"""GSM8K-port of flow_map.py — information-flow map of CODI-GPT-2 on GSM8K.

For every (phase ∈ {latent, decode}, step, layer) cell, captures:
  - attention weights from the last token to all prior key positions, per head.
  - ||attn_block_output||_2 at the last token.
  - ||mlp_block_output||_2 at the last token.

Aggregates attention into position classes (Q, BOT, L1..L6, EOT, D0..D9),
averaged across all GSM8K test problems.

Outputs:
  flow_map_gsm8k.npz
  flow_map_gsm8k_meta.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

N_DECODE_STEPS = 10
CLASS_NAMES = ["Q", "BOT", "L1", "L2", "L3", "L4", "L5", "L6", "EOT"] + [
    f"D{i}" for i in range(N_DECODE_STEPS)
]


def main():
    BS = 16

    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *args, **k: _orig(*args, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments

    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=["c_attn", "c_proj", "c_fc"],
                          init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_fmgsm", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=768,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=6,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id

    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    N_LAYERS = len(transformer.h)
    cfg = model.codi.config
    N_HEADS = cfg.n_head
    HID = cfg.n_embd
    print(f"  GPT-2: {N_LAYERS} layers, {N_HEADS} heads, hidden={HID}")
    N_LATENT = 6
    N_CLASSES = len(CLASS_NAMES)

    CAP = {
        "active": False, "phase": "off", "step": -1,
        "attn_norm": [None] * N_LAYERS,
        "mlp_norm":  [None] * N_LAYERS,
        "attn_w":    [None] * N_LAYERS,
    }

    def make_attn_hook(idx):
        def fn(_module, _inputs, output):
            if not CAP["active"]: return output
            a = output[0]
            CAP["attn_norm"][idx] = a[:, -1, :].norm(dim=-1).float().detach().cpu().numpy()
            if len(output) > 2 and output[2] is not None:
                w = output[2]
                CAP["attn_w"][idx] = w[:, :, -1, :].float().detach().cpu().numpy()
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_module, _inputs, output):
            if not CAP["active"]: return output
            CAP["mlp_norm"][idx] = output[:, -1, :].norm(dim=-1).float().detach().cpu().numpy()
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))

    # ---- GSM8K data ----
    ds = load_dataset("gsm8k", "main")["test"]
    questions = [ex["question"].strip().replace("  ", " ") for ex in ds]
    N = len(questions)
    print(f"  N={N} GSM8K test problems")

    sum_attn = np.zeros((2, N_DECODE_STEPS, N_LAYERS, N_HEADS, N_CLASSES), dtype=np.float64)
    cnt_attn = np.zeros((2, N_DECODE_STEPS), dtype=np.int64)
    sum_attn_norm = np.zeros((2, N_DECODE_STEPS, N_LAYERS), dtype=np.float64)
    sum_mlp_norm  = np.zeros((2, N_DECODE_STEPS, N_LAYERS), dtype=np.float64)
    cnt_norm = np.zeros((2, N_DECODE_STEPS), dtype=np.int64)

    @torch.no_grad()
    def run_batch(qs):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        q_lens = batch["attention_mask"].sum(dim=-1).cpu().numpy()
        Lq_pad = batch["input_ids"].shape[1]
        bot_id_tok = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot_id_tok], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot_id_tok)], dim=1)

        CAP["active"] = False
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)

        CAP["active"] = True
        for step in range(N_LATENT):
            CAP["phase"] = "latent"; CAP["step"] = step
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             output_attentions=True, past_key_values=past)
            past = out.past_key_values
            aggregate_phase(0, step, B, q_lens, Lq_pad)
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)

        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        done = [False] * B
        for dec_step in range(N_DECODE_STEPS):
            CAP["phase"] = "decode"; CAP["step"] = dec_step
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              output_attentions=True, past_key_values=past)
            past = sout.past_key_values
            aggregate_phase(1, dec_step, B, q_lens, Lq_pad)
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if not done[b] and int(next_ids[b].item()) == eos_id:
                    done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        CAP["active"] = False

    def aggregate_phase(phase_id, step, B, q_lens, Lq_pad):
        for l in range(N_LAYERS):
            w = CAP["attn_w"][l]
            if w is None: continue
            k_len = w.shape[-1]
            for b in range(B):
                pad_amt_b = Lq_pad - int(q_lens[b])
                if Lq_pad > pad_amt_b:
                    sum_attn[phase_id, step, l, :, 0] += w[b, :, pad_amt_b:Lq_pad].sum(axis=1)
            if Lq_pad < k_len:
                sum_attn[phase_id, step, l, :, 1] += w[:, :, Lq_pad].sum(axis=0)
            n_latents_present = (step + 1) if phase_id == 0 else 6
            for li in range(n_latents_present):
                p = Lq_pad + 1 + li
                if p < k_len:
                    sum_attn[phase_id, step, l, :, 2 + li] += w[:, :, p].sum(axis=0)
            if phase_id == 1:
                p = Lq_pad + 7
                if p < k_len:
                    sum_attn[phase_id, step, l, :, 8] += w[:, :, p].sum(axis=0)
                for d in range(step + 1):
                    p = Lq_pad + 8 + d
                    if p < k_len:
                        sum_attn[phase_id, step, l, :, 9 + d] += w[:, :, p].sum(axis=0)
            sum_attn_norm[phase_id, step, l] += float(CAP["attn_norm"][l].sum())
            sum_mlp_norm[phase_id, step, l]  += float(CAP["mlp_norm"][l].sum())
        cnt_attn[phase_id, step] += B
        cnt_norm[phase_id, step] += B

    t0 = time.time()
    for s in range(0, N, BS):
        run_batch(questions[s:s+BS])
        done = s + BS
        if done % 64 == 0 or done >= N:
            print(f"  {min(done, N)}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    mean_attn = np.zeros_like(sum_attn)
    mean_attn_norm = np.zeros_like(sum_attn_norm)
    mean_mlp_norm  = np.zeros_like(sum_mlp_norm)
    for phase_id in range(2):
        for step in range(N_DECODE_STEPS):
            n = cnt_attn[phase_id, step]
            if n > 0:
                mean_attn[phase_id, step] = sum_attn[phase_id, step] / n
            n2 = cnt_norm[phase_id, step]
            if n2 > 0:
                mean_attn_norm[phase_id, step] = sum_attn_norm[phase_id, step] / n2
                mean_mlp_norm[phase_id, step]  = sum_mlp_norm[phase_id, step] / n2

    out_npz = PD / "flow_map_gsm8k.npz"
    np.savez(out_npz,
             mean_attn=mean_attn.astype(np.float32),
             mean_attn_norm=mean_attn_norm.astype(np.float32),
             mean_mlp_norm=mean_mlp_norm.astype(np.float32),
             cnt_attn=cnt_attn, cnt_norm=cnt_norm)
    meta = {
        "N_total": int(N), "N_layers": int(N_LAYERS), "N_heads": int(N_HEADS),
        "N_latent_steps": int(N_LATENT), "N_decode_steps": int(N_DECODE_STEPS),
        "class_names": CLASS_NAMES,
        "phases": ["latent", "decode"],
        "dataset": "gsm8k/main/test",
    }
    (PD / "flow_map_gsm8k_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out_npz} and meta")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
