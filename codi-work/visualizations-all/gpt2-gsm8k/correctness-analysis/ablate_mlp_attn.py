"""Mean-ablation sweep: for each (latent step, layer, block ∈ {mlp, attn}),
replace that block's output with its mean across N=1000 SVAMP examples and
measure the resulting greedy-decode accuracy.

Procedure:
  Pass 1: forward all 1000 examples through the latent loop (no intervention),
          capturing the mean MLP output and mean attn-block output at the last
          token for each (step, layer). Stored as (6, 12, 768) bf16.
  Pass 2: for each condition (step, layer, block, mode='mean'):
          run the model with a forward-hook on that block replacing the
          block's output at the last token with the captured mean vector.
          Then complete the rest of the latent loop + decode, parse the
          predicted answer, score against gold.

Output:
  ablation_mlp_attn.json — per condition: accuracy, n_changed_vs_baseline,
                          n_correct_after, delta_acc, sample preds.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

OUT_JSON = PD / "ablation_mlp_attn_gsm8k.json"


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    BS = 16

    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=["c_attn", "c_proj", "c_fc"],
                          init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_ab", bf16=True,
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
    N_LAYERS = model.codi.config.n_layer
    HID = model.codi.config.n_embd
    N_LAT = 6
    print(f"  layers={N_LAYERS}  hidden={HID}", flush=True)

    # Hook state: capture mode or ablation mode.
    CAP = {"mode": "off", "step": -1,
           "ablate_step": -1, "ablate_layer": -1, "ablate_block": ""}
    # Captured buffers (per step, per layer): summed last-token output for averaging.
    cap_mlp_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    cap_attn_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    cap_count = 0
    # Mean tensors (computed at end of pass 1)
    mean_mlp = None   # torch.Tensor (N_LAT, N_LAYERS, HID) on GPU
    mean_attn = None

    def make_attn_hook(idx):
        def fn(_module, _inputs, output):
            if CAP["mode"] == "capture" and CAP["step"] >= 0:
                # output[0] is the attn block output (B, T, HID), already post-c_proj
                a = output[0]
                cap_attn_sum[CAP["step"], idx] += a[:, -1, :].float().detach().cpu().numpy().sum(axis=0)
            elif (CAP["mode"] == "ablate" and CAP["step"] == CAP["ablate_step"]
                  and idx == CAP["ablate_layer"] and CAP["ablate_block"] == "attn"):
                a = output[0].clone()
                a[:, -1, :] = mean_attn[CAP["step"], idx].to(a.device, dtype=a.dtype)
                return (a,) + output[1:]
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_module, _inputs, output):
            if CAP["mode"] == "capture" and CAP["step"] >= 0:
                cap_mlp_sum[CAP["step"], idx] += output[:, -1, :].float().detach().cpu().numpy().sum(axis=0)
            elif (CAP["mode"] == "ablate" and CAP["step"] == CAP["ablate_step"]
                  and idx == CAP["ablate_layer"] and CAP["ablate_block"] == "mlp"):
                o = output.clone()
                o[:, -1, :] = mean_mlp[CAP["step"], idx].to(o.device, dtype=o.dtype)
                return o
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))

    ds = load_dataset("gsm8k", "main")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    N = len(questions)

    @torch.no_grad()
    def run_batch(start):
        qs = questions[start:start + BS]
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for step in range(N_LAT):
            CAP["step"] = step
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            o = model.codi(inputs_embeds=latent, attention_mask=attn,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        CAP["step"] = -1   # ablation only happens during latent loop
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(64):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if done[b]: continue
                tid = int(next_ids[b].item()); tokens[b].append(tid)
                if tid == eos_id: done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    # --- Pass 1: capture means ---
    print("\n=== PASS 1: capturing means (no intervention) ===", flush=True)
    CAP["mode"] = "capture"
    cap_count = 0
    t0 = time.time()
    base_strs = []
    for s in range(0, N, BS):
        strs = run_batch(s)
        base_strs += strs
        cap_count += len(strs)
        if (s + BS) % 64 == 0:
            print(f"  {min(s + BS, N)}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    mean_mlp_np = cap_mlp_sum / cap_count
    mean_attn_np = cap_attn_sum / cap_count
    mean_mlp = torch.from_numpy(mean_mlp_np).to("cuda", dtype=torch.bfloat16)
    mean_attn = torch.from_numpy(mean_attn_np).to("cuda", dtype=torch.bfloat16)
    print(f"  means captured. shapes: mlp {tuple(mean_mlp.shape)} attn {tuple(mean_attn.shape)}")

    base_ints = [codi_extract(s) for s in base_strs]
    base_correct = np.array([v is not None and abs(v - golds[i]) < 1e-3
                              for i, v in enumerate(base_ints)])
    base_acc = float(base_correct.mean())
    print(f"  baseline accuracy: {base_acc*100:.1f}%")

    # --- Pass 2: 144 ablation conditions ---
    CAP["mode"] = "ablate"
    results = {"baseline_accuracy": base_acc,
               "baseline_n_correct": int(base_correct.sum()),
               "N": N, "conditions": {}}
    n_cells = N_LAT * N_LAYERS * 2
    cell_i = 0
    print(f"\n=== PASS 2: {n_cells} ablation conditions ===", flush=True)
    t0 = time.time()
    for step in range(N_LAT):
        for layer in range(N_LAYERS):
            for block in ("mlp", "attn"):
                CAP["ablate_step"] = step
                CAP["ablate_layer"] = layer
                CAP["ablate_block"] = block
                strs = []
                for s in range(0, N, BS):
                    strs += run_batch(s)
                ints = [codi_extract(s) for s in strs]
                correct = np.array([v is not None and abs(v - golds[i]) < 1e-3
                                    for i, v in enumerate(ints)])
                acc = float(correct.mean())
                n_changed = int(sum(1 for i in range(N) if ints[i] != base_ints[i]))
                wr = int(((~base_correct) & correct).sum())
                rw = int((base_correct & ~correct).sum())
                k = f"step{step+1}_L{layer}_{block}"
                results["conditions"][k] = {
                    "accuracy": acc, "delta_acc": acc - base_acc,
                    "n_correct": int(correct.sum()), "n_changed": n_changed,
                    "wrong_to_right": wr, "right_to_wrong": rw,
                }
                cell_i += 1
                print(f"  [{cell_i:3d}/{n_cells}] {k}  acc={acc*100:5.1f}%  "
                      f"delta={(acc-base_acc)*100:+5.1f}pp  "
                      f"changed={n_changed:4d}  ({time.time()-t0:.0f}s)", flush=True)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
