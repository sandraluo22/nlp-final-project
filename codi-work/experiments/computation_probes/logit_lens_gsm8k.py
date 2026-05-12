"""Logit-lens over CODI's 6 latent steps on GSM8K.

For each problem at each (step ∈ 1..6, layer ∈ 0..11, sublayer ∈ {resid_pre,
attn_out, mlp_out, resid_post}) at the LAST-TOKEN position:
  - apply the final LayerNorm + LM head to that hidden state
  - record the top-K tokens and their softmax probabilities

Aggregate across N problems:
  - mean confidence of each cell's top-1 token
  - which token is most-often the top-1 prediction (modal token)
  - average rank of the GOLD answer's first-digit token (if force-decode would
    emit it) at each cell

Outputs:
  logit_lens_gsm8k.npz
  logit_lens_gsm8k.json   (a summary: per (step, layer, sublayer) mean top-1
                            confidence + modal top-1 token + sample top-K
                            tokens for the first 5 problems)
"""
from __future__ import annotations

import json, os, re, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "codi"))

N_EXAMPLES = 50    # subset for the logit-lens table
TOP_K = 5
SUBLAYERS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


def main():
    BS = 8
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI from {ckpt}", flush=True)
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
    targs = TrainingArguments(output_dir="/tmp/_llens", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=768,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=6,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False); model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16); model.eval()

    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    N_LAT = 6
    N_LAYERS = len(transformer.h)
    H = model.codi.config.n_embd
    # Final LayerNorm of GPT-2
    final_ln = transformer.ln_f
    lm_head = model.codi.get_output_embeddings()
    print(f"  N_LAYERS={N_LAYERS} H={H}")

    # Hooks to capture per-(step, layer) {resid_pre, attn_out, mlp_out, resid_post}
    CAP = {"active": False, "step": -1, "buf": None}

    def make_attn_hook(idx):
        def fn(_m, _i, output):
            if not CAP["active"] or CAP["step"] < 0: return output
            a = output[0] if isinstance(output, tuple) else output
            CAP["buf"]["attn_out"][CAP["step"], idx] = a[:, -1, :].float().cpu()
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_m, _i, output):
            if not CAP["active"] or CAP["step"] < 0: return output
            CAP["buf"]["mlp_out"][CAP["step"], idx] = output[:, -1, :].float().cpu()
            return output
        return fn

    def make_block_pre_hook(idx):
        def fn(_m, inputs):
            if not CAP["active"] or CAP["step"] < 0: return None
            h = inputs[0]
            CAP["buf"]["resid_pre"][CAP["step"], idx] = h[:, -1, :].float().cpu()
            return None
        return fn

    def make_block_post_hook(idx):
        def fn(_m, _i, output):
            if not CAP["active"] or CAP["step"] < 0: return output
            h = output[0] if isinstance(output, tuple) else output
            CAP["buf"]["resid_post"][CAP["step"], idx] = h[:, -1, :].float().cpu()
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.register_forward_pre_hook(make_block_pre_hook(i)))
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))
        handles.append(blk.register_forward_hook(make_block_post_hook(i)))

    ds = load_dataset("gsm8k", "main")["test"]
    questions = []
    gold_finals = []
    for ex in ds:
        m = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if m is None: continue
        questions.append(ex["question"].strip().replace("  ", " "))
        gold_finals.append(float(m.group(1)))
        if len(questions) >= N_EXAMPLES: break

    # Output buffer: per-(step, layer, sublayer) accumulation across examples
    sum_top1_conf = np.zeros((N_LAT, N_LAYERS, len(SUBLAYERS)), dtype=np.float64)
    top1_token_counter = [[[Counter() for _ in SUBLAYERS] for _ in range(N_LAYERS)] for _ in range(N_LAT)]
    cnt_examples = 0
    # Save per-example top-5 tokens + probs for the first 5 problems (sample for table)
    sample_logits = []

    @torch.no_grad()
    def run_batch(qs, save_samples_until):
        nonlocal cnt_examples
        B = len(qs)
        CAP["buf"] = {sl: torch.zeros(N_LAT, N_LAYERS, B, H) for sl in SUBLAYERS}
        CAP["step"] = -1; CAP["active"] = True
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for s in range(targs.inf_latent_iterations):
            CAP["step"] = s
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        CAP["active"] = False; CAP["step"] = -1

        # Apply LayerNorm + LM head to each captured hidden state
        for sl_i, sl_name in enumerate(SUBLAYERS):
            # captured shape: (N_LAT, N_LAYERS, B, H)
            for s in range(N_LAT):
                for L in range(N_LAYERS):
                    h = CAP["buf"][sl_name][s, L].to("cuda").to(torch.bfloat16)
                    h_ln = final_ln(h)
                    logits = lm_head(h_ln)[:, :model.codi.config.vocab_size - 1]
                    probs = torch.softmax(logits.float(), dim=-1).cpu()
                    topk = probs.topk(TOP_K, dim=-1)
                    top1_conf = topk.values[:, 0].numpy()
                    top1_id = topk.indices[:, 0].numpy()
                    sum_top1_conf[s, L, sl_i] += top1_conf.sum()
                    for b in range(B):
                        top1_token_counter[s][L][sl_i][int(top1_id[b])] += 1
                        if cnt_examples + b < save_samples_until:
                            if cnt_examples + b == len(sample_logits):
                                sample_logits.append({"q": qs[b], "gold": gold_finals[cnt_examples + b],
                                                       "cells": {}})
                            entry = sample_logits[cnt_examples + b]
                            key = f"step{s+1}_L{L}_{sl_name}"
                            entry["cells"][key] = {
                                "top_tokens": [tok.decode([int(i)]) for i in topk.indices[b].tolist()],
                                "top_probs": [float(p) for p in topk.values[b].tolist()],
                            }
        cnt_examples += B

    t0 = time.time()
    for s in range(0, len(questions), BS):
        run_batch(questions[s:s+BS], save_samples_until=5)
        print(f"  {min(s+BS, len(questions))}/{len(questions)}  ({time.time()-t0:.0f}s)", flush=True)

    mean_top1_conf = sum_top1_conf / max(1, cnt_examples)
    # Most-common top-1 token per cell
    modal_token = [[[None for _ in SUBLAYERS] for _ in range(N_LAYERS)] for _ in range(N_LAT)]
    modal_freq = np.zeros((N_LAT, N_LAYERS, len(SUBLAYERS)), dtype=np.float64)
    for s in range(N_LAT):
        for L in range(N_LAYERS):
            for sl_i in range(len(SUBLAYERS)):
                c = top1_token_counter[s][L][sl_i]
                if not c: continue
                tok_id, freq = c.most_common(1)[0]
                modal_token[s][L][sl_i] = tok.decode([tok_id])
                modal_freq[s, L, sl_i] = freq / cnt_examples

    OUT_NPZ = PD / "logit_lens_gsm8k.npz"
    np.savez(OUT_NPZ,
             mean_top1_conf=mean_top1_conf.astype(np.float32),
             modal_freq=modal_freq.astype(np.float32),
             sublayers=np.array(SUBLAYERS))

    summary = {
        "N_examples": int(cnt_examples), "N_LAT": int(N_LAT),
        "N_LAYERS": int(N_LAYERS), "TOP_K": TOP_K,
        "SUBLAYERS": SUBLAYERS,
        "mean_top1_conf": mean_top1_conf.tolist(),
        "modal_token": [[[(modal_token[s][L][sl_i] or "")
                          for sl_i in range(len(SUBLAYERS))]
                         for L in range(N_LAYERS)] for s in range(N_LAT)],
        "modal_freq": modal_freq.tolist(),
        "sample_top5": sample_logits,
    }
    OUT_JSON = PD / "logit_lens_gsm8k.json"
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(f"saved {OUT_NPZ} and {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
