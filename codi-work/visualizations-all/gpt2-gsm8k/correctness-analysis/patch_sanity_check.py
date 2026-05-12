"""Sanity check the mean-patching pipeline.

Two tests:
  1. ZERO ablation per step: replace ALL attn_out + mlp_out at every layer of
     step k with ZEROS (not the mean). If recovery rate stays ~100%, the hook
     is broken; if accuracy collapses, the hook works and the mean-patch
     result is genuinely "this is a small perturbation."
  2. Activation-vs-mean L2 stats: for each (step, layer, block) report
     mean ‖x_i‖, ‖mean(x)‖, and mean ‖x_i − mean(x)‖. If the deviation is
     tiny relative to the activation norm, mean-patching is a near-no-op.

Run on vary_numerals (N=80, baseline 72.5%) and vary_a_2digit (N=80, 100%).
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
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = REPO / "experiments" / "computation_probes"
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

CF_SETS = ["gsm8k_cf_op_strict"]


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def load_cf(name):
    rows = json.load(open(CF_DIR / f"{name}.json"))
    qs = [r["question_concat"].strip().replace("  ", " ") for r in rows]
    golds = [float(r["answer"]) for r in rows]
    return qs, golds


def main():
    BS = 16
    OUT_JSON = PD / "patch_sanity_check_gsm8k.json"

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
    targs = TrainingArguments(output_dir="/tmp/_san", bf16=True,
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

    CAP = {"mode": "off", "step": -1,
           "zero_step": -1, "zero_blocks": set(),
           # capture buffers (per step, per layer, per block):
           # we keep running sums + sums of squared norms + sum of vectors.
           }
    # For stats:
    attn_norm_sum_sq = np.zeros((N_LAT, N_LAYERS), dtype=np.float64)  # sum of ||a_i||^2
    mlp_norm_sum_sq = np.zeros((N_LAT, N_LAYERS), dtype=np.float64)
    attn_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    mlp_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    cap_count = [0]

    def make_attn_hook(idx):
        def fn(_module, _inputs, output):
            if CAP["mode"] == "capture" and CAP["step"] >= 0:
                a = output[0]
                last = a[:, -1, :].float().detach().cpu().numpy()
                attn_sum[CAP["step"], idx] += last.sum(axis=0)
                attn_norm_sum_sq[CAP["step"], idx] += float((last ** 2).sum())
            elif CAP["mode"] == "zero" and CAP["step"] == CAP["zero_step"] and "attn" in CAP["zero_blocks"]:
                a = output[0].clone()
                a[:, -1, :] = 0
                return (a,) + output[1:]
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_module, _inputs, output):
            if CAP["mode"] == "capture" and CAP["step"] >= 0:
                last = output[:, -1, :].float().detach().cpu().numpy()
                mlp_sum[CAP["step"], idx] += last.sum(axis=0)
                mlp_norm_sum_sq[CAP["step"], idx] += float((last ** 2).sum())
            elif CAP["mode"] == "zero" and CAP["step"] == CAP["zero_step"] and "mlp" in CAP["zero_blocks"]:
                o = output.clone()
                o[:, -1, :] = 0
                return o
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))

    @torch.no_grad()
    def run_batch(qs):
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
        CAP["step"] = -1
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

    def run_all(qs):
        out = []
        for s in range(0, len(qs), BS):
            out += run_batch(qs[s:s+BS])
        return out

    results = {}
    for cf_name in CF_SETS:
        print(f"\n=== {cf_name} ===", flush=True)
        qs, golds = load_cf(cf_name)
        N = len(qs)
        golds_arr = np.array(golds)
        # Pass 1: capture
        CAP["mode"] = "capture"
        attn_sum[:] = 0; mlp_sum[:] = 0
        attn_norm_sum_sq[:] = 0; mlp_norm_sum_sq[:] = 0
        cap_count[0] = 0
        base_strs = []
        for s in range(0, N, BS):
            base_strs += run_batch(qs[s:s+BS])
            cap_count[0] += min(BS, N - s)
        nn = cap_count[0]
        mean_attn = attn_sum / nn
        mean_mlp = mlp_sum / nn
        mean_attn_norm = np.linalg.norm(mean_attn, axis=-1)  # (S, L)
        mean_mlp_norm = np.linalg.norm(mean_mlp, axis=-1)
        # Mean over examples of ||a_i||
        mean_perex_attn_norm = np.sqrt(attn_norm_sum_sq / nn)
        mean_perex_mlp_norm = np.sqrt(mlp_norm_sum_sq / nn)
        # Mean ||a_i − mean(a)||^2 = E[||a||^2] − ||E[a]||^2
        var_attn = attn_norm_sum_sq / nn - (mean_attn_norm ** 2)
        var_mlp = mlp_norm_sum_sq / nn - (mean_mlp_norm ** 2)
        std_attn = np.sqrt(np.clip(var_attn, 0, None))
        std_mlp = np.sqrt(np.clip(var_mlp, 0, None))

        base_ints = [codi_extract(s) for s in base_strs]
        base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                  for i, v in enumerate(base_ints)])
        base_acc = float(base_correct.mean())
        print(f"  N={N} baseline acc={base_acc*100:.1f}%")

        # Print activation stats per step / layer
        print(f"\n  Per-cell activation stats (across {nn} examples):")
        print(f"  attn-block at last token:")
        print(f"        layer:  " + "    ".join(f"L{l:2d}" for l in range(N_LAYERS)))
        for s in range(N_LAT):
            row_ratio = std_attn[s] / np.maximum(mean_attn_norm[s], 1e-9)
            print(f"    step {s+1} ‖mean‖={mean_attn_norm[s].mean():5.1f}  "
                  f"std/‖mean‖: " + "  ".join(f"{v:4.2f}" for v in row_ratio))
        print(f"\n  mlp-block at last token:")
        for s in range(N_LAT):
            row_ratio = std_mlp[s] / np.maximum(mean_mlp_norm[s], 1e-9)
            print(f"    step {s+1} ‖mean‖={mean_mlp_norm[s].mean():5.1f}  "
                  f"std/‖mean‖: " + "  ".join(f"{v:4.2f}" for v in row_ratio))

        # Zero-ablation per step
        print(f"\n  -- ZERO ablation per step (all 12 layers attn+mlp → 0) --")
        CAP["mode"] = "zero"
        zero_per_step = {}
        for step in range(N_LAT):
            CAP["zero_step"] = step
            CAP["zero_blocks"] = {"attn", "mlp"}
            strs = run_all(qs)
            ints = [codi_extract(s) for s in strs]
            recovered = sum(1 for i in range(N) if ints[i] == base_ints[i])
            correct = sum(1 for i in range(N) if ints[i] is not None and abs(ints[i] - golds_arr[i]) < 1e-3)
            n_changed = N - recovered
            zero_per_step[f"step{step+1}"] = {
                "recovery_rate": recovered / N,
                "accuracy": correct / N,
                "delta_acc": correct / N - base_acc,
                "n_changed": n_changed,
            }
            print(f"    step {step+1}: recovery={recovered/N*100:5.1f}%  "
                  f"acc={correct/N*100:5.1f}%  changed={n_changed}/{N}")

        results[cf_name] = {
            "N": N, "baseline_accuracy": base_acc,
            "mean_attn_norm": mean_attn_norm.tolist(),
            "mean_mlp_norm": mean_mlp_norm.tolist(),
            "std_attn_norm": std_attn.tolist(),
            "std_mlp_norm": std_mlp.tolist(),
            "std_over_mean_attn": (std_attn / np.maximum(mean_attn_norm, 1e-9)).tolist(),
            "std_over_mean_mlp": (std_mlp / np.maximum(mean_mlp_norm, 1e-9)).tolist(),
            "zero_ablation_per_step": zero_per_step,
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
