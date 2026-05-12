"""Recovery-rate sweep across 4 tiers of granularity.

For each CF set (shared-template, varied-number examples):
  Pass 1 (capture, no intervention): mean attn_out, mlp_out, and residual-stream
         output per (step, layer) across N examples.
  Pass 2 (4 ablation sweeps), each measuring recovery rate = fraction of
         examples where patched prediction matches baseline prediction:
    Sweep A — per STEP (whole step ablated): replace every layer's attn+mlp
              outputs at step k with their means. 6 conditions.
    Sweep B — per-LAYER residual stream: replace the entire block output
              (post-residual) at layer L across ALL 6 steps. 12 conditions.
    Sweep C — per-LAYER attn (all heads): replace attn_out at layer L across
              ALL 6 steps. 12 conditions.
    Sweep D — per-LAYER mlp:  replace mlp_out at layer L across ALL 6 steps.
              12 conditions.

Total: 42 conditions per CF set.

Lower recovery rate = patching killed more outputs = cell more essential.
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

CF_SETS = ["vary_a_2digit", "vary_numerals"]  # 100% baseline + diverse-numbers


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
    OUT_JSON = PD / "patch_recovery_sweep_gsm8k.json"

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
    targs = TrainingArguments(output_dir="/tmp/_pr", bf16=True,
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

    CAP = {
        "mode": "off", "step": -1,
        # Ablation params (set per condition):
        # ablate_steps: list of step indices (0..5) where ablation is active
        # ablate_layers: list of layer indices (0..11)
        # ablate_blocks: subset of {"attn", "mlp", "residual"}
        "ablate_steps": [], "ablate_layers": [], "ablate_blocks": set(),
        "mean_attn": None, "mean_mlp": None, "mean_resid": None,
    }
    cap_attn_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    cap_mlp_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    cap_resid_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    cap_count = [0]

    def in_ablate_set(step, layer, block):
        return (step in CAP["ablate_steps"] and layer in CAP["ablate_layers"]
                and block in CAP["ablate_blocks"])

    def make_attn_hook(idx):
        def fn(_module, _inputs, output):
            if CAP["mode"] == "capture" and CAP["step"] >= 0:
                a = output[0]
                cap_attn_sum[CAP["step"], idx] += a[:, -1, :].float().detach().cpu().numpy().sum(axis=0)
            elif CAP["mode"] == "ablate" and in_ablate_set(CAP["step"], idx, "attn"):
                a = output[0].clone()
                a[:, -1, :] = CAP["mean_attn"][CAP["step"], idx].to(a.device, dtype=a.dtype)
                return (a,) + output[1:]
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_module, _inputs, output):
            if CAP["mode"] == "capture" and CAP["step"] >= 0:
                cap_mlp_sum[CAP["step"], idx] += output[:, -1, :].float().detach().cpu().numpy().sum(axis=0)
            elif CAP["mode"] == "ablate" and in_ablate_set(CAP["step"], idx, "mlp"):
                o = output.clone()
                o[:, -1, :] = CAP["mean_mlp"][CAP["step"], idx].to(o.device, dtype=o.dtype)
                return o
            return output
        return fn

    def make_block_hook(idx):
        # GPT2Block returns (hidden_states, present_kv, ...) where hidden_states
        # is the residual-stream output of this layer (post-attn + post-mlp).
        def fn(_module, _inputs, output):
            if CAP["mode"] == "capture" and CAP["step"] >= 0:
                h = output[0]
                cap_resid_sum[CAP["step"], idx] += h[:, -1, :].float().detach().cpu().numpy().sum(axis=0)
            elif CAP["mode"] == "ablate" and in_ablate_set(CAP["step"], idx, "residual"):
                h = output[0].clone()
                h[:, -1, :] = CAP["mean_resid"][CAP["step"], idx].to(h.device, dtype=h.dtype)
                return (h,) + output[1:]
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))
        handles.append(blk.register_forward_hook(make_block_hook(i)))

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

    results = {"cf_sets": {}, "N_LAYERS": N_LAYERS, "N_LAT": N_LAT}

    for cf_name in CF_SETS:
        print(f"\n=== CF set: {cf_name} ===", flush=True)
        qs, golds = load_cf(cf_name)
        N = len(qs)
        golds_arr = np.array(golds)
        # Pass 1: capture
        CAP["mode"] = "capture"
        cap_attn_sum[:] = 0; cap_mlp_sum[:] = 0; cap_resid_sum[:] = 0
        cap_count[0] = 0
        base_strs = []
        for s in range(0, N, BS):
            base_strs += run_batch(qs[s:s+BS])
            cap_count[0] += min(BS, N - s)
        mean_attn = torch.from_numpy(cap_attn_sum / cap_count[0]).to("cuda", dtype=torch.bfloat16)
        mean_mlp = torch.from_numpy(cap_mlp_sum / cap_count[0]).to("cuda", dtype=torch.bfloat16)
        mean_resid = torch.from_numpy(cap_resid_sum / cap_count[0]).to("cuda", dtype=torch.bfloat16)
        CAP["mean_attn"] = mean_attn; CAP["mean_mlp"] = mean_mlp; CAP["mean_resid"] = mean_resid
        base_ints = [codi_extract(s) for s in base_strs]
        base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                  for i, v in enumerate(base_ints)])
        base_acc = float(base_correct.mean())
        print(f"  N={N}, baseline acc={base_acc*100:.1f}%")

        def measure(strs):
            ints = [codi_extract(s) for s in strs]
            recovered = sum(1 for i in range(N) if ints[i] == base_ints[i])
            correct = sum(1 for i in range(N) if ints[i] is not None and abs(ints[i] - golds_arr[i]) < 1e-3)
            n_changed = N - recovered
            return {"recovery_rate": recovered / N,
                    "accuracy": correct / N,
                    "delta_acc": correct / N - base_acc,
                    "n_changed_from_baseline": n_changed}

        CAP["mode"] = "ablate"
        sw = {}

        # Sweep A: per step (ablate all blocks at ALL layers at this step)
        print(f"\n  -- Sweep A: per-step (all 12 layers × attn+mlp at step k) --", flush=True)
        for step in range(N_LAT):
            CAP["ablate_steps"] = [step]
            CAP["ablate_layers"] = list(range(N_LAYERS))
            CAP["ablate_blocks"] = {"attn", "mlp"}
            strs = run_all(qs)
            m = measure(strs)
            sw[f"step{step+1}"] = m
            print(f"    step {step+1}: recovery={m['recovery_rate']*100:5.1f}%  "
                  f"acc={m['accuracy']*100:5.1f}%  changed={m['n_changed_from_baseline']:3d}/{N}")

        # Sweep B: per-layer residual stream (across all 6 steps)
        print(f"\n  -- Sweep B: per-layer residual stream (all 6 steps, layer L block output) --", flush=True)
        for layer in range(N_LAYERS):
            CAP["ablate_steps"] = list(range(N_LAT))
            CAP["ablate_layers"] = [layer]
            CAP["ablate_blocks"] = {"residual"}
            strs = run_all(qs)
            m = measure(strs)
            sw[f"resid_L{layer}"] = m
            print(f"    layer {layer}: recovery={m['recovery_rate']*100:5.1f}%  "
                  f"acc={m['accuracy']*100:5.1f}%  changed={m['n_changed_from_baseline']:3d}/{N}")

        # Sweep C: per-layer attn (across all 6 steps)
        print(f"\n  -- Sweep C: per-layer attention (all 6 steps, layer L attn) --", flush=True)
        for layer in range(N_LAYERS):
            CAP["ablate_steps"] = list(range(N_LAT))
            CAP["ablate_layers"] = [layer]
            CAP["ablate_blocks"] = {"attn"}
            strs = run_all(qs)
            m = measure(strs)
            sw[f"attn_L{layer}"] = m
            print(f"    layer {layer}: recovery={m['recovery_rate']*100:5.1f}%  "
                  f"acc={m['accuracy']*100:5.1f}%  changed={m['n_changed_from_baseline']:3d}/{N}")

        # Sweep D: per-layer mlp (across all 6 steps)
        print(f"\n  -- Sweep D: per-layer MLP (all 6 steps, layer L mlp) --", flush=True)
        for layer in range(N_LAYERS):
            CAP["ablate_steps"] = list(range(N_LAT))
            CAP["ablate_layers"] = [layer]
            CAP["ablate_blocks"] = {"mlp"}
            strs = run_all(qs)
            m = measure(strs)
            sw[f"mlp_L{layer}"] = m
            print(f"    layer {layer}: recovery={m['recovery_rate']*100:5.1f}%  "
                  f"acc={m['accuracy']*100:5.1f}%  changed={m['n_changed_from_baseline']:3d}/{N}")

        results["cf_sets"][cf_name] = {
            "N": N, "baseline_accuracy": base_acc, "sweeps": sw,
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
