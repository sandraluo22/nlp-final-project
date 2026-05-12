"""Mean-CF activation patching to localize NUMBER-DEPENDENT computation.

For each CF set with a shared template and varying numbers, capture per-example
activations at every (latent step, layer, block ∈ {mlp, attn}). Compute the
mean activation across the set (the "different-numbers centroid"). Then for
each of 144 cells, run the model with that cell's activation replaced by the
mean — if the prediction changes, the cell is computing on the number-specific
content.

This is the canonical activation-patching / DCM design with a population-mean
counterfactual.

Per CF set, per cell, we record:
  baseline_acc, patched_acc, n_changed_int (vs baseline), wrong_to_right,
  right_to_wrong, n_correct (after).

Output: patch_cf_mean.json
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

CF_SETS = ["gsm8k_vary_operator", "gsm8k_cf_op_strict"]


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
    OUT_JSON = PD / "patch_cf_mean_gsm8k.json"

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
    targs = TrainingArguments(output_dir="/tmp/_pc", bf16=True,
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
           "ablate_step": -1, "ablate_layer": -1, "ablate_block": "",
           "mean_attn": None, "mean_mlp": None}
    # per-layer per-step buffers used in capture mode
    cap_mlp_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    cap_attn_sum = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    cap_count = [0]

    def make_attn_hook(idx):
        def fn(_module, _inputs, output):
            if CAP["mode"] == "capture" and CAP["step"] >= 0:
                a = output[0]
                cap_attn_sum[CAP["step"], idx] += a[:, -1, :].float().detach().cpu().numpy().sum(axis=0)
            elif (CAP["mode"] == "ablate" and CAP["step"] == CAP["ablate_step"]
                  and idx == CAP["ablate_layer"] and CAP["ablate_block"] == "attn"):
                a = output[0].clone()
                a[:, -1, :] = CAP["mean_attn"][CAP["step"], idx].to(a.device, dtype=a.dtype)
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
                o[:, -1, :] = CAP["mean_mlp"][CAP["step"], idx].to(o.device, dtype=o.dtype)
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

    results = {"cf_sets": {}, "N_LAYERS": N_LAYERS, "N_LAT": N_LAT}

    for cf_name in CF_SETS:
        print(f"\n=== CF set: {cf_name} ===", flush=True)
        try:
            qs, golds = load_cf(cf_name)
        except Exception as e:
            print(f"  skip: {e}"); continue
        N = len(qs)
        golds_arr = np.array(golds)
        # Pass 1: capture
        CAP["mode"] = "capture"
        cap_mlp_sum[:] = 0; cap_attn_sum[:] = 0; cap_count[0] = 0
        t0 = time.time()
        base_strs = []
        for s in range(0, N, BS):
            base_strs += run_batch(qs[s:s+BS])
            cap_count[0] += min(BS, N - s)
        mean_mlp = torch.from_numpy(cap_mlp_sum / cap_count[0]).to("cuda", dtype=torch.bfloat16)
        mean_attn = torch.from_numpy(cap_attn_sum / cap_count[0]).to("cuda", dtype=torch.bfloat16)
        CAP["mean_mlp"] = mean_mlp
        CAP["mean_attn"] = mean_attn
        base_ints = [codi_extract(s) for s in base_strs]
        base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                  for i, v in enumerate(base_ints)])
        base_acc = float(base_correct.mean())
        print(f"  N={N}, baseline accuracy={base_acc*100:.1f}%  (capture+baseline in {time.time()-t0:.0f}s)", flush=True)

        # Pass 2: ablate each cell
        CAP["mode"] = "ablate"
        per_cell = {}
        t1 = time.time()
        n_cells = N_LAT * N_LAYERS * 2
        ci = 0
        for step in range(N_LAT):
            for layer in range(N_LAYERS):
                for block in ("mlp", "attn"):
                    CAP["ablate_step"] = step
                    CAP["ablate_layer"] = layer
                    CAP["ablate_block"] = block
                    strs = run_all(qs)
                    ints = [codi_extract(s) for s in strs]
                    correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                        for i, v in enumerate(ints)])
                    n_changed = int(sum(1 for i in range(N) if ints[i] != base_ints[i]))
                    wr = int(((~base_correct) & correct).sum())
                    rw = int((base_correct & ~correct).sum())
                    per_cell[f"step{step+1}_L{layer}_{block}"] = {
                        "accuracy": float(correct.mean()),
                        "delta_acc": float(correct.mean() - base_acc),
                        "n_correct": int(correct.sum()), "n_changed": n_changed,
                        "wrong_to_right": wr, "right_to_wrong": rw,
                    }
                    ci += 1
                    if ci % 12 == 0 or ci == n_cells:
                        print(f"    [{ci:3d}/{n_cells}]  ({time.time()-t1:.0f}s)", flush=True)
        results["cf_sets"][cf_name] = {
            "N": N, "baseline_accuracy": base_acc,
            "baseline_n_correct": int(base_correct.sum()),
            "conditions": per_cell,
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
