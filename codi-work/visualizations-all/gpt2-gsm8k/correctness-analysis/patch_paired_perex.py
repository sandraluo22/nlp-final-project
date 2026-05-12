"""Multi-step paired-CF patching + per-example dumps.

Two questions this script answers:
1) Are steps 2-6 "repairing" the patch?  Sweep K ∈ {1, 2, 3, 4, 5, 6}: patch
   steps 1..K simultaneously with source B's residuals.  If transfer rises
   with K, downstream steps were absorbing the perturbation.  If it stays at
   0, even the full latent loop's residuals can't carry B's answer.
2) Among examples that DO break under the most disruptive cell, what's the
   pattern?  Save per-example predictions so we can join them with operator,
   magnitudes, baseline correctness.

For each multi-step setting we record:
  - n_followed_source / n_followed_target / n_other / n_unparseable
  - per-example list of (target_pred, source_pred, patched_pred, gold_a, gold_b)
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
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

CF_SETS = ["gsm8k_vary_operator"]
SEED = 0
OUT_JSON = PD / "patch_paired_perex_gsm8k.json"


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
    return qs, golds, rows


def derangement(n, rng, max_tries=100):
    for _ in range(max_tries):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p
    return np.r_[np.arange(1, n), 0]


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
    targs = TrainingArguments(output_dir="/tmp/_px", bf16=True,
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
    model = model.to("cuda").to(torch.bfloat16); model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id

    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    N_LAYERS = model.codi.config.n_layer
    HID = model.codi.config.n_embd
    N_LAT = 6

    CAP = {
        "mode": "off", "step": -1, "ex_idx_in_batch": None,
        "patch_steps": set(),   # set of step indices to patch (multi-step support)
        "patch_layers": set(),  # set of layer indices to patch (e.g. all 12 for whole-step)
        "patch_block": "",      # "attn" | "mlp" | "resid" | "step_all"
        "cap_attn": None, "cap_mlp": None, "cap_resid": None,
        "patch_attn": None, "patch_mlp": None, "patch_resid": None,
    }

    def make_attn_hook(idx):
        def fn(_m, _i, output):
            m = CAP["mode"]
            if m == "capture" and CAP["step"] >= 0:
                last = output[0][:, -1, :].detach().to(torch.bfloat16).cpu()
                CAP["cap_attn"][CAP["ex_idx_in_batch"], CAP["step"], idx, :] = last
            elif m == "patch" and CAP["step"] in CAP["patch_steps"]:
                pb = CAP["patch_block"]
                if (pb == "attn" and idx in CAP["patch_layers"]) or pb == "step_all":
                    a = output[0].clone()
                    src = CAP["patch_attn"][CAP["ex_idx_in_batch"], CAP["step"], idx, :]
                    a[:, -1, :] = src.to(a.device, dtype=a.dtype)
                    return (a,) + output[1:]
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_m, _i, output):
            m = CAP["mode"]
            if m == "capture" and CAP["step"] >= 0:
                last = output[:, -1, :].detach().to(torch.bfloat16).cpu()
                CAP["cap_mlp"][CAP["ex_idx_in_batch"], CAP["step"], idx, :] = last
            elif m == "patch" and CAP["step"] in CAP["patch_steps"]:
                pb = CAP["patch_block"]
                if (pb == "mlp" and idx in CAP["patch_layers"]) or pb == "step_all":
                    o = output.clone()
                    src = CAP["patch_mlp"][CAP["ex_idx_in_batch"], CAP["step"], idx, :]
                    o[:, -1, :] = src.to(o.device, dtype=o.dtype)
                    return o
            return output
        return fn

    def make_block_hook(idx):
        def fn(_m, _i, output):
            m = CAP["mode"]
            if m == "capture" and CAP["step"] >= 0:
                h = output[0] if isinstance(output, tuple) else output
                last = h[:, -1, :].detach().to(torch.bfloat16).cpu()
                CAP["cap_resid"][CAP["ex_idx_in_batch"], CAP["step"], idx, :] = last
            elif (m == "patch" and CAP["step"] in CAP["patch_steps"]
                  and idx in CAP["patch_layers"]
                  and CAP["patch_block"] == "resid"):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                src = CAP["patch_resid"][CAP["ex_idx_in_batch"], CAP["step"], idx, :]
                h[:, -1, :] = src.to(h.device, dtype=h.dtype)
                if isinstance(output, tuple): return (h,) + output[1:]
                return h
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
        tokens = [[] for _ in range(B)]; done = [False] * B
        for _ in range(64):
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past)
            past = s.past_key_values
            nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
            for b in range(B):
                if done[b]: continue
                tid = int(nid[b].item()); tokens[b].append(tid)
                if tid == eos_id: done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(nid).unsqueeze(1)
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    def run_all_with_idx(qs, idx_arr):
        out = []
        for s in range(0, len(qs), BS):
            CAP["ex_idx_in_batch"] = np.asarray(idx_arr[s:s+BS], dtype=np.int64)
            out += run_batch(qs[s:s+BS])
        CAP["ex_idx_in_batch"] = None
        return out

    results = {}
    for cf_name in CF_SETS:
        print(f"\n=== CF set: {cf_name} ===", flush=True)
        qs, golds, rows = load_cf(cf_name)
        N = len(qs); golds_arr = np.array(golds)

        # Capture
        CAP["cap_attn"] = torch.zeros((N, N_LAT, N_LAYERS, HID), dtype=torch.bfloat16)
        CAP["cap_mlp"]  = torch.zeros((N, N_LAT, N_LAYERS, HID), dtype=torch.bfloat16)
        CAP["cap_resid"] = torch.zeros((N, N_LAT, N_LAYERS, HID), dtype=torch.bfloat16)
        CAP["mode"] = "capture"
        base_strs = run_all_with_idx(qs, np.arange(N))
        base_ints = [codi_extract(s) for s in base_strs]
        base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                  for i, v in enumerate(base_ints)])
        print(f"  N={N}, baseline acc={base_correct.mean()*100:.1f}%", flush=True)

        # Pairing
        rng = np.random.default_rng(SEED)
        for _ in range(50):
            pi = derangement(N, rng)
            coll = sum(1 for i in range(N)
                       if base_ints[i] is not None and base_ints[pi[i]] is not None
                       and base_ints[i] == base_ints[pi[i]])
            if coll <= N // 20: break
        CAP["patch_attn"] = CAP["cap_attn"][pi].clone()
        CAP["patch_mlp"]  = CAP["cap_mlp"][pi].clone()
        CAP["patch_resid"] = CAP["cap_resid"][pi].clone()
        source_ints = [base_ints[pi[i]] for i in range(N)]

        CAP["mode"] = "patch"
        cf_results = {
            "N": N, "baseline_accuracy": float(base_correct.mean()),
            "pairing": pi.tolist(),
            "base_preds": [None if v is None else float(v) for v in base_ints],
            "source_preds": [None if v is None else float(v) for v in source_ints],
            "golds": golds, "rows": rows,
            "conditions": {},
        }

        def score_and_save(strs, key):
            ints = [codi_extract(s) for s in strs]
            n_src = n_tgt = n_oth = n_unp = 0
            eq = lambda a, b: a is not None and b is not None and abs(a - b) < 1e-3
            per_ex = []
            for i in range(N):
                v = ints[i]; si = source_ints[i]; ti = base_ints[i]
                if v is None:
                    cat = "unparseable"; n_unp += 1
                elif eq(v, si):
                    cat = "followed_source"; n_src += 1
                elif eq(v, ti):
                    cat = "followed_target"; n_tgt += 1
                else:
                    cat = "other"; n_oth += 1
                per_ex.append({
                    "idx": i, "pi": int(pi[i]),
                    "patched": None if v is None else float(v),
                    "target_pred": None if ti is None else float(ti),
                    "source_pred": None if si is None else float(si),
                    "gold_a": golds_arr[i], "gold_b": golds_arr[pi[i]],
                    "category": cat,
                })
            cf_results["conditions"][key] = {
                "n_followed_source": n_src, "n_followed_target": n_tgt,
                "n_other": n_oth, "n_unparseable": n_unp,
                "transfer_rate": n_src / N,
                "per_example": per_ex,
            }
            return cf_results["conditions"][key]

        # Sweep D: patch steps 1..K simultaneously, whole-step (all layers attn+mlp).
        for K in range(1, N_LAT + 1):
            CAP["patch_steps"] = set(range(K))
            CAP["patch_layers"] = set()
            CAP["patch_block"] = "step_all"
            t0 = time.time()
            strs = run_all_with_idx(qs, np.arange(N))
            r = score_and_save(strs, f"steps_1to{K}_ALL")
            print(f"  multi-step 1..{K}_ALL: transfer={r['transfer_rate']:.2f}  "
                  f"src={r['n_followed_source']:3d}  tgt={r['n_followed_target']:3d}  "
                  f"other={r['n_other']:3d}  unp={r['n_unparseable']:3d}  ({time.time()-t0:.0f}s)",
                  flush=True)

        # Sweep E: most disruptive single cell — step1 resid L0 — save per-example.
        CAP["patch_steps"] = {0}
        CAP["patch_layers"] = {0}
        CAP["patch_block"] = "resid"
        strs = run_all_with_idx(qs, np.arange(N))
        r = score_and_save(strs, "step1_L0_resid")
        print(f"  step1_L0_resid: transfer={r['transfer_rate']:.2f}  "
              f"src={r['n_followed_source']}  tgt={r['n_followed_target']}  "
              f"other={r['n_other']}", flush=True)

        # Sweep F: residual stream patched at step1 across ALL 12 layers.
        CAP["patch_steps"] = {0}
        CAP["patch_layers"] = set(range(N_LAYERS))
        CAP["patch_block"] = "resid"
        strs = run_all_with_idx(qs, np.arange(N))
        r = score_and_save(strs, "step1_resid_ALL_layers")
        print(f"  step1_resid_ALL_layers: transfer={r['transfer_rate']:.2f}  "
              f"src={r['n_followed_source']}  tgt={r['n_followed_target']}  "
              f"other={r['n_other']}  unp={r['n_unparseable']}", flush=True)

        # Sweep G: patch ALL 6 steps' residuals at ALL layers (most aggressive).
        CAP["patch_steps"] = set(range(N_LAT))
        CAP["patch_layers"] = set(range(N_LAYERS))
        CAP["patch_block"] = "resid"
        strs = run_all_with_idx(qs, np.arange(N))
        r = score_and_save(strs, "ALL_steps_resid_ALL_layers")
        print(f"  ALL_steps_resid_ALL_layers: transfer={r['transfer_rate']:.2f}  "
              f"src={r['n_followed_source']}  tgt={r['n_followed_target']}  "
              f"other={r['n_other']}  unp={r['n_unparseable']}", flush=True)

        results[cf_name] = cf_results
        CAP["cap_attn"] = None; CAP["cap_mlp"] = None; CAP["cap_resid"] = None
        CAP["patch_attn"] = None; CAP["patch_mlp"] = None; CAP["patch_resid"] = None

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
