"""Paired-CF activation patching (interchange intervention) on vary_numerals.

Mean-CF patching (see patch_cf_mean.py) under-localized because the CF residuals
cluster tightly: replacing an activation with the population mean barely moves
it. Paired-CF instead swaps in a SPECIFIC source example's activation, so the
intervention has real force.

Setup
-----
For each example A_i in vary_numerals, pick a paired source example B_{π(i)}
from the same CF set such that they differ in numerals (and therefore differ
in their gold answer). Capture B's activations across every (step, layer,
block) cell. Then run A through the model, but at one cell replace A's
activation with B's. Read off the resulting prediction.

Per cell we record, over the N pairs:
  - n_followed_source: predictions that equal B's baseline answer (clean
    interchange — the cell carried B's answer to A)
  - n_followed_target: predictions that equal A's baseline answer (no
    transfer — the cell was uninvolved or downstream computation overrode it)
  - n_other: predictions that changed but matched neither A nor B (the cell
    is necessary but doesn't itself carry the answer — gate / structural role)
  - n_unparseable: empty/garbage predictions
  - transfer_rate = n_followed_source / N

A clean causal map: transfer_rate close to 1 means the cell is sufficient to
carry the answer; transfer_rate close to 0 means it isn't where the answer
lives. n_other tells the gating story.

Pairing
-------
Random derangement of indices with seed=0. Ensures no self-pairs and (for
vary_numerals) guarantees A.gold ≠ B.gold for ~all pairs (we double-check
and re-derange if any collision).

Output: patch_paired_cf.json (and a plotting script will turn it into a PDF
heatmap separately).
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

REPO = Path(__file__).resolve().parents[3]   # codi-work/
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

# Target CF set. vary_numerals varies both operands so answers differ widely
# across pairs — best contrast for interchange patching.
CF_SETS = ["vary_numerals", "vary_both_2digit"]

SEED = 0


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


def derangement(n: int, rng: np.random.Generator, max_tries: int = 100):
    """Return a permutation pi with pi[i] != i for all i."""
    for _ in range(max_tries):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p
    # Fallback: cyclic shift by 1.
    return np.r_[np.arange(1, n), 0]


def main():
    BS = 16
    OUT_JSON = PD / "patch_paired_cf.json"

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
    targs = TrainingArguments(output_dir="/tmp/_pp", bf16=True,
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
        "mode": "off",
        "step": -1,
        "ex_idx_in_batch": None,    # numpy array of dataset indices for the batch
        # Single-cell patch knobs (sweep A):
        "patch_step": -1,
        "patch_layer": -1,
        "patch_block": "",          # "attn" | "mlp" | "resid"
        # Whole-step patch (sweep B): patch every layer's attn AND mlp at this
        # step.  When patch_block == "step_all" we ignore patch_layer.
        # Per-step capture buffers:
        "cap_attn": None,           # (N, N_LAT, N_LAYERS, HID) bf16 CPU tensor
        "cap_mlp": None,
        "cap_resid": None,          # (N, N_LAT, N_LAYERS, HID) bf16 CPU tensor
        "patch_attn": None,         # (N, N_LAT, N_LAYERS, HID) — source acts to inject
        "patch_mlp": None,
        "patch_resid": None,
    }

    def make_attn_hook(idx):
        def fn(_module, _inputs, output):
            mode = CAP["mode"]
            if mode == "capture" and CAP["step"] >= 0:
                a = output[0]
                last = a[:, -1, :].detach().to(torch.bfloat16).cpu()
                CAP["cap_attn"][CAP["ex_idx_in_batch"], CAP["step"], idx, :] = last
            elif mode == "patch" and CAP["step"] == CAP["patch_step"]:
                pb = CAP["patch_block"]
                if (pb == "attn" and idx == CAP["patch_layer"]) or pb == "step_all":
                    a = output[0].clone()
                    src = CAP["patch_attn"][CAP["ex_idx_in_batch"], CAP["step"], idx, :]
                    a[:, -1, :] = src.to(a.device, dtype=a.dtype)
                    return (a,) + output[1:]
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_module, _inputs, output):
            mode = CAP["mode"]
            if mode == "capture" and CAP["step"] >= 0:
                last = output[:, -1, :].detach().to(torch.bfloat16).cpu()
                CAP["cap_mlp"][CAP["ex_idx_in_batch"], CAP["step"], idx, :] = last
            elif mode == "patch" and CAP["step"] == CAP["patch_step"]:
                pb = CAP["patch_block"]
                if (pb == "mlp" and idx == CAP["patch_layer"]) or pb == "step_all":
                    o = output.clone()
                    src = CAP["patch_mlp"][CAP["ex_idx_in_batch"], CAP["step"], idx, :]
                    o[:, -1, :] = src.to(o.device, dtype=o.dtype)
                    return o
            return output
        return fn

    def make_block_hook(idx):
        """Hook on the GPT2Block's forward — captures / patches the full
        residual stream after both attn and mlp have been added at layer idx.
        Block forward returns a tuple whose first element is the hidden state.
        """
        def fn(_module, _inputs, output):
            mode = CAP["mode"]
            if mode == "capture" and CAP["step"] >= 0:
                h = output[0] if isinstance(output, tuple) else output
                last = h[:, -1, :].detach().to(torch.bfloat16).cpu()
                CAP["cap_resid"][CAP["ex_idx_in_batch"], CAP["step"], idx, :] = last
            elif (mode == "patch"
                  and CAP["step"] == CAP["patch_step"]
                  and idx == CAP["patch_layer"]
                  and CAP["patch_block"] == "resid"):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                src = CAP["patch_resid"][CAP["ex_idx_in_batch"], CAP["step"], idx, :]
                h[:, -1, :] = src.to(h.device, dtype=h.dtype)
                if isinstance(output, tuple):
                    return (h,) + output[1:]
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

    def run_all_with_idx(qs, idx_arr):
        out = []
        for s in range(0, len(qs), BS):
            CAP["ex_idx_in_batch"] = np.asarray(idx_arr[s:s+BS], dtype=np.int64)
            out += run_batch(qs[s:s+BS])
        CAP["ex_idx_in_batch"] = None
        return out

    results = {"cf_sets": {}, "N_LAYERS": N_LAYERS, "N_LAT": N_LAT, "seed": SEED}

    for cf_name in CF_SETS:
        print(f"\n=== CF set: {cf_name} ===", flush=True)
        try:
            qs, golds = load_cf(cf_name)
        except Exception as e:
            print(f"  skip: {e}"); continue
        N = len(qs)
        golds_arr = np.array(golds)

        # Allocate per-example capture buffers (CPU bf16 ~ N * 6 * 12 * 768 * 2B
        # ≈ 1.7 MB per buffer for N=80; three buffers ≈ 5 MB).
        CAP["cap_attn"] = torch.zeros((N, N_LAT, N_LAYERS, HID), dtype=torch.bfloat16)
        CAP["cap_mlp"]  = torch.zeros((N, N_LAT, N_LAYERS, HID), dtype=torch.bfloat16)
        CAP["cap_resid"] = torch.zeros((N, N_LAT, N_LAYERS, HID), dtype=torch.bfloat16)

        # PASS 1 — capture per-example activations on the clean run.
        CAP["mode"] = "capture"
        t0 = time.time()
        base_strs = run_all_with_idx(qs, np.arange(N))
        base_ints = [codi_extract(s) for s in base_strs]
        base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                  for i, v in enumerate(base_ints)])
        base_acc = float(base_correct.mean())
        print(f"  N={N}, baseline accuracy={base_acc*100:.1f}%  "
              f"(capture+baseline in {time.time()-t0:.0f}s)", flush=True)

        # Build pairing: derangement, but also enforce A.pred != B.pred so
        # that "followed source" is meaningful.
        rng = np.random.default_rng(SEED)
        for attempt in range(50):
            pi = derangement(N, rng)
            collisions = sum(1 for i in range(N)
                             if base_ints[i] is not None
                             and base_ints[pi[i]] is not None
                             and base_ints[i] == base_ints[pi[i]])
            if collisions <= N // 20:
                break
        # Pairing source acts: gather B = pi[i] for each i.
        CAP["patch_attn"] = CAP["cap_attn"][pi].clone()
        CAP["patch_mlp"]  = CAP["cap_mlp"][pi].clone()
        CAP["patch_resid"] = CAP["cap_resid"][pi].clone()
        # Source baseline predictions for the (paired) source = pi[i].
        source_ints = [base_ints[pi[i]] for i in range(N)]
        print(f"  pairing derangement seed={SEED}, prediction-collisions={collisions} / {N}", flush=True)

        def score(strs):
            ints = [codi_extract(s) for s in strs]
            n_src = n_tgt = n_other = n_unp = n_gold_b = n_gold_a = 0
            eq = lambda a, b: a is not None and b is not None and abs(a - b) < 1e-3
            for i in range(N):
                v = ints[i]
                if v is None:
                    n_unp += 1; continue
                si = source_ints[i]; ti = base_ints[i]
                gA = golds_arr[i]; gB = golds_arr[pi[i]]
                if eq(v, si): n_src += 1
                elif eq(v, ti): n_tgt += 1
                else: n_other += 1
                if eq(v, gB): n_gold_b += 1
                if eq(v, gA): n_gold_a += 1
            return {
                "transfer_rate": n_src / N,
                "n_followed_source": n_src,
                "n_followed_target": n_tgt,
                "n_other": n_other,
                "n_unparseable": n_unp,
                "n_followed_gold_b": n_gold_b,
                "n_followed_gold_a": n_gold_a,
            }

        # PASS 2 — three sweeps with progressively coarser intervention.
        #   A) per-cell  (step, layer, attn|mlp)         → 144 cells
        #   B) per-cell  (step, layer, resid)            → 72 cells
        #   C) per-step  (all layers' attn+mlp at step) → 6 cells
        CAP["mode"] = "patch"
        per_cell = {}
        ci = 0
        n_cells_total = N_LAT * N_LAYERS * 2 + N_LAT * N_LAYERS + N_LAT
        t1 = time.time()

        # Sweep A: single (attn|mlp) cell per (step, layer).
        for step in range(N_LAT):
            for layer in range(N_LAYERS):
                for block in ("mlp", "attn"):
                    CAP["patch_step"] = step
                    CAP["patch_layer"] = layer
                    CAP["patch_block"] = block
                    strs = run_all_with_idx(qs, np.arange(N))
                    per_cell[f"step{step+1}_L{layer}_{block}"] = score(strs)
                    ci += 1
                    if ci % 24 == 0:
                        print(f"    A [{ci:3d}/{n_cells_total}]  "
                              f"({time.time()-t1:.0f}s)  "
                              f"latest = {per_cell[f'step{step+1}_L{layer}_{block}']['transfer_rate']:.2f}",
                              flush=True)

        # Sweep B: residual stream after block L (cumulative state).
        for step in range(N_LAT):
            for layer in range(N_LAYERS):
                CAP["patch_step"] = step
                CAP["patch_layer"] = layer
                CAP["patch_block"] = "resid"
                strs = run_all_with_idx(qs, np.arange(N))
                per_cell[f"step{step+1}_L{layer}_resid"] = score(strs)
                ci += 1
                if ci % 24 == 0:
                    print(f"    B [{ci:3d}/{n_cells_total}]  "
                          f"({time.time()-t1:.0f}s)  "
                          f"latest = {per_cell[f'step{step+1}_L{layer}_resid']['transfer_rate']:.2f}",
                          flush=True)

        # Sweep C: whole step (every layer's attn AND mlp at this step).
        for step in range(N_LAT):
            CAP["patch_step"] = step
            CAP["patch_layer"] = -1  # ignored
            CAP["patch_block"] = "step_all"
            strs = run_all_with_idx(qs, np.arange(N))
            per_cell[f"step{step+1}_ALL"] = score(strs)
            ci += 1
            print(f"    C [{ci:3d}/{n_cells_total}]  "
                  f"({time.time()-t1:.0f}s)  "
                  f"step{step+1}_ALL transfer = {per_cell[f'step{step+1}_ALL']['transfer_rate']:.2f}",
                  flush=True)

        results["cf_sets"][cf_name] = {
            "N": N,
            "baseline_accuracy": base_acc,
            "baseline_n_correct": int(base_correct.sum()),
            "pairing": pi.tolist(),
            "base_preds": [None if v is None else float(v) for v in base_ints],
            "source_preds": [None if v is None else float(v) for v in source_ints],
            "golds": golds,
            "conditions": per_cell,
        }
        # Free the big tensors before the next CF set.
        CAP["cap_attn"] = None; CAP["cap_mlp"] = None; CAP["cap_resid"] = None
        CAP["patch_attn"] = None; CAP["patch_mlp"] = None; CAP["patch_resid"] = None

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
