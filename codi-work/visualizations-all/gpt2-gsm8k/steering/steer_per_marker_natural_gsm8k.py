"""Marker-specific causal add↔mul steering on gsm8k_cf_natural.

For each marker position m ∈ {1..4}:
  1. Look up that marker's "best cell" (step_m, layer_m) from the REFINED probe
     (correct + length-matched filter) in multi_op_probe_refined_gsm8k.json.
  2. Fit a 4-class LDA at (step_m, layer_m) on gsm8k_latent_acts.pt using
     marker m's first-operator labels.
  3. Extract the add→mul direction at that cell.

Then on cf_natural problems with ≥2 markers (chains where different m's can
have different operators):
  - Compute the baseline (no-steer) emitted answer per problem.
  - Compute alternative chain answers: for each (j, new_op), what the answer
    would be if ONLY the j-th marker's operator were replaced with new_op.
  - For each m_steered ∈ {1..4}: hook at cell_{m_steered}, add α·v_{m_steered}.
    Decode, parse the emitted answer, score against:
       baseline  : no flip
       target    : matches alt(m_steered → mul)
       off_target: matches alt(j → mul) for some j ≠ m_steered

The marker-specificity ratio per α:
   P(target | steered at m) − P(off_target | steered at m)

High ratio → cell_m causally controls *only* marker m. Low/zero → intervention
non-specifically scrambles the chain.

Outputs:
  steer_per_marker_natural_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, math, os, re, sys, time
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
REFINED_JSON = REPO / "visualizations-all" / "gpt2-gsm8k" / "operator-probe" / "multi_op_probe_refined_gsm8k.json"
CODI_COT_JSON = REPO.parent / "cf-datasets" / "gsm8k_codi_cot.json"
sys.path.insert(0, str(REPO / "codi"))

OUT_JSON = PD / "steer_per_marker_natural_gsm8k.json"
OUT_PDF = PD / "steer_per_marker_natural_gsm8k.pdf"

M_MAX = 4
OPS = ["+", "-", "*", "/"]
OP_NAMES = {"+": "Addition", "-": "Subtraction",
            "*": "Multiplication", "/": "Common-Division"}
NAME_TO_OP = {v: k for k, v in OP_NAMES.items()}


def parse_markers(s):
    s = s.replace(",", "")
    ms = re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)
    return [(float(a), op, float(b), float(c)) for a, op, b, c in ms]


def codi_extract(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def apply_op(a, op, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/":
        return a / b if b != 0 else None
    return None


def chain_answer(markers, ops_override=None):
    """Compute the left-fold chain result. If ops_override is given, replace
    each operator with the override; otherwise use the markers' original ops.
    markers: list of (a, op, b, c). For a left-fold chain, marker_k's 'a' is
    marker_{k-1}'s c (for k>0)."""
    if not markers: return None
    cur = markers[0][0]  # initial operand a_1
    for k, (a, op, b, c) in enumerate(markers):
        use_op = (ops_override[k] if ops_override else op)
        cur = apply_op(cur, use_op, b)
        if cur is None: return None
    # round to integer if very close
    if abs(cur - round(cur)) < 1e-6: cur = round(cur)
    return float(cur)


def main():
    BS = 8
    print("loading refined probe results", flush=True)
    refined = json.load(open(REFINED_JSON))
    pref_filter = "correct_and_length_matched" if "correct_and_length_matched" in refined else \
                  ("correct_only" if "correct_only" in refined else "length_matched")
    print(f"  using filter: {pref_filter}")
    cells = {}
    for m in range(1, M_MAX + 1):
        info = refined[pref_filter].get("op", {}).get(str(m))
        if info is None or info["best"] is None:
            print(f"  m={m}: no best cell in refined; falling back")
            info = refined.get("original", {}).get("op", {}).get(str(m))
        if info is None or info["best"] is None:
            print(f"  m={m}: no usable cell, skipping"); continue
        cells[m] = {"step_1idx": info["best"]["step"],
                    "layer": info["best"]["layer"],
                    "probe_acc": info["best"]["acc"]}
        print(f"  m={m}: cell = step{cells[m]['step_1idx']} L{cells[m]['layer']}  "
              f"(probe_acc={cells[m]['probe_acc']:.3f})")

    # --- Fit per-marker LDA at each cell using gsm8k_latent_acts.pt ---
    print("loading gsm8k activations + GSM8K labels", flush=True)
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    ds = load_dataset("gsm8k", "main")["test"]
    markers_per_problem = [parse_markers(ex["answer"]) for ex in ds]
    N = len(markers_per_problem)

    # Try to load CODI-correct mask
    correct_mask = np.zeros(N, dtype=bool)
    try:
        cot = json.load(open(CODI_COT_JSON))
        emits = np.array([r.get("codi_pred_int", np.nan) for r in cot], dtype=float)
        golds = np.array([r.get("gold", np.nan) for r in cot], dtype=float)
        correct_mask = (~np.isnan(emits)) & (~np.isnan(golds)) & (np.abs(emits - golds) < 1e-3)
        print(f"  CODI-correct: {correct_mask.sum()}/{N}")
    except Exception as e:
        print(f"  no CODI-correct mask: {e}")

    v_per_m = {}
    for m in cells:
        s_idx = cells[m]["step_1idx"] - 1  # 0-indexed
        ly = cells[m]["layer"]
        # Filter to length-matched (chain length == m) AND CODI-correct
        len_eq_m = np.array([len(ms) == m for ms in markers_per_problem])
        idx = np.where(len_eq_m & correct_mask)[0] if correct_mask.any() else np.where(len_eq_m)[0]
        if len(idx) < 20:
            print(f"  m={m}: too few clean examples ({len(idx)}); using length-matched only")
            idx = np.where(len_eq_m)[0]
        y_names = np.array([OP_NAMES.get(markers_per_problem[i][m-1][1], "") for i in idx])
        keep = y_names != ""
        idx = idx[keep]; y_names = y_names[keep]
        if "Addition" not in y_names or "Multiplication" not in y_names:
            print(f"  m={m}: missing Add or Mul in training set; skipping"); continue
        X = acts[idx, s_idx, ly, :]
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        lda = LinearDiscriminantAnalysis(solver="svd").fit(Xs, y_names)
        cl = list(lda.classes_)
        add_i = cl.index("Addition"); mul_i = cl.index("Multiplication")
        v_std = lda.means_[mul_i] - lda.means_[add_i]
        v_orig = sc.scale_ * v_std
        v_unit = v_orig / max(np.linalg.norm(v_orig), 1e-9)
        v_per_m[m] = v_unit
        print(f"  m={m}: LDA at step{cells[m]['step_1idx']} L{ly} fit on N={len(idx)} "
              f"(classes: {dict(zip(*np.unique(y_names, return_counts=True)))})")

    # --- Build cf_natural eval set ---
    rows = json.load(open(CF_DIR / "gsm8k_cf_natural.json"))
    print(f"cf_natural N={len(rows)}")
    eval_data = []
    for r in rows:
        ms_tr = parse_markers(r["answer_trace"])
        if not (2 <= len(ms_tr) <= 4): continue
        n_distinct = len(set(op for _, op, _, _ in ms_tr))
        if n_distinct < 2: continue   # need mixed ops to detect specificity
        base_ans = chain_answer(ms_tr)
        if base_ans is None or abs(base_ans - float(r["answer"])) > 1e-3: continue
        # Per-(j, new_op) alternative answers (only flip the j-th marker)
        alt = {}
        K = len(ms_tr)
        orig_ops = [op for _, op, _, _ in ms_tr]
        for j in range(K):
            for new_op in OPS:
                if new_op == orig_ops[j]: continue
                ovr = list(orig_ops); ovr[j] = new_op
                v = chain_answer(ms_tr, ovr)
                if v is None: continue
                alt[(j + 1, new_op)] = v
        eval_data.append({
            "q": r["question_concat"].strip().replace("  ", " "),
            "operands": r["operands"], "K": K,
            "orig_ops": orig_ops, "gold": float(r["answer"]),
            "alt": alt,
        })
    # cap
    eval_data = eval_data[:80]
    print(f"  usable mixed-op problems: {len(eval_data)} (capped)")

    # --- Load CODI ---
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
    targs = TrainingArguments(output_dir="/tmp/_steerpm", bf16=True,
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
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id
    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)

    HOOK = {"active": False, "latent_step": -1, "target_step": -1,
            "target_layer": -1, "alpha": 0.0, "vec": None}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if block_idx != HOOK["target_layer"]: return output
            if HOOK["latent_step"] != HOOK["target_step"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, target_step, target_layer, alpha, vec, max_new=128):
        B = len(qs)
        HOOK["target_step"] = target_step; HOOK["target_layer"] = target_layer
        HOOK["alpha"] = float(alpha); HOOK["vec"] = vec
        HOOK["active"] = True; HOOK["latent_step"] = -1
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
            HOOK["latent_step"] = s
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        HOOK["active"] = False; HOOK["latent_step"] = -1
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]; done = [False] * B
        for _ in range(max_new):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if not done[b]:
                    tokens[b].append(int(next_ids[b].item()))
                    if int(next_ids[b].item()) == eos_id: done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    # Score a batch against (target_m, target_op) flip
    def classify(emit_val, ex_d, m_steered, target_op):
        if emit_val is None: return "unparsed"
        if abs(emit_val - ex_d["gold"]) < 1e-3: return "baseline"
        # check if matches alt[m_steered → target_op]
        v_target = ex_d["alt"].get((m_steered, target_op))
        if v_target is not None and abs(emit_val - v_target) < 1e-3:
            return "target"
        # off-target: any OTHER j → target_op matches
        for (j, op), v in ex_d["alt"].items():
            if op == target_op and j != m_steered and abs(emit_val - v) < 1e-3:
                return "off_target"
        # any-other-flip: matches some (j, op) but not exactly (m_steered, target_op)
        for (j, op), v in ex_d["alt"].items():
            if abs(emit_val - v) < 1e-3:
                return "other_alt"
        return "scrambled"

    # --- Sweep ---
    qs = [d["q"] for d in eval_data]
    alphas = [-80, -40, -20, -10, 0, 10, 20, 40, 80]

    results = {
        "cells": cells, "filter": pref_filter,
        "n_problems": len(eval_data), "alphas": alphas,
        "per_m": {},   # results per m_steered
    }

    t0 = time.time()
    # Run baseline once (α=0): no actual hook effect; record emitted answers
    HOOK_DUMMY_V = torch.zeros(768, dtype=torch.float32)
    print("\n=== baseline (α=0) ===", flush=True)
    base_strs = []
    for s in range(0, len(qs), BS):
        base_strs += run_batch(qs[s:s+BS], target_step=-99, target_layer=-99,
                                alpha=0.0, vec=HOOK_DUMMY_V)
    base_vals = [codi_extract(s) for s in base_strs]
    n_emit = sum(1 for v in base_vals if v is not None)
    n_match_gold = sum(1 for v, d in zip(base_vals, eval_data)
                       if v is not None and abs(v - d["gold"]) < 1e-3)
    print(f"  baseline acc: {n_match_gold}/{len(eval_data)} matched gold, "
          f"{n_emit}/{len(eval_data)} parseable")
    results["baseline"] = {
        "n_match_gold": n_match_gold, "n_emit": n_emit, "n": len(eval_data),
        "emits": base_strs[:10],   # sample
    }

    for m_steered in cells:
        if m_steered not in v_per_m:
            print(f"\n=== m={m_steered}: no LDA direction; skipping ==="); continue
        target_step = cells[m_steered]["step_1idx"] - 1
        target_layer = cells[m_steered]["layer"]
        v_torch = torch.tensor(v_per_m[m_steered], dtype=torch.float32)
        results["per_m"][m_steered] = {}
        for alpha in alphas:
            sign_dir = "add→mul" if alpha > 0 else ("mul→add" if alpha < 0 else "baseline")
            print(f"\n=== m={m_steered} (cell step{target_step+1} L{target_layer}), "
                  f"α={alpha} ({sign_dir}) ===", flush=True)
            strs = []
            for s in range(0, len(qs), BS):
                strs += run_batch(qs[s:s+BS], target_step=target_step,
                                  target_layer=target_layer,
                                  alpha=alpha, vec=v_torch)
            vals = [codi_extract(s) for s in strs]
            # Choose target_op: positive alpha means push toward mul; negative toward add.
            target_op = "*" if alpha > 0 else "+"
            counts = {"baseline": 0, "target": 0, "off_target": 0,
                      "other_alt": 0, "scrambled": 0, "unparsed": 0}
            for v, d in zip(vals, eval_data):
                counts[classify(v, d, m_steered, target_op)] += 1
            results["per_m"][m_steered][str(alpha)] = {
                "target_op": target_op, "counts": counts,
                "emits": strs[:5],
            }
            elapsed = time.time() - t0
            print(f"  target={target_op}  baseline={counts['baseline']}  "
                  f"target={counts['target']}  off_target={counts['off_target']}  "
                  f"other_alt={counts['other_alt']}  scrambled={counts['scrambled']}  "
                  f"unparsed={counts['unparsed']}  ({elapsed:.0f}s)")
            OUT_JSON.write_text(json.dumps(results, indent=2))

    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()

    # --- Plot ---
    with PdfPages(OUT_PDF) as pdf:
        # Title
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = ("Marker-specific add↔mul steering on gsm8k_cf_natural\n\n"
                f"  Filter for cell selection: {pref_filter}\n"
                f"  Cells per marker (probe accuracy):\n")
        for m, c in cells.items():
            body += f"    m={m}: step{c['step_1idx']} L{c['layer']}  probe_acc={c['probe_acc']:.3f}\n"
        body += f"\n  Eval set: {len(eval_data)} mixed-op cf_natural problems\n"
        body += f"  Baseline acc on mixed-op subset: {n_match_gold}/{len(eval_data)}\n\n"
        body += "Each panel below = one m_steered. X-axis = α. Curves:\n"
        body += "  target      : emit matches alt(m_steered → target_op) (clean flip)\n"
        body += "  off_target  : emit matches alt(j ≠ m → target_op) (specificity failure)\n"
        body += "  baseline    : emit matches gold (no flip)\n"
        body += "  scrambled   : emit doesn't match any expected variant\n"
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Per-m panels
        fig, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 5))
        if len(cells) == 1: axes = [axes]
        for col, m in enumerate(cells):
            if m not in results["per_m"]: continue
            ax = axes[col]
            data = results["per_m"][m]
            a_keys = sorted(data.keys(), key=lambda x: float(x))
            a_vals = [float(a) for a in a_keys]
            n = len(eval_data)
            for key, color in [("baseline", "C0"), ("target", "C2"),
                               ("off_target", "C3"), ("scrambled", "C7")]:
                ys = [data[a]["counts"][key] / n for a in a_keys]
                ax.plot(a_vals, ys, "-o", color=color, label=key)
            ax.axvline(0, color="gray", ls="--", alpha=0.5)
            ax.set_xlabel("α"); ax.set_ylabel("fraction of N")
            ax.set_title(f"m_steered={m}  (cell step{cells[m]['step_1idx']} L{cells[m]['layer']})",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.05)
        fig.suptitle("Per-marker steering: target = clean m-specific flip, "
                     "off_target = wrong-marker flip",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Specificity ratio plot: target − off_target per α per m
        fig, ax = plt.subplots(figsize=(11, 5))
        for m in cells:
            if m not in results["per_m"]: continue
            data = results["per_m"][m]
            a_keys = sorted(data.keys(), key=lambda x: float(x))
            a_vals = [float(a) for a in a_keys]
            n = len(eval_data)
            ys = [(data[a]["counts"]["target"] - data[a]["counts"]["off_target"]) / n
                  for a in a_keys]
            ax.plot(a_vals, ys, "-o", label=f"m={m}", lw=2)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("α (positive=push toward mul, negative=push toward add)")
        ax.set_ylabel("P(target flip) − P(off-target flip)")
        ax.set_title("Marker-specificity of intervention per α",
                     fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
