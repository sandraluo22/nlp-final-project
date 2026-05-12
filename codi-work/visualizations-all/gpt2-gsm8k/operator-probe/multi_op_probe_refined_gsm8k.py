"""Refined multi-op probe on GSM8K.

Adds two filters to the original multi_op_probe to clean up the per-marker
emergence signal:

  (A) CORRECT-ONLY: train probes only on problems where CODI's emitted answer
      matches the gold. Removes noise from problems the model failed on.
  (B) LENGTH-MATCHED: train probe for marker m only on problems whose chain
      has EXACTLY m markers. Removes the confound that m=4's training set is
      always 4+ step chains (a different distribution).

For each (probe_type, marker, step, layer) cell, fits a RidgeClassifier on
the filtered training set with stratified 80/20 split.

Outputs:
  multi_op_probe_refined_gsm8k.json
  multi_op_probe_refined_gsm8k.pdf
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
GSM8K_CODI_JSON = REPO.parent / "cf-datasets" / "gsm8k_codi_cot.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "multi_op_probe_refined_gsm8k.json"
OUT_PDF = PD / "multi_op_probe_refined_gsm8k.pdf"

PROBES = ["op", "a_ld", "c_ld"]
M_MAX = 4
CHANCE = {"op": 0.25, "a_ld": 0.10, "c_ld": 0.10}


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def last_digit(x):
    return int(abs(int(round(float(x)))) % 10)


def codi_extract(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def fit_one(X, y):
    if len(set(y)) < 2 or len(y) < 20:
        return None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(Counter(y).values()) >= 2 else None
    )
    sc = StandardScaler().fit(Xtr)
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = float((pred == yte).mean())
    cls = sorted(set(y))
    f1 = float(f1_score(yte, pred, labels=cls, average="macro", zero_division=0))
    return {"acc": acc, "f1": f1, "n_test": len(yte)}


def sweep(acts, idx, y, S, L):
    accG = np.full((S, L), np.nan)
    f1G = np.full((S, L), np.nan)
    nG = np.full((S, L), 0, dtype=int)
    for k in range(S):
        for l in range(L):
            X = acts[idx, k, l, :]
            r = fit_one(X, y)
            if r is not None:
                accG[k, l] = r["acc"]
                f1G[k, l] = r["f1"]
                nG[k, l] = r["n_test"]
    return accG, f1G, nG


def main():
    print("loading GSM8K + activations", flush=True)
    ds = load_dataset("gsm8k", "main")["test"]
    markers_per_problem = [parse_markers(ex["answer"]) for ex in ds]
    gold_finals = []
    for ex in ds:
        m = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        gold_finals.append(float(m.group(1)) if m else np.nan)
    gold_finals = np.array(gold_finals)
    N = len(markers_per_problem)
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    Nact, S, L, H = acts.shape
    assert Nact == N

    # Try to load CODI's emitted answers for the correct-only filter
    correct_mask = None
    if GSM8K_CODI_JSON.exists():
        try:
            cot = json.load(open(GSM8K_CODI_JSON))
            # gsm8k_codi_cot may have per-problem emitted final answer
            keys = list(cot[0].keys()) if cot else []
            print(f"  gsm8k_codi_cot keys sample: {keys}")
            # Try common field names
            emit_field = None
            for cand in ("codi_pred_int", "emit_int", "emit_answer", "model_answer",
                         "codi_int", "pred", "answer_emit"):
                if cand in keys:
                    emit_field = cand; break
            if emit_field is None and "emit" in keys:
                emit_field = "emit"
            if emit_field:
                emits = []
                for r in cot:
                    v = r.get(emit_field)
                    if v is None: emits.append(np.nan)
                    elif isinstance(v, (int, float)): emits.append(float(v))
                    else:
                        x = codi_extract(str(v))
                        emits.append(x if x is not None else np.nan)
                emits = np.array(emits)
                if len(emits) == N:
                    correct_mask = (~np.isnan(emits)) & (~np.isnan(gold_finals)) & \
                                    (np.abs(emits - gold_finals) < 1e-3)
                    print(f"  CODI-correct filter: {correct_mask.sum()}/{N} examples got the answer right")
        except Exception as e:
            print(f"  could not load CODI correct labels: {e}")
    if correct_mask is None:
        print("  no CODI correct labels available; correct-only filter disabled, only length-matched")

    # Build per-marker labels (same as original probe)
    marker_meta = {}
    for m in range(1, M_MAX + 1):
        mask_geq = np.array([len(ms) >= m for ms in markers_per_problem])
        mask_eq = np.array([len(ms) == m for ms in markers_per_problem])
        ops_m = np.array([
            (markers_per_problem[i][m-1][1] if mask_geq[i] else "")
            for i in range(N)
        ])
        a_m_ld = np.array([
            (last_digit(markers_per_problem[i][m-1][0]) if mask_geq[i] else -1)
            for i in range(N)
        ])
        c_m_ld = np.array([
            (last_digit(markers_per_problem[i][m-1][3]) if mask_geq[i] else -1)
            for i in range(N)
        ])
        marker_meta[m] = {
            "mask_geq": mask_geq, "mask_eq": mask_eq,
            "ops": ops_m, "a_ld": a_m_ld, "c_ld": c_m_ld,
        }
        print(f"  m={m}: ≥m={mask_geq.sum()}, exact-m={mask_eq.sum()}")

    # ---- Define 3 filters: ORIGINAL, CORRECT-ONLY, LENGTH-MATCHED ----
    filters = {"original": {m: marker_meta[m]["mask_geq"] for m in range(1, M_MAX + 1)},
               "length_matched": {m: marker_meta[m]["mask_eq"] for m in range(1, M_MAX + 1)}}
    if correct_mask is not None:
        filters["correct_only"] = {m: marker_meta[m]["mask_geq"] & correct_mask for m in range(1, M_MAX + 1)}
        filters["correct_and_length_matched"] = {m: marker_meta[m]["mask_eq"] & correct_mask for m in range(1, M_MAX + 1)}

    print(f"\nFilters: {list(filters.keys())}")

    # ---- Run sweeps ----
    results = {}
    t0 = time.time()
    for filt_name, marker_filters in filters.items():
        print(f"\n=== filter: {filt_name} ===", flush=True)
        results[filt_name] = {}
        for p in PROBES:
            results[filt_name][p] = {}
            for m in range(1, M_MAX + 1):
                mask = marker_filters[m]
                idx = np.where(mask)[0]
                if len(idx) < 30:
                    print(f"  {p} m={m}: too few samples ({len(idx)}); skipping")
                    continue
                if p == "op":
                    y = marker_meta[m]["ops"][mask]
                elif p == "a_ld":
                    y = marker_meta[m]["a_ld"][mask].astype(int)
                else:
                    y = marker_meta[m]["c_ld"][mask].astype(int)
                acc, f1, nt = sweep(acts, idx, y, S, L)
                # Best cell and best-anywhere-per-step trajectory
                if np.all(np.isnan(acc)):
                    best = None; cell_acc = None
                else:
                    flat = acc.flatten()
                    i_best = int(np.nanargmax(flat))
                    s_b, l_b = i_best // L, i_best % L
                    best = {"step": s_b + 1, "layer": l_b,
                            "acc": float(acc[s_b, l_b]),
                            "f1": float(f1[s_b, l_b])}
                # max-over-layers per step
                best_per_step = []
                for k in range(S):
                    row = acc[k, :]
                    if np.all(np.isnan(row)):
                        best_per_step.append(None)
                    else:
                        ly = int(np.nanargmax(row))
                        best_per_step.append({"layer": ly, "acc": float(row[ly])})
                results[filt_name][p][m] = {
                    "n_train_total": int(len(idx)),
                    "best": best,
                    "best_per_step": best_per_step,
                    "acc": acc.tolist(),
                    "f1": f1.tolist(),
                }
                if best:
                    print(f"  {p} m={m}: best step{best['step']} L{best['layer']} acc={best['acc']:.3f} "
                          f"(N={len(idx)})  ({time.time()-t0:.0f}s)")
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # ---- Plot: per-step best-anywhere trajectories per filter ----
    with PdfPages(OUT_PDF) as pdf:
        # Title
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = ("Multi-op probe — refined filters\n\n"
                "Three (or four) filters comparison:\n"
                "  original: all problems with ≥m markers (the original probe's training set)\n"
                "  length_matched: only problems with exactly m markers\n")
        if "correct_only" in filters:
            body += "  correct_only: only problems CODI got right\n"
            body += "  correct_and_length_matched: both filters combined (smallest, cleanest)\n"
        body += "\nPer-step trajectory (max-over-layers acc) plotted per filter."
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Refined multi-op probe", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Per probe, 4 panels (one per marker), each with 1 curve per filter
        for p in PROBES:
            fig, axes = plt.subplots(1, M_MAX, figsize=(20, 4.5))
            for col, m in enumerate(range(1, M_MAX + 1)):
                ax = axes[col]
                steps = np.arange(1, S + 1)
                for filt_name in filters:
                    info = results[filt_name].get(p, {}).get(m)
                    if info is None or not info["best_per_step"]: continue
                    bps = info["best_per_step"]
                    ys = [b["acc"] if b else np.nan for b in bps]
                    ax.plot(steps, ys, "-o", label=f"{filt_name} (N={info['n_train_total']})", lw=1.5)
                ax.axhline(CHANCE[p], color="black", ls=":", alpha=0.5, label=f"chance={CHANCE[p]:.2f}")
                ax.set_xlabel("latent step"); ax.set_ylabel("max-over-layers acc")
                ax.set_xticks(steps)
                ax.set_title(f"{p}, m={m}", fontsize=10, fontweight="bold")
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
            fig.suptitle(f"{p} probe — per-marker best-anywhere-per-step, per filter",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Best-cell summary table
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.axis("off")
        txt = ""
        for p in PROBES:
            txt += f"\n=== {p} ===\n"
            txt += f"{'marker':<7s} {'filter':<32s} {'best (step, layer)':<22s} {'acc':<8s} {'n':<7s}\n"
            for m in range(1, M_MAX + 1):
                for filt_name in filters:
                    info = results[filt_name].get(p, {}).get(m)
                    if info is None or info["best"] is None: continue
                    b = info["best"]
                    txt += f"  m={m}    {filt_name:<30s}  step{b['step']} L{b['layer']:<2d}            "
                    txt += f"{b['acc']:<8.3f} {info['n_train_total']:<7d}\n"
                txt += "\n"
        ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=8)
        ax.set_title("Best-cell summary across filters", fontsize=12, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
