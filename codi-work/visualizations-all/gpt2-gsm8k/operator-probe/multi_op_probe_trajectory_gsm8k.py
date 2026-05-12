"""Per-example marker-emergence trajectory probe.

The multi_op_probe shows population-level (step, layer) accuracy. This script
asks the per-example question: for problem i, at which latent step does the
probe FIRST correctly predict each gold marker m?

Procedure:
  - Pick the best layer at each step for each (probe, marker) cell using the
    saved acc grid from multi_op_probe_gsm8k.json.
  - At each (step, marker) cell, fit a 5-fold cross-validated RidgeClassifier
    → get out-of-fold predictions for every example. That gives 6 OOF
    predictions per example per marker per probe (one per step).
  - For each example, find the smallest step k at which OOF pred == gold (or
    "never"). Aggregate per marker.

Outputs:
  multi_op_probe_trajectory_gsm8k.json
  multi_op_probe_trajectory_gsm8k.pdf
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
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
PD = Path(__file__).resolve().parent
REAL_JSON = PD / "multi_op_probe_gsm8k.json"
OUT_JSON = PD / "multi_op_probe_trajectory_gsm8k.json"
OUT_PDF = PD / "multi_op_probe_trajectory_gsm8k.pdf"

OPS = ["+", "-", "*", "/"]
M_MAX = 4
PROBES = ["op", "a_ld", "c_ld"]
CHANCE = {"op": 0.25, "a_ld": 0.10, "c_ld": 0.10}


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def last_digit(x):
    return int(abs(int(round(float(x)))) % 10)


def main():
    print("loading GSM8K", flush=True)
    ds = load_dataset("gsm8k", "main")["test"]
    markers_per_problem = [parse_markers(ex["answer"]) for ex in ds]
    N = len(markers_per_problem)
    print(f"N={N}")

    print("loading activations", flush=True)
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    Nact, S, L, H = acts.shape
    assert Nact == N
    print(f"  shape={acts.shape}")

    # Load best-layer-at-each-step from the multi_op_probe grid
    D = json.load(open(REAL_JSON))
    acc_grids = {p: {m: np.array(D["acc"][p][str(m)]) for m in range(1, M_MAX + 1)} for p in PROBES}

    # Build per-marker mask + labels
    marker_meta = {}
    for m in range(1, M_MAX + 1):
        mask = np.array([len(ms) >= m for ms in markers_per_problem])
        ops_m = np.array([
            (markers_per_problem[i][m-1][1] if mask[i] else "")
            for i in range(N)
        ])
        a_m_ld = np.array([
            (last_digit(markers_per_problem[i][m-1][0]) if mask[i] else -1)
            for i in range(N)
        ])
        c_m_ld = np.array([
            (last_digit(markers_per_problem[i][m-1][3]) if mask[i] else -1)
            for i in range(N)
        ])
        marker_meta[m] = {
            "mask": mask, "ops": ops_m, "a_ld": a_m_ld, "c_ld": c_m_ld,
            "n": int(mask.sum()),
        }
        print(f"  m={m}: N={mask.sum()}")

    # Compute per-(probe, marker, step) OOF predictions at the best layer at
    # that step (from the saved grid).
    t0 = time.time()
    results = {p: {m: {} for m in range(1, M_MAX + 1)} for p in PROBES}
    correctness_per_step = {p: {m: np.full(S, np.nan) for m in range(1, M_MAX + 1)} for p in PROBES}
    first_correct_step = {p: {m: None for m in range(1, M_MAX + 1)} for p in PROBES}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for p in PROBES:
        for m in range(1, M_MAX + 1):
            mask = marker_meta[m]["mask"]
            idx = np.where(mask)[0]
            if p == "op":
                y = marker_meta[m]["ops"][mask]
            elif p == "a_ld":
                y = marker_meta[m]["a_ld"][mask].astype(int)
            else:
                y = marker_meta[m]["c_ld"][mask].astype(int)
            if len(set(y)) < 2: continue
            # OOF prediction per example, per step (using the best layer for
            # this (probe, marker, step) cell from the original grid).
            oof_per_step = np.full((len(idx), S), -999, dtype=object)
            for k in range(S):
                # best layer at step k for this (probe, marker)
                row = acc_grids[p][m][k, :]
                if np.all(np.isnan(row)): continue
                best_layer = int(np.nanargmax(row))
                X = acts[idx, k, best_layer, :]
                # Standardize then cv-predict
                scaler = StandardScaler().fit(X)
                Xs = scaler.transform(X)
                clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
                try:
                    pred = cross_val_predict(clf, Xs, y, cv=skf, n_jobs=1)
                except Exception:
                    # Fallback for very imbalanced classes
                    pred = cross_val_predict(clf, Xs, y, cv=3, n_jobs=1)
                oof_per_step[:, k] = pred
                correct_at_k = (pred == y).mean()
                correctness_per_step[p][m][k] = float(correct_at_k)
                results[p][m][f"step{k+1}"] = {
                    "best_layer": best_layer, "acc_oof": float(correct_at_k),
                }
            # First-correct-step per example
            fcs = np.full(len(idx), -1, dtype=int)
            for i in range(len(idx)):
                for k in range(S):
                    if oof_per_step[i, k] == y[i]:
                        fcs[i] = k + 1; break
            never = int((fcs == -1).sum())
            dist = dict(Counter(fcs.tolist()))
            first_correct_step[p][m] = {
                "distribution": dist, "n_never": never, "n_total": len(idx),
            }
            print(f"  {p} m={m}: per-step acc {correctness_per_step[p][m].round(3).tolist()} "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    out = {
        "N": N, "S": S, "L": L,
        "correctness_per_step": {p: {m: correctness_per_step[p][m].tolist()
                                      for m in range(1, M_MAX + 1)} for p in PROBES},
        "first_correct_step": first_correct_step,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    # ---- Slideshow ----
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: setup
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = ("Multi-op probe — per-example chain trajectory\n\n"
                "For each (probe, marker), at each step k:\n"
                "  - Pick best layer from saved (step, layer) grid.\n"
                "  - 5-fold cross-validated probe → out-of-fold prediction per example.\n"
                "  - 'correct@k' = fraction of examples where OOF pred == gold marker m's label.\n\n"
                "Hypothesis (from even/odd analysis):\n"
                "  - c_ld m's correctness rises sharply at step 2m-1 (odd).\n"
                "  - op m's correctness rises at step ≈ m (then plateaus).\n\n")
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: per-step accuracy curves, one panel per probe, 4 markers per panel
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for col, p in enumerate(PROBES):
            ax = axes[col]
            steps = np.arange(1, S + 1)
            for m in range(1, M_MAX + 1):
                accs = correctness_per_step[p][m]
                ax.plot(steps, accs, "-o", label=f"m={m}")
            ax.axhline(CHANCE[p], color="black", ls=":", alpha=0.5,
                       label=f"chance={CHANCE[p]:.2f}")
            ax.set_xlabel("latent step"); ax.set_ylabel("OOF accuracy")
            ax.set_xticks(steps)
            ax.set_title(f"{p}: per-step OOF probe accuracy", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_ylim(0, max(0.6, max(correctness_per_step[p][m].max() for m in range(1, M_MAX + 1)) + 0.05))
            # Mark predicted "emergence step" per marker for c_ld: 2m-1
            if p == "c_ld":
                for m in range(1, M_MAX + 1):
                    em = min(S, 2 * m - 1)
                    ax.axvline(em, ls="--", lw=0.6, alpha=0.3, color=f"C{m-1}")
        fig.suptitle("When does each marker emerge across the latent loop?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: first-correct-step distribution per marker per probe
        fig, axes = plt.subplots(M_MAX, 3, figsize=(15, 11))
        for row_m, m in enumerate(range(1, M_MAX + 1)):
            for col_p, p in enumerate(PROBES):
                ax = axes[row_m, col_p]
                fcs = first_correct_step[p][m]
                if fcs is None: continue
                xs = list(range(1, S + 1)) + [S + 1]
                heights = [fcs["distribution"].get(k, 0) for k in range(1, S + 1)] + [fcs["n_never"]]
                colors = ["#4c72b0"] * S + ["#d62728"]
                ax.bar(range(len(xs)), heights, color=colors, edgecolor="black")
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([str(k) for k in range(1, S + 1)] + ["never"], fontsize=8)
                ax.set_xlabel("first correct step", fontsize=8)
                ax.set_title(f"{p}, m={m} — N={fcs['n_total']} examples",
                             fontsize=9, fontweight="bold")
                ax.set_ylabel("# examples", fontsize=8)
                ax.grid(axis="y", alpha=0.3)
                for i, h in enumerate(heights):
                    if h > 0:
                        ax.text(i, h + 0.5, str(h), ha="center", fontsize=7)
        fig.suptitle("Per-example: smallest latent step at which the OOF probe was correct  "
                     "(blue=steps 1-6; red=never)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
