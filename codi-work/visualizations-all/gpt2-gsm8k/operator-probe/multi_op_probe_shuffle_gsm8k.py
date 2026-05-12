"""Shuffle-step null baseline for multi_op_probe_gsm8k.

The original multi_op_probe asks: at which (latent step, layer) cell does a
per-marker (op | a_ld | c_ld) probe peak? If CODI computes the GSM8K chain
sequentially, later markers should peak at later steps.

This baseline tests: if we permute the 6 latent-step indices PER EXAMPLE,
does the per-marker (step, layer) pattern survive? Permutation destroys any
information carried by the *position* of an activation in the latent loop
while preserving (a) the marginal distribution of activations per example
and (b) the per-marker label distribution.

For each shuffle seed and each (probe_type, marker, step, layer) cell, we
fit the same RidgeClassifier as multi_op_probe_gsm8k.py and record acc/F1.
We compare best-cell acc real vs shuffled-mean ± std.

Outputs: multi_op_probe_shuffle_gsm8k.{json,pdf}
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
PD = Path(__file__).resolve().parent
REAL_JSON = PD / "multi_op_probe_gsm8k.json"
OUT_JSON = PD / "multi_op_probe_shuffle_gsm8k.json"
OUT_PDF = PD / "multi_op_probe_shuffle_gsm8k.pdf"

OPS = ["+", "-", "*", "/"]
M_MAX = 4
N_DIGITS = 10
N_SHUFFLE_SEEDS = 3   # number of independent permutations to average over


def parse_markers(answer_text: str):
    ans = answer_text.replace(",", "")
    ms = re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", ans)
    return [(float(a), op, float(b), float(c)) for a, op, b, c in ms]


def last_digit(x: float) -> int:
    return int(abs(int(round(x))) % 10)


def fit_probe(X, y):
    if len(set(y)) < 2:
        return None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(Counter(y).values()) >= 2 else None
    )
    scaler = StandardScaler().fit(Xtr)
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))
    acc = float((pred == yte).mean())
    classes = sorted(set(y))
    f1 = float(f1_score(yte, pred, labels=classes, average="macro", zero_division=0))
    return {"acc": acc, "f1": f1}


def sweep_grid(acts, idx, y_for_marker, S, L):
    """Sweep (step, layer) for one marker's probe."""
    acc_g = np.full((S, L), np.nan)
    f1_g = np.full((S, L), np.nan)
    for s in range(S):
        for l in range(L):
            X = acts[idx, s, l, :]
            r = fit_probe(X, y_for_marker)
            if r is not None:
                acc_g[s, l] = r["acc"]
                f1_g[s, l] = r["f1"]
    return acc_g, f1_g


def main():
    print("loading GSM8K", flush=True)
    ds = load_dataset("gsm8k", "main")["test"]
    markers_per_problem = [parse_markers(ex["answer"]) for ex in ds]
    N_total = len(markers_per_problem)
    print(f"N={N_total}")

    print("loading activations", flush=True)
    acts_real = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = acts_real.shape
    assert N == N_total
    print(f"  shape={acts_real.shape}")

    # Build per-marker label arrays + masks (identical to multi_op_probe_gsm8k.py)
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
            "mask": mask, "n": int(mask.sum()),
            "ops": ops_m, "a_ld": a_m_ld, "c_ld": c_m_ld,
        }

    probes = ["op", "a_ld", "c_ld"]

    # Real grids (load from previously-saved JSON if present, else recompute)
    if REAL_JSON.exists():
        print(f"loading real grids from {REAL_JSON.name}", flush=True)
        real = json.load(open(REAL_JSON))
        real_acc = {p: {m: np.array(real["acc"][p][str(m)]) for m in range(1, M_MAX+1)}
                    for p in probes}
        real_f1 = {p: {m: np.array(real["f1"][p][str(m)]) for m in range(1, M_MAX+1)}
                   for p in probes}
        real_best = real["best"]
    else:
        raise FileNotFoundError(f"{REAL_JSON} not found — run multi_op_probe_gsm8k.py first.")

    # Shuffle baseline: K seeds × probes × markers × (S, L) cells
    print(f"running {N_SHUFFLE_SEEDS} shuffle seeds...", flush=True)
    t0 = time.time()
    shuf_acc = {p: {m: [] for m in range(1, M_MAX+1)} for p in probes}
    shuf_f1  = {p: {m: [] for m in range(1, M_MAX+1)} for p in probes}

    for seed in range(N_SHUFFLE_SEEDS):
        print(f"  seed {seed} permuting steps per example...", flush=True)
        rng = np.random.default_rng(seed)
        # For each example, generate a random permutation of [0..S-1]
        perms = np.stack([rng.permutation(S) for _ in range(N)])  # (N, S)
        # Apply permutation along step axis per example
        rows = np.arange(N)[:, None]
        acts_shuf = acts_real[rows, perms, :, :]  # (N, S, L, H)

        for probe in probes:
            for m in range(1, M_MAX + 1):
                mask = marker_meta[m]["mask"]
                idx = np.where(mask)[0]
                if probe == "op":
                    y = marker_meta[m]["ops"][mask]
                elif probe == "a_ld":
                    y = marker_meta[m]["a_ld"][mask].astype(int)
                else:
                    y = marker_meta[m]["c_ld"][mask].astype(int)
                acc_g, f1_g = sweep_grid(acts_shuf, idx, y, S, L)
                shuf_acc[probe][m].append(acc_g)
                shuf_f1[probe][m].append(f1_g)
                print(f"    {probe} m={m} done  ({time.time()-t0:.0f}s)", flush=True)

    # Aggregate across seeds
    shuf_acc_mean = {p: {m: np.nanmean(np.stack(shuf_acc[p][m]), axis=0)
                          for m in range(1, M_MAX+1)} for p in probes}
    shuf_acc_std  = {p: {m: np.nanstd(np.stack(shuf_acc[p][m]), axis=0)
                          for m in range(1, M_MAX+1)} for p in probes}

    # Find best cell of shuffled (per probe/marker) — for null comparison
    shuf_best = {p: {} for p in probes}
    for p in probes:
        for m in range(1, M_MAX+1):
            G = shuf_acc_mean[p][m]
            if np.all(np.isnan(G)):
                shuf_best[p][m] = None; continue
            i = int(np.nanargmax(G.flatten()))
            s, l = i // L, i % L
            shuf_best[p][m] = {
                "step": s+1, "layer": l,
                "acc_mean": float(G[s, l]),
                "acc_std": float(shuf_acc_std[p][m][s, l]),
            }

    # Save
    out = {
        "N_total": N_total, "M_MAX": M_MAX, "N_SHUFFLE_SEEDS": N_SHUFFLE_SEEDS,
        "shuf_acc_mean": {p: {m: shuf_acc_mean[p][m].tolist() for m in range(1, M_MAX+1)} for p in probes},
        "shuf_acc_std":  {p: {m: shuf_acc_std[p][m].tolist()  for m in range(1, M_MAX+1)} for p in probes},
        "shuf_best": shuf_best,
        "real_best": real_best,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    # ---- Slideshow: per probe, real vs shuffled-mean heatmaps + best summary ----
    chance = {"op": 0.25, "a_ld": 0.10, "c_ld": 0.10}
    probe_title = {"op": "operator (4-cls)",
                   "a_ld": "a-operand last digit (10-cls)",
                   "c_ld": "result last digit (10-cls)"}

    with PdfPages(OUT_PDF) as pdf:
        # PAGE 1: setup
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        txt = (f"Multi-op probe — shuffle-step null baseline\n\n"
               f"  Real probes from {REAL_JSON.name}\n"
               f"  Permutation: per-example random shuffle of the 6 latent steps\n"
               f"  N_SHUFFLE_SEEDS = {N_SHUFFLE_SEEDS}\n\n"
               f"  Hypothesis: if step ORDERING carries marker-specific information,\n"
               f"  per-marker best-cell acc should drop when we shuffle steps.\n"
               f"  If the structure is purely about which latent acts are present\n"
               f"  (irrespective of step index), shuffle ≈ real.\n")
        ax.text(0.04, 0.96, txt, va="top", ha="left", family="monospace", fontsize=11)
        ax.set_title("Setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGES 2-4: per probe — 4×2 grid of (real, shuffle-mean) heatmaps per marker
        for probe in probes:
            fig, axes = plt.subplots(M_MAX, 2, figsize=(11, 12))
            for m in range(1, M_MAX+1):
                G_real = real_acc[probe][m]
                G_shuf = shuf_acc_mean[probe][m]
                vmin = max(0.0, chance[probe] - 0.05)
                vmax = max(0.3, max(float(np.nanmax(G_real)), float(np.nanmax(G_shuf))) + 0.02)
                ax = axes[m-1, 0]
                im = ax.imshow(G_real, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
                ax.set_title(f"REAL m={m}  best={real_best[probe][str(m)]['acc']:.3f}", fontsize=9)
                ax.set_xticks(range(L)); ax.set_yticks(range(S))
                ax.set_yticklabels([str(i+1) for i in range(S)])
                ax.set_xlabel("layer"); ax.set_ylabel("step")
                rb = real_best[probe][str(m)]
                if rb: ax.scatter([rb["layer"]], [rb["step"]-1], marker="*", s=80, c="white", edgecolors="black")
                plt.colorbar(im, ax=ax, fraction=0.045)

                ax = axes[m-1, 1]
                im = ax.imshow(G_shuf, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
                sb = shuf_best[probe][m]
                title = f"SHUFFLE m={m}"
                if sb: title += f"  best mean={sb['acc_mean']:.3f}±{sb['acc_std']:.3f}"
                ax.set_title(title, fontsize=9)
                ax.set_xticks(range(L)); ax.set_yticks(range(S))
                ax.set_yticklabels([str(i+1) for i in range(S)])
                ax.set_xlabel("layer"); ax.set_ylabel("step")
                if sb: ax.scatter([sb["layer"]], [sb["step"]-1], marker="*", s=80, c="white", edgecolors="black")
                plt.colorbar(im, ax=ax, fraction=0.045)
            fig.suptitle(f"Per-marker {probe_title[probe]} probe — REAL vs SHUFFLED-STEP "
                         f"(chance={chance[probe]:.2f})",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 5: best-acc real vs shuffled summary (does the gap shrink?)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        ms = list(range(1, M_MAX+1))
        for col, probe in enumerate(probes):
            real_acc_m = [real_best[probe][str(m)]["acc"] if real_best[probe][str(m)] else np.nan for m in ms]
            shuf_acc_m = [shuf_best[probe][m]["acc_mean"] if shuf_best[probe][m] else np.nan for m in ms]
            shuf_std_m = [shuf_best[probe][m]["acc_std"] if shuf_best[probe][m] else 0 for m in ms]
            ax = axes[col]
            ax.errorbar(ms, shuf_acc_m, yerr=shuf_std_m, marker="s", ls="--",
                        color="gray", label="shuffle-step best (±std)")
            ax.plot(ms, real_acc_m, "o-", color="#2ca02c", label="real best")
            ax.axhline(chance[probe], ls=":", c="black", alpha=0.5, label=f"chance={chance[probe]:.2f}")
            ax.set_xlabel("marker position m"); ax.set_xticks(ms)
            ax.set_ylabel("best (step, layer) accuracy")
            ax.set_title(probe_title[probe], fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left")
            ax.set_ylim(0, max(max(real_acc_m), max(shuf_acc_m)) + 0.05)
            ax.grid(alpha=0.3)
        fig.suptitle("Real vs shuffle-step best-cell accuracy per marker  "
                     "(closing gap = step-order matters less than acts themselves)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
