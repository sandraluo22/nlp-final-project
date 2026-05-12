"""LDA-as-probe on GSM8K + add↔mul steering at the winning cell.

Section A — LDA as probe
  - For each (step, layer), fit LinearDiscriminantAnalysis on GSM8K
    latent acts using gsm8k first-op Type (4-class).  Uses LDA itself as
    the classifier (predict()), 5-fold stratified 80/20 split.
  - Reports per-(step, layer) accuracy + macro-F1.

Section B — Add↔Mul direction
  - At the winning (step, layer), compute add_centroid and mul_centroid
    over training rows.  v_add→mul = unit(mul - add).

Section C — Activation-space steering surrogate
  - Take held-out Add and Mul rows.
  - For α in a sweep grid, shift each Add row by +α·v_add→mul and each
    Mul row by −α·v_add→mul.
  - Use the already-fit 4-class LDA to predict the shifted rows.
  - Report flip rates:
      P(Add → Mul) as α grows positively
      P(Mul → Add) as α grows positively
    plus side-effects (flips to Sub or Div).

This is an ACTIVATION-space test, not real CODI steering — we shift the
activation and ask the probe, not the model.  A direction that fails
this test certainly won't steer the model causally; passing it is
necessary-but-not-sufficient for a real causal effect.

Output: lda_probe_and_steer_addmul_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parents[3]
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
PD = Path(__file__).resolve().parent
OUT_PDF = PD / "lda_probe_and_steer_addmul_gsm8k.pdf"
OUT_JSON = PD / "lda_probe_and_steer_addmul_gsm8k.json"

OP_SYM_TO_NAME = {
    "+": "Addition", "-": "Subtraction",
    "*": "Multiplication", "/": "Common-Division",
}
OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
OP_COLORS = {
    "Addition":        "#ff7f0e",
    "Subtraction":     "#1f77b4",
    "Multiplication":  "#d62728",
    "Common-Division": "#2ca02c",
}


def gsm8k_first_op(ds_test) -> list[str]:
    labels = []
    for ex in ds_test:
        ans = ex["answer"].replace(",", "")
        m = re.search(r"<<.*?([+\-*/]).*?=", ans)
        labels.append(OP_SYM_TO_NAME[m.group(1)] if m else "Addition")
    return labels


def fit_eval_lda(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42):
    accs, f1s = [], []
    per_class = {op: {"p": [], "r": [], "f": []} for op in OPS}
    for s in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed + s, stratify=y
        )
        lda = LinearDiscriminantAnalysis(n_components=3, solver="svd")
        lda.fit(Xtr, ytr)
        pred = lda.predict(Xte)
        accs.append(float((pred == yte).mean()))
        f1s.append(float(f1_score(yte, pred, labels=OPS, average="macro", zero_division=0)))
        p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=OPS, zero_division=0)
        for j, op in enumerate(OPS):
            per_class[op]["p"].append(p[j])
            per_class[op]["r"].append(r[j])
            per_class[op]["f"].append(f[j])
    return {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "per_class": {op: {k: float(np.mean(v)) for k, v in d.items()} for op, d in per_class.items()},
    }


def _eval_shifted(lda, Xte, yte, v_unit, scale, alphas, add_mask_te, mul_mask_te):
    sweep = []
    for a in alphas:
        Xshift = Xte.copy()
        Xshift[add_mask_te] = Xshift[add_mask_te] + a * v_unit * scale
        Xshift[mul_mask_te] = Xshift[mul_mask_te] - a * v_unit * scale
        pred = lda.predict(Xshift)
        add_pred_dist = Counter(pred[add_mask_te])
        mul_pred_dist = Counter(pred[mul_mask_te])
        n_add = int(add_mask_te.sum()); n_mul = int(mul_mask_te.sum())
        sweep.append({
            "alpha": float(a),
            "add_to_mul": add_pred_dist.get("Multiplication", 0) / max(n_add, 1),
            "add_kept": add_pred_dist.get("Addition", 0) / max(n_add, 1),
            "add_to_sub": add_pred_dist.get("Subtraction", 0) / max(n_add, 1),
            "add_to_div": add_pred_dist.get("Common-Division", 0) / max(n_add, 1),
            "mul_to_add": mul_pred_dist.get("Addition", 0) / max(n_mul, 1),
            "mul_kept": mul_pred_dist.get("Multiplication", 0) / max(n_mul, 1),
            "mul_to_sub": mul_pred_dist.get("Subtraction", 0) / max(n_mul, 1),
            "mul_to_div": mul_pred_dist.get("Common-Division", 0) / max(n_mul, 1),
            "overall_acc": float((pred == yte).mean()),
            "n_add": n_add, "n_mul": n_mul,
        })
    return sweep


def steering_sweep(X: np.ndarray, y: np.ndarray, alphas: list[float], seed: int = 42):
    """At a single (step, layer), fit one LDA on the train split.  Then
    shift held-out Add rows by +α·v and Mul rows by −α·v.  v points
    add→mul (i.e. +α should push Add → Mul predictions).  Also runs a
    random-direction baseline to show the effect is operator-specific."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    lda = LinearDiscriminantAnalysis(n_components=3, solver="svd")
    lda.fit(Xtr, ytr)
    base_pred = lda.predict(Xte)
    base_acc = float((base_pred == yte).mean())

    add_centroid = Xtr[ytr == "Addition"].mean(axis=0)
    mul_centroid = Xtr[ytr == "Multiplication"].mean(axis=0)
    v = mul_centroid - add_centroid
    v_norm = float(np.linalg.norm(v))
    v_unit = v / (v_norm + 1e-12)

    # baseline residual norm at this (step, layer): use median ||x|| over train
    median_norm = float(np.median(np.linalg.norm(Xtr, axis=1)))

    add_mask_te = yte == "Addition"
    mul_mask_te = yte == "Multiplication"

    sweep = _eval_shifted(lda, Xte, yte, v_unit, median_norm, alphas, add_mask_te, mul_mask_te)

    # Random-direction baseline: mean over N_RAND random unit vectors
    rng = np.random.RandomState(0)
    N_RAND = 8
    rand_sweeps = []
    for k in range(N_RAND):
        rv = rng.randn(X.shape[1]).astype(np.float32)
        rv = rv / (np.linalg.norm(rv) + 1e-12)
        rand_sweeps.append(_eval_shifted(
            lda, Xte, yte, rv, median_norm, alphas, add_mask_te, mul_mask_te))
    keys = ["add_to_mul", "add_kept", "add_to_sub", "add_to_div",
            "mul_to_add", "mul_kept", "mul_to_sub", "mul_to_div", "overall_acc"]
    rand_avg = []
    for i, a in enumerate(alphas):
        row = {"alpha": float(a)}
        for k in keys:
            row[k] = float(np.mean([rand_sweeps[j][i][k] for j in range(N_RAND)]))
        rand_avg.append(row)
    return {
        "base_acc": base_acc,
        "v_norm": v_norm,
        "median_train_norm": median_norm,
        "sweep": sweep,
        "random_baseline": rand_avg,
        "alphas": [float(a) for a in alphas],
    }


def main():
    print("loading GSM8K Type labels", flush=True)
    ds = load_dataset("gsm8k", "main")
    types = np.array(gsm8k_first_op(ds["test"]))
    print(f"  N={len(types)}  by op: {dict(Counter(types.tolist()))}")

    print("loading activations", flush=True)
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    print(f"  shape={acts.shape}")  # (N, S, L, H)
    N, S, L, H = acts.shape
    assert N == len(types), f"act/label N mismatch: {N} vs {len(types)}"

    # ----- Section A: per (step, layer) LDA probe sweep -----
    print("\nfitting LDA probe per (step, layer)", flush=True)
    acc = np.full((S, L), np.nan)
    f1m = np.full((S, L), np.nan)
    detail = {}
    t0 = time.time()
    for step in range(S):
        for layer in range(L):
            X = acts[:, step, layer, :]
            res = fit_eval_lda(X, types)
            acc[step, layer] = res["acc_mean"]
            f1m[step, layer] = res["f1_mean"]
            detail[(step, layer)] = res
        print(f"  step {step+1}/{S} done ({time.time()-t0:.0f}s)", flush=True)

    best_idx = int(np.nanargmax(acc.flatten()))
    bs, bl = best_idx // L, best_idx % L
    print(f"\nLDA-probe winner: step {bs+1}, layer {bl}  "
          f"acc={acc[bs, bl]:.3f}  F1={f1m[bs, bl]:.3f}")

    # ----- Section B/C: add↔mul direction + steering at winner -----
    print("running add↔mul steering sweep at winner", flush=True)
    Xwin = acts[:, bs, bl, :]
    alphas = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    steer = steering_sweep(Xwin, types, alphas)
    for r in steer["sweep"]:
        print(f"  α={r['alpha']:>4.2f}  Add→Mul={r['add_to_mul']:.3f}  "
              f"Mul→Add={r['mul_to_add']:.3f}  acc={r['overall_acc']:.3f}")

    # Also run steering at a few neighboring cells for robustness
    neighbors = [(bs, bl)]
    for ds_, dl_ in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        ns, nl = bs + ds_, bl + dl_
        if 0 <= ns < S and 0 <= nl < L:
            neighbors.append((ns, nl))
    neighbor_results = {}
    for (ns, nl) in neighbors:
        if (ns, nl) == (bs, bl): continue
        Xn = acts[:, ns, nl, :]
        neighbor_results[f"step{ns+1}_L{nl}"] = steering_sweep(Xn, types, alphas)

    out = {
        "N": int(N), "by_op": {k: int(v) for k, v in Counter(types.tolist()).items()},
        "acc_grid": acc.tolist(), "f1_grid": f1m.tolist(),
        "winner": {"step": bs+1, "layer": bl,
                   "acc": float(acc[bs, bl]), "f1": float(f1m[bs, bl]),
                   "per_class": detail[(bs, bl)]["per_class"]},
        "steer_at_winner": steer,
        "steer_neighbors": neighbor_results,
        "alphas": alphas,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    # ----- Slideshow -----
    print("rendering slideshow", flush=True)
    with PdfPages(OUT_PDF) as pdf:
        # PAGE 1: heatmaps acc + F1
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, M, t in [(axes[0], acc, "accuracy"), (axes[1], f1m, "macro-F1")]:
            im = ax.imshow(M, origin="lower", aspect="auto",
                           vmin=0.2, vmax=1.0, cmap="viridis")
            ax.set_xlabel("layer"); ax.set_ylabel("step")
            ax.set_xticks(range(L)); ax.set_yticks(range(S))
            ax.set_yticklabels([str(i+1) for i in range(S)])
            ax.set_title(f"LDA probe — {t}\nwinner: step{bs+1}, layer {bl}: "
                         f"acc={acc[bs,bl]:.3f}, F1={f1m[bs,bl]:.3f}")
            ax.scatter([bl], [bs], marker="*", s=180, c="white",
                       edgecolors="black", linewidths=1, zorder=5)
            plt.colorbar(im, ax=ax, fraction=0.045)
        fig.suptitle(f"LDA-as-probe on GSM8K (4-class, N={N})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 2: per-class P/R/F1 at winner
        d = detail[(bs, bl)]["per_class"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for col, m in enumerate(["p", "r", "f"]):
            vals = [d[op][m] for op in OPS]
            axes[col].bar(range(len(OPS)), vals, color=[OP_COLORS[o] for o in OPS])
            axes[col].set_ylim(0, 1.0)
            axes[col].set_title({"p": "precision", "r": "recall", "f": "F1"}[m])
            axes[col].set_xticks(range(len(OPS)))
            axes[col].set_xticklabels(OPS, rotation=20, ha="right")
            for i, v in enumerate(vals):
                axes[col].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        fig.suptitle(f"Per-class metrics at winner (step{bs+1}, layer {bl})",
                     fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 3: steering sweep at winner — Add→Mul and Mul→Add curves
        sweep = steer["sweep"]; rand = steer["random_baseline"]
        a_grid = [r["alpha"] for r in sweep]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        # Add row (target = Mul) — show 2 key metrics (kept + target) + random baseline
        axes[0].plot(a_grid, [r["add_kept"]   for r in sweep], "-o", color=OP_COLORS["Addition"], label="P(Add | shifted Add) — kept")
        axes[0].plot(a_grid, [r["add_to_mul"] for r in sweep], "-o", color=OP_COLORS["Multiplication"], label="P(Mul | shifted Add) — TARGET")
        axes[0].plot(a_grid, [r["add_kept"]   for r in rand],  "--x", color=OP_COLORS["Addition"], alpha=0.5, label="random dir: kept")
        axes[0].plot(a_grid, [r["add_to_mul"] for r in rand],  "--x", color=OP_COLORS["Multiplication"], alpha=0.5, label="random dir: → Mul")
        axes[0].set_xlabel("α (× median train norm)"); axes[0].set_ylabel("P(predict op | held-out Add row)")
        axes[0].set_title("Steering Add → Mul   (random dir = baseline)")
        axes[0].grid(alpha=0.3); axes[0].legend(loc="center right", fontsize=7)
        axes[0].set_ylim(0, 1.05)
        # Mul row (target = Add)
        axes[1].plot(a_grid, [r["mul_kept"]   for r in sweep], "-o", color=OP_COLORS["Multiplication"], label="P(Mul | shifted Mul) — kept")
        axes[1].plot(a_grid, [r["mul_to_add"] for r in sweep], "-o", color=OP_COLORS["Addition"], label="P(Add | shifted Mul) — TARGET")
        axes[1].plot(a_grid, [r["mul_kept"]   for r in rand],  "--x", color=OP_COLORS["Multiplication"], alpha=0.5, label="random dir: kept")
        axes[1].plot(a_grid, [r["mul_to_add"] for r in rand],  "--x", color=OP_COLORS["Addition"], alpha=0.5, label="random dir: → Add")
        axes[1].set_xlabel("α (× median train norm)"); axes[1].set_ylabel("P(predict op | held-out Mul row)")
        axes[1].set_title("Steering Mul → Add   (random dir = baseline)")
        axes[1].grid(alpha=0.3); axes[1].legend(loc="center right", fontsize=7)
        axes[1].set_ylim(0, 1.05)
        fig.suptitle(f"Add↔Mul activation-space steering at step{bs+1} L{bl}  "
                     f"(direction norm={steer['v_norm']:.2f}; "
                     f"random baseline avg over 8 unit vectors)",
                     fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 4: summary text + neighbor robustness
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.axis("off")
        txt = "LDA-as-probe + Add↔Mul steering — summary\n\n"
        txt += f"  Winner cell: step{bs+1}, layer {bl}   acc={acc[bs,bl]:.3f}  F1={f1m[bs,bl]:.3f}\n"
        txt += f"  Direction ||mul - add|| = {steer['v_norm']:.3f}\n"
        txt += f"  Median train-row norm   = {steer['median_train_norm']:.3f}\n"
        txt += f"  (α is in units of median train-row norm)\n\n"
        txt += "Steering at winner (Add→Mul):\n"
        for r in sweep:
            txt += (f"  α={r['alpha']:>4.2f}  "
                    f"add_kept={r['add_kept']:.3f}  add→mul={r['add_to_mul']:.3f}  "
                    f"add→sub={r['add_to_sub']:.3f}  add→div={r['add_to_div']:.3f}\n")
        txt += "\nSteering at winner (Mul→Add):\n"
        for r in sweep:
            txt += (f"  α={r['alpha']:>4.2f}  "
                    f"mul_kept={r['mul_kept']:.3f}  mul→add={r['mul_to_add']:.3f}  "
                    f"mul→sub={r['mul_to_sub']:.3f}  mul→div={r['mul_to_div']:.3f}\n")
        txt += "\nNeighbor cells (acc at α=0 / Add→Mul at α=1.0):\n"
        for k, nb in neighbor_results.items():
            row1 = next(r for r in nb["sweep"] if r["alpha"] == 1.0)
            txt += f"  {k:<14s}  base_acc={nb['base_acc']:.3f}  add→mul@1.0={row1['add_to_mul']:.3f}\n"
        ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9)
        ax.set_title("Summary", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
