"""LDA-on-CF dim sweep with held-out scoring + cross-dataset transfer.

':' analog of cf_lda_80_20_slideshow.py. For each k in {1,2,3}:
  Fit LDA(n_components=k) on 80% of cf_balanced ':' residuals (stratified
  on problem_type) per layer. Project held-out 20% and full SVAMP ':' into
  the k-dim space. Fit a fresh LDA classifier on the k-dim projection and
  report acc.

Layout: dim × 2 datasets × 13 layers = 78 slides + summaries.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[4]
CF_ACTS = REPO / "experiments" / "computation_probes" / "cf_balanced_colon_acts.pt"
CF_META = REPO / "experiments" / "computation_probes" / "cf_balanced_colon_acts_meta.json"
ORIG_ACTS = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
ORIG_META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2-gsm8k" / "cf_lda_80_20_slideshow_colon.pdf"
OUT_STATS = REPO / "visualizations-all" / "gpt2-gsm8k" / "cf_lda_80_20_colon_stats.json"

PROBLEM_TYPE_COLORS = {"Subtraction": "#1f77b4", "Addition": "#ff7f0e",
                       "Common-Division": "#2ca02c", "Multiplication": "#d62728"}
OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
SEED = 13
DIMS = [1, 2, 3]


def load(acts_p, meta_p):
    acts = torch.load(acts_p, map_location="cpu", weights_only=True).float().numpy()
    types = np.array(json.load(open(meta_p))["types"])
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y = np.array([op_to_idx.get(t, -1) for t in types])
    valid = y >= 0
    return acts[valid], y[valid]


def render_slide(pdf, dim, title, X_proj, y, acc, op_names=OPS):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title + f"  (acc={acc*100:.1f}%)", fontsize=11, fontweight="bold")
    if dim == 1:
        ax = fig.add_subplot(111)
        for c, name in enumerate(op_names):
            mask = y == c
            if not mask.any(): continue
            ax.hist(X_proj[mask, 0], bins=30, alpha=0.55,
                    color=PROBLEM_TYPE_COLORS[name], label=name, density=True)
        ax.set_xlabel("LD1"); ax.set_ylabel("density")
    elif dim == 2:
        ax = fig.add_subplot(111)
        for c, name in enumerate(op_names):
            mask = y == c
            if not mask.any(): continue
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1], s=8,
                       c=PROBLEM_TYPE_COLORS[name], label=name, alpha=0.6)
        ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.grid(alpha=0.3)
    else:  # 3D
        ax = fig.add_subplot(111, projection="3d")
        for c, name in enumerate(op_names):
            mask = y == c
            if not mask.any(): continue
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1], X_proj[mask, 2], s=8,
                       c=PROBLEM_TYPE_COLORS[name], label=name, alpha=0.6,
                       linewidths=0, depthshade=True)
        ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.set_zlabel("LD3")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig, dpi=140); plt.close(fig)


def main():
    cf_acts, cf_y = load(CF_ACTS, CF_META)
    orig_acts, orig_y = load(ORIG_ACTS, ORIG_META)
    print(f"cf={cf_acts.shape}, types={Counter(cf_y.tolist())}")
    print(f"orig={orig_acts.shape}, types={Counter(orig_y.tolist())}")
    Ncf = cf_acts.shape[0]; L = cf_acts.shape[1]
    idx_tr, idx_te = train_test_split(np.arange(Ncf), test_size=0.2,
                                       random_state=SEED, stratify=cf_y)
    cf_test_acc = {k: np.zeros(L) for k in DIMS}
    orig_acc = {k: np.zeros(L) for k in DIMS}

    print(f"writing {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        # Summary slide: dim-by-dim acc per layer for both datasets
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        # Compute first
        for k in DIMS:
            for l in range(L):
                Xtr = cf_acts[idx_tr, l, :]; ytr = cf_y[idx_tr]
                lda = LinearDiscriminantAnalysis(n_components=k).fit(Xtr, ytr)
                # k-dim classifier on the projection
                clf = LinearDiscriminantAnalysis()
                clf.fit(lda.transform(Xtr), ytr)
                Xte = lda.transform(cf_acts[idx_te, l, :])
                cf_test_acc[k][l] = clf.score(Xte, cf_y[idx_te])
                Xorig = lda.transform(orig_acts[:, l, :])
                orig_acc[k][l] = clf.score(Xorig, orig_y)
        xs = np.arange(L)
        for k, c in zip(DIMS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            axes[0].plot(xs, cf_test_acc[k] * 100, "o-", color=c, label=f"k={k}")
            axes[1].plot(xs, orig_acc[k] * 100, "o-", color=c, label=f"k={k}")
        for ax, title in [(axes[0], "cf_balanced held-out 20%"),
                          (axes[1], "SVAMP (cross-dataset transfer)")]:
            ax.set_xlabel("layer"); ax.set_ylabel("k-dim LDA acc (%)")
            ax.set_title(title); ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(20, 100)
        fig.suptitle("LDA(operator) trained on cf_balanced ':' residuals, evaluated at k-dim projection",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Per-(dim, layer) slides
        for k in DIMS:
            print(f"  dim={k} CF held-out slides...")
            for l in range(L):
                Xtr = cf_acts[idx_tr, l, :]; ytr = cf_y[idx_tr]
                lda = LinearDiscriminantAnalysis(n_components=k).fit(Xtr, ytr)
                Xte = lda.transform(cf_acts[idx_te, l, :])
                render_slide(pdf, k, f"cf_balanced (held-out) — layer {l} — dim={k}",
                             Xte, cf_y[idx_te], cf_test_acc[k][l])
            print(f"  dim={k} SVAMP transfer slides...")
            for l in range(L):
                Xtr = cf_acts[idx_tr, l, :]; ytr = cf_y[idx_tr]
                lda = LinearDiscriminantAnalysis(n_components=k).fit(Xtr, ytr)
                Xorig = lda.transform(orig_acts[:, l, :])
                render_slide(pdf, k, f"SVAMP (transfer) — layer {l} — dim={k}",
                             Xorig, orig_y, orig_acc[k][l])
    OUT_STATS.write_text(json.dumps({
        "L": int(L),
        "cf_test_acc": {k: v.tolist() for k, v in cf_test_acc.items()},
        "orig_acc": {k: v.tolist() for k, v in orig_acc.items()},
    }, indent=2))
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")
    print(f"stats -> {OUT_STATS}")


if __name__ == "__main__":
    main()
