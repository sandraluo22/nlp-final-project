"""Per-(problem-type, layer) 3D PCA at ':' residual on SVAMP.

':' analog of pca_2_slideshow.py. Fits PCA per (operator subset, layer) and
plots the resulting 3D point cloud. One slide per (op, layer). 4 ops × 13
layers = 52 slides.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
ACTS = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2-gsm8k" / "pca_2_slideshow_colon.pdf"

OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
PROBLEM_TYPE_COLORS = {"Subtraction": "#1f77b4", "Addition": "#ff7f0e",
                       "Common-Division": "#2ca02c", "Multiplication": "#d62728"}


def main():
    acts = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(META))
    N, L, H = acts.shape
    types = np.array(meta["types"])

    print(f"writing {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        for op in OPS:
            mask = types == op
            if mask.sum() < 3:
                continue
            sub = acts[mask]
            print(f"  {op}: n={int(mask.sum())}")
            for l in range(L):
                X = sub[:, l, :]
                n_pc = min(3, X.shape[0] - 1, H)
                if n_pc < 1: continue
                pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0)
                xyz = pca.fit_transform(X)
                v = pca.explained_variance_ratio_
                fig = plt.figure(figsize=(8, 7))
                fig.suptitle(f"{op}  —  layer {l}  —  PCA of ':' residual (subset only)",
                             fontsize=11, fontweight="bold")
                if n_pc >= 3:
                    ax = fig.add_subplot(111, projection="3d")
                    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=10,
                               c=PROBLEM_TYPE_COLORS[op], alpha=0.7, linewidths=0,
                               depthshade=True, rasterized=True)
                    ax.set_xlabel(f"PC1 ({v[0]*100:.1f}%)", fontsize=8)
                    ax.set_ylabel(f"PC2 ({v[1]*100:.1f}%)", fontsize=8)
                    ax.set_zlabel(f"PC3 ({v[2]*100:.1f}%)", fontsize=8)
                    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
                else:
                    ax = fig.add_subplot(111)
                    ax.scatter(xyz[:, 0], xyz[:, 1] if n_pc > 1 else np.zeros(xyz.shape[0]),
                               s=10, c=PROBLEM_TYPE_COLORS[op], alpha=0.7,
                               linewidths=0, rasterized=True)
                    ax.set_xlabel("PC1"); ax.grid(alpha=0.3)
                fig.tight_layout()
                pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"done -> {OUT_PDF}")


if __name__ == "__main__":
    main()
