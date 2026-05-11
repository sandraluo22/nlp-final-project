"""3D PCA at ':' residual on vary_operator dataset. ':' analog of
vary_operator_pca_slideshow.pdf. One slide per layer, colored by operator type.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
ACTS = REPO / "experiments" / "computation_probes" / "vary_operator_colon_acts.pt"
META = REPO / "experiments" / "computation_probes" / "vary_operator_colon_acts_meta.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2" / "vary_operator_pca_slideshow_colon.pdf"

PROBLEM_TYPE_COLORS = {"Subtraction": "#1f77b4", "Addition": "#ff7f0e",
                       "Common-Division": "#2ca02c", "Multiplication": "#d62728"}


def main():
    acts = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(META))
    types = np.array(meta["types"])
    N, L, H = acts.shape
    print(f"vary_operator: shape={acts.shape}  types={dict(zip(*np.unique(types, return_counts=True)))}")

    print(f"writing {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        for coloring in ["plain", "operator"]:
            for l in range(L):
                X = acts[:, l, :]
                n_pc = min(3, N - 1, H)
                pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0)
                xyz = pca.fit_transform(X)
                v = pca.explained_variance_ratio_
                fig = plt.figure(figsize=(8, 7))
                fig.suptitle(f"vary_operator — layer {l} — PCA at ':' residual — coloring: {coloring}",
                             fontsize=11, fontweight="bold")
                ax = fig.add_subplot(111, projection="3d") if n_pc >= 3 else fig.add_subplot(111)
                if coloring == "plain":
                    if n_pc >= 3:
                        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=16,
                                   c="#3357aa", alpha=0.7, linewidths=0, depthshade=True)
                    else:
                        ax.scatter(xyz[:, 0], xyz[:, 1] if n_pc > 1 else np.zeros(N),
                                   s=16, c="#3357aa", alpha=0.7, linewidths=0)
                else:
                    legend = []
                    for op, color in PROBLEM_TYPE_COLORS.items():
                        mask = types == op
                        if not mask.any(): continue
                        if n_pc >= 3:
                            ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], s=20,
                                       c=color, alpha=0.85, linewidths=0, depthshade=True)
                        else:
                            ax.scatter(xyz[mask, 0],
                                       xyz[mask, 1] if n_pc > 1 else np.zeros(mask.sum()),
                                       s=20, c=color, alpha=0.85, linewidths=0)
                        legend.append(Line2D([0], [0], marker="o", linestyle="",
                                             color=color, label=f"{op} ({int(mask.sum())})"))
                    if legend: ax.legend(handles=legend, fontsize=8, loc="upper right")
                if n_pc >= 3:
                    ax.set_xlabel(f"PC1 ({v[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({v[1]*100:.1f}%)")
                    ax.set_zlabel(f"PC3 ({v[2]*100:.1f}%)")
                    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
                else:
                    ax.set_xlabel("PC1"); ax.grid(alpha=0.3)
                fig.tight_layout()
                pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"done -> {OUT_PDF}")


if __name__ == "__main__":
    main()
