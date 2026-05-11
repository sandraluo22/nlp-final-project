"""3D PCA at ':' residual on vary_numerals dataset. ':' analog of the
vary_numerals_pca_slideshow.pdf (originally from number_isolation_slideshows.py).
One slide per layer, colorings: plain, by `a`, by `b`, by `answer`.
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
ACTS = REPO / "experiments" / "computation_probes" / "vary_numerals_colon_acts.pt"
DATA = REPO.parent / "cf-datasets" / "vary_numerals.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2" / "vary_numerals_pca_slideshow_colon.pdf"


def main():
    acts = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(DATA))
    N, L, H = acts.shape
    a_arr = np.array([r.get("a", np.nan) for r in rows[:N]], dtype=float)
    b_arr = np.array([r.get("b", np.nan) for r in rows[:N]], dtype=float)
    ans = np.array([r.get("answer", np.nan) for r in rows[:N]], dtype=float)
    print(f"vary_numerals: shape={acts.shape}")

    print(f"writing {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        for coloring, vals, cmap, label in [
            ("plain", None, None, None),
            ("a", a_arr, "viridis", "a"),
            ("b", b_arr, "plasma", "b"),
            ("answer", ans, "viridis", "answer"),
        ]:
            for l in range(L):
                X = acts[:, l, :]
                n_pc = min(3, N - 1, H)
                pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0)
                xyz = pca.fit_transform(X)
                v = pca.explained_variance_ratio_
                fig = plt.figure(figsize=(8, 7))
                fig.suptitle(f"vary_numerals — layer {l} — PCA at ':' residual — coloring: {coloring}",
                             fontsize=11, fontweight="bold")
                ax = fig.add_subplot(111, projection="3d") if n_pc >= 3 else fig.add_subplot(111)
                if vals is None:
                    if n_pc >= 3:
                        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=14,
                                   c="#3357aa", alpha=0.7, linewidths=0, depthshade=True)
                    else:
                        ax.scatter(xyz[:, 0], xyz[:, 1] if n_pc > 1 else np.zeros(N),
                                   s=14, c="#3357aa", alpha=0.7, linewidths=0)
                else:
                    if n_pc >= 3:
                        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=14,
                                        c=vals, cmap=cmap, alpha=0.7, linewidths=0, depthshade=True)
                    else:
                        sc = ax.scatter(xyz[:, 0], xyz[:, 1] if n_pc > 1 else np.zeros(N),
                                        s=14, c=vals, cmap=cmap, alpha=0.7, linewidths=0)
                    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.08)
                    cb.set_label(label, fontsize=8)
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
