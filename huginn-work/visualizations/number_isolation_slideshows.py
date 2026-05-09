"""3D PCA slideshows for the two number-isolation datasets on Huginn-3.5B.

Outputs two PDFs into huginn-work/visualizations/:
  vary_numerals_pca_slideshow.pdf   — 4 core blocks × 32 recurrence steps
                                       Colorings: plain, by `a`, by `answer`
  vary_operator_pca_slideshow.pdf   — 4 core blocks × 32 recurrence steps
                                       Colorings: plain, by operator

Activations:
  vary_numerals : (80, 32, 4, 5280)  Subtraction-only, varied (a, b)
  vary_operator : (24, 32, 4, 5280)  fixed (a, b)=(12, 4), varied operator/scenario
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA


HUGINN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HUGINN_ROOT.parent
ACTS_NUM = HUGINN_ROOT / "latent-sweep" / "huginn_vary_numerals" / "K32" / "activations.pt"
ACTS_OP  = HUGINN_ROOT / "latent-sweep" / "huginn_vary_operator" / "K32" / "activations.pt"
DATA_NUM = PROJECT_ROOT / "cf-datasets" / "vary_numerals.json"
DATA_OP  = PROJECT_ROOT / "cf-datasets" / "vary_operator.json"
OUT_NUM = HUGINN_ROOT / "visualizations" / "vary_numerals_pca_slideshow.pdf"
OUT_OP  = HUGINN_ROOT / "visualizations" / "vary_operator_pca_slideshow.pdf"


PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}


def load_acts(path: Path) -> np.ndarray:
    print(f"loading {path}", flush=True)
    a = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    print(f"  shape={a.shape}", flush=True)
    return a


def fit_all_pca_3d(acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N, S, L, _H = acts.shape
    out = np.empty((L, S, N, 3), dtype=np.float32)
    var = np.empty((L, S, 3), dtype=np.float32)
    print(f"fitting {L*S} PCAs (3D)", flush=True)
    for layer in range(L):
        for step in range(S):
            X = acts[:, step, layer, :]
            pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
            out[layer, step] = pca.fit_transform(X)
            var[layer, step] = pca.explained_variance_ratio_
        print(f"  block {layer}/{L-1}", flush=True)
    return out, var


def _strip_3d_axes(ax, vr):
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0, pad=-2)
    ax.set_xlabel(f"PC1 ({vr[0]*100:.1f}%)", fontsize=7, labelpad=-10)
    ax.set_ylabel(f"PC2 ({vr[1]*100:.1f}%)", fontsize=7, labelpad=-10)
    ax.set_zlabel(f"PC3 ({vr[2]*100:.1f}%)", fontsize=7, labelpad=-10)
    ax.grid(True, alpha=0.25)


def render_slide(pdf, layer, pca_for_layer, var_for_layer, coloring, meta,
                 title_prefix=""):
    S = pca_for_layer.shape[0]
    fig, axes = plt.subplots(4, 8, figsize=(24, 13), subplot_kw={"projection": "3d"})
    fig.suptitle(
        f"Core block {layer}  —  {title_prefix}3D PCA  —  coloring: {coloring}",
        fontsize=12,
    )
    legend_proxies = []
    cbar_mappable = None
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off"); continue
        xy = pca_for_layer[s]

        if coloring == "plain":
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=6, c="#3357aa",
                       alpha=0.6, linewidths=0, depthshade=True, rasterized=True)
        elif coloring == "operator":
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = meta["problem_type"] == cls
                if mask.sum() == 0: continue
                ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2],
                           s=12, c=color, alpha=0.8, linewidths=0,
                           depthshade=True, rasterized=True)
            if s == 0:
                for cls, color in PROBLEM_TYPE_COLORS.items():
                    n = (meta["problem_type"] == cls).sum()
                    if n == 0: continue
                    legend_proxies.append(Line2D([0], [0], marker="o",
                        linestyle="", color=color, label=f"{cls} ({n})"))
        elif coloring in ("a", "answer"):
            values = meta[coloring]
            norm = Normalize(vmin=values.min(), vmax=values.max())
            sc = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=8, c=values,
                            cmap="viridis", norm=norm, alpha=0.85,
                            linewidths=0, depthshade=True, rasterized=True)
            if s == 0:
                cbar_mappable = ScalarMappable(norm=norm, cmap="viridis")
                cbar_mappable.set_array([])
        else:
            raise ValueError(coloring)

        ax.set_title(f"K={s+1}", fontsize=10)
        _strip_3d_axes(ax, var_for_layer[s])

    if legend_proxies:
        fig.legend(handles=legend_proxies, loc="lower center",
                   ncol=len(legend_proxies), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    elif cbar_mappable is not None:
        fig.subplots_adjust(right=0.95)
        cax = fig.add_axes([0.965, 0.10, 0.012, 0.80])
        cb = fig.colorbar(cbar_mappable, cax=cax)
        cb.set_label(coloring, fontsize=10)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.02, right=0.95,
                        wspace=0.08, hspace=0.12)
    pdf.savefig(fig, dpi=130)
    plt.close(fig)


def render_slideshow(acts_path, data_path, out_pdf, dataset_label, colorings):
    acts = load_acts(acts_path)
    rows = json.load(open(data_path))
    N = acts.shape[0]
    meta = {
        "problem_type": np.array([r["type"] for r in rows[:N]]),
        "a": np.array([r["a"] for r in rows[:N]], dtype=np.float64),
        "b": np.array([r["b"] for r in rows[:N]], dtype=np.float64),
        "answer": np.array([r["answer"] for r in rows[:N]], dtype=np.float64),
    }
    print(f"  N={N}  type counts={dict(Counter(meta['problem_type']))}")

    pca_all, var_all = fit_all_pca_3d(acts)
    L = pca_all.shape[0]

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        for coloring in colorings:
            print(f"  coloring: {coloring}")
            for layer in range(L):
                render_slide(pdf, layer, pca_all[layer], var_all[layer],
                             coloring, meta, title_prefix=f"{dataset_label}: ")
    print(f"done -> {out_pdf}  ({out_pdf.stat().st_size/1e6:.1f} MB)")


def main():
    print("=== vary_numerals (Subtraction, varied a/b) ===")
    render_slideshow(
        ACTS_NUM, DATA_NUM, OUT_NUM,
        dataset_label="vary_numerals",
        colorings=["plain", "a", "answer"],
    )
    print("\n=== vary_operator (fixed a=12 b=4, varied operator) ===")
    render_slideshow(
        ACTS_OP, DATA_OP, OUT_OP,
        dataset_label="vary_operator",
        colorings=["plain", "operator"],
    )


if __name__ == "__main__":
    main()
