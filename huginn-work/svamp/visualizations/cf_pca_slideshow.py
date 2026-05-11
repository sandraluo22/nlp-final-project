"""Per-(core_block, recurrence-step) 3D PCA of Huginn-3.5B activations on the
*counterfactual* magnitude-balanced SVAMP set, colored three ways.

Activations: (N_cf=676, S=32, L=4, H=5280)

Layout (one slide per page, 32 subplots per slide in a 4x8 grid):
  - one slide per core block, plain                  (4 slides)
  - one slide per core block, problem type           (4 slides)
  - one slide per core block, output bucket          (4 slides)
Total: 12 slides.

Colorings:
  - plain          : sanity / overall geometry
  - problem_type   : does PCA find operator structure unsupervised?
  - output_bucket  : on the magnitude-balanced CF set, does PCA still latch
                     onto magnitude even though it's no longer correlated with
                     operator? (the disentanglement test)
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.decomposition import PCA


HUGINN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HUGINN_ROOT.parent
ACTS_PATH = HUGINN_ROOT / "latent-sweep" / "huginn_cf_balanced" / "K32" / "activations.pt"
CF_DATA = PROJECT_ROOT / "cf-datasets" / "cf_balanced.json"
OUT_PDF = HUGINN_ROOT / "visualizations" / "cf_pca_slideshow.pdf"

PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}
BUCKETS_ORDER = ["<10", "10-99", "100-999", "1000+"]
BUCKET_COLORS = {
    "<10": "#440154",
    "10-99": "#3b528b",
    "100-999": "#21918c",
    "1000+": "#fde725",
}


def load_metadata() -> dict:
    rows = json.load(open(CF_DATA))
    return {
        "n": len(rows),
        "problem_type": np.array([r["type"] for r in rows]),
        "output_bucket": np.array([r["output_bucket"] for r in rows]),
    }


def load_activations() -> np.ndarray:
    print(f"loading activations from {ACTS_PATH}", flush=True)
    t = torch.load(ACTS_PATH, map_location="cpu", weights_only=True)
    a = t.float().numpy()  # (N, S, L, H)
    print(f"  shape={a.shape}  bytes={a.nbytes/1e9:.2f} GB", flush=True)
    return a


def fit_all_pca_3d(acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return projected coords (L, S, N, 3) and var-ratio (L, S, 3)."""
    N, S, L, _H = acts.shape
    out = np.empty((L, S, N, 3), dtype=np.float32)
    var_ratio = np.empty((L, S, 3), dtype=np.float32)
    print(f"fitting {L*S} PCAs (3D, per core block × per recurrence step)", flush=True)
    for layer in range(L):
        for step in range(S):
            X = acts[:, step, layer, :]
            pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
            out[layer, step] = pca.fit_transform(X)
            var_ratio[layer, step] = pca.explained_variance_ratio_
        print(f"  block {layer}/{L-1} done", flush=True)
    return out, var_ratio


def _strip_3d_axes(ax, var_ratio: np.ndarray | None = None):
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0, pad=-2)
    if var_ratio is not None:
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=7, labelpad=-10)
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=7, labelpad=-10)
        ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)", fontsize=7, labelpad=-10)
    ax.grid(True, alpha=0.25)


def render_slide(
    pdf: PdfPages,
    layer: int,
    pca_for_layer: np.ndarray,   # (S, N, 3)
    var_for_layer: np.ndarray,   # (S, 3)
    coloring: str,
    meta: dict,
):
    S = pca_for_layer.shape[0]
    fig, axes = plt.subplots(4, 8, figsize=(24, 13), subplot_kw={"projection": "3d"})
    fig.suptitle(
        f"Core block {layer}  —  3D PCA of CF latent activations  —  coloring: {coloring}",
        fontsize=12,
    )
    legend_proxies: list[Line2D] = []
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off")
            continue
        xy = pca_for_layer[s]

        if coloring == "plain":
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2],
                       s=4, c="#3357aa", alpha=0.5, linewidths=0,
                       depthshade=True, rasterized=True)

        elif coloring == "problem_type":
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = meta["problem_type"] == cls
                ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2],
                           s=4, c=color, alpha=0.55, linewidths=0,
                           depthshade=True, rasterized=True)
            if s == 0:
                for cls, color in PROBLEM_TYPE_COLORS.items():
                    n = (meta["problem_type"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=color, label=f"{cls} ({n})")
                    )

        elif coloring == "output_bucket":
            for cls in BUCKETS_ORDER:
                mask = meta["output_bucket"] == cls
                ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2],
                           s=4, c=BUCKET_COLORS[cls], alpha=0.6, linewidths=0,
                           depthshade=True, rasterized=True)
            if s == 0:
                for cls in BUCKETS_ORDER:
                    n = (meta["output_bucket"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=BUCKET_COLORS[cls], label=f"{cls} ({n})")
                    )
        else:
            raise ValueError(coloring)

        ax.set_title(f"K={s+1}", fontsize=10)
        _strip_3d_axes(ax, var_ratio=var_for_layer[s])

    if legend_proxies:
        fig.legend(handles=legend_proxies, loc="lower center",
                   ncol=len(legend_proxies), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.02, right=0.98,
                        wspace=0.08, hspace=0.12)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main():
    meta = load_metadata()
    print("CF labels: type=", dict(Counter(meta["problem_type"])),
          " | output_bucket=", dict(Counter(meta["output_bucket"])),
          flush=True)

    acts = load_activations()
    pca_all, var_all = fit_all_pca_3d(acts)
    L = pca_all.shape[0]

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT_PDF}", flush=True)
    with PdfPages(OUT_PDF) as pdf:
        for coloring in ["plain", "problem_type", "output_bucket"]:
            print(f"  coloring: {coloring}", flush=True)
            for layer in range(L):
                render_slide(pdf, layer, pca_all[layer], var_all[layer], coloring, meta)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
