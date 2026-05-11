"""Per-(layer, latent-step) 2D LDA on the *counterfactual* magnitude-matched
SVAMP activations, supervised on problem type. Colorings:
  - problem_type   : sanity check that the LDA finds operator separation
  - output_bucket  : does the operator-supervised axis also separate magnitude
                     buckets? On magnitude-matched CF, magnitude is decoupled
                     from operator by construction, so the same axes should
                     NOT separate buckets — that's the disentanglement test.

Layout:
  - 4 slides, problem_type coloring
  - 4 slides, output_bucket coloring
Total: 34 slides.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


HUGINN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HUGINN_ROOT.parent
ACTS_PATH = HUGINN_ROOT / "latent-sweep" / "huginn_cf_balanced" / "K32" / "activations.pt"
CF_DATA = PROJECT_ROOT / "cf-datasets" / "cf_balanced.json"
OUT_PDF = HUGINN_ROOT / "visualizations" / "cf_lda_slideshow.pdf"

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


def fit_all_lda_2d(
    acts: np.ndarray, problem_type: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit LDA per (layer, latent_step) supervised on problem_type. Returns
    coords (L, S, N, 2) and var-ratio (L, S, 2)."""
    N, S, L, _H = acts.shape
    out = np.empty((L, S, N, 2), dtype=np.float32)
    var_ratio = np.empty((L, S, 2), dtype=np.float32)
    print(
        f"fitting {L*S} LDAs (2D, supervised on CF problem_type, "
        f"per core block × per recurrence step)",
        flush=True,
    )
    for layer in range(L):
        for step in range(S):
            X = acts[:, step, layer, :]
            lda = LinearDiscriminantAnalysis(n_components=2, solver="svd")
            out[layer, step] = lda.fit_transform(X, problem_type)
            var_ratio[layer, step] = lda.explained_variance_ratio_[:2]
        print(f"  block {layer}/{L-1} done", flush=True)
    return out, var_ratio


def render_slide(
    pdf: PdfPages,
    layer: int,
    proj_for_layer: np.ndarray,  # (S, N, 2)
    var_for_layer: np.ndarray,   # (S, 2)
    coloring: str,
    meta: dict,
):
    S = proj_for_layer.shape[0]
    fig, axes = plt.subplots(4, 8, figsize=(24, 13))
    fig.suptitle(
        f"Core block {layer}  —  CF 2D LDA (supervised on problem_type)  "
        f"—  coloring: {coloring}",
        fontsize=12,
    )
    legend_proxies: list[Line2D] = []
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off")
            continue
        xy = proj_for_layer[s]

        if coloring == "problem_type":
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = meta["problem_type"] == cls
                ax.scatter(xy[mask, 0], xy[mask, 1],
                           s=4, c=color, alpha=0.55, linewidths=0)
            if s == 0:
                for cls, color in PROBLEM_TYPE_COLORS.items():
                    n = (meta["problem_type"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=color, label=f"{cls} ({n})"))
        elif coloring == "output_bucket":
            for cls in BUCKETS_ORDER:
                mask = meta["output_bucket"] == cls
                ax.scatter(xy[mask, 0], xy[mask, 1],
                           s=4, c=BUCKET_COLORS[cls], alpha=0.55, linewidths=0)
            if s == 0:
                for cls in BUCKETS_ORDER:
                    n = (meta["output_bucket"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=BUCKET_COLORS[cls], label=f"{cls} ({n})"))
        else:
            raise ValueError(coloring)

        ax.set_title(f"K={s+1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"LD1 ({var_for_layer[s, 0]*100:.1f}%)", fontsize=8)
        ax.set_ylabel(f"LD2 ({var_for_layer[s, 1]*100:.1f}%)", fontsize=8)

    if legend_proxies:
        fig.legend(handles=legend_proxies, loc="lower center",
                   ncol=len(legend_proxies), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.04, right=0.98,
                        wspace=0.18, hspace=0.25)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main():
    meta = load_metadata()
    print("CF labels:",
          " problem_type=", dict(Counter(meta["problem_type"])),
          " | output_bucket=", dict(Counter(meta["output_bucket"])),
          flush=True)

    acts = load_activations()
    proj_all, var_all = fit_all_lda_2d(acts, meta["problem_type"])
    L = proj_all.shape[0]

    print(f"writing {OUT_PDF}", flush=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for coloring in ["problem_type", "output_bucket"]:
            print(f"  coloring: {coloring}", flush=True)
            for layer in range(L):
                render_slide(pdf, layer, proj_all[layer], var_all[layer], coloring, meta)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
