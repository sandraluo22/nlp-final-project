"""Per-(problem-type, layer, latent-step) 3D PCA of student activations on
SVAMP. PCA is fit on the subset of points that share a problem type, then
rendered per layer like the other slideshows.

Layout (one slide per page, 6 subplots per slide):
  - 17 slides for problem type Addition
  - 17 slides for problem type Subtraction
  - 17 slides for problem type Multiplication
  - 17 slides for problem type Common-Division
Total: 68 slides.

Each slide is a single layer; subplots are the 6 latent steps. Points within a
slide are colored by the problem type's signature color (i.e., it's the "plain"
analogue — no overlay), since every point on the slide belongs to that type.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.decomposition import PCA


REPO = Path(__file__).resolve().parent.parent.parent
ACTS_PATH = REPO / "inference" / "runs" / "svamp_student" / "activations.pt"
OUT_PDF = REPO / "visualizations-student-correct" / "sc-v2-B" / "pca_2_slideshow.pdf"

# Order: Addition, Subtraction, Multiplication, Common-Division.
PROBLEM_TYPES_ORDER = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}


def load_problem_types() -> np.ndarray:
    print("loading SVAMP types", flush=True)
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    return np.array(types)


def load_activations() -> np.ndarray:
    print(f"loading activations from {ACTS_PATH}", flush=True)
    t = torch.load(ACTS_PATH, map_location="cpu", weights_only=True)
    a = t.float().numpy()
    print(f"  shape={a.shape}  bytes={a.nbytes/1e9:.2f} GB", flush=True)
    return a


def fit_pca_3d_for_subset(
    acts_subset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """acts_subset: (n_sub, S, L, H). Returns coords (L, S, n_sub, 3) and
    var-ratio (L, S, 3)."""
    n_sub, S, L, _H = acts_subset.shape
    out = np.empty((L, S, n_sub, 3), dtype=np.float32)
    var_ratio = np.empty((L, S, 3), dtype=np.float32)
    for layer in range(L):
        for step in range(S):
            X = acts_subset[:, step, layer, :]
            pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
            out[layer, step] = pca.fit_transform(X)
            var_ratio[layer, step] = pca.explained_variance_ratio_
    return out, var_ratio


def _strip_3d_axes(ax, var_ratio: np.ndarray | None = None):
    """Hide tick labels but keep axis labels (with variance % if provided)."""
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0, pad=-2)
    if var_ratio is not None:
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=7, labelpad=-10)
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=7, labelpad=-10)
        ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)", fontsize=7, labelpad=-10)
    else:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_visible(False)
    ax.grid(True, alpha=0.25)


def render_slide(
    pdf: PdfPages,
    layer: int,
    proj_for_layer: np.ndarray,  # (S, n_sub, 3)
    var_for_layer: np.ndarray,  # (S, 3)
    problem_type: str,
    n_sub: int,
):
    S = proj_for_layer.shape[0]
    color = PROBLEM_TYPE_COLORS[problem_type]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.875), subplot_kw={"projection": "3d"})
    fig.suptitle(
        f"Layer {layer}  —  3D PCA fit on {problem_type} only  (n={n_sub})",
        fontsize=12,
    )
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off")
            continue
        xy = proj_for_layer[s]
        ax.scatter(
            xy[:, 0], xy[:, 1], xy[:, 2],
            s=5, c=color, alpha=0.6, linewidths=0,
            depthshade=True, rasterized=True,
        )
        ax.set_title(f"latent step {s+1}", fontsize=10)
        _strip_3d_axes(ax, var_ratio=var_for_layer[s])

    fig.subplots_adjust(top=0.93, bottom=0.04, left=0.02, right=0.98, wspace=0.08, hspace=0.12)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main():
    problem_type = load_problem_types()
    acts = load_activations()

    # Filter to student-correct.
    sr = json.load(open(REPO / "inference/runs/svamp_student/results.json"))
    mask = np.array([r["correct"] for r in sr], dtype=bool)
    print(f"filtering to student-correct: {int(mask.sum())}/{len(mask)} kept", flush=True)
    acts = acts[mask]
    problem_type = problem_type[mask]
    print("type histogram (student-correct only):",
          dict(Counter(problem_type)), flush=True)

    print(f"writing {OUT_PDF}", flush=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for ptype in PROBLEM_TYPES_ORDER:
            mask = problem_type == ptype
            n_sub = int(mask.sum())
            print(f"  problem type: {ptype}  (n={n_sub}) — fitting PCA on subset", flush=True)
            sub = acts[mask]  # (n_sub, S, L, H)
            proj, var_ratio = fit_pca_3d_for_subset(sub)
            L = proj.shape[0]
            for layer in range(L):
                render_slide(pdf, layer, proj[layer], var_ratio[layer], ptype, n_sub)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
