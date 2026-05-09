"""Combined number-isolation slideshow for CODI-GPT2.

ONE PDF: codi-work/visualizations-all/gpt2/number_isolation_combined.pdf
Per-dataset section (vary_numerals, vary_operator):
  For each layer (0..12) × each coloring in {a, b, answer, operator}:
    one 2D PCA slide showing all 6 latent steps in a 2x3 grid.
Note: degenerate colorings (operator in vary_numerals; a, b in vary_operator)
are rendered for symmetry but show a single uniform color since the coloring
axis doesn't vary in that dataset.

Final section: cosine-similarity heatmap between the principal components of
vary_numerals and vary_operator at every (layer, latent_step). Both PCAs live
in the same 768-dim feature space.
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


CODI_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODI_ROOT.parent
OUT_PDF = CODI_ROOT / "visualizations-all" / "gpt2" / "number_isolation_combined.pdf"
DATASETS = {
    "vary_numerals": {
        "acts": CODI_ROOT / "inference" / "runs" / "gpt2_vary_numerals" / "activations.pt",
        "data": PROJECT_ROOT / "cf-datasets" / "vary_numerals.json",
        "colorings": ["a", "b", "answer"],     # operator trivial (Sub-only)
        "step_label": "latent step",
    },
    "vary_a": {
        "acts": CODI_ROOT / "inference" / "runs" / "gpt2_vary_a" / "activations.pt",
        "data": PROJECT_ROOT / "cf-datasets" / "vary_a.json",
        "colorings": ["a", "answer"],          # b fixed=4, operator fixed → trivial
        "step_label": "latent step",
    },
    "vary_b": {
        "acts": CODI_ROOT / "inference" / "runs" / "gpt2_vary_b" / "activations.pt",
        "data": PROJECT_ROOT / "cf-datasets" / "vary_b.json",
        "colorings": ["b", "answer"],          # a fixed=200, operator fixed → trivial
        "step_label": "latent step",
    },
    "vary_operator": {
        "acts": CODI_ROOT / "inference" / "runs" / "gpt2_vary_operator" / "activations.pt",
        "data": PROJECT_ROOT / "cf-datasets" / "vary_operator.json",
        "colorings": ["answer", "operator"],   # a, b trivial (constants)
        "step_label": "latent step",
    },
}
COSSIM_JSON = CODI_ROOT / "visualizations-all" / "gpt2" / "number_isolation_cossim.json"

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


def fit_pca(acts: np.ndarray, n_components: int = 3
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns proj (L,S,N,k), var (L,S,k), comps (L,S,k,H) — unit PC axes."""
    N, S, L, H = acts.shape
    out = np.empty((L, S, N, n_components), dtype=np.float32)
    var = np.empty((L, S, n_components), dtype=np.float32)
    comps = np.empty((L, S, n_components, H), dtype=np.float32)
    for layer in range(L):
        for step in range(S):
            X = acts[:, step, layer, :]
            pca = PCA(n_components=n_components, svd_solver="randomized",
                      random_state=0)
            out[layer, step] = pca.fit_transform(X)
            var[layer, step] = pca.explained_variance_ratio_
            comps[layer, step] = pca.components_
    return out, var, comps


def cos_sim_pcs(comps_a: np.ndarray, comps_b: np.ndarray) -> np.ndarray:
    L, S, k, H = comps_a.shape
    out = np.empty((L, S, k, k), dtype=np.float32)
    for layer in range(L):
        for step in range(S):
            out[layer, step] = comps_a[layer, step] @ comps_b[layer, step].T
    return out


def render_slide(pdf, layer, pca_for_layer, var_for_layer, coloring, meta,
                 dim: int, dataset_label: str, step_label: str):
    S = pca_for_layer.shape[0]
    if dim == 3:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.875),
                                 subplot_kw={"projection": "3d"})
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.875))
    fig.suptitle(
        f"{dataset_label}  —  Layer {layer}  —  {dim}D PCA  —  coloring: {coloring}",
        fontsize=12,
    )
    legend_proxies: list[Line2D] = []
    cbar_mappable = None
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off"); continue
        xy = pca_for_layer[s]

        if coloring == "operator":
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = meta["problem_type"] == cls
                if mask.sum() == 0: continue
                if dim == 3:
                    ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2],
                               s=14, c=color, alpha=0.85, linewidths=0,
                               depthshade=True, rasterized=True)
                else:
                    ax.scatter(xy[mask, 0], xy[mask, 1],
                               s=22, c=color, alpha=0.85, linewidths=0,
                               rasterized=True)
            if s == 0:
                for cls, color in PROBLEM_TYPE_COLORS.items():
                    n = (meta["problem_type"] == cls).sum()
                    if n == 0: continue
                    legend_proxies.append(Line2D([0], [0], marker="o",
                        linestyle="", color=color, label=f"{cls} ({n})"))
        elif coloring in ("a", "b", "answer"):
            values = meta[coloring]
            norm = Normalize(vmin=values.min(), vmax=values.max())
            if dim == 3:
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=12, c=values,
                           cmap="viridis", norm=norm, alpha=0.9, linewidths=0,
                           depthshade=True, rasterized=True)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], s=20, c=values,
                           cmap="viridis", norm=norm, alpha=0.9, linewidths=0,
                           rasterized=True)
            if s == 0:
                cbar_mappable = ScalarMappable(norm=norm, cmap="viridis")
                cbar_mappable.set_array([])
        else:
            raise ValueError(coloring)

        ax.set_title(f"{step_label} {s+1}", fontsize=10)
        if dim == 3:
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.tick_params(axis="both", which="both", length=0, pad=-2)
            ax.set_xlabel(f"PC1 ({var_for_layer[s,0]*100:.1f}%)", fontsize=7, labelpad=-10)
            ax.set_ylabel(f"PC2 ({var_for_layer[s,1]*100:.1f}%)", fontsize=7, labelpad=-10)
            ax.set_zlabel(f"PC3 ({var_for_layer[s,2]*100:.1f}%)", fontsize=7, labelpad=-10)
        else:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"PC1 ({var_for_layer[s,0]*100:.1f}%)", fontsize=7)
            ax.set_ylabel(f"PC2 ({var_for_layer[s,1]*100:.1f}%)", fontsize=7)
        ax.grid(True, alpha=0.25)

    if legend_proxies:
        fig.legend(handles=legend_proxies, loc="lower center",
                   ncol=len(legend_proxies), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    elif cbar_mappable is not None:
        fig.subplots_adjust(right=0.93)
        cax = fig.add_axes([0.94, 0.10, 0.015, 0.80])
        cb = fig.colorbar(cbar_mappable, cax=cax)
        cb.set_label(coloring, fontsize=10)
    fig.subplots_adjust(top=0.92, bottom=0.07, left=0.04,
                        right=0.93 if cbar_mappable is not None else 0.98,
                        wspace=0.18, hspace=0.25)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def render_cossim_slide(pdf, layer, cossim_for_layer, step_label="latent step"):
    """cossim_for_layer shape (S, k, k). 2x3 grid of heatmaps."""
    S, k, _ = cossim_for_layer.shape
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.875))
    fig.suptitle(
        f"Layer {layer}  —  cos_sim(PC_i_numerals, PC_j_operator)  [signed]",
        fontsize=12,
    )
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off"); continue
        M = cossim_for_layer[s]
        ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels([f"OP{i+1}" for i in range(k)], fontsize=9)
        ax.set_yticklabels([f"NUM{i+1}" for i in range(k)], fontsize=9)
        ax.set_title(f"{step_label} {s+1}  max|cs|={np.abs(M).max():.2f}",
                     fontsize=10)
        for ii in range(k):
            for jj in range(k):
                ax.text(jj, ii, f"{M[ii,jj]:+.2f}", ha="center", va="center",
                        fontsize=9, color="black"
                        if abs(M[ii, jj]) < 0.5 else "white")
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.98,
                        wspace=0.30, hspace=0.40)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    cooked = {}
    for ds_name, cfg in DATASETS.items():
        print(f"\n=== {ds_name} ===", flush=True)
        acts = load_acts(cfg["acts"])
        rows = json.load(open(cfg["data"]))
        N = acts.shape[0]
        meta = {
            "problem_type": np.array([r["type"] for r in rows[:N]]),
            "a": np.array([r["a"] for r in rows[:N]], dtype=np.float64),
            "b": np.array([r["b"] for r in rows[:N]], dtype=np.float64),
            "answer": np.array([r["answer"] for r in rows[:N]], dtype=np.float64),
        }
        print(f"  N={N}  type counts={dict(Counter(meta['problem_type']))}")
        proj, var, comps = fit_pca(acts, n_components=3)
        cooked[ds_name] = {"proj": proj, "var": var, "comps": comps,
                           "meta": meta, "cfg": cfg}

    # Multiple cos_sim comparisons:
    pairs = [
        ("vary_numerals", "vary_operator"),
        ("vary_a",        "vary_operator"),
        ("vary_b",        "vary_operator"),
        ("vary_a",        "vary_b"),         # Direct number-vs-number isolation test
    ]
    cs_dict = {}
    summary = {}
    for a_name, b_name in pairs:
        if a_name not in cooked or b_name not in cooked:
            continue
        cs = cos_sim_pcs(cooked[a_name]["comps"], cooked[b_name]["comps"])
        cs_dict[(a_name, b_name)] = cs
        summary[f"{a_name}__VS__{b_name}"] = {
            "max_abs_per_step": np.abs(cs).max(axis=(2, 3)).tolist(),
            "mean_abs_per_step": np.abs(cs).mean(axis=(2, 3)).tolist(),
            "global_max_abs": float(np.abs(cs).max()),
            "global_mean_abs": float(np.abs(cs).mean()),
        }
        print(f"\ncos_sim {a_name} vs {b_name}:  max|cs|={float(np.abs(cs).max()):.3f}  "
              f"mean|cs|={float(np.abs(cs).mean()):.3f}", flush=True)
    COSSIM_JSON.write_text(json.dumps(summary, indent=2))

    print(f"\nwriting {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        for ds_name, info in cooked.items():
            proj, var, meta, cfg = info["proj"], info["var"], info["meta"], info["cfg"]
            for layer in range(proj.shape[0]):
                for coloring in cfg["colorings"]:
                    render_slide(pdf, layer,
                                 proj[layer, :, :, :2], var[layer, :, :2],
                                 coloring, meta, dim=2, dataset_label=ds_name,
                                 step_label=cfg["step_label"])
        # Cos-sim sections — one per pair
        for (a_name, b_name), cs in cs_dict.items():
            for layer in range(cs.shape[0]):
                fig = plt.figure(figsize=(14, 7.875))
                fig.suptitle(
                    f"cos_sim  ·  {a_name}  vs  {b_name}  ·  layer {layer}",
                    fontsize=12,
                )
                S, k, _ = cs[layer].shape
                ncols, nrows = 3, 2
                for s in range(S):
                    ax = fig.add_subplot(nrows, ncols, s + 1)
                    M = cs[layer, s]
                    ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
                    ax.set_xticks(range(k)); ax.set_yticks(range(k))
                    ax.set_xticklabels([f"{b_name[:8]}-PC{i+1}" for i in range(k)], fontsize=7)
                    ax.set_yticklabels([f"{a_name[:8]}-PC{i+1}" for i in range(k)], fontsize=7)
                    ax.set_title(f"latent step {s+1}  max|cs|={np.abs(M).max():.2f}",
                                 fontsize=10)
                    for ii in range(k):
                        for jj in range(k):
                            ax.text(jj, ii, f"{M[ii,jj]:+.2f}",
                                    ha="center", va="center", fontsize=8,
                                    color="black" if abs(M[ii, jj]) < 0.5 else "white")
                fig.subplots_adjust(top=0.92, bottom=0.05, left=0.06, right=0.98,
                                    wspace=0.35, hspace=0.40)
                pdf.savefig(fig, dpi=140)
                plt.close(fig)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
