"""Per-(layer, latent-step) 2D LDA of student activations on SVAMP, where the
two projected axes are chosen to maximize separation between the four SVAMP
problem types (Subtraction / Addition / Common-Division / Multiplication).
Same coloring schemes and slide ordering as pca_slideshow.py.

LDA is supervised on problem_type. The faithful/magnitude/plain colorings
re-use those same directions to ask whether problem-type-discriminating axes
also happen to separate other labels.

Layout (one slide per page, 6 subplots per slide):
  - one slide per layer, plain                  (17 slides)
  - one slide per layer, faithful/unfaithful    (17 slides)
  - one slide per layer, problem type           (17 slides)
  - one slide per layer, magnitude bucket       (17 slides)
Total: 68 slides.
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
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


REPO = Path(__file__).resolve().parents[3]
ACTS_PATH = REPO / "inference" / "runs" / "svamp_student" / "activations.pt"
RESULTS_PATH = REPO / "inference" / "runs" / "svamp_teacher" / "results.json"
JUDGED_PATH = REPO.parent / "cf-datasets" / "svamp_judged.json"
OUT_PDF = REPO / "visualizations-all" / "v1" / "lda_slideshow.pdf"

PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}
FAITHFUL_COLORS = {
    "faithful": "#2ca02c",
    "unfaithful": "#d62728",
    "teacher_incorrect": "#cccccc",
}
MAG_BUCKETS = ["<10", "10-99", "100-999", "1000+"]
MAG_COLORS = {
    "<10": "#440154",
    "10-99": "#3b528b",
    "100-999": "#21918c",
    "1000+": "#fde725",
}


def bucket_magnitude(ans: np.ndarray) -> np.ndarray:
    out = np.empty(len(ans), dtype=object)
    out[ans < 10] = "<10"
    out[(ans >= 10) & (ans < 100)] = "10-99"
    out[(ans >= 100) & (ans < 1000)] = "100-999"
    out[ans >= 1000] = "1000+"
    return out


def load_metadata() -> dict:
    print("loading metadata", flush=True)
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    answers = np.array(
        [float(str(a).replace(",", "")) for a in full["Answer"]], dtype=np.float64
    )
    teacher = json.load(open(RESULTS_PATH))
    judged = json.load(open(JUDGED_PATH))
    label_by_idx = {j["idx"]: j["label"] for j in judged}
    faithful = [
        label_by_idx[t["idx"]] if t["idx"] in label_by_idx else "teacher_incorrect"
        for t in teacher
    ]
    return {
        "n": len(types),
        "problem_type": np.array(types),
        "answer": answers,
        "faithful": np.array(faithful),
        "magnitude": bucket_magnitude(answers),
    }


def load_activations() -> np.ndarray:
    print(f"loading activations from {ACTS_PATH}", flush=True)
    t = torch.load(ACTS_PATH, map_location="cpu", weights_only=True)
    a = t.float().numpy()  # (N, latent_steps, layers+1, hidden)
    print(f"  shape={a.shape}  bytes={a.nbytes/1e9:.2f} GB", flush=True)
    return a


def fit_all_lda_2d(
    acts: np.ndarray, problem_type: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected coords (L, S, N, 2) and var-ratio (L, S, 2)."""
    N, S, L, _H = acts.shape
    out = np.empty((L, S, N, 2), dtype=np.float32)
    var_ratio = np.empty((L, S, 2), dtype=np.float32)
    print(
        f"fitting {L*S} LDAs (2D, supervised on problem_type, "
        f"per layer × per latent step)",
        flush=True,
    )
    for layer in range(L):
        for step in range(S):
            X = acts[:, step, layer, :]
            lda = LinearDiscriminantAnalysis(n_components=2, solver="svd")
            out[layer, step] = lda.fit_transform(X, problem_type)
            var_ratio[layer, step] = lda.explained_variance_ratio_[:2]
        print(f"  layer {layer}/{L-1} done", flush=True)
    return out, var_ratio


def render_slide(
    pdf: PdfPages,
    layer: int,
    proj_for_layer: np.ndarray,  # (S, N, 2)
    var_for_layer: np.ndarray,  # (S, 2)
    coloring: str,
    meta: dict,
):
    S = proj_for_layer.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.875))
    fig.suptitle(
        f"Layer {layer}  —  2D LDA (supervised on problem_type)  —  coloring: {coloring}",
        fontsize=12,
    )
    legend_proxies: list[Line2D] = []
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off")
            continue
        xy = proj_for_layer[s]

        if coloring == "plain":
            ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#3357aa", alpha=0.5, linewidths=0)

        elif coloring == "faithful":
            order = ["teacher_incorrect", "faithful", "unfaithful"]  # rare on top
            for cls in order:
                mask = meta["faithful"] == cls
                ax.scatter(
                    xy[mask, 0], xy[mask, 1],
                    s=10 if cls == "unfaithful" else 4,
                    c=FAITHFUL_COLORS[cls],
                    alpha=0.9 if cls == "unfaithful" else 0.4,
                    linewidths=0,
                )
            if s == 0:
                for cls in order:
                    n = (meta["faithful"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=FAITHFUL_COLORS[cls], label=f"{cls} ({n})")
                    )

        elif coloring == "problem_type":
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = meta["problem_type"] == cls
                ax.scatter(
                    xy[mask, 0], xy[mask, 1],
                    s=4, c=color, alpha=0.55, linewidths=0,
                )
            if s == 0:
                for cls, color in PROBLEM_TYPE_COLORS.items():
                    n = (meta["problem_type"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=color, label=f"{cls} ({n})")
                    )

        elif coloring == "magnitude":
            for cls in MAG_BUCKETS:
                mask = meta["magnitude"] == cls
                ax.scatter(
                    xy[mask, 0], xy[mask, 1],
                    s=4, c=MAG_COLORS[cls], alpha=0.6, linewidths=0,
                )
            if s == 0:
                for cls in MAG_BUCKETS:
                    n = (meta["magnitude"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=MAG_COLORS[cls], label=f"{cls} ({n})")
                    )
        else:
            raise ValueError(coloring)

        ax.set_title(f"latent step {s+1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"LD1 ({var_for_layer[s, 0]*100:.1f}%)", fontsize=8)
        ax.set_ylabel(f"LD2 ({var_for_layer[s, 1]*100:.1f}%)", fontsize=8)

    if legend_proxies:
        fig.legend(
            handles=legend_proxies,
            loc="lower center",
            ncol=len(legend_proxies),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.04, right=0.98, wspace=0.18, hspace=0.25)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main():
    meta = load_metadata()
    print(
        "labels: faithful=", dict(Counter(meta["faithful"])),
        " | type=", dict(Counter(meta["problem_type"])),
        " | mag=", dict(Counter(meta["magnitude"])),
        flush=True,
    )

    acts = load_activations()
    proj_all, var_all = fit_all_lda_2d(acts, meta["problem_type"])
    L = proj_all.shape[0]

    print(f"writing {OUT_PDF}", flush=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for coloring in ["faithful", "problem_type"]:
            print(f"  coloring: {coloring}", flush=True)
            for layer in range(L):
                render_slide(pdf, layer, proj_all[layer], var_all[layer], coloring, meta)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
