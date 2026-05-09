"""Project magnitude-matched counterfactual activations onto the LDA operator
axes that were fit on the original SVAMP activations. If operator clusters
survive, the LDA direction is genuinely operator-coding (not magnitude in
disguise). If clusters collapse, the original LDA was magnitude.

Slideshow: per (layer × latent step), one subplot showing
  - original SVAMP points (small gray, operator-colored borders) — reference
  - CF points (filled, operator-colored, opaque) — held-out projection
On the same LDA axes (fit on original).

Layout: 17 slides, one per layer, 6 latent-step subplots each.

Reports a per-(layer, latent-step) operator-classification accuracy on the CF
projection (using the LDA classifier fit on original) so we can quantify how
well the operator direction generalizes.
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
ORIG_ACTS = REPO / "inference" / "runs" / "svamp_student" / "activations.pt"
CF_ACTS = REPO / "inference" / "runs" / "cf_balanced_student" / "activations.pt"
CF_DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"
OUT_PDF = REPO / "visualizations-all" / "v2" / "cf_lda_compare.pdf"
OUT_STATS = REPO / "visualizations-all" / "v2" / "cf_lda_stats.json"

PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}


def load_orig_problem_types() -> np.ndarray:
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    return np.array(types)


def load_cf_metadata() -> tuple[np.ndarray, np.ndarray]:
    rows = json.load(open(CF_DATA))
    types = np.array([r["type"] for r in rows])
    bucket = np.array([r["output_bucket"] for r in rows])
    return types, bucket


def load_acts(path: Path) -> np.ndarray:
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t.float().numpy()  # (N, S, L, H)


def main():
    print("loading metadata + activations")
    orig_types = load_orig_problem_types()
    cf_types, cf_bucket = load_cf_metadata()
    orig = load_acts(ORIG_ACTS)
    cf = load_acts(CF_ACTS)
    print(f"  orig: {orig.shape}  cf: {cf.shape}")
    print(f"  cf type counts:   {dict(Counter(cf_types))}")
    print(f"  cf bucket counts: {dict(Counter(cf_bucket))}")

    L = orig.shape[2]
    S = orig.shape[1]

    # Fit LDA on each (layer, latent_step) using ORIGINAL SVAMP, transform both.
    proj_orig = np.empty((L, S, orig.shape[0], 2), dtype=np.float32)
    proj_cf = np.empty((L, S, cf.shape[0], 2), dtype=np.float32)
    var_ratio = np.empty((L, S, 2), dtype=np.float32)
    cf_acc = np.empty((L, S), dtype=np.float32)
    orig_acc = np.empty((L, S), dtype=np.float32)
    print(f"fitting {L*S} LDAs on original SVAMP, projecting CF, scoring")
    for layer in range(L):
        for step in range(S):
            X_orig = orig[:, step, layer, :]
            X_cf = cf[:, step, layer, :]
            lda = LinearDiscriminantAnalysis(n_components=2, solver="svd")
            lda.fit(X_orig, orig_types)
            proj_orig[layer, step] = lda.transform(X_orig)[:, :2]
            proj_cf[layer, step] = lda.transform(X_cf)[:, :2]
            var_ratio[layer, step] = lda.explained_variance_ratio_[:2]
            orig_acc[layer, step] = (lda.predict(X_orig) == orig_types).mean()
            cf_acc[layer, step] = (lda.predict(X_cf) == cf_types).mean()
        print(f"  layer {layer}/{L-1}  cf-acc(step1..6)={cf_acc[layer]}", flush=True)

    print("\nrendering slideshow")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for layer in range(L):
            fig, axes = plt.subplots(2, 3, figsize=(14, 7.875))
            fig.suptitle(
                f"Layer {layer}  —  CF (filled) vs original SVAMP (faint) on "
                f"original-fit LDA axes",
                fontsize=12,
            )
            legend_proxies = []
            for s, ax in enumerate(axes.ravel()):
                if s >= S:
                    ax.axis("off")
                    continue
                # Original (faint background)
                for cls, color in PROBLEM_TYPE_COLORS.items():
                    m = orig_types == cls
                    ax.scatter(
                        proj_orig[layer, s, m, 0], proj_orig[layer, s, m, 1],
                        s=3, c=color, alpha=0.12, linewidths=0,
                    )
                # Counterfactual (foreground)
                for cls, color in PROBLEM_TYPE_COLORS.items():
                    m = cf_types == cls
                    ax.scatter(
                        proj_cf[layer, s, m, 0], proj_cf[layer, s, m, 1],
                        s=8, c=color, alpha=0.7, linewidths=0,
                    )
                if s == 0 and not legend_proxies:
                    for cls, color in PROBLEM_TYPE_COLORS.items():
                        n_orig = (orig_types == cls).sum()
                        n_cf = (cf_types == cls).sum()
                        legend_proxies.append(
                            Line2D([0], [0], marker="o", linestyle="",
                                   color=color, label=f"{cls} (orig {n_orig} / cf {n_cf})")
                        )
                ax.set_title(
                    f"latent step {s+1}\n"
                    f"orig acc {orig_acc[layer, s]*100:.0f}% | cf acc {cf_acc[layer, s]*100:.0f}%",
                    fontsize=9,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(f"LD1 ({var_ratio[layer, s, 0]*100:.1f}%)", fontsize=8)
                ax.set_ylabel(f"LD2 ({var_ratio[layer, s, 1]*100:.1f}%)", fontsize=8)

            if legend_proxies:
                fig.legend(handles=legend_proxies, loc="lower center",
                           ncol=len(legend_proxies), fontsize=8, frameon=False,
                           bbox_to_anchor=(0.5, 0.0))
            fig.subplots_adjust(top=0.92, bottom=0.10, left=0.04, right=0.98,
                                wspace=0.18, hspace=0.4)
            pdf.savefig(fig, dpi=140)
            plt.close(fig)

    # Save accuracies
    stats = {
        "orig_acc": orig_acc.tolist(),
        "cf_acc": cf_acc.tolist(),
        "var_ratio": var_ratio.tolist(),
        "shape": {"layers_plus_emb": L, "latent_steps": S},
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"\nclassifier accuracy (trained on orig SVAMP, applied to held-out CF):")
    print("layer | step1   step2   step3   step4   step5   step6  | mean")
    for layer in range(L):
        row = "  ".join(f"{cf_acc[layer, s]*100:5.1f}%" for s in range(S))
        print(f" {layer:>4d} | {row}  | {cf_acc[layer].mean()*100:5.1f}%")
    chance = max(Counter(orig_types).values()) / len(orig_types)
    print(f"\nmajority-class baseline: {chance*100:.1f}%")
    print(f"saved -> {OUT_PDF}  +  {OUT_STATS}")


if __name__ == "__main__":
    main()
