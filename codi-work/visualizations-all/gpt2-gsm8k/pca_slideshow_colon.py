"""Per-layer 3D PCA of CODI-GPT-2 ':' residuals on SVAMP, 5 colorings.

Mirrors pca_slideshow.py but on the (N, 13, 768) ':' tensor (one position only)
instead of the (N, 6, 13, 768) latent-loop tensor. Each slide = one layer, 5
colorings across pages.

Layout: 13 layers × 5 colorings = 65 slides.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
ACTS = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
JUDGED = REPO.parent / "cf-datasets" / "gsm8k_judged.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2" / "pca_slideshow_colon.pdf"

PROBLEM_TYPE_COLORS = {"Subtraction": "#1f77b4", "Addition": "#ff7f0e",
                       "Common-Division": "#2ca02c", "Multiplication": "#d62728"}
FAITHFUL_COLORS = {"faithful": "#2ca02c", "unfaithful": "#d62728", "teacher_incorrect": "#cccccc"}
MAG_BUCKETS = ["<10", "10-99", "100-999", "1000+"]
MAG_COLORS = {"<10": "#440154", "10-99": "#3b528b", "100-999": "#21918c", "1000+": "#fde725"}


def bucket_magnitude(ans):
    out = np.empty(len(ans), dtype=object)
    a = np.asarray(ans, dtype=float)
    out[a < 10] = "<10"
    out[(a >= 10) & (a < 100)] = "10-99"
    out[(a >= 100) & (a < 1000)] = "100-999"
    out[a >= 1000] = "1000+"
    return out


def main():
    print("loading", ACTS)
    acts = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(META))
    N, L, H = acts.shape
    print(f"  shape={acts.shape}")

    # Pull problem types from SVAMP (matches meta["types"] but use both for sanity)
    types = np.array(meta["types"])
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    answers = np.where(np.isnan(gold), 0.0, gold)
    judged = json.load(open(JUDGED)) if JUDGED.exists() else []
    label_by_idx = {j["idx"]: j["label"] for j in judged}
    faithful = np.array([label_by_idx.get(i, "teacher_incorrect") for i in range(N)])
    log_answer = np.log10(np.maximum(answers, 1) + 1)
    magnitude = bucket_magnitude(answers)
    pred = np.array([np.nan if v is None else float(v) for v in meta["pred_int_extracted"]])
    correct = (~np.isnan(pred)) & (~np.isnan(gold)) & (np.abs(pred - gold) < 1e-3)
    print(f"  faithful: {dict(Counter(faithful))}")
    print(f"  types: {dict(Counter(types))}")
    print(f"  correct: {int(correct.sum())}/{N}")

    print("fitting PCA-3 per layer (13 layers)...")
    xyz = np.zeros((L, N, 3), dtype=np.float32)
    var_ratio = np.zeros((L, 3), dtype=np.float32)
    for l in range(L):
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
        xyz[l] = pca.fit_transform(acts[:, l, :])
        var_ratio[l] = pca.explained_variance_ratio_

    print(f"writing {OUT_PDF}")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for coloring in ["plain", "faithful", "problem_type", "magnitude", "log_answer", "correct"]:
            print(f"  coloring: {coloring}")
            for l in range(L):
                fig = plt.figure(figsize=(8, 7))
                fig.suptitle(f"Layer {l}  —  PCA of CODI-GPT-2 ':' residual on SVAMP  —  coloring: {coloring}",
                             fontsize=12, fontweight="bold")
                ax = fig.add_subplot(111, projection="3d")
                xy = xyz[l]
                v = var_ratio[l]
                legend_proxies = []
                if coloring == "plain":
                    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=6, c="#3357aa",
                               alpha=0.5, linewidths=0, depthshade=True, rasterized=True)
                elif coloring == "faithful":
                    for cls in ["teacher_incorrect", "faithful", "unfaithful"]:
                        mask = faithful == cls
                        ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2],
                                   s=14 if cls == "unfaithful" else 6,
                                   c=FAITHFUL_COLORS[cls],
                                   alpha=0.9 if cls == "unfaithful" else 0.4,
                                   linewidths=0, depthshade=True, rasterized=True)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=FAITHFUL_COLORS[cls],
                                                     label=f"{cls} ({int(mask.sum())})"))
                elif coloring == "problem_type":
                    for cls, color in PROBLEM_TYPE_COLORS.items():
                        mask = types == cls
                        ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2],
                                   s=6, c=color, alpha=0.6, linewidths=0,
                                   depthshade=True, rasterized=True)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=color, label=f"{cls} ({int(mask.sum())})"))
                elif coloring == "magnitude":
                    for cls in MAG_BUCKETS:
                        mask = magnitude == cls
                        ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2],
                                   s=6, c=MAG_COLORS[cls], alpha=0.6, linewidths=0,
                                   depthshade=True, rasterized=True)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=MAG_COLORS[cls], label=f"{cls} ({int(mask.sum())})"))
                elif coloring == "log_answer":
                    sc = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=6, c=log_answer,
                                    cmap="viridis", alpha=0.6, linewidths=0,
                                    depthshade=True, rasterized=True)
                    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.08)
                    cb.set_label("log10(answer+1)", fontsize=8)
                elif coloring == "correct":
                    for val, color, lbl in [(False, "#d62728", "wrong"),
                                            (True, "#2ca02c", "right")]:
                        mask = correct == val
                        ax.scatter(xy[mask, 0], xy[mask, 1], xy[mask, 2],
                                   s=6, c=color, alpha=0.55, linewidths=0,
                                   depthshade=True, rasterized=True)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=color,
                                                     label=f"{lbl} ({int(mask.sum())})"))
                ax.set_xlabel(f"PC1 ({v[0]*100:.1f}%)", fontsize=8)
                ax.set_ylabel(f"PC2 ({v[1]*100:.1f}%)", fontsize=8)
                ax.set_zlabel(f"PC3 ({v[2]*100:.1f}%)", fontsize=8)
                ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
                if legend_proxies:
                    ax.legend(handles=legend_proxies, fontsize=7, loc="upper right")
                fig.tight_layout()
                pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
