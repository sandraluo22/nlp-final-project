"""Per-layer 2D LDA on CF_balanced ':' residuals, supervised on problem type.

Operator separation should be clean (since LDA is trained on problem type).
Magnitude (output bucket) coloring tests whether the operator-supervised axes
also separate magnitude buckets — for cf_balanced this is the disentanglement
test because magnitude is decoupled from operator by construction.

Output PDF: 13 layers × 2 colorings = 26 slides.
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

REPO = Path(__file__).resolve().parents[4]
ACTS = REPO / "experiments" / "computation_probes" / "cf_balanced_colon_acts.pt"
META = REPO / "experiments" / "computation_probes" / "cf_balanced_colon_acts_meta.json"
CF_DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2-gsm8k" / "cf_lda_slideshow_colon.pdf"

PROBLEM_TYPE_COLORS = {"Subtraction": "#1f77b4", "Addition": "#ff7f0e",
                       "Common-Division": "#2ca02c", "Multiplication": "#d62728"}
BUCKETS_ORDER = ["<10", "10-99", "100-999", "1000+"]
BUCKET_COLORS = {"<10": "#440154", "10-99": "#3b528b", "100-999": "#21918c", "1000+": "#fde725"}


def main():
    acts = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(META))
    N, L, H = acts.shape
    types = np.array(meta["types"])

    rows = json.load(open(CF_DATA))
    buckets = np.array([r.get("output_bucket", "?") for r in rows[:N]])
    pred = np.array([np.nan if v is None else float(v) for v in meta["pred_int_extracted"]])
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    correct = (~np.isnan(pred)) & (~np.isnan(gold)) & (np.abs(pred - gold) < 1e-3)
    print(f"N={N} types={dict(Counter(types))} buckets={dict(Counter(buckets))}  correct={int(correct.sum())}/{N}")

    OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y = np.array([op_to_idx.get(t, -1) for t in types])
    valid = y >= 0
    print(f"  valid={int(valid.sum())}")

    print("fitting LDA per layer...")
    xy = np.zeros((L, N, 2), dtype=np.float32)
    acc = np.zeros(L)
    for l in range(L):
        X = acts[valid, l, :]; yv = y[valid]
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, yv)
        xy[l, valid] = lda.transform(X).astype(np.float32)
        acc[l] = lda.score(X, yv)
    print(f"  LDA in-sample acc: {[f'{a:.2f}' for a in acc]}")

    print(f"writing {OUT_PDF}")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for coloring in ["problem_type", "output_bucket", "correct"]:
            for l in range(L):
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.suptitle(f"Layer {l}  —  LDA(operator) on cf_balanced ':' residual  "
                             f"—  coloring: {coloring}  (acc={acc[l]*100:.1f}%)",
                             fontsize=11, fontweight="bold")
                pp = xy[l]
                legend_proxies = []
                if coloring == "problem_type":
                    for cls, color in PROBLEM_TYPE_COLORS.items():
                        mask = (types == cls) & valid
                        ax.scatter(pp[mask, 0], pp[mask, 1], s=6, c=color, alpha=0.6, linewidths=0)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=color, label=f"{cls} ({int(mask.sum())})"))
                elif coloring == "output_bucket":
                    for cls in BUCKETS_ORDER:
                        mask = (buckets == cls) & valid
                        if not mask.any(): continue
                        ax.scatter(pp[mask, 0], pp[mask, 1], s=6, c=BUCKET_COLORS[cls],
                                   alpha=0.6, linewidths=0)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=BUCKET_COLORS[cls],
                                                     label=f"{cls} ({int(mask.sum())})"))
                else:  # "correct"
                    for val, color, lbl in [(False, "#d62728", "wrong"),
                                            (True, "#2ca02c", "right")]:
                        mask = (correct == val) & valid
                        ax.scatter(pp[mask, 0], pp[mask, 1], s=6, c=color,
                                   alpha=0.55, linewidths=0)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=color,
                                                     label=f"{lbl} ({int(mask.sum())})"))
                ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.grid(alpha=0.3)
                ax.legend(handles=legend_proxies, fontsize=7, loc="upper right")
                fig.tight_layout(rect=(0, 0, 1, 0.94))
                pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"done -> {OUT_PDF}")


if __name__ == "__main__":
    main()
