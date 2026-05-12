"""Per-layer 2D LDA of CODI-GPT-2 ':' residuals on SVAMP, operator-supervised.

Direct ':' analog of lda_slideshow.py. LDA is fit on operator (4 classes);
the 2D projection is then re-colored by:
  - plain, faithful, problem_type, magnitude, log_answer.

Layout: 13 layers × 5 colorings = 65 slides.
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
ACTS = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
JUDGED = REPO.parent / "cf-datasets" / "gsm8k_judged.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2-gsm8k" / "lda_slideshow_colon.pdf"

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
    print(f"  correct: {int(correct.sum())}/{N}")

    # Map labels for LDA
    OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y = np.array([op_to_idx.get(t, -1) for t in types])
    valid = y >= 0
    print(f"  N={N} valid={int(valid.sum())} class_counts={dict(Counter(types[valid]))}")

    print("fitting LDA per layer...")
    xy_per_layer = np.zeros((L, N, 2), dtype=np.float32)
    lda_acc = np.zeros(L)
    for l in range(L):
        X = acts[valid, l, :]
        yv = y[valid]
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, yv)
        xy_per_layer[l, valid] = lda.transform(X).astype(np.float32)
        lda_acc[l] = lda.score(X, yv)
    print(f"  per-layer LDA accuracy: {[f'{a:.2f}' for a in lda_acc]}")

    print(f"writing {OUT_PDF}")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for coloring in ["plain", "faithful", "problem_type", "magnitude", "log_answer", "correct"]:
            print(f"  coloring: {coloring}")
            for l in range(L):
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.suptitle(f"Layer {l}  —  LDA(operator) of CODI-GPT-2 ':' residual  "
                             f"—  coloring: {coloring}  (in-sample acc={lda_acc[l]*100:.1f}%)",
                             fontsize=11, fontweight="bold")
                xy = xy_per_layer[l]
                legend_proxies = []
                if coloring == "plain":
                    ax.scatter(xy[valid, 0], xy[valid, 1], s=6, c="#3357aa", alpha=0.5, linewidths=0)
                elif coloring == "faithful":
                    for cls in ["teacher_incorrect", "faithful", "unfaithful"]:
                        mask = (faithful == cls) & valid
                        ax.scatter(xy[mask, 0], xy[mask, 1], s=14 if cls == "unfaithful" else 6,
                                   c=FAITHFUL_COLORS[cls],
                                   alpha=0.9 if cls == "unfaithful" else 0.4, linewidths=0)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=FAITHFUL_COLORS[cls],
                                                     label=f"{cls} ({int(mask.sum())})"))
                elif coloring == "problem_type":
                    for cls, color in PROBLEM_TYPE_COLORS.items():
                        mask = (types == cls) & valid
                        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, c=color, alpha=0.6, linewidths=0)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=color, label=f"{cls} ({int(mask.sum())})"))
                elif coloring == "magnitude":
                    for cls in MAG_BUCKETS:
                        mask = (magnitude == cls) & valid
                        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, c=MAG_COLORS[cls],
                                   alpha=0.6, linewidths=0)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=MAG_COLORS[cls],
                                                     label=f"{cls} ({int(mask.sum())})"))
                elif coloring == "log_answer":
                    sc = ax.scatter(xy[valid, 0], xy[valid, 1], s=6,
                                    c=log_answer[valid], cmap="viridis", alpha=0.6, linewidths=0)
                    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
                    cb.set_label("log10(answer+1)", fontsize=8)
                elif coloring == "correct":
                    for val, color, lbl in [(False, "#d62728", "wrong"),
                                            (True, "#2ca02c", "right")]:
                        mask = (correct == val) & valid
                        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, c=color,
                                   alpha=0.55, linewidths=0)
                        legend_proxies.append(Line2D([0], [0], marker="o", linestyle="",
                                                     color=color,
                                                     label=f"{lbl} ({int(mask.sum())})"))
                ax.set_xlabel("LD1", fontsize=9); ax.set_ylabel("LD2", fontsize=9)
                ax.grid(alpha=0.3)
                if legend_proxies:
                    ax.legend(handles=legend_proxies, fontsize=7, loc="upper right")
                fig.tight_layout(rect=(0, 0, 1, 0.94))
                pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
