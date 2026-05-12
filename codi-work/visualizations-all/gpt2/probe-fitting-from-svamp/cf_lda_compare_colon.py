"""Project cf_balanced ':' residuals onto the LDA(operator) axes that were
FIT on the original SVAMP ':' residuals. If operator clusters survive in CF,
the LDA direction is genuinely operator-coding (not magnitude in disguise).

':' analog of cf_lda_compare.py. 13 slides (one per layer); each slide
overlays SVAMP (small gray markers) with CF (filled colored) on the
operator-LDA axes fit on SVAMP.

Reports per-layer operator-classification accuracy on CF projection using a
classifier fit on SVAMP's projection.
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
CF_ACTS = REPO / "experiments" / "computation_probes" / "cf_balanced_colon_acts.pt"
CF_META = REPO / "experiments" / "computation_probes" / "cf_balanced_colon_acts_meta.json"
ORIG_ACTS = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
ORIG_META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2-gsm8k" / "cf_lda_compare_colon.pdf"

OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
PROBLEM_TYPE_COLORS = {"Subtraction": "#1f77b4", "Addition": "#ff7f0e",
                       "Common-Division": "#2ca02c", "Multiplication": "#d62728"}


def load(acts_p, meta_p):
    acts = torch.load(acts_p, map_location="cpu", weights_only=True).float().numpy()
    types = np.array(json.load(open(meta_p))["types"])
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y = np.array([op_to_idx.get(t, -1) for t in types])
    valid = y >= 0
    return acts[valid], y[valid]


def main():
    cf_acts, cf_y = load(CF_ACTS, CF_META)
    orig_acts, orig_y = load(ORIG_ACTS, ORIG_META)
    L = cf_acts.shape[1]
    print(f"cf={cf_acts.shape}, svamp={orig_acts.shape}")

    transfer_acc = np.zeros(L)
    print(f"writing {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        # Summary slide
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for l in range(L):
            lda = LinearDiscriminantAnalysis(n_components=2).fit(orig_acts[:, l, :], orig_y)
            cf_proj = lda.transform(cf_acts[:, l, :])
            # classifier on orig projection
            clf = LinearDiscriminantAnalysis().fit(lda.transform(orig_acts[:, l, :]), orig_y)
            transfer_acc[l] = clf.score(cf_proj, cf_y)
        ax.plot(np.arange(L), transfer_acc * 100, "o-", color="#1f77b4")
        ax.set_xlabel("layer"); ax.set_ylabel("CF operator-classification acc (%)")
        ax.set_title("LDA fit on SVAMP — projected CF acc (operator direction transfer)",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3); ax.set_ylim(20, 100)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # Per-layer overlay
        for l in range(L):
            lda = LinearDiscriminantAnalysis(n_components=2).fit(orig_acts[:, l, :], orig_y)
            orig_xy = lda.transform(orig_acts[:, l, :])
            cf_xy = lda.transform(cf_acts[:, l, :])
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.suptitle(f"layer {l} — LDA fit on SVAMP, CF overlay  (CF transfer acc={transfer_acc[l]*100:.1f}%)",
                         fontsize=11, fontweight="bold")
            for c, name in enumerate(OPS):
                m_orig = orig_y == c; m_cf = cf_y == c
                if m_orig.any():
                    ax.scatter(orig_xy[m_orig, 0], orig_xy[m_orig, 1], s=10,
                               edgecolors=PROBLEM_TYPE_COLORS[name], facecolors="none",
                               linewidths=0.6, alpha=0.4, label=f"SVAMP {name}")
                if m_cf.any():
                    ax.scatter(cf_xy[m_cf, 0], cf_xy[m_cf, 1], s=20,
                               c=PROBLEM_TYPE_COLORS[name], alpha=0.85,
                               linewidths=0, label=f"CF {name}")
            ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc="upper right")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"done -> {OUT_PDF}")


if __name__ == "__main__":
    main()
