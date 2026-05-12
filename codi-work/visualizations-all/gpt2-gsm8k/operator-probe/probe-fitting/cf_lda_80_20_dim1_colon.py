"""Per-LD-axis 1D analysis at the ':' residual.

For each layer, fit LDA(n_components=3) supervised on 4-class problem-type on
80% of cf_balanced ':' residuals. Then *project onto each axis individually*
(LD1, LD2, LD3) and treat that as a 1D feature. For each axis and each grouping
scheme, score the 1D classifier on held-out CF and on SVAMP.

Groupings:
  - indiv : Add / Sub / Mul / Div
  - AS|M|D: {Add+Sub} vs Mul vs Div
  - AS|MD : {Add+Sub} vs {Mul+Div}
  - AM|SD : {Add+Mul} vs {Sub+Div}
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[4]
CF_ACTS = REPO / "experiments" / "computation_probes" / "cf_balanced_colon_acts.pt"
CF_META = REPO / "experiments" / "computation_probes" / "cf_balanced_colon_acts_meta.json"
ORIG_ACTS = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
ORIG_META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2" / "cf_lda_80_20_dim1_colon.pdf"
OUT_STATS = REPO / "visualizations-all" / "gpt2" / "cf_lda_80_20_dim1_colon_stats.json"

OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
PROBLEM_TYPE_COLORS = {"Subtraction": "#1f77b4", "Addition": "#ff7f0e",
                       "Common-Division": "#2ca02c", "Multiplication": "#d62728"}
SEED = 13

GROUPINGS = {
    "indiv":  np.array([0, 1, 2, 3]),
    "AS|M|D": np.array([0, 0, 1, 2]),   # Add=Sub=0, Mul=1, Div=2
    "AS|MD":  np.array([0, 0, 1, 1]),
    "AM|SD":  np.array([0, 1, 0, 1]),
}


def load(acts_p, meta_p):
    acts = torch.load(acts_p, map_location="cpu", weights_only=True).float().numpy()
    types = np.array(json.load(open(meta_p))["types"])
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y = np.array([op_to_idx.get(t, -1) for t in types])
    valid = y >= 0
    return acts[valid], y[valid]


def acc_1d(X1d_tr, y_tr, X1d_te, y_te):
    # 1D LDA classifier (effectively a thresholded line per class)
    clf = LinearDiscriminantAnalysis().fit(X1d_tr.reshape(-1, 1), y_tr)
    return float(clf.score(X1d_te.reshape(-1, 1), y_te))


def main():
    cf_acts, cf_y = load(CF_ACTS, CF_META)
    orig_acts, orig_y = load(ORIG_ACTS, ORIG_META)
    L = cf_acts.shape[1]
    idx_tr, idx_te = train_test_split(np.arange(cf_acts.shape[0]), test_size=0.2,
                                       random_state=SEED, stratify=cf_y)

    # accs[axis_idx, grouping_name, dataset] per layer
    cf_acc = {ax: {g: np.zeros(L) for g in GROUPINGS} for ax in range(3)}
    orig_accs = {ax: {g: np.zeros(L) for g in GROUPINGS} for ax in range(3)}

    print(f"writing {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        # Per-(axis, layer) histograms + accs
        for ax_i in range(3):
            for l in range(L):
                Xtr = cf_acts[idx_tr, l, :]; ytr = cf_y[idx_tr]
                lda = LinearDiscriminantAnalysis(n_components=3).fit(Xtr, ytr)
                T_tr = lda.transform(Xtr)[:, ax_i]
                T_cfte = lda.transform(cf_acts[idx_te, l, :])[:, ax_i]
                T_orig = lda.transform(orig_acts[:, l, :])[:, ax_i]
                for g_name, g_map in GROUPINGS.items():
                    g_tr = g_map[ytr]
                    g_cfte = g_map[cf_y[idx_te]]
                    g_orig = g_map[orig_y]
                    cf_acc[ax_i][g_name][l] = acc_1d(T_tr, g_tr, T_cfte, g_cfte)
                    orig_accs[ax_i][g_name][l] = acc_1d(T_tr, g_tr, T_orig, g_orig)

                # Render one slide per (axis, layer): histogram per class
                fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
                for which, (T_test, y_test, title) in enumerate([
                    (T_cfte, cf_y[idx_te], "cf_balanced (held-out)"),
                    (T_orig, orig_y, "SVAMP (transfer)"),
                ]):
                    ax = axes[which]
                    for c, name in enumerate(OPS):
                        mask = y_test == c
                        if not mask.any(): continue
                        ax.hist(T_test[mask], bins=30, alpha=0.55,
                                color=PROBLEM_TYPE_COLORS[name], label=name, density=True)
                    ax.set_xlabel(f"LD{ax_i+1}"); ax.set_ylabel("density")
                    ax.set_title(f"{title}: indiv={cf_acc[ax_i]['indiv'][l]*100:.0f}% "
                                 f"AS|M|D={cf_acc[ax_i]['AS|M|D'][l]*100:.0f}% "
                                 f"AS|MD={cf_acc[ax_i]['AS|MD'][l]*100:.0f}% "
                                 f"AM|SD={cf_acc[ax_i]['AM|SD'][l]*100:.0f}%", fontsize=8)
                    ax.legend(fontsize=7)
                fig.suptitle(f"layer {l} — LD{ax_i+1} 1D projection (':' residual)",
                             fontsize=11, fontweight="bold")
                fig.tight_layout(rect=(0, 0, 1, 0.94))
                pdf.savefig(fig, dpi=140); plt.close(fig)

        # Summary slide: accuracy per layer for each grouping × axis × dataset
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for di, (which, accs, title) in enumerate([
            (0, cf_acc, "cf_balanced (held-out)"),
            (1, orig_accs, "SVAMP (transfer)"),
        ]):
            for gi, g in enumerate(GROUPINGS):
                ax = axes[di, gi]
                for ax_i in range(3):
                    ax.plot(np.arange(L), accs[ax_i][g] * 100,
                            "o-", label=f"LD{ax_i+1}")
                ax.set_xlabel("layer")
                ax.set_ylabel("acc (%)")
                ax.set_title(f"{title}: {g}", fontsize=9)
                ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(20, 100)
        fig.suptitle("1D LDA accuracy per layer (':' residual)", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

    stats = {"L": int(L),
             "cf_acc": {ax: {g: v.tolist() for g, v in d.items()} for ax, d in cf_acc.items()},
             "orig_acc": {ax: {g: v.tolist() for g, v in d.items()} for ax, d in orig_accs.items()}}
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"done -> {OUT_PDF}")


if __name__ == "__main__":
    main()
