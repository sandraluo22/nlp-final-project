"""Operator probe on GSM8K vary_operator CF dataset.

This is the cleanest possible operator test: 80 templates, each with 4
variants sharing the SAME operands (a, b) but using different operators.
So operand magnitude and scenario context are HELD CONSTANT within each
template; only the operator differs.

Tests:
  1. Per (step, layer) 4-class accuracy on the full 320 examples.
  2. Within-template accuracy — for each template's 4 variants, can the
     probe assign each to the correct operator? This is the strongest
     control because it equates magnitude and narrative.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
LAT = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_vary_operator_latent_acts.pt"
COL = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_vary_operator_colon_acts.pt"
CF = REPO.parent / "cf-datasets" / "gsm8k_vary_operator.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "operator_probe_vary_op_gsm8k.json"
OUT_PDF = PD / "operator_probe_vary_op_gsm8k.pdf"

OP_NAMES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
op_to_int = {n: i for i, n in enumerate(OP_NAMES)}
SEED = 0


def cv_acc(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def leave_one_template_out(X, y, groups):
    """For each template_id, train on all other templates and test on it.
    This is the harshest test — the test template's (a, b) is unseen."""
    logo = LeaveOneGroupOut()
    accs = []
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 4: continue
        sc = StandardScaler().fit(X[tr])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def main():
    lat = torch.load(LAT, map_location="cpu", weights_only=True).float().numpy()
    col = torch.load(COL, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(CF))
    y = np.array([op_to_int[r["type"]] for r in rows])
    groups = np.array([r["template_id"] for r in rows])
    print(f"vary_operator: N={len(rows)}, types={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  templates: {len(np.unique(groups))}, 4 variants per template")
    S, L, H = lat.shape[1], lat.shape[2], lat.shape[3]
    Lc = col.shape[1]

    # Per-(step, layer) standard 5-fold CV accuracy
    lat_acc = np.zeros((S, L))
    for s in range(S):
        for l in range(L):
            lat_acc[s, l] = cv_acc(lat[:, s, l, :], y)
        print(f"  step {s+1}: best L{int(lat_acc[s].argmax())} 5fold-acc={lat_acc[s].max():.3f}")
    col_acc = np.zeros(Lc)
    for l in range(Lc):
        col_acc[l] = cv_acc(col[:, l, :], y)
    print(f"  colon best L{int(col_acc.argmax())} 5fold-acc={col_acc.max():.3f}")

    # Leave-one-template-out (LOTO) — strongest control:
    print("\nLeave-one-template-out (harshest control)...")
    s_b, l_b = np.unravel_index(lat_acc.argmax(), lat_acc.shape)
    loto_acc_latent = leave_one_template_out(lat[:, s_b, l_b, :], y, groups)
    print(f"  latent best cell (step={s_b+1}, L={l_b}):  LOTO acc = {loto_acc_latent:.3f}")
    l_bc = int(col_acc.argmax())
    loto_acc_colon = leave_one_template_out(col[:, l_bc, :], y, groups)
    print(f"  colon best L{l_bc}:                       LOTO acc = {loto_acc_colon:.3f}")

    # Loto across every (step, layer)
    loto_lat = np.zeros((S, L))
    for s in range(S):
        for l in range(L):
            loto_lat[s, l] = leave_one_template_out(lat[:, s, l, :], y, groups)
    loto_col = np.zeros(Lc)
    for l in range(Lc):
        loto_col[l] = leave_one_template_out(col[:, l, :], y, groups)

    OUT_JSON.write_text(json.dumps({
        "N": int(len(rows)), "S": int(S), "L": int(L), "Lc": int(Lc),
        "n_templates": int(len(np.unique(groups))),
        "latent_acc_5fold": lat_acc.tolist(),
        "colon_acc_5fold": col_acc.tolist(),
        "latent_acc_loto": loto_lat.tolist(),
        "colon_acc_loto": loto_col.tolist(),
        "best_latent_5fold": float(lat_acc.max()),
        "best_latent_loto": float(loto_lat.max()),
        "best_colon_5fold": float(col_acc.max()),
        "best_colon_loto": float(loto_col.max()),
    }, indent=2))
    print(f"saved {OUT_JSON}")

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        for ax, M, title in [
            (axes[0, 0], lat_acc, "Latent — 5-fold CV"),
            (axes[0, 1], loto_lat, "Latent — Leave-One-Template-Out"),
        ]:
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                            vmin=0.25, vmax=1.0)
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_xticks(range(L)); ax.set_yticks(range(S))
            ax.set_yticklabels([str(s + 1) for s in range(S)])
            ax.set_title(title, fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04)
            for s in range(S):
                for l in range(L):
                    v = M[s, l]
                    if v >= 0.4 or v <= 0.3:
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.6 else "black")
        # Colon bars
        ax = axes[1, 0]
        ax.bar(range(Lc), col_acc, color="#1f77b4")
        ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance")
        ax.set_xticks(range(Lc)); ax.set_xlabel("colon layer"); ax.set_ylabel("5-fold acc")
        ax.set_title(f"Colon 5-fold CV — best L{int(col_acc.argmax())} = {col_acc.max():.2f}",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3); ax.legend()
        ax = axes[1, 1]
        ax.bar(range(Lc), loto_col, color="#d62728")
        ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance")
        ax.set_xticks(range(Lc)); ax.set_xlabel("colon layer"); ax.set_ylabel("LOTO acc")
        ax.set_title(f"Colon LOTO — best L{int(loto_col.argmax())} = {loto_col.max():.2f}",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3); ax.legend()
        fig.suptitle("Operator probe on vary_operator (same a,b within template; 80 templates × 4 variants)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
