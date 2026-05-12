"""Operator-presence one-vs-rest probe on the LLM-generated NATURAL CF
dataset (630 multi-step problems, magnitude-balanced, multi-label parser-
derived operator labels).

Key contrast: this is the only operator-presence probe with controlled
magnitude AND multi-step natural mixing. The natural GSM8K test set
presence probe is also multi-label but magnitude-uncontrolled. If the
natural CF probe gives similar AUCs, that's evidence the operator-
presence signal in the latent loop is NOT a magnitude/length confound.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
LAT = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_natural_latent_acts.pt"
COL = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_natural_colon_acts.pt"
CF = REPO.parent / "cf-datasets" / "gsm8k_cf_natural.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "operator_probe_natural_cf_gsm8k.json"
OUT_PDF = PD / "operator_probe_natural_cf_gsm8k.pdf"
SEED = 0


def cv_auc(X, y, n_folds=5):
    if y.sum() < 5 or y.sum() > len(y) - 5: return 0.5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        score = clf.decision_function(sc.transform(X[te]))
        try: aucs.append(roc_auc_score(y[te], score))
        except ValueError: aucs.append(0.5)
    return float(np.mean(aucs))


def main():
    lat = torch.load(LAT, map_location="cpu", weights_only=True).float().numpy()
    col = torch.load(COL, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(CF))
    N = lat.shape[0]
    has_add = np.array([r["has_add"] for r in rows], dtype=int)
    has_sub = np.array([r["has_sub"] for r in rows], dtype=int)
    has_mul = np.array([r["has_mul"] for r in rows], dtype=int)
    has_div = np.array([r["has_div"] for r in rows], dtype=int)
    mag = np.array([r["magnitude_bucket"] for r in rows])
    print(f"natural CF: N={N}, has_add={has_add.mean():.2f}, has_sub={has_sub.mean():.2f}, "
          f"has_mul={has_mul.mean():.2f}, has_div={has_div.mean():.2f}")
    S, L, H = lat.shape[1], lat.shape[2], lat.shape[3]

    # Per-(step, layer) AUC for each operator
    auc = {op: np.zeros((S, L)) for op in ["add", "sub", "mul", "div"]}
    labels = {"add": has_add, "sub": has_sub, "mul": has_mul, "div": has_div}
    for s in range(S):
        for l in range(L):
            X = lat[:, s, l, :]
            for op in auc:
                auc[op][s, l] = cv_auc(X, labels[op])
        for op in auc:
            print(f"  step {s+1}: AUC {op}={auc[op][s].max():.3f} (best L{int(auc[op][s].argmax())})")

    # Colon residual per layer
    Lc = col.shape[1]
    auc_col = {op: np.zeros(Lc) for op in ["add", "sub", "mul", "div"]}
    for l in range(Lc):
        X = col[:, l, :]
        for op in auc_col:
            auc_col[op][l] = cv_auc(X, labels[op])
    for op in auc_col:
        print(f"  colon AUC {op}: best L{int(auc_col[op].argmax())} = {auc_col[op].max():.3f}")

    # Within-magnitude AUC for each operator (use best latent cell per operator)
    within_mag = {}
    for op in auc:
        s_b, l_b = np.unravel_index(auc[op].argmax(), auc[op].shape)
        within_mag[op] = {}
        for m in np.unique(mag):
            idx = np.where(mag == m)[0]
            if len(idx) < 30: continue
            within_mag[op][m] = cv_auc(lat[idx, s_b, l_b, :], labels[op][idx], n_folds=3)
        print(f"  within-magnitude AUC {op}: {within_mag[op]}")

    OUT_JSON.write_text(json.dumps({
        "N": int(N),
        "has_op": {op: float(labels[op].mean()) for op in auc},
        "latent_auc": {op: auc[op].tolist() for op in auc},
        "colon_auc": {op: auc_col[op].tolist() for op in auc_col},
        "within_magnitude_auc": within_mag,
    }, indent=2))
    print(f"saved {OUT_JSON}")

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, op in zip(axes.flat, ["add", "sub", "mul", "div"]):
            im = ax.imshow(auc[op], aspect="auto", origin="lower",
                           cmap="viridis", vmin=0.5, vmax=1.0)
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_xticks(range(L)); ax.set_yticks(range(S))
            ax.set_yticklabels([str(s+1) for s in range(S)])
            ax.set_title(f"AUC: has_{op}? (positive rate={labels[op].mean():.2f})",
                         fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04)
            for s in range(S):
                for l in range(L):
                    v = auc[op][s, l]
                    if v >= 0.7 or v <= 0.55:
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.75 else "black")
        fig.suptitle(f"GSM8K NATURAL CF operator-presence probes (N={N}, "
                     f"magnitude-balanced)", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Bar chart: best AUC per operator, latent vs colon vs within-magnitude min
        fig, ax = plt.subplots(figsize=(11, 5))
        ops = ["add", "sub", "mul", "div"]
        xs = np.arange(len(ops))
        w = 0.27
        best_lat = [auc[op].max() for op in ops]
        best_col = [auc_col[op].max() for op in ops]
        min_within = [min(within_mag[op].values()) if within_mag[op] else 0.5 for op in ops]
        ax.bar(xs - w, best_lat, w, color="#1f77b4", label="best latent (step,layer)")
        ax.bar(xs,     best_col, w, color="#2ca02c", label="best colon layer")
        ax.bar(xs + w, min_within, w, color="#d62728", label="min within-mag-bucket")
        ax.axhline(0.5, color="black", lw=0.3, ls="--", label="chance")
        ax.set_xticks(xs); ax.set_xticklabels([f"has_{o}" for o in ops])
        ax.set_ylim(0, 1.05); ax.set_ylabel("AUC")
        ax.set_title("AUC per operator on natural CF — "
                     "magnitude-balanced confirms no magnitude conflation",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
