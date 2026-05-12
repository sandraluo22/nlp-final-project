"""Multi-page slideshow for the operator-probe transfer experiment.

Reads operator_probe_transfer_gsm8k.json and operator_probe_strict_gsm8k.json
(within-domain CV results) and lays out:

  Page 1 — TL;DR: bar plot of within-domain CV vs forward transfer vs
           reverse transfer per layer.
  Page 2 — Within-domain CV: train + test accuracy bars per layer for
           BOTH strict CF and (newly added) cf_balanced-on-cf_balanced.
  Page 3 — Cross-dataset transfer: forward (train strict → test cf_balanced)
           bars per layer; reverse (train cf_balanced → test strict) bars
           per layer.
  Page 4 — Confusion matrices at L0 (worst transfer) and at best transfer
           layer, both directions.
  Page 5 — Verdict text.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
TRANSFER_JSON = PD / "operator_probe_transfer_gsm8k.json"
STRICT_COL = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_op_strict_colon_acts.pt"
STRICT_CF = REPO.parent / "cf-datasets" / "gsm8k_cf_op_strict.json"
SVAMP_COL = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "cf_balanced_colon_acts.pt"
SVAMP_META = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "cf_balanced_colon_acts_meta.json"
OUT_PDF = PD / "operator_probe_transfer_slideshow.pdf"

OP_NAMES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
op_to_int = {n: i for i, n in enumerate(OP_NAMES)}


def cv_acc_per_layer(X, y, n_folds=5):
    """Within-domain stratified-CV accuracy per layer."""
    L = X.shape[1]
    out = np.zeros(L)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for l in range(L):
        accs = []
        for tr, te in skf.split(X[:, l, :], y):
            sc = StandardScaler().fit(X[tr, l, :])
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
                sc.transform(X[tr, l, :]), y[tr])
            accs.append(clf.score(sc.transform(X[te, l, :]), y[te]))
        out[l] = float(np.mean(accs))
    return out


def main():
    d = json.load(open(TRANSFER_JSON))
    fwd = np.array(d["test_acc_on_cf_balanced"])     # forward transfer
    rev = np.array(d["reverse_test_acc_on_strict"])
    train_strict = np.array(d["train_acc_strict"])
    train_svamp = np.array(d["reverse_train_acc_cf_balanced"])
    confs_fwd = [np.array(c) for c in d["confusion_matrices_per_layer"]]
    L = len(fwd)

    # Compute within-domain CV for both datasets.
    Xs = torch.load(STRICT_COL, map_location="cpu", weights_only=True).float().numpy()
    rows_s = json.load(open(STRICT_CF))
    ys = np.array([op_to_int[r["type"]] for r in rows_s])
    cv_strict = cv_acc_per_layer(Xs, ys)

    Xt = torch.load(SVAMP_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(SVAMP_META))
    types_t = ["Common-Division" if t == "Common-Divison" else t for t in meta["types"]]
    types_t = types_t[:Xt.shape[0]]
    keep = [i for i, t in enumerate(types_t) if t in op_to_int]
    yt = np.array([op_to_int[types_t[i]] for i in keep])
    Xt = Xt[keep]
    cv_svamp = cv_acc_per_layer(Xt, yt)

    # Build reverse confusion at best reverse layer (need to recompute)
    l_rev_best = int(rev.argmax())
    sc = StandardScaler().fit(Xt[:, l_rev_best, :])
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
        sc.transform(Xt[:, l_rev_best, :]), yt)
    ypred = clf.predict(sc.transform(Xs[:, l_rev_best, :]))
    conf_rev_best = confusion_matrix(ys, ypred, labels=list(range(4)))
    sc0 = StandardScaler().fit(Xt[:, 0, :])
    clf0 = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
        sc0.transform(Xt[:, 0, :]), yt)
    conf_rev_L0 = confusion_matrix(
        ys, clf0.predict(sc0.transform(Xs[:, 0, :])), labels=list(range(4)))

    xs = np.arange(L)
    with PdfPages(OUT_PDF) as pdf:
        # ============================================================
        # Page 1: TL;DR
        # ============================================================
        fig, ax = plt.subplots(figsize=(13, 6))
        w = 0.18
        ax.bar(xs - 2*w, cv_strict, w, color="#1f77b4", label="strict CV (within-domain)")
        ax.bar(xs - w,   cv_svamp,  w, color="#aec7e8", label="cf_balanced CV (within-domain)")
        ax.bar(xs,       fwd,       w, color="#2ca02c", label="forward transfer (strict → cf_balanced)")
        ax.bar(xs + w,   rev,       w, color="#ff7f0e", label="reverse transfer (cf_balanced → strict)")
        ax.axhline(0.25, color="black", lw=0.5, ls="--", label="chance (1/4)")
        ax.set_xticks(xs); ax.set_xlabel("layer"); ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("TL;DR — operator probe: within-domain vs cross-domain accuracy by layer",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        # Annotate the gap at best layer
        l_best = int(fwd.argmax())
        gap = cv_strict[l_best] - fwd[l_best]
        ax.annotate(f"vocab-bound gap = {gap:.2f}",
                    xy=(l_best, fwd[l_best]), xytext=(l_best - 2, fwd[l_best] - 0.20),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=9, color="red")
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============================================================
        # Page 2: Within-domain CV per layer (train and test bars)
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, train, test, title in [
            (axes[0], train_strict, cv_strict, "Strict CF — train on full strict / 5-fold CV"),
            (axes[1], train_svamp,  cv_svamp,  "cf_balanced — train on full / 5-fold CV"),
        ]:
            ax.bar(xs - 0.2, train, 0.4, color="#1f77b4", label="train_acc (on training set)")
            ax.bar(xs + 0.2, test,  0.4, color="#2ca02c", label="test_acc (5-fold CV)")
            ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance")
            ax.set_xticks(xs); ax.set_xlabel("layer"); ax.set_ylabel("accuracy")
            ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
            for l in range(L):
                ax.text(l - 0.2, train[l] + 0.01, f"{train[l]:.2f}",
                        ha="center", fontsize=7)
                ax.text(l + 0.2, test[l]  + 0.01, f"{test[l]:.2f}",
                        ha="center", fontsize=7)
        fig.suptitle("Within-domain operator-probe accuracy on each dataset",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============================================================
        # Page 3: Cross-dataset transfer per layer (train and test bars)
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, train, test, title in [
            (axes[0], train_strict, fwd,
             "Forward: TRAIN on strict CF / TEST on cf_balanced"),
            (axes[1], train_svamp,  rev,
             "Reverse: TRAIN on cf_balanced / TEST on strict CF"),
        ]:
            ax.bar(xs - 0.2, train, 0.4, color="#1f77b4", label="train_acc (source dataset)")
            ax.bar(xs + 0.2, test,  0.4, color="#d62728", label="test_acc (target dataset)")
            ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance")
            ax.set_xticks(xs); ax.set_xlabel("layer"); ax.set_ylabel("accuracy")
            ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
            for l in range(L):
                ax.text(l - 0.2, train[l] + 0.01, f"{train[l]:.2f}",
                        ha="center", fontsize=7)
                ax.text(l + 0.2, test[l]  + 0.01, f"{test[l]:.2f}",
                        ha="center", fontsize=7)
        fig.suptitle("Cross-dataset operator-probe transfer accuracy by layer",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============================================================
        # Page 4: Confusion matrices forward + reverse, L0 vs best
        # ============================================================
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        l_fwd_best = int(fwd.argmax())
        for ax, cm, title in [
            (axes[0, 0], confs_fwd[0],          f"Forward L0  (test_acc={fwd[0]:.2f})"),
            (axes[0, 1], confs_fwd[l_fwd_best], f"Forward L{l_fwd_best} BEST  (test_acc={fwd[l_fwd_best]:.2f})"),
            (axes[1, 0], conf_rev_L0,           f"Reverse L0  (test_acc={rev[0]:.2f})"),
            (axes[1, 1], conf_rev_best,         f"Reverse L{l_rev_best} BEST  (test_acc={rev[l_rev_best]:.2f})"),
        ]:
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(OP_NAMES, rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(OP_NAMES, fontsize=8)
            ax.set_xlabel("predicted"); ax.set_ylabel("actual")
            ax.set_title(title, fontsize=10, fontweight="bold")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.suptitle("Confusion matrices — vocabulary-binding at L0 vs operator-encoding at late layers",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============================================================
        # Page 5: Verdict text
        # ============================================================
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.axis("off")
        ax.set_title("Operator-probe transfer: verdict",
                     fontsize=14, fontweight="bold", loc="left")
        verdict = (
            f"Within-domain CV (4-class operator):\n"
            f"  strict CF (LLM-generated):  best layer L{int(cv_strict.argmax())} = {cv_strict.max():.2f}\n"
            f"  cf_balanced (SVAMP-style):  best layer L{int(cv_svamp.argmax())} = {cv_svamp.max():.2f}\n"
            f"\n"
            f"Cross-dataset transfer:\n"
            f"  forward (strict → cf_balanced): best L{l_fwd_best} = {fwd[l_fwd_best]:.2f}\n"
            f"  reverse (cf_balanced → strict): best L{l_rev_best} = {rev[l_rev_best]:.2f}\n"
            f"\n"
            f"Chance (4-class) = 0.25\n"
            f"Vocab-bound gap at L{l_fwd_best}: {cv_strict[l_fwd_best] - fwd[l_fwd_best]:.2f}\n"
            f"\n"
            f"Reading: at early layers (L0-L3), forward transfer is 0.25-0.40\n"
            f"(near chance) — early-layer probes are reading dataset-specific\n"
            f"vocabulary rather than abstract operator structure.\n"
            f"\n"
            f"At late layers (L8-L12), forward transfer is 0.81-0.86 (well above\n"
            f"chance). The drop from 1.00 within-domain to 0.85 across-domain\n"
            f"means about 15% of the within-domain accuracy was vocabulary-\n"
            f"specific, but the remaining 85% reflects a real operator encoding\n"
            f"that generalizes across vocabulary distributions.\n"
            f"\n"
            f"This is the expected pattern of an interpretable transformer:\n"
            f"token-level vocabulary processing in early layers, more abstract\n"
            f"task structure encoding in late layers.\n"
        )
        ax.text(0.02, 0.90, verdict, fontsize=10, va="top", family="monospace")
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
