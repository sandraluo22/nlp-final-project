"""Cross-dataset operator-probe transfer test.

Train operator probe on the LLM-generated GSM8K-style strict CF dataset
(`gsm8k_cf_op_strict`, vocabulary biased by GPT-5-mini's word choices).
Test on the SVAMP-style cf_balanced dataset (human-template-derived, very
different vocabulary). If transfer accuracy stays high, the operator
signal generalizes beyond the training-set vocabulary. If accuracy drops
to chance, the probe was reading vocabulary, not operator structure.

Uses the COLON residual (we don't have cf_balanced latent acts captured).
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
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
STRICT_COL = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_op_strict_colon_acts.pt"
STRICT_CF = REPO.parent / "cf-datasets" / "gsm8k_cf_op_strict.json"
SVAMP_COL = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "cf_balanced_colon_acts.pt"
SVAMP_META = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "cf_balanced_colon_acts_meta.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "operator_probe_transfer_gsm8k.json"
OUT_PDF = PD / "operator_probe_transfer_gsm8k.pdf"

OP_NAMES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
op_to_int = {n: i for i, n in enumerate(OP_NAMES)}


def main():
    # Load training data: strict CF colon acts + 4-class labels
    Xs = torch.load(STRICT_COL, map_location="cpu", weights_only=True).float().numpy()
    rows_s = json.load(open(STRICT_CF))
    Ns = Xs.shape[0]
    assert Ns == len(rows_s)
    ys = np.array([op_to_int[r["type"]] for r in rows_s])
    Lc = Xs.shape[1]
    print(f"strict CF: N={Ns}, distribution: {dict(zip(*np.unique(ys, return_counts=True)))}")

    # Load test data: cf_balanced colon acts + types
    Xt = torch.load(SVAMP_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(SVAMP_META))
    types_t = meta["types"]
    Nt = Xt.shape[0]
    types_t = types_t[:Nt]
    # Map Common-Divison typo if present
    types_t = ["Common-Division" if t == "Common-Divison" else t for t in types_t]
    keep_t = [i for i, t in enumerate(types_t) if t in op_to_int]
    yt = np.array([op_to_int[types_t[i]] for i in keep_t])
    Xt = Xt[keep_t]
    print(f"cf_balanced: N_kept={len(yt)}, distribution: {dict(zip(*np.unique(yt, return_counts=True)))}")

    # Per-layer transfer: train on strict, test on cf_balanced.
    train_acc = np.zeros(Lc)
    test_acc = np.zeros(Lc)
    confs = []
    for l in range(Lc):
        sc = StandardScaler().fit(Xs[:, l, :])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
            sc.transform(Xs[:, l, :]), ys)
        train_acc[l] = clf.score(sc.transform(Xs[:, l, :]), ys)
        test_acc[l] = clf.score(sc.transform(Xt[:, l, :]), yt)
        ypred = clf.predict(sc.transform(Xt[:, l, :]))
        confs.append(confusion_matrix(yt, ypred, labels=list(range(4))).tolist())

    print("\nLayer | train_acc | test_acc (cf_balanced)")
    for l in range(Lc):
        print(f"  L{l:2d}   {train_acc[l]:.3f}     {test_acc[l]:.3f}")
    print(f"\nbest transfer layer: L{int(test_acc.argmax())}  acc={test_acc.max():.3f}")
    print(f"chance: 0.25 (4-class)")

    # Also reverse direction: train on cf_balanced, test on strict
    rev_train = np.zeros(Lc); rev_test = np.zeros(Lc)
    for l in range(Lc):
        sc = StandardScaler().fit(Xt[:, l, :])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
            sc.transform(Xt[:, l, :]), yt)
        rev_train[l] = clf.score(sc.transform(Xt[:, l, :]), yt)
        rev_test[l] = clf.score(sc.transform(Xs[:, l, :]), ys)
    print("\nReverse: train on cf_balanced, test on strict CF")
    for l in range(Lc):
        print(f"  L{l:2d}   train={rev_train[l]:.3f}  test={rev_test[l]:.3f}")
    print(f"best reverse layer: L{int(rev_test.argmax())}  acc={rev_test.max():.3f}")

    OUT_JSON.write_text(json.dumps({
        "Ns": int(Ns), "Nt": int(len(yt)),
        "train_acc_strict": train_acc.tolist(),
        "test_acc_on_cf_balanced": test_acc.tolist(),
        "best_layer": int(test_acc.argmax()),
        "best_transfer_acc": float(test_acc.max()),
        "reverse_train_acc_cf_balanced": rev_train.tolist(),
        "reverse_test_acc_on_strict": rev_test.tolist(),
        "best_reverse_layer": int(rev_test.argmax()),
        "best_reverse_acc": float(rev_test.max()),
        "confusion_matrices_per_layer": confs,
    }, indent=2))
    print(f"saved {OUT_JSON}")

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # Panel 1: forward and reverse transfer
        ax = axes[0]
        ls = np.arange(Lc); w = 0.4
        ax.bar(ls - w/2, test_acc, w, color="#1f77b4", label="train strict → test cf_balanced")
        ax.bar(ls + w/2, rev_test, w, color="#ff7f0e", label="train cf_balanced → test strict")
        ax.axhline(0.25, color="black", lw=0.5, ls="--", label="4-class chance")
        ax.set_xticks(ls); ax.set_xlabel("layer"); ax.set_ylabel("transfer accuracy")
        ax.set_title("Cross-dataset operator-probe transfer (colon residual)",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        # Panel 2: confusion at best forward layer
        ax = axes[1]
        l_best = int(test_acc.argmax())
        cm = np.array(confs[l_best])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(OP_NAMES, rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(OP_NAMES, fontsize=7)
        ax.set_xlabel("predicted (probe trained on strict CF)")
        ax.set_ylabel("actual (cf_balanced)")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=10, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"Confusion at best layer L{l_best}  acc={test_acc[l_best]:.2f}",
                     fontsize=10, fontweight="bold")
        fig.suptitle("Operator probe — does it transfer across vocabulary domains?",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
