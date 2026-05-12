"""Union-training operator transfer probe.

Train on 70 % of cf_balanced PLUS 70 % of gsm8k_cf_op_strict combined.
Evaluate on:
  - 30 % held-out cf_balanced (in-domain SVAMP-style)
  - 30 % held-out gsm8k_cf_op_strict (in-domain LLM-generated)
  - Natural GSM8K test (single-op subset, N=174) — fully out-of-distribution
  - Natural GSM8K test (full 1318) — multi-op problems, scored by FIRST
    operator in the chain (noisier label).

This tests whether mixing two operator-labeled datasets gives a probe
that captures a more universal operator representation. If accuracy on
the natural GSM8K (totally different distribution) jumps relative to
either dataset alone, training on both helped the probe abstract away
dataset-specific vocabulary.

Uses the colon-position residual (cf_balanced has no latent acts).
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
STRICT_COL = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_op_strict_colon_acts.pt"
STRICT_CF = REPO.parent / "cf-datasets" / "gsm8k_cf_op_strict.json"
SVAMP_COL = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "cf_balanced_colon_acts.pt"
SVAMP_META = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "cf_balanced_colon_acts_meta.json"
NAT_COL = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
NAT_META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"

PD = Path(__file__).resolve().parent
OUT_JSON = PD / "operator_probe_union_transfer.json"
OUT_PDF = PD / "operator_probe_union_transfer.pdf"

OP_NAMES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
op_to_int = {n: i for i, n in enumerate(OP_NAMES)}
SEED = 0


def load_strict():
    X = torch.load(STRICT_COL, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(STRICT_CF))
    y = np.array([op_to_int[r["type"]] for r in rows])
    return X, y


def load_svamp_cf_balanced():
    X = torch.load(SVAMP_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(SVAMP_META))
    types = ["Common-Division" if t == "Common-Divison" else t for t in meta["types"]]
    types = types[:X.shape[0]]
    keep = [i for i, t in enumerate(types) if t in op_to_int]
    return X[keep], np.array([op_to_int[types[i]] for i in keep])


def load_natural_gsm8k():
    """Returns (X_single_op, y_single_op, X_full, y_full_first_op).
    Multi-op problems' y_full uses first-operator-in-chain (noisier label)."""
    X = torch.load(NAT_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(NAT_META))
    # Use augmented Type from meta
    types = meta.get("types", [])[:X.shape[0]]
    ops_used = meta.get("operators_used", [])[:X.shape[0]]
    # Single-op subset
    keep_single = [i for i, ou in enumerate(ops_used) if isinstance(ou, list) and len(ou) == 1]
    X_single = X[keep_single]
    y_single = np.array([op_to_int[ops_used[i][0]] for i in keep_single])
    # Full set, first-op label (drop 'Multi' as that means multiple ops)
    keep_full = [i for i, t in enumerate(types) if t in op_to_int]
    X_full = X[keep_full]
    y_full_firstop = np.array([op_to_int[types[i]] for i in keep_full])
    # First-op-from-chain for Multi cases too (use first element of operators_used)
    keep_first_in_chain = [i for i, ou in enumerate(ops_used)
                            if isinstance(ou, list) and len(ou) >= 1 and ou[0] in op_to_int]
    X_first = X[keep_first_in_chain]
    y_first = np.array([op_to_int[ops_used[i][0]] for i in keep_first_in_chain])
    return X_single, y_single, X_first, y_first


def main():
    Xs, ys = load_strict()
    Xb, yb = load_svamp_cf_balanced()
    X_nat_single, y_nat_single, X_nat_first, y_nat_first = load_natural_gsm8k()
    Lc = Xs.shape[1]
    print(f"strict CF:       N={Xs.shape[0]}, label dist={dict(zip(*np.unique(ys, return_counts=True)))}")
    print(f"cf_balanced:     N={Xb.shape[0]}, label dist={dict(zip(*np.unique(yb, return_counts=True)))}")
    print(f"natural single-op: N={X_nat_single.shape[0]}, dist={dict(zip(*np.unique(y_nat_single, return_counts=True)))}")
    print(f"natural first-op:  N={X_nat_first.shape[0]}, dist={dict(zip(*np.unique(y_nat_first, return_counts=True)))}")

    # 70/30 stratified split for each
    rng = np.random.default_rng(SEED)
    idx_s_tr, idx_s_te = train_test_split(np.arange(Xs.shape[0]), test_size=0.30,
                                            random_state=SEED, stratify=ys)
    idx_b_tr, idx_b_te = train_test_split(np.arange(Xb.shape[0]), test_size=0.30,
                                            random_state=SEED, stratify=yb)
    print(f"\nSplit sizes:")
    print(f"  strict   train={len(idx_s_tr)}  test={len(idx_s_te)}")
    print(f"  balanced train={len(idx_b_tr)}  test={len(idx_b_te)}")

    # Per-layer train union, evaluate on all targets
    test_strict = np.zeros(Lc)
    test_balanced = np.zeros(Lc)
    test_nat_single = np.zeros(Lc)
    test_nat_first = np.zeros(Lc)
    confs = []
    for l in range(Lc):
        Xtr = np.vstack([Xs[idx_s_tr, l, :], Xb[idx_b_tr, l, :]])
        ytr = np.concatenate([ys[idx_s_tr], yb[idx_b_tr]])
        sc = StandardScaler().fit(Xtr)
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        test_strict[l]   = clf.score(sc.transform(Xs[idx_s_te, l, :]), ys[idx_s_te])
        test_balanced[l] = clf.score(sc.transform(Xb[idx_b_te, l, :]), yb[idx_b_te])
        test_nat_single[l] = clf.score(sc.transform(X_nat_single[:, l, :]), y_nat_single)
        test_nat_first[l]  = clf.score(sc.transform(X_nat_first[:, l, :]), y_nat_first)
        if l == 0 or l == Lc - 1 or l == Lc // 2:
            ypred = clf.predict(sc.transform(X_nat_first[:, l, :]))
            confs.append((l, confusion_matrix(y_nat_first, ypred,
                                               labels=list(range(4))).tolist()))

    print("\nLayer | held-out strict | held-out balanced | natural-single-op | natural-first-op")
    for l in range(Lc):
        print(f"  L{l:2d}   {test_strict[l]:.3f}            "
              f"{test_balanced[l]:.3f}             {test_nat_single[l]:.3f}              "
              f"{test_nat_first[l]:.3f}")
    print(f"\nBest layers:")
    print(f"  strict held-out:   L{int(test_strict.argmax())} = {test_strict.max():.3f}")
    print(f"  balanced held-out: L{int(test_balanced.argmax())} = {test_balanced.max():.3f}")
    print(f"  natural single-op: L{int(test_nat_single.argmax())} = {test_nat_single.max():.3f}")
    print(f"  natural first-op:  L{int(test_nat_first.argmax())} = {test_nat_first.max():.3f}")
    print(f"chance (4-class):   0.25")

    OUT_JSON.write_text(json.dumps({
        "Ns_train": int(len(idx_s_tr)), "Ns_test": int(len(idx_s_te)),
        "Nb_train": int(len(idx_b_tr)), "Nb_test": int(len(idx_b_te)),
        "N_nat_single": int(X_nat_single.shape[0]),
        "N_nat_first":  int(X_nat_first.shape[0]),
        "test_acc_strict_heldout":   test_strict.tolist(),
        "test_acc_balanced_heldout": test_balanced.tolist(),
        "test_acc_natural_singleop": test_nat_single.tolist(),
        "test_acc_natural_firstop":  test_nat_first.tolist(),
        "confs_natural_firstop": [{"layer": l, "matrix": cm} for l, cm in confs],
    }, indent=2))
    print(f"saved {OUT_JSON}")

    with PdfPages(OUT_PDF) as pdf:
        # Page 1: per-layer accuracy bars
        fig, ax = plt.subplots(figsize=(14, 6))
        xs = np.arange(Lc); w = 0.22
        ax.bar(xs - 1.5*w, test_strict,    w, color="#1f77b4", label="held-out strict (30%)")
        ax.bar(xs - 0.5*w, test_balanced,  w, color="#aec7e8", label="held-out cf_balanced (30%)")
        ax.bar(xs + 0.5*w, test_nat_single, w, color="#2ca02c", label="natural GSM8K single-op")
        ax.bar(xs + 1.5*w, test_nat_first,  w, color="#d62728", label="natural GSM8K first-op")
        ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance (4-class)")
        ax.set_xticks(xs); ax.set_xlabel("colon layer"); ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("Union-trained operator probe (70% strict + 70% cf_balanced) — "
                     "per-layer test accuracy",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
        for l in range(Lc):
            for xi, v in [(xs[l] - 1.5*w, test_strict[l]),
                           (xs[l] - 0.5*w, test_balanced[l]),
                           (xs[l] + 0.5*w, test_nat_single[l]),
                           (xs[l] + 1.5*w, test_nat_first[l])]:
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 2: confusion matrices on natural-first-op at L0, mid, last
        fig, axes = plt.subplots(1, len(confs), figsize=(5 * len(confs), 5))
        if len(confs) == 1: axes = [axes]
        for ax, (l, cm) in zip(axes, confs):
            cm = np.array(cm)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(OP_NAMES, rotation=30, ha="right", fontsize=7)
            ax.set_yticklabels(OP_NAMES, fontsize=7)
            ax.set_xlabel("predicted"); ax.set_ylabel("actual (first-op-in-chain)")
            ax.set_title(f"L{l}  acc={test_nat_first[l]:.2f}",
                         fontsize=10, fontweight="bold")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=10, color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.suptitle("Union probe → natural GSM8K first-operator confusion at selected layers",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
