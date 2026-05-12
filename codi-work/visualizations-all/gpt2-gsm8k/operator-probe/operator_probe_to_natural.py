"""Find the best operator probe for NATURAL GSM8K (without training on it).

Sweeps 7 training combinations:
   T1: strict CF only
   T2: cf_balanced only
   T3: vary_operator only
   T4: strict ∪ cf_balanced
   T5: strict ∪ vary_operator
   T6: cf_balanced ∪ vary_operator
   T7: strict ∪ cf_balanced ∪ vary_operator

For each: train on ALL of the source data (no held-out from training set),
test on natural GSM8K (a) single-op subset, (b) first-op-in-chain labels.

Reports per-layer accuracy, macro precision/recall/F1, per-class metrics.
Picks the layer-and-train-combo with best macro F1 on natural single-op
as the final winner.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
STRICT_COL = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_op_strict_colon_acts.pt"
STRICT_CF = REPO.parent / "cf-datasets" / "gsm8k_cf_op_strict.json"
SVAMP_COL = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "cf_balanced_colon_acts.pt"
SVAMP_META = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "cf_balanced_colon_acts_meta.json"
VARY_OP_COL = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_vary_operator_colon_acts.pt"
VARY_OP_CF = REPO.parent / "cf-datasets" / "gsm8k_vary_operator.json"
NAT_COL = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
NAT_META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
OUT_PDF = PD / "operator_probe_to_natural.pdf"
OUT_JSON = PD / "operator_probe_to_natural.json"

OP_NAMES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
op_to_int = {n: i for i, n in enumerate(OP_NAMES)}


def metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_per_class": precision_score(y_true, y_pred, average=None,
                                                labels=list(range(4)), zero_division=0).tolist(),
        "recall_per_class":    recall_score(y_true, y_pred, average=None,
                                             labels=list(range(4)), zero_division=0).tolist(),
        "f1_per_class":        f1_score(y_true, y_pred, average=None,
                                         labels=list(range(4)), zero_division=0).tolist(),
    }


def load_strict():
    X = torch.load(STRICT_COL, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(STRICT_CF))
    return X, np.array([op_to_int[r["type"]] for r in rows])


def load_balanced():
    X = torch.load(SVAMP_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(SVAMP_META))
    types = ["Common-Division" if t == "Common-Divison" else t for t in meta["types"]]
    types = types[:X.shape[0]]
    keep = [i for i, t in enumerate(types) if t in op_to_int]
    return X[keep], np.array([op_to_int[types[i]] for i in keep])


def load_vary_op():
    X = torch.load(VARY_OP_COL, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(VARY_OP_CF))
    return X, np.array([op_to_int[r["type"]] for r in rows])


def load_natural():
    X = torch.load(NAT_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(NAT_META))
    ops_used = meta.get("operators_used", [])[:X.shape[0]]
    keep_single = [i for i, ou in enumerate(ops_used)
                    if isinstance(ou, list) and len(ou) == 1]
    keep_first  = [i for i, ou in enumerate(ops_used)
                    if isinstance(ou, list) and len(ou) >= 1 and ou[0] in op_to_int]
    return (X[keep_single], np.array([op_to_int[ops_used[i][0]] for i in keep_single]),
            X[keep_first],  np.array([op_to_int[ops_used[i][0]] for i in keep_first]))


def main():
    print("Loading...")
    Xs, ys = load_strict()
    Xb, yb = load_balanced()
    Xv, yv = load_vary_op()
    X_nat_single, y_nat_single, X_nat_first, y_nat_first = load_natural()
    Lc = Xs.shape[1]
    # All sources must have the same Lc
    assert Xb.shape[1] == Lc and Xv.shape[1] == Lc and X_nat_single.shape[1] == Lc

    sources = {"strict": (Xs, ys), "balanced": (Xb, yb), "vary_op": (Xv, yv)}
    combos = [
        ("T1_strict",                   ("strict",)),
        ("T2_balanced",                 ("balanced",)),
        ("T3_vary_op",                  ("vary_op",)),
        ("T4_strict+balanced",          ("strict", "balanced")),
        ("T5_strict+vary_op",           ("strict", "vary_op")),
        ("T6_balanced+vary_op",         ("balanced", "vary_op")),
        ("T7_all_three",                ("strict", "balanced", "vary_op")),
    ]

    results = {}
    for name, srcs in combos:
        X_tr = np.concatenate([sources[s][0] for s in srcs], axis=0)
        y_tr = np.concatenate([sources[s][1] for s in srcs])
        print(f"\n{name}: trained on {srcs}  N_train={len(y_tr)}")
        per_layer_single = []
        per_layer_first = []
        for l in range(Lc):
            sc = StandardScaler().fit(X_tr[:, l, :])
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
                sc.transform(X_tr[:, l, :]), y_tr)
            yp_single = clf.predict(sc.transform(X_nat_single[:, l, :]))
            yp_first  = clf.predict(sc.transform(X_nat_first[:, l, :]))
            per_layer_single.append(metrics(y_nat_single, yp_single))
            per_layer_first.append(metrics(y_nat_first, yp_first))
        results[name] = {"sources": list(srcs),
                          "per_layer_single": per_layer_single,
                          "per_layer_first":  per_layer_first}
        accs_s = [m["acc"] for m in per_layer_single]
        f1s_s = [m["f1_macro"] for m in per_layer_single]
        accs_f = [m["acc"] for m in per_layer_first]
        f1s_f = [m["f1_macro"] for m in per_layer_first]
        ls_b_s = int(np.argmax(f1s_s)); ls_b_f = int(np.argmax(f1s_f))
        print(f"  natural single-op best: L{ls_b_s}  acc={accs_s[ls_b_s]:.3f}  F1={f1s_s[ls_b_s]:.3f}")
        print(f"  natural first-op  best: L{ls_b_f}  acc={accs_f[ls_b_f]:.3f}  F1={f1s_f[ls_b_f]:.3f}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # Winner table
    print("\n=== Winner table (best by F1 macro on natural single-op) ===")
    print(f"{'combo':<22} {'best L':<7} {'acc':<6} {'F1':<6} {'prec':<6} {'recall':<6}")
    for name in results:
        per = results[name]["per_layer_single"]
        f1s = [m["f1_macro"] for m in per]
        lb = int(np.argmax(f1s)); m = per[lb]
        print(f"{name:<22} {lb:<7} {m['acc']:<6.3f} {m['f1_macro']:<6.3f} "
              f"{m['precision_macro']:<6.3f} {m['recall_macro']:<6.3f}")
    print()
    print(f"=== Winner table (best by F1 macro on natural first-op) ===")
    print(f"{'combo':<22} {'best L':<7} {'acc':<6} {'F1':<6} {'prec':<6} {'recall':<6}")
    for name in results:
        per = results[name]["per_layer_first"]
        f1s = [m["f1_macro"] for m in per]
        lb = int(np.argmax(f1s)); m = per[lb]
        print(f"{name:<22} {lb:<7} {m['acc']:<6.3f} {m['f1_macro']:<6.3f} "
              f"{m['precision_macro']:<6.3f} {m['recall_macro']:<6.3f}")

    # PDF
    COLORS = {
        "T1_strict": "#1f77b4", "T2_balanced": "#aec7e8", "T3_vary_op": "#17becf",
        "T4_strict+balanced": "#2ca02c", "T5_strict+vary_op": "#9467bd",
        "T6_balanced+vary_op": "#ff7f0e", "T7_all_three": "#d62728",
    }
    xs = np.arange(Lc)
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: F1 macro per layer for both targets
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, target, title in [
            (axes[0], "per_layer_single", "Natural GSM8K single-op (N=174)"),
            (axes[1], "per_layer_first",  "Natural GSM8K first-op (N=1289, multi-op)"),
        ]:
            for name in results:
                vals = [m["f1_macro"] for m in results[name][target]]
                ax.plot(xs, vals, "o-", lw=1.5, color=COLORS[name], label=name)
            ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance F1")
            ax.set_xlabel("colon layer"); ax.set_ylabel("macro F1")
            ax.set_xticks(xs); ax.set_ylim(0, 1.05)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
        fig.suptitle("Operator probe transfer to natural GSM8K — F1 macro per training combo",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 2: per-class P/R/F1 at WINNER training combo, best layer (single-op)
        # Find winner
        winner_name = max(results, key=lambda n:
                          max(m["f1_macro"] for m in results[n]["per_layer_single"]))
        per = results[winner_name]["per_layer_single"]
        f1s = [m["f1_macro"] for m in per]
        lb = int(np.argmax(f1s)); m_best = per[lb]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        ax = axes[0]
        w = 0.27; xs_op = np.arange(4)
        ax.bar(xs_op - w, m_best["precision_per_class"], w, color="#1f77b4", label="precision")
        ax.bar(xs_op,     m_best["recall_per_class"],    w, color="#2ca02c", label="recall")
        ax.bar(xs_op + w, m_best["f1_per_class"],        w, color="#d62728", label="F1")
        ax.set_xticks(xs_op); ax.set_xticklabels(["Add","Sub","Mul","Div"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("score")
        ax.set_title(f"WINNER → natural single-op\n{winner_name}, L{lb}, "
                     f"acc={m_best['acc']:.2f}, macro F1={m_best['f1_macro']:.2f}",
                     fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        # Same for first-op winner
        winner_name_f = max(results, key=lambda n:
                            max(m["f1_macro"] for m in results[n]["per_layer_first"]))
        per_f = results[winner_name_f]["per_layer_first"]
        f1s_f = [m["f1_macro"] for m in per_f]
        lb_f = int(np.argmax(f1s_f)); m_best_f = per_f[lb_f]
        ax = axes[1]
        ax.bar(xs_op - w, m_best_f["precision_per_class"], w, color="#1f77b4", label="precision")
        ax.bar(xs_op,     m_best_f["recall_per_class"],    w, color="#2ca02c", label="recall")
        ax.bar(xs_op + w, m_best_f["f1_per_class"],        w, color="#d62728", label="F1")
        ax.set_xticks(xs_op); ax.set_xticklabels(["Add","Sub","Mul","Div"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("score")
        ax.set_title(f"WINNER → natural first-op\n{winner_name_f}, L{lb_f}, "
                     f"acc={m_best_f['acc']:.2f}, macro F1={m_best_f['f1_macro']:.2f}",
                     fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Best training-combo's per-class precision/recall/F1 on natural GSM8K",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 3: Summary table
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.axis("off")
        rows = [["combo", "target", "best L", "acc", "F1", "prec", "recall"]]
        for target, label in [("per_layer_single", "natural single-op"),
                               ("per_layer_first",  "natural first-op")]:
            for name in results:
                per = results[name][target]
                f1s = [m["f1_macro"] for m in per]
                lb = int(np.argmax(f1s)); m = per[lb]
                rows.append([name, label, str(lb), f"{m['acc']:.3f}",
                             f"{m['f1_macro']:.3f}", f"{m['precision_macro']:.3f}",
                             f"{m['recall_macro']:.3f}"])
        tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        tbl.scale(1, 1.3)
        for j in range(len(rows[0])):
            tbl[0, j].set_facecolor("#dddddd")
        ax.set_title("All 7 training combos → both natural-GSM8K targets",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
