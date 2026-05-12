"""Comprehensive operator-probe slideshow.

Aggregates every operator-probe variant we've run on GSM8K and reports
per-layer accuracy, precision, recall, F1 (macro and per-class) and a
final comparison page across probes.

Probes covered:
  P1. Strict CF 4-class within-CV
  P2. Forward transfer: train strict → test cf_balanced
  P3. Reverse transfer: train cf_balanced → test strict
  P4. Union: train 70% strict ∪ 70% cf_balanced; test on:
        a) held-out strict 30%
        b) held-out cf_balanced 30%
        c) natural GSM8K single-op (N=174)
        d) natural GSM8K first-op (N=1289)
  P5. vary_operator within-CV
  P6. vary_operator LOTO
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                      train_test_split)
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
OUT_PDF = PD / "operator_probe_full_slideshow.pdf"
OUT_JSON = PD / "operator_probe_full_slideshow.json"

OP_NAMES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
op_to_int = {n: i for i, n in enumerate(OP_NAMES)}
SEED = 0


def metrics(y_true, y_pred):
    """Return dict with acc, macro-precision/recall/F1, per-class precision/recall/F1."""
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


def fit_predict_per_layer(X_tr, y_tr, X_te, y_te):
    """For each layer, fit a probe on train, predict test. Return list of
    metric dicts."""
    L = X_tr.shape[1]
    out = []
    for l in range(L):
        sc = StandardScaler().fit(X_tr[:, l, :])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
            sc.transform(X_tr[:, l, :]), y_tr)
        ypred = clf.predict(sc.transform(X_te[:, l, :]))
        out.append(metrics(y_te, ypred))
    return out


def cv_metrics_per_layer(X, y, n_folds=5):
    """5-fold CV metrics per layer."""
    L = X.shape[1]
    out_layers = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for l in range(L):
        accs, p_macros, r_macros, f1_macros = [], [], [], []
        p_per, r_per, f_per = [], [], []
        for tr, te in skf.split(X[:, l, :], y):
            sc = StandardScaler().fit(X[tr, l, :])
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
                sc.transform(X[tr, l, :]), y[tr])
            ypred = clf.predict(sc.transform(X[te, l, :]))
            m = metrics(y[te], ypred)
            accs.append(m["acc"])
            p_macros.append(m["precision_macro"])
            r_macros.append(m["recall_macro"])
            f1_macros.append(m["f1_macro"])
            p_per.append(m["precision_per_class"])
            r_per.append(m["recall_per_class"])
            f_per.append(m["f1_per_class"])
        out_layers.append({
            "acc": float(np.mean(accs)),
            "precision_macro": float(np.mean(p_macros)),
            "recall_macro":    float(np.mean(r_macros)),
            "f1_macro":        float(np.mean(f1_macros)),
            "precision_per_class": np.mean(p_per, axis=0).tolist(),
            "recall_per_class":    np.mean(r_per, axis=0).tolist(),
            "f1_per_class":        np.mean(f_per, axis=0).tolist(),
        })
    return out_layers


def loto_metrics_per_layer(X, y, groups):
    """LeaveOneGroupOut metrics per layer."""
    L = X.shape[1]
    out = []
    logo = LeaveOneGroupOut()
    for l in range(L):
        ytrue_all = []; ypred_all = []
        for tr, te in logo.split(X[:, l, :], y, groups):
            if len(np.unique(y[tr])) < 4: continue
            sc = StandardScaler().fit(X[tr, l, :])
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
                sc.transform(X[tr, l, :]), y[tr])
            ypred_all.append(clf.predict(sc.transform(X[te, l, :])))
            ytrue_all.append(y[te])
        if not ytrue_all:
            out.append(None); continue
        out.append(metrics(np.concatenate(ytrue_all), np.concatenate(ypred_all)))
    return out


def load_strict():
    X = torch.load(STRICT_COL, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(STRICT_CF))
    return X, np.array([op_to_int[r["type"]] for r in rows])


def load_svamp():
    X = torch.load(SVAMP_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(SVAMP_META))
    types = ["Common-Division" if t == "Common-Divison" else t for t in meta["types"]]
    types = types[:X.shape[0]]
    keep = [i for i, t in enumerate(types) if t in op_to_int]
    return X[keep], np.array([op_to_int[types[i]] for i in keep])


def load_vary_op():
    X = torch.load(VARY_OP_COL, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(VARY_OP_CF))
    y = np.array([op_to_int[r["type"]] for r in rows])
    groups = np.array([r["template_id"] for r in rows])
    return X, y, groups


def load_natural():
    X = torch.load(NAT_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(NAT_META))
    ops_used = meta.get("operators_used", [])[:X.shape[0]]
    keep_single = [i for i, ou in enumerate(ops_used) if isinstance(ou, list) and len(ou) == 1]
    keep_first = [i for i, ou in enumerate(ops_used)
                  if isinstance(ou, list) and len(ou) >= 1 and ou[0] in op_to_int]
    return (X[keep_single], np.array([op_to_int[ops_used[i][0]] for i in keep_single]),
            X[keep_first], np.array([op_to_int[ops_used[i][0]] for i in keep_first]))


def main():
    print("Loading...")
    Xs, ys = load_strict()
    Xb, yb = load_svamp()
    Xv, yv, gv = load_vary_op()
    Xn_single, yn_single, Xn_first, yn_first = load_natural()
    Lc = Xs.shape[1]

    print(f"strict: N={len(ys)}, balanced: N={len(yb)}, vary_op: N={len(yv)}")
    print(f"natural single-op: N={len(yn_single)}, first-op: N={len(yn_first)}")

    print("\nP1. Strict CF 5-fold CV per layer")
    P1 = cv_metrics_per_layer(Xs, ys)
    print("P2. Forward transfer (train strict full → test cf_balanced)")
    P2 = fit_predict_per_layer(Xs, ys, Xb, yb)
    print("P3. Reverse transfer (train cf_balanced full → test strict)")
    P3 = fit_predict_per_layer(Xb, yb, Xs, ys)
    print("P4. Union training (70% strict + 70% cf_balanced)")
    idx_s_tr, idx_s_te = train_test_split(np.arange(len(ys)), test_size=0.30,
                                            random_state=SEED, stratify=ys)
    idx_b_tr, idx_b_te = train_test_split(np.arange(len(yb)), test_size=0.30,
                                            random_state=SEED, stratify=yb)
    Xu_tr = np.concatenate([Xs[idx_s_tr], Xb[idx_b_tr]], axis=0)
    yu_tr = np.concatenate([ys[idx_s_tr], yb[idx_b_tr]])
    P4a = fit_predict_per_layer(Xu_tr, yu_tr, Xs[idx_s_te], ys[idx_s_te])
    P4b = fit_predict_per_layer(Xu_tr, yu_tr, Xb[idx_b_te], yb[idx_b_te])
    P4c = fit_predict_per_layer(Xu_tr, yu_tr, Xn_single, yn_single)
    P4d = fit_predict_per_layer(Xu_tr, yu_tr, Xn_first, yn_first)
    print("P5. vary_operator 5-fold CV")
    P5 = cv_metrics_per_layer(Xv, yv)
    print("P6. vary_operator LOTO")
    P6 = loto_metrics_per_layer(Xv, yv, gv)

    probes = {
        "P1_strict_CV":   P1,
        "P2_forward":     P2,
        "P3_reverse":     P3,
        "P4a_union_strict_30":    P4a,
        "P4b_union_balanced_30":  P4b,
        "P4c_union_nat_singleop": P4c,
        "P4d_union_nat_firstop":  P4d,
        "P5_varyop_CV":   P5,
        "P6_varyop_LOTO": P6,
    }

    OUT_JSON.write_text(json.dumps(probes, indent=2))
    print(f"saved {OUT_JSON}")

    # SLIDESHOW
    xs_layer = np.arange(Lc)
    COLORS = {
        "P1_strict_CV":   "#1f77b4",
        "P2_forward":     "#2ca02c",
        "P3_reverse":     "#ff7f0e",
        "P4a_union_strict_30":    "#9467bd",
        "P4b_union_balanced_30":  "#aec7e8",
        "P4c_union_nat_singleop": "#d62728",
        "P4d_union_nat_firstop":  "#8c564b",
        "P5_varyop_CV":   "#17becf",
        "P6_varyop_LOTO": "#7f7f7f",
    }

    def per_layer(probe, key):
        return [(m[key] if m is not None else 0.0) for m in probes[probe]]

    with PdfPages(OUT_PDF) as pdf:
        # ============================================================
        # Page 1: All probes — accuracy per layer
        # ============================================================
        fig, ax = plt.subplots(figsize=(15, 6))
        for name in probes:
            ax.plot(xs_layer, per_layer(name, "acc"), "o-", lw=1.6,
                    color=COLORS[name], label=name)
        ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance (1/4)")
        ax.set_xlabel("layer"); ax.set_ylabel("accuracy")
        ax.set_xticks(xs_layer); ax.set_ylim(0, 1.05)
        ax.set_title("All operator probes — accuracy per layer",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============================================================
        # Page 2: Macro precision / recall / F1
        # ============================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, key, title in [
            (axes[0], "precision_macro", "Macro precision"),
            (axes[1], "recall_macro",    "Macro recall"),
            (axes[2], "f1_macro",        "Macro F1"),
        ]:
            for name in probes:
                ax.plot(xs_layer, per_layer(name, key), "o-", lw=1.5,
                        color=COLORS[name], label=name)
            ax.axhline(0.25, color="black", ls="--", lw=0.5)
            ax.set_xlabel("layer"); ax.set_ylabel(key); ax.set_xticks(xs_layer)
            ax.set_ylim(0, 1.05)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)
            if ax is axes[0]: ax.legend(fontsize=7, loc="lower right")
        fig.suptitle("Macro precision / recall / F1 per layer, all probes",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============================================================
        # Page 3: Per-class precision/recall/F1 — at the best layer for each probe
        # ============================================================
        # 9 probes × 4 ops × {precision, recall, F1} ≈ a lot.
        # Plot the BEST layer per probe with class-wise bars.
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        names = list(probes.keys())
        for ax, name in zip(axes.flat, names):
            accs = per_layer(name, "acc")
            l_best = int(np.argmax(accs))
            m = probes[name][l_best]
            if m is None:
                ax.axis("off"); continue
            xs = np.arange(4); w = 0.27
            ax.bar(xs - w, m["precision_per_class"], w, color="#1f77b4", label="precision")
            ax.bar(xs,     m["recall_per_class"],    w, color="#2ca02c", label="recall")
            ax.bar(xs + w, m["f1_per_class"],        w, color="#d62728", label="F1")
            ax.set_xticks(xs); ax.set_xticklabels(["Add","Sub","Mul","Div"], fontsize=8)
            ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
            ax.set_title(f"{name} (best L{l_best}, acc={m['acc']:.2f})",
                         fontsize=9, fontweight="bold")
            if ax is axes[0, 0]:
                ax.legend(fontsize=8)
            for xi, v in zip(xs - w, m["precision_per_class"]):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
            for xi, v in zip(xs, m["recall_per_class"]):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
            for xi, v in zip(xs + w, m["f1_per_class"]):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
        fig.suptitle("Per-class precision / recall / F1 at each probe's BEST layer",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============================================================
        # Page 4: Mixed-probe panels — union variants vs single-source
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        ax = axes[0]
        for name in ["P1_strict_CV", "P2_forward", "P3_reverse",
                     "P4a_union_strict_30", "P4b_union_balanced_30"]:
            ax.plot(xs_layer, per_layer(name, "acc"), "o-", lw=1.6,
                    color=COLORS[name], label=name)
        ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance")
        ax.set_xlabel("layer"); ax.set_ylabel("accuracy")
        ax.set_xticks(xs_layer); ax.set_ylim(0, 1.05)
        ax.set_title("In-domain: union vs single-source probes",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1]
        for name in ["P2_forward", "P3_reverse",
                     "P4c_union_nat_singleop", "P4d_union_nat_firstop"]:
            ax.plot(xs_layer, per_layer(name, "acc"), "o-", lw=1.6,
                    color=COLORS[name], label=name)
        ax.axhline(0.25, color="black", ls="--", lw=0.5, label="chance")
        ax.set_xlabel("layer"); ax.set_ylabel("accuracy")
        ax.set_xticks(xs_layer); ax.set_ylim(0, 1.05)
        ax.set_title("Out-of-domain: did mixed training help?",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle("Mixed (union) probe vs single-source: in-domain & out-of-domain",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============================================================
        # Page 5: Summary table
        # ============================================================
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.axis("off")
        rows = [["probe", "best L", "best acc", "macro F1", "macro prec.", "macro recall"]]
        for name in probes:
            accs = per_layer(name, "acc")
            l_best = int(np.argmax(accs))
            m = probes[name][l_best]
            rows.append([name, str(l_best), f"{m['acc']:.3f}",
                         f"{m['f1_macro']:.3f}",
                         f"{m['precision_macro']:.3f}",
                         f"{m['recall_macro']:.3f}"])
        tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        for j in range(len(rows[0])):
            tbl[0, j].set_facecolor("#dddddd")
        ax.set_title("Summary across all operator probes (best layer per probe)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
