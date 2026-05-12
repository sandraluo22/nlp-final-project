"""One-vs-rest operator-PRESENCE probing on natural GSM8K activations.

For each operator (+, -, *, /), train a binary probe at every (step, layer)
that predicts "does this problem's solution chain contain operator X?"
from CODI's latent residual.

Tests whether CODI's latent loop linearly encodes which operators the
problem requires — multi-label, since most GSM8K problems use multiple.

Also runs a clean 4-class probe restricted to the single-operator-only
subset (174 examples) for comparison with SVAMP's operator probe.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[3]
LAT_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
LABELS_PATH = REPO.parent / "cf-datasets" / "gsm8k_op_presence.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "operator_probe_presence_gsm8k.json"
OUT_PDF = PD / "operator_probe_presence_gsm8k.pdf"

SEED = 0


def cv_auc(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs",
                                  class_weight="balanced").fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te], p))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs))


def cv_multiclass_acc(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs",
                                  class_weight="balanced", multi_class="multinomial"
                                  ).fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    acts = torch.load(LAT_PATH, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = acts.shape
    rows = json.load(open(LABELS_PATH))
    by_idx = {r["idx"]: r for r in rows}
    keep_idx = [i for i in range(N) if i in by_idx]
    acts = acts[keep_idx]
    has_add = np.array([by_idx[i]["has_add"] for i in keep_idx], dtype=int)
    has_sub = np.array([by_idx[i]["has_sub"] for i in keep_idx], dtype=int)
    has_mul = np.array([by_idx[i]["has_mul"] for i in keep_idx], dtype=int)
    has_div = np.array([by_idx[i]["has_div"] for i in keep_idx], dtype=int)
    n_distinct = np.array([by_idx[i]["n_distinct_ops"] for i in keep_idx])
    print(f"GSM8K presence labels: N_kept={acts.shape[0]}, "
          f"add={has_add.mean():.2f}, sub={has_sub.mean():.2f}, "
          f"mul={has_mul.mean():.2f}, div={has_div.mean():.2f}")

    auc = {op: np.zeros((S, L)) for op in ["add", "sub", "mul", "div"]}
    for s in range(S):
        for l in range(L):
            X = acts[:, s, l, :]
            auc["add"][s, l] = cv_auc(X, has_add)
            auc["sub"][s, l] = cv_auc(X, has_sub)
            auc["mul"][s, l] = cv_auc(X, has_mul)
            auc["div"][s, l] = cv_auc(X, has_div)
        for op in auc:
            print(f"  step {s+1}: AUC {op}={auc[op][s].max():.3f} (best L{int(auc[op][s].argmax())})")

    # Single-op-only 4-class probe
    single_mask = n_distinct == 1
    print(f"\nSingle-op subset: N={single_mask.sum()}")
    if single_mask.sum() >= 40:
        y_single = (1*has_add + 2*has_sub + 3*has_mul + 4*has_div)[single_mask] - 1  # 0..3
        X_single_all = acts[single_mask]
        single_acc = np.zeros((S, L))
        for s in range(S):
            for l in range(L):
                single_acc[s, l] = cv_multiclass_acc(X_single_all[:, s, l, :], y_single)
            print(f"  step {s+1}: 4-class acc max={single_acc[s].max():.3f} "
                  f"(best L{int(single_acc[s].argmax())})")
    else:
        single_acc = None

    OUT_JSON.write_text(json.dumps({
        "N": int(acts.shape[0]),
        "auc": {op: auc[op].tolist() for op in auc},
        "single_op_n": int(single_mask.sum()),
        "single_op_acc": None if single_acc is None else single_acc.tolist(),
    }, indent=2))
    print(f"saved {OUT_JSON}")

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, op in zip(axes.flat, ["add", "sub", "mul", "div"]):
            M = auc[op]
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                            vmin=0.5, vmax=1.0)
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_xticks(range(L)); ax.set_yticks(range(S))
            ax.set_yticklabels([str(s + 1) for s in range(S)])
            ax.set_title(f"AUC: has_{op}?", fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04)
            for s in range(S):
                for l in range(L):
                    v = M[s, l]
                    if v >= 0.7 or v <= 0.55:
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.75 else "black")
        fig.suptitle("GSM8K operator-presence probes (one-vs-rest on natural GSM8K test)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        if single_acc is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(single_acc, aspect="auto", origin="lower", cmap="magma",
                            vmin=0.25, vmax=1.0)
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_xticks(range(L)); ax.set_yticks(range(S))
            ax.set_yticklabels([str(s + 1) for s in range(S)])
            ax.set_title(f"4-class operator probe on SINGLE-OP subset (N={int(single_mask.sum())})",
                         fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04)
            for s in range(S):
                for l in range(L):
                    v = single_acc[s, l]
                    if v >= 0.4 or v <= 0.3:
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color="white" if v < 0.5 else "black")
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
