"""Held-out correctness probe at the ':' residual on SVAMP (CODI-GPT-2).

Direct analog to ../../visualizations-all/gpt2/correctness_probe.py, but on
the ':' residual (shape N x 13 x 768) instead of the latent loop residual.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PD = Path(__file__).resolve().parent
ACTS = PD / "svamp_colon_acts.pt"
META = PD / "svamp_colon_acts_meta.json"
OUT_PDF = PD / "correctness_probe_colon.pdf"
OUT_JSON = PD / "correctness_probe_colon.json"


def main():
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(META))
    N, L, H = a.shape
    print(f"colon acts shape={a.shape}  N={N}")

    # Build correctness label = (extracted prediction matches gold) per example.
    pred = np.array([np.nan if v is None else float(v) for v in meta["pred_int_extracted"]])
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    y = np.array([
        (not np.isnan(pred[i])) and (not np.isnan(gold[i])) and abs(pred[i] - gold[i]) < 1e-3
        for i in range(N)
    ], dtype=int)
    baseline = max(y.mean(), 1 - y.mean())
    print(f"  positives={int(y.sum())}/{N} = {y.mean()*100:.1f}%; baseline={baseline*100:.1f}%")

    idx_tr, idx_te = train_test_split(
        np.arange(N), test_size=0.2, random_state=0, stratify=y,
    )
    y_tr, y_te = y[idx_tr], y[idx_te]

    train_acc = np.zeros(L); test_acc = np.zeros(L)
    train_auc = np.zeros(L); test_auc = np.zeros(L)
    for l in range(L):
        X = a[:, l, :]
        sc = StandardScaler().fit(X[idx_tr])
        Xtr, Xte = sc.transform(X[idx_tr]), sc.transform(X[idx_te])
        clf = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs").fit(Xtr, y_tr)
        tr_prob = clf.predict_proba(Xtr)[:, 1]
        te_prob = clf.predict_proba(Xte)[:, 1]
        train_acc[l] = (clf.predict(Xtr) == y_tr).mean()
        test_acc[l]  = (clf.predict(Xte) == y_te).mean()
        train_auc[l] = roc_auc_score(y_tr, tr_prob)
        test_auc[l]  = roc_auc_score(y_te, te_prob)

    print("\nLayer | test_acc | test_AUC")
    for l in range(L):
        print(f"  {l:>2}  | {test_acc[l]*100:6.2f}% | {test_auc[l]:.3f}")
    l_best = int(test_auc.argmax())
    print(f"\nbest test AUC: layer={l_best}, auc={test_auc[l_best]:.3f}, acc={test_acc[l_best]*100:.1f}% (baseline {baseline*100:.1f}%)")

    OUT_JSON.write_text(json.dumps({
        "shape": [N, L, H], "n_pos": int(y.sum()), "baseline": float(baseline),
        "train_acc": train_acc.tolist(), "test_acc": test_acc.tolist(),
        "train_auc": train_auc.tolist(), "test_auc": test_auc.tolist(),
        "best_layer": l_best, "best_test_auc": float(test_auc[l_best]),
        "best_test_acc": float(test_acc[l_best]),
        "note": "probe on residual at the position whose INPUT is ':' (canonical pre-answer cell)",
    }, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(L)
    axes[0].plot(xs, train_acc * 100, "o-", label="train")
    axes[0].plot(xs, test_acc * 100, "s-", label="test")
    axes[0].axhline(baseline * 100, color="gray", ls=":", label=f"baseline {baseline*100:.1f}%")
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Held-out correctness probe accuracy @ ':' residual")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(xs, train_auc, "o-", label="train")
    axes[1].plot(xs, test_auc, "s-", label="test")
    axes[1].axhline(0.5, color="gray", ls=":", label="chance")
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("Held-out correctness probe AUC @ ':' residual")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0.4, 1.0)
    fig.suptitle(f"CODI-GPT-2 correctness probe at ':' residual (SVAMP N={N})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=140)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
