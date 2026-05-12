"""Held-out linear probe for faithfulness on CODI-GPT-2 latent activations.

Labels: svamp_judged.json -> {faithful, unfaithful, teacher_incorrect}
We train a BINARY probe (faithful vs unfaithful) on the 581 + 41 = 622
labeled examples. Stratified 70/30 split.

For each (latent step, layer) of svamp_student_gpt2/activations.pt:
  - StandardScaler + LogisticRegression on 70%.
  - Test acc + ROC AUC on held-out 30%.
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

REPO = Path(__file__).resolve().parents[2]
ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
JUDGED = REPO.parent / "cf-datasets" / "gsm8k_judged.json"
OUT_JSON = Path(__file__).resolve().parent / "faithfulness_probe.json"
OUT_PDF = Path(__file__).resolve().parent / "faithfulness_probe.pdf"


def main():
    print(f"loading activations {ACTS}")
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    print(f"  shape={a.shape}")

    print(f"loading judged labels {JUDGED}")
    judged = json.load(open(JUDGED))
    label_by_idx = {j["idx"]: j["label"] for j in judged}
    labels = np.array([label_by_idx.get(i, "teacher_incorrect") for i in range(N)])
    print(f"  counts: {dict(zip(*np.unique(labels, return_counts=True)))}")

    keep = np.where((labels == "faithful") | (labels == "unfaithful"))[0]
    y = np.array([1 if labels[i] == "faithful" else 0 for i in keep])
    print(f"  using {len(keep)} examples (faithful=1, unfaithful=0).")
    print(f"  faithful: {(y==1).sum()}; unfaithful: {(y==0).sum()}")

    rng = 0
    idx_tr, idx_te = train_test_split(np.arange(len(keep)), test_size=0.3,
                                      random_state=rng, stratify=y)
    print(f"  train {len(idx_tr)} / test {len(idx_te)}")
    y_tr, y_te = y[idx_tr], y[idx_te]
    baseline = max(y.mean(), 1 - y.mean())
    print(f"  majority-class baseline = {baseline*100:.1f}%")

    train_acc = np.zeros((S, L))
    test_acc = np.zeros((S, L))
    train_auc = np.zeros((S, L))
    test_auc = np.zeros((S, L))
    # also a "kept" subset accuracy on unfaithful only (sensitivity)
    test_unf_recall = np.zeros((S, L))
    test_fai_recall = np.zeros((S, L))

    for s in range(S):
        for l in range(L):
            X = a[keep[:, None].squeeze(), s, l, :]   # (Nkeep, H)
            sc = StandardScaler().fit(X[idx_tr])
            Xtr = sc.transform(X[idx_tr]); Xte = sc.transform(X[idx_te])
            clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs",
                                     class_weight="balanced")
            clf.fit(Xtr, y_tr)
            pred_te = clf.predict(Xte)
            prob_te = clf.predict_proba(Xte)[:, 1]
            train_acc[s, l] = (clf.predict(Xtr) == y_tr).mean()
            test_acc[s, l]  = (pred_te == y_te).mean()
            train_auc[s, l] = roc_auc_score(y_tr, clf.predict_proba(Xtr)[:, 1])
            test_auc[s, l]  = roc_auc_score(y_te, prob_te)
            # per-class recall
            test_fai_recall[s, l] = (pred_te[y_te == 1] == 1).mean() if (y_te == 1).any() else np.nan
            test_unf_recall[s, l] = (pred_te[y_te == 0] == 0).mean() if (y_te == 0).any() else np.nan
        print(f"  step {s+1}/{S} done")

    s_b, l_b = np.unravel_index(test_auc.argmax(), test_auc.shape)
    print(f"\nbest test AUC: step={s_b+1}, layer={l_b}, "
          f"auc={test_auc[s_b, l_b]:.3f}, acc={test_acc[s_b, l_b]*100:.1f}%")

    OUT_JSON.write_text(json.dumps({
        "shape": [N, S, L, H], "baseline": float(baseline),
        "n_faithful": int((y == 1).sum()), "n_unfaithful": int((y == 0).sum()),
        "n_train": len(idx_tr), "n_test": len(idx_te),
        "train_acc": train_acc.tolist(), "test_acc": test_acc.tolist(),
        "train_auc": train_auc.tolist(), "test_auc": test_auc.tolist(),
        "test_faithful_recall": test_fai_recall.tolist(),
        "test_unfaithful_recall": test_unf_recall.tolist(),
        "best_step_1indexed": int(s_b + 1), "best_layer": int(l_b),
        "best_test_auc": float(test_auc[s_b, l_b]),
        "best_test_acc": float(test_acc[s_b, l_b]),
    }, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, mat, title in [
        (axes[0, 0], test_acc, "test acc"),
        (axes[0, 1], test_auc, "test AUC"),
        (axes[1, 0], test_fai_recall, "test recall (faithful)"),
        (axes[1, 1], test_unf_recall, "test recall (unfaithful)"),
    ]:
        vmin = 0.5 if "AUC" in title else (baseline if "acc" in title else 0)
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=1.0)
        ax.set_xlabel("layer"); ax.set_ylabel("latent step")
        ax.set_yticks(range(S)); ax.set_yticklabels([str(s+1) for s in range(S)])
        ax.set_title(title, fontsize=10)
        for s in range(S):
            for l in range(L):
                v = mat[s, l]
                if v >= vmin + (1 - vmin) * 0.4:
                    ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < vmin + (1-vmin)*0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"Faithfulness probe (70/30 stratified split, balanced LR) — "
                 f"baseline {baseline*100:.1f}%",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PDF, dpi=140)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
