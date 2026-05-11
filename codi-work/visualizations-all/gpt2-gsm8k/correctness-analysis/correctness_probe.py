"""Held-out linear probe for student_correct on per-(step, layer) activations.

Tests whether the PC1 bifurcation reflects a learnable correctness signal that
generalizes, or merely per-example structure that happens to correlate with
correctness. Output: per-(step, layer) train/test accuracy + ROC AUC, heatmap.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
STUDENT_RESULTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "results.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2" / "correctness_probe.pdf"
OUT_JSON = REPO / "visualizations-all" / "gpt2" / "correctness_probe.json"


def main():
    print("loading metadata", flush=True)
    student = json.load(open(STUDENT_RESULTS))
    y = np.array([bool(s["correct"]) for s in student], dtype=int)
    # SVAMP test/train union — match the 1000 in activations
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    assert len(full) == len(y), f"meta/results mismatch: {len(full)} vs {len(y)}"
    print(f"  n={len(y)}  positives={int(y.sum())} ({y.mean()*100:.1f}%)")

    print(f"loading activations {ACTS}", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    print(f"  shape={a.shape}")

    rng = 0
    train_acc = np.zeros((S, L)); test_acc = np.zeros((S, L))
    train_auc = np.zeros((S, L)); test_auc = np.zeros((S, L))
    baseline = max(y.mean(), 1 - y.mean())

    # one 80/20 split, used at every cell so cells are comparable.
    idx_tr, idx_te = train_test_split(
        np.arange(N), test_size=0.2, random_state=rng, stratify=y,
    )
    y_tr, y_te = y[idx_tr], y[idx_te]
    print(f"  split: train {len(idx_tr)} / test {len(idx_te)}  baseline={baseline*100:.1f}%")

    for s in range(S):
        for l in range(L):
            X = a[:, s, l, :]
            sc = StandardScaler().fit(X[idx_tr])
            Xtr, Xte = sc.transform(X[idx_tr]), sc.transform(X[idx_te])
            clf = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs")
            clf.fit(Xtr, y_tr)
            tr_pred = clf.predict(Xtr); te_pred = clf.predict(Xte)
            tr_prob = clf.predict_proba(Xtr)[:, 1]; te_prob = clf.predict_proba(Xte)[:, 1]
            train_acc[s, l] = (tr_pred == y_tr).mean()
            test_acc[s, l]  = (te_pred == y_te).mean()
            train_auc[s, l] = roc_auc_score(y_tr, tr_prob)
            test_auc[s, l]  = roc_auc_score(y_te, te_prob)
        print(f"  step {s+1}/{S} done", flush=True)

    print("\n=== Test accuracy heatmap (rows=step, cols=layer) ===")
    print(f"  baseline (majority-class) = {baseline*100:.1f}%")
    for s in range(S):
        row = "  ".join(f"{test_acc[s, l]*100:5.1f}" for l in range(L))
        print(f"  step {s+1}: {row}")
    print("\n=== Test ROC AUC ===")
    for s in range(S):
        row = "  ".join(f"{test_auc[s, l]:.3f}" for l in range(L))
        print(f"  step {s+1}: {row}")

    s_best, l_best = np.unravel_index(test_auc.argmax(), test_auc.shape)
    print(f"\nbest test AUC: step={s_best+1}, layer={l_best}, "
          f"auc={test_auc[s_best, l_best]:.3f}, "
          f"acc={test_acc[s_best, l_best]*100:.1f}% (vs baseline {baseline*100:.1f}%)")

    OUT_JSON.write_text(json.dumps({
        "shape": [N, S, L, H], "n_pos": int(y.sum()), "baseline": float(baseline),
        "train_acc": train_acc.tolist(), "test_acc": test_acc.tolist(),
        "train_auc": train_auc.tolist(), "test_auc": test_auc.tolist(),
        "best_step_1indexed": int(s_best + 1), "best_layer": int(l_best),
        "best_test_auc": float(test_auc[s_best, l_best]),
        "best_test_acc": float(test_acc[s_best, l_best]),
    }, indent=2))
    print(f"saved {OUT_JSON}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, mat, title in [
        (axes[0, 0], train_acc, "train acc"),
        (axes[0, 1], test_acc,  "test acc"),
        (axes[1, 0], train_auc, "train AUC"),
        (axes[1, 1], test_auc,  "test AUC"),
    ]:
        vmin = 0.5 if "AUC" in title else baseline
        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=1.0, cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("layer"); ax.set_ylabel("latent step")
        ax.set_yticks(range(S)); ax.set_yticklabels([str(s + 1) for s in range(S)])
        ax.set_xticks(range(0, L, 2))
        for s_ in range(S):
            for l_ in range(L):
                v = mat[s_, l_]
                col = "white" if v < (vmin + 1) / 2 else "black"
                ax.text(l_, s_, f"{v:.2f}", ha="center", va="center", fontsize=6, color=col)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Held-out linear probe: student_correct from per-(step, layer) activations\n"
                 f"baseline = {baseline*100:.1f}% (majority class)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PDF, dpi=140)
    plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
