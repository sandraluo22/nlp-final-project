"""Faithfulness probe v2: baselines + cross-validation + regularization sweep.

Adds to v1:
1. Permutation null:  shuffle labels N_PERM times, refit, get null AUC dist
   per cell. Tells us if 0.77 AUC is real or noise.
2. Random Gaussian feature baseline: replace activations with N(0,1) of same
   shape. Should give chance AUC; serves as a "structureless feature" floor.
3. 5-fold stratified CV: reduces variance from the small test set.
4. C-sweep: {0.01, 0.1, 1.0, 10.0} per cell, report best C and its CV-mean AUC.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
JUDGED = REPO.parent / "cf-datasets" / "svamp_judged.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "faithfulness_probe_v2.json"
OUT_PDF = PD / "faithfulness_probe_v2.pdf"

N_PERM = 30          # label permutations per cell
N_FOLDS = 5
C_GRID = [0.01, 0.1, 1.0, 10.0]
SEED = 0


def cv_auc(X, y, C, n_folds=N_FOLDS, seed=SEED):
    """Return mean test AUC across stratified folds."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=4000, C=C, solver="lbfgs",
                                 class_weight="balanced").fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    print(f"acts={a.shape}")
    labels = np.array(["teacher_incorrect"] * N, dtype=object)
    judged = json.load(open(JUDGED))
    for j in judged: labels[j["idx"]] = j["label"]
    keep = np.where((labels == "faithful") | (labels == "unfaithful"))[0]
    y = (labels[keep] == "faithful").astype(int)
    print(f"  faithful={int((y==1).sum())}  unfaithful={int((y==0).sum())}")
    baseline = max(y.mean(), 1 - y.mean())

    # Baseline 1: random Gaussian features at each cell. We use ONE Gaussian
    # tensor of the same shape (N_keep, H) and run the same CV pipeline.
    rng = np.random.default_rng(SEED)
    rand_feats = rng.standard_normal((len(keep), H)).astype(np.float32)

    # Pre-compute keep subset of acts.
    a_keep = a[keep]   # (N_keep, S, L, H)

    cv_best_auc = np.zeros((S, L))
    cv_best_C = np.full((S, L), -1.0)
    cv_best_std = np.zeros((S, L))
    perm_mean_auc = np.zeros((S, L))
    perm_std_auc = np.zeros((S, L))
    perm_95 = np.zeros((S, L))
    rand_auc = np.zeros((S, L))

    rng_perm = np.random.default_rng(SEED + 1)

    for s in range(S):
        for l in range(L):
            X = a_keep[:, s, l, :]
            # C-sweep with 5-fold CV
            best_auc, best_C, best_std = -1.0, -1.0, 0.0
            for C in C_GRID:
                m, sd = cv_auc(X, y, C)
                if m > best_auc:
                    best_auc, best_C, best_std = m, C, sd
            cv_best_auc[s, l] = best_auc
            cv_best_C[s, l] = best_C
            cv_best_std[s, l] = best_std

            # Permutation null at the best C
            null_aucs = []
            for p in range(N_PERM):
                y_sh = rng_perm.permutation(y)
                m, _ = cv_auc(X, y_sh, best_C)
                null_aucs.append(m)
            perm_mean_auc[s, l] = float(np.mean(null_aucs))
            perm_std_auc[s, l] = float(np.std(null_aucs))
            perm_95[s, l] = float(np.percentile(null_aucs, 95))

            # Random Gaussian features baseline (one run)
            r_auc, _ = cv_auc(rand_feats, y, best_C)
            rand_auc[s, l] = r_auc

        print(f"  step {s+1}/{S} done. "
              f"layer-wise best AUC at step {s+1}: {cv_best_auc[s].max():.3f}")

    s_b, l_b = np.unravel_index(cv_best_auc.argmax(), cv_best_auc.shape)
    print(f"\nbest CV-mean test AUC: step={s_b+1}, layer={l_b}, "
          f"auc={cv_best_auc[s_b, l_b]:.3f} (std {cv_best_std[s_b, l_b]:.3f}, "
          f"C={cv_best_C[s_b, l_b]:.2f})")
    print(f"  permutation null mean = {perm_mean_auc[s_b, l_b]:.3f} "
          f"(p95 = {perm_95[s_b, l_b]:.3f})")
    print(f"  random-feature baseline AUC = {rand_auc[s_b, l_b]:.3f}")
    print(f"  delta to permutation null = "
          f"{cv_best_auc[s_b, l_b] - perm_mean_auc[s_b, l_b]:+.3f}")

    OUT_JSON.write_text(json.dumps({
        "n_faithful": int((y == 1).sum()), "n_unfaithful": int((y == 0).sum()),
        "baseline": float(baseline), "N_PERM": N_PERM, "N_FOLDS": N_FOLDS,
        "cv_best_auc": cv_best_auc.tolist(),
        "cv_best_C": cv_best_C.tolist(),
        "cv_best_std": cv_best_std.tolist(),
        "perm_mean_auc": perm_mean_auc.tolist(),
        "perm_std_auc": perm_std_auc.tolist(),
        "perm_95_auc": perm_95.tolist(),
        "rand_auc": rand_auc.tolist(),
        "best_step_1indexed": int(s_b + 1), "best_layer": int(l_b),
        "best_cv_auc": float(cv_best_auc[s_b, l_b]),
        "best_perm_mean": float(perm_mean_auc[s_b, l_b]),
        "best_rand_auc": float(rand_auc[s_b, l_b]),
    }, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, mat, title, vmin, vmax in [
        (axes[0, 0], cv_best_auc, f"CV-mean AUC (best C per cell)", 0.4, 1.0),
        (axes[0, 1], perm_mean_auc, "Permutation null mean AUC", 0.4, 1.0),
        (axes[1, 0], rand_auc, "Random Gaussian features AUC", 0.4, 1.0),
        (axes[1, 1], cv_best_auc - perm_mean_auc, "AUC over null (signal)", -0.1, 0.3),
    ]:
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel("layer"); ax.set_ylabel("latent step")
        ax.set_yticks(range(S)); ax.set_yticklabels([str(s+1) for s in range(S)])
        ax.set_title(title, fontsize=10)
        for s in range(S):
            for l in range(L):
                v = mat[s, l]
                if abs(v - (vmin + vmax) / 2) >= (vmax - vmin) * 0.25:
                    ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if v < (vmin + vmax) / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"Faithfulness probe v2 — 5-fold CV + C-sweep + null/random baselines "
                 f"(majority baseline acc = {baseline*100:.1f}%)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PDF, dpi=140)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
