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
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]   # codi-work/
ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
JUDGED = REPO.parent / "cf-datasets" / "gsm8k_judged.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "faithfulness_probe_v2.json"
OUT_PDF = PD / "faithfulness_probe_v2.pdf"

N_PERM = 30          # label permutations per cell
N_FOLDS = 5
C_GRID = [0.01, 0.1, 1.0, 10.0]
SEED = 0


def cv_metrics(X, y, C, n_folds=N_FOLDS, seed=SEED):
    """Return CV-mean (auc, acc, recall_unfaithful, recall_faithful) and stds.

    Recall is per-class on the held-out fold.  recall_0 (unfaithful) and
    recall_1 (faithful) average together to give balanced accuracy.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs, accs, r0s, r1s = [], [], [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=4000, C=C, solver="lbfgs",
                                 class_weight="balanced").fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], yhat))
        r0s.append(recall_score(y[te], yhat, pos_label=0, zero_division=0))
        r1s.append(recall_score(y[te], yhat, pos_label=1, zero_division=0))
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "recall_unfaithful_mean": float(np.mean(r0s)),
        "recall_unfaithful_std": float(np.std(r0s)),
        "recall_faithful_mean": float(np.mean(r1s)),
        "recall_faithful_std": float(np.std(r1s)),
    }


def cv_auc(X, y, C, n_folds=N_FOLDS, seed=SEED):
    """Backwards-compatible wrapper: returns (auc_mean, auc_std)."""
    m = cv_metrics(X, y, C, n_folds=n_folds, seed=seed)
    return m["auc_mean"], m["auc_std"]


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
    cv_best_acc = np.zeros((S, L))
    cv_best_r0 = np.zeros((S, L))     # recall_unfaithful
    cv_best_r1 = np.zeros((S, L))     # recall_faithful
    cv_best_C = np.full((S, L), -1.0)
    cv_best_std = np.zeros((S, L))
    perm_mean_auc = np.zeros((S, L))
    perm_std_auc = np.zeros((S, L))
    perm_95 = np.zeros((S, L))
    perm_mean_acc = np.zeros((S, L))
    perm_mean_r0 = np.zeros((S, L))
    perm_mean_r1 = np.zeros((S, L))
    rand_auc = np.zeros((S, L))
    rand_acc = np.zeros((S, L))
    rand_r0 = np.zeros((S, L))
    rand_r1 = np.zeros((S, L))

    rng_perm = np.random.default_rng(SEED + 1)

    for s in range(S):
        for l in range(L):
            X = a_keep[:, s, l, :]
            # C-sweep with 5-fold CV on the real probe; pick C by AUC
            best = None
            for C in C_GRID:
                m = cv_metrics(X, y, C)
                if best is None or m["auc_mean"] > best["auc_mean"]:
                    best = m; best["C"] = C
            cv_best_auc[s, l] = best["auc_mean"]
            cv_best_acc[s, l] = best["acc_mean"]
            cv_best_r0[s, l] = best["recall_unfaithful_mean"]
            cv_best_r1[s, l] = best["recall_faithful_mean"]
            cv_best_C[s, l] = best["C"]
            cv_best_std[s, l] = best["auc_std"]

            # Permutation null at the best C (AUC + acc + recalls)
            null_aucs, null_accs, null_r0s, null_r1s = [], [], [], []
            for _ in range(N_PERM):
                y_sh = rng_perm.permutation(y)
                m = cv_metrics(X, y_sh, best["C"])
                null_aucs.append(m["auc_mean"])
                null_accs.append(m["acc_mean"])
                null_r0s.append(m["recall_unfaithful_mean"])
                null_r1s.append(m["recall_faithful_mean"])
            perm_mean_auc[s, l] = float(np.mean(null_aucs))
            perm_std_auc[s, l] = float(np.std(null_aucs))
            perm_95[s, l] = float(np.percentile(null_aucs, 95))
            perm_mean_acc[s, l] = float(np.mean(null_accs))
            perm_mean_r0[s, l] = float(np.mean(null_r0s))
            perm_mean_r1[s, l] = float(np.mean(null_r1s))

            # Random Gaussian features baseline (one run, full metrics)
            mr = cv_metrics(rand_feats, y, best["C"])
            rand_auc[s, l] = mr["auc_mean"]
            rand_acc[s, l] = mr["acc_mean"]
            rand_r0[s, l] = mr["recall_unfaithful_mean"]
            rand_r1[s, l] = mr["recall_faithful_mean"]

        print(f"  step {s+1}/{S} done. "
              f"layer-wise best AUC at step {s+1}: {cv_best_auc[s].max():.3f}  "
              f"best acc: {cv_best_acc[s].max():.3f}")

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
        "cv_best_acc": cv_best_acc.tolist(),
        "cv_best_recall_unfaithful": cv_best_r0.tolist(),
        "cv_best_recall_faithful": cv_best_r1.tolist(),
        "cv_best_C": cv_best_C.tolist(),
        "cv_best_std": cv_best_std.tolist(),
        "perm_mean_auc": perm_mean_auc.tolist(),
        "perm_std_auc": perm_std_auc.tolist(),
        "perm_95_auc": perm_95.tolist(),
        "perm_mean_acc": perm_mean_acc.tolist(),
        "perm_mean_recall_unfaithful": perm_mean_r0.tolist(),
        "perm_mean_recall_faithful": perm_mean_r1.tolist(),
        "rand_auc": rand_auc.tolist(),
        "rand_acc": rand_acc.tolist(),
        "rand_recall_unfaithful": rand_r0.tolist(),
        "rand_recall_faithful": rand_r1.tolist(),
        "best_step_1indexed": int(s_b + 1), "best_layer": int(l_b),
        "best_cv_auc": float(cv_best_auc[s_b, l_b]),
        "best_cv_acc": float(cv_best_acc[s_b, l_b]),
        "best_cv_recall_unfaithful": float(cv_best_r0[s_b, l_b]),
        "best_cv_recall_faithful": float(cv_best_r1[s_b, l_b]),
        "best_perm_mean_auc": float(perm_mean_auc[s_b, l_b]),
        "best_perm_mean_acc": float(perm_mean_acc[s_b, l_b]),
        "best_rand_auc": float(rand_auc[s_b, l_b]),
        "best_rand_acc": float(rand_acc[s_b, l_b]),
    }, indent=2))

    # PDF: page 1 = the 4 AUC heatmaps; page 2 = 4 accuracy heatmaps;
    # page 3 = 4 per-class recall heatmaps; page 4 = bar chart at best cell
    # comparing real / permutation / random across all 4 metrics.
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: AUC heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, mat, title, vmin, vmax in [
            (axes[0, 0], cv_best_auc, "CV-mean AUC (best C per cell)", 0.4, 1.0),
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
        fig.suptitle(f"Faithfulness probe v2 — AUC (majority baseline acc = "
                     f"{baseline*100:.1f}%)", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 2: Accuracy heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, mat, title, vmin, vmax in [
            (axes[0, 0], cv_best_acc, "CV-mean accuracy (real probe)", 0.4, 1.0),
            (axes[0, 1], perm_mean_acc, "Permutation null mean accuracy", 0.4, 1.0),
            (axes[1, 0], rand_acc, "Random Gaussian features accuracy", 0.4, 1.0),
            (axes[1, 1], cv_best_acc - perm_mean_acc, "Accuracy over null (signal)", -0.1, 0.3),
        ]:
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma",
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
        fig.suptitle(f"Faithfulness probe v2 — accuracy (majority baseline "
                     f"= {baseline*100:.1f}%)", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 3: per-class recall (real vs random)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, mat, title, vmin, vmax in [
            (axes[0, 0], cv_best_r0, "Real probe — recall(unfaithful=0)", 0.0, 1.0),
            (axes[0, 1], cv_best_r1, "Real probe — recall(faithful=1)", 0.0, 1.0),
            (axes[1, 0], rand_r0, "Random feats — recall(unfaithful=0)", 0.0, 1.0),
            (axes[1, 1], rand_r1, "Random feats — recall(faithful=1)", 0.0, 1.0),
        ]:
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="cividis",
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
        fig.suptitle("Faithfulness probe v2 — per-class recall (5-fold CV)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 4: bar chart at the best (s, l) cell — real vs permutation vs random
        fig, ax = plt.subplots(figsize=(11, 5.5))
        groups = ["AUC", "Accuracy", "Recall (unfaithful)", "Recall (faithful)"]
        real_vals = [cv_best_auc[s_b, l_b], cv_best_acc[s_b, l_b],
                     cv_best_r0[s_b, l_b], cv_best_r1[s_b, l_b]]
        perm_vals = [perm_mean_auc[s_b, l_b], perm_mean_acc[s_b, l_b],
                     perm_mean_r0[s_b, l_b], perm_mean_r1[s_b, l_b]]
        rand_vals = [rand_auc[s_b, l_b], rand_acc[s_b, l_b],
                     rand_r0[s_b, l_b], rand_r1[s_b, l_b]]
        x = np.arange(len(groups))
        w = 0.27
        b1 = ax.bar(x - w, real_vals, w, color="#2ca02c",
                    label=f"real probe (C={cv_best_C[s_b, l_b]:.2f})")
        b2 = ax.bar(x,     perm_vals, w, color="#7f7f7f",
                    label=f"permutation null (N={N_PERM})")
        b3 = ax.bar(x + w, rand_vals, w, color="#d62728",
                    label="random Gaussian features")
        for bs in (b1, b2, b3):
            for b in bs:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                        f"{b.get_height():.2f}", ha="center", fontsize=8)
        ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.6)
        ax.axhline(baseline, color="#1f77b4", lw=0.7, ls="--", alpha=0.6,
                   label=f"majority baseline acc ({baseline*100:.1f}%)")
        ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel("CV-mean metric")
        ax.set_title(f"Best cell: step {s_b+1}, layer {l_b}  —  "
                     f"real vs. permutation vs. random-feature probe",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
