"""Tighter test of whether step 1->2 moves the residual along the correctness
direction:

For each layer L:
  - Fit a correctness probe (binary logreg) on STEP-2 activations at layer L,
    target = student_correct_at_step_2. Hold out 30% (stratified).
  - For each cohort (wr, rw, all):
      - Compute the probe's P(correct | step-2 residual at L) for each example.
      - Compute cos(step-2_residual, w_correct_at_L) per example.
      - Compute cos(delta, w_correct_at_L) per example (delta = s2 - s1).
  - Report distributions: mean, median, std per cohort.

Also at the best-AUC cell (probe fit on step-2, eval on the test split):
  - Report P(correct) statistics for wr vs rw vs same-correct vs same-wrong.

Outputs: step1to2_correctness_distribution.{json,pdf}
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
FD = PD / "force_decode_per_step.json"
OUT_JSON = PD / "step1to2_correctness_distribution.json"
OUT_PDF = PD / "step1to2_correctness_distribution.pdf"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def main():
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    fd = json.load(open(FD))
    correct = np.array(fd["correct_per_step"])
    s1_correct = correct[0].astype(bool)
    s2_correct = correct[1].astype(bool)
    wr = (~s1_correct) & s2_correct
    rw = s1_correct & (~s2_correct)
    same_right = s1_correct & s2_correct
    same_wrong = (~s1_correct) & (~s2_correct)
    print(f"N={N}  wr={int(wr.sum())}  rw={int(rw.sum())}  "
          f"same_right={int(same_right.sum())}  same_wrong={int(same_wrong.sum())}")
    print(f"  step1 acc={s1_correct.mean()*100:.1f}%  step2 acc={s2_correct.mean()*100:.1f}%")

    # Fit per-layer correctness probe on step-2 residuals, target = s2_correct
    rng = 0
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=rng,
                                      stratify=s2_correct.astype(int))
    y_tr = s2_correct[idx_tr].astype(int)
    y_te = s2_correct[idx_te].astype(int)

    test_auc = np.zeros(L)
    proba_s2 = np.zeros((N, L))      # P(correct) at step 2
    proba_s1 = np.zeros((N, L))      # P(correct) at step 1 (same direction)
    cos_s2 = np.zeros((N, L))        # cos(step-2 residual, correctness_dir)
    cos_s1 = np.zeros((N, L))
    cos_delta = np.zeros((N, L))     # cos(delta, correctness_dir)
    for l in range(L):
        X_s2 = a[:, 1, l, :]
        X_s1 = a[:, 0, l, :]
        sc = StandardScaler().fit(X_s2[idx_tr])
        clf = LogisticRegression(max_iter=4000, C=0.1, solver="lbfgs",
                                 class_weight="balanced")
        clf.fit(sc.transform(X_s2[idx_tr]), y_tr)
        test_auc[l] = roc_auc_score(y_te, clf.predict_proba(sc.transform(X_s2[idx_te]))[:, 1])
        # Predict P(correct) on ALL examples at step 2 (these include train; for
        # cohort means we keep all examples — we're not estimating generalization
        # here, just the probe's confidence projection).
        proba_s2[:, l] = clf.predict_proba(sc.transform(X_s2))[:, 1]
        proba_s1[:, l] = clf.predict_proba(sc.transform(X_s1))[:, 1]
        # Direction in raw space:
        w_raw = unit(clf.coef_[0] / sc.scale_)
        cos_s2[:, l] = np.array([float(np.dot(X_s2[i], w_raw) / (np.linalg.norm(X_s2[i]) + 1e-12))
                                  for i in range(N)])
        cos_s1[:, l] = np.array([float(np.dot(X_s1[i], w_raw) / (np.linalg.norm(X_s1[i]) + 1e-12))
                                  for i in range(N)])
        d = X_s2 - X_s1
        cos_delta[:, l] = np.array([float(np.dot(d[i], w_raw) / (np.linalg.norm(d[i]) + 1e-12))
                                     for i in range(N)])
        if l in (0, 6, 11):
            print(f"  L{l}: test AUC = {test_auc[l]:.3f}")

    l_best = int(test_auc.argmax())
    print(f"\nbest probe layer = L{l_best}, test AUC = {test_auc[l_best]:.3f}")

    def stats(arr, mask):
        x = arr[mask]
        return {"n": int(mask.sum()),
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "std": float(x.std()),
                "p10": float(np.percentile(x, 10)),
                "p90": float(np.percentile(x, 90))}

    cohorts = {"wr": wr, "rw": rw, "same_right": same_right, "same_wrong": same_wrong}
    rec = {"test_auc_per_layer": test_auc.tolist(), "best_layer": l_best,
           "by_layer": {}, "best_layer_summary": {}}
    print("\nProbe P(correct | step-2 residual) per cohort at best layer:")
    for c, m in cohorts.items():
        s2 = stats(proba_s2[:, l_best], m)
        s1 = stats(proba_s1[:, l_best], m)
        print(f"  {c:12s} P_s2 mean={s2['mean']:.3f}  median={s2['median']:.3f}  "
              f"n={s2['n']}    P_s1 mean={s1['mean']:.3f}  median={s1['median']:.3f}")
        rec["best_layer_summary"][c] = {"P_s2": s2, "P_s1": s1,
                                        "cos_s2": stats(cos_s2[:, l_best], m),
                                        "cos_s1": stats(cos_s1[:, l_best], m),
                                        "cos_delta": stats(cos_delta[:, l_best], m)}

    print("\nP(correct) gain from step 1 to step 2, mean per cohort:")
    for c, m in cohorts.items():
        gain = (proba_s2[m, l_best] - proba_s1[m, l_best]).mean()
        print(f"  {c:12s} ΔP(correct) = {gain:+.3f}")

    print("\nPer-layer mean P(correct) at step 2 per cohort:")
    by_layer_mean_s2 = {c: [float(proba_s2[m, l].mean()) for l in range(L)] for c, m in cohorts.items()}
    by_layer_mean_s1 = {c: [float(proba_s1[m, l].mean()) for l in range(L)] for c, m in cohorts.items()}
    rec["by_layer"]["mean_P_s2_per_cohort"] = by_layer_mean_s2
    rec["by_layer"]["mean_P_s1_per_cohort"] = by_layer_mean_s1
    rec["by_layer"]["mean_cos_delta_per_cohort"] = {c: [float(cos_delta[m, l].mean()) for l in range(L)] for c, m in cohorts.items()}
    rec["by_layer"]["mean_cos_s2_per_cohort"] = {c: [float(cos_s2[m, l].mean()) for l in range(L)] for c, m in cohorts.items()}

    OUT_JSON.write_text(json.dumps(rec, indent=2))
    print(f"saved {OUT_JSON}")

    # Plot histograms at best layer
    with PdfPages(OUT_PDF) as pdf:
        # Slide 1: per-layer test AUC + cohort mean P(correct)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].plot(range(L), test_auc, "o-", lw=2, color="#1f77b4")
        axes[0].axhline(0.5, color="gray", ls=":", label="chance")
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("test AUC")
        axes[0].set_title(f"Per-layer correctness probe AUC (best L{l_best} = {test_auc[l_best]:.3f})",
                          fontsize=10)
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
        for c, color in [("wr", "#2ca02c"), ("rw", "#d62728"),
                         ("same_right", "#7f7f7f"), ("same_wrong", "#bcbd22")]:
            axes[1].plot(range(L), by_layer_mean_s2[c], "o-", lw=2, color=color,
                         label=f"{c} (n={int(cohorts[c].sum())})")
        axes[1].axhline(0.5, color="gray", ls=":")
        axes[1].set_xlabel("layer"); axes[1].set_ylabel("mean P(correct | step-2 residual)")
        axes[1].set_title("Cohort mean probe-confidence at step 2", fontsize=10)
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 2: distribution at best layer
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        bins = np.linspace(0, 1, 30)
        for c, color in [("wr", "#2ca02c"), ("rw", "#d62728"),
                         ("same_right", "#7f7f7f"), ("same_wrong", "#bcbd22")]:
            axes[0].hist(proba_s2[cohorts[c], l_best], bins=bins, alpha=0.55,
                         color=color, label=f"{c} (n={int(cohorts[c].sum())})",
                         density=True)
        axes[0].set_xlabel(f"P(correct | step-2 residual) at L{l_best}")
        axes[0].set_ylabel("density")
        axes[0].set_title(f"Probe P(correct) distribution at best layer (L{l_best})",
                          fontsize=10)
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        # Step-1 vs step-2 P(correct) shift, paired
        for c, color in [("wr", "#2ca02c"), ("rw", "#d62728")]:
            mask = cohorts[c]
            axes[1].scatter(proba_s1[mask, l_best], proba_s2[mask, l_best],
                            color=color, alpha=0.7,
                            label=f"{c} (n={int(mask.sum())})", s=18)
        axes[1].plot([0, 1], [0, 1], color="black", lw=0.5, ls="--", label="no change")
        axes[1].set_xlabel("P(correct) at step 1 residual"); axes[1].set_ylabel("P(correct) at step 2 residual")
        axes[1].set_title(f"Per-example P(correct) shift at L{l_best}", fontsize=10)
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3); axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 3: cos_delta distributions
        fig, ax = plt.subplots(figsize=(11, 5))
        bins = np.linspace(-0.3, 0.3, 40)
        for c, color in [("wr", "#2ca02c"), ("rw", "#d62728"),
                         ("same_right", "#7f7f7f"), ("same_wrong", "#bcbd22")]:
            ax.hist(cos_delta[cohorts[c], l_best], bins=bins, alpha=0.55,
                    color=color, label=f"{c} (n={int(cohorts[c].sum())})",
                    density=True)
        ax.set_xlabel(f"cos(step2 − step1 delta, correctness_dir) at L{l_best}")
        ax.set_ylabel("density")
        ax.set_title(f"Per-example cos(delta, correctness_dir) at best layer (L{l_best})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.axvline(0, color="black", lw=0.5)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
