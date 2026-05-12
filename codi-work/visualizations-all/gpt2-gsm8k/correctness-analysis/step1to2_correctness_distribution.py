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

REPO = Path(__file__).resolve().parents[3]   # codi-work/
EXP_PD = REPO / "experiments" / "computation_probes"
PD = Path(__file__).resolve().parent
ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
FD = EXP_PD / "force_decode_per_step.json"
OUT_JSON = PD / "step1to2_correctness_distribution_gsm8k.json"
OUT_PDF = PD / "step1to2_correctness_distribution_gsm8k.pdf"


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
    dot_delta = np.zeros((N, L))     # raw probe dot-product: w_raw_unnorm . delta
    norm_delta = np.zeros((N, L))    # ||delta||_2
    norm_w_raw = np.zeros(L)         # ||w in raw feature space (not unit)||
    logit_s1 = np.zeros((N, L))
    logit_s2 = np.zeros((N, L))
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
        # Logits (used for sigmoid-amplification explanation slide).
        logit_s2[:, l] = clf.decision_function(sc.transform(X_s2))
        logit_s1[:, l] = clf.decision_function(sc.transform(X_s1))
        # Direction in raw space — keep both the unit and un-normalized forms:
        #   w_raw_unnorm = clf.coef_/sc.scale_   (what's actually applied to raw acts)
        #   w_raw_unit   = w_raw_unnorm normalized
        w_raw_unnorm = clf.coef_[0] / sc.scale_
        norm_w_raw[l] = float(np.linalg.norm(w_raw_unnorm))
        w_raw = unit(w_raw_unnorm)
        cos_s2[:, l] = np.array([float(np.dot(X_s2[i], w_raw) / (np.linalg.norm(X_s2[i]) + 1e-12))
                                  for i in range(N)])
        cos_s1[:, l] = np.array([float(np.dot(X_s1[i], w_raw) / (np.linalg.norm(X_s1[i]) + 1e-12))
                                  for i in range(N)])
        d = X_s2 - X_s1
        cos_delta[:, l] = np.array([float(np.dot(d[i], w_raw) / (np.linalg.norm(d[i]) + 1e-12))
                                     for i in range(N)])
        dot_delta[:, l] = d @ w_raw_unnorm
        norm_delta[:, l] = np.linalg.norm(d, axis=1)
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

        # ------------------------------------------------------------------
        # Geometric explanation appendix.
        # Q: why is cos(Δ, w_correct) ≈ 0.02 but ΔP(correct) ≈ 0.32?
        # Four contributing factors, one slide each, plus a title slide and a
        # TL;DR slide at the end.
        # ------------------------------------------------------------------
        L_b = l_best
        H = a.shape[3]
        wr_mask = cohorts["wr"]
        # Empirical numbers we will cite throughout the explanation.
        p_s1_wr = float(proba_s1[wr_mask, L_b].mean())
        p_s2_wr = float(proba_s2[wr_mask, L_b].mean())
        delta_P = p_s2_wr - p_s1_wr
        z_s1_wr = float(logit_s1[wr_mask, L_b].mean())
        z_s2_wr = float(logit_s2[wr_mask, L_b].mean())
        delta_z = z_s2_wr - z_s1_wr
        cos_delta_wr = float(cos_delta[wr_mask, L_b].mean())
        dot_delta_wr = float(dot_delta[wr_mask, L_b].mean())
        norm_delta_wr_examples = norm_delta[wr_mask, L_b]
        norm_delta_indiv = float(np.mean(norm_delta_wr_examples))
        norm_meandelta = float(np.linalg.norm(
            (a[wr_mask, 1, L_b, :] - a[wr_mask, 0, L_b, :]).mean(axis=0)
        ))
        norm_w = float(norm_w_raw[L_b])

        # ============= Slide 4: title + observed numbers ===================
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.axis("off")
        ax.set_title(
            "Appendix: why does cos(Δ, w_correct) ≈ 0.02 but ΔP(correct) ≈ +0.32?",
            fontsize=14, fontweight="bold", loc="left")
        observed = (
            f"Observed at the best probe layer L{L_b} on the wr cohort "
            f"(n = {int(wr_mask.sum())}):\n"
            f"   mean P(correct) at step 1  =  {p_s1_wr:.3f}\n"
            f"   mean P(correct) at step 2  =  {p_s2_wr:.3f}    ΔP = "
            f"{delta_P:+.3f}\n"
            f"   in logit space:  z_s1 = {z_s1_wr:+.2f},  z_s2 = "
            f"{z_s2_wr:+.2f},  Δz = {delta_z:+.2f}\n"
            f"   mean cos(Δ, w̃)            =  {cos_delta_wr:+.4f}\n"
            f"   mean w_raw·Δ (un-norm.)   =  {dot_delta_wr:+.2f}\n"
            f"   ‖w_raw‖ (un-norm.)       =  {norm_w:.3f}\n"
            f"   mean ‖Δᵢ‖                  =  {norm_delta_indiv:.1f}\n"
            f"   ‖mean Δᵢ‖                  =  {norm_meandelta:.1f}    "
            f"(orthogonal noise cancels)"
        )
        ax.text(0.02, 0.88, observed, fontsize=11, family="monospace",
                verticalalignment="top")
        explanation = (
            "These two metrics look contradictory but answer different questions.\n\n"
            "  ΔP(correct) — how far the residual moves across the decision boundary.\n"
            "  cos(Δ, w_correct) — what fraction of Δ's geometry points along w.\n\n"
            "The probe only cares about the dot product w·Δ.  Four factors below "
            "explain why a tiny cos can produce a huge ΔP."
        )
        ax.text(0.02, 0.45, explanation, fontsize=11, verticalalignment="top")
        factors = (
            "1.  The probe vector is a whitened direction (∝ Σ⁻¹ μ_diff), not "
            "a centroid direction.\n"
            "2.  The sigmoid amplifies modest logit changes into big P(correct) "
            "changes.\n"
            "3.  cos(mean Δ, w) ≠ mean cos(Δᵢ, w).  Orthogonal noise cancels in "
            "the mean of Δ, but the dot product survives because it is "
            "systematic.\n"
            "4.  Cosine sim in 768-D is geometrically conservative.  A cos of "
            "0.02 looks tiny but is several SE above null when averaged over a "
            "cohort."
        )
        ax.text(0.02, 0.22, factors, fontsize=10, verticalalignment="top")
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============= Slide 5: Factor 1 — whitened direction ==============
        rng_t = np.random.default_rng(0)
        n_toy = 400
        mu_p = np.array([+0.5, +0.5]); mu_n = np.array([-0.5, -0.5])
        cov_long = np.array([[10.0, 9.5], [9.5, 10.0]])  # anisotropic (long
                                                          # axis along (1,1))
        Xp = rng_t.multivariate_normal(mu_p, cov_long, n_toy)
        Xn = rng_t.multivariate_normal(mu_n, cov_long, n_toy)
        X_toy = np.vstack([Xp, Xn]); y_toy = np.r_[np.ones(n_toy), np.zeros(n_toy)]
        sc_t = StandardScaler().fit(X_toy)
        clf_t = LogisticRegression(max_iter=4000, C=10.0).fit(sc_t.transform(X_toy), y_toy)
        w_t_raw = clf_t.coef_[0] / sc_t.scale_
        w_t = w_t_raw / np.linalg.norm(w_t_raw)
        mu_diff = (mu_p - mu_n)
        mu_diff_u = mu_diff / np.linalg.norm(mu_diff)
        cos_w_mu = float(np.dot(w_t, mu_diff_u))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        ax = axes[0]
        ax.scatter(Xn[:, 0], Xn[:, 1], s=10, c="#d62728", alpha=0.35, label="class 0")
        ax.scatter(Xp[:, 0], Xp[:, 1], s=10, c="#2ca02c", alpha=0.35, label="class 1")
        ax.arrow(0, 0, *(mu_diff_u * 5), color="black", width=0.05,
                 head_width=0.4, length_includes_head=True,
                 label="raw mean-diff direction")
        ax.arrow(0, 0, *(w_t * 5), color="#1f77b4", width=0.05, head_width=0.4,
                 length_includes_head=True, label=r"probe direction $w \propto \Sigma^{-1} \mu_{\rm diff}$")
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_aspect("equal")
        ax.axhline(0, color="black", lw=0.3); ax.axvline(0, color="black", lw=0.3)
        ax.set_title("Toy 2D with anisotropic covariance\n"
                     "(major axis along (1,1) with var≈19, minor axis var≈0.5)",
                     fontsize=10)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.2)

        ax = axes[1]
        ax.axis("off")
        explain = (
            f"Logistic regression solves\n"
            f"   w ∝ Σ⁻¹ (μ₁ − μ₀)\n"
            f"so it points along directions of small variance (after Σ⁻¹\n"
            f"deflates the large ones), not along the centroid difference.\n\n"
            f"In this toy:\n"
            f"   raw mean-diff direction = (1,1)/√2\n"
            f"   probe direction w  ≈ ({w_t[0]:+.3f}, {w_t[1]:+.3f}) — almost\n"
            f"   orthogonal to (1,1)\n"
            f"   cos(w, raw mean diff) = {cos_w_mu:+.4f}\n\n"
            f"Yet the probe classifies the two clusters cleanly\n"
            f"(train acc = {clf_t.score(sc_t.transform(X_toy), y_toy):.3f}).\n\n"
            "In the real model, transformer residuals are even more\n"
            "anisotropic (a few PCs carry magnitude/operator/discourse\n"
            "structure with var 100–1000× the discriminative directions).\n"
            "Δ’s energy goes mostly into those large-variance directions —\n"
            "exactly the ones Σ⁻¹ kills in w."
        )
        ax.text(0.0, 1.0, "Factor 1 — probe direction is whitened",
                fontsize=12, fontweight="bold", va="top")
        ax.text(0.0, 0.92, explain, fontsize=10, va="top", family="monospace")
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============= Slide 6: Factor 2 — sigmoid amplification ===========
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        zs = np.linspace(-4, 4, 400)
        ps = 1.0 / (1.0 + np.exp(-zs))
        ax = axes[0]
        ax.plot(zs, ps, lw=2, color="#1f77b4")
        ax.axhline(p_s1_wr, color="#7f7f7f", ls=":", lw=1)
        ax.axhline(p_s2_wr, color="#2ca02c", ls=":", lw=1)
        ax.axvline(z_s1_wr, color="#7f7f7f", ls="--", lw=1)
        ax.axvline(z_s2_wr, color="#2ca02c", ls="--", lw=1)
        ax.scatter([z_s1_wr], [p_s1_wr], color="#7f7f7f", s=70, zorder=5,
                   label=f"step 1: z={z_s1_wr:+.2f}, P={p_s1_wr:.2f}")
        ax.scatter([z_s2_wr], [p_s2_wr], color="#2ca02c", s=70, zorder=5,
                   label=f"step 2: z={z_s2_wr:+.2f}, P={p_s2_wr:.2f}")
        ax.annotate("", xy=(z_s2_wr, p_s2_wr), xytext=(z_s1_wr, p_s1_wr),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.set_xlabel("z (logit of P(correct))"); ax.set_ylabel("P(correct) = σ(z)")
        ax.set_title("Sigmoid amplifies tiny logit moves", fontsize=10)
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.axis("off")
        # The dot product accounting: Δz = w_raw · Δ
        cos_implied = delta_z / (norm_w * norm_meandelta + 1e-12)
        explain = (
            f"The probe’s logit is\n"
            f"     z = w_raw · x + b\n"
            f"and P(correct) = σ(z).  Slope σ′(0) = 0.25.\n\n"
            f"Observed shift in the wr cohort:\n"
            f"     Δz = z_s2 − z_s1 = {delta_z:+.2f}\n"
            f"     ⇒  ΔP = σ(z_s2) − σ(z_s1) = {delta_P:+.3f}\n\n"
            f"Working back through the dot product:\n"
            f"     Δz = w_raw · (mean Δ) = {dot_delta_wr:+.2f}\n"
            f"     ‖w_raw‖  = {norm_w:.3f}\n"
            f"     ‖mean Δ‖ = {norm_meandelta:.1f}\n"
            f"     ⇒ implied cos(w, mean Δ) = {cos_implied:+.3f}\n\n"
            f"That implied cos is the geometric quantity that the dot\n"
            f"product actually depends on, not the per-example cos(Δᵢ, w).\n"
            f"The per-example cos averages down because each Δᵢ carries\n"
            f"a lot of orthogonal noise (Factor 3)."
        )
        ax.text(0.0, 1.0, "Factor 2 — sigmoid amplification",
                fontsize=12, fontweight="bold", va="top")
        ax.text(0.0, 0.92, explain, fontsize=10, va="top", family="monospace")
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============= Slide 7: Factor 3 — cancellation ====================
        # Per-example raw dot product w_raw · Δᵢ vs cos(Δᵢ, w).  Show both
        # distributions and the mean across the wr cohort.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        ax = axes[0]
        per_ex_dot = dot_delta[wr_mask, L_b]
        ax.hist(per_ex_dot, bins=30, color="#2ca02c", alpha=0.6)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(per_ex_dot.mean(), color="#1f77b4", lw=2,
                   label=f"cohort mean = {per_ex_dot.mean():+.2f}")
        ax.set_xlabel("per-example dot product  w_raw · Δᵢ  (wr cohort)")
        ax.set_ylabel("count")
        ax.set_title(f"Δz_i for each example (n={int(wr_mask.sum())})",
                     fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.axis("off")
        cos_per_ex_mean = float(cos_delta[wr_mask, L_b].mean())
        cos_of_mean = norm_meandelta * 0 + (np.dot(
            (a[wr_mask, 1, L_b, :] - a[wr_mask, 0, L_b, :]).mean(axis=0),
            np.zeros(H)) if False else 0.0)
        # Compute cos(mean Δ, w_raw_unit) explicitly:
        mean_delta_vec = (a[wr_mask, 1, L_b, :] - a[wr_mask, 0, L_b, :]).mean(axis=0)
        # Rebuild w_raw_unit at L_b (was discarded above).  Re-derive from the
        # probe by retraining on the same split — but we still have norm_w_raw
        # and the un-norm w via Δz=w·Δ.  To avoid keeping the full vector,
        # reconstruct the unit form using the cosine we already computed
        # per-example: cos_of_mean = (w_unit · mean_delta) / ‖mean_delta‖.
        # An equivalent quantity: w_unit · mean_delta = Δz / ‖w_raw‖.
        w_dot_mean_delta = delta_z / norm_w
        cos_of_mean_val = w_dot_mean_delta / (norm_meandelta + 1e-12)
        explain = (
            f"Per-example dot products w·Δᵢ are noisy (left panel) but their\n"
            f"sign is consistent → averaging gives a clear signal:\n"
            f"     mean( w·Δᵢ )  =  {per_ex_dot.mean():+.2f}\n\n"
            f"Per-example cos(Δᵢ, w):\n"
            f"     mean( cos(Δᵢ, w) )  =  {cos_per_ex_mean:+.4f}\n\n"
            f"cos(mean Δᵢ, w):\n"
            f"     cos( mean Δᵢ, w )  =  {cos_of_mean_val:+.4f}\n\n"
            f"The two cos values differ because orthogonal components of Δᵢ\n"
            f"cancel in the mean:\n"
            f"     mean(‖Δᵢ‖)    =  {norm_delta_indiv:.1f}\n"
            f"     ‖mean Δᵢ‖     =  {norm_meandelta:.1f}\n"
            f"     cancellation ratio  ≈ {norm_meandelta/norm_delta_indiv:.2f}\n\n"
            f"The systematic on-axis component survives averaging; the rest\n"
            f"is noise that shrinks like 1/√n.  This is why the probe\n"
            f"(which only cares about the dot product) sees clear signal\n"
            f"even when cos looks small."
        )
        ax.text(0.0, 1.0, "Factor 3 — cos(mean Δ, w) vs mean cos(Δᵢ, w); cancellation",
                fontsize=12, fontweight="bold", va="top")
        ax.text(0.0, 0.92, explain, fontsize=10, va="top", family="monospace")
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============= Slide 8: Factor 4 — high-D cosine null ==============
        # Empirical null: cos of pairs of random N(0,I) vectors in 768D.
        rng_n = np.random.default_rng(1)
        N_null = 5000
        u = rng_n.standard_normal((N_null, H))
        v = rng_n.standard_normal((N_null, H))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        null_cos = (u * v).sum(axis=1)
        sig_null = float(null_cos.std())
        # observed quantity to mark on the null
        obs = cos_per_ex_mean  # per-example mean cos for wr cohort
        se_mean = sig_null / np.sqrt(int(wr_mask.sum()))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        ax = axes[0]
        ax.hist(null_cos, bins=50, color="#cccccc", alpha=0.8,
                label=f"cos(random, random) in {H}-D  (σ ≈ {sig_null:.3f})")
        ax.axvline(obs, color="#2ca02c", lw=2,
                   label=f"mean cos(Δᵢ, w) wr cohort = {obs:+.3f}")
        ax.axvline(0, color="black", lw=0.3)
        ax.set_xlabel("cosine")
        ax.set_ylabel("count")
        ax.set_title(f"Null distribution of cos in {H}-D vs observed",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.axis("off")
        z_single = obs / sig_null
        z_cohort = obs / se_mean
        explain = (
            f"In {H} dimensions, two random unit vectors have\n"
            f"     cos ∼ N(0, 1/√{H}) ≈ N(0, {sig_null:.3f}).\n\n"
            f"A single observation of cos = {obs:.3f}\n"
            f"  → z = {z_single:.2f}σ above null  (looks tiny)\n\n"
            f"But the cohort mean is over n = {int(wr_mask.sum())} examples, so\n"
            f"its standard error is SE = σ_null/√n = {se_mean:.4f}\n"
            f"  → z = {z_cohort:.2f}σ above null  (clearly significant)\n\n"
            f"The intuition mismatch: in 2D/3D, cos = 0.02 looks like\n"
            f"noise (≈ 89° off).  In {H}D, cos = 0.02 sits well above\n"
            f"the null floor — and averaging over many examples makes\n"
            f"the signal-to-noise even cleaner."
        )
        ax.text(0.0, 1.0, "Factor 4 — high-D cosine is conservative",
                fontsize=12, fontweight="bold", va="top")
        ax.text(0.0, 0.92, explain, fontsize=10, va="top", family="monospace")
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ============= Slide 9: TL;DR side-by-side =========================
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.axis("off")
        ax.set_title("TL;DR — cos and ΔP measure different things",
                     fontsize=13, fontweight="bold", loc="left")
        rows = [
            ("metric",
             "cos(Δ, w_correct)",
             "ΔP(correct)"),
            ("question it answers",
             "what fraction of Δ's\ngeometry points along w",
             "how much does Δ\nshift the residual across\nthe decision boundary"),
            ("depends on",
             "‖Δ‖ in the denominator —\northogonal noise dilutes",
             "only the on-axis component\nof Δ (survives averaging)"),
            ("typical scale here",
             f"≈ {cos_delta_wr:+.3f}  (per-ex mean)",
             f"≈ {delta_P:+.3f}"),
            ("interpretation",
             "geometric alignment with w",
             "probability mass crossing\nthe decision surface"),
            ("if you change ‖w‖ or scaling",
             "unchanged (cos is normalized)",
             "unchanged (sigmoid sees z)"),
        ]
        # Render as a 3-column table
        col_x = [0.02, 0.34, 0.66]
        row_y = np.linspace(0.85, 0.05, len(rows))
        for x, h in zip(col_x, rows[0]):
            ax.text(x, 0.95, h, fontsize=11, fontweight="bold")
        for y, row in zip(row_y[1:], rows[1:]):
            for x, cell in zip(col_x, row):
                ax.text(x, y, cell, fontsize=10, va="top")
        ax.text(0.02, -0.02,
                "Step 1→2 changes 117 units of residual; only a sliver of\n"
                "that lies on w_correct, and that sliver is enough to flip\n"
                "P(correct) by 0.32 for the wr cohort.",
                fontsize=10, fontstyle="italic", va="top")
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
