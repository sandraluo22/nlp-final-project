"""Out-of-distribution extrapolation test for the Clock encoding.

If a, b are encoded on a TRUE helix at period T, then a probe trained on
operands in one range should generalize to operands in another (as long as
both ranges sample the circle reasonably). If a, b are encoded LINEARLY in
n, the probe still extrapolates well over short distances but its predicted
angles get arbitrarily far from the true wrapped angle once n exceeds the
training range by more than T/2.

This script trains the cos/sin probe on the LOW half of operand values and
tests on the HIGH half. We compare:
  - In-distribution R² (5-fold CV on the full data)
  - Out-of-distribution R² (train low → test high)
  - Cos-alignment of predicted vs true wrapped angle on the test split

A drop in cos-alignment from in- to out-of-distribution is a sign that the
encoding is NOT genuinely periodic — the probe fit a smooth-in-n feature
that happens to look like cos(2πn/T) within the training range only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[3]
CF_DIR = REPO.parent / "cf-datasets"
LAT_DIR = REPO / "visualizations-all" / "gpt2" / "counterfactuals"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "clock_extrapolation_gsm8k.json"
OUT_PDF = PD / "clock_extrapolation_gsm8k.pdf"

CF_SETS = ["vary_numerals", "vary_both_2digit"]
PERIODS = [5, 10, 50, 100]   # the non-trivial periods
RIDGE_ALPHA = 1.0
SEED = 0


def fourier_target(n, T):
    return np.stack([np.cos(2 * np.pi * n / T), np.sin(2 * np.pi * n / T)], axis=-1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def angles_from_pred(p):
    return np.arctan2(p[:, 1], p[:, 0])


def cv_r2(X, Y, alpha=RIDGE_ALPHA, n_folds=5, seed=SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y, dtype=np.float64)
    for tr, te in kf.split(X):
        clf = Ridge(alpha=alpha).fit(X[tr], Y[tr])
        preds[te] = clf.predict(X[te])
    ss_res = float(np.sum((Y - preds) ** 2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12), preds


def fit_test(X_tr, Y_tr, X_te, Y_te, alpha=RIDGE_ALPHA):
    """Fit on train split, evaluate on test split."""
    clf = Ridge(alpha=alpha).fit(X_tr, Y_tr)
    pred_tr = clf.predict(X_tr)
    pred_te = clf.predict(X_te)
    r2_tr = 1.0 - float(np.sum((Y_tr - pred_tr) ** 2)) / max(float(np.sum((Y_tr - Y_tr.mean(axis=0)) ** 2)), 1e-12)
    r2_te = 1.0 - float(np.sum((Y_te - pred_te) ** 2)) / max(float(np.sum((Y_te - Y_te.mean(axis=0)) ** 2)), 1e-12)
    return r2_tr, r2_te, pred_te


def load_cf(name):
    acts = torch.load(LAT_DIR / f"{name}_latent_acts.pt", map_location="cpu",
                      weights_only=True).float().numpy()
    rows = json.load(open(CF_DIR / f"{name}.json"))
    N = acts.shape[0]
    a = np.array([r.get("a", np.nan) for r in rows[:N]], dtype=float)
    b = np.array([r.get("b", np.nan) for r in rows[:N]], dtype=float)
    gold = np.array([r.get("answer", np.nan) for r in rows[:N]], dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b) | np.isnan(gold))
    return acts[keep], a[keep], b[keep], gold[keep]


def main():
    # We already know the best cells from helix_clock_test_latent_gsm8k.json.
    helix = json.load(open(PD / "helix_clock_test_latent_gsm8k.json"))
    results = {}
    for cf_name in CF_SETS:
        acts, a, b, gold = load_cf(cf_name)
        N, S, L, H = acts.shape
        ck_per_T = helix["cf"][cf_name]["clock_per_T"]
        results[cf_name] = {"periods": PERIODS, "by_T": {}}
        print(f"\n=== {cf_name}: N={N}, a∈[{a.min():.0f},{a.max():.0f}] ===")
        for T in PERIODS:
            # Pull best cell coords for this period from the prior sweep.
            key = str(T) if isinstance(list(ck_per_T.keys())[0], str) else T
            best_s = ck_per_T[key]["best_step_1indexed"] - 1
            best_l = ck_per_T[key]["best_layer"]
            X = acts[:, best_s, best_l, :]   # (N, H)

            # Split: train on LOWER half of operand a, test on UPPER half.
            median_a = float(np.median(a))
            tr_mask = a <= median_a
            te_mask = a > median_a
            if tr_mask.sum() < 10 or te_mask.sum() < 10:
                print(f"  T={T}: skip — split too uneven")
                continue
            Y_a = fourier_target(a, T)
            Y_b = fourier_target(b, T)
            Y_g = fourier_target(gold, T)
            # In-distribution: 5-fold CV on the FULL data (same as helix test).
            r2_a_id, _ = cv_r2(X, Y_a)
            r2_b_id, _ = cv_r2(X, Y_b)
            r2_g_id, _ = cv_r2(X, Y_g)
            # OOD: train low, test high.
            r2_a_tr, r2_a_te, pred_a_te = fit_test(X[tr_mask], Y_a[tr_mask],
                                                    X[te_mask], Y_a[te_mask])
            r2_b_tr, r2_b_te, pred_b_te = fit_test(X[tr_mask], Y_b[tr_mask],
                                                    X[te_mask], Y_b[te_mask])
            r2_g_tr, r2_g_te, pred_g_te = fit_test(X[tr_mask], Y_g[tr_mask],
                                                    X[te_mask], Y_g[te_mask])
            # cos-align on test split (do predicted angles match true wrapped angles?)
            true_a_te = wrap(2 * np.pi * a[te_mask] / T)
            true_b_te = wrap(2 * np.pi * b[te_mask] / T)
            true_g_te = wrap(2 * np.pi * gold[te_mask] / T)
            pred_a_ang = angles_from_pred(pred_a_te)
            pred_b_ang = angles_from_pred(pred_b_te)
            pred_g_ang = angles_from_pred(pred_g_te)
            cos_a = float(np.mean(np.cos(pred_a_ang - true_a_te)))
            cos_b = float(np.mean(np.cos(pred_b_ang - true_b_te)))
            cos_g = float(np.mean(np.cos(pred_g_ang - true_g_te)))
            # Closure on OOD test split
            sub_resid = wrap(pred_a_ang - pred_b_ang - pred_g_ang)
            results[cf_name]["by_T"][T] = {
                "best_step_1indexed": best_s + 1, "best_layer": best_l,
                "median_a_split": median_a, "N_train": int(tr_mask.sum()),
                "N_test": int(te_mask.sum()),
                "in_dist_R2": {"a": r2_a_id, "b": r2_b_id, "gold": r2_g_id},
                "ood_R2_train": {"a": r2_a_tr, "b": r2_b_tr, "gold": r2_g_tr},
                "ood_R2_test":  {"a": r2_a_te, "b": r2_b_te, "gold": r2_g_te},
                "ood_cos_align": {"a": cos_a, "b": cos_b, "gold": cos_g},
                "ood_sub_closure_mean_abs": float(np.mean(np.abs(sub_resid))),
                "a_train_range": (float(a[tr_mask].min()), float(a[tr_mask].max())),
                "a_test_range": (float(a[te_mask].min()), float(a[te_mask].max())),
            }
            r = results[cf_name]["by_T"][T]
            print(f"  T={T:4d}  step={best_s+1} L{best_l:2d}  "
                  f"train a∈[{r['a_train_range'][0]:.0f},{r['a_train_range'][1]:.0f}]  "
                  f"test a∈[{r['a_test_range'][0]:.0f},{r['a_test_range'][1]:.0f}]")
            print(f"        in-dist R²(a)={r2_a_id:+.2f}   OOD R²(a) train={r2_a_tr:+.2f} test={r2_a_te:+.2f}")
            print(f"        OOD cos-align  a={cos_a:+.2f}  b={cos_b:+.2f}  gold={cos_g:+.2f}")
            print(f"        OOD Sub closure |resid|={r['ood_sub_closure_mean_abs']:.3f} rad")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # PDF: per CF set, a bar chart comparing in-dist vs OOD R² and cos-align.
    with PdfPages(OUT_PDF) as pdf:
        for cf_name, r in results.items():
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            Ts = list(r["by_T"].keys())
            xs = np.arange(len(Ts))
            # Panel 1: R² in-dist vs OOD for operand a
            id_r2 = [r["by_T"][T]["in_dist_R2"]["a"] for T in Ts]
            tr_r2 = [r["by_T"][T]["ood_R2_train"]["a"] for T in Ts]
            te_r2 = [r["by_T"][T]["ood_R2_test"]["a"]  for T in Ts]
            w = 0.27
            ax = axes[0]
            ax.bar(xs - w, id_r2, w, color="#1f77b4", label="in-dist (5-fold CV)")
            ax.bar(xs,     tr_r2, w, color="#2ca02c", label="OOD train (low a)")
            ax.bar(xs + w, te_r2, w, color="#d62728", label="OOD test (high a)")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(xs); ax.set_xticklabels([f"T={T}" for T in Ts])
            ax.set_ylim(-1.0, 1.0); ax.set_ylabel("R²(a)")
            ax.set_title(f"{cf_name} — operand-a probe R², OOD extrapolation",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(axis="y", alpha=0.3)
            for ti, v in enumerate(te_r2):
                ax.text(ti + w, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

            # Panel 2: cos-align on OOD test split
            cos_a = [r["by_T"][T]["ood_cos_align"]["a"] for T in Ts]
            cos_b = [r["by_T"][T]["ood_cos_align"]["b"] for T in Ts]
            cos_g = [r["by_T"][T]["ood_cos_align"]["gold"] for T in Ts]
            ax = axes[1]
            ax.bar(xs - w, cos_a, w, color="#1f77b4", label="a")
            ax.bar(xs,     cos_b, w, color="#ff7f0e", label="b")
            ax.bar(xs + w, cos_g, w, color="#2ca02c", label="gold")
            ax.axhline(0, color="black", lw=0.5)
            ax.axhline(1, color="gray", ls=":", lw=0.5, label="perfect")
            ax.set_xticks(xs); ax.set_xticklabels([f"T={T}" for T in Ts])
            ax.set_ylim(-1.0, 1.1); ax.set_ylabel("OOD cos-align of predicted vs true angle")
            ax.set_title(f"{cf_name} — angle alignment on held-out high-a test set",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(axis="y", alpha=0.3)
            for ti, v in enumerate(cos_a):
                ax.text(ti - w, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
            fig.suptitle(f"Clock extrapolation — train on low-a, test on high-a "
                         f"(if encoding is truly helical, high cos-align should survive)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
