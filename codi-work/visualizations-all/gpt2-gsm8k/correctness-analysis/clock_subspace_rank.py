"""Subspace-rank test for shared Clock geometry.

If a, b, gold live on the SAME 2D helix at period T, a single 2D subspace
in the residual should suffice to predict all three of their cos/sin
features (6 outputs total). If a, b, gold are encoded independently, you
need ~6 dimensions.

This script fits reduced-rank regression: predict the stacked
[cos(a), sin(a), cos(b), sin(b), cos(g), sin(g)] from the residual, with
rank constraints K ∈ {1, 2, 3, 4, 5, 6}. R²-vs-K shows how much linear
structure can be shared.

Also computes the PRINCIPAL ANGLES between the 2D subspaces fit separately
to a vs b. Small principal angles → shared circle (Clock). Large angles
→ separate dimensions (linear).
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
OUT_JSON = PD / "clock_subspace_rank_gsm8k.json"
OUT_PDF = PD / "clock_subspace_rank_gsm8k.pdf"

CF_SETS = ["vary_numerals", "vary_both_2digit"]
PERIODS = [5, 10, 50, 100]
SEED = 0


def fourier_target(n, T):
    return np.stack([np.cos(2 * np.pi * n / T), np.sin(2 * np.pi * n / T)], axis=-1)


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


def reduced_rank_r2(X, Y, K, alpha=1.0, n_folds=5, seed=SEED):
    """Cross-validated R² for reduced-rank regression of Y on X with rank K.

    Uses ridge to fit the full-rank coefficient B (D, P), then truncates B to
    rank K via SVD. Reports per-output R² averaged over folds (mean) and the
    aggregate R² across all outputs.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    P = Y.shape[1]
    preds = np.zeros_like(Y, dtype=np.float64)
    for tr, te in kf.split(X):
        # Center
        x_mean = X[tr].mean(axis=0); y_mean = Y[tr].mean(axis=0)
        Xc = X[tr] - x_mean; Yc = Y[tr] - y_mean
        clf = Ridge(alpha=alpha, fit_intercept=False).fit(Xc, Yc)
        B = clf.coef_.T   # (D, P)
        # Reduced-rank: SVD of (Xc B) - take top K components.
        # Equivalently: B_K = B U[:K] U[:K]^T where U comes from SVD of Yhat_train.
        Yhat_train = Xc @ B
        U, S, Vt = np.linalg.svd(Yhat_train, full_matrices=False)  # U: (Ntr, P) S: (P,) Vt: (P, P)
        # Project predictions onto top-K right singular vectors.
        if K >= P: V_K = Vt
        else:      V_K = Vt[:K]
        proj = V_K.T @ V_K   # (P, P) — projects Y onto top-K subspace
        # Predictions on test fold
        Xte_c = X[te] - x_mean
        Yhat_te = Xte_c @ B
        Yhat_te = Yhat_te @ proj
        preds[te] = Yhat_te + y_mean
    ss_res = float(np.sum((Y - preds) ** 2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    aggregate_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    per_output_r2 = []
    for p in range(P):
        ss_res_p = float(np.sum((Y[:, p] - preds[:, p]) ** 2))
        ss_tot_p = float(np.sum((Y[:, p] - Y[:, p].mean()) ** 2))
        per_output_r2.append(1.0 - ss_res_p / max(ss_tot_p, 1e-12))
    return aggregate_r2, per_output_r2


def principal_angles(W1, W2):
    """Principal angles between column spans of W1, W2 (both (D, K)).
    Returns array of K cosines (sorted descending — closest pair first)."""
    Q1, _ = np.linalg.qr(W1)
    Q2, _ = np.linalg.qr(W2)
    _, sigs, _ = np.linalg.svd(Q1.T @ Q2, full_matrices=False)
    return sigs  # cosines of principal angles, in [0, 1]


def main():
    helix = json.load(open(PD / "helix_clock_test_latent_gsm8k.json"))
    results = {}
    for cf_name in CF_SETS:
        acts, a, b, gold = load_cf(cf_name)
        ck_per_T = helix["cf"][cf_name]["clock_per_T"]
        results[cf_name] = {"by_T": {}}
        print(f"\n=== {cf_name}: N={acts.shape[0]} ===")
        for T in PERIODS:
            key = str(T) if isinstance(list(ck_per_T.keys())[0], str) else T
            best_s = ck_per_T[key]["best_step_1indexed"] - 1
            best_l = ck_per_T[key]["best_layer"]
            X = acts[:, best_s, best_l, :]   # (N, 768)

            # Stack all 6 outputs (cos/sin × {a, b, gold}).
            Y_full = np.hstack([fourier_target(a, T),
                                fourier_target(b, T),
                                fourier_target(gold, T)])  # (N, 6)

            r2_by_K = {}
            r2_per_output_by_K = {}
            for K in range(1, 7):
                r2, per_p = reduced_rank_r2(X, Y_full, K)
                r2_by_K[K] = r2
                r2_per_output_by_K[K] = per_p

            # Fit separate 2D probes for a, b, gold (full rank within each)
            # and compute principal angles between subspaces.
            clf_a = Ridge(alpha=1.0).fit(X, fourier_target(a, T))
            clf_b = Ridge(alpha=1.0).fit(X, fourier_target(b, T))
            clf_g = Ridge(alpha=1.0).fit(X, fourier_target(gold, T))
            W_a = clf_a.coef_.T   # (768, 2)
            W_b = clf_b.coef_.T
            W_g = clf_g.coef_.T

            pa_ab = principal_angles(W_a, W_b)  # cosines
            pa_ag = principal_angles(W_a, W_g)
            pa_bg = principal_angles(W_b, W_g)

            results[cf_name]["by_T"][T] = {
                "best_step_1indexed": best_s + 1, "best_layer": best_l,
                "r2_by_rank": r2_by_K,
                "r2_per_output_by_rank": r2_per_output_by_K,
                "principal_angle_cosines": {
                    "ab": pa_ab.tolist(),
                    "ag": pa_ag.tolist(),
                    "bg": pa_bg.tolist(),
                },
            }
            print(f"  T={T:4d}  step={best_s+1} L{best_l:2d}")
            for K in range(1, 7):
                print(f"    rank-{K}  agg R²={r2_by_K[K]:+.3f}  "
                      f"per-output: {[f'{v:.2f}' for v in r2_per_output_by_K[K]]}")
            print(f"    principal-angle cosines (a vs b): {pa_ab.round(3)}  "
                  f"(1=aligned, 0=orthogonal)")
            print(f"    principal-angle cosines (a vs gold): {pa_ag.round(3)}")
            print(f"    principal-angle cosines (b vs gold): {pa_bg.round(3)}")

    OUT_JSON.write_text(json.dumps(results, indent=2,
                                    default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else list(v) if isinstance(v, np.ndarray) else v))
    print(f"\nsaved {OUT_JSON}")

    # PDF
    with PdfPages(OUT_PDF) as pdf:
        for cf_name, r in results.items():
            Ts = list(r["by_T"].keys())
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            ax = axes[0]
            ks = list(range(1, 7))
            for T in Ts:
                vals = [r["by_T"][T]["r2_by_rank"][K] for K in ks]
                ax.plot(ks, vals, "o-", lw=2, label=f"T={T}")
            ax.set_xlabel("rank K of fit"); ax.set_ylabel("aggregate R² for [cos(a),sin(a),cos(b),sin(b),cos(g),sin(g)]")
            ax.set_title(f"{cf_name} — reduced-rank R²\n"
                         "(if Clock: K=2 should suffice; if linear: need K=6)",
                         fontsize=10, fontweight="bold")
            ax.set_xticks(ks); ax.set_ylim(-0.2, 1.0); ax.axhline(0, color="black", lw=0.3)
            ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

            # Panel 2: principal-angle cosines (avg) per T
            ax = axes[1]
            cats = ["a vs b", "a vs gold", "b vs gold"]
            xs = np.arange(len(Ts))
            w = 0.27
            for ci, c in enumerate(cats):
                key = c.split(" vs ")
                key_letter = ("ab" if c == "a vs b" else
                              "ag" if c == "a vs gold" else "bg")
                # Use max cosine of the two principal angles (rank-2 subspaces give 2 cosines).
                vals = [max(r["by_T"][T]["principal_angle_cosines"][key_letter])
                        for T in Ts]
                ax.bar(xs + (ci - 1) * w, vals, w, label=c)
            ax.set_xticks(xs); ax.set_xticklabels([f"T={T}" for T in Ts])
            ax.set_ylim(0, 1.05)
            ax.axhline(1, color="black", lw=0.3, ls=":")
            ax.set_ylabel("max principal-angle cosine\n(1=subspaces aligned, 0=orthogonal)")
            ax.set_title(f"{cf_name} — subspace alignment between a, b, gold probes",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(axis="y", alpha=0.3)
            fig.suptitle(f"Clock subspace-rank test on LATENT residual",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
