"""Test whether CODI-GPT-2's latent residuals follow the 'Clock' algorithm.

Kantamneni & Tegmark (2024) showed that some LMs encode integers on helices
in residual space: a number n is represented as (cos(2πn/T), sin(2πn/T)) for
a set of periods T. Addition a+b is implemented by adding angles in those
helices (the "Clock"), i.e. angle(a) + angle(b) ≈ angle(a+b) at each period.

Two tests here:

1) HELIX ENCODING.  For each (step, layer) and each period T:
       fit a linear probe from residual → [cos(2π·n/T), sin(2π·n/T)]
       report cross-validated R² (5-fold).
   High R² = the residual encodes n on a helix of period T.

2) CLOCK CLOSURE.  At the best-encoding (step, layer, T) for the operand
   axis:
       compute model-estimated angles for a, b, and gold answer per example.
       check whether  angle(a) ± angle(b) − angle(gold)  is distributed
       around 0 (where the sign depends on operator: + for Addition, − for
       Subtraction).
   If yes, the latent loop is doing addition by angle rotation — the Clock.

Data:
   - SVAMP colon residuals (N=1000) — broad gold-answer helix coverage.
   - vary_numerals (80, all Subtraction) — controlled (a, b, gold).
   - vary_a_2digit / vary_b_2digit (80 each) — single-operand sweeps; tests
     whether one operand has its own helix even when the other is constant.
   - vary_both_2digit (80) — both operands vary across 10–99.

Output: helix_clock_test.{json, pdf}
   PDF pages:
       1. R² heatmap (step, layer) for gold answer on SVAMP, per period.
       2. R² heatmap for operand a, operand b, gold on vary_numerals.
       3. Clock-closure scatter plots at the best cell.
       4. TL;DR summary table.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[3]  # codi-work/
CF_DIR = REPO.parent / "cf-datasets"
COL_DIR = REPO / "visualizations-all" / "gpt2" / "counterfactuals"
SVAMP_COL = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
SVAMP_META = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "helix_clock_test_gsm8k.json"
OUT_PDF = PD / "helix_clock_test_gsm8k.pdf"

PERIODS = [2, 5, 10, 50, 100, 1000]
RIDGE_ALPHA = 1.0
N_FOLDS = 5
SEED = 0


def fourier_target(n, T):
    """Return shape (2,) feature [cos, sin] for value n at period T."""
    return np.stack([np.cos(2 * np.pi * n / T), np.sin(2 * np.pi * n / T)], axis=-1)


def cv_r2(X, Y, alpha=RIDGE_ALPHA, n_folds=N_FOLDS, seed=SEED):
    """5-fold CV R² for a linear ridge probe X → Y (multi-output OK)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y, dtype=np.float64)
    for tr, te in kf.split(X):
        clf = Ridge(alpha=alpha).fit(X[tr], Y[tr])
        preds[te] = clf.predict(X[te])
    ss_res = float(np.sum((Y - preds) ** 2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12), preds


def angles_from_pred(pred_cos_sin):
    """pred_cos_sin shape (N, 2) → angles in [-π, π]."""
    return np.arctan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])


def wrap_angle(a):
    """Wrap to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_svamp():
    acts = torch.load(SVAMP_COL, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(SVAMP_META))
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    keep = ~np.isnan(gold)
    return acts[keep], {"gold": gold[keep]}


def load_cf(name):
    """Load colon acts + meta for a CF set; extract a, b, gold."""
    acts = torch.load(COL_DIR / f"{name}_colon_acts.pt", map_location="cpu",
                      weights_only=True).float().numpy()
    meta = json.load(open(COL_DIR / f"{name}_colon_acts_meta.json"))
    rows = json.load(open(CF_DIR / f"{name}.json"))
    N = acts.shape[0]
    a_arr = np.array([r.get("a", np.nan) for r in rows[:N]], dtype=float)
    b_arr = np.array([r.get("b", np.nan) for r in rows[:N]], dtype=float)
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    # operator: pick first; for vary_numerals it's all Subtraction
    ops = meta.get("types") or [None] * N
    op = ops[0] if ops else None
    return acts, {"a": a_arr, "b": b_arr, "gold": gold, "op": op,
                  "ops": ops}


def run_helix_sweep(acts, target, periods=PERIODS):
    """Run helix R² for a single (N,L,H) tensor over all layers, all periods.

    Returns (n_layers, n_periods) R² matrix.
    """
    N, L, H = acts.shape
    R2 = np.zeros((L, len(periods)))
    for l in range(L):
        X = acts[:, l, :]
        for ti, T in enumerate(periods):
            Y = fourier_target(target, T)
            r2, _ = cv_r2(X, Y)
            R2[l, ti] = r2
    return R2


def run_helix_sweep_latent(acts4d, target, periods=PERIODS):
    """(N, S, L, H) → R² matrix shape (S, L, n_periods)."""
    N, S, L, H = acts4d.shape
    R2 = np.zeros((S, L, len(periods)))
    for s in range(S):
        for l in range(L):
            X = acts4d[:, s, l, :]
            for ti, T in enumerate(periods):
                Y = fourier_target(target, T)
                r2, _ = cv_r2(X, Y)
                R2[s, l, ti] = r2
    return R2


def heatmap_2d(ax, M, title, xticks, yticks, xlabel, ylabel, vmin=0.0, vmax=1.0,
               cmap="viridis"):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xticks))); ax.set_xticklabels(xticks, fontsize=8)
    ax.set_yticks(range(len(yticks))); ax.set_yticklabels(yticks, fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for j in range(M.shape[0]):
        for i in range(M.shape[1]):
            v = M[j, i]
            if v >= 0.4 or v <= 0.05:
                ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.5 else "black")
    return im


def main():
    print("loading SVAMP colon acts...")
    svamp_acts, svamp_meta = load_svamp()
    print(f"  SVAMP shape={svamp_acts.shape}, gold range [{svamp_meta['gold'].min():.0f}, "
          f"{svamp_meta['gold'].max():.0f}]")

    print("loading CF colon acts...")
    cf_data = {}
    for name in ["vary_numerals", "vary_both_2digit", "vary_a_2digit", "vary_b_2digit"]:
        try:
            acts, info = load_cf(name)
            cf_data[name] = (acts, info)
            print(f"  {name}: shape={acts.shape}, "
                  f"a∈[{np.nanmin(info['a']):.0f},{np.nanmax(info['a']):.0f}], "
                  f"b∈[{np.nanmin(info['b']):.0f},{np.nanmax(info['b']):.0f}], "
                  f"gold∈[{np.nanmin(info['gold']):.0f},{np.nanmax(info['gold']):.0f}], "
                  f"op={info['op']}")
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")

    results = {"periods": PERIODS, "svamp": {}, "cf": {}}

    # Sweep 1: SVAMP gold helix R² across layers, periods.
    print("\nSWEEP 1: SVAMP gold-answer helix R² per (layer, period)")
    R2_svamp = run_helix_sweep(svamp_acts, svamp_meta["gold"])
    results["svamp"]["R2_gold"] = R2_svamp.tolist()
    for ti, T in enumerate(PERIODS):
        best_l = int(np.argmax(R2_svamp[:, ti]))
        print(f"  T={T:5d}: best layer={best_l}, R²={R2_svamp[best_l, ti]:.3f}")

    # Sweep 2: CF per-operand helices.
    print("\nSWEEP 2: CF operand and gold helix R²")
    for cf_name, (acts, info) in cf_data.items():
        results["cf"][cf_name] = {"op": info["op"]}
        # Drop NaNs
        keep = ~(np.isnan(info["a"]) | np.isnan(info["b"]) | np.isnan(info["gold"]))
        ac = acts[keep]; a = info["a"][keep]; b = info["b"][keep]; gold = info["gold"][keep]
        print(f"  {cf_name}: N_kept={int(keep.sum())}")
        for tgt_name, tgt_vals in [("a", a), ("b", b), ("gold", gold)]:
            R2 = run_helix_sweep(ac, tgt_vals)
            results["cf"][cf_name][f"R2_{tgt_name}"] = R2.tolist()
            best_T = PERIODS[int(np.argmax(R2.max(axis=0)))]
            best_l = int(np.unravel_index(R2.argmax(), R2.shape)[0])
            print(f"    {tgt_name:4s}: best (layer, T) = ({best_l}, {best_T}), "
                  f"R²={R2.max():.3f}")

    # Clock-closure test on vary_numerals (Subtraction).  Run at every period
    # so we can see at which T the Clock actually closes.  T=1000 is NEAR-LINEAR
    # for our operand range (max ~200), so closure there is largely trivial;
    # T ≤ 100 is where the test is non-trivial (operands wrap modulo T).
    print("\nCLOCK CLOSURE TEST on vary_numerals (Subtraction) — every period")
    results["clock"] = {}
    if "vary_numerals" in cf_data:
        acts, info = cf_data["vary_numerals"]
        keep = ~(np.isnan(info["a"]) | np.isnan(info["b"]) | np.isnan(info["gold"]))
        ac = acts[keep]; a = info["a"][keep]; b = info["b"][keep]; gold = info["gold"][keep]
        R2_a = np.array(results["cf"]["vary_numerals"]["R2_a"])  # (L, P)
        random_resid_std = np.std(np.random.default_rng(0).uniform(-np.pi, np.pi, len(a)))
        per_T = {}
        for ti, T in enumerate(PERIODS):
            # Pick the layer with best R² FOR THIS PERIOD on operand a.
            best_l = int(np.argmax(R2_a[:, ti]))
            X = ac[:, best_l, :]
            _, pred_a = cv_r2(X, fourier_target(a, T))
            _, pred_b = cv_r2(X, fourier_target(b, T))
            _, pred_g = cv_r2(X, fourier_target(gold, T))
            ang_a = angles_from_pred(pred_a)
            ang_b = angles_from_pred(pred_b)
            ang_g = angles_from_pred(pred_g)
            true_a_ang = wrap_angle(2 * np.pi * a / T)
            # How aligned are the PREDICTED angles with the TRUE angles?  This
            # tells us whether the probe is reading something Fourier-like or
            # just a generic linear feature.  For T=1000 (near-linear), even a
            # linear feature gives high cos-sim; for T=10, only a true helix
            # feature would.
            cos_align_a = float(np.mean(np.cos(ang_a - true_a_ang)))

            clock_resid_sub = wrap_angle(ang_a - ang_b - ang_g)
            clock_resid_add = wrap_angle(ang_a + ang_b - ang_g)
            # Operand-wrap fraction: how much of the operand range exceeds T?
            wrap_frac = float(np.mean(np.abs(a) > T)) + float(np.mean(np.abs(b) > T))

            per_T[T] = {
                "best_layer": best_l,
                "R2_a": float(R2_a[best_l, ti]),
                "operand_a_max_over_T": float(a.max() / T),
                "cos_align_predicted_vs_true_angle": cos_align_a,
                "clock_resid_sub_mean_abs": float(np.mean(np.abs(clock_resid_sub))),
                "clock_resid_sub_std": float(np.std(clock_resid_sub)),
                "clock_resid_add_mean_abs": float(np.mean(np.abs(clock_resid_add))),
                "clock_resid_add_std": float(np.std(clock_resid_add)),
                "ang_a": ang_a.tolist(),
                "ang_b": ang_b.tolist(),
                "ang_g": ang_g.tolist(),
                "clock_resid_sub": clock_resid_sub.tolist(),
            }
            print(f"  T={T:5d}: best_L={best_l}, R²(a)={R2_a[best_l, ti]:+.3f}, "
                  f"max(a)/T={a.max()/T:.2f}  "
                  f"cos(ang_a, true)={cos_align_a:+.2f}  "
                  f"Sub|resid|={np.mean(np.abs(clock_resid_sub)):.3f}  "
                  f"Add|resid|={np.mean(np.abs(clock_resid_add)):.3f}  "
                  f"(random≈1.57)")

        results["clock"]["per_T"] = per_T
        results["clock"]["a"] = a.tolist()
        results["clock"]["b"] = b.tolist()
        results["clock"]["gold"] = gold.tolist()
        results["clock"]["random_resid_std"] = random_resid_std

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # PDF
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: SVAMP gold-answer R² heatmap
        fig, ax = plt.subplots(figsize=(10, 5))
        heatmap_2d(ax, R2_svamp.T,  # (P, L)
                   "SVAMP — gold-answer Fourier R² (5-fold CV)",
                   xticks=[str(l) for l in range(R2_svamp.shape[0])],
                   yticks=[f"T={T}" for T in PERIODS],
                   xlabel="layer", ylabel="period")
        fig.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02, label="R²")
        fig.suptitle("Helix encoding: how much of cos(2π·gold/T), sin(2π·gold/T) "
                     "is linear in residual?",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 2-N: CF datasets — per (target, period) heatmaps
        for cf_name in cf_data.keys():
            r = results["cf"][cf_name]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, tgt in zip(axes, ["a", "b", "gold"]):
                R2 = np.array(r[f"R2_{tgt}"])
                heatmap_2d(ax, R2.T,
                           f"{cf_name} — {tgt} helix R²",
                           xticks=[str(l) for l in range(R2.shape[0])],
                           yticks=[f"T={T}" for T in PERIODS],
                           xlabel="layer", ylabel="period")
                fig.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02, label="R²")
            fig.suptitle(f"{cf_name} (op={r['op']})  —  per-operand helix R²",
                         fontsize=11, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # Clock-closure scatter PER PERIOD
        if results.get("clock", {}).get("per_T"):
            per_T = results["clock"]["per_T"]
            # Page A: summary bar chart of |residual| vs T (Sub and Add).
            fig, ax = plt.subplots(figsize=(11, 5))
            Ts = [str(T) for T in PERIODS]
            sub_vals = [per_T[T]["clock_resid_sub_mean_abs"] for T in PERIODS]
            add_vals = [per_T[T]["clock_resid_add_mean_abs"] for T in PERIODS]
            xs = np.arange(len(Ts))
            w = 0.35
            ax.bar(xs - w/2, sub_vals, w, color="#2ca02c",
                   label="Sub:  |angle(a) − angle(b) − angle(gold)|")
            ax.bar(xs + w/2, add_vals, w, color="#d62728",
                   label="Add:  |angle(a) + angle(b) − angle(gold)|  (control)")
            ax.axhline(np.pi / 2, color="black", ls="--", lw=0.7,
                       label="random baseline (π/2)")
            ax.set_xticks(xs); ax.set_xticklabels([f"T={T}" for T in PERIODS])
            ax.set_ylabel("mean |residual|  (rad)")
            ax.set_title("Clock closure on vary_numerals — does angle(a) − angle(b) "
                         "= angle(gold) at each period T?",
                         fontsize=11, fontweight="bold")
            for x, v in zip(xs - w/2, sub_vals):
                ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
            for x, v in zip(xs + w/2, add_vals):
                ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
            for ti, T in enumerate(PERIODS):
                R2 = per_T[T]["R2_a"]
                mx = per_T[T]["operand_a_max_over_T"]
                ax.text(ti, -0.10, f"R²(a)={R2:.2f}\nmax(a)/T={mx:.1f}",
                        ha="center", fontsize=7, transform=ax.get_xaxis_transform(),
                        va="top", color="gray")
            ax.set_ylim(0, max(1.8, max(sub_vals + add_vals) * 1.1))
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)

            # Page B: per-T scatter plot of (angle(a) - angle(b)) vs angle(gold)
            non_triv = [T for T in PERIODS if per_T[T]["operand_a_max_over_T"] >= 0.5]
            for T in non_triv:
                ck = per_T[T]
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                # angle(a) predicted vs true
                ax = axes[0]
                true_a_ang = wrap_angle(2 * np.pi * np.array(results["clock"]["a"]) / T)
                ax.scatter(true_a_ang, ck["ang_a"], s=14, alpha=0.7, c="#1f77b4")
                ax.plot([-np.pi, np.pi], [-np.pi, np.pi], ls="--", color="gray", lw=0.7)
                ax.set_xlabel("true angle(a)"); ax.set_ylabel("predicted angle(a)")
                ax.set_title(f"angle(a) at L{ck['best_layer']}  (T={T}, "
                             f"R²={ck['R2_a']:.2f}, cos-align={ck['cos_align_predicted_vs_true_angle']:.2f})",
                             fontsize=10)
                ax.grid(alpha=0.3)
                # Predicted angle(a)-angle(b) vs predicted angle(gold)
                ax = axes[1]
                clock_pred = wrap_angle(np.array(ck["ang_a"]) - np.array(ck["ang_b"]))
                ax.scatter(ck["ang_g"], clock_pred, s=14, alpha=0.7, c="#2ca02c")
                ax.plot([-np.pi, np.pi], [-np.pi, np.pi], ls="--", color="gray", lw=0.7)
                ax.set_xlabel("predicted angle(gold)")
                ax.set_ylabel("predicted angle(a) − predicted angle(b)")
                ax.set_title(f"Clock closure (Sub)  |resid|={ck['clock_resid_sub_mean_abs']:.3f}",
                             fontsize=10)
                ax.grid(alpha=0.3)
                # Residual histogram
                ax = axes[2]
                ax.hist(ck["clock_resid_sub"], bins=40, color="#2ca02c", alpha=0.8,
                        label=f"Sub: |resid|={ck['clock_resid_sub_mean_abs']:.2f}")
                ax.axvline(0, color="black", lw=0.5)
                ax.set_xlabel("angle(a) − angle(b) − angle(gold)  [rad, wrapped]")
                ax.set_ylabel("count")
                ax.set_title("residual histogram", fontsize=10)
                ax.legend(fontsize=9); ax.grid(alpha=0.3)
                ax.set_xlim(-np.pi, np.pi)
                fig.suptitle(f"Clock test at T={T}  (operand_a / T ≈ {ck['operand_a_max_over_T']:.1f}, "
                             f"so the helix wraps non-trivially)",
                             fontsize=11, fontweight="bold")
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                pdf.savefig(fig, dpi=140); plt.close(fig)

        # TL;DR
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.axis("off")
        ax.set_title("Helix / Clock test summary", fontsize=14, fontweight="bold",
                     loc="left")
        lines = []
        lines.append("SVAMP — gold answer encoded on a helix?")
        for ti, T in enumerate(PERIODS):
            best_l = int(np.argmax(R2_svamp[:, ti]))
            lines.append(f"   T={T:5d}: best layer L{best_l}, R²={R2_svamp[best_l, ti]:.3f}")
        lines.append("")
        lines.append("vary_numerals (80, Subtraction) — operand a helix:")
        if "vary_numerals" in cf_data:
            R2_a = np.array(results["cf"]["vary_numerals"]["R2_a"])
            best = np.unravel_index(R2_a.argmax(), R2_a.shape)
            lines.append(f"   best (layer, T) = (L{best[0]}, T={PERIODS[best[1]]}), "
                         f"R²={R2_a.max():.3f}")
        if results.get("clock", {}).get("per_T"):
            per_T = results["clock"]["per_T"]
            lines.append("")
            lines.append("Clock closure on vary_numerals (Subtraction), per period:")
            lines.append("   T       R²(a)  max(a)/T  Sub|resid|  Add|resid|  random")
            for T in PERIODS:
                ck = per_T[T]
                lines.append(f"   {T:5d}   {ck['R2_a']:+.2f}    {ck['operand_a_max_over_T']:.2f}      "
                             f"{ck['clock_resid_sub_mean_abs']:.3f}      "
                             f"{ck['clock_resid_add_mean_abs']:.3f}     ≈π/2")
            # Verdict: look at the smallest T where the Clock closes well.
            sub_closure_at_nontrivial = [
                (T, per_T[T]["clock_resid_sub_mean_abs"])
                for T in PERIODS if per_T[T]["operand_a_max_over_T"] >= 0.5
            ]
            if sub_closure_at_nontrivial:
                best_T, best_resid = min(sub_closure_at_nontrivial, key=lambda x: x[1])
                lines.append("")
                if best_resid < 0.3:
                    lines.append(f"   Verdict: STRONG Clock evidence — at T={best_T} (non-trivial)")
                    lines.append(f"            |residual| = {best_resid:.3f} rad ≪ π/2")
                elif best_resid < 0.7:
                    lines.append(f"   Verdict: MODERATE Clock evidence — at T={best_T}")
                    lines.append(f"            |residual| = {best_resid:.3f} rad")
                else:
                    lines.append(f"   Verdict: NO non-trivial Clock — closure breaks at T<200")
                    lines.append(f"            best non-trivial |residual| = {best_resid:.3f} rad")
        ax.text(0.02, 0.92, "\n".join(lines), fontsize=11, family="monospace",
                verticalalignment="top")
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
