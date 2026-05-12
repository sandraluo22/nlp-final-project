"""Clock test on CODI-GPT-2's LATENT-LOOP residuals (not the ':' position).

Mirrors helix_clock_test.py but operates on the (N, 6 steps, 13 layers, 768)
latent activations. Tests whether the answer-arithmetic happens via the Clock
algorithm WITHIN the latent loop — which is where the actual computation
should live, if it lives anywhere.

Sweep: (step ∈ 1..6) × (layer ∈ 0..12) × (period T ∈ {2, 5, 10, 50, 100, 1000}).
For each cell × period, fit a ridge probe residual → [cos(2π·n/T), sin(2π·n/T)]
for each of n ∈ {a, b, gold}. 5-fold CV R².

Clock closure: at every period (especially the non-trivial ones where operand
max wraps ≥ 2× through T), check |angle(a) − angle(b) − angle(gold)| at the
best (step, layer) cell for that period.

Output: helix_clock_test_latent.{json, pdf}.
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
OUT_JSON = PD / "helix_clock_test_latent_gsm8k.json"
OUT_PDF = PD / "helix_clock_test_latent_gsm8k.pdf"

CF_SETS = ["vary_numerals", "vary_both_2digit", "vary_a_2digit", "vary_b_2digit"]
PERIODS = [2, 5, 10, 50, 100, 1000]
RIDGE_ALPHA = 1.0
N_FOLDS = 5
SEED = 0


def fourier_target(n, T):
    return np.stack([np.cos(2 * np.pi * n / T), np.sin(2 * np.pi * n / T)], axis=-1)


def cv_r2(X, Y, alpha=RIDGE_ALPHA, n_folds=N_FOLDS, seed=SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y, dtype=np.float64)
    for tr, te in kf.split(X):
        clf = Ridge(alpha=alpha).fit(X[tr], Y[tr])
        preds[te] = clf.predict(X[te])
    ss_res = float(np.sum((Y - preds) ** 2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12), preds


def angles_from_pred(pred):
    return np.arctan2(pred[:, 1], pred[:, 0])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_cf_latent(name):
    """Returns (acts (N, S, L, H), info)."""
    acts = torch.load(LAT_DIR / f"{name}_latent_acts.pt", map_location="cpu",
                      weights_only=True).float().numpy()
    rows = json.load(open(CF_DIR / f"{name}.json"))
    N = acts.shape[0]
    a_arr = np.array([r.get("a", np.nan) for r in rows[:N]], dtype=float)
    b_arr = np.array([r.get("b", np.nan) for r in rows[:N]], dtype=float)
    gold = np.array([r.get("answer", np.nan) for r in rows[:N]], dtype=float)
    return acts, {"a": a_arr, "b": b_arr, "gold": gold}


def sweep_helix(acts, target):
    """Return (S, L, P) R² array."""
    N, S, L, H = acts.shape
    R2 = np.zeros((S, L, len(PERIODS)))
    for s in range(S):
        for l in range(L):
            X = acts[:, s, l, :]
            for ti, T in enumerate(PERIODS):
                Y = fourier_target(target, T)
                r2, _ = cv_r2(X, Y)
                R2[s, l, ti] = r2
    return R2


def main():
    print("loading CF latent acts...")
    data = {}
    for name in CF_SETS:
        try:
            acts, info = load_cf_latent(name)
            keep = ~(np.isnan(info["a"]) | np.isnan(info["b"]) | np.isnan(info["gold"]))
            data[name] = (acts[keep], {"a": info["a"][keep], "b": info["b"][keep],
                                        "gold": info["gold"][keep]})
            a, b, g = info["a"], info["b"], info["gold"]
            print(f"  {name}: shape={acts.shape} (kept {int(keep.sum())})  "
                  f"a∈[{int(np.nanmin(a))},{int(np.nanmax(a))}], "
                  f"b∈[{int(np.nanmin(b))},{int(np.nanmax(b))}], "
                  f"gold∈[{int(np.nanmin(g))},{int(np.nanmax(g))}]")
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")

    results = {"periods": PERIODS, "cf": {}}

    for name, (acts, info) in data.items():
        print(f"\n=== {name} ===")
        S, L = acts.shape[1], acts.shape[2]
        out = {}
        for tgt_name, tgt_vals in (("a", info["a"]), ("b", info["b"]),
                                    ("gold", info["gold"])):
            R2 = sweep_helix(acts, tgt_vals)   # (S, L, P)
            out[f"R2_{tgt_name}"] = R2.tolist()
            # Best cell per period
            for ti, T in enumerate(PERIODS):
                s_b, l_b = np.unravel_index(R2[:, :, ti].argmax(), R2[:, :, ti].shape)
                print(f"  R²({tgt_name}, T={T:5d}): best=(step={s_b+1}, L={l_b}) "
                      f"R²={R2[s_b, l_b, ti]:+.3f}    "
                      f"max(n)/T={(np.nanmax(np.abs(tgt_vals))/T):.2f}")
        # Clock closure: at every period, pick best (step, layer) for operand a,
        # compute angle predictions for a, b, gold, check closure.
        R2_a = np.array(out["R2_a"])   # (S, L, P)
        a, b, g = info["a"], info["b"], info["gold"]
        per_T = {}
        for ti, T in enumerate(PERIODS):
            s_b, l_b = np.unravel_index(R2_a[:, :, ti].argmax(), R2_a[:, :, ti].shape)
            X = acts[:, s_b, l_b, :]
            _, pred_a = cv_r2(X, fourier_target(a, T))
            _, pred_b = cv_r2(X, fourier_target(b, T))
            _, pred_g = cv_r2(X, fourier_target(g, T))
            ang_a = angles_from_pred(pred_a)
            ang_b = angles_from_pred(pred_b)
            ang_g = angles_from_pred(pred_g)
            true_ang_a = wrap_angle(2 * np.pi * a / T)
            cos_align = float(np.mean(np.cos(ang_a - true_ang_a)))
            sub_resid = wrap_angle(ang_a - ang_b - ang_g)
            add_resid = wrap_angle(ang_a + ang_b - ang_g)
            per_T[T] = {
                "best_step_1indexed": int(s_b + 1),
                "best_layer": int(l_b),
                "R2_a": float(R2_a[s_b, l_b, ti]),
                "operand_a_max_over_T": float(np.nanmax(np.abs(a)) / T),
                "cos_align_predicted_vs_true_angle": cos_align,
                "clock_resid_sub_mean_abs": float(np.mean(np.abs(sub_resid))),
                "clock_resid_sub_std": float(np.std(sub_resid)),
                "clock_resid_add_mean_abs": float(np.mean(np.abs(add_resid))),
                "ang_a": ang_a.tolist(), "ang_b": ang_b.tolist(),
                "ang_g": ang_g.tolist(),
                "clock_resid_sub": sub_resid.tolist(),
            }
            print(f"  T={T:5d}: best (step={s_b+1}, L={l_b}), R²(a)={R2_a[s_b, l_b, ti]:+.3f}, "
                  f"max(a)/T={(np.nanmax(np.abs(a))/T):.2f}  "
                  f"cos-align={cos_align:+.2f}  "
                  f"Sub|resid|={np.mean(np.abs(sub_resid)):.3f}  "
                  f"Add|resid|={np.mean(np.abs(add_resid)):.3f}  (random≈1.57)")
        out["clock_per_T"] = per_T
        out["a"] = a.tolist(); out["b"] = b.tolist(); out["gold"] = g.tolist()
        results["cf"][name] = out

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # PDF: for each CF set, R² heatmap (step × layer) per period and per target,
    # plus a closure-vs-T bar chart, plus scatter plots at non-trivial T.
    with PdfPages(OUT_PDF) as pdf:
        # TL;DR table
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        ax.set_title("Clock test on LATENT residuals — TL;DR",
                     fontsize=14, fontweight="bold", loc="left")
        lines = ["Periods: " + " | ".join(f"T={T}" for T in PERIODS), ""]
        for name in results["cf"]:
            ck = results["cf"][name]["clock_per_T"]
            lines.append(f"=== {name} (Subtraction) ===")
            lines.append(" T      best(step,L)  R²(a)  max(a)/T  Sub|resid|  Add|resid|")
            for T in PERIODS:
                c = ck[T]
                lines.append(f"  {T:5d}  ({c['best_step_1indexed']},{c['best_layer']:2d})       "
                             f"{c['R2_a']:+.2f}   {c['operand_a_max_over_T']:.2f}      "
                             f"{c['clock_resid_sub_mean_abs']:.3f}      "
                             f"{c['clock_resid_add_mean_abs']:.3f}")
            sub_at_nt = [(T, ck[T]["clock_resid_sub_mean_abs"]) for T in PERIODS
                         if ck[T]["operand_a_max_over_T"] >= 0.5]
            if sub_at_nt:
                best_T, best_resid = min(sub_at_nt, key=lambda x: x[1])
                verdict = (f"  ⇒ best non-trivial closure: T={best_T}, "
                           f"|resid|={best_resid:.3f} rad  "
                           f"({'STRONG' if best_resid < 0.3 else 'MODERATE' if best_resid < 0.7 else 'NONE'})")
                lines.append(verdict)
            lines.append("")
        ax.text(0.0, 0.95, "\n".join(lines), fontsize=8, family="monospace", va="top")
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # Per-CF: R² heatmaps (one slide per target, with one heatmap per period)
        for name in results["cf"]:
            out = results["cf"][name]
            for tgt in ["a", "b", "gold"]:
                R2 = np.array(out[f"R2_{tgt}"])   # (S, L, P)
                S, L, P = R2.shape
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                for ti, T in enumerate(PERIODS):
                    ax = axes[ti // 3, ti % 3]
                    im = ax.imshow(R2[:, :, ti], aspect="auto", origin="lower",
                                   cmap="viridis", vmin=-0.5, vmax=1.0)
                    ax.set_xlabel("layer"); ax.set_ylabel("latent step")
                    ax.set_yticks(range(S)); ax.set_yticklabels([str(s + 1) for s in range(S)])
                    ax.set_xticks(range(L))
                    ax.set_title(f"T={T}, R²={R2[:, :, ti].max():.2f} max",
                                 fontsize=9)
                    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                    for s in range(S):
                        for l in range(L):
                            v = R2[s, l, ti]
                            if v >= 0.4 or v <= -0.3:
                                ax.text(l, s, f"{v:.1f}", ha="center", va="center",
                                        fontsize=5, color="white" if v < 0.5 else "black")
                fig.suptitle(f"{name} — helix R² for operand {tgt}  (latent residual)",
                             fontsize=11, fontweight="bold")
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                pdf.savefig(fig, dpi=140); plt.close(fig)

        # Per-CF: closure-vs-T bar
        for name in results["cf"]:
            ck = results["cf"][name]["clock_per_T"]
            fig, ax = plt.subplots(figsize=(11, 5))
            Ts = [str(T) for T in PERIODS]
            sub_vals = [ck[T]["clock_resid_sub_mean_abs"] for T in PERIODS]
            add_vals = [ck[T]["clock_resid_add_mean_abs"] for T in PERIODS]
            xs = np.arange(len(Ts))
            w = 0.35
            ax.bar(xs - w/2, sub_vals, w, color="#2ca02c",
                   label="Sub:  |angle(a) − angle(b) − angle(gold)|")
            ax.bar(xs + w/2, add_vals, w, color="#d62728",
                   label="Add:  |angle(a) + angle(b) − angle(gold)|  (control)")
            ax.axhline(np.pi / 2, color="black", ls="--", lw=0.7, label="random≈π/2")
            ax.set_xticks(xs); ax.set_xticklabels([f"T={T}" for T in PERIODS])
            ax.set_ylabel("mean |residual|  (rad)")
            ax.set_title(f"{name} — Clock closure on LATENT residual (best (step, L) per T)",
                         fontsize=11, fontweight="bold")
            for ti, T in enumerate(PERIODS):
                c = ck[T]
                ax.text(ti, -0.10, f"R²(a)={c['R2_a']:.2f}\nstep={c['best_step_1indexed']}, L={c['best_layer']}\nmax(a)/T={c['operand_a_max_over_T']:.1f}",
                        ha="center", fontsize=6.5, transform=ax.get_xaxis_transform(),
                        va="top", color="gray")
            ax.set_ylim(0, max(1.8, max(sub_vals + add_vals) * 1.1))
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
