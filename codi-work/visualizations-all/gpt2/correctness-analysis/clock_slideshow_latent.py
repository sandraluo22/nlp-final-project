"""Detailed per-period slideshow for the Clock test on LATENT residuals.

Reads helix_clock_test_latent.json and produces clock_slideshow_latent.pdf with
one detailed page per (CF set × period T). Each page shows:
  - Three "predicted angle vs true angle" scatters for a, b, gold.
  - "predicted angle(a) − predicted angle(b)" vs "predicted angle(gold)"
    closure scatter.
  - Residual histogram for Sub closure and Add closure (control).
  - Stats panel with R², cos-align, closure means.

Plus a TL;DR summary page per CF set comparing all periods.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
IN_JSON = PD / "helix_clock_test_latent.json"
OUT_PDF = PD / "clock_slideshow_latent.pdf"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    d = json.load(open(IN_JSON))
    PERIODS = d["periods"]
    with PdfPages(OUT_PDF) as pdf:
        for cf_name in d["cf"]:
            cf = d["cf"][cf_name]
            ck_all = cf["clock_per_T"]
            a_vals = np.array(cf["a"]); b_vals = np.array(cf["b"]); g_vals = np.array(cf["gold"])

            # ============================================================
            # Summary page per CF set: closure vs T with Sub & Add bars.
            # ============================================================
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            ax = axes[0]
            sub_vals = [ck_all[str(T) if isinstance(list(ck_all.keys())[0], str) else T]["clock_resid_sub_mean_abs"]
                        for T in PERIODS]
            add_vals = [ck_all[str(T) if isinstance(list(ck_all.keys())[0], str) else T]["clock_resid_add_mean_abs"]
                        for T in PERIODS]
            r2_vals  = [ck_all[str(T) if isinstance(list(ck_all.keys())[0], str) else T]["R2_a"]
                        for T in PERIODS]
            xs = np.arange(len(PERIODS)); w = 0.32
            ax.bar(xs - w/2, sub_vals, w, color="#2ca02c", label="Sub closure |a − b − g|")
            ax.bar(xs + w/2, add_vals, w, color="#d62728", label="Add closure |a + b − g| (control)")
            ax.axhline(np.pi / 2, ls="--", color="black", lw=0.7, label="random≈π/2")
            ax.set_xticks(xs); ax.set_xticklabels([f"T={T}" for T in PERIODS])
            ax.set_ylabel("|residual|  (rad)")
            ax.set_title(f"{cf_name}: Clock closure vs period",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(axis="y", alpha=0.3)
            for ti, T in enumerate(PERIODS):
                key = str(T) if isinstance(list(ck_all.keys())[0], str) else T
                c = ck_all[key]
                ax.text(ti, -0.10,
                        f"R²(a)={c['R2_a']:.2f}\n"
                        f"step{c['best_step_1indexed']}, L{c['best_layer']}\n"
                        f"max(a)/T={c['operand_a_max_over_T']:.2f}",
                        ha="center", fontsize=7,
                        transform=ax.get_xaxis_transform(),
                        va="top", color="gray")
            ax.set_ylim(0, 2.0)

            # R²(a) vs T panel
            ax = axes[1]
            ax.bar(xs, r2_vals, color="#1f77b4")
            ax.set_xticks(xs); ax.set_xticklabels([f"T={T}" for T in PERIODS])
            ax.set_ylim(-0.2, 1.0)
            ax.axhline(0, color="black", lw=0.3)
            ax.set_ylabel("R²(a) at best cell")
            ax.set_title(f"{cf_name}: probe quality vs period",
                         fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            for ti, v in enumerate(r2_vals):
                ax.text(ti, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
            fig.suptitle(f"Clock test on CODI-GPT-2 latent residuals: {cf_name}",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

            # ============================================================
            # Per-T detail page
            # ============================================================
            for T in PERIODS:
                key = str(T) if isinstance(list(ck_all.keys())[0], str) else T
                c = ck_all[key]
                ang_a = np.array(c["ang_a"]); ang_b = np.array(c["ang_b"])
                ang_g = np.array(c["ang_g"])
                sub_resid = np.array(c["clock_resid_sub"])
                true_a = wrap(2 * np.pi * a_vals / T)
                true_b = wrap(2 * np.pi * b_vals / T)
                true_g = wrap(2 * np.pi * g_vals / T)

                fig = plt.figure(figsize=(15, 9))
                gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

                # Row 1: predicted vs true for a, b, gold
                for col, (name, true_ang, pred_ang, color) in enumerate([
                    ("operand a", true_a, ang_a, "#1f77b4"),
                    ("operand b", true_b, ang_b, "#ff7f0e"),
                    ("gold",      true_g, ang_g, "#2ca02c"),
                ]):
                    ax = fig.add_subplot(gs[0, col])
                    ax.scatter(true_ang, pred_ang, s=14, alpha=0.7, c=color)
                    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], ls="--", color="gray", lw=0.7)
                    ax.set_xlabel(f"true angle({name})")
                    ax.set_ylabel(f"predicted angle({name})")
                    cos_align = float(np.mean(np.cos(pred_ang - true_ang)))
                    ax.set_title(f"{name} — cos-align={cos_align:+.2f}",
                                 fontsize=10)
                    ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
                    ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
                    ax.grid(alpha=0.3)

                # Row 2 col 0: closure scatter
                ax = fig.add_subplot(gs[1, 0])
                clock_pred = wrap(ang_a - ang_b)
                ax.scatter(ang_g, clock_pred, s=14, alpha=0.7, c="#9467bd")
                ax.plot([-np.pi, np.pi], [-np.pi, np.pi], ls="--", color="gray", lw=0.7)
                ax.set_xlabel("predicted angle(gold)")
                ax.set_ylabel("predicted angle(a) − angle(b)")
                ax.set_title(f"Sub closure  |resid|={c['clock_resid_sub_mean_abs']:.3f}",
                             fontsize=10)
                ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
                ax.grid(alpha=0.3)

                # Row 2 col 1: closure histogram
                ax = fig.add_subplot(gs[1, 1])
                ax.hist(sub_resid, bins=30, color="#2ca02c", alpha=0.7,
                        label=f"Sub: mean|resid|={c['clock_resid_sub_mean_abs']:.3f}")
                # Also show Add closure for comparison (compute fresh)
                add_resid = wrap(ang_a + ang_b - ang_g)
                ax.hist(add_resid, bins=30, color="#d62728", alpha=0.5,
                        label=f"Add: mean|resid|={c['clock_resid_add_mean_abs']:.3f}")
                ax.axvline(0, color="black", lw=0.5)
                ax.set_xlim(-np.pi, np.pi)
                ax.set_xlabel("angle residual  [rad, wrapped]")
                ax.set_ylabel("count")
                ax.set_title("Sub vs Add closure residual", fontsize=10)
                ax.legend(fontsize=8); ax.grid(alpha=0.3)

                # Row 2 col 2: stats panel
                ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
                non_triv = "YES (operands wrap)" if c["operand_a_max_over_T"] >= 1.0 else "NO (near-linear)"
                strength = "STRONG (likely Clock-consistent)" if (
                    c["clock_resid_sub_mean_abs"] < 0.5 and
                    c["operand_a_max_over_T"] >= 1.0
                ) else "WEAK" if c["clock_resid_sub_mean_abs"] >= 1.0 else "MODERATE"
                stats_txt = (
                    f"Period T = {T}\n"
                    f"Best cell:  step={c['best_step_1indexed']}, layer={c['best_layer']}\n"
                    f"R²(a) = {c['R2_a']:+.3f}\n"
                    f"cos-align(pred angle, true angle) = "
                    f"{c['cos_align_predicted_vs_true_angle']:+.3f}\n"
                    f"max(operand)/T = {c['operand_a_max_over_T']:.2f}\n"
                    f"  Non-trivial Clock test? {non_triv}\n"
                    "\n"
                    f"|angle(a) − angle(b) − angle(gold)|  =  "
                    f"{c['clock_resid_sub_mean_abs']:.3f} rad\n"
                    f"|angle(a) + angle(b) − angle(gold)|  =  "
                    f"{c['clock_resid_add_mean_abs']:.3f} rad  (control)\n"
                    f"random baseline ≈ π/2 ≈ 1.57 rad\n"
                    "\n"
                    f"Verdict: {strength}\n"
                    "\n"
                    "Note: this consistency test cannot distinguish\n"
                    "between a true helical Clock geometry and\n"
                    "independent linear encoding of a, b, gold +\n"
                    "arithmetic.  See clock_subspace_rank.py and\n"
                    "clock_causal_rotation.py for that."
                )
                ax.text(0.0, 1.0, stats_txt, fontsize=9, va="top",
                        family="monospace")

                fig.suptitle(f"{cf_name} — Clock test at T={T}",
                             fontsize=13, fontweight="bold")
                pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
