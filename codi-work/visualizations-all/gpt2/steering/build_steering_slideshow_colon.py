"""Combined steering meta-deck at the ':' residual.

Aggregates the colon-position steering JSONs from this session:
  - steering_operator_colon.json: A (centroid), C (cross-patch), DAS at ':'
  - steering_correctness.json: asymmetric +/- α causal test (latent cell)
  - steering_operator_all.json: A_latent / A_decode / C_latent / C_decode for comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
OUT = PD / "steering_slideshow_colon.pdf"
OPS = ["Add", "Sub", "Mul", "Div"]
FULL_OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


def text_slide(pdf, title, lines):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax = fig.add_axes([0.05, 0.04, 0.9, 0.86]); ax.axis("off")
    y = 0.97
    for ln in lines:
        if ln.startswith("# "):
            ax.text(0.0, y, ln[2:], fontsize=12, fontweight="bold", transform=ax.transAxes); y -= 0.045
        elif ln.startswith("- "):
            ax.text(0.02, y, "•  " + ln[2:], fontsize=10, transform=ax.transAxes); y -= 0.034
        elif ln == "":
            y -= 0.020
        else:
            ax.text(0.0, y, ln, fontsize=10, transform=ax.transAxes); y -= 0.034
    pdf.savefig(fig, dpi=140); plt.close(fig)


def flip_matrix(d):
    M = np.full((4, 4), np.nan)
    for i, src in enumerate(FULL_OPS):
        for j, tgt in enumerate(FULL_OPS):
            if src == tgt: continue
            k = f"{src}->{tgt}"
            if k in d: M[i, j] = d[k]["frac_tgt"]
    return M


def heat(ax, M, title, vmax_floor=0.10):
    vmax = max(vmax_floor, np.nanmax(M) if not np.all(np.isnan(M)) else vmax_floor)
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(OPS)
    ax.set_yticks(range(4)); ax.set_yticklabels(OPS)
    ax.set_xlabel("target op"); ax.set_ylabel("source op")
    ax.set_title(title, fontsize=10)
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            s = "—" if np.isnan(v) else f"{v*100:.1f}%"
            ax.text(j, i, s, ha="center", va="center", fontsize=8,
                    color="white" if (np.isnan(v) or v < 0.5 * vmax) else "black")
    return im


def main():
    def _load(name):
        p = PD / name
        if p.exists():
            try: return json.load(open(p))
            except Exception: return None
        return None

    colon = _load("steering_operator_colon.json")
    multi = _load("steering_operator_all.json")
    corr = _load("steering_correctness.json")

    with PdfPages(OUT) as pdf:
        text_slide(pdf, "CODI-GPT-2 steering meta-deck at the ':' residual",
            [
                "# What's combined here",
                "- Operator steering at ':' (decode position 4) — A / C / DAS rank-4.",
                "- Operator steering at the LATENT cell (step 4, layer 10) for comparison.",
                "- Operator steering at 'The' (decode position 1, layer 8) for comparison.",
                "- Correctness-direction steering (latent step 2, layer 6) — asymmetric +/-α test.",
                "",
                "# Methods reminder",
                "- A (centroid): replace last-token residual with per-op mean.",
                "- C (cross-patch): replace with a real target-op partner residual.",
                "- DAS rank-k: swap ONLY the projection onto the operator subspace.",
                "  (orthonormalized row-space of a 4-class logreg fit on this cell)",
                "",
                "# Flip-rate metric",
                "On src-op problems with single-op Equation '(a op b)', fraction whose new",
                "prediction equals target_op(a, b).",
            ])

        # ====== Operator @ ':' panel ======
        if colon is not None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            for ax, key, title in [
                (axes[0], "A_centroid_patch", "A: centroid"),
                (axes[1], "C_cross_patch", "C: cross-patch"),
                (axes[2], "DAS_subspace_patch", "DAS: rank-4 subspace"),
            ]:
                heat(ax, flip_matrix(colon.get(key, {})), title)
            fig.suptitle(f"Operator steering at ':' (decode pos {colon.get('target_decode_pos', 4)}, "
                         f"layer {colon.get('target_layer', 9)})",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # ====== Compare ':' vs latent vs 'The' ======
        if colon is not None and multi is not None:
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            # row 0: A: centroid at each cell
            heat(axes[0, 0], flip_matrix(multi.get("A_centroid_patch", {})),
                 "A_latent @ (step 4, layer 10)")
            heat(axes[0, 1], flip_matrix(multi.get("A_centroid_patch_decode", {})),
                 "A_decode 'The' @ (pos 1, layer 8)")
            heat(axes[0, 2], flip_matrix(colon.get("A_centroid_patch", {})),
                 "A_decode ':' @ (pos 4, layer 9)")
            heat(axes[1, 0], flip_matrix(multi.get("C_cross_patch", {})),
                 "C_latent @ (step 4, layer 10)")
            heat(axes[1, 1], flip_matrix(multi.get("C_cross_patch_decode", {})),
                 "C_decode 'The' @ (pos 1, layer 8)")
            heat(axes[1, 2], flip_matrix(colon.get("C_cross_patch", {})),
                 "C_decode ':' @ (pos 4, layer 9)")
            fig.suptitle("Operator-steering flip rates: latent vs 'The' vs ':' — all converge",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # ====== Correctness direction steering ======
        if corr is not None:
            xs = []
            for d in ["correctness", "permuted", "random"]:
                pass  # placeholder
            rows = corr["alphas"]
            for d in ["correctness", "permuted", "random"]:
                pass
            fig, ax = plt.subplots(figsize=(10, 5))
            for d in ["correctness", "permuted", "random"]:
                xs = [r["alpha"] for r in rows if r["direction"] == d]
                ys = [r["delta_acc"] * 100 for r in rows if r["direction"] == d]
                ax.plot(xs, ys, "o-", label=d)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("alpha"); ax.set_ylabel("Δ accuracy (pp)")
            ax.set_title(f"Correctness-direction steering at (step {corr['target_step_1indexed']}, "
                         f"layer {corr['target_layer']}); baseline {corr['baseline_accuracy']*100:.1f}%",
                         fontsize=11, fontweight="bold")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # ====== Synthesis ======
        lines = ["# Headline numbers"]
        if colon is not None:
            def mx(d):
                vs = [v["frac_tgt"] for v in d.values()]
                return max(vs) if vs else 0
            lines.append(f"- ':' operator steering: A_centroid max {mx(colon['A_centroid_patch'])*100:.1f}%, "
                         f"C_cross max {mx(colon['C_cross_patch'])*100:.1f}%, DAS max {mx(colon['DAS_subspace_patch'])*100:.1f}%.")
        if multi is not None:
            lines.append(f"- latent steering: max ~6%.  'The' steering: max ~6%.  ':' steering: max ~7%.")
            lines.append("- 11/12 src->tgt pairs flip <=1.4% under every method at every cell tested.")
            lines.append("- The Sub->Common-Division ~6% outlier is consistent across latent / 'The' / ':'.")
        if corr is not None:
            d_neg = next((r for r in corr["alphas"] if r["direction"] == "correctness" and r["alpha"] == -16), None)
            d_pos = next((r for r in corr["alphas"] if r["direction"] == "correctness" and r["alpha"] == 16), None)
            if d_neg and d_pos:
                lines.append(f"- correctness direction at (step 2, layer 6): α=-16 Δacc={d_neg['delta_acc']*100:+.1f}pp; "
                             f"α=+16 Δacc={d_pos['delta_acc']*100:+.1f}pp. Asymmetric causal direction.")
        lines.append("")
        lines.append("# Interpretation")
        lines.append("- The operator is encoded but not steerable via single-position residual swap")
        lines.append("  at ANY cell we have tested: latent (step 1-6, layer 10), decode 'The' (pos 1,")
        lines.append("  layer 8), or decode ':' (pos 4, layer 9). Probe ≠ causal direction is robust.")
        lines.append("- The correctness direction IS causal in the destructive direction (-α breaks)")
        lines.append("  but not in the constructive direction (+α saturates).")
        text_slide(pdf, "Synthesis — steering at ':' vs everywhere else", lines)

    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
