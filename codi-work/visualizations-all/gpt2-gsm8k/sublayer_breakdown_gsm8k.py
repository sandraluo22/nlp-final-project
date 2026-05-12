"""Side-by-side attn_out vs mlp_out logit-lens per (step, layer) on GSM8K.

Reads logit_lens_gsm8k.json and produces a per-step page comparing:
  - attn_out modal token + confidence at each layer
  - mlp_out modal token + confidence at each layer
  - resid_post modal token + confidence (the cumulative residual exiting the
    block, what the LM head would emit)

This makes visible the finding that ATTENTION carries operator/marker/digit
tokens while MLP outputs are largely noise-word fragments at most layers.

Output: sublayer_breakdown_gsm8k.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
IN_JSON = REPO / "experiments" / "computation_probes" / "logit_lens_gsm8k.json"
OUT_PDF = PD / "sublayer_breakdown_gsm8k.pdf"


def main():
    d = json.load(open(IN_JSON))
    SUBL = d["SUBLAYERS"]
    N_LAT = d["N_LAT"]; N_LAYERS = d["N_LAYERS"]
    modal = d["modal_token"]
    conf = np.array(d["mean_top1_conf"])

    SUBL_TO_PLOT = ["attn_out", "mlp_out", "resid_post"]
    sub_idx = [SUBL.index(s) for s in SUBL_TO_PLOT]
    colors = {"attn_out": "#4c72b0", "mlp_out": "#dd8452", "resid_post": "#2ca02c"}

    with PdfPages(OUT_PDF) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")
        body = (
            "Sublayer breakdown — what does each (step, layer) sublayer commit to?\n\n"
            "For each of the 6 latent steps, we plot per-layer:\n"
            "  - attn_out  : the layer's attention output (residual contribution from attn block)\n"
            "  - mlp_out   : the layer's MLP output (residual contribution from MLP block)\n"
            "  - resid_post: residual stream AFTER this block (the cumulative state)\n\n"
            "Each bar shows the mean top-1 confidence at that cell; text labels the\n"
            "MODAL top-1 token across N={N} GSM8K test problems.\n\n"
            "Key visual takeaway: at most layers, ATTN tokens are interpretable\n"
            "(numbers, '<<', '>>', operators); MLP tokens are mostly noise word\n"
            "fragments. Attention is doing the structural work; MLP magnitudes are\n"
            "large but their per-layer token-space projections are not coherent.\n"
        ).format(N=d["N_examples"])
        ax.text(0.04, 0.95, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Sublayer breakdown — interpretation guide",
                     fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # One page per step
        for s in range(N_LAT):
            fig, axes = plt.subplots(len(SUBL_TO_PLOT), 1,
                                      figsize=(14, 4 * len(SUBL_TO_PLOT)),
                                      sharex=True)
            for row, sl_name in enumerate(SUBL_TO_PLOT):
                sl_i = SUBL.index(sl_name)
                ax = axes[row]
                vals = conf[s, :, sl_i]
                ax.bar(range(N_LAYERS), vals, color=colors[sl_name],
                       edgecolor="black")
                for L in range(N_LAYERS):
                    tk = (modal[s][L][sl_i] or "").replace("\n", "\\n")
                    tk_disp = repr(tk)[:8]
                    ax.text(L, vals[L] + 0.005, tk_disp,
                            ha="center", fontsize=8, rotation=30)
                ax.set_ylabel(f"{sl_name}\ntop-1 conf",
                              fontsize=10, fontweight="bold")
                ax.set_xticks(range(N_LAYERS))
                ax.set_ylim(0, max(0.7, float(conf.max()) + 0.12))
                ax.grid(axis="y", alpha=0.3)
            axes[-1].set_xlabel("layer")
            fig.suptitle(f"Latent step {s+1}: per-layer modal token + confidence "
                         f"(attn vs mlp vs resid_post)",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Focused comparison: step 2 vs step 3 side by side, attn vs mlp only
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for col, FS in enumerate([1, 2]):  # 0-indexed step 2, 3
            for row, sl_name in enumerate(["attn_out", "mlp_out"]):
                sl_i = SUBL.index(sl_name)
                ax = axes[row, col]
                vals = conf[FS, :, sl_i]
                ax.bar(range(N_LAYERS), vals, color=colors[sl_name],
                       edgecolor="black")
                for L in range(N_LAYERS):
                    tk = (modal[FS][L][sl_i] or "").replace("\n", "\\n")
                    ax.text(L, vals[L] + 0.005, repr(tk)[:8],
                            ha="center", fontsize=8, rotation=30)
                ax.set_xticks(range(N_LAYERS))
                ax.set_xlabel("layer")
                ax.set_ylabel("top-1 conf")
                ax.set_ylim(0, max(0.7, float(conf.max()) + 0.12))
                ax.set_title(f"step {FS+1} — {sl_name}",
                             fontsize=11, fontweight="bold")
                ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Step 2 (commit) vs Step 3 (synthesize): attn_out vs mlp_out",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
