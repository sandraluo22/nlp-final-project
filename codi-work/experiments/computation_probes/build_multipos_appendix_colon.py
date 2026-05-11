"""Multipos decode-position probe figure, with the ':' position (decode index 4)
highlighted. Direct rebuild of build_multipos_figure.py focusing on the colon."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
J = json.load(open(PD / "gpt2_multipos_probes.json"))
OUT = PD / "computation_probes_multipos_appendix_colon.pdf"
units = np.array(J["units_acc"])
tens = np.array(J["tens_acc"])
oper = np.array(J["operator_acc"])
P, Lp1 = units.shape

TEMPLATE_LABEL = {0: "EOT→The", 1: "The→ans", 2: "ans→is", 3: "is→:",
                  4: ":→<num>", 5: "<num>→...", 6: "...", 7: "...",
                  8: "...", 9: "...", 10: "...", 11: "...", 12: "...",
                  13: "...", 14: "...", 15: "..."}


def pos_label(p): return TEMPLATE_LABEL.get(p, f"D{p}")


def main():
    with PdfPages(OUT) as pdf:
        # Headline: heatmaps with ':' column highlighted
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, M, name, vmax in zip(axes, [units, tens, oper],
                                     ["units digit (chance 10%)",
                                      "tens digit (chance 10%)",
                                      "operator (chance 25%)"],
                                     [0.6, 0.6, 1.0]):
            im = ax.imshow(M.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
            ax.set_xlabel("decode position (token whose residual we read)")
            ax.set_ylabel("layer")
            ax.set_xticks(range(P))
            ax.set_xticklabels([pos_label(p) for p in range(P)],
                               rotation=45, fontsize=7, ha="right")
            ax.set_title(name, fontsize=10)
            # Highlight ':' position
            ax.axvline(4, color="red", lw=1.5, ls="--", alpha=0.7)
            ax.text(4.05, Lp1 - 0.5, "':' position", color="red", fontsize=8, va="top")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.suptitle("Multi-position decode probes — ':' (decode pos 4) highlighted",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide: per-variable accuracy at ':' column vs other positions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, M, name in zip(axes, [units, tens, oper],
                               ["units", "tens", "operator"]):
            for p in range(P):
                lw = 2.0 if p == 4 else 0.7
                alpha = 1.0 if p == 4 else 0.4
                color = "red" if p == 4 else None
                label = "':' (pos 4)" if p == 4 else (pos_label(p) if p < 6 else None)
                ax.plot(range(Lp1), M[p], lw=lw, alpha=alpha, color=color, label=label)
            ax.set_xlabel("layer")
            ax.set_ylabel(f"{name} probe acc")
            ax.set_title(f"{name}", fontsize=10)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.3); ax.set_ylim(0, 1)
        fig.suptitle("Per-position probe accuracy per layer (':' highlighted in red)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide: peak acc per variable at ':'
        fig, ax = plt.subplots(figsize=(8, 4))
        names = ["operator", "tens", "units"]
        peak_at_colon = [oper[4].max(), tens[4].max(), units[4].max()]
        peak_global   = [oper.max(),    tens.max(),    units.max()]
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, peak_at_colon, width, color="#d62728", label="peak @ ':' (decode pos 4)")
        ax.bar(x + width/2, peak_global, width, color="#1f77b4", label="peak across all positions")
        for i, (a_, b_) in enumerate(zip(peak_at_colon, peak_global)):
            ax.text(i - width/2, a_ + 0.01, f"{a_:.2f}", ha="center", fontsize=8)
            ax.text(i + width/2, b_ + 0.01, f"{b_:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("probe accuracy"); ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_title("Probe accuracy at ':' vs global peak", fontsize=11, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
