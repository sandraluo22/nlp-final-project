"""Context-isolation appendix focused on the ':' decode position.

The original experiment captures probe accuracy for units/tens/operator across
multiple decode positions × layers under 5 context-isolation methods.
This rebuild extracts the ':' column (decode pos 4) from each method and shows
per-layer accuracy curves.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
J = json.load(open(PD / "context_isolation.json"))
OUT = PD / "context_isolation_slideshow_colon.pdf"

POS_COLON = 4  # ':' position in decode
methods = [k for k in J if k.startswith("method") or k == "baseline"]
VARS = ["units", "tens", "op"]


def main():
    L = J.get("L_plus_1", 13)
    P = J.get("P", 16)
    with PdfPages(OUT) as pdf:
        # Headline: per-variable, per-method, per-layer accuracy at ':' position
        for var in VARS:
            fig, ax = plt.subplots(figsize=(10, 5))
            for m in methods:
                arr = np.array(J[m][var])  # (P, L+1)
                if arr.shape[0] < P or arr.shape[1] < L:
                    continue
                ax.plot(range(L), arr[POS_COLON], "o-", label=m, lw=1.5)
            ax.set_xlabel("layer"); ax.set_ylabel(f"{var} probe acc")
            ax.set_title(f"context isolation @ ':' (decode pos {POS_COLON}) — {var}",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
            chance = 0.10 if var in ("units", "tens") else 0.25
            ax.axhline(chance, color="gray", ls=":", lw=0.7, label=f"chance ({chance})")
            ax.set_ylim(0, 1.0)
            fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # Heatmap: per-method, per-variable peak across layers at ':'
        fig, axes = plt.subplots(1, len(VARS), figsize=(15, 4))
        for ai, var in enumerate(VARS):
            ax = axes[ai]
            mat = np.zeros((len(methods), L))
            for mi, m in enumerate(methods):
                arr = np.array(J[m][var])
                if arr.shape[0] > POS_COLON: mat[mi] = arr[POS_COLON, :L]
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
            ax.set_xticks(range(0, L, 2)); ax.set_xlabel("layer")
            ax.set_title(f"{var} acc at ':' — per method × layer", fontsize=10)
            for mi in range(len(methods)):
                for l in range(L):
                    v = mat[mi, l]
                    if v >= 0.4:
                        ax.text(l, mi, f"{v:.2f}", ha="center", va="center",
                                fontsize=5, color="white" if v < 0.6 else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.suptitle("Context-isolation probe accuracy at ':' position (per method × layer)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
