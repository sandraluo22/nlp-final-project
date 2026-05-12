"""Bar graphs of the 4-tier recovery-rate sweep.

Reads patch_recovery_sweep.json and produces:
  patch_recovery_sweep.pdf — 1 page per CF set with 4 bar panels (Sweep A: per step,
                              Sweep B: per-layer residual, Sweep C: per-layer attn,
                              Sweep D: per-layer mlp). 100% recovery = full bar
                              (cell is robust); shorter bar = more disruption.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
IN_JSON = REPO / "experiments" / "computation_probes" / "patch_recovery_sweep_gsm8k.json"
D = json.load(open(IN_JSON))
OUT_PDF = PD / "patch_recovery_sweep_gsm8k.pdf"

N_LAYERS = D["N_LAYERS"]
N_LAT = D["N_LAT"]


def bar(ax, labels, values, title, color, ylabel="recovery rate (%)", baseline=None,
        threshold=None):
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 102)
    ax.axhline(100, color="gray", lw=0.5, ls=":")
    if threshold is not None:
        ax.axhline(threshold, color="red", lw=0.7, ls="--",
                   label=f"baseline acc {threshold:.0f}%")
        ax.legend(fontsize=7, loc="lower right")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, min(v + 1, 100.5),
                f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def main():
    with PdfPages(OUT_PDF) as pdf:
        for cf_name, r in D["cf_sets"].items():
            sw = r["sweeps"]
            N = r["N"]; base = r["baseline_accuracy"]
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.suptitle(f"Recovery-rate sweep — {cf_name}  (N={N}, baseline acc={base*100:.1f}%)",
                         fontsize=12, fontweight="bold")
            # Sweep A
            labels = [f"s{s+1}" for s in range(N_LAT)]
            vals = [sw[f"step{s+1}"]["recovery_rate"] * 100 for s in range(N_LAT)]
            bar(axes[0, 0], labels, vals, "Sweep A: ablate whole STEP (all 12 layers × attn+mlp)", "#1f77b4")
            # Sweep B
            labels = [f"L{l}" for l in range(N_LAYERS)]
            vals = [sw[f"resid_L{l}"]["recovery_rate"] * 100 for l in range(N_LAYERS)]
            bar(axes[0, 1], labels, vals, "Sweep B: ablate residual stream at LAYER (across all 6 steps)", "#9467bd")
            # Sweep C
            vals = [sw[f"attn_L{l}"]["recovery_rate"] * 100 for l in range(N_LAYERS)]
            bar(axes[1, 0], labels, vals, "Sweep C: ablate ATTN at LAYER (across all 6 steps)", "#ff7f0e")
            # Sweep D
            vals = [sw[f"mlp_L{l}"]["recovery_rate"] * 100 for l in range(N_LAYERS)]
            bar(axes[1, 1], labels, vals, "Sweep D: ablate MLP at LAYER (across all 6 steps)", "#2ca02c")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, dpi=140); plt.close(fig)

            # Also add an accuracy view (delta vs baseline)
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.suptitle(f"Accuracy delta vs baseline — {cf_name}  (baseline acc={base*100:.1f}%)",
                         fontsize=12, fontweight="bold")
            labels_step = [f"s{s+1}" for s in range(N_LAT)]
            labels_lay = [f"L{l}" for l in range(N_LAYERS)]
            for ax, (key_prefix, labels, color, title) in zip(axes.ravel(), [
                ("step", labels_step, "#1f77b4", "A: per STEP"),
                ("resid_L", labels_lay, "#9467bd", "B: per-LAYER residual"),
                ("attn_L", labels_lay, "#ff7f0e", "C: per-LAYER attn"),
                ("mlp_L", labels_lay, "#2ca02c", "D: per-LAYER mlp"),
            ]):
                vals = []
                for lab in labels:
                    if key_prefix == "step":
                        k = f"{key_prefix}{lab[1:]}"
                    else:
                        k = f"{key_prefix}{lab[1:]}"
                    vals.append(sw[k]["delta_acc"] * 100)
                xs = np.arange(len(labels))
                colors = [color if v >= 0 else "#d62728" for v in vals]
                bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.5)
                ax.axhline(0, color="black", lw=0.5)
                ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylabel("Δ accuracy (pp)", fontsize=9)
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.grid(axis="y", alpha=0.3)
                for b, v in zip(bars, vals):
                    if abs(v) > 0.1:
                        ax.text(b.get_x() + b.get_width() / 2,
                                v + (0.2 if v >= 0 else -0.4),
                                f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                                fontsize=7)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
