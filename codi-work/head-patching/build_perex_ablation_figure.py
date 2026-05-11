"""Per-example baseline-vs-ablated scatter plots for the residual ablation.

For each layer L (and at stage 0 = prompt position), plot:
  x-axis: example index 0..199
  y-axis: log10(|int|+1) for both baseline (dotted) and ablated-resid-L (line)

Also: scatter of baseline_int vs ablated_int (log-log) per layer.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PD = Path(__file__).resolve().parent
d = np.load(PD / "ablation_codi_gpt2.perex.npz")
base = d["base_ints"]                # (200,)
per = d["per_ex_resid"]              # (7, 12, 200)

n_stages, n_layers, N = per.shape
print(f"loaded n_stages={n_stages}, n_layers={n_layers}, N={N}")

# === Per-example trace plot at stage 0 (the cleanest) ===
STAGE = 0
order = np.argsort(np.abs(base))   # sort examples by baseline magnitude
fig, axes = plt.subplots(3, 4, figsize=(20, 11), sharey=True)
for L in range(n_layers):
    ax = axes[L // 4, L % 4]
    base_sorted = np.abs(base[order])
    abl_sorted  = np.abs(per[STAGE, L, order])
    base_log = np.log10(np.maximum(base_sorted, 1))
    abl_log  = np.log10(np.maximum(abl_sorted, 1))
    ax.plot(np.arange(N), base_log, "k:", lw=1, label="baseline", alpha=0.7)
    ax.plot(np.arange(N), abl_log,  "-", lw=1, label=f"resid-L{L} ablated", color="#d62728", alpha=0.85)
    ax.set_title(f"resid L{L:02d}  median ratio={np.median(abl_sorted/np.maximum(base_sorted,1)):.2f}",
                 fontsize=10)
    ax.set_xlabel("example (sorted by |baseline|)", fontsize=8)
    ax.set_ylabel("log10(|int|+1)", fontsize=8)
    ax.set_ylim(0, 12); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
plt.suptitle(f"Per-example outputs after zero-ablating residual stream at (stage 0, layer L)\n"
             f"(N={N} SVAMP examples, sorted by baseline magnitude)",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = PD / "ablation_perex_traces_stage0.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {out}")

# === Scatter: baseline vs ablated per layer (stage 0) ===
fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey=True, sharex=True)
for L in range(n_layers):
    ax = axes[L // 4, L % 4]
    x = np.maximum(np.abs(base), 1).astype(float)
    y = np.maximum(np.abs(per[STAGE, L]), 1).astype(float)
    ax.scatter(x, y, s=12, alpha=0.4, c="#1f77b4")
    # diagonal
    lo, hi = 1, max(x.max(), y.max())*1.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, hi); ax.set_ylim(1, hi)
    median_ratio = np.median(y / x)
    ax.set_title(f"resid L{L:02d}  median(y/x)={median_ratio:.2f}", fontsize=10)
    ax.set_xlabel("|baseline int|", fontsize=8)
    if L % 4 == 0: ax.set_ylabel("|ablated int|", fontsize=8)
    ax.grid(alpha=0.3, which="both")
plt.suptitle(f"Baseline vs resid-L-ablated outputs (log-log) at stage 0\n"
             f"Diagonal = no change. Above = ablation grew output. Below = ablation shrank it.",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = PD / "ablation_perex_scatter_stage0.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {out}")

# === Layer-summary line plot: median |int| ratio per layer for each stage ===
fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, n_stages))
for s in range(n_stages):
    ratios = []
    for L in range(n_layers):
        x = np.maximum(np.abs(base), 1).astype(float)
        y = np.maximum(np.abs(per[s, L]), 1).astype(float)
        ratios.append(np.median(y / x))
    ax.plot(range(n_layers), ratios, "o-", color=colors[s],
            label=f"stage {s}" if s < 7 else None, linewidth=1.5, markersize=6)
ax.axhline(1, color="black", ls="--", lw=1, alpha=0.5, label="no change")
ax.set_xlabel("layer (0..11)")
ax.set_ylabel("median(|ablated int| / |baseline int|)")
ax.set_yscale("log")
ax.set_xticks(range(n_layers))
ax.set_title("Per-layer multiplicative effect on output magnitude (residual ablation)\n"
             "Above 1 = ablating this layer GROWS the output. Below 1 = SHRINKS it.")
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
out = PD / "ablation_perex_layer_summary.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {out}")
