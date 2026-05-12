"""Re-render attn_to_numbers PDF from the saved npz (no GPU)."""
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
d = np.load(PD / "attn_to_numbers_gsm8k.npz")
meta = json.load(open(PD / "attn_to_numbers_gsm8k.json"))
mean_num = d["mean_num"]
mean_q = d["mean_q"]
num_frac = d["num_frac"]
N_LAT = meta["N_latent_steps"]
N_DEC = meta["N_decode_steps"]
N_LAYERS = meta["N_layers"]
N_HEADS = meta["N_heads"]

OUT_PDF = PD / "attn_to_numbers_gsm8k.pdf"
with PdfPages(OUT_PDF) as pdf:
    # Page 1: per-step number-attention fraction (avg over layers + heads)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    steps = np.arange(1, N_LAT + 1)
    ax.bar(steps, [num_frac[0, s].mean() * 100 for s in range(N_LAT)],
           color="#4c72b0", edgecolor="black")
    for s in steps:
        ax.text(s, num_frac[0, s-1].mean() * 100 + 0.3,
                f"{num_frac[0, s-1].mean()*100:.1f}", ha="center", fontsize=8)
    ax.set_xlabel("latent step"); ax.set_ylabel("% of Q attention to NUMBER tokens")
    ax.set_xticks(steps)
    ax.set_title("Fraction of Q-attention to number tokens — LATENT phase",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    decs = np.arange(1, N_DEC + 1)
    ax.bar(decs, [num_frac[1, d_].mean() * 100 for d_ in range(N_DEC)],
           color="#dd8452", edgecolor="black")
    ax.set_xlabel("decode step"); ax.set_ylabel("% of Q attention to NUMBER tokens")
    ax.set_xticks(decs)
    ax.set_title("DECODE phase", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("How much of each step's question-attention is to NUMBER tokens?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Per-(layer, head) heatmaps: ALL 6 latent steps + first 4 decode steps
    targets = [(0, s, f"latent step {s+1}") for s in range(N_LAT)] + \
              [(1, d_, f"decode step {d_+1}") for d_ in range(4)]
    for phase, step, label in targets:
        fig, ax = plt.subplots(figsize=(13, 6))
        G = num_frac[phase, step] * 100   # (layers, heads)
        im = ax.imshow(G, aspect="auto", cmap="viridis",
                       vmin=0, vmax=max(50, float(G.max())))
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_LAYERS))
        for L in range(N_LAYERS):
            for H in range(N_HEADS):
                if G[L, H] > 5:
                    ax.text(H, L, f"{G[L,H]:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if G[L,H] < 30 else "black")
        ax.set_title(f"{label}: % of Q attention to NUMBER tokens, per (layer, head)",
                     fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.045, label="% to numbers")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
print(f"saved {OUT_PDF}")
