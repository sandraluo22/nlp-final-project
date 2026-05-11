"""Visualize the per-(stage, layer, component) ablation grid."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PD = Path(__file__).resolve().parent
J = json.load(open(PD / "ablation_codi_gpt2.json"))
N = J["n_eval"]
COMPS = ["resid", "attn", "mlp"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, c in zip(axes, COMPS):
    g = np.array(J["grid_changed"][c]) / N * 100
    im = ax.imshow(g, aspect="auto", cmap="viridis", vmin=0, vmax=80)
    ax.set_title(f"{c.upper()}  (mean {g.mean():.1f}%)")
    ax.set_xlabel("layer")
    ax.set_ylabel("stage  (0=prompt, 1-6=latent)")
    ax.set_xticks(range(g.shape[1]))
    ax.set_yticks(range(g.shape[0]))
    yt = ["prompt"] + [f"latent {s}" for s in range(1, g.shape[0])]
    ax.set_yticklabels(yt, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04, label="% outputs changed")
plt.suptitle(f"Zero-ablation: % of {N} outputs that change vs baseline\n"
             f"(CODI-GPT-2 on SVAMP)", fontsize=12, fontweight="bold")
plt.tight_layout()
out = PD / "ablation_grid.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {out}")
