"""Visualize bare-math vs SVAMP op-direction cos similarity per (pos, layer)."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PD = Path(__file__).resolve().parent
d = np.load(PD / "bare_vs_svamp_op_dir.npz")
keys = [k for k in d.files if k not in ("P", "Lp1")]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, k in zip(axes.flat, keys):
    g = d[k]
    im = ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{k}   mean={g.mean():+.3f}")
    ax.set_xlabel("layer"); ax.set_ylabel("decode pos")
    ax.set_xticks(range(g.shape[1])); ax.set_yticks(range(g.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.05, label="cos sim")
plt.suptitle("cos_sim(bare-math op direction, SVAMP op direction)\n"
             "Red = same direction in both contexts. Blue = opposite. White = orthogonal.",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = PD / "bare_vs_svamp_op_dir.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {out}")
