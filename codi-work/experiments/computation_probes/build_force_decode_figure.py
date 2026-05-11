"""Bar graph of right->wrong and wrong->right per latent-step transition."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

PD = Path(__file__).resolve().parent
J = json.load(open(PD / "force_decode_per_step.json"))
# CODI was trained with K=6 latent steps — show only the trained regime
K_SHOW = 6
trs = J["transitions"][:K_SHOW - 1]   # 5 transitions: 1->2, 2->3, 3->4, 4->5, 5->6

# Pull arrays
from_steps = [t["from_step"] for t in trs]
rtw = np.array([t["right_to_wrong"] for t in trs])
wtr = np.array([t["wrong_to_right"] for t in trs])
acc = np.array(J["accuracy_per_step"][:K_SHOW]) * 100

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# (a) Transitions
ax = axes[0]
width = 0.4
xs = np.arange(len(from_steps))
ax.bar(xs - width/2, wtr, width, label="wrong → right", color="#2ca02c")
ax.bar(xs + width/2, -rtw, width, label="right → wrong (shown negative)", color="#d62728")
ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(xs)
ax.set_xticklabels([f"{f}→{f+1}" for f in from_steps], rotation=45, fontsize=8)
ax.set_ylabel("# examples")
ax.set_title("Per-step transitions: right→wrong (red, below 0) and wrong→right (green, above 0)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Annotate net
for i, (a, b) in enumerate(zip(wtr, rtw)):
    net = a - b
    ax.text(xs[i], max(a, 0) + 0.4, f"{net:+d}", ha="center", fontsize=7,
            color="black", fontweight="bold")

# (b) accuracy per step
ax = axes[1]
ax.plot(range(1, len(acc)+1), acc, "o-", color="#1f77b4", linewidth=2, markersize=8)
ax.set_xlabel("latent step K (force-decode after K iterations)")
ax.set_ylabel("accuracy (%)")
ax.set_title(f"Force-decoded accuracy per K, K=1..{K_SHOW} (CODI trained value)\n200 SVAMP examples")
ax.set_xticks(range(1, len(acc)+1))
ax.set_ylim(35, 43)
ax.grid(alpha=0.3)
# annotate each point
for k, v in enumerate(acc, start=1):
    ax.annotate(f"{v:.1f}%", (k, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

plt.tight_layout()
out = PD / "force_decode_per_step.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
