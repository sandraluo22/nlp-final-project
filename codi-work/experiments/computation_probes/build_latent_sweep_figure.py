"""Plot accuracy vs K, and cos sim between consecutive steps trajectory."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

PD = Path(__file__).resolve().parent
J = json.load(open(PD / "latent_steps_sweep.json"))

K_values = J["K_values"]
acc = [J["accuracy"][str(k)] * 100 for k in K_values]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Accuracy vs K
ax = axes[0]
ax.plot(K_values, acc, "o-", color="#1f77b4", linewidth=2, markersize=8)
ax.axvline(6, color="red", ls="--", lw=1, alpha=0.6, label="trained K=6")
ax.set_xlabel("number of latent loop steps K")
ax.set_ylabel("accuracy on 200 SVAMP (%)")
ax.set_title("CODI-GPT-2 accuracy vs. # latent loop iterations")
ax.set_xticks(K_values)
ax.grid(alpha=0.3)
ax.legend()

# (b) Cos sim between consecutive steps for the K=20 run
ax = axes[1]
cs20 = J["cos_sim_consec_per_K"]["20"]
ax.plot(np.arange(1, len(cs20)+1), cs20, "o-", color="#d62728", linewidth=2, markersize=8)
ax.axhline(1.0, color="gray", ls="--", lw=0.5, alpha=0.5, label="cos=1 (fixed point)")
ax.axvline(6, color="red", ls="--", lw=1, alpha=0.6, label="trained K=6")
ax.set_xlabel("transition step k → k+1")
ax.set_ylabel("cos_sim(residual at step k, residual at step k+1)")
ax.set_title("Convergence of latent residual: consecutive-step cos sim\n(at canonical cell layer 8, K=20 run)")
ax.set_xticks(np.arange(1, len(cs20)+1))
ax.set_ylim(0.4, 1.0)
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
out = PD / "latent_steps_sweep.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
