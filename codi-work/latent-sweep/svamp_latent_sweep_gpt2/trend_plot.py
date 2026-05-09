"""Trend plot: CODI-GPT2 SVAMP accuracy vs forced number of latent steps N.
Reads existing results.json files from N0..N6 and renders a line plot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "trend.png"


def main():
    rows = []
    for d in sorted(HERE.glob("N*"), key=lambda p: int(p.name[1:])):
        rj = d / "results.json"
        if not rj.exists():
            continue
        rs = json.load(open(rj))
        nc = sum(r["correct"] for r in rs)
        rows.append((int(d.name[1:]), nc, len(rs), nc / len(rs)))
    Ns = np.array([r[0] for r in rows])
    accs = np.array([r[3] for r in rows])
    print(f"  N values: {Ns.tolist()}")
    print(f"  accuracies: {[f'{a:.3f}' for a in accs]}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ns, accs, "o-", color="#d62728", lw=2, markersize=8,
            label="CODI-GPT2 (N=0..6)")
    for n, a in zip(Ns, accs):
        ax.text(n, a + 0.012, f"{a*100:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("forced number of latent steps N (0 = no latent thinking)")
    ax.set_ylabel("SVAMP accuracy")
    ax.set_title("CODI-GPT2: SVAMP accuracy vs early latent stopping (N = 0..6)")
    ax.set_xticks(Ns)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(accs) * 1.15 + 0.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
