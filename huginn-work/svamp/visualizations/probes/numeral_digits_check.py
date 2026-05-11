"""Sanity check: in vary_a (b fixed = 4, varied a), is PCA picking up the
CONTINUOUS value of a, or just the DIGIT COUNT (2- vs 3-digit numerals)?

The two are correlated because string length is monotone in magnitude, but
they're not the same:
  - value     : continuous, 11..200
  - n_digits  : discrete in {2, 3} (a in 11..99 → 2 digits; a in 100..200 → 3)

For each (layer, step) cell, fit PCA(3) on the 80 activations, then compute:
  - r2(PC1, value),  r2(PC1, n_digits)
  - r2(PC2, value),  r2(PC2, n_digits)
  - r2(PC3, value),  r2(PC3, n_digits)
Higher r² to value than to n_digits ⇒ PC tracks magnitude continuously, not
the token-length discontinuity. Pick the cell with the most extreme split
and render a side-by-side scatter.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA


HUGINN_ROOT = Path(__file__).resolve().parents[2]   # huginn-work/
PROJECT_ROOT = HUGINN_ROOT.parent
OUT_DIR = HUGINN_ROOT / "visualizations" / "probes"

EXPERIMENTS = [
    ("Huginn",   HUGINN_ROOT / "latent-sweep" / "huginn_vary_a" / "K32" / "activations.pt",
                 "K"),
    ("CODI-GPT-2",
                 PROJECT_ROOT / "codi-work" / "inference" / "runs" / "gpt2_vary_a" / "activations.pt",
                 "latent step"),
]
DATA_VARY_A = PROJECT_ROOT / "cf-datasets" / "vary_a.json"


def r2(x, y):
    """Pearson r squared (coefficient of determination of linear fit y ~ x)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) ** 2 / ((x @ x) * (y @ y) + 1e-12))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(DATA_VARY_A))
    a_vals = np.array([r["a"] for r in rows], dtype=float)
    n_dig = np.array([len(str(int(r["a"]))) for r in rows], dtype=float)
    print(f"vary_a: N={len(rows)},  a range {int(a_vals.min())}..{int(a_vals.max())},  "
          f"digit counts: {dict(zip(*np.unique(n_dig.astype(int), return_counts=True)))}")
    print()

    summary = {}
    fig, axes = plt.subplots(len(EXPERIMENTS), 2, figsize=(11, 5.5 * len(EXPERIMENTS)))
    if len(EXPERIMENTS) == 1:
        axes = axes[None, :]

    for row_i, (label, acts_path, step_label) in enumerate(EXPERIMENTS):
        if not acts_path.exists():
            print(f"[{label}] skipping — missing {acts_path}")
            continue
        a = torch.load(acts_path, map_location="cpu", weights_only=True).float().numpy()
        N, S, L, H = a.shape
        print(f"=== {label} ({a.shape}) ===")

        # Per-(layer, step) PCA + r² with value vs n_digits
        n_components = 3
        r2_val = np.empty((L, S, n_components))
        r2_dig = np.empty((L, S, n_components))
        proj_all = np.empty((L, S, N, n_components), dtype=np.float32)
        for l in range(L):
            for s in range(S):
                X = a[:, s, l, :]
                pca = PCA(n_components=n_components, svd_solver="randomized",
                          random_state=0)
                P = pca.fit_transform(X)
                proj_all[l, s] = P
                for k in range(n_components):
                    r2_val[l, s, k] = r2(P[:, k], a_vals)
                    r2_dig[l, s, k] = r2(P[:, k], n_dig)

        # Pick the (layer, step) cell where PC1's r² with value is highest;
        # report whether PC1 looks continuous or digit-count-driven.
        best = np.unravel_index(np.argmax(r2_val[:, :, 0]), r2_val[:, :, 0].shape)
        bl, bs = best
        print(f"  best PC1 r² with VALUE: {r2_val[bl, bs, 0]:.3f} at "
              f"(layer={bl}, {step_label}={bs+1})")
        print(f"  same cell — PC1 r² with N_DIGITS: {r2_dig[bl, bs, 0]:.3f}")
        print(f"  ratio (value / digits): {r2_val[bl, bs, 0] / max(r2_dig[bl, bs, 0], 1e-6):.2f}×")
        # Aggregate
        avg_val = r2_val[:, :, 0].mean()
        avg_dig = r2_dig[:, :, 0].mean()
        print(f"  averaged across all (layer × step): PC1 r²(value)={avg_val:.3f}  "
              f"r²(n_digits)={avg_dig:.3f}  ratio={avg_val/max(avg_dig, 1e-6):.2f}×")
        summary[label] = {
            "shape": list(a.shape),
            "best_cell": [int(bl), int(bs)],
            "PC1_r2_value":   r2_val[:, :, 0].tolist(),
            "PC1_r2_ndigits": r2_dig[:, :, 0].tolist(),
            "avg_PC1_r2_value":   float(avg_val),
            "avg_PC1_r2_ndigits": float(avg_dig),
        }

        P = proj_all[bl, bs]
        # Left: colored by VALUE (continuous viridis)
        ax = axes[row_i, 0]
        norm = Normalize(vmin=a_vals.min(), vmax=a_vals.max())
        sc = ax.scatter(P[:, 0], P[:, 1], c=a_vals, cmap="viridis", norm=norm,
                        s=22, alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax, label="a (continuous)")
        ax.set_title(f"{label}  ·  layer {bl}, {step_label}={bs+1}\n"
                     f"colored by a (continuous)  ·  PC1 r²(a)={r2_val[bl, bs, 0]:.2f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.3)

        # Right: colored by N_DIGITS (discrete 2 vs 3)
        ax = axes[row_i, 1]
        for d, color in [(2, "#1f77b4"), (3, "#d62728")]:
            mask = n_dig == d
            if mask.sum() == 0: continue
            ax.scatter(P[mask, 0], P[mask, 1], c=color, s=28,
                       label=f"{d}-digit ({int(mask.sum())})", linewidths=0,
                       alpha=0.85)
        ax.legend()
        ax.set_title(f"{label}  ·  same cell  ·  colored by digit count\n"
                     f"PC1 r²(n_digits)={r2_dig[bl, bs, 0]:.2f}  ·  "
                     f"value/digit ratio = {r2_val[bl, bs, 0]/max(r2_dig[bl, bs, 0], 1e-6):.2f}×")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.3)

    fig.suptitle("vary_a sanity check  ·  PC1 captures continuous value, not just digit count",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_png = OUT_DIR / "numeral_digits_check.png"
    fig.savefig(out_png, dpi=140)
    (OUT_DIR / "numeral_digits_check.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
