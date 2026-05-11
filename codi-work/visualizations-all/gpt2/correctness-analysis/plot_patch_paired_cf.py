"""Heatmaps for paired-CF interchange patching results.

Reads patch_paired_cf.json (produced by patch_paired_cf.py) and writes
patch_paired_cf.pdf — for each CF set, four pages:

  Page 1: transfer rate (followed source) heatmap, attn & mlp side-by-side.
          Cells where patching reliably carries the source answer over.
  Page 2: "other" rate heatmap (prediction changed but matched neither A nor
          B). Cells that are necessary structurally but don't themselves carry
          the answer — gating roles.
  Page 3: "followed target" rate heatmap. Robust cells where patching had no
          effect.
  Page 4: summary bar — best (step, layer, block) per CF set ranked by
          transfer rate.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
IN_JSON = PD / "patch_paired_cf.json"
OUT_PDF = PD / "patch_paired_cf.pdf"


def cell_matrix(cells: dict, n_steps: int, n_layers: int, block: str, key: str):
    """Materialize a (n_steps, n_layers) matrix for a given block ('attn' or
    'mlp') and metric ('transfer_rate', 'n_followed_target', 'n_other')."""
    M = np.zeros((n_steps, n_layers))
    for step in range(1, n_steps + 1):
        for layer in range(n_layers):
            c = cells.get(f"step{step}_L{layer}_{block}")
            if c is None: continue
            v = c[key]
            M[step - 1, layer] = v
    return M


def heatmap(ax, M, title, cmap, vmin, vmax, n_for_pct=None):
    """Draw a (steps, layers) heatmap with cell labels."""
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax)
    n_steps, n_layers = M.shape
    ax.set_xlabel("layer"); ax.set_ylabel("latent step")
    ax.set_yticks(range(n_steps)); ax.set_yticklabels([str(s + 1) for s in range(n_steps)])
    ax.set_xticks(range(n_layers))
    ax.set_title(title, fontsize=10, fontweight="bold")
    for s in range(n_steps):
        for l in range(n_layers):
            v = M[s, l]
            txt = f"{v*100:.0f}" if vmax <= 1.5 else f"{v:.0f}"
            ax.text(l, s, txt, ha="center", va="center", fontsize=7,
                    color="white" if v < (vmin + vmax) / 2 else "black")
    return im


def main():
    D = json.load(open(IN_JSON))
    N_LAYERS = D["N_LAYERS"]; N_LAT = D["N_LAT"]
    with PdfPages(OUT_PDF) as pdf:
        for cf_name, r in D["cf_sets"].items():
            N = r["N"]; cells = r["conditions"]
            # Per-cell rates (fraction of N).
            tr_attn = cell_matrix(cells, N_LAT, N_LAYERS, "attn", "n_followed_source") / N
            tr_mlp  = cell_matrix(cells, N_LAT, N_LAYERS, "mlp",  "n_followed_source") / N
            tt_attn = cell_matrix(cells, N_LAT, N_LAYERS, "attn", "n_followed_target") / N
            tt_mlp  = cell_matrix(cells, N_LAT, N_LAYERS, "mlp",  "n_followed_target") / N
            ot_attn = cell_matrix(cells, N_LAT, N_LAYERS, "attn", "n_other") / N
            ot_mlp  = cell_matrix(cells, N_LAT, N_LAYERS, "mlp",  "n_other") / N
            base_acc = r["baseline_accuracy"]

            # Page 1: transfer rate (followed source)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            im0 = heatmap(axes[0], tr_attn, f"{cf_name} — transfer rate (attn)",
                          "viridis", 0.0, max(0.6, tr_attn.max() * 1.05))
            im1 = heatmap(axes[1], tr_mlp,  f"{cf_name} — transfer rate (mlp)",
                          "viridis", 0.0, max(0.6, tr_mlp.max() * 1.05))
            fig.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.03,
                         label="P(followed source)")
            fig.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.03,
                         label="P(followed source)")
            fig.suptitle(f"Paired-CF interchange patching — transfer rate "
                         f"(N={N}, baseline acc={base_acc*100:.0f}%)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

            # Page 2: 'other' rate (changed but matched neither A nor B)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            im0 = heatmap(axes[0], ot_attn, f"{cf_name} — 'other' rate (attn)",
                          "magma", 0.0, max(0.6, ot_attn.max() * 1.05))
            im1 = heatmap(axes[1], ot_mlp,  f"{cf_name} — 'other' rate (mlp)",
                          "magma", 0.0, max(0.6, ot_mlp.max() * 1.05))
            fig.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.03,
                         label="P(broken: neither A nor B)")
            fig.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.03,
                         label="P(broken: neither A nor B)")
            fig.suptitle(f"Paired-CF interchange patching — 'other' rate "
                         f"(cell is necessary but doesn't carry the answer)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

            # Page 3: followed target (no effect)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            im0 = heatmap(axes[0], tt_attn, f"{cf_name} — robust (followed target, attn)",
                          "cividis", 0.0, 1.05)
            im1 = heatmap(axes[1], tt_mlp,  f"{cf_name} — robust (followed target, mlp)",
                          "cividis", 0.0, 1.05)
            fig.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.03,
                         label="P(no effect: stayed on A)")
            fig.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.03,
                         label="P(no effect: stayed on A)")
            fig.suptitle(f"Paired-CF interchange patching — robust cells",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

            # Page 4: top-10 cells by transfer rate
            rows = []
            for key, c in cells.items():
                rows.append((key, c["n_followed_source"] / N,
                             c["n_followed_target"] / N, c["n_other"] / N))
            rows.sort(key=lambda r: -r[1])
            top = rows[:15]
            fig, ax = plt.subplots(figsize=(12, 6))
            xs = np.arange(len(top))
            tr = [r[1] for r in top]
            tt = [r[2] for r in top]
            ot = [r[3] for r in top]
            w = 0.27
            ax.bar(xs - w, tr, w, color="#2ca02c", label="followed source (transfer)")
            ax.bar(xs,     ot, w, color="#d62728", label="other (broken)")
            ax.bar(xs + w, tt, w, color="#cccccc", label="followed target (no effect)")
            ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in top],
                                                  rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("fraction of N"); ax.set_ylim(0, 1.05)
            ax.set_title(f"{cf_name} — top-15 cells by transfer rate",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
