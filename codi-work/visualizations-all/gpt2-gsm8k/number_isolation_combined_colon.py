"""Per-layer PCA at ':' residual for vary_* and numeral_pairs_* CF datasets,
colored by 'a', 'b', and 'answer' continuous variables. ':' analog of
number_isolation_combined.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
CF_DIR = REPO.parent / "cf-datasets"
OUT_PDF = REPO / "visualizations-all" / "gpt2" / "number_isolation_combined_colon.pdf"

DATASETS = {
    "vary_numerals": {"colorings": ["a", "b", "answer", "correct"]},
    "vary_a":        {"colorings": ["a", "answer", "correct"]},
    "vary_a_2digit": {"colorings": ["a", "answer", "correct"]},
    "vary_b":        {"colorings": ["b", "answer", "correct"]},
    "vary_b_2digit": {"colorings": ["b", "answer", "correct"]},
    "vary_both_2digit": {"colorings": ["a", "b", "answer", "correct"]},
    "vary_operator": {"colorings": ["answer", "correct"]},  # operator varies; show answer color
}


def load(ds_name):
    acts = torch.load(PD / f"{ds_name}_colon_acts.pt", map_location="cpu",
                      weights_only=True).float().numpy()
    meta = json.load(open(PD / f"{ds_name}_colon_acts_meta.json"))
    rows = json.load(open(CF_DIR / f"{ds_name}.json"))
    N = acts.shape[0]
    a_arr = np.array([r.get("a", np.nan) for r in rows[:N]], dtype=float)
    b_arr = np.array([r.get("b", np.nan) for r in rows[:N]], dtype=float)
    ans = np.array([r.get("answer", np.nan) for r in rows[:N]], dtype=float)
    pred = np.array([np.nan if v is None else float(v) for v in meta["pred_int_extracted"]])
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    correct = ((~np.isnan(pred)) & (~np.isnan(gold))
               & (np.abs(pred - gold) < 1e-3)).astype(float)  # 0/1
    return acts, {"a": a_arr, "b": b_arr, "answer": ans,
                  "correct": correct, "types": np.array(meta["types"])}


def main():
    print(f"writing {OUT_PDF}")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for ds_name, cfg in DATASETS.items():
            try:
                acts, info = load(ds_name)
            except FileNotFoundError as e:
                print(f"SKIP {ds_name}: {e}")
                continue
            N, L, H = acts.shape
            print(f"  {ds_name}: shape={acts.shape}")
            # Fit PCA-3 per layer (if N is small, fewer components)
            n_pc = min(3, N - 1, H)
            xyz = np.zeros((L, N, n_pc), dtype=np.float32)
            var_ratio = np.zeros((L, n_pc), dtype=np.float32)
            for l in range(L):
                pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0)
                xyz[l] = pca.fit_transform(acts[:, l, :])
                var_ratio[l] = pca.explained_variance_ratio_

            for coloring in cfg["colorings"]:
                vals = info[coloring]
                is_binary = coloring == "correct"
                if not is_binary:
                    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                for l in range(L):
                    fig = plt.figure(figsize=(8, 7))
                    fig.suptitle(f"{ds_name}  —  layer {l}  —  PCA at ':' residual  —  colored by {coloring}",
                                 fontsize=11, fontweight="bold")
                    if is_binary:
                        if n_pc >= 3:
                            ax = fig.add_subplot(111, projection="3d")
                            for v, color, lbl in [(0.0, "#d62728", "wrong"),
                                                  (1.0, "#2ca02c", "right")]:
                                m = vals == v
                                ax.scatter(xyz[l, m, 0], xyz[l, m, 1], xyz[l, m, 2],
                                           s=14, c=color, alpha=0.7, linewidths=0,
                                           depthshade=True, rasterized=True,
                                           label=f"{lbl} ({int(m.sum())})")
                            ax.set_xlabel(f"PC1 ({var_ratio[l, 0]*100:.1f}%)", fontsize=8)
                            ax.set_ylabel(f"PC2 ({var_ratio[l, 1]*100:.1f}%)", fontsize=8)
                            ax.set_zlabel(f"PC3 ({var_ratio[l, 2]*100:.1f}%)", fontsize=8)
                            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
                        else:
                            ax = fig.add_subplot(111)
                            for v, color, lbl in [(0.0, "#d62728", "wrong"),
                                                  (1.0, "#2ca02c", "right")]:
                                m = vals == v
                                ax.scatter(xyz[l, m, 0], xyz[l, m, 1] if n_pc > 1 else np.zeros(int(m.sum())),
                                           s=14, c=color, alpha=0.7, linewidths=0,
                                           label=f"{lbl} ({int(m.sum())})")
                            ax.set_xlabel("PC1"); ax.grid(alpha=0.3)
                        ax.legend(fontsize=8, loc="upper right")
                    else:
                        if n_pc >= 3:
                            ax = fig.add_subplot(111, projection="3d")
                            sc = ax.scatter(xyz[l, :, 0], xyz[l, :, 1], xyz[l, :, 2],
                                            c=vals, cmap="viridis", s=14, alpha=0.7,
                                            linewidths=0, depthshade=True, rasterized=True,
                                            vmin=vmin, vmax=vmax)
                            ax.set_xlabel(f"PC1 ({var_ratio[l, 0]*100:.1f}%)", fontsize=8)
                            ax.set_ylabel(f"PC2 ({var_ratio[l, 1]*100:.1f}%)", fontsize=8)
                            ax.set_zlabel(f"PC3 ({var_ratio[l, 2]*100:.1f}%)", fontsize=8)
                            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
                        else:
                            ax = fig.add_subplot(111)
                            sc = ax.scatter(xyz[l, :, 0], xyz[l, :, 1] if n_pc > 1 else np.zeros(N),
                                            c=vals, cmap="viridis", s=14, alpha=0.7,
                                            linewidths=0, rasterized=True,
                                            vmin=vmin, vmax=vmax)
                            ax.set_xlabel("PC1"); ax.grid(alpha=0.3)
                        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.08)
                        cb.set_label(coloring, fontsize=8)
                    fig.tight_layout()
                    pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
