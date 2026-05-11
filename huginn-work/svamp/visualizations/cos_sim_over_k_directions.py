"""Slideshow: per-pair cos_sim of top-PC directions across K=1..32 for every
Huginn vary_* comparison. One slide per pair, each showing:
  - heatmap of signed cos(PC1_A[block, K], PC1_B[block, K]) over (block, K)
  - line plot overlaying the 4 blocks' trajectories vs K
  - line plot of the rotation of the difference vector v_op - v_num across K

Output: huginn-work/visualizations/cos_sim_over_k_directions.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA


HUGINN = Path(__file__).resolve().parent.parent
OUT_PDF = HUGINN / "visualizations" / "cos_sim_over_k_directions.pdf"

ACTS = {
    "vary_numerals":     HUGINN / "latent-sweep" / "huginn_vary_numerals" / "K32" / "activations.pt",
    "vary_a":            HUGINN / "latent-sweep" / "huginn_vary_a" / "K32" / "activations.pt",
    "vary_b":            HUGINN / "latent-sweep" / "huginn_vary_b" / "K32" / "activations.pt",
    "vary_a_2digit":     HUGINN / "latent-sweep" / "huginn_vary_a_2digit" / "K32" / "activations.pt",
    "vary_b_2digit":     HUGINN / "latent-sweep" / "huginn_vary_b_2digit" / "K32" / "activations.pt",
    "vary_both_2digit":  HUGINN / "latent-sweep" / "huginn_vary_both_2digit" / "K32" / "activations.pt",
    "vary_operator":     HUGINN / "latent-sweep" / "huginn_vary_operator" / "K32" / "activations.pt",
}
PAIRS = [
    ("vary_numerals",    "vary_operator"),
    ("vary_a",           "vary_operator"),
    ("vary_b",           "vary_operator"),
    ("vary_a",           "vary_b"),
    ("vary_a_2digit",    "vary_b_2digit"),
    ("vary_a_2digit",    "vary_operator"),
    ("vary_b_2digit",    "vary_operator"),
    ("vary_both_2digit", "vary_operator"),
]


def load(p): return torch.load(p, map_location="cpu", weights_only=True).float().numpy()


def top_pcs(acts):
    """Per (block, K), return PC1 unit vectors. Sign-aligned across K within
    each block so adjacent K's don't flip arbitrarily."""
    N, S, L, H = acts.shape
    out = np.zeros((L, S, H), dtype=np.float32)
    for blk in range(L):
        for K in range(S):
            X = acts[:, K, blk, :]
            pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
            pca.fit(X); v = pca.components_[0]
            if K > 0 and np.dot(v, out[blk, K-1]) < 0: v = -v
            out[blk, K] = v
    return out


def signed_cos_grid(A, B):
    """A, B shape (L, S, H). Returns (L, S) signed cosine."""
    L, S, _ = A.shape
    out = np.empty((L, S), dtype=np.float32)
    for l in range(L):
        for s in range(S):
            out[l, s] = float(np.dot(A[l, s], B[l, s]) /
                              (np.linalg.norm(A[l, s]) * np.linalg.norm(B[l, s]) + 1e-12))
    return out


def render_pair_slide(pdf, A_name, B_name, A_pcs, B_pcs):
    cos_grid = signed_cos_grid(A_pcs, B_pcs)
    L, S = cos_grid.shape
    Ks = np.arange(1, S + 1)

    fig = plt.figure(figsize=(15, 7.5))
    fig.suptitle(
        f"Huginn  ·  cos(PC1[{A_name}, K, block], PC1[{B_name}, K, block]) across K",
        fontsize=12, fontweight="bold",
    )

    # --- Top-left: signed-cos heatmap ---
    ax_h = fig.add_axes([0.05, 0.55, 0.42, 0.35])
    im = ax_h.imshow(cos_grid, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax_h.set_xticks(np.arange(0, S, 2))
    ax_h.set_xticklabels([str(k+1) for k in range(0, S, 2)], fontsize=8)
    ax_h.set_yticks(range(L)); ax_h.set_yticklabels([f"block {b}" for b in range(L)])
    ax_h.set_xlabel("recurrence step K"); ax_h.set_ylabel("core block")
    ax_h.set_title(f"signed cos  (mean {cos_grid.mean():.2f}, |max| {np.abs(cos_grid).max():.2f})",
                   fontsize=10)
    cb = plt.colorbar(im, ax=ax_h, label="cos(PC1_A, PC1_B)")
    for r in range(L):
        for c in range(S):
            v = cos_grid[r, c]
            ax_h.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=5,
                      color="black" if abs(v) < 0.5 else "white")

    # --- Top-right: trajectory of cos vs K, one line per block ---
    ax_t = fig.add_axes([0.55, 0.55, 0.40, 0.35])
    for b in range(L):
        ax_t.plot(Ks, cos_grid[b], "o-", label=f"block {b}", lw=1.5, ms=4)
    ax_t.axhline(0, color="gray", lw=1, ls=":")
    ax_t.set_xlabel("recurrence step K"); ax_t.set_ylabel("cos(PC1_A, PC1_B)")
    ax_t.set_title("trajectory across K, per block")
    ax_t.set_xticks(np.arange(1, S+1, 2)); ax_t.set_ylim(-1.05, 1.05)
    ax_t.grid(alpha=0.3); ax_t.legend(fontsize=8)

    # --- Bottom: rotation of the difference direction (v_A − v_B) ---
    ax_d = fig.add_axes([0.05, 0.10, 0.90, 0.32])
    # compute diff and its rotation across K, per block
    rot_per_block = np.zeros((L, S), dtype=np.float32)
    align_to_K1 = np.zeros((L, S), dtype=np.float32)
    for b in range(L):
        diff = A_pcs[b] - B_pcs[b]                        # (S, H)
        diff = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12)
        rot_per_block[b, 0] = 1.0
        for K in range(1, S):
            rot_per_block[b, K] = float(np.dot(diff[K], diff[K-1]))
        align_to_K1[b] = diff @ diff[0]
    for b in range(L):
        ax_d.plot(Ks, align_to_K1[b], "o-", label=f"block {b}  (mean {align_to_K1[b].mean():+.2f})",
                  lw=1.5, ms=4)
    ax_d.axhline(0, color="gray", lw=1, ls=":")
    ax_d.set_xlabel("recurrence step K"); ax_d.set_ylabel("cos(diff[K], diff[1])")
    ax_d.set_title("does the difference vector v_A − v_B point in a consistent direction?  "
                   f"(==1 means K-th diff is colinear with K=1's diff)")
    ax_d.set_xticks(np.arange(1, S+1, 2)); ax_d.set_ylim(-1.05, 1.05)
    ax_d.grid(alpha=0.3); ax_d.legend(fontsize=8, loc="lower right")

    pdf.savefig(fig, dpi=130); plt.close(fig)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print("loading & PCA-ing all datasets:", flush=True)
    pcs = {}
    for name, path in ACTS.items():
        if not path.exists():
            print(f"  SKIP missing: {name}"); continue
        a = load(path)
        print(f"  {name:<22} shape={a.shape}", flush=True)
        pcs[name] = top_pcs(a)

    print(f"\nwriting {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        for A_name, B_name in PAIRS:
            if A_name not in pcs or B_name not in pcs:
                print(f"  SKIP {A_name} vs {B_name} (missing acts)"); continue
            print(f"  slide: {A_name} vs {B_name}", flush=True)
            render_pair_slide(pdf, A_name, B_name, pcs[A_name], pcs[B_name])
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
