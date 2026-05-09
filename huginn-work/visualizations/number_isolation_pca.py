"""Compare principal-component subspaces between two number-isolation datasets:

  vary_numerals : Subtraction-only, fixed scenario template, varied (a, b)
                  → PCA picks up directions encoding numerical inputs (and
                  the answer a-b).
  vary_operator : fixed (a, b) = (12, 4), 4 operators × 6 scenarios each
                  → PCA picks up directions encoding operator/scenario.

If number-encoding directions and operator-encoding directions are linearly
*disentangled* in the residual stream, the two principal subspaces should
have small overlap (near-orthogonal). If they're entangled, the overlap will
be high — and that has implications for steering.

Quantification: subspace alignment = ‖U_A^T U_B‖_F^2 / k where U_A, U_B are
the top-k principal axes of each dataset (k=3). Random chance for a
hidden-dim H subspace is k^2/H.

Outputs:
  huginn-work/visualizations/probes/number_isolation_pca.png
  huginn-work/visualizations/probes/number_isolation_pca.json
Per-(layer, step) PCA, and a 2-panel plot of the dataset comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


HUGINN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HUGINN_ROOT.parent
HUGINN_NUM = HUGINN_ROOT / "latent-sweep" / "huginn_vary_numerals" / "K32" / "activations.pt"
HUGINN_OP  = HUGINN_ROOT / "latent-sweep" / "huginn_vary_operator" / "K32" / "activations.pt"
GPT2_NUM   = PROJECT_ROOT / "codi-work" / "inference" / "runs" / "gpt2_vary_numerals" / "activations.pt"
GPT2_OP    = PROJECT_ROOT / "codi-work" / "inference" / "runs" / "gpt2_vary_operator" / "activations.pt"

OUT_DIR = HUGINN_ROOT / "visualizations" / "probes"


def load(path: Path) -> np.ndarray:
    print(f"loading {path}", flush=True)
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


def subspace_alignment(U_A: np.ndarray, U_B: np.ndarray) -> float:
    """U_A, U_B are (d, k) orthonormal bases. Return ‖U_A^T U_B‖_F^2 / k.
    1.0 = identical span, 0.0 = orthogonal subspaces.
    """
    M = U_A.T @ U_B
    return float((M ** 2).sum() / U_A.shape[1])


def per_step_alignment(acts_a: np.ndarray, acts_b: np.ndarray, k: int = 3
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each (S, L) pair, compute top-k PCA on acts_a, top-k on acts_b,
    and the subspace alignment between them. acts shape (N, S, L, H).
    Returns (alignment_grid, var_a_grid, var_b_grid) all (S, L)."""
    Na, S, L, H = acts_a.shape
    Nb = acts_b.shape[0]
    assert acts_b.shape[1:] == (S, L, H)
    align = np.empty((S, L), dtype=np.float32)
    var_a = np.empty((S, L), dtype=np.float32)
    var_b = np.empty((S, L), dtype=np.float32)
    print(f"  computing PCA × 2 over (S={S}, L={L}) = {S*L} positions, k={k}", flush=True)
    for s in range(S):
        for l in range(L):
            X_a = acts_a[:, s, l, :]
            X_b = acts_b[:, s, l, :]
            pa = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(X_a)
            pb = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(X_b)
            U_a = pa.components_.T   # (H, k)
            U_b = pb.components_.T
            align[s, l] = subspace_alignment(U_a, U_b)
            var_a[s, l] = pa.explained_variance_ratio_.sum()
            var_b[s, l] = pb.explained_variance_ratio_.sum()
    return align, var_a, var_b


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for label, path_a, path_b in [
        ("huginn", HUGINN_NUM, HUGINN_OP),
        ("gpt2",   GPT2_NUM,   GPT2_OP),
    ]:
        if not (path_a.exists() and path_b.exists()):
            print(f"\n[{label}] skipping — missing activations:")
            print(f"  {path_a.exists()}: {path_a}")
            print(f"  {path_b.exists()}: {path_b}")
            continue
        print(f"\n=== {label} ===", flush=True)
        acts_a = load(path_a)
        acts_b = load(path_b)
        H = acts_a.shape[-1]
        chance = (3 ** 2) / H
        print(f"  shapes: vary_numerals={acts_a.shape}  vary_operator={acts_b.shape}")
        print(f"  random-chance subspace alignment for k=3, H={H}: {chance:.4f}")
        align, va, vb = per_step_alignment(acts_a, acts_b, k=3)
        summary[label] = {
            "shape_numerals": list(acts_a.shape),
            "shape_operator": list(acts_b.shape),
            "alignment": align.tolist(),
            "var_explained_numerals": va.tolist(),
            "var_explained_operator": vb.tolist(),
            "chance_alignment": chance,
            "alignment_min":  float(align.min()),
            "alignment_max":  float(align.max()),
            "alignment_mean": float(align.mean()),
        }
        print(f"  alignment grid: min={align.min():.3f}  mean={align.mean():.3f}  max={align.max():.3f}")
        print(f"  (≫ chance {chance:.4f}: numbers + operator share principal subspaces — entangled)")

    (OUT_DIR / "number_isolation_pca.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR/'number_isolation_pca.json'}")

    # ---------- plot ----------
    fams = list(summary.keys())
    if not fams:
        print("nothing to plot — re-run after inference completes")
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(7 * len(fams), 5),
                             squeeze=False)
    for ax, fam in zip(axes[0], fams):
        align = np.array(summary[fam]["alignment"])  # (S, L)
        chance = summary[fam]["chance_alignment"]
        S, L = align.shape
        im = ax.imshow(align.T, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("step (CODI: latent step / Huginn: recurrence step)")
        ax.set_ylabel("layer / core block")
        ax.set_title(
            f"{fam}: subspace alignment (numerals-PCA vs operator-PCA)\n"
            f"chance={chance:.4f}  mean={align.mean():.3f}  max={align.max():.3f}"
        )
        plt.colorbar(im, ax=ax, label="alignment (1=same span, 0=orthogonal)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "number_isolation_pca.png", dpi=130)
    print(f"saved {OUT_DIR/'number_isolation_pca.png'}")

    print("\n=== INTERPRETATION ===")
    for fam in fams:
        m = summary[fam]["alignment_mean"]
        c = summary[fam]["chance_alignment"]
        ratio = m / c
        print(f"  {fam}: mean alignment {m:.3f} = {ratio:.0f}× chance ({c:.4f})")
        if ratio > 100:
            print(f"    → number and operator subspaces share substantial structure (entangled)")
        elif ratio > 10:
            print(f"    → modest overlap — partial entanglement")
        else:
            print(f"    → near-orthogonal — number and operator are linearly disentangled")


if __name__ == "__main__":
    main()
