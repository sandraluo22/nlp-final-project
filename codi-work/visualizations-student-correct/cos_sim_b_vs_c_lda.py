"""Cosine similarity between sc-v2-B and sc-v2-C operator directions, this
time using LDA discriminant axes instead of centroid differences.

Centroid-difference directions are non-orthogonal across operators (e.g.,
Add and Mul are anti-parallel because they sit on a single "additive vs
multiplicative" axis). LDA constructs *orthogonal* discriminant axes by
maximizing between-class / within-class scatter ratio, giving cleaner
per-operator directions.

For each (layer, latent_step):
  1. Restrict each dataset to the 3 common operators {Add, Sub, Mul}.
  2. Fit LDA(n_components=2) on each (3 classes -> 2 LD axes).
  3. Compute the per-operator direction = LDA's classifier coef_ row for
     that operator (one-vs-rest classifier weight). These are NOT orthogonal
     to each other, but each is the best linear discriminator for its op.
  4. Also compute the LDA SUBSPACE alignment between B and C (Procrustes /
     principal angles) — measures whether the discriminative subspaces are
     the same, regardless of how each op is parameterized within them.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

REPO = Path(__file__).resolve().parent.parent

OUT_PDF = REPO / "visualizations-student-correct" / "cos_sim_b_vs_c_lda.pdf"
OUT_STATS = REPO / "visualizations-student-correct" / "cos_sim_b_vs_c_lda_stats.json"


def load(name_acts: str, name_meta: str, op_filter: list[str] | None = None):
    acts = torch.load(
        REPO / "inference" / "runs" / f"{name_acts}_student" / "activations.pt",
        map_location="cpu", weights_only=True,
    ).float().numpy()
    results = json.load(
        open(REPO / "inference" / "runs" / f"{name_acts}_student" / "results.json")
    )
    meta = json.load(open(REPO.parent / "cf-datasets" / f"{name_meta}.json"))
    types = np.array([r["type"] for r in meta])
    correct = np.array([r["correct"] for r in results], dtype=bool)
    keep = correct
    if op_filter is not None:
        keep = keep & np.isin(types, op_filter)
    return acts[keep], types[keep]


def cos(u, v):
    n = np.linalg.norm(u) * np.linalg.norm(v) + 1e-12
    return float(u @ v / n)


def principal_angles_cos(B_basis, C_basis):
    """Cosines of principal angles between two subspaces given as orthonormal bases (D x k)."""
    Q_B, _ = np.linalg.qr(B_basis)
    Q_C, _ = np.linalg.qr(C_basis)
    M = Q_B.T @ Q_C
    s = np.linalg.svd(M, compute_uv=False)
    return s  # cosines of principal angles, sorted descending


def main():
    print("loading B (cf_under99_b, common ops only)")
    acts_b, types_b = load("cf_under99_b", "cf_under99_b",
                           op_filter=["Addition", "Subtraction", "Multiplication"])
    print(f"  shape={acts_b.shape}  types={dict(Counter(types_b))}")
    print("loading C (cf_under99 minus Div)")
    acts_c, types_c = load("cf_under99", "cf_under99",
                           op_filter=["Addition", "Subtraction", "Multiplication"])
    print(f"  shape={acts_c.shape}  types={dict(Counter(types_c))}")

    common_ops = ["Addition", "Subtraction", "Multiplication"]
    op_to_idx = {op: i for i, op in enumerate(common_ops)}

    L = acts_b.shape[2]
    S = acts_b.shape[1]

    # Per (layer, step):
    #   - fit LDA on each dataset
    #   - compare classifier coef_ rows (one-vs-rest direction per op)
    #   - compare LDA discriminant subspaces (principal angles)
    cos_within = {op: np.zeros((L, S)) for op in common_ops}
    cos_cross = {f"{a} vs {b}": np.zeros((L, S))
                 for a in common_ops for b in common_ops if a != b}
    subspace_angles = np.zeros((L, S, 2))  # (cos of principal angle 1, angle 2) for the 2D LDA subspace

    for layer in range(L):
        for step in range(S):
            X_b = acts_b[:, step, layer, :]
            X_c = acts_c[:, step, layer, :]
            lda_b = LinearDiscriminantAnalysis(n_components=2, solver="svd")
            lda_b.fit(X_b, types_b)
            lda_c = LinearDiscriminantAnalysis(n_components=2, solver="svd")
            lda_c.fit(X_c, types_c)
            # coef_ shape: (n_classes, H). LDA orders classes alphabetically:
            # ['Addition', 'Multiplication', 'Subtraction']
            ord_b = list(lda_b.classes_)
            ord_c = list(lda_c.classes_)
            for op_a in common_ops:
                for op_b in common_ops:
                    v_a = lda_b.coef_[ord_b.index(op_a)]
                    w_b = lda_c.coef_[ord_c.index(op_b)]
                    if op_a == op_b:
                        cos_within[op_a][layer, step] = cos(v_a, w_b)
                    else:
                        cos_cross[f"{op_a} vs {op_b}"][layer, step] = cos(v_a, w_b)
            # Subspace angle: scalings_ contains the LD axes as columns.
            sc_b = lda_b.scalings_[:, :2]
            sc_c = lda_c.scalings_[:, :2]
            subspace_angles[layer, step] = principal_angles_cos(sc_b, sc_c)
        if layer % 4 == 0 or layer == L - 1:
            print(f"  layer {layer:>2d}/{L-1} done", flush=True)

    print("\n=== Within-op LDA-coef cos sim (B vs C, mean across 6 latent steps) ===")
    print(f"{'layer':>5s}  " + "  ".join(f"{op[:11]:>11s}" for op in common_ops))
    for layer in range(L):
        row = "  ".join(f"{cos_within[op][layer].mean():>+10.3f} " for op in common_ops)
        print(f"  {layer:>3d}  {row}")

    print("\n=== LDA subspace alignment (cos of principal angles, mean across steps) ===")
    print("  cos = 1.0 means subspaces coincide; cos = 0 means orthogonal.")
    print(f"{'layer':>5s}  {'angle 1':>9s}  {'angle 2':>9s}")
    for layer in range(L):
        m = subspace_angles[layer].mean(axis=0)
        print(f"  {layer:>3d}  {m[0]:>+8.3f}  {m[1]:>+8.3f}")

    # Plot
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: within-op LDA-coef cos sim
        ax = axes[0]
        for op in common_ops:
            ax.plot(range(L), cos_within[op].mean(axis=1), marker="o", label=op)
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.axhline(0.0, color="gray", ls=":", lw=1)
        ax.set_xticks(range(L))
        ax.set_xlabel("layer")
        ax.set_ylabel("cos sim")
        ax.set_title("Within-op LDA-coef direction\n(B vs C, same operator)")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Panel 2: cross-op LDA-coef cos sim
        ax = axes[1]
        for k, v in cos_cross.items():
            ax.plot(range(L), v.mean(axis=1), marker="s", label=k)
        ax.axhline(0.0, color="gray", ls=":", lw=1)
        ax.set_xticks(range(L))
        ax.set_xlabel("layer")
        ax.set_ylabel("cos sim")
        ax.set_title("Cross-op control\n(B-op_X vs C-op_Y)")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

        # Panel 3: subspace principal angles (LDA discriminant subspace alignment)
        ax = axes[2]
        ax.plot(range(L), subspace_angles[:, :, 0].mean(axis=1), marker="o", label="cos(angle 1)")
        ax.plot(range(L), subspace_angles[:, :, 1].mean(axis=1), marker="s", label="cos(angle 2)")
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xticks(range(L))
        ax.set_xlabel("layer")
        ax.set_ylabel("cos sim")
        ax.set_title("LDA discriminant subspace alignment\n(cos of principal angles)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        fig.suptitle("LDA-based B vs C cosine similarity (orthogonal discriminant directions)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140)
        plt.close(fig)

    stats = {
        "within_op_lda_coef_cos": {op: cos_within[op].tolist() for op in common_ops},
        "cross_op_lda_coef_cos": {k: v.tolist() for k, v in cos_cross.items()},
        "subspace_principal_angle_cos": subspace_angles.tolist(),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"\nsaved -> {OUT_PDF}")
    print(f"saved -> {OUT_STATS}")


if __name__ == "__main__":
    main()
