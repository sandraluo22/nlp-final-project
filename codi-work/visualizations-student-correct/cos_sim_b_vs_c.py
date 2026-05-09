"""Cosine similarity between sc-v2-B and sc-v2-C operator centroids.

For each operator (Add, Sub, Mul) — the 3 operators that exist in both — and
each (layer, latent_step), compute:

    mu_B(L, S, op) = mean activation of B's student-correct rows for op
    mu_C(L, S, op) = mean activation of C's student-correct rows for op
    cos(mu_B, mu_C)

If the operator representation is dataset-invariant (e.g., "this is Addition"
encodes the same way regardless of whether numerals are <100 with at-least-one-
small-input vs <100 with no-Common-Division), the cosines should be high.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parent.parent

OUT_PDF = REPO / "visualizations-student-correct" / "cos_sim_b_vs_c.pdf"
OUT_STATS = REPO / "visualizations-student-correct" / "cos_sim_b_vs_c_stats.json"


def load(name_acts: str, name_meta: str, op_filter: list[str] | None = None):
    """Load a CF dataset's activations + types, filtered to student-correct."""
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


def centroids(acts, types, ops):
    """Return dict op -> (L, S, H) of per-(layer, step) centroid for each op."""
    out = {}
    N, S, L, H = acts.shape
    for op in ops:
        mask = types == op
        if mask.sum() == 0:
            out[op] = None
            continue
        # acts[mask] is (n_op, S, L, H); mean over axis 0 -> (S, L, H); transpose to (L, S, H)
        m = acts[mask].mean(axis=0).transpose(1, 0, 2)
        out[op] = m  # shape (L, S, H)
    return out


def main():
    print("loading B (cf_under99_b, student-correct, 4 ops)")
    acts_b, types_b = load("cf_under99_b", "cf_under99_b")
    print(f"  shape={acts_b.shape}  types={dict(Counter(types_b))}")

    print("loading C (cf_under99 minus Div, student-correct, 3 ops)")
    acts_c, types_c = load(
        "cf_under99", "cf_under99",
        op_filter=["Addition", "Subtraction", "Multiplication"],
    )
    print(f"  shape={acts_c.shape}  types={dict(Counter(types_c))}")

    common_ops = ["Addition", "Subtraction", "Multiplication"]

    # We compare the operator direction defined as
    #   v_op = mu_op - mean(mu_other_ops)
    # i.e. the direction "this operator minus the rest". This is invariant to
    # global-mean and to differences in the OTHER classes' composition.
    def op_directions(acts, types):
        mu = {}
        for op in common_ops:
            m = (types == op)
            if m.sum() == 0:
                mu[op] = None
                continue
            mu[op] = acts[m].mean(axis=0).transpose(1, 0, 2)  # (L, S, H)
        out = {}
        for op in common_ops:
            others = [o for o in common_ops if o != op and mu[o] is not None]
            if mu[op] is None or not others:
                out[op] = None
                continue
            mu_rest = np.mean([mu[o] for o in others], axis=0)
            out[op] = mu[op] - mu_rest
        return out
    cb = op_directions(acts_b, types_b)
    cc = op_directions(acts_c, types_c)

    # Per (op, layer, step) cosine similarity.
    L = acts_b.shape[2]
    S = acts_b.shape[1]
    cos_per = {op: np.zeros((L, S)) for op in common_ops}
    for op in common_ops:
        if cb[op] is None or cc[op] is None:
            cos_per[op][:] = np.nan
            continue
        for layer in range(L):
            for step in range(S):
                cos_per[op][layer, step] = cos(cb[op][layer, step], cc[op][layer, step])

    # Cross-op control: cos sim between mu_B(op_X) and mu_C(op_Y) for op_X != op_Y.
    # Should be much LOWER than within-op if the operator direction is meaningful.
    cross_pairs = [
        ("Addition", "Subtraction"),
        ("Addition", "Multiplication"),
        ("Subtraction", "Multiplication"),
    ]
    cos_cross = {f"{a} vs {b}": np.zeros((L, S)) for a, b in cross_pairs}
    for a, b in cross_pairs:
        for layer in range(L):
            for step in range(S):
                cos_cross[f"{a} vs {b}"][layer, step] = cos(
                    cb[a][layer, step], cc[b][layer, step]
                )

    # ---- Print summary ----
    print("\n=== Within-op cosine similarity (B vs C, mean across 6 latent steps) ===")
    print(f"{'layer':>5s}  " + "  ".join(f"{op[:11]:>11s}" for op in common_ops))
    for layer in range(L):
        row = "  ".join(f"{cos_per[op][layer].mean():>+10.3f} " for op in common_ops)
        print(f"  {layer:>3d}  {row}")
    print("\n  (1.0 = identical direction; 0 = orthogonal; -1 = anti-parallel)")

    print("\n=== Cross-op control (B-op_X vs C-op_Y; should be lower) ===")
    print(f"{'layer':>5s}  " + "  ".join(f"{k[:18]:>18s}" for k in cos_cross))
    for layer in range(L):
        row = "  ".join(f"{v[layer].mean():>+17.3f} " for v in cos_cross.values())
        print(f"  {layer:>3d}  {row}")

    # ---- Plot: per-op line plot of cos sim by layer (mean across steps) ----
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: within-op cos sim by layer
        ax = axes[0]
        for op in common_ops:
            ax.plot(range(L), cos_per[op].mean(axis=1), marker="o", label=op)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(range(L))
        ax.set_xlabel("layer")
        ax.set_ylabel("cosine similarity")
        ax.set_title("Within-op centroid cos sim: B vs C\n(same operator across the two CFs)")
        ax.set_ylim(-0.2, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Right: cross-op control
        ax = axes[1]
        for k, v in cos_cross.items():
            ax.plot(range(L), v.mean(axis=1), marker="s", label=k)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(range(L))
        ax.set_xlabel("layer")
        ax.set_ylabel("cosine similarity")
        ax.set_title("Cross-op control\n(B-op_X vs C-op_Y, should be lower)")
        ax.set_ylim(-0.2, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        fig.suptitle("Operator centroid cosine similarity: sc-v2-B (cf_under99_b) vs sc-v2-C (cf_under99 − Div)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140)
        plt.close(fig)

    # Save numeric stats
    stats = {
        "within_op_per_layer_mean": {
            op: cos_per[op].mean(axis=1).tolist() for op in common_ops
        },
        "within_op_per_layer_step": {
            op: cos_per[op].tolist() for op in common_ops
        },
        "cross_op_per_layer_mean": {
            k: v.mean(axis=1).tolist() for k, v in cos_cross.items()
        },
        "shape": {"L": L, "S": S},
        "n_per_dataset": {
            "B": int(acts_b.shape[0]),
            "C": int(acts_c.shape[0]),
        },
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"\nsaved -> {OUT_PDF}")
    print(f"saved -> {OUT_STATS}")


if __name__ == "__main__":
    main()
