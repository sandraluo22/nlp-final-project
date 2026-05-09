"""Test for parallelogram structure between operator centroids.

If operators have compositional structure in activation space, certain
centroid-difference vectors should be approximately parallel:

  (Mul − Add) ?≈ (Div − Sub)        "additive → multiplicative" axis
  (Add − Sub) ?≈ (Mul − Div)        "forward → inverse" axis
  (Mul − Sub) ?≈ (Div − Add)        cross-diagonal

For each (layer, latent_step) we compute centroids per operator on the
cf_balanced student activations and report cosine similarities of these
candidate parallelogram pairs. A score near 1 means the two diff-vectors
align (parallelogram present); near 0 means orthogonal; near -1 means
antiparallel.

Also computes |v|/|w| ratios — a true parallelogram has ratio ≈ 1.
"""

import json
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent
ACTS = REPO / "inference" / "runs" / "cf_balanced_student" / "activations.pt"
DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"
OUT = REPO / "experiments" / "parallelogram_results.json"

CLASSES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
PAIRINGS = {
    "(Mul-Add) vs (Div-Sub)": (
        ("Multiplication", "Addition"),
        ("Common-Division", "Subtraction"),
    ),
    "(Add-Sub) vs (Mul-Div)": (
        ("Addition", "Subtraction"),
        ("Multiplication", "Common-Division"),
    ),
    "(Mul-Sub) vs (Div-Add)": (
        ("Multiplication", "Subtraction"),
        ("Common-Division", "Addition"),
    ),
}


def cos(u: np.ndarray, v: np.ndarray) -> float:
    n = np.linalg.norm(u) * np.linalg.norm(v) + 1e-9
    return float(u @ v / n)


def main():
    print(f"loading {ACTS}")
    acts = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(DATA))
    types = np.array([r["type"] for r in rows])
    N, S, L, H = acts.shape
    print(f"  shape={acts.shape}  types={dict(zip(*np.unique(types, return_counts=True)))}")

    # Per (layer, step) results
    layers = list(range(L))
    steps = list(range(S))
    results = {p: {"cos": np.zeros((L, S)), "norm_a": np.zeros((L, S)),
                   "norm_b": np.zeros((L, S))} for p in PAIRINGS}
    for layer in layers:
        for step in steps:
            slc = acts[:, step, layer, :]
            mu = {c: slc[types == c].mean(axis=0) for c in CLASSES}
            for p_name, ((aA, aB), (bA, bB)) in PAIRINGS.items():
                v = mu[aA] - mu[aB]
                w = mu[bA] - mu[bB]
                results[p_name]["cos"][layer, step] = cos(v, w)
                results[p_name]["norm_a"][layer, step] = float(np.linalg.norm(v))
                results[p_name]["norm_b"][layer, step] = float(np.linalg.norm(w))

    # Summary
    print("\n=== Parallelogram cosine similarities (mean across 6 latent steps) ===")
    print(f"{'layer':>5s}  " + "  ".join(f"{p:>22s}" for p in PAIRINGS))
    for layer in layers:
        row = [f"{results[p]['cos'][layer].mean():>+.3f}" for p in PAIRINGS]
        print(f"{layer:>5d}  " + "  ".join(f"{c:>22s}" for c in row))

    # Best (layer, step) per pairing
    print("\n=== Best (layer, step) per pairing ===")
    for p_name in PAIRINGS:
        m = results[p_name]["cos"]
        idx = np.unravel_index(np.argmax(m), m.shape)
        layer, step = idx
        nv = results[p_name]["norm_a"][layer, step]
        nw = results[p_name]["norm_b"][layer, step]
        print(f"  {p_name:<25s}  best layer={layer} step={step+1}  "
              f"cos={m[layer, step]:+.3f}  ‖v‖={nv:.2f}  ‖w‖={nw:.2f}  "
              f"ratio={nv/max(nw, 1e-9):.2f}")

    # Random-vector baseline (control): how much cosine do we expect from random
    # centroid differences? Sample from per-class data covariance.
    print("\n=== Random-vector control (per layer, mean across steps) ===")
    print("  (cosine between two random class-difference vectors of similar norm)")
    rng = np.random.default_rng(0)
    rand_cos_per_layer = []
    for layer in layers:
        cosines = []
        for step in steps:
            slc = acts[:, step, layer, :]
            mu = {c: slc[types == c].mean(axis=0) for c in CLASSES}
            # 50 random pairings of (cls_a - cls_b) where cls != cls
            for _ in range(50):
                a, b = rng.choice(CLASSES, size=2, replace=False)
                c, d = rng.choice(CLASSES, size=2, replace=False)
                if (a, b) == (c, d) or (a, b) == (d, c):
                    continue
                v = mu[a] - mu[b]
                w = mu[c] - mu[d]
                cosines.append(abs(cos(v, w)))
        rand_cos_per_layer.append(float(np.mean(cosines)))
    for layer in layers:
        print(f"  layer {layer:>2d}: mean |cos| of random class-diff pairs = "
              f"{rand_cos_per_layer[layer]:.3f}")

    # Save
    out_data = {
        "shape": {"L": L, "S": S, "H": H},
        "pairings": {
            p: {
                "cos": results[p]["cos"].tolist(),
                "norm_a": results[p]["norm_a"].tolist(),
                "norm_b": results[p]["norm_b"].tolist(),
            }
            for p in PAIRINGS
        },
        "random_control_per_layer": rand_cos_per_layer,
    }
    OUT.write_text(json.dumps(out_data, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
