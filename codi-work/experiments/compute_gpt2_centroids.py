"""Compute per-class centroids of CODI-GPT2 cf_balanced activations, save to
a pickle for use as steering directions.

Mirrors the lda_means field of the Huginn steering cache: shape
  (n_classes=4, latent_steps=6, layers+1=13, hidden=768)
Plus per-class counts and the class label order.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]   # codi-work/
ACTS = REPO / "inference" / "runs" / "cf_balanced_student_gpt2" / "activations.pt"
CF_DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"
OUT = REPO / "experiments" / "gpt2_cf_centroids.pkl"


def main():
    print(f"loading {ACTS}", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    print(f"  shape={a.shape}  bytes={a.nbytes/1e9:.2f} GB", flush=True)
    rows = json.load(open(CF_DATA))
    types = np.array([r["type"] for r in rows])
    print(f"  type counts: {dict(Counter(types))}", flush=True)

    classes = sorted(set(types))
    print(f"  classes (alphabetical): {classes}")
    N, S, L, H = a.shape

    means = np.empty((len(classes), S, L, H), dtype=np.float32)
    counts = np.empty((len(classes),), dtype=np.int32)
    for ci, cls in enumerate(classes):
        mask = types == cls
        counts[ci] = int(mask.sum())
        means[ci] = a[mask].mean(axis=0)
        print(f"  centroid for {cls}: n={counts[ci]}  ‖μ‖_avg(per layer×step)={np.linalg.norm(means[ci], axis=-1).mean():.2f}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(
            {
                "means": means,         # (n_classes, S, L, H)
                "classes": classes,     # sorted list
                "counts": counts,
                "shape": list(a.shape),
            },
            f,
        )
    print(f"\nsaved -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
