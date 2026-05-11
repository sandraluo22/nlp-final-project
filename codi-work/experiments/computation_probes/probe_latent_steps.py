"""Probe operator at each (latent_step, layer) cell + prompt_end (per-layer).

Tells us where in CODI's pipeline operator info first appears and how it
evolves across the 6 latent loop steps.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

PD = Path(__file__).resolve().parent


def fit_probe(X, y, n_splits=5):
    y = np.asarray(y); mask = y >= 0
    X = X[mask]; y = y[mask]
    if len(np.unique(y)) < 2: return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))
        clf.fit(X[tr], y[tr]); accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    print("loading svamp_like latent activations...")
    blob = torch.load(PD / "svamp_like_latent_acts.pt", map_location="cpu")
    latent = blob["latent"].to(torch.float32).numpy()        # (N, 6, L+1, H)
    pe = blob["prompt_end"].to(torch.float32).numpy()        # (N, L+1, H)
    meta = json.load(open(PD / "svamp_like_latent_meta.json"))
    op_idx = np.array([m["op_idx"] for m in meta["meta"]])
    correct = np.array(meta["correct"], dtype=bool)
    N, S, Lp1, H = latent.shape
    print(f"  latent: {latent.shape}, prompt_end: {pe.shape}")

    # === Per-(latent_step, layer) probe accuracy for operator ===
    print("\n=== Operator probe accuracy per (latent_step, layer) ===")
    grid = np.zeros((S + 1, Lp1))   # row 0 = prompt_end, rows 1..6 = latent steps 0..5
    # prompt_end probes
    for l in range(Lp1):
        grid[0, l] = fit_probe(pe[:, l, :], op_idx)
    print(f"  prompt_end:    " + "  ".join(f"L{l:02d}={grid[0, l]*100:5.1f}" for l in range(Lp1)))
    # latent step probes
    for s in range(S):
        for l in range(Lp1):
            grid[s+1, l] = fit_probe(latent[:, s, l, :], op_idx)
        print(f"  latent step {s}: " + "  ".join(f"L{l:02d}={grid[s+1, l]*100:5.1f}" for l in range(Lp1)))

    # peak per row
    print("\n=== Peak per stage ===")
    print(f"  prompt_end:    peak {grid[0].max()*100:.1f}% @ L{int(np.argmax(grid[0]))}")
    for s in range(S):
        print(f"  latent step {s}: peak {grid[s+1].max()*100:.1f}% @ L{int(np.argmax(grid[s+1]))}")

    # also do the same restricted to CORRECT subset (optional)
    print("\n=== Operator probe (correct-only subset) ===")
    grid_c = np.zeros((S + 1, Lp1))
    pe_c = pe[correct]; lat_c = latent[correct]; op_c = op_idx[correct]
    for l in range(Lp1):
        grid_c[0, l] = fit_probe(pe_c[:, l, :], op_c)
    for s in range(S):
        for l in range(Lp1):
            grid_c[s+1, l] = fit_probe(lat_c[:, s, l, :], op_c)
    print(f"  prompt_end:    peak {grid_c[0].max()*100:.1f}% @ L{int(np.argmax(grid_c[0]))}")
    for s in range(S):
        print(f"  latent step {s}: peak {grid_c[s+1].max()*100:.1f}% @ L{int(np.argmax(grid_c[s+1]))}")

    out = PD / "probe_latent_steps.json"
    out.write_text(json.dumps({
        "stages": ["prompt_end"] + [f"latent_step_{s}" for s in range(S)],
        "all_examples":  grid.tolist(),
        "correct_only":  grid_c.tolist(),
        "majority_baseline_4class": float(np.bincount(op_idx[op_idx >= 0]).max() / (op_idx >= 0).sum()),
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
