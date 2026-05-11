"""Use the SAME probe method that gave 96.5% held-out on real SVAMP at
(decode pos 1, layer 8): train on 800 of 1000 SVAMP examples, evaluate on
held-out 200. Apply this method consistently across:
  - prompt_end at every layer
  - each of 6 latent steps × 13 layers
  - decode position 1, layer 8 (sanity check that we reproduce 96.5%)

Output: a (8 stages × 13 layers) grid of held-out probe accuracy.
Stages: prompt_end, latent_step_0..5, decode_pos1.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PD = Path(__file__).resolve().parent


def fit_eval(X_tr, y_tr, X_te, y_te):
    mask_tr = y_tr >= 0; mask_te = y_te >= 0
    sc = StandardScaler().fit(X_tr[mask_tr])
    clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs").fit(
        sc.transform(X_tr[mask_tr]), y_tr[mask_tr])
    return float(clf.score(sc.transform(X_te[mask_te]), y_te[mask_te]))


def main():
    print("loading real SVAMP latent + prompt_end activations...")
    blob = torch.load(PD / "svamp_real_latent_acts.pt", map_location="cpu")
    latent = blob["latent"].to(torch.float32).numpy()      # (N=1000, 6, L+1=13, H=768)
    pe = blob["prompt_end"].to(torch.float32).numpy()      # (N, L+1, H)
    decode_acts = torch.load(PD / "svamp_fixed_acts.pt", map_location="cpu").to(torch.float32).numpy()
    # decode_acts: (N, P=16, L+1, H), use position 1
    print(f"  latent {latent.shape}, prompt_end {pe.shape}, decode {decode_acts.shape}")

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])

    np.random.seed(0)
    perm = np.random.permutation(len(operators))
    train_idx = perm[:800]
    test_idx  = perm[800:]
    y_tr = operators[train_idx]; y_te = operators[test_idx]

    Lp1 = pe.shape[1]
    n_stages = 1 + 6 + 1   # prompt_end, 6 latent steps, decode pos 1
    grid = np.zeros((n_stages, Lp1))
    stage_names = ["prompt_end"] + [f"latent_{s}" for s in range(6)] + ["decode_pos1"]

    print("\n=== Probe accuracy (train on 800, eval on held-out 200) ===")
    # prompt_end
    for l in range(Lp1):
        Xa = pe[train_idx, l, :]; Xb = pe[test_idx, l, :]
        grid[0, l] = fit_eval(Xa, y_tr, Xb, y_te)
    # 6 latent steps
    for s in range(6):
        for l in range(Lp1):
            Xa = latent[train_idx, s, l, :]; Xb = latent[test_idx, s, l, :]
            grid[1 + s, l] = fit_eval(Xa, y_tr, Xb, y_te)
    # decode pos 1
    for l in range(Lp1):
        Xa = decode_acts[train_idx, 1, l, :]; Xb = decode_acts[test_idx, 1, l, :]
        grid[7, l] = fit_eval(Xa, y_tr, Xb, y_te)

    print(f"{'stage':>14s}: " + "  ".join(f"L{l:02d}" for l in range(Lp1)))
    for i, name in enumerate(stage_names):
        print(f"{name:>14s}: " + "  ".join(f"{grid[i, l]*100:5.1f}" for l in range(Lp1)))
    # peak per stage
    print("\n=== Peak per stage ===")
    for i, name in enumerate(stage_names):
        print(f"  {name:>14s}: {grid[i].max()*100:.1f}% @ L{int(np.argmax(grid[i]))}")

    out = PD / "probe_real_svamp_latent.json"
    out.write_text(json.dumps({
        "grid": grid.tolist(),
        "stage_names": stage_names,
        "majority_baseline": float(np.bincount(y_te[y_te >= 0]).max() / (y_te >= 0).sum()),
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
