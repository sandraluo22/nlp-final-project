"""Canonical operator probe: LDA fit on cf_balanced (80% train, 20% held-out),
applied at each (layer, latent_step) cell. Evaluate on held-out CF AND on
original SVAMP (cross-distribution transfer). Pick the cell with best peak
transfer to SVAMP — that's the probe we use everywhere downstream.

Output:
  canonical_probe_grid.json — full grid of held-out CF and SVAMP transfer acc
  canonical_probe.pkl — fitted LDA + scaler at the chosen (layer, step) cell
"""

from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"

CF_ACTS = REPO / "inference" / "runs" / "cf_balanced_student_gpt2" / "activations.pt"
ORIG_ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
CF_DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"

CLASSES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
CL2IDX = {c: i for i, c in enumerate(CLASSES)}


def main():
    # Load CF activations + labels
    print("loading activations...")
    cf_blob = torch.load(CF_ACTS, map_location="cpu")
    print(f"  CF blob keys: {list(cf_blob.keys()) if isinstance(cf_blob, dict) else type(cf_blob)}")
    if isinstance(cf_blob, dict):
        for k, v in cf_blob.items():
            print(f"    {k}: shape={tuple(v.shape) if hasattr(v, 'shape') else type(v)}")
        # Likely keys: latent / hidden_states / activations
        cf_acts = cf_blob[list(cf_blob.keys())[0]]
    else:
        cf_acts = cf_blob
    cf_acts = cf_acts.to(torch.float32).numpy()
    print(f"  CF acts shape: {cf_acts.shape}")
    cf_data = json.load(open(CF_DATA))
    cf_types = [d["type"] for d in cf_data][:cf_acts.shape[0]]
    cf_labels = np.array([CL2IDX.get(t, -1) for t in cf_types])
    print(f"  CF labels: n={(cf_labels >= 0).sum()}  dist={np.bincount(cf_labels[cf_labels >= 0])}")

    orig_blob = torch.load(ORIG_ACTS, map_location="cpu")
    orig_acts = (orig_blob[list(orig_blob.keys())[0]] if isinstance(orig_blob, dict) else orig_blob).to(torch.float32).numpy()
    print(f"  ORIG SVAMP acts shape: {orig_acts.shape}")
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    orig_types = [ex["Type"] for ex in full][:orig_acts.shape[0]]
    orig_labels = np.array([CL2IDX.get(t, -1) for t in orig_types])
    print(f"  ORIG SVAMP labels: n={(orig_labels >= 0).sum()}  dist={np.bincount(orig_labels[orig_labels >= 0])}")

    # Activation tensor convention: probably (N, layers=13, latent_steps=6, hidden=768)
    # OR (N, latent_steps, layers, hidden). Let's check.
    if cf_acts.shape[1] == 13 and cf_acts.shape[2] == 6:
        Lp1, S = 13, 6
        def get(acts, l, s): return acts[:, l, s, :]
    elif cf_acts.shape[1] == 6 and cf_acts.shape[2] == 13:
        S, Lp1 = 6, 13
        def get(acts, l, s): return acts[:, s, l, :]
    else:
        raise RuntimeError(f"unexpected shape {cf_acts.shape}")

    # 80/20 train/test split on CF (same seed as previous probe)
    SEED = 13
    cf_mask = cf_labels >= 0
    cf_idx = np.arange(len(cf_labels))[cf_mask]
    tr_idx, te_idx = train_test_split(cf_idx, test_size=0.2, random_state=SEED,
                                       stratify=cf_labels[cf_mask])

    grid_train = np.zeros((Lp1, S))
    grid_cftest = np.zeros((Lp1, S))
    grid_orig = np.zeros((Lp1, S))
    print("\n=== LDA per (layer, latent_step) | accuracy on (CF train | CF held-out | SVAMP) ===")
    fitted = {}
    orig_mask = orig_labels >= 0
    for l in range(Lp1):
        for s in range(S):
            X_cf = get(cf_acts, l, s)
            X_or = get(orig_acts, l, s)
            X_tr = X_cf[tr_idx]; y_tr = cf_labels[tr_idx]
            X_te = X_cf[te_idx]; y_te = cf_labels[te_idx]
            sc = StandardScaler().fit(X_tr)
            clf = LinearDiscriminantAnalysis(n_components=3).fit(sc.transform(X_tr), y_tr)
            grid_train[l, s] = clf.score(sc.transform(X_tr), y_tr)
            grid_cftest[l, s] = clf.score(sc.transform(X_te), y_te)
            grid_orig[l, s] = clf.score(sc.transform(X_or[orig_mask]), orig_labels[orig_mask])
            fitted[(l, s)] = (sc, clf)
        print(f"  L{l:02d}: " + " ".join(
            f"s{s}({grid_train[l,s]*100:.0f}/{grid_cftest[l,s]*100:.0f}/{grid_orig[l,s]*100:.0f})"
            for s in range(S)), flush=True)

    # Pick the cell with best (CF_test + SVAMP_transfer)/2 — generalization-balanced
    score = (grid_cftest + grid_orig) / 2
    bl, bs = np.unravel_index(int(np.argmax(score)), score.shape)
    print(f"\n=== Best generalization cell: layer={bl}, latent_step={bs} ===")
    print(f"  CF train={grid_train[bl, bs]*100:.1f}%  CF test={grid_cftest[bl, bs]*100:.1f}%  "
          f"SVAMP transfer={grid_orig[bl, bs]*100:.1f}%")

    out_grid = PD / "canonical_probe_grid.json"
    out_grid.write_text(json.dumps({
        "shape_convention": "(layer, latent_step)",
        "Lp1": Lp1, "S": S,
        "best_cell": [int(bl), int(bs)],
        "best_cf_train": float(grid_train[bl, bs]),
        "best_cf_test": float(grid_cftest[bl, bs]),
        "best_svamp_transfer": float(grid_orig[bl, bs]),
        "grid_cf_train": grid_train.tolist(),
        "grid_cf_test": grid_cftest.tolist(),
        "grid_svamp_transfer": grid_orig.tolist(),
    }, indent=2))
    print(f"saved {out_grid}")

    # Save the fitted scaler+clf at the best cell
    out_pkl = PD / "canonical_probe.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"layer": int(bl), "latent_step": int(bs),
                      "scaler": fitted[(bl, bs)][0],
                      "clf": fitted[(bl, bs)][1],
                      "shape_convention": "(layer, latent_step) on (N, 13, 6, 768)" if cf_acts.shape[1] == 13 else "(latent_step, layer)",
                      "classes": CLASSES}, f)
    print(f"saved {out_pkl}")


if __name__ == "__main__":
    main()
