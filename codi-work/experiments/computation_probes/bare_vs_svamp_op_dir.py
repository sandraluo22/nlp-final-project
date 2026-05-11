"""Compare op-direction vectors derived from bare arithmetic vs from full
SVAMP. Cos similarity per (pos, layer) tells us whether the bare-math
'add - sub' direction lives in the same subspace as the SVAMP one — i.e.,
whether the operator feature is universal or context-dependent.
"""

from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset

PD = Path(__file__).resolve().parent


def main():
    # Load SVAMP activations + operator labels
    print("loading SVAMP activations...")
    svamp_acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N_s, P_s, Lp1_s, H = svamp_acts.shape
    print(f"  svamp shape {svamp_acts.shape}")

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])

    # SVAMP op means at SVAMP positions
    svamp_mean_op = np.zeros((4, P_s, Lp1_s, H), dtype=np.float32)
    for c in range(4):
        m = operators == c
        if m.sum() == 0: continue
        svamp_mean_op[c] = svamp_acts[m].mean(axis=0)

    # Load bare-math op means
    print("loading bare-math op directions...")
    bare = np.load(PD / "bare_math_op_dirs.npz")
    bare_mean_op = bare["mean_op"]   # (4, P_b, L+1, H)
    print(f"  bare mean_op shape: {bare_mean_op.shape}")

    P_b, Lp1_b = bare_mean_op.shape[1], bare_mean_op.shape[2]
    P_compare = min(P_s, P_b)
    Lp1 = min(Lp1_s, Lp1_b)

    # for each operator pair (s, t), compare:
    #   svamp_diff = mean_t - mean_s  (SVAMP)
    #   bare_diff  = mean_t - mean_s  (bare)
    # cos sim per (pos, layer)
    OP_NAMES = ["add", "sub", "mul", "div"]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    cos_sim_per_pair = {}
    for s, t in pairs:
        svamp_diff = svamp_mean_op[t] - svamp_mean_op[s]   # (P, L+1, H)
        bare_diff  = bare_mean_op[t]  - bare_mean_op[s]
        cs = np.zeros((P_compare, Lp1))
        for p in range(P_compare):
            for l in range(Lp1):
                a = svamp_diff[p, l]; b = bare_diff[p, l]
                cs[p, l] = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        cos_sim_per_pair[f"{OP_NAMES[s]}->{OP_NAMES[t]}"] = cs
        print(f"  {OP_NAMES[s]:>3}->{OP_NAMES[t]:<3}: mean cos_sim={cs.mean():+.3f}  "
              f"max={cs.max():+.3f}  min={cs.min():+.3f}")

    out = PD / "bare_vs_svamp_op_dir.npz"
    np.savez(out, **cos_sim_per_pair, P=P_compare, Lp1=Lp1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
