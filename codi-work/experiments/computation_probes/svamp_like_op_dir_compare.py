"""From the svamp_like activations (84% accuracy), compute per-op mean
activations on the CORRECT subset, then:
  (1) cos sim with real-SVAMP correct-subset op direction
  (2) cos sim with bare-math op direction
  (3) save the new op direction for steering"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

PD = Path(__file__).resolve().parent
OPS = ["add", "sub", "mul", "div"]


def main():
    print("loading svamp_like activations...")
    acts = torch.load(PD / "svamp_like_acts.pt", map_location="cpu").to(torch.float32).numpy()
    meta = json.load(open(PD / "svamp_like_meta.json"))
    correct = np.array(meta["correct"], dtype=bool)
    op_idx = np.array([m["op_idx"] for m in meta["meta"]])
    print(f"  acts shape: {acts.shape}")
    print(f"  correct: {correct.sum()}/{correct.size}")

    # CORRECT-subset per-op means
    N, P, Lp1, H = acts.shape
    means_correct = np.zeros((4, P, Lp1, H), dtype=np.float32)
    counts = []
    for c in range(4):
        m = (op_idx == c) & correct
        counts.append(int(m.sum()))
        if m.sum() == 0: continue
        means_correct[c] = acts[m].mean(axis=0)
    print(f"  correct-only counts per op: {counts}")

    # Pairwise diffs
    diffs = {}
    for s in range(4):
        for t in range(4):
            if s == t: continue
            d = means_correct[t] - means_correct[s]
            diffs[f"{OPS[s]}->{OPS[t]}"] = d
            print(f"  ||{OPS[s]}->{OPS[t]}||_F = {np.linalg.norm(d):.2f}")

    # Compare: svamp_like vs real SVAMP correct-subset
    print("\n=== svamp_like vs real SVAMP correct-subset op direction ===")
    real = np.load(PD / "redo_op_dir.npz")
    real_means = real["correct_op_means"]   # (4, P=16, L+1=13, H=768)
    cs_grids = {}
    for s in range(4):
        for t in range(s+1, 4):
            key = f"{OPS[s]}->{OPS[t]}"
            v_like = means_correct[t] - means_correct[s]
            v_real = real_means[t] - real_means[s]
            cs = np.zeros((P, Lp1))
            for p in range(P):
                for l in range(Lp1):
                    a = v_like[p, l]; b = v_real[p, l]
                    cs[p, l] = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
            cs_grids[f"like_vs_real_{key}"] = cs
            print(f"  {key:>15s}: mean cos={cs.mean():+.3f}  max={cs.max():+.3f}  min={cs.min():+.3f}")

    # Compare: svamp_like vs bare-math
    print("\n=== svamp_like vs bare-math op direction ===")
    bare = np.load(PD / "bare_math_op_dirs.npz")
    bare_means = bare["mean_op"]   # (4, P=12, L+1=13, H=768)
    P_compare = min(P, bare_means.shape[1])
    for s in range(4):
        for t in range(s+1, 4):
            key = f"{OPS[s]}->{OPS[t]}"
            v_like = means_correct[t][:P_compare] - means_correct[s][:P_compare]
            v_bare = bare_means[t] - bare_means[s]
            cs = np.zeros((P_compare, Lp1))
            for p in range(P_compare):
                for l in range(Lp1):
                    a = v_like[p, l]; b = v_bare[p, l]
                    cs[p, l] = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
            cs_grids[f"like_vs_bare_{key}"] = cs
            print(f"  {key:>15s}: mean cos={cs.mean():+.3f}  max={cs.max():+.3f}  min={cs.min():+.3f}")

    out_npz = PD / "svamp_like_op_dir.npz"
    np.savez(out_npz, correct_op_means=means_correct,
             **{k: v for k, v in diffs.items()},
             **cs_grids)
    print(f"\nsaved {out_npz}")


if __name__ == "__main__":
    main()
