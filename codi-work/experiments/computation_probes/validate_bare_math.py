"""Three validations of the bare-math operator direction:

1) Cos sim WITHIN bare math:
   - Between the 4 op-mean vectors per (pos, layer): are they well-separated?
   - Between the 6 pairwise difference vectors: are they orthogonal where
     expected, opposite where expected?

2) Confound check:
   - Compute mean activation grouped by NUMBER MAGNITUDE buckets, and by
     position of operator token. Are these similar to op directions?

3) Bare-math accuracy:
   - For each prompt 'a OP b = ', compare the model's first-integer
     prediction to a OP b. Per-operator accuracy.
"""

from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import torch

PD = Path(__file__).resolve().parent
OPS = ["addition", "subtraction", "multiplication", "division"]


def parse_first_int(s):
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def main():
    print("loading bare_math activations + meta...")
    acts = torch.load(PD / "bare_math_acts.pt", map_location="cpu").to(torch.float32).numpy()
    meta = json.load(open(PD / "bare_math_meta.json"))
    examples = meta["meta"]
    preds = meta["preds"]
    P_DECODE = meta["p_decode"]
    N, P, Lp1, H = acts.shape
    print(f"  acts shape: {acts.shape}")

    op_idx = np.array([m["op_idx"] for m in examples])
    a_arr = np.array([m["a"] for m in examples])
    b_arr = np.array([m["b"] for m in examples])

    # === 1. Cos sim within bare math ===
    print("\n=== Cos sim within bare math ===")
    # mean per op
    mean_op = np.zeros((4, P, Lp1, H), dtype=np.float32)
    for c in range(4):
        mean_op[c] = acts[op_idx == c].mean(axis=0)

    # Pairwise: mean op direction in absolute terms (not difference)
    # cos sim averaged over (pos, layer)
    print("  cos sim between mean_op vectors (centered by overall mean):")
    grand_mean = acts.mean(axis=0)            # (P, L+1, H)
    centered = mean_op - grand_mean[None]     # (4, P, L+1, H)
    for s in range(4):
        for t in range(s+1, 4):
            cs_ij = []
            for p in range(P):
                for l in range(Lp1):
                    a = centered[s, p, l]; b = centered[t, p, l]
                    cs_ij.append(float(np.dot(a, b) /
                                       (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)))
            cs_arr = np.array(cs_ij)
            print(f"    {OPS[s]:>15s} vs {OPS[t]:<15s}  mean={cs_arr.mean():+.3f}  "
                  f"max={cs_arr.max():+.3f}  min={cs_arr.min():+.3f}")

    # Pairwise of difference vectors
    print("\n  cos sim between difference vectors:")
    diff_vectors = {}
    for s in range(4):
        for t in range(4):
            if s == t: continue
            diff_vectors[(s, t)] = mean_op[t] - mean_op[s]
    pairs = [
        ((0, 1), (1, 0), "add->sub vs sub->add (should be -1)"),
        ((0, 1), (0, 2), "add->sub vs add->mul (operator-orthogonality?)"),
        ((0, 1), (2, 3), "add->sub vs mul->div (different op pair)"),
        ((0, 2), (1, 3), "add->mul vs sub->div"),
    ]
    for (s1, t1), (s2, t2), desc in pairs:
        v1 = diff_vectors[(s1, t1)]; v2 = diff_vectors[(s2, t2)]
        cs_ij = []
        for p in range(P):
            for l in range(Lp1):
                a = v1[p, l]; b = v2[p, l]
                cs_ij.append(float(np.dot(a, b) /
                                   (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)))
        cs_arr = np.array(cs_ij)
        print(f"    {desc}")
        print(f"      mean={cs_arr.mean():+.3f}  max={cs_arr.max():+.3f}  min={cs_arr.min():+.3f}")

    # === 2. Confound check: magnitude, value-of-a, value-of-b ===
    print("\n=== Confound check ===")
    # Group by a-bucket (4 buckets: 1-5, 6-10, 11-15, 16-20)
    a_bucket = np.digitize(a_arr, [5, 10, 15])  # 0..3
    b_bucket = np.digitize(b_arr, [5, 10, 15])
    mean_a_bucket = np.zeros((4, P, Lp1, H), dtype=np.float32)
    mean_b_bucket = np.zeros((4, P, Lp1, H), dtype=np.float32)
    for c in range(4):
        mean_a_bucket[c] = acts[a_bucket == c].mean(axis=0)
        mean_b_bucket[c] = acts[b_bucket == c].mean(axis=0)

    print("  ||add_dir||_F vs ||a_bucket_diff||_F vs ||b_bucket_diff||_F (sanity)")
    add_diff = mean_op[1] - mean_op[0]                         # add->sub
    a_diff   = mean_a_bucket[3] - mean_a_bucket[0]             # large-a vs small-a
    b_diff   = mean_b_bucket[3] - mean_b_bucket[0]
    print(f"    ||add->sub|| = {np.linalg.norm(add_diff):.2f}")
    print(f"    ||large-a - small-a|| = {np.linalg.norm(a_diff):.2f}")
    print(f"    ||large-b - small-b|| = {np.linalg.norm(b_diff):.2f}")

    print("\n  cos sim of add->sub direction vs a-magnitude direction (per (pos, layer)):")
    cs_amag = []; cs_bmag = []
    for p in range(P):
        for l in range(Lp1):
            v = add_diff[p, l]; a = a_diff[p, l]; b = b_diff[p, l]
            cs_amag.append(float(np.dot(v, a) / (np.linalg.norm(v)*np.linalg.norm(a) + 1e-12)))
            cs_bmag.append(float(np.dot(v, b) / (np.linalg.norm(v)*np.linalg.norm(b) + 1e-12)))
    print(f"    add->sub vs a-magnitude: mean cos={np.mean(cs_amag):+.3f}  "
          f"max abs={np.max(np.abs(cs_amag)):.3f}")
    print(f"    add->sub vs b-magnitude: mean cos={np.mean(cs_bmag):+.3f}  "
          f"max abs={np.max(np.abs(cs_bmag)):.3f}")

    # === 3. Bare-math accuracy ===
    print("\n=== Bare math accuracy ===")
    correct_per_op = {c: [0, 0] for c in range(4)}   # [n_correct, n_total]
    for ex, pred in zip(examples, preds):
        a, b, op_i = ex["a"], ex["b"], ex["op_idx"]
        if op_i == 0: gold = a + b
        elif op_i == 1: gold = a - b
        elif op_i == 2: gold = a * b
        elif op_i == 3:
            if b == 0: continue
            gold = a / b
            if abs(gold - round(gold)) > 0.001:   # non-integer division
                continue
            gold = int(round(gold))
        pred_int = parse_first_int(pred)
        correct_per_op[op_i][1] += 1
        if pred_int == gold:
            correct_per_op[op_i][0] += 1
    total_correct = total_n = 0
    for c in range(4):
        n_c, n_t = correct_per_op[c]
        total_correct += n_c; total_n += n_t
        print(f"  {OPS[c]:>15s}: {n_c:3d}/{n_t:3d}  ({100*n_c/max(n_t,1):.1f}%)")
    print(f"  {'TOTAL':>15s}: {total_correct}/{total_n}  ({100*total_correct/max(total_n,1):.1f}%)")

    # save
    out = PD / "validate_bare_math.json"
    summary = {
        "accuracy_per_op": {OPS[c]: {"correct": correct_per_op[c][0],
                                       "total": correct_per_op[c][1]}
                              for c in range(4)},
        "n_examples": int(N),
        "preds_sample": preds[:10],
        "examples_sample": examples[:10],
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
