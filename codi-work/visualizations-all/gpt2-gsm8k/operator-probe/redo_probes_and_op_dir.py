"""With the fixed SVAMP pipeline (39.2% accuracy), redo:
  - per-(pos, layer) probes for {gold_units, gold_tens, gold_operator,
    model_correct} using the corrected activations
  - operator direction derived from CORRECT examples only — these are the
    cases where the model genuinely solved the problem
  - cos similarity of clean-op-direction vs bare-math op-direction
"""

from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

PD = Path(__file__).resolve().parent


def fit_probe(X, y, n_splits=5):
    y = np.asarray(y); mask = y >= 0
    X = X[mask]; y = y[mask]
    if len(np.unique(y)) < 2: return 0.0
    counts = np.bincount(y); valid = counts[counts > 0]
    n_splits = min(n_splits, int(valid.min()))
    if n_splits < 2: return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))
        clf.fit(X[tr], y[tr]); accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    print("loading fixed SVAMP activations + preds...")
    acts = torch.load(PD / "svamp_fixed_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N, P, Lp1, H = acts.shape
    print(f"  acts shape {acts.shape}")
    preds = json.load(open(PD / "svamp_fixed_preds.json"))
    correct_mask = np.array(preds["correct"], dtype=bool)
    pred_floats = [None if pf is None else pf for pf in preds["pred_floats"]]
    print(f"  model exact-correct: {correct_mask.sum()}/{N} = {100*correct_mask.mean():.1f}%")

    ds = load_dataset("gsm8k")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])
    answers = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    gold_units = np.array([int(abs(int(round(a))) % 10) for a in answers])
    gold_tens  = np.array([int((abs(int(round(a))) // 10) % 10) for a in answers])

    # Print per-op accuracy
    print("\n  per-op accuracy:")
    for c, name in [(0, "addition"), (1, "subtraction"), (2, "multiplication"), (3, "division")]:
        m = operators == c
        if m.sum() == 0: continue
        n_c = int(correct_mask[m].sum()); n_total = int(m.sum())
        print(f"    {name:>15s}: {n_c}/{n_total} = {100*n_c/n_total:.1f}%")

    # =====================================================================
    # PROBES: per-(pos, layer) for op (gold), gold_units, gold_tens, correctness
    # =====================================================================
    print("\n=== Probes per (pos, layer): {gold operator, gold units, gold tens, correctness} ===")
    op_acc = np.zeros((P, Lp1)); units_acc = np.zeros((P, Lp1))
    tens_acc = np.zeros((P, Lp1)); cor_acc = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            op_acc[p, l]    = fit_probe(X, operators)
            units_acc[p, l] = fit_probe(X, gold_units)
            tens_acc[p, l]  = fit_probe(X, gold_tens)
            cor_acc[p, l]   = fit_probe(X, correct_mask.astype(int))
        print(f"  pos {p:2d}: op {op_acc[p].max()*100:.1f}  units {units_acc[p].max()*100:.1f}  "
              f"tens {tens_acc[p].max()*100:.1f}  correct? {cor_acc[p].max()*100:.1f}", flush=True)

    # =====================================================================
    # Operator direction from CORRECT subset only (clean computation)
    # =====================================================================
    print("\n=== Op direction from CORRECT subset ===")
    correct_acts = acts[correct_mask]
    correct_ops = operators[correct_mask]
    print(f"  correct subset: {correct_acts.shape[0]} examples")
    correct_op_means = np.zeros((4, P, Lp1, H), dtype=np.float32)
    for c in range(4):
        m = correct_ops == c
        if m.sum() == 0: continue
        correct_op_means[c] = correct_acts[m].mean(axis=0)
        print(f"  op {c}: n={m.sum()} examples")

    # Mean diff vectors
    OP_NAMES = ["add", "sub", "mul", "div"]
    diff_clean = {}
    for s in range(4):
        for t in range(4):
            if s == t: continue
            v = correct_op_means[t] - correct_op_means[s]
            diff_clean[f"{OP_NAMES[s]}->{OP_NAMES[t]}"] = v
            print(f"  ||{OP_NAMES[s]}->{OP_NAMES[t]}||_F = {np.linalg.norm(v):.2f}")

    # =====================================================================
    # Cos sim clean vs bare-math
    # =====================================================================
    print("\n=== Cos sim: clean (correct-only SVAMP) vs bare-math op direction ===")
    bare = np.load(PD / "bare_math_op_dirs.npz")
    P_compare = min(P, bare["mean_op"].shape[1])
    cs_grids = {}
    for s in range(4):
        for t in range(s+1, 4):
            key = f"{OP_NAMES[s]}->{OP_NAMES[t]}"
            v_clean = correct_op_means[t] - correct_op_means[s]
            v_bare  = bare["mean_op"][t] - bare["mean_op"][s]
            cs = np.zeros((P_compare, Lp1))
            for p in range(P_compare):
                for l in range(Lp1):
                    a = v_clean[p, l]; b = v_bare[p, l]
                    cs[p, l] = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
            cs_grids[key] = cs
            print(f"  {key:>15s}: mean cos={cs.mean():+.3f}  max={cs.max():+.3f}  min={cs.min():+.3f}")

    # =====================================================================
    # SAVE
    # =====================================================================
    out_json = PD / "redo_probes_summary.json"
    out_json.write_text(json.dumps({
        "n": int(N),
        "n_correct": int(correct_mask.sum()),
        "accuracy": float(correct_mask.mean()),
        "op_acc":    op_acc.tolist(),
        "units_acc": units_acc.tolist(),
        "tens_acc":  tens_acc.tolist(),
        "cor_acc":   cor_acc.tolist(),
        "P": int(P), "L_plus_1": int(Lp1),
    }, indent=2))
    print(f"\nsaved {out_json}")
    out_npz = PD / "redo_op_dir.npz"
    np.savez(out_npz, correct_op_means=correct_op_means,
             **{k: v for k, v in diff_clean.items()},
             **{f"cossim_{k}": v for k, v in cs_grids.items()})
    print(f"saved {out_npz}")


if __name__ == "__main__":
    main()
