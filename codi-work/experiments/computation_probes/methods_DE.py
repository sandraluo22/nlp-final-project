"""Add Methods D (subtract grand mean) and E (orthogonal to top-3 PCs) to
the existing context_isolation.json — these aim to remove 'context' WITHOUT
removing operator/digit info."""

from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

PD = Path(__file__).resolve().parent


def parse_int(s):
    m = re.search(r"answer is\s*:\s*(-?\d+)", s)
    return int(m.group(1)) if m else None


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
    print("loading multi-pos decode activations + preds...")
    acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N, P, Lp1, H = acts.shape
    print(f"  acts shape {acts.shape}")
    preds = json.load(open(PD / "svamp_multipos_decode_preds.json"))
    model_ans_py = [parse_int(s) for s in preds["preds"]]
    model_units = np.array([(abs(v) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)
    model_tens  = np.array([((abs(v)//10) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)

    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])

    # METHOD D: grand-mean subtraction
    print("\n=== METHOD D: subtract grand mean ===")
    acts_D = acts - acts.mean(axis=0, keepdims=True)
    D_units = np.zeros((P, Lp1)); D_tens = np.zeros((P, Lp1)); D_op = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            D_units[p, l] = fit_probe(acts_D[:, p, l, :], model_units)
            D_tens[p, l]  = fit_probe(acts_D[:, p, l, :], model_tens)
            D_op[p, l]    = fit_probe(acts_D[:, p, l, :], operators)
        print(f"  pos {p:2d}: units {D_units[p].max()*100:.1f}  "
              f"tens {D_tens[p].max()*100:.1f}  op {D_op[p].max()*100:.1f}", flush=True)

    # METHOD E: orthogonal to top-3 PCs
    print("\n=== METHOD E: orthogonal to top-3 PCs ===")
    K_E = 3
    acts_E = acts.copy()
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            ctx = Vt[:K_E]
            proj = Xc @ ctx.T @ ctx
            acts_E[:, p, l, :] = Xc - proj
    E_units = np.zeros((P, Lp1)); E_tens = np.zeros((P, Lp1)); E_op = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            E_units[p, l] = fit_probe(acts_E[:, p, l, :], model_units)
            E_tens[p, l]  = fit_probe(acts_E[:, p, l, :], model_tens)
            E_op[p, l]    = fit_probe(acts_E[:, p, l, :], operators)
        print(f"  pos {p:2d}: units {E_units[p].max()*100:.1f}  "
              f"tens {E_tens[p].max()*100:.1f}  op {E_op[p].max()*100:.1f}", flush=True)

    # Merge into existing JSON
    out = PD / "context_isolation.json"
    if out.exists():
        J = json.load(open(out))
    else:
        J = {"P": P, "L_plus_1": Lp1}
    J["methodD"] = {"units": D_units.tolist(), "tens": D_tens.tolist(), "op": D_op.tolist()}
    J["methodE"] = {"units": E_units.tolist(), "tens": E_tens.tolist(), "op": E_op.tolist()}
    out.write_text(json.dumps(J, indent=2))
    print(f"\nupdated {out}")


if __name__ == "__main__":
    main()
