"""Three methods to subtract 'context' from CODI-GPT-2 activations and see
what's left for the actual numbers / operators. Run probes on the residual
to see if the math content survives context subtraction.

Method A: Operator-conditional mean subtraction.
    For each operator class c, compute the mean activation over examples with
    op=c at each (pos, layer). For each example, subtract its operator's
    mean. What remains should be operator-orthogonal — anything that varies
    BETWEEN examples WITH THE SAME OPERATOR. That's where digit/value info
    should live (since the operator is averaged out).

Method B: PCA-based context-subspace removal.
    For each (pos, layer), do PCA on the 1000 activations. Define the
    'context subspace' as top-k PCs that explain X% of variance. Project
    each activation orthogonal to it. Probe the residual.
    (k chosen to capture 80% of variance.)

Method C: CF-difference numeral subspace (uses NEW captures of vary_a/vary_b).
    Δx = x(vary_a) - x(original) per (pos, layer). PCA → 'numeral subspace'.
    Project x onto its complement → 'context'. Subtract → 'numeral signal'.
    Requires capturing activations on vary_a/vary_b examples (this script
    uses cached activations if present, otherwise just runs A and B).

After each method, refit per-(pos, layer) probes for {units, tens, operator}
on the residualized activations and compare accuracy to the baseline (no
residualization).
"""

from __future__ import annotations
import json, os, re, sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"


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

    # model labels
    model_ans_py = [parse_int(s) for s in preds["preds"]]
    model_units = np.array([(abs(v) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)
    model_tens  = np.array([((abs(v)//10) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])

    # =====================================================================
    # BASELINE probes (no residualization) — copy from earlier results
    # =====================================================================
    print("\n=== BASELINE (no residualization) — fitting probes ===")
    base_units = np.zeros((P, Lp1)); base_tens = np.zeros((P, Lp1)); base_op = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            base_units[p, l] = fit_probe(acts[:, p, l, :], model_units)
            base_tens[p, l]  = fit_probe(acts[:, p, l, :], model_tens)
            base_op[p, l]    = fit_probe(acts[:, p, l, :], operators)
        print(f"  pos {p:2d}: units max={base_units[p].max()*100:.1f}  "
              f"tens max={base_tens[p].max()*100:.1f}  op max={base_op[p].max()*100:.1f}", flush=True)

    # =====================================================================
    # METHOD A: subtract operator-conditional mean
    # =====================================================================
    print("\n=== METHOD A: subtract operator-conditional mean ===")
    acts_A = acts.copy()
    for c in range(4):
        mask = operators == c
        if mask.sum() == 0: continue
        mu_c = acts[mask].mean(axis=0, keepdims=True)  # (1, P, L+1, H)
        acts_A[mask] = acts[mask] - mu_c
    print(f"  per-op means subtracted ({(operators >= 0).sum()} examples)")
    A_units = np.zeros((P, Lp1)); A_tens = np.zeros((P, Lp1)); A_op = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            A_units[p, l] = fit_probe(acts_A[:, p, l, :], model_units)
            A_tens[p, l]  = fit_probe(acts_A[:, p, l, :], model_tens)
            A_op[p, l]    = fit_probe(acts_A[:, p, l, :], operators)
        print(f"  pos {p:2d}: units {A_units[p].max()*100:.1f}  "
              f"tens {A_tens[p].max()*100:.1f}  op {A_op[p].max()*100:.1f}  "
              f"(op should drop sharply)", flush=True)

    # =====================================================================
    # METHOD B: PCA context-subspace orthogonalization
    # =====================================================================
    print("\n=== METHOD B: project orthogonal to top-k PCA (k captures 80% var) ===")
    acts_B = acts.copy()
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            Xc = X - X.mean(axis=0, keepdims=True)
            # SVD
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            var = S ** 2; cumvar = np.cumsum(var) / var.sum()
            k = int(np.searchsorted(cumvar, 0.80) + 1)
            # context subspace = top-k right singular vectors (Vt[:k])
            ctx_basis = Vt[:k]                  # (k, H)
            proj = Xc @ ctx_basis.T @ ctx_basis  # (N, H)
            acts_B[:, p, l, :] = Xc - proj      # orthogonal complement
        print(f"  pos {p:2d}: residualized ({k} PCs removed at last cell)", flush=True)
    B_units = np.zeros((P, Lp1)); B_tens = np.zeros((P, Lp1)); B_op = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            B_units[p, l] = fit_probe(acts_B[:, p, l, :], model_units)
            B_tens[p, l]  = fit_probe(acts_B[:, p, l, :], model_tens)
            B_op[p, l]    = fit_probe(acts_B[:, p, l, :], operators)
        print(f"  pos {p:2d}: units {B_units[p].max()*100:.1f}  "
              f"tens {B_tens[p].max()*100:.1f}  op {B_op[p].max()*100:.1f}", flush=True)

    # =====================================================================
    # METHOD C: prose-prediction residual
    # We approximate a 'prose-only' baseline by predicting each example's
    # activation from a low-dim summary of its prose tokens (here: the
    # operator-class as a 4-way one-hot, since we don't have prose embeddings
    # at hand — this is essentially Method A but framed regressively, so we
    # use it as a sanity check).
    # =====================================================================
    print("\n=== METHOD C: residual after predicting from operator one-hot ===")
    print("  (regression-based version of Method A; expect very similar result)")
    from sklearn.linear_model import Ridge
    op_onehot = np.eye(4)[operators]  # (N, 4); rows where op<0 get a row of zeros
    op_onehot[operators < 0] = 0
    acts_C = acts.copy()
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            ridge = Ridge(alpha=1.0).fit(op_onehot, X)
            acts_C[:, p, l, :] = X - ridge.predict(op_onehot)
    C_units = np.zeros((P, Lp1)); C_tens = np.zeros((P, Lp1)); C_op = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            C_units[p, l] = fit_probe(acts_C[:, p, l, :], model_units)
            C_tens[p, l]  = fit_probe(acts_C[:, p, l, :], model_tens)
            C_op[p, l]    = fit_probe(acts_C[:, p, l, :], operators)
        print(f"  pos {p:2d}: units {C_units[p].max()*100:.1f}  "
              f"tens {C_tens[p].max()*100:.1f}  op {C_op[p].max()*100:.1f}", flush=True)

    # =====================================================================
    # METHOD D: subtract grand mean only (zero-center)  — removes constant
    # offset but preserves operator/digit info entirely
    # =====================================================================
    print("\n=== METHOD D: subtract grand mean (preserve operator+digit) ===")
    acts_D = acts - acts.mean(axis=0, keepdims=True)
    D_units = np.zeros((P, Lp1)); D_tens = np.zeros((P, Lp1)); D_op = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            D_units[p, l] = fit_probe(acts_D[:, p, l, :], model_units)
            D_tens[p, l]  = fit_probe(acts_D[:, p, l, :], model_tens)
            D_op[p, l]    = fit_probe(acts_D[:, p, l, :], operators)
        print(f"  pos {p:2d}: units {D_units[p].max()*100:.1f}  "
              f"tens {D_tens[p].max()*100:.1f}  op {D_op[p].max()*100:.1f}", flush=True)

    # =====================================================================
    # METHOD E: project orthogonal to top-3 PCs only (small k) — removes the
    # biggest "shared mode" (likely template/context) but should keep operator
    # and digits which are typically NOT in the very top PCs.
    # =====================================================================
    print("\n=== METHOD E: orthogonal to top-3 PCs (preserve operator+digit) ===")
    acts_E = acts.copy()
    K_E = 3
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            ctx_basis = Vt[:K_E]
            proj = Xc @ ctx_basis.T @ ctx_basis
            acts_E[:, p, l, :] = Xc - proj
    E_units = np.zeros((P, Lp1)); E_tens = np.zeros((P, Lp1)); E_op = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            E_units[p, l] = fit_probe(acts_E[:, p, l, :], model_units)
            E_tens[p, l]  = fit_probe(acts_E[:, p, l, :], model_tens)
            E_op[p, l]    = fit_probe(acts_E[:, p, l, :], operators)
        print(f"  pos {p:2d}: units {E_units[p].max()*100:.1f}  "
              f"tens {E_tens[p].max()*100:.1f}  op {E_op[p].max()*100:.1f}", flush=True)

    # =====================================================================
    # SAVE
    # =====================================================================
    out = PD / "context_isolation.json"
    out.write_text(json.dumps({
        "baseline": {"units": base_units.tolist(), "tens": base_tens.tolist(), "op": base_op.tolist()},
        "methodA":  {"units": A_units.tolist(),    "tens": A_tens.tolist(),    "op": A_op.tolist()},
        "methodB":  {"units": B_units.tolist(),    "tens": B_tens.tolist(),    "op": B_op.tolist()},
        "methodC":  {"units": C_units.tolist(),    "tens": C_tens.tolist(),    "op": C_op.tolist()},
        "methodD":  {"units": D_units.tolist(),    "tens": D_tens.tolist(),    "op": D_op.tolist()},
        "methodE":  {"units": E_units.tolist(),    "tens": E_tens.tolist(),    "op": E_op.tolist()},
        "P": P, "L_plus_1": Lp1,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
