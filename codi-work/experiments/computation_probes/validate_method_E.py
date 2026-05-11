"""Validate Method E (orthogonal to top-3 PCs):

(1) Does E REMOVE context? Probe for question-length quartile (a context-y
    label) on baseline vs Method-E activations. If E is removing context,
    the question-length probe accuracy should drop after E.

(2) Is the REMOVED-part (top-3 PCs) actually context, or did it carry op?
    Compute removed = x - E(x) = projection onto top-3 PCs. Run an LDA on
    those 3-dim projections with operator labels. If E is clean, op
    accuracy should be at majority baseline (~53%). If E damaged op, it
    should be high.
"""

from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

PD = Path(__file__).resolve().parent


def parse_int(s):
    m = re.search(r"answer is\s*:\s*(-?\d+)", s)
    return int(m.group(1)) if m else None


def fit_probe(X, y, n_splits=5, model="lr"):
    y = np.asarray(y); mask = y >= 0
    X = X[mask]; y = y[mask]
    if len(np.unique(y)) < 2: return 0.0
    counts = np.bincount(y); valid = counts[counts > 0]
    n_splits = min(n_splits, int(valid.min()))
    if n_splits < 2: return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        if model == "lr":
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))
        elif model == "lda":
            clf = LinearDiscriminantAnalysis()
        clf.fit(X[tr], y[tr]); accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    print("loading multi-pos decode activations + preds...")
    acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N, P, Lp1, H = acts.shape
    print(f"  acts shape {acts.shape}")
    preds = json.load(open(PD / "svamp_multipos_decode_preds.json"))

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    # context label: question length in chars binned into 5 quartiles (5 buckets)
    qlens = np.array([len(q) for q in questions])
    qlen_bucket = np.digitize(qlens, np.percentile(qlens, [20, 40, 60, 80])).astype(np.int64)
    print(f"  q-length quartile distribution: {np.bincount(qlen_bucket)}")
    print(f"  q-length majority baseline: {np.bincount(qlen_bucket).max()/N*100:.1f}%")

    # Compute Method E activations + the removed part (projection onto top-3 PCs)
    K_E = 3
    print("\n=== Computing Method E (orthogonal to top-3 PCs) and 'removed' (top-3 PC projection) ===")
    E_units_acc = np.zeros((P, Lp1)); E_op = np.zeros((P, Lp1)); E_qlen = np.zeros((P, Lp1))
    base_qlen = np.zeros((P, Lp1)); base_op = np.zeros((P, Lp1))
    rem_op_lda = np.zeros((P, Lp1)); rem_qlen_lda = np.zeros((P, Lp1))
    rem_op_lr = np.zeros((P, Lp1)); rem_qlen_lr = np.zeros((P, Lp1))
    var_explained_top3 = np.zeros((P, Lp1))

    # parse model labels for op-on-emission stuff later if needed
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            ctx = Vt[:K_E]                                    # (3, H)
            top3_proj = Xc @ ctx.T @ ctx                      # the 'removed' part
            E_act = Xc - top3_proj                            # what E keeps
            top3_coords = Xc @ ctx.T                          # the 3-dim top-3-PC coords
            var_explained_top3[p, l] = (S[:K_E]**2).sum() / (S**2).sum()

            # baseline probes for op + qlen
            base_op[p, l]   = fit_probe(X, operators, model="lr")
            base_qlen[p, l] = fit_probe(X, qlen_bucket, model="lr")
            # Method E probes
            E_op[p, l]   = fit_probe(E_act, operators, model="lr")
            E_qlen[p, l] = fit_probe(E_act, qlen_bucket, model="lr")
            # 'removed' probes via LDA + LR (3-dim coords; LDA fine here)
            rem_op_lda[p, l]   = fit_probe(top3_coords, operators, model="lda")
            rem_qlen_lda[p, l] = fit_probe(top3_coords, qlen_bucket, model="lda")
            rem_op_lr[p, l]    = fit_probe(top3_coords, operators, model="lr")
            rem_qlen_lr[p, l]  = fit_probe(top3_coords, qlen_bucket, model="lr")
        print(f"  pos {p:2d}: top3 var={var_explained_top3[p].mean()*100:.1f}%  | "
              f"base_op={base_op[p].max()*100:.1f}  E_op={E_op[p].max()*100:.1f}  | "
              f"base_qlen={base_qlen[p].max()*100:.1f}  E_qlen={E_qlen[p].max()*100:.1f}  | "
              f"removed_op_LDA={rem_op_lda[p].max()*100:.1f}  removed_qlen_LDA={rem_qlen_lda[p].max()*100:.1f}",
              flush=True)

    out = PD / "validate_method_E.json"
    out.write_text(json.dumps({
        "P": P, "L_plus_1": Lp1, "K_E": K_E,
        "qlen_majority_baseline": float(np.bincount(qlen_bucket).max()/N),
        "var_explained_top3":    var_explained_top3.tolist(),
        "base_op":         base_op.tolist(),
        "base_qlen":       base_qlen.tolist(),
        "E_op":            E_op.tolist(),
        "E_qlen":          E_qlen.tolist(),
        "removed_op_lda":  rem_op_lda.tolist(),
        "removed_qlen_lda":rem_qlen_lda.tolist(),
        "removed_op_lr":   rem_op_lr.tolist(),
        "removed_qlen_lr": rem_qlen_lr.tolist(),
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
