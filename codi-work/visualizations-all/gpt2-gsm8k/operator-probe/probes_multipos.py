"""Per-(decode-position × layer) variable probes on the multi-position decode
activations. Tells us at which token position the units/tens/operator info is
strongest. Operator answer position should show the answer NUMBER digits."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"


def fit_probe(X, y, n_splits=5):
    """Cross-validated logistic regression accuracy. Filters y<0."""
    y = np.asarray(y)
    mask = y >= 0
    X = X[mask]; y = y[mask]
    if len(np.unique(y)) < 2:
        return 0.0
    counts = np.bincount(y)
    valid = counts[counts > 0]
    n_splits = min(n_splits, int(valid.min()))
    if n_splits < 2:
        return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0,
                                               solver="lbfgs"))
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    print("loading multi-position decode activations...")
    acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    # shape (N, P, L+1, H)
    N, P, Lp1, H = acts.shape
    print(f"  acts shape {acts.shape}")

    # pull labels from SVAMP
    ds = load_dataset("gsm8k")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2, "division": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])
    answers = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    units = np.array([int(abs(int(round(a))) % 10) for a in answers])
    tens = np.array([int((abs(int(round(a))) // 10) % 10) for a in answers])

    # also load preds to know what the model is actually emitting at each pos
    preds = json.load(open(PD / "svamp_multipos_decode_preds.json"))
    pred_ids_per_pos = np.array(preds["pred_ids_per_pos"])  # (N, P)
    # which token id is most common per position
    from collections import Counter
    top_tok_per_pos = []
    for p in range(P):
        c = Counter(pred_ids_per_pos[:, p].tolist())
        top_tok_per_pos.append(c.most_common(3))
    print(f"  most-common token per position (id -> count):")
    for p, tops in enumerate(top_tok_per_pos):
        print(f"    pos {p}: {tops}")

    # fit per (pos, layer) probes
    units_acc = np.zeros((P, Lp1))
    tens_acc = np.zeros((P, Lp1))
    op_acc = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            units_acc[p, l] = fit_probe(X, units)
            tens_acc[p, l] = fit_probe(X, tens)
            op_acc[p, l] = fit_probe(X, operators)
        print(f"  pos {p:2d}: units max={units_acc[p].max()*100:.1f}%  "
              f"tens max={tens_acc[p].max()*100:.1f}%  "
              f"op max={op_acc[p].max()*100:.1f}%", flush=True)

    out = PD / "gpt2_multipos_probes.json"
    out.write_text(json.dumps({
        "units_acc": units_acc.tolist(),
        "tens_acc": tens_acc.tolist(),
        "operator_acc": op_acc.tolist(),
        "P": P, "L_plus_1": Lp1,
        "top_tok_per_pos": top_tok_per_pos,
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
