"""Refit per-(pos × layer) units/tens probes against the MODEL'S emitted
answer instead of the gold answer. This isolates 'what the residual encodes
about the model's own commitment' from 'what the residual encodes about the
ground truth'."""

from __future__ import annotations
import json, re
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


def fit_probe(X, y, n_splits=5):
    y = np.asarray(y)
    mask = y >= 0
    X = X[mask]; y = y[mask]
    if len(np.unique(y)) < 2: return 0.0
    counts = np.bincount(y)
    valid = counts[counts > 0]
    n_splits = min(n_splits, int(valid.min()))
    if n_splits < 2: return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def parse_model_answer(pred_str):
    """Extract the first integer the model emits after 'answer is: '."""
    m = re.search(r"answer is\s*:\s*(-?\d+)", pred_str)
    if not m: return None
    try: return int(m.group(1))
    except: return None


def main():
    print("loading multi-pos decode activations + preds...")
    acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N, P, Lp1, H = acts.shape
    print(f"  acts shape {acts.shape}")
    preds = json.load(open(PD / "svamp_multipos_decode_preds.json"))

    # parse model answers (Python ints; can be arbitrarily large)
    model_ans_py = [parse_model_answer(s) for s in preds["preds"]]
    valid_mask = np.array([v is not None for v in model_ans_py])
    print(f"  parsed model-answer for {valid_mask.sum()}/{N} examples")

    # model digit labels — keep math in Python to avoid int64 overflow
    model_units = np.array([(abs(v) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)
    model_tens  = np.array([((abs(v) // 10) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)

    # gold labels for comparison
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    answers = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    gold_units = np.array([int(abs(int(round(a))) % 10) for a in answers])
    gold_tens = np.array([int((abs(int(round(a))) // 10) % 10) for a in answers])

    # operator: derive what op the model "implicitly used" by checking
    # which of {a+b, a-b, a*b, a/b} the model's answer matches.
    # SVAMP examples have ex.Equation; or we can scan for numbers in question.
    # Simplest: keep gold-operator labels (we already trust those).
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])

    # match-with-gold counts (Python ints, then to bool array)
    gold_int = [int(round(a)) for a in answers]
    correct_mask = np.array([(v is not None and v == g)
                             for v, g in zip(model_ans_py, gold_int)])
    print(f"  model exact-correct: {correct_mask.sum()}/{N} = {correct_mask.mean()*100:.1f}%")
    print(f"  units agree with gold (when valid): "
          f"{((model_units == gold_units) & valid_mask).sum()}/{valid_mask.sum()}")
    print(f"  tens agree with gold (when valid): "
          f"{((model_tens == gold_tens) & valid_mask).sum()}/{valid_mask.sum()}")

    # baseline distributions
    cu = Counter(model_units[valid_mask].tolist())
    ct = Counter(model_tens[valid_mask].tolist())
    print(f"  model-units distribution: {dict(sorted(cu.items()))}")
    print(f"  model-tens  distribution: {dict(sorted(ct.items()))}")
    print(f"  model-units majority baseline: {max(cu.values())/sum(cu.values())*100:.1f}%")
    print(f"  model-tens  majority baseline: {max(ct.values())/sum(ct.values())*100:.1f}%")

    # fit per (pos, layer)
    units_acc = np.zeros((P, Lp1))
    tens_acc = np.zeros((P, Lp1))
    op_acc = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            units_acc[p, l] = fit_probe(X, model_units)
            tens_acc[p, l] = fit_probe(X, model_tens)
            op_acc[p, l] = fit_probe(X, operators)
        print(f"  pos {p:2d}: model-units max={units_acc[p].max()*100:.1f}%  "
              f"model-tens max={tens_acc[p].max()*100:.1f}%  "
              f"op max={op_acc[p].max()*100:.1f}%", flush=True)

    out = PD / "gpt2_multipos_probes_modelans.json"
    out.write_text(json.dumps({
        "model_units_acc": units_acc.tolist(),
        "model_tens_acc": tens_acc.tolist(),
        "operator_acc": op_acc.tolist(),
        "majority_units": max(cu.values())/sum(cu.values()),
        "majority_tens":  max(ct.values())/sum(ct.values()),
        "n_valid": int(valid_mask.sum()),
        "model_correct_rate": float(correct_mask.mean()),
        "P": P, "L_plus_1": Lp1,
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
