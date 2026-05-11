"""Correct-only probes with consistent 70/30 train/test split:
  1. Refit canonical 4-class OPERATOR probe on cf_balanced CORRECT-ONLY (152
     examples), evaluate on held-out 30% CF + full SVAMP (correct only).
  2. Binary parity probe: gold answer odd vs even (real SVAMP correct subset).
  3. Binary 2 vs 4 probe.
  4. Binary 1 vs 3 probe.

All probes use LDA, 70/30 split, applied at every (layer, latent_step) cell.
Activations: latent-loop only (the existing inference/runs captures), shape
(N, 6 latent_steps, 13 layers, 768 hidden).
"""

from __future__ import annotations
import json
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
CF_RES = REPO / "inference" / "runs" / "cf_balanced_student_gpt2" / "results.json"
SV_ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
SV_RES = REPO / "inference" / "runs" / "svamp_student_gpt2" / "results.json"
CF_DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"

CLASSES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
CL2IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 13


def fit_eval_lda(X_tr, y_tr, X_te, y_te, n_components=None):
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None
    n_classes = len(np.unique(y_tr))
    if n_components is None:
        n_components = min(n_classes - 1, X_tr.shape[1])
    sc = StandardScaler().fit(X_tr)
    clf = LinearDiscriminantAnalysis(n_components=n_components).fit(sc.transform(X_tr), y_tr)
    return float(clf.score(sc.transform(X_te), y_te))


def main():
    print("loading cf_balanced + svamp_student activations...")
    cf_acts = torch.load(CF_ACTS, map_location="cpu").to(torch.float32).numpy()
    sv_acts = torch.load(SV_ACTS, map_location="cpu").to(torch.float32).numpy()
    print(f"  cf shape: {cf_acts.shape}, svamp shape: {sv_acts.shape}")
    # axes: (N, 6 latent_steps, 13 layers, 768)
    N_cf, S, Lp1, H = cf_acts.shape
    print(f"  axes: (N, latent_steps={S}, layers={Lp1}, hidden={H})")

    cf_results = json.load(open(CF_RES))
    sv_results = json.load(open(SV_RES))
    cf_correct = np.array([ex.get("correct", False) for ex in cf_results])
    sv_correct = np.array([ex.get("correct", False) for ex in sv_results])

    # CF labels (4-class operator)
    cf_data = json.load(open(CF_DATA))
    cf_op = np.array([CL2IDX.get(d["type"], -1) for d in cf_data])[:N_cf]

    # SVAMP labels (4-class operator + gold answer for parity/value tasks)
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    sv_op = np.array([CL2IDX.get(ex["Type"], -1) for ex in full])[:sv_acts.shape[0]]
    sv_gold = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])[:sv_acts.shape[0]]
    sv_gold_int = np.array([int(round(x)) for x in sv_gold])
    sv_parity = (sv_gold_int % 2).astype(int)   # 0 = even, 1 = odd

    summary = {}

    # =====================================================================
    # 1. OPERATOR probe — CORRECT-ONLY cf_balanced, 70/30 split
    # =====================================================================
    print("\n=== TASK 1: 4-class OPERATOR probe, CF correct-only, 70/30 ===")
    cf_keep = cf_correct & (cf_op >= 0)
    print(f"  CF correct examples: {cf_keep.sum()}/{N_cf}  per-op: {np.bincount(cf_op[cf_keep])}")
    cf_idx_keep = np.arange(N_cf)[cf_keep]
    tr_idx, te_idx = train_test_split(cf_idx_keep, test_size=0.3, random_state=SEED,
                                       stratify=cf_op[cf_keep])
    print(f"  train: {len(tr_idx)}, test: {len(te_idx)}")

    grid_cf = np.full((S, Lp1), np.nan)
    grid_sv = np.full((S, Lp1), np.nan)
    sv_keep = sv_correct & (sv_op >= 0)
    print(f"  SVAMP correct (for transfer): {sv_keep.sum()}/{len(sv_op)}")
    sv_idx_keep = np.arange(len(sv_op))[sv_keep]

    for s in range(S):
        for l in range(Lp1):
            X_tr = cf_acts[tr_idx, s, l]; y_tr = cf_op[tr_idx]
            X_te = cf_acts[te_idx, s, l]; y_te = cf_op[te_idx]
            grid_cf[s, l] = fit_eval_lda(X_tr, y_tr, X_te, y_te) or np.nan
            # SVAMP transfer (using same train data, eval on correct-only SVAMP)
            X_sv = sv_acts[sv_idx_keep, s, l]; y_sv = sv_op[sv_idx_keep]
            sc = StandardScaler().fit(X_tr)
            n_comp = min(len(np.unique(y_tr)) - 1, X_tr.shape[1])
            clf = LinearDiscriminantAnalysis(n_components=n_comp).fit(sc.transform(X_tr), y_tr)
            grid_sv[s, l] = float(clf.score(sc.transform(X_sv), y_sv))

    print("  CF held-out 30% accuracy (rows = latent step, cols = layer):")
    print(np.array2string(grid_cf*100, formatter={"float_kind": lambda x: f"{x:5.1f}"}, max_line_width=200))
    print("  SVAMP transfer (correct-only) accuracy:")
    print(np.array2string(grid_sv*100, formatter={"float_kind": lambda x: f"{x:5.1f}"}, max_line_width=200))
    bb = np.unravel_index(int(np.nanargmax(grid_cf)), grid_cf.shape)
    print(f"  best CF held-out cell: step {bb[0]} L{bb[1]} = {grid_cf[bb]*100:.1f}%  "
          f"(SVAMP transfer at same cell = {grid_sv[bb]*100:.1f}%)")
    summary["operator_correct_only"] = {
        "cf_grid": grid_cf.tolist(), "sv_transfer_grid": grid_sv.tolist(),
        "n_train": len(tr_idx), "n_test_cf": len(te_idx), "n_svamp": len(sv_idx_keep),
        "best_cell": [int(bb[0]), int(bb[1])],
        "best_cf_acc": float(grid_cf[bb]),
        "best_sv_transfer": float(grid_sv[bb]),
    }

    # =====================================================================
    # 2-4. BINARY probes on correct-only SVAMP, 70/30 split
    # =====================================================================
    def binary_probe(name, label_fn, filter_fn=None):
        """Binary probe: label_fn maps gold -> {0, 1, -1}; filter_fn optional."""
        print(f"\n=== TASK: BINARY probe '{name}' on correct-only SVAMP ===")
        labels = np.array([label_fn(g) for g in sv_gold_int])
        # filter
        keep = sv_correct & (labels >= 0)
        if filter_fn is not None:
            keep = keep & filter_fn(sv_gold_int)
        idx = np.arange(len(sv_op))[keep]
        labs = labels[idx]
        if len(np.unique(labs)) < 2:
            print("  not enough classes — skipping")
            return None
        n_per_class = np.bincount(labs)
        print(f"  examples: {len(idx)}  per-class: {n_per_class}")
        if n_per_class.min() < 6:
            print("  warning: very few examples per class")
            return None
        tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=labs)
        grid = np.full((S, Lp1), np.nan)
        for s in range(S):
            for l in range(Lp1):
                X_tr = sv_acts[tr_idx, s, l]; y_tr = labels[tr_idx]
                X_te = sv_acts[te_idx, s, l]; y_te = labels[te_idx]
                grid[s, l] = fit_eval_lda(X_tr, y_tr, X_te, y_te) or np.nan
        bb = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
        majority_baseline = max(n_per_class) / sum(n_per_class)
        print(f"  majority baseline: {majority_baseline*100:.1f}%")
        print(f"  held-out 30% accuracy grid (rows = latent step, cols = layer):")
        print(np.array2string(grid*100, formatter={"float_kind": lambda x: f"{x:5.1f}"}, max_line_width=200))
        print(f"  best cell: step {bb[0]} L{bb[1]} = {grid[bb]*100:.1f}%")
        return {"grid": grid.tolist(), "best_cell": [int(bb[0]), int(bb[1])],
                "best_acc": float(grid[bb]), "majority_baseline": float(majority_baseline),
                "n_per_class": n_per_class.tolist()}

    summary["parity"]    = binary_probe("parity (odd vs even)", lambda g: g % 2)
    summary["2_vs_4"]    = binary_probe("2 vs 4",  lambda g: 0 if g == 2 else (1 if g == 4 else -1))
    summary["1_vs_3"]    = binary_probe("1 vs 3",  lambda g: 0 if g == 1 else (1 if g == 3 else -1))
    summary["odd_vs_3"]  = binary_probe("3 vs 5",  lambda g: 0 if g == 3 else (1 if g == 5 else -1))

    out = PD / "correct_only_probes.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
