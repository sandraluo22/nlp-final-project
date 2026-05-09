"""Probe Huginn's recurrent-depth activations for operator information.

The accuracy phase transition between K=8 and K=16 is a behavioral fact: when
forced to stop after K=8 recurrence iterations on SVAMP, Huginn fails (2.6%);
at K=16 it suddenly works (13.1%). The interpretability question is whether
something inside the residual stream undergoes a *matching* internal phase
transition — does the operator type become linearly decodable at the same K?

Setup:
  acts shape: (N=1000, S=32, L=4, H=5280)
  labels   : SVAMP problem type (Subtraction / Addition / Common-Division /
             Multiplication) — 4 classes, ~250 each
  probe    : LDA(n_components=3), fit on 80% (stratified) of acts[:, k, l, :]
             and tested on the held-out 20%, for every (k, l) ∈ [0..31]×[0..3].
             Use a fresh LDA classifier on the 3-dim projection (matches the
             "k-dim only" accuracy from the CODI cf_lda_80_20 script).

Outputs:
  huginn/probes/operator_lda_depth.png   — line plot of per-K probe accuracy
  huginn/probes/operator_lda_depth.json  — full (32×4) accuracy grid + summary
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parent.parent
ACTS = REPO / "latent-sweep" / "huginn_svamp" / "K32" / "activations.pt"
OUT_DIR = REPO / "visualizations" / "probes"


def load_metadata(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (operator_label, magnitude_bucket) of length n."""
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array(
        [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    )[:n]
    answers = np.array(
        [float(str(a).replace(",", "")) for a in full["Answer"]], dtype=np.float64
    )[:n]
    mag = np.empty(n, dtype=object)
    mag[answers < 10] = "<10"
    mag[(answers >= 10) & (answers < 100)] = "10-99"
    mag[(answers >= 100) & (answers < 1000)] = "100-999"
    mag[answers >= 1000] = "1000+"
    return types, mag


def load_acts() -> np.ndarray:
    print(f"loading {ACTS}", flush=True)
    t = torch.load(ACTS, map_location="cpu", weights_only=True)
    a = t.float().numpy()
    print(f"  shape={a.shape}  bytes={a.nbytes/1e9:.2f} GB", flush=True)
    return a


def probe_grid(acts: np.ndarray, labels: np.ndarray, n_components: int = 3,
               seed: int = 0) -> np.ndarray:
    """Fit an LDA(n_components) per (step, block); return (S, L) test acc grid.
    The classifier is a *second* LDA fit on the projected training points (so
    the accuracy is the accuracy of a classifier that only sees the n_components
    visualized dimensions). Mirrors visualizations-all/cf_lda_80_20."""
    N, S, L, H = acts.shape
    classes = np.unique(labels)
    n_components = min(n_components, len(classes) - 1)
    grid = np.zeros((S, L), dtype=np.float32)
    idx = np.arange(N)
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=seed
    )
    y_tr, y_te = labels[tr_idx], labels[te_idx]
    for s in range(S):
        for l in range(L):
            X = acts[:, s, l, :]
            X_tr, X_te = X[tr_idx], X[te_idx]
            proj = LDA(n_components=n_components, solver="svd")
            P_tr = proj.fit_transform(X_tr, y_tr)
            P_te = proj.transform(X_te)
            clf = LDA(solver="svd")
            clf.fit(P_tr, y_tr)
            grid[s, l] = clf.score(P_te, y_te)
        print(f"  step {s+1}/{S}: best_block_acc={grid[s].max():.3f}  mean={grid[s].mean():.3f}", flush=True)
    return grid


SVAMP_HUGINN_ACC = {1: 0.004, 2: 0.011, 4: 0.026, 8: 0.026, 16: 0.131, 32: 0.142}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    acts = load_acts()
    N = acts.shape[0]
    op_labels, mag_labels = load_metadata(N)
    from collections import Counter
    print(f"operator counts: {dict(Counter(op_labels))}")
    print(f"magnitude counts: {dict(Counter(mag_labels))}")

    print("\n=== probing OPERATOR (4-class) ===")
    op_grid = probe_grid(acts, op_labels, n_components=3)
    print("\n=== probing MAGNITUDE bucket (4-class) ===")
    mag_grid = probe_grid(acts, mag_labels, n_components=3)

    op_mean_per_step  = op_grid.mean(axis=1)
    op_best_per_step  = op_grid.max(axis=1)
    mag_mean_per_step = mag_grid.mean(axis=1)
    mag_best_per_step = mag_grid.max(axis=1)

    # ---------------- save full results ----------------
    summary = {
        "shape": list(acts.shape),
        "operator_grid":  op_grid.tolist(),
        "magnitude_grid": mag_grid.tolist(),
        "operator_mean_per_step":  op_mean_per_step.tolist(),
        "operator_best_per_step":  op_best_per_step.tolist(),
        "magnitude_mean_per_step": mag_mean_per_step.tolist(),
        "magnitude_best_per_step": mag_best_per_step.tolist(),
        "huginn_accuracy_at_K":    SVAMP_HUGINN_ACC,
    }
    (OUT_DIR / "operator_lda_depth.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR / 'operator_lda_depth.json'}")

    # ---------------- plot ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    Ks = np.arange(1, op_grid.shape[0] + 1)
    ax = axes[0]
    ax.plot(Ks, op_best_per_step,  "o-", label="operator (best block)", color="#d62728")
    ax.plot(Ks, op_mean_per_step,  "o--", label="operator (mean)",      color="#d62728", alpha=0.5)
    ax.plot(Ks, mag_best_per_step, "s-", label="magnitude (best block)", color="#1f77b4")
    ax.plot(Ks, mag_mean_per_step, "s--", label="magnitude (mean)",      color="#1f77b4", alpha=0.5)
    ax.axhline(0.25, ls=":", color="gray", lw=1, label="chance (4-class)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32])
    ax.set_xlabel("recurrence step K (log scale)")
    ax.set_ylabel("LDA(3-dim) classification accuracy")
    ax.set_title("Internal: 4-class probe accuracy vs recurrence depth")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Behavioral curve for comparison
    ax = axes[1]
    K_acc = sorted(SVAMP_HUGINN_ACC)
    accs  = [SVAMP_HUGINN_ACC[k] for k in K_acc]
    ax.plot(K_acc, accs, "o-", color="black", lw=2, label="Huginn SVAMP accuracy")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32])
    ax.set_xlabel("recurrence step K (log scale)")
    ax.set_ylabel("SVAMP accuracy")
    ax.set_title("Behavioral: SVAMP accuracy vs recurrence depth")
    ax.set_ylim(0, 0.20)
    ax.axvspan(8, 16, color="red", alpha=0.1, label="phase transition")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Huginn-3.5B on SVAMP — operator/magnitude become decodable BEFORE accuracy "
        f"phase transition?\n(activations: {acts.shape})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "operator_lda_depth.png"
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")

    # ---------------- terse human-readable summary ----------------
    print("\n=== KEY NUMBERS ===")
    print(f"  Operator best probe acc at K=1   : {op_best_per_step[0]:.3f}")
    print(f"  Operator best probe acc at K=8   : {op_best_per_step[7]:.3f}")
    print(f"  Operator best probe acc at K=16  : {op_best_per_step[15]:.3f}")
    print(f"  Operator best probe acc at K=32  : {op_best_per_step[-1]:.3f}")
    print(f"  Magnitude best probe acc at K=1  : {mag_best_per_step[0]:.3f}")
    print(f"  Magnitude best probe acc at K=8  : {mag_best_per_step[7]:.3f}")
    print(f"  Magnitude best probe acc at K=16 : {mag_best_per_step[15]:.3f}")
    print(f"  Magnitude best probe acc at K=32 : {mag_best_per_step[-1]:.3f}")
    # Find the K where operator-probe accuracy first crosses 0.5 and 0.8
    for thresh in (0.50, 0.80, 0.90):
        idxs = np.where(op_best_per_step >= thresh)[0]
        if len(idxs):
            print(f"  Operator probe acc ≥ {thresh:.2f} first at K={idxs[0]+1}")
        else:
            print(f"  Operator probe acc ≥ {thresh:.2f} never reached")


if __name__ == "__main__":
    main()
