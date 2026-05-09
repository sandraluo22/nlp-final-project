"""Compare LDA operator-decodability on ALL examples vs CORRECT-ONLY examples.

For each (core_block, recurrence_step) we fit a 4-class LDA(n_components=3),
80/20 stratified split, and report the held-out test accuracy. Run twice:
  - all examples
  - only examples where Huginn's K=32 prediction equals the gold answer
Then compare the per-(block, step) and per-step accuracy curves.

Cf_balanced has 676 examples but only ~68 are correct (10.1% acc), which is
borderline-too-few for a 4-class LDA on 5280-dim features — so we use SVAMP
(1000 examples, 142 correct at K=32) instead.

Outputs:
  huginn-work/visualizations/probes/correct_only_compare.png
  huginn-work/visualizations/probes/correct_only_compare.json
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


HUGINN_ROOT = Path(__file__).resolve().parents[1]
ACTS = HUGINN_ROOT / "latent-sweep" / "huginn_svamp" / "K32" / "activations.pt"
RESULTS = HUGINN_ROOT / "latent-sweep" / "huginn_svamp" / "K32" / "results.json"
OUT_DIR = HUGINN_ROOT / "visualizations" / "probes"


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"loading activations: {ACTS}", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    print(f"  shape={a.shape}", flush=True)
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array(
        [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    )[: a.shape[0]]
    res = json.load(open(RESULTS))
    correct = np.array([bool(r["correct"]) for r in res])[: a.shape[0]]
    print(f"  N={a.shape[0]}  N_correct={int(correct.sum())} ({correct.mean()*100:.1f}%)",
          flush=True)
    return a, types, correct


def probe_grid(acts: np.ndarray, types: np.ndarray, seed: int = 0,
               n_components: int = 3) -> np.ndarray:
    """Return (S, L) test accuracy grid using stratified 80/20."""
    N, S, L, _H = acts.shape
    classes = np.unique(types)
    n_components = min(n_components, len(classes) - 1)
    grid = np.zeros((S, L), dtype=np.float32)
    idx = np.arange(N)
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, stratify=types, random_state=seed
    )
    y_tr, y_te = types[tr_idx], types[te_idx]
    for s in range(S):
        for l in range(L):
            X_tr = acts[tr_idx, s, l, :]
            X_te = acts[te_idx, s, l, :]
            proj = LDA(n_components=n_components, solver="svd")
            P_tr = proj.fit_transform(X_tr, y_tr)
            P_te = proj.transform(X_te)
            clf = LDA(solver="svd")
            clf.fit(P_tr, y_tr)
            grid[s, l] = clf.score(P_te, y_te)
        if s % 8 == 0 or s == S - 1:
            print(f"  step {s+1}/{S}: best_block={grid[s].max():.3f} "
                  f"mean={grid[s].mean():.3f}", flush=True)
    return grid


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    acts, types, correct = load()

    from collections import Counter
    print(f"\nfull set type distribution: {dict(Counter(types))}")
    print(f"correct subset type distribution: {dict(Counter(types[correct]))}")

    print("\n=== probing on ALL examples (N={}) ===".format(len(types)))
    grid_all = probe_grid(acts, types)
    print("\n=== probing on CORRECT-ONLY examples (N={}) ===".format(int(correct.sum())))
    if correct.sum() < 40:
        print(f"WARNING: only {int(correct.sum())} correct examples — LDA will be very noisy")
    grid_corr = probe_grid(acts[correct], types[correct])

    summary = {
        "n_all": int(len(types)),
        "n_correct": int(correct.sum()),
        "type_distribution_all": {k: int(v) for k, v in Counter(types).items()},
        "type_distribution_correct": {k: int(v) for k, v in Counter(types[correct]).items()},
        "grid_all": grid_all.tolist(),
        "grid_correct_only": grid_corr.tolist(),
        "best_per_step_all":  grid_all.max(axis=1).tolist(),
        "best_per_step_corr": grid_corr.max(axis=1).tolist(),
        "mean_per_step_all":  grid_all.mean(axis=1).tolist(),
        "mean_per_step_corr": grid_corr.mean(axis=1).tolist(),
    }
    (OUT_DIR / "correct_only_compare.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR/'correct_only_compare.json'}")

    # Plot
    Ks = np.arange(1, grid_all.shape[0] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, label_set, label_str in [
        (axes[0], "best", "best block per K"),
        (axes[1], "mean", "mean across blocks"),
    ]:
        if label_set == "best":
            y_all = grid_all.max(axis=1); y_corr = grid_corr.max(axis=1)
        else:
            y_all = grid_all.mean(axis=1); y_corr = grid_corr.mean(axis=1)
        ax.plot(Ks, y_all, "o-", color="#1f77b4",
                label=f"all examples (N={len(types)})")
        ax.plot(Ks, y_corr, "s-", color="#d62728",
                label=f"correct only (N={int(correct.sum())})")
        ax.axhline(0.25, ls=":", color="gray", lw=1, label="chance (4-class)")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32])
        ax.set_xlabel("recurrence step K")
        ax.set_ylabel("LDA(3-dim) accuracy")
        ax.set_title(label_str)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Operator-LDA probe: all examples vs correct-only on Huginn-3.5B SVAMP K=32",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "correct_only_compare.png"
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")

    print("\n=== KEY NUMBERS ===")
    print(f"  All examples best probe acc at K=8   : {grid_all[7].max():.3f}")
    print(f"  All examples best probe acc at K=16  : {grid_all[15].max():.3f}")
    print(f"  All examples best probe acc at K=32  : {grid_all[-1].max():.3f}")
    print(f"  Correct-only best probe acc at K=8   : {grid_corr[7].max():.3f}")
    print(f"  Correct-only best probe acc at K=16  : {grid_corr[15].max():.3f}")
    print(f"  Correct-only best probe acc at K=32  : {grid_corr[-1].max():.3f}")


if __name__ == "__main__":
    main()
