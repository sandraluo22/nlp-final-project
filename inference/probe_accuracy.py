"""Linear probe for per-question correctness on saved CODI activations.

For every (step, layer) cell of saved teacher / student activations, train a
linear probe to predict whether the model's final answer was correct. Reports
5-fold stratified CV accuracy alongside two controls (label permutation,
random features) at every cell, plus heatmaps.

Mirrors the structure of inference/aggregate_logit_lens.py.

Usage:
  python inference/probe_accuracy.py \\
      --teacher_acts    inference/runs/svamp_teacher/activations.pt \\
      --teacher_results inference/runs/svamp_teacher/results.json \\
      --student_acts    inference/runs/svamp_student/activations.pt \\
      --student_results inference/runs/svamp_student/results.json \\
      --out_dir         inference/analysis/svamp/probe_accuracy

Notes:
  - activations.pt is gitignored (~800MB). Pull from the companion HF dataset
    (sandrajyluo/nlp-final-project-activations) or regenerate via
    inference/run_eval_with_hooks.py.
  - LogReg uses L2 with StandardScaler (regularization is scale-sensitive).
  - LDA uses solver='lsqr' with shrinkage='auto' because H=2048 > N_train.
  - Controls: shuffled labels (permutation invariance) and random Gaussian
    features (feature-information ablation). Both should sit at the majority
    baseline; if the real probe doesn't beat both, the cell carries no
    linearly-readable correctness signal.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Probe builders
# ---------------------------------------------------------------------------

def make_logreg():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs"),
    )


def make_lda():
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


# ---------------------------------------------------------------------------
# Per-cell probing
# ---------------------------------------------------------------------------

def probe_cell(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
               seed: int = 0) -> dict:
    """Return mean held-out accuracy for: real LogReg, real LDA, shuffled-label
    LogReg, random-feature LogReg. Plus N and n_pos."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)
    X_random = rng.standard_normal(size=X.shape).astype(np.float32)

    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "acc_logreg": float(cross_val_score(make_logreg(), X, y, cv=cv).mean()),
        "acc_lda": float(cross_val_score(make_lda(), X, y, cv=cv).mean()),
        "acc_shuffled": float(cross_val_score(make_logreg(), X, y_shuffled, cv=cv).mean()),
        "acc_random_feat": float(cross_val_score(make_logreg(), X_random, y, cv=cv).mean()),
    }


def majority_baseline(y: np.ndarray) -> float:
    return max(float((y == 1).mean()), float((y == 0).mean()))


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_teacher(acts: torch.Tensor, y: np.ndarray, seed: int = 0) -> list:
    """acts: (N, L, H). Returns one row per layer."""
    L = acts.shape[1]
    rows = []
    for layer in range(L):
        X = acts[:, layer, :].float().numpy()
        scores = probe_cell(X, y, seed=seed)
        scores["layer"] = layer
        rows.append(scores)
        print(
            f"[probe] teacher layer={layer:>2} "
            f"acc_logreg={scores['acc_logreg']:.3f} "
            f"acc_lda={scores['acc_lda']:.3f} "
            f"shuf={scores['acc_shuffled']:.3f} "
            f"rand={scores['acc_random_feat']:.3f}",
            flush=True,
        )
    return rows


def sweep_student(acts: torch.Tensor, y: np.ndarray, seed: int = 0) -> list:
    """acts: (N, S, L, H). Returns one row per (step, layer)."""
    _, S, L, _ = acts.shape
    rows = []
    for step in range(S):
        for layer in range(L):
            X = acts[:, step, layer, :].float().numpy()
            scores = probe_cell(X, y, seed=seed)
            scores["step"] = step
            scores["layer"] = layer
            rows.append(scores)
            print(
                f"[probe] student step={step} layer={layer:>2} "
                f"acc_logreg={scores['acc_logreg']:.3f} "
                f"acc_lda={scores['acc_lda']:.3f} "
                f"shuf={scores['acc_shuffled']:.3f} "
                f"rand={scores['acc_random_feat']:.3f}",
                flush=True,
            )
    return rows


# ---------------------------------------------------------------------------
# Best-cell out-of-fold predictions (for downstream cross-tabbing)
# ---------------------------------------------------------------------------

def oof_predictions(X: np.ndarray, y: np.ndarray, seed: int = 0):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.full(len(y), -1, dtype=np.int64)
    probs = np.full(len(y), -1.0, dtype=np.float32)
    for tr, te in cv.split(X, y):
        clf = make_logreg().fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
        probs[te] = clf.predict_proba(X[te])[:, 1]
    return preds, probs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_teacher(rows: list, baseline: float, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe] matplotlib missing; skipping PNG", flush=True)
        return
    layers = [r["layer"] for r in rows]
    real = [r["acc_logreg"] for r in rows]
    lda = [r["acc_lda"] for r in rows]
    shuf = [r["acc_shuffled"] for r in rows]
    rand = [r["acc_random_feat"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, real, marker="o", label="LogReg")
    ax.plot(layers, lda, marker="s", label="LDA")
    ax.plot(layers, shuf, marker="x", linestyle="--", label="shuffled labels")
    ax.plot(layers, rand, marker="^", linestyle=":", label="random features")
    ax.axhline(baseline, color="gray", linestyle="-",
               label=f"majority={baseline:.3f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV accuracy")
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[probe] saved {out_path}", flush=True)


def plot_student_grid(rows: list, S: int, L: int, baseline: float,
                      out_path: Path, title: str, key: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe] matplotlib missing; skipping PNG", flush=True)
        return
    grid = np.zeros((S, L), dtype=np.float32)
    for r in rows:
        grid[r["step"], r["layer"]] = r[key]

    fig, ax = plt.subplots(figsize=(14, 5))
    vmin = min(baseline, float(grid.min()))
    vmax = max(baseline + 0.01, float(grid.max()))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Latent step")
    ax.set_xticks(range(L))
    ax.set_yticks(range(S))
    ax.set_title(f"{title} (majority={baseline:.3f})")
    midpoint = (vmin + vmax) / 2
    for s in range(S):
        for l in range(L):
            v = grid[s, l]
            color = "white" if v < midpoint else "black"
            ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
    fig.colorbar(im, ax=ax, label="CV accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[probe] saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_labels(results_path: str) -> np.ndarray:
    with open(results_path) as f:
        results = json.load(f)
    return np.array([int(r["correct"]) for r in results], dtype=np.int64)


def require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"missing {path}; activations are gitignored. Pull from the "
            "companion HF dataset (sandrajyluo/nlp-final-project-activations) "
            "or regenerate via inference/run_eval_with_hooks.py."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_teacher(args, out_dir: Path) -> None:
    require_file(args.teacher_acts)
    print(f"[probe] loading teacher activations: {args.teacher_acts}", flush=True)
    acts = torch.load(args.teacher_acts, map_location="cpu", weights_only=False)
    y = load_labels(args.teacher_results)
    assert acts.shape[0] == len(y), (
        f"N mismatch: acts={acts.shape[0]} results={len(y)}"
    )
    if acts.dim() != 3:
        raise ValueError(f"expected teacher 3D activations, got {tuple(acts.shape)}")
    print(f"[probe]   shape={tuple(acts.shape)} N={len(y)} pos={int(y.sum())}",
          flush=True)

    baseline = majority_baseline(y)
    rows = sweep_teacher(acts, y, seed=args.seed)
    with open(out_dir / "teacher.json", "w") as f:
        json.dump({"baseline": baseline, "rows": rows}, f, indent=2)
    plot_teacher(rows, baseline, out_dir / "teacher_acc.png",
                 "SVAMP teacher: P(correct) probe by layer")

    if args.save_predictions:
        best = max(rows, key=lambda r: r["acc_logreg"])
        X = acts[:, best["layer"], :].float().numpy()
        preds, probs = oof_predictions(X, y, seed=args.seed)
        torch.save(
            {
                "layer": best["layer"],
                "preds": torch.from_numpy(preds),
                "probs": torch.from_numpy(probs),
                "labels": torch.from_numpy(y),
            },
            out_dir / "teacher_predictions.pt",
        )
        print(f"[probe] saved teacher predictions at layer={best['layer']} "
              f"(acc_logreg={best['acc_logreg']:.3f})", flush=True)


def run_student(args, out_dir: Path) -> None:
    require_file(args.student_acts)
    print(f"[probe] loading student activations: {args.student_acts}", flush=True)
    acts = torch.load(args.student_acts, map_location="cpu", weights_only=False)
    y = load_labels(args.student_results)
    assert acts.shape[0] == len(y), (
        f"N mismatch: acts={acts.shape[0]} results={len(y)}"
    )
    if acts.dim() != 4:
        raise ValueError(f"expected student 4D activations, got {tuple(acts.shape)}")
    print(f"[probe]   shape={tuple(acts.shape)} N={len(y)} pos={int(y.sum())}",
          flush=True)

    _, S, L, _ = acts.shape
    baseline = majority_baseline(y)
    rows = sweep_student(acts, y, seed=args.seed)
    with open(out_dir / "student.json", "w") as f:
        json.dump({"baseline": baseline, "shape": [S, L], "rows": rows}, f, indent=2)

    plot_student_grid(rows, S, L, baseline,
                      out_dir / "student_acc_logreg.png",
                      "SVAMP student: P(correct) probe (LogReg) by (step, layer)",
                      key="acc_logreg")
    plot_student_grid(rows, S, L, baseline,
                      out_dir / "student_acc_lda.png",
                      "SVAMP student: P(correct) probe (LDA) by (step, layer)",
                      key="acc_lda")
    plot_student_grid(rows, S, L, baseline,
                      out_dir / "student_acc_shuffled.png",
                      "SVAMP student: shuffled-label control by (step, layer)",
                      key="acc_shuffled")

    if args.save_predictions:
        best = max(rows, key=lambda r: r["acc_logreg"])
        X = acts[:, best["step"], best["layer"], :].float().numpy()
        preds, probs = oof_predictions(X, y, seed=args.seed)
        torch.save(
            {
                "step": best["step"],
                "layer": best["layer"],
                "preds": torch.from_numpy(preds),
                "probs": torch.from_numpy(probs),
                "labels": torch.from_numpy(y),
            },
            out_dir / "student_predictions.pt",
        )
        print(f"[probe] saved student predictions at "
              f"step={best['step']} layer={best['layer']} "
              f"(acc_logreg={best['acc_logreg']:.3f})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_acts",
                   default="inference/runs/svamp_teacher/activations.pt")
    p.add_argument("--teacher_results",
                   default="inference/runs/svamp_teacher/results.json")
    p.add_argument("--student_acts",
                   default="inference/runs/svamp_student/activations.pt")
    p.add_argument("--student_results",
                   default="inference/runs/svamp_student/results.json")
    p.add_argument("--out_dir",
                   default="inference/analysis/svamp/probe_accuracy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_teacher", action="store_true")
    p.add_argument("--skip_student", action="store_true")
    p.add_argument("--save_predictions", action="store_true",
                   help="dump (N,) preds/probs at the best (step, layer) cell")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_teacher:
        run_teacher(args, out_dir)
    if not args.skip_student:
        run_student(args, out_dir)

    print(f"[probe] done. outputs in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
