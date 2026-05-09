"""Layer-aligned cosine distance between teacher residual and student latent.

For every (student step, layer) cell, compute the per-question distance between
the student's latent at (step, layer) and the teacher's residual at the same
layer. Aggregate within each four-cell teacher x student outcome bucket plus
ALL, and emit per-cell heatmaps.

Mirrors the structure of inference/aggregate_logit_lens.py and
inference/probe_accuracy.py.

Outputs (under --out_dir):
  distances.pt                        {distances, metric, outcome_cells}
  aggregate.json                      {cell -> {n, mean_grid, sem_grid}}
  distance_{TC-SC,TC-SI,TI-SC,TI-SI,ALL}.png
  distance_combined.png               2x2 panel of the four outcome cells
  distance_diff_TCSI_minus_TCSC.png   diverging-colormap diff plot

Usage:
  python inference/distance_heatmap.py
  python inference/distance_heatmap.py --metric euclidean

Notes:
  - activations.pt is gitignored. Pull from the companion HF dataset
    (sandrajyluo/nlp-final-project-activations) or regenerate via
    inference/run_eval_with_hooks.py.
  - bf16 is cast to float32 before any cosine/norm computation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CELLS = ["TC-SC", "TC-SI", "TI-SC", "TI-SI", "ALL"]
PLOT_CELLS = ["TC-SC", "TC-SI", "TI-SC", "TI-SI"]
CELL_TITLES = {
    "TC-SC": "Teacher right, Student right",
    "TC-SI": "Teacher right, Student wrong",
    "TI-SC": "Teacher wrong, Student right",
    "TI-SI": "Teacher wrong, Student wrong",
    "ALL":   "All questions",
}


def require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"missing {path}; activations are gitignored. Pull from the "
            "companion HF dataset (sandrajyluo/nlp-final-project-activations) "
            "or regenerate via inference/run_eval_with_hooks.py."
        )


def outcome_cell(t_correct: bool, s_correct: bool) -> str:
    return f"{'TC' if t_correct else 'TI'}-{'SC' if s_correct else 'SI'}"


def load_outcomes(student_results_path: str, teacher_results_path: str, n: int) -> list:
    with open(student_results_path) as f:
        student = json.load(f)
    with open(teacher_results_path) as f:
        teacher = json.load(f)
    assert len(student) == len(teacher) == n, (
        f"length mismatch: student={len(student)} teacher={len(teacher)} acts={n}"
    )
    cells = []
    for sr, tr in zip(student, teacher):
        assert sr["idx"] == tr["idx"], f"idx mismatch at {sr['idx']} vs {tr['idx']}"
        cells.append(outcome_cell(bool(tr["correct"]), bool(sr["correct"])))
    return cells


def compute_distances(teacher: torch.Tensor, student: torch.Tensor, metric: str) -> torch.Tensor:
    """teacher: (N, L, H), student: (N, S, L, H). Returns (N, S, L) float32."""
    N, S, L, H = student.shape
    # Broadcast teacher over the S axis so each (q, s, l, :) student vector is
    # compared against the same (q, l, :) teacher vector at every latent step.
    teacher_b = teacher.float().unsqueeze(1).expand(-1, S, -1, -1)
    student_f = student.float()
    if metric == "cosine":
        sim = F.cosine_similarity(student_f, teacher_b, dim=-1)
        return 1.0 - sim
    if metric == "euclidean":
        return (student_f - teacher_b).norm(dim=-1)
    raise ValueError(f"unknown metric: {metric}")


def aggregate(distances: torch.Tensor, outcome_cells: list) -> dict:
    """{cell -> {'n', 'mean_grid' (S,L), 'sem_grid' (S,L)}}."""
    cells_arr = np.array(outcome_cells)
    dists_np = distances.numpy()
    S, L = distances.shape[1], distances.shape[2]
    out = {}
    for cell in CELLS:
        mask = np.ones(len(outcome_cells), dtype=bool) if cell == "ALL" else (cells_arr == cell)
        n = int(mask.sum())
        if n == 0:
            out[cell] = {"n": 0,
                         "mean_grid": np.zeros((S, L), dtype=np.float32),
                         "sem_grid": np.zeros((S, L), dtype=np.float32)}
            continue
        sel = dists_np[mask]
        mean_grid = sel.mean(axis=0).astype(np.float32)
        if n > 1:
            sem_grid = (sel.std(axis=0, ddof=1) / np.sqrt(n)).astype(np.float32)
        else:
            sem_grid = np.zeros_like(mean_grid)
        out[cell] = {"n": n, "mean_grid": mean_grid, "sem_grid": sem_grid}
    return out


def annotate(ax, grid, vmin, vmax, fmt="{:.3f}"):
    midpoint = (vmin + vmax) / 2.0
    S, L = grid.shape
    for s in range(S):
        for l in range(L):
            v = float(grid[s, l])
            color = "white" if v < midpoint else "black"
            ax.text(l, s, fmt.format(v), ha="center", va="center",
                    fontsize=7, color=color)


def plot_single(grid, n, cell, vmin, vmax, dataset_name, metric, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"{dataset_name}: mean {metric} distance, {cell} (n={n})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Latent step")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    annotate(ax, grid, vmin, vmax)
    fig.colorbar(im, ax=ax, label=f"{metric} distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[dist] saved {out_path}", flush=True)


def plot_combined(agg, vmin, vmax, dataset_name, metric, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    im = None
    for ax, cell in zip(axes.ravel(), PLOT_CELLS):
        info = agg[cell]
        grid = info["mean_grid"]
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{CELL_TITLES[cell]} (n={info['n']})", fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Latent step")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        annotate(ax, grid, vmin, vmax)
    fig.suptitle(
        f"{dataset_name}: {metric} distance between student latent "
        "and teacher residual at same layer",
        fontsize=12,
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=f"{metric} distance")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[dist] saved {out_path}", flush=True)


def plot_diff(agg, dataset_name, metric, out_path):
    import matplotlib.pyplot as plt
    diff = agg["TC-SI"]["mean_grid"] - agg["TC-SC"]["mean_grid"]
    bound = max(abs(float(diff.min())), abs(float(diff.max())), 1e-6)
    vmin, vmax = -bound, bound
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"{dataset_name}: distance(TC-SI) - distance(TC-SC) per (step, layer)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Latent step")
    ax.set_xticks(range(diff.shape[1]))
    ax.set_yticks(range(diff.shape[0]))
    for s in range(diff.shape[0]):
        for l in range(diff.shape[1]):
            v = float(diff[s, l])
            color = "white" if abs(v) > bound * 0.5 else "black"
            ax.text(l, s, f"{v:+.3f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label=f"{metric} distance(TC-SI) - distance(TC-SC)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[dist] saved {out_path}", flush=True)


def print_summary(agg, dataset_name, metric):
    print(f"\n=== Layer-aligned distance summary ({dataset_name}, metric={metric}) ===")
    print(f"{'Cell':<10}{'n':<8}{'mean dist':<14}"
          f"{'cell with max dist (s,l)':<30}{'cell with min dist (s,l)'}")
    for cell in CELLS:
        info = agg[cell]
        n = info["n"]
        if n == 0:
            print(f"{cell:<10}{n:<8}(empty)")
            continue
        grid = info["mean_grid"]
        mean = float(grid.mean())
        max_idx = np.unravel_index(int(np.argmax(grid)), grid.shape)
        min_idx = np.unravel_index(int(np.argmin(grid)), grid.shape)
        print(f"{cell:<10}{n:<8}{mean:<14.3f}"
              f"({max_idx[0]},{max_idx[1]}) = {float(grid[max_idx]):<18.3f}"
              f"({min_idx[0]},{min_idx[1]}) = {float(grid[min_idx]):.3f}")

    diff = agg["TC-SI"]["mean_grid"] - agg["TC-SC"]["mean_grid"]
    pos = np.unravel_index(int(np.argmax(diff)), diff.shape)
    neg = np.unravel_index(int(np.argmin(diff)), diff.shape)
    print(f"\nDifference TC-SI minus TC-SC:")
    print(f"  Largest positive difference: ({pos[0]},{pos[1]}) = {float(diff[pos]):+.3f} "
          f"(student diverges most from teacher here on TC-SI)")
    print(f"  Largest negative difference: ({neg[0]},{neg[1]}) = {float(diff[neg]):+.3f}")
    print(f"  Mean difference across all cells: {float(diff.mean()):+.3f}")


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
                   default="inference/analysis/svamp/distance_heatmap")
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--dataset_name", default="SVAMP")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for path in (args.teacher_acts, args.teacher_results,
                 args.student_acts, args.student_results):
        require_file(path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dist] loading teacher activations: {args.teacher_acts}", flush=True)
    teacher = torch.load(args.teacher_acts, map_location="cpu", weights_only=False)
    print(f"[dist]   shape={tuple(teacher.shape)}", flush=True)
    if teacher.dim() != 3:
        raise ValueError(f"expected teacher 3D activations, got {tuple(teacher.shape)}")

    print(f"[dist] loading student activations: {args.student_acts}", flush=True)
    student = torch.load(args.student_acts, map_location="cpu", weights_only=False)
    print(f"[dist]   shape={tuple(student.shape)}", flush=True)
    if student.dim() != 4:
        raise ValueError(f"expected student 4D activations, got {tuple(student.shape)}")

    N = teacher.shape[0]
    if (student.shape[0] != N
            or student.shape[2] != teacher.shape[1]
            or student.shape[3] != teacher.shape[2]):
        raise ValueError(
            f"shape mismatch: teacher {tuple(teacher.shape)} vs student "
            f"{tuple(student.shape)}; expected (N, L, H) and (N, S, L, H) "
            "with matching N, L, H"
        )

    outcome_cells = load_outcomes(args.student_results, args.teacher_results, N)

    print(f"[dist] computing {args.metric} distances...", flush=True)
    distances = compute_distances(teacher, student, args.metric)
    print(f"[dist]   distances shape={tuple(distances.shape)}", flush=True)

    if args.metric == "cosine":
        dmin, dmax = float(distances.min()), float(distances.max())
        assert -1e-4 <= dmin and dmax <= 2.0 + 1e-4, (
            f"cosine distances out of [0, 2]: min={dmin}, max={dmax}"
        )

    # Layer 0 is the embedding output; both models share the base, so the
    # prompt embedding should match and mean distance here should be ~0.
    layer0_mean = float(distances[:, :, 0].mean())
    if args.metric == "cosine" and layer0_mean > 0.05:
        print(f"[dist] WARNING: mean layer-0 cosine distance = {layer0_mean:.4f}; "
              "expected near zero (shared embedding). Check layer indexing.",
              flush=True)
    else:
        print(f"[dist]   layer-0 mean distance = {layer0_mean:.4f}", flush=True)

    agg = aggregate(distances, outcome_cells)

    n_sum = sum(agg[c]["n"] for c in PLOT_CELLS)
    counts_str = ", ".join(f"{c}={agg[c]['n']}" for c in CELLS)
    print(f"[dist]   per-cell counts: {counts_str}  "
          f"(sum of 4 cells = {n_sum}, expected {N})", flush=True)
    assert n_sum == N, f"cell counts {n_sum} do not sum to N={N}"

    torch.save(
        {
            "distances": distances,
            "metric": args.metric,
            "outcome_cells": outcome_cells,
        },
        out_dir / "distances.pt",
    )
    print(f"[dist] saved {out_dir / 'distances.pt'}", flush=True)

    agg_json = {
        cell: {
            "n": info["n"],
            "mean_grid": info["mean_grid"].tolist(),
            "sem_grid": info["sem_grid"].tolist(),
        }
        for cell, info in agg.items()
    }
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(agg_json, f, indent=2)
    print(f"[dist] saved {out_dir / 'aggregate.json'}", flush=True)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[dist] matplotlib missing; skipping PNGs.", flush=True)
        return

    grids = [agg[c]["mean_grid"] for c in CELLS]
    vmin = float(min(g.min() for g in grids))
    vmax = float(max(g.max() for g in grids))

    for cell in CELLS:
        plot_single(agg[cell]["mean_grid"], agg[cell]["n"], cell,
                    vmin, vmax, args.dataset_name, args.metric,
                    out_dir / f"distance_{cell}.png")
    plot_combined(agg, vmin, vmax, args.dataset_name, args.metric,
                  out_dir / "distance_combined.png")
    plot_diff(agg, args.dataset_name, args.metric,
              out_dir / "distance_diff_TCSI_minus_TCSC.png")

    print_summary(agg, args.dataset_name, args.metric)

    n_pngs = len(CELLS) + 2
    print(f"\nSaved:")
    print(f"  {out_dir}/distances.pt")
    print(f"  {out_dir}/aggregate.json")
    print(f"  {out_dir}/distance_*.png ({n_pngs} files)")


if __name__ == "__main__":
    main()
