"""Pairwise adjacent-step logit-lens analysis across ALL latent steps.

For each question, compute the gold-answer rank at every latent step (at a
chosen decoder layer), classify as "right" (rank < k) or "wrong" (rank >= k),
then build a 2x2 transition matrix for every adjacent pair of steps.

Outputs (under --out_dir):
  step_overview.json        per-step counts, pairwise transitions, rank trajectories
  step_overview.png         per-step right/wrong bar chart (same as before)
  transitions.png           grouped bar chart of gained/lost per adjacent pair
  transitions/stepA_stepB.json  per-transition question lists (gained/lost/stable)

Usage:
  python latent_filter_step_overview.py \
      --activations runs/svamp_student/activations.pt \
      --results runs/svamp_student/results.json \
      --out_dir analysis/svamp_step_overview \
      --topk 5 --layer -1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_INF = Path(__file__).resolve().parent
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

from aggregate_logit_lens import gold_to_token_ids, load_lm_head  # noqa: E402
from question_features import agg_group, print_breakdown_table, question_features  # noqa: E402


def _resolve_layer(layer: int, num_layers_plus_one: int) -> int:
    L = num_layers_plus_one
    if layer < 0:
        layer = L + layer
    if not (0 <= layer < L):
        raise ValueError(f"layer must be in [0, {L - 1}] or negative, got {layer}")
    return layer


def _batch_ranks(
    hidden: torch.Tensor,
    lm_head: torch.Tensor,
    gold_ids_per_q: list[list[int]],
) -> list[int]:
    """hidden: (N, H) -> rank of gold token per row; -1 if no gold ids."""
    N, H = hidden.shape
    logits = hidden.float() @ lm_head.T
    out: list[int] = []
    for q in range(N):
        ids = gold_ids_per_q[q]
        if not ids:
            out.append(-1)
            continue
        lv = logits[q]
        best: int | None = None
        for tid in ids:
            g = lv[tid].item()
            r = int((lv > g).sum().item())
            if best is None or r < best:
                best = r
        out.append(best if best is not None else -1)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--activations", required=True, help="Student activations.pt (N,S,L,H)")
    p.add_argument("--results", required=True, help="Matching results.json")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--topk", type=int, default=5, help="'right' means gold rank < topk (default: 5)")
    p.add_argument("--layer", type=int, default=-1, help="Decoder layer index (default: -1 = final layer)")
    p.add_argument(
        "--dataset_name",
        default=None,
        help="Label for plots (auto-detected from path if omitted)",
    )
    args = p.parse_args()

    acts_path = Path(args.activations).expanduser()
    results_path = Path(args.results).expanduser()
    if not acts_path.is_file():
        raise SystemExit(f"Activations file not found: {acts_path}")
    if not results_path.is_file():
        raise SystemExit(f"Results file not found: {results_path}")

    acts = torch.load(str(acts_path), map_location="cpu", weights_only=False)
    if acts.dim() != 4:
        raise ValueError(f"expected (N,S,L,H), got {tuple(acts.shape)}")
    N, S, L, _ = acts.shape
    layer = _resolve_layer(args.layer, L)

    with open(results_path) as f:
        results = json.load(f)
    if len(results) != N:
        raise ValueError(f"results ({len(results)}) != activations N ({N})")

    dataset_name = args.dataset_name or acts_path.parent.name
    k = args.topk

    features: list[dict] = [question_features(r["question"], r["gold"]) for r in results]

    lm_head, tokenizer = load_lm_head(args.base_model, trust_remote_code=args.trust_remote_code)

    gold_ids_per_q: list[list[int]] = []
    skipped = 0
    for r in results:
        ids = gold_to_token_ids(r["gold"], tokenizer)
        gold_ids_per_q.append(ids)
        if not ids:
            skipped += 1

    # ---- Compute ranks at every step ----
    ranks_all: list[list[int]] = []
    for si in range(S):
        print(
            f"[overview] computing ranks for step {si + 1}/{S} at layer {layer}",
            flush=True,
        )
        h = acts[:, si, layer, :]
        ranks_all.append(_batch_ranks(h, lm_head, gold_ids_per_q))

    # right_at[si] = set of question indices that are "right" at step si
    right_at: list[set[int]] = []
    for si in range(S):
        right_at.append({q for q in range(N) if ranks_all[si][q] >= 0 and ranks_all[si][q] < k})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-step counts (kept for the static bar chart) ----
    step_summaries = []
    for si in range(S):
        n_right = len(right_at[si])
        n_skip = sum(1 for q in range(N) if ranks_all[si][q] < 0)
        step_summaries.append(
            {
                "step": si + 1,
                "right": n_right,
                "wrong": N - n_right - n_skip,
                "skipped": n_skip,
            }
        )

    # ---- Pairwise adjacent transitions ----
    def _q_entry(q: int, step_a: int, step_b: int) -> dict:
        res = results[q]
        return {
            "idx": res["idx"],
            "question": res["question"],
            "gold": res["gold"],
            "pred": res["pred"],
            "student_final_correct": res["correct"],
            f"rank_step{step_a}": ranks_all[step_a - 1][q],
            f"rank_step{step_b}": ranks_all[step_b - 1][q],
            "features": features[q],
        }

    transitions_dir = out_dir / "transitions"
    transitions_dir.mkdir(exist_ok=True)

    transition_summaries = []
    for si in range(S - 1):
        step_a, step_b = si + 1, si + 2
        ra, rb = right_at[si], right_at[si + 1]
        valid = {q for q in range(N) if ranks_all[si][q] >= 0}

        gained = sorted(valid & rb - ra)  # wrong->right
        lost = sorted(valid & ra - rb)  # right->wrong
        stable_right = sorted(valid & ra & rb)
        stable_wrong = sorted(valid - ra - rb)

        breakdown = {
            "gained": agg_group([features[q] for q in gained]),
            "lost": agg_group([features[q] for q in lost]),
            "stable_right": agg_group([features[q] for q in stable_right]),
            "stable_wrong": agg_group([features[q] for q in stable_wrong]),
        }

        t = {
            "from_step": step_a,
            "to_step": step_b,
            "gained_wrong_to_right": len(gained),
            "lost_right_to_wrong": len(lost),
            "stable_right": len(stable_right),
            "stable_wrong": len(stable_wrong),
            "feature_breakdown": breakdown,
        }
        transition_summaries.append(t)

        t_detail = {
            **t,
            "topk": k,
            "layer": layer,
            "gained_questions": [_q_entry(q, step_a, step_b) for q in gained],
            "lost_questions": [_q_entry(q, step_a, step_b) for q in lost],
        }
        t_file = transitions_dir / f"step{step_a}_step{step_b}.json"
        with open(t_file, "w") as f:
            json.dump(t_detail, f, indent=2, ensure_ascii=False)

    # ---- Per-question rank trajectory ----
    trajectories = []
    for q in range(N):
        res = results[q]
        trajectories.append(
            {
                "idx": res["idx"],
                "gold": res["gold"],
                "pred": res["pred"],
                "student_final_correct": res["correct"],
                "ranks": [ranks_all[si][q] for si in range(S)],
            }
        )

    overview = {
        "dataset": dataset_name,
        "N": N,
        "S": S,
        "layer": layer,
        "topk": k,
        "skipped_gold_not_single_token": skipped,
        "per_step": step_summaries,
        "transitions": transition_summaries,
    }

    overview_path = out_dir / "step_overview.json"
    with open(overview_path, "w") as f:
        json.dump({"summary": overview, "trajectories": trajectories}, f, indent=2, ensure_ascii=False)
    print(f"\n[overview] wrote {overview_path}", flush=True)

    # ---- Print tables ----
    print(f"\n{'Step':<6}{'Right':<8}{'Wrong':<8}{'Skipped':<8}")
    for row in step_summaries:
        print(f"{row['step']:<6}{row['right']:<8}{row['wrong']:<8}{row['skipped']:<8}")

    print(f"\n{'Transition':<14}{'Gained':<10}{'Lost':<10}{'Stable R':<10}{'Stable W':<10}")
    for t in transition_summaries:
        label = f"{t['from_step']}->{t['to_step']}"
        print(
            f"{label:<14}{t['gained_wrong_to_right']:<10}{t['lost_right_to_wrong']:<10}"
            f"{t['stable_right']:<10}{t['stable_wrong']:<10}"
        )

    print_breakdown_table(
        transition_summaries,
        lambda t: f"{t['from_step']}->{t['to_step']}",
    )

    # ---- Plots ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[overview] matplotlib not installed; skipping plots.")
        return

    # Plot 1: per-step bar chart
    steps = [row["step"] for row in step_summaries]
    rights = [row["right"] for row in step_summaries]
    wrongs = [row["wrong"] for row in step_summaries]
    skippeds = [row["skipped"] for row in step_summaries]

    fig, ax = plt.subplots(figsize=(max(6, S * 1.2), 5))
    bar_w = 0.55
    x = range(len(steps))
    ax.bar(x, rights, bar_w, label=f"Right (rank < {k})", color="#4CAF50")
    ax.bar(x, wrongs, bar_w, bottom=rights, label=f"Wrong (rank ≥ {k})", color="#F44336")
    ax.bar(
        x,
        skippeds,
        bar_w,
        bottom=[r + w for r, w in zip(rights, wrongs)],
        label="Skipped (no single token)",
        color="#BDBDBD",
    )
    for i, (r, w) in enumerate(zip(rights, wrongs)):
        if r > 0:
            ax.text(i, r / 2, str(r), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        if w > 0:
            ax.text(i, r + w / 2, str(w), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: gold-answer rank per step (layer {layer}, top-{k})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "step_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[overview] saved step_overview.png", flush=True)

    # Plot 2: pairwise transitions
    labels = [f"{t['from_step']}→{t['to_step']}" for t in transition_summaries]
    gained = [t["gained_wrong_to_right"] for t in transition_summaries]
    lost = [t["lost_right_to_wrong"] for t in transition_summaries]

    fig, ax = plt.subplots(figsize=(max(6, (S - 1) * 1.5), 5))
    x = range(len(labels))
    w = 0.35
    bars_g = ax.bar([i - w / 2 for i in x], gained, w, label="Gained (wrong→right)", color="#4CAF50")
    bars_l = ax.bar([i + w / 2 for i in x], lost, w, label="Lost (right→wrong)", color="#F44336")
    for bar in bars_g:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                str(int(h)),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    for bar in bars_l:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                str(int(h)),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Adjacent step transition")
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: questions gained/lost between adjacent steps (layer {layer}, top-{k})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "transitions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[overview] saved transitions.png", flush=True)
    print(f"[overview] per-transition question lists in {transitions_dir}/", flush=True)


if __name__ == "__main__":
    main()

