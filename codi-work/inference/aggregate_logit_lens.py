"""Aggregate logit-lens analysis for student CODI activations.

For each question, find the rank of the gold-answer token in the LM-head
distribution at every (step, layer) cell. Aggregate into hit rates
P(rank < k) per cell, broken down by the four-cell teacher x student
outcome variable.

Outputs (under --out_dir):
  ranks.pt              (N, S, L) long tensor; -1 = skipped
                        (S, L) = (6, 17) for CODI student, (num_steps, L_core)
                        for Huginn-0125, etc.
  outcomes.json         per-question outcome cell + gold-token info
  aggregate.json        for each k and outcome cell, the (S, L) hit-rate grid
  hit_rate_topk{K}.png  2x2 heatmap (one panel per outcome cell), per k

Usage:
  python aggregate_logit_lens.py \\
      --student_acts    runs/svamp_student/activations.pt \\
      --student_results runs/svamp_student/results.json \\
      --teacher_results runs/svamp_teacher/results.json \\
      --out_dir         analysis/svamp \\
      --topk 5 10 50

Requires: torch, transformers, matplotlib (optional; PNGs skipped if absent).
"""

import argparse
import json
from pathlib import Path

import torch


def load_lm_head(base_model: str, trust_remote_code: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[agg] loading {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=trust_remote_code,
    )
    head = model.lm_head.weight.detach().clone().float()
    del model
    print(f"[agg]   lm_head shape = {tuple(head.shape)}", flush=True)
    return head, tokenizer


def gold_to_token_ids(gold, tokenizer):
    """Return single-token candidate IDs for the gold answer.

    Tries both " 145" (with leading space, common BPE form) and "145" (bare).
    Returns empty list if neither tokenizes to a single token (e.g. multi-digit
    numbers that BPE splits, like "1234" -> ["12", "34"]).
    """
    s = str(int(gold)) if float(gold) == int(gold) else str(gold)
    candidates = []
    for variant in (" " + s, s):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            candidates.append(ids[0])
    return candidates


def compute_ranks_chunk(acts_chunk, lm_head, gold_ids_per_q):
    """
    acts_chunk: (B, S, L, H) float32
    Returns:    (B, S, L) long, rank of gold token in each cell. -1 if skipped.
    """
    B, S, L, H = acts_chunk.shape
    flat = acts_chunk.reshape(-1, H)                         # (B*S*L, H)
    logits = (flat @ lm_head.T).reshape(B, S, L, -1)         # (B, S, L, V)

    ranks = torch.full((B, S, L), -1, dtype=torch.long)
    for q in range(B):
        ids = gold_ids_per_q[q]
        if not ids:
            continue
        # Take the best (lowest) rank across candidate tokenizations.
        per_candidate = []
        for tid in ids:
            gold_logit = logits[q, :, :, tid].unsqueeze(-1)  # (S, L, 1)
            r = (logits[q] > gold_logit).sum(dim=-1)         # (S, L)
            per_candidate.append(r)
        ranks[q] = torch.stack(per_candidate, dim=0).min(dim=0).values
    return ranks


def compute_all_ranks(acts, lm_head, gold_ids_per_q, batch_size):
    N, S, L, _ = acts.shape
    out = torch.empty((N, S, L), dtype=torch.long)
    for i in range(0, N, batch_size):
        out[i:i + batch_size] = compute_ranks_chunk(
            acts[i:i + batch_size].float(), lm_head, gold_ids_per_q[i:i + batch_size]
        )
        print(f"[agg] ranks {min(i + batch_size, N)}/{N}", flush=True)
    return out


def outcome_cell(t_correct, s_correct):
    return f"{'TC' if t_correct else 'TI'}-{'SC' if s_correct else 'SI'}"


def aggregate(ranks, outcomes, topks):
    """For each k, produce {cell -> {'n': int, 'grid': (S, L) tensor of hit rates}}."""
    _, S, L = ranks.shape
    cells = ["ALL", "TC-SC", "TC-SI", "TI-SC", "TI-SI"]
    out = {}
    for k in topks:
        per_cell = {}
        for cell in cells:
            mask = torch.tensor([
                o["valid"] and (cell == "ALL" or o["cell"] == cell)
                for o in outcomes
            ])
            n = int(mask.sum())
            if n == 0:
                per_cell[cell] = {"n": 0, "grid": torch.zeros(S, L)}
                continue
            sel = ranks[mask]
            hits = (sel >= 0) & (sel < k)                # (n, S, L)
            per_cell[cell] = {"n": n, "grid": hits.float().mean(dim=0)}
        out[k] = per_cell
    return out


def plot_heatmaps(agg_for_k, k, out_path, dataset_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[agg] matplotlib not installed; skipping PNG. `pip install matplotlib` to enable.")
        return

    cells = ["TC-SC", "TC-SI", "TI-SC", "TI-SI"]
    titles = {
        "TC-SC": "Teacher right, Student right",
        "TC-SI": "Teacher right, Student wrong",
        "TI-SC": "Teacher wrong, Student right",
        "TI-SI": "Teacher wrong, Student wrong",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    grids = [agg_for_k[c]["grid"].numpy() for c in cells]
    vmax = max(0.05, max(g.max() for g in grids))
    S, L = grids[0].shape

    im = None
    for ax, cell, grid in zip(axes.ravel(), cells, grids):
        info = agg_for_k[cell]
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(f"{titles[cell]} (n={info['n']})", fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Latent step")
        ax.set_xticks(range(L))
        ax.set_yticks(range(S))
        for s in range(S):
            for l in range(L):
                v = grid[s, l]
                if v > 0:
                    color = "white" if v < vmax * 0.6 else "black"
                    ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color=color)
    fig.suptitle(f"{dataset_name}: P(gold in top-{k}) per (step, layer)", fontsize=12)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="hit rate")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[agg] saved {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_acts", required=True)
    p.add_argument("--student_results", required=True)
    p.add_argument("--teacher_results", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="required for models with custom code (e.g. Huginn-0125)")
    p.add_argument("--topk", type=int, nargs="+", default=[5, 10, 50])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dataset_name", default="SVAMP")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load activations + results.
    print(f"[agg] loading {args.student_acts}", flush=True)
    acts = torch.load(args.student_acts, map_location="cpu", weights_only=False)
    print(f"[agg]   shape = {tuple(acts.shape)}", flush=True)
    if acts.dim() != 4:
        raise ValueError(f"expected student 4D activations, got {acts.shape}")

    with open(args.student_results) as f:
        student_results = json.load(f)
    with open(args.teacher_results) as f:
        teacher_results = json.load(f)

    # Activations + student_results must agree positionally; teacher_results may
    # be a superset (e.g. full-1000 teacher run vs 50-question student smoke),
    # so we align on `idx` rather than position.
    if len(student_results) != acts.shape[0]:
        raise ValueError(
            f"student_results length {len(student_results)} != acts.shape[0] {acts.shape[0]}"
        )
    teacher_by_idx = {tr["idx"]: tr for tr in teacher_results}
    missing = [sr["idx"] for sr in student_results if sr["idx"] not in teacher_by_idx]
    if missing:
        raise ValueError(
            f"{len(missing)} student questions have no matching teacher result by idx; "
            f"first few: {missing[:5]}"
        )

    # 2. LM head + tokenizer.
    lm_head, tokenizer = load_lm_head(args.base_model, trust_remote_code=args.trust_remote_code)

    # 3. Gold token IDs + outcome cells.
    outcomes, gold_id_lists = [], []
    n_skipped = 0
    for q in range(acts.shape[0]):
        sr = student_results[q]
        tr = teacher_by_idx[sr["idx"]]
        ids = gold_to_token_ids(sr["gold"], tokenizer)
        outcomes.append({
            "idx": sr["idx"],
            "gold": sr["gold"],
            "teacher_correct": bool(tr["correct"]),
            "student_correct": bool(sr["correct"]),
            "cell": outcome_cell(bool(tr["correct"]), bool(sr["correct"])),
            "gold_token_ids": ids,
            "gold_token_strs": [tokenizer.decode([i]) for i in ids],
            "valid": len(ids) > 0,
        })
        gold_id_lists.append(ids)
        if not ids:
            n_skipped += 1
    print(f"[agg] gold-token coverage: {acts.shape[0] - n_skipped}/{acts.shape[0]} "
          f"({n_skipped} skipped, gold not single-token)", flush=True)

    # 4. Compute ranks.
    ranks = compute_all_ranks(acts, lm_head, gold_id_lists, args.batch_size)
    torch.save(ranks, out_dir / "ranks.pt")
    with open(out_dir / "outcomes.json", "w") as f:
        json.dump(outcomes, f, indent=2)
    print(f"[agg] saved ranks.pt and outcomes.json", flush=True)

    # 5. Aggregate.
    agg = aggregate(ranks, outcomes, args.topk)
    agg_json = {
        str(k): {cell: {"n": info["n"], "grid": info["grid"].tolist()}
                 for cell, info in per_cell.items()}
        for k, per_cell in agg.items()
    }
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(agg_json, f, indent=2)
    print(f"[agg] saved aggregate.json", flush=True)

    # 6. Summary table.
    print(f"\n[agg] Best-cell hit rate per outcome cell per k:")
    print(f"{'k':<6}{'cell':<8}{'n':<8}{'best_rate':<12}{'best (s,l)':<14}")
    for k in args.topk:
        for cell in ["ALL", "TC-SC", "TC-SI", "TI-SC", "TI-SI"]:
            info = agg[k][cell]
            if info["n"] == 0:
                continue
            grid = info["grid"]
            best = float(grid.max())
            flat_idx = int(grid.argmax())
            s, l = divmod(flat_idx, grid.shape[1])
            print(f"{k:<6}{cell:<8}{info['n']:<8}{best:<12.3f}({s},{l})")

    # 7. Heatmaps.
    for k in args.topk:
        plot_heatmaps(agg[k], k, out_dir / f"hit_rate_topk{k}.png", args.dataset_name)

    print(f"\n[agg] done. Outputs in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
