"""Corrected logit lens + narrowing-down analysis using CODI-GPT-2's actual
lm_head (not vanilla GPT-2's).

For every SVAMP problem:
  Candidate set = {a-b, a+b, a*b, a÷b}  (the 4 operator answers)
  At every (layer, latent_step) for the LAST PROMPT TOKEN and at every
  layer for the ANSWER POSITION, compute:

  - top-1 token id  (and whether it's the gold-token)
  - rank of the gold-answer token
  - rank of each candidate-answer token
  - probability mass on each candidate
  - distribution entropy

Outputs:
  codi-work/experiments/computation_probes/gpt2_corrected_lens.json
  codi-work/experiments/computation_probes/gpt2_narrowing_summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset


REPO = Path(__file__).resolve().parents[2]
ACTS_PROMPT = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
ACTS_DECODE = REPO / "experiments" / "computation_probes" / "svamp_decode_acts.pt"
LM_HEAD = REPO / "experiments" / "computation_probes" / "codi_gpt2_lm_head.npy"
OUT = REPO / "experiments" / "computation_probes"


def first_token_id(tok, n):
    """First token id of the leading-space numeric string ' n'."""
    s = f" {int(n) if n == int(n) else n}"
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else tok.eos_token_id


def main():
    print("loading activations + lm_head", flush=True)
    a_prompt = torch.load(ACTS_PROMPT, map_location="cpu", weights_only=True).float().numpy()
    a_decode = torch.load(ACTS_DECODE, map_location="cpu", weights_only=True).float().numpy()
    W = np.load(LM_HEAD)                                            # (V, H)
    print(f"  prompt acts: {a_prompt.shape}  decode acts: {a_decode.shape}  W: {W.shape}", flush=True)
    N, S, L, H = a_prompt.shape

    # Load SVAMP labels + numerals
    print("loading SVAMP", flush=True)
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array([t.replace("Common-Divison", "Common-Division") for t in full["Type"]])[:N]
    answers = np.array([float(str(x).replace(",", "")) for x in full["Answer"]])[:N]
    nums_per = []
    for ex in full[:N]["Equation"] if False else [e["Equation"] for e in full][:N]:
        toks = re.findall(r"\d+\.?\d*", ex)
        nums = [float(t) for t in toks[:2]] if len(toks) >= 2 else [None, None]
        nums_per.append(nums)

    tok = transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # Build candidate-token-ids per problem: {a-b, a+b, a*b, a/b (round)}
    gold_id = np.zeros(N, dtype=np.int64)
    cand_ids = np.zeros((N, 4), dtype=np.int64)              # (sub, add, mul, div)
    cand_vals = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        a, b = nums_per[i]
        gold_id[i] = first_token_id(tok, answers[i])
        if a is None or b is None:
            cand_ids[i] = gold_id[i]
            cand_vals[i] = 0.0
            continue
        big, small = (a, b) if a >= b else (b, a)
        sub = big - small
        add = a + b
        mul = a * b
        div = big / small if small != 0 else 0.0
        cand_vals[i] = (sub, add, mul, div)
        for k, v in enumerate((sub, add, mul, div)):
            cand_ids[i, k] = first_token_id(tok, v)

    # ------------ Helper: compute per-layer logits ------------
    def lens_metrics(acts, layer_axis="step"):
        """acts shape (N, S, L, H) for prompt or (N, L, H) for decode.
        Returns dict of arrays keyed by 'shape'."""
        if acts.ndim == 4:
            N, S, L, H = acts.shape
            shapes = (L, S)
            iter_LS = ((l, s) for l in range(L) for s in range(S))
        else:
            N, L, H = acts.shape; S = 1
            shapes = (L,)
            iter_LS = ((l, None) for l in range(L))
        # Rank, prob, entropy per (layer[, step]) per example
        rank_gold = np.empty((N, *shapes), dtype=np.int32)
        rank_cand = np.empty((N, *shapes, 4), dtype=np.int32)
        prob_gold = np.empty((N, *shapes), dtype=np.float32)
        prob_cand = np.empty((N, *shapes, 4), dtype=np.float32)
        entropy   = np.empty((N, *shapes), dtype=np.float32)
        top1_eq_gold = np.zeros((N, *shapes), dtype=bool)

        if acts.ndim == 4:
            for l in range(L):
                for s in range(S):
                    R = acts[:, s, l, :]                           # (N, H)
                    logits = R @ W.T                               # (N, V)
                    # softmax
                    logits -= logits.max(axis=1, keepdims=True)
                    p = np.exp(logits)
                    p /= p.sum(axis=1, keepdims=True)
                    H_ent = -(p * np.where(p > 0, np.log(p + 1e-30), 0)).sum(axis=1)
                    entropy[:, l, s] = H_ent
                    gold_logit = logits[np.arange(N), gold_id]
                    rank_gold[:, l, s] = (logits > gold_logit[:, None]).sum(axis=1)
                    prob_gold[:, l, s] = p[np.arange(N), gold_id]
                    top1_eq_gold[:, l, s] = (logits.argmax(axis=1) == gold_id)
                    for k in range(4):
                        cl = logits[np.arange(N), cand_ids[:, k]]
                        rank_cand[:, l, s, k] = (logits > cl[:, None]).sum(axis=1)
                        prob_cand[:, l, s, k] = p[np.arange(N), cand_ids[:, k]]
        else:
            for l in range(L):
                R = acts[:, l, :]
                logits = R @ W.T
                logits -= logits.max(axis=1, keepdims=True)
                p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
                H_ent = -(p * np.where(p > 0, np.log(p + 1e-30), 0)).sum(axis=1)
                entropy[:, l] = H_ent
                gold_logit = logits[np.arange(N), gold_id]
                rank_gold[:, l] = (logits > gold_logit[:, None]).sum(axis=1)
                prob_gold[:, l] = p[np.arange(N), gold_id]
                top1_eq_gold[:, l] = (logits.argmax(axis=1) == gold_id)
                for k in range(4):
                    cl = logits[np.arange(N), cand_ids[:, k]]
                    rank_cand[:, l, k] = (logits > cl[:, None]).sum(axis=1)
                    prob_cand[:, l, k] = p[np.arange(N), cand_ids[:, k]]
        return dict(rank_gold=rank_gold, rank_cand=rank_cand,
                    prob_gold=prob_gold, prob_cand=prob_cand,
                    entropy=entropy, top1_eq_gold=top1_eq_gold)

    print("\n=== prompt-position logit lens ===", flush=True)
    pm = lens_metrics(a_prompt)
    print(f"  prompt: top-1=={int(pm['top1_eq_gold'].any(axis=(1,2)).sum())}/{N} ever in any (L, S)")
    pct_top1_per_LS = pm["top1_eq_gold"].mean(axis=0) * 100      # (L, S)
    print(f"  pct top-1 (gold) at any (L, S) max: {pct_top1_per_LS.max():.1f}%")

    print("\n=== decode-position logit lens ===", flush=True)
    dm = lens_metrics(a_decode)
    print(f"  decode: top-1=={int(dm['top1_eq_gold'].any(axis=1).sum())}/{N} ever per layer")
    print(f"  pct top-1 per layer: {[f'{v:.1f}%' for v in dm['top1_eq_gold'].mean(axis=0)*100]}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "gpt2_corrected_lens.npz",
             prompt_rank_gold=pm["rank_gold"], prompt_rank_cand=pm["rank_cand"],
             prompt_prob_gold=pm["prob_gold"], prompt_prob_cand=pm["prob_cand"],
             prompt_entropy=pm["entropy"], prompt_top1=pm["top1_eq_gold"],
             decode_rank_gold=dm["rank_gold"], decode_rank_cand=dm["rank_cand"],
             decode_prob_gold=dm["prob_gold"], decode_prob_cand=dm["prob_cand"],
             decode_entropy=dm["entropy"], decode_top1=dm["top1_eq_gold"],
             cand_ids=cand_ids, gold_id=gold_id, types=types,
             cand_vals=cand_vals, answers=answers)
    print(f"saved {OUT/'gpt2_corrected_lens.npz'}")

    # ---------- summary table & narrowing plot ----------
    # Average rank of each candidate at decode position, per layer
    cand_names = ["sub (a-b)", "add (a+b)", "mul (a*b)", "div (a/b)"]
    cand_color = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]

    # Decode-position narrowing
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    Ls = np.arange(dm["entropy"].shape[1])

    ax = axes[0]
    for k, (name, color) in enumerate(zip(cand_names, cand_color)):
        med_rank = np.median(dm["rank_cand"][:, :, k], axis=0)
        ax.plot(Ls, med_rank, "o-", color=color, label=name, lw=1.8, ms=5)
    med_gold = np.median(dm["rank_gold"], axis=0)
    ax.plot(Ls, med_gold, "k--", label="gold (= correct cand)", lw=2, alpha=0.7)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("layer (decode position)"); ax.set_ylabel("median rank in vocab")
    ax.set_title("(narrowing) median rank of each candidate token")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for k, (name, color) in enumerate(zip(cand_names, cand_color)):
        mean_p = dm["prob_cand"][:, :, k].mean(axis=0)
        ax.plot(Ls, mean_p, "o-", color=color, label=name, lw=1.8, ms=5)
    mean_p_gold = dm["prob_gold"].mean(axis=0)
    ax.plot(Ls, mean_p_gold, "k--", label="gold", lw=2, alpha=0.7)
    ax.set_xlabel("layer"); ax.set_ylabel("mean probability")
    ax.set_title("mean probability mass on each candidate")
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    mean_ent = dm["entropy"].mean(axis=0)
    pct_top1 = dm["top1_eq_gold"].mean(axis=0) * 100
    ax.plot(Ls, mean_ent, "o-", color="#9467bd", label="mean entropy", lw=2)
    ax.set_xlabel("layer"); ax.set_ylabel("mean entropy", color="#9467bd")
    ax.tick_params(axis="y", labelcolor="#9467bd")
    ax2 = ax.twinx()
    ax2.plot(Ls, pct_top1, "s-", color="#d62728", label="% top-1 = gold", lw=2)
    ax2.set_ylabel("% top-1 = gold", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("entropy and top-1-= gold % (decode position)")
    ax.grid(alpha=0.3)

    fig.suptitle("CODI-GPT-2  ·  decode-position narrowing of operator candidates  (N=1000)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "gpt2_narrowing_decode.png", dpi=140); plt.close(fig)
    print(f"saved {OUT/'gpt2_narrowing_decode.png'}")

    # Prompt-position narrowing — heatmap of median rank per (layer, step) per candidate
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    for ax, k, name in zip(axes.ravel(), range(4), cand_names):
        med = np.median(pm["rank_cand"][:, :, :, k], axis=0)        # (L, S)
        im = ax.imshow(np.log10(med + 1), aspect="auto", origin="lower", cmap="viridis_r")
        ax.set_title(f"{name}  median log10(rank+1)")
        ax.set_xlabel("latent step"); ax.set_ylabel("layer")
        ax.set_xticks(range(med.shape[1]))
        ax.set_xticklabels([str(s+1) for s in range(med.shape[1])])
        ax.set_yticks(range(med.shape[0]))
        plt.colorbar(im, ax=ax, label="log10(rank+1)")
    fig.suptitle("CODI-GPT-2  ·  prompt-position median rank per candidate (lower = closer to top)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "gpt2_narrowing_prompt.png", dpi=140); plt.close(fig)
    print(f"saved {OUT/'gpt2_narrowing_prompt.png'}")

    # ---------- text summary ----------
    summary = {
        "decode_layers": dm["entropy"].shape[1],
        "decode_pct_top1_gold_per_layer": (dm["top1_eq_gold"].mean(axis=0) * 100).tolist(),
        "decode_mean_entropy_per_layer": dm["entropy"].mean(axis=0).tolist(),
        "decode_median_rank_gold_per_layer": np.median(dm["rank_gold"], axis=0).tolist(),
        "decode_median_rank_cand_per_layer": [
            np.median(dm["rank_cand"][:, :, k], axis=0).tolist() for k in range(4)
        ],
        "candidate_names": cand_names,
    }
    (OUT / "gpt2_narrowing_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT/'gpt2_narrowing_summary.json'}")

    # Print compact table
    print("\n" + "="*80)
    print("Decode-position table per layer (median ranks, % top1, entropy)")
    print("="*80)
    print(f"  L  | rank_gold | rank_sub  rank_add  rank_mul  rank_div | %top1 | entropy")
    for L_ in range(dm["entropy"].shape[1]):
        rg = np.median(dm["rank_gold"][:, L_])
        rs, ra, rm, rd = [np.median(dm["rank_cand"][:, L_, k]) for k in range(4)]
        pt = dm["top1_eq_gold"][:, L_].mean() * 100
        ent = dm["entropy"][:, L_].mean()
        print(f"  {L_:>2} | {rg:>9.0f} | {rs:>8.0f}  {ra:>8.0f}  {rm:>8.0f}  {rd:>8.0f} | {pt:>5.1f} | {ent:>6.2f}")


if __name__ == "__main__":
    main()
