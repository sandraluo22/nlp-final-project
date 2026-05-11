"""K-scan zero-patch sweep on Huginn attn output.

For each (core_block, recurrence step K) cell in 4 × 32, zero out the attn-
proj input at that cell (last prompt token only) and measure the fraction of
outputs that change from the unhooked baseline. High change rate ⇒ that
attn cell is causally load-bearing for these problems.

Also includes a "all 4 blocks at one K" row, to show the joint effect of
losing all attention at a given recurrence step.

Output:
  huginn-work/visualizations/probes/patching_K_scan_b1_sub.json
  huginn-work/visualizations/probes/patching_K_scan_b1_sub.png
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers


MODEL_NAME = "tomg-group-umd/huginn-0125"
PROMPT_TEMPLATE = (
    "You are a careful step-by-step math problem solver. Solve the problem "
    "and end with 'The answer is <number>.'\n\n"
    "Question: {q}\nAnswer:"
)
ROOT = Path(__file__).resolve().parents[2]
PAIRS_PATH = ROOT / "cf-datasets" / "numeral_pairs_b1_sub.json"
OUT_DIR = ROOT / "huginn-work" / "visualizations" / "probes"

NUM_STEPS = 32


def extract_answer(text):
    s = text.replace(",", "")
    for stop in ("\n\nQuestion:", "\nQuestion:"):
        idx = s.find(stop)
        if idx > 0: s = s[:idx]; break
    m = re.search(r"answer is\s*\$?\s*(-?\d+\.?\d*)", s, re.IGNORECASE)
    if m: return float(m.group(1))
    pred = re.findall(r"-?\d+\.?\d*", s)
    return float(pred[-1]) if pred else float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=20)
    p.add_argument("--batch", type=int, default=20)
    args = p.parse_args()

    pairs = json.load(open(PAIRS_PATH))[:args.num]
    n_pairs = len(pairs)
    print(f"loaded {n_pairs} pairs", flush=True)

    print("loading Huginn", flush=True)
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda",
    )
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    n_blocks = len(model.transformer.core_block)

    state = {"target_block": -1, "K_target": -1, "block_iter": [0]*n_blocks}

    def make_hook(block_idx):
        def pre_hook(_mod, inputs):
            x = inputs[0]
            if x.shape[1] < 2: return None
            if state["block_iter"][block_idx] >= NUM_STEPS:
                state["block_iter"][block_idx] = 0
            state["block_iter"][block_idx] += 1
            k = state["block_iter"][block_idx]
            if k != state["K_target"]: return None
            if state["target_block"] != -1 and block_idx != state["target_block"]:
                return None
            x_new = x.clone()
            x_new[:, -1, :] = 0.0
            return (x_new,)
        return pre_hook

    handles = [blk.attn.proj.register_forward_pre_hook(make_hook(i))
               for i, blk in enumerate(model.transformer.core_block)]

    @torch.no_grad()
    def gen(target_block, K_target, fixed_max_len, prompts):
        for i in range(n_blocks): state["block_iter"][i] = 0
        state["target_block"], state["K_target"] = target_block, K_target
        inp = tok(prompts, return_tensors="pt", padding="max_length",
                  max_length=fixed_max_len, truncation=True,
                  return_token_type_ids=False).to("cuda")
        out = model.generate(**inp, max_new_tokens=64, num_steps=NUM_STEPS,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        return [extract_answer(t) for t in
                tok.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True)]

    prompts_corr = [PROMPT_TEMPLATE.format(q=p["corrupted"]["text"]) for p in pairs]
    fixed_max_len = max(len(tok(t, return_token_type_ids=False)["input_ids"])
                        for t in prompts_corr)
    print(f"  fixed_max_len = {fixed_max_len}", flush=True)

    # Baseline (no patching)
    print("\n=== BASELINE (no patching) ===")
    base = gen(-1, -1, fixed_max_len, prompts_corr)   # K_target=-1 disables hook
    n_corr_correct = sum(1 for i, p in enumerate(pairs) if base[i] == p["corrupted"]["answer"])
    print(f"  baseline correct: {n_corr_correct}/{n_pairs}")

    # Per-(block, K) zero patch — 4 × 32 = 128 cells
    grid = np.zeros((n_blocks + 1, NUM_STEPS), dtype=np.float32)   # +1 row for "all blocks"
    print(f"\n=== zero-patch per (block, K), {n_blocks * NUM_STEPS} cells ===")
    for blk in range(n_blocks):
        for K in range(1, NUM_STEPS + 1):
            preds = gen(blk, K, fixed_max_len, prompts_corr)
            n_changed = sum(1 for i in range(n_pairs) if preds[i] != base[i])
            grid[blk, K-1] = n_changed / n_pairs
        print(f"  block {blk}: row max change rate = {grid[blk].max()*100:.0f}% "
              f"(at K={int(np.argmax(grid[blk]))+1})", flush=True)

    print(f"\n=== zero-patch ALL 4 blocks × one K ===")
    for K in range(1, NUM_STEPS + 1):
        preds = gen(-1, K, fixed_max_len, prompts_corr)   # target_block=-1 => all blocks
        n_changed = sum(1 for i in range(n_pairs) if preds[i] != base[i])
        grid[n_blocks, K-1] = n_changed / n_pairs
    print(f"  all-blocks row max change = {grid[n_blocks].max()*100:.0f}% "
          f"(at K={int(np.argmax(grid[n_blocks]))+1})", flush=True)

    for h in handles: h.remove()

    # Save JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "patching_K_scan_b1_sub.json"
    summary = {
        "n_pairs": n_pairs, "n_corr_correct_baseline": n_corr_correct,
        "K_range": [1, NUM_STEPS],
        "row_labels": [f"block {b}" for b in range(n_blocks)] + ["all blocks"],
        "grid_change_rate": grid.tolist(),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_json}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(grid * 100, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(np.arange(NUM_STEPS))
    ax.set_xticklabels([str(k+1) for k in range(NUM_STEPS)], fontsize=8)
    ax.set_yticks(np.arange(n_blocks + 1))
    ax.set_yticklabels(summary["row_labels"], fontsize=10)
    ax.set_xlabel("recurrence step K")
    ax.set_ylabel("attn-zero target")
    ax.set_title(
        f"Huginn  ·  zero-patch attn output, change rate from baseline "
        f"(N={n_pairs} corrupted-prompt pairs)\n"
        f"low change ⇒ attn at that (block, K) is dispensable    ·    "
        f"high change ⇒ that cell is load-bearing"
    )
    cb = plt.colorbar(im, ax=ax, label="% of outputs that changed")
    # annotate cells
    for r in range(n_blocks + 1):
        for c in range(NUM_STEPS):
            v = grid[r, c] * 100
            if v >= 5:
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if v < 50 else "black")
    fig.tight_layout()
    out_png = OUT_DIR / "patching_K_scan_b1_sub.png"
    fig.savefig(out_png, dpi=140)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
