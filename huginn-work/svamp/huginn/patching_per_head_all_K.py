"""For each (core_block, head), zero out that head's contribution at ALL 32
recurrence steps simultaneously. Tests whether ANY single head is essential
when ablated across the entire recurrent depth.

Compared to the per-(block, head, K=14) sweep: that hit ~0% because losing one
head at one K can be compensated by other K's. Here we deny the model that
escape route by killing the head at every K.

Output:
  huginn-work/visualizations/probes/patching_per_head_all_K.png
  huginn-work/visualizations/probes/patching_per_head_all_K.json
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
PAIRS = ROOT / "cf-datasets" / "numeral_pairs_b1_sub.json"
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
    args = p.parse_args()
    pairs = json.load(open(PAIRS))[:args.num]
    print(f"loaded {len(pairs)} pairs", flush=True)

    print("loading Huginn", flush=True)
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    H = model.config.n_embd; n_heads = model.config.n_heads
    d_head = H // n_heads; n_blocks = len(model.transformer.core_block)
    print(f"  H={H}  n_heads={n_heads}  d_head={d_head}", flush=True)

    state = {"target_block": -1, "target_head": -1, "block_iter": [0]*n_blocks}

    def make_hook(b):
        def pre_hook(_mod, inputs):
            x = inputs[0]
            if x.shape[1] < 2: return None
            if state["block_iter"][b] >= NUM_STEPS:
                state["block_iter"][b] = 0
            state["block_iter"][b] += 1
            if state["target_block"] != b or state["target_head"] < 0:
                return None
            # Zero this head at LAST PROMPT TOKEN, every K (no K filter).
            h = state["target_head"]
            x_new = x.clone()
            x_new[:, -1, h*d_head:(h+1)*d_head] = 0.0
            return (x_new,)
        return pre_hook

    handles = [blk.attn.proj.register_forward_pre_hook(make_hook(i))
               for i, blk in enumerate(model.transformer.core_block)]

    @torch.no_grad()
    def gen(prompts, fixed_max_len, target_block, target_head):
        for i in range(n_blocks): state["block_iter"][i] = 0
        state["target_block"], state["target_head"] = target_block, target_head
        inp = tok(prompts, return_tensors="pt", padding="max_length",
                  max_length=fixed_max_len, truncation=True,
                  return_token_type_ids=False).to("cuda")
        out = model.generate(**inp, max_new_tokens=64, num_steps=NUM_STEPS,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        return [extract_answer(t) for t in
                tok.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True)]

    prompts = [PROMPT_TEMPLATE.format(q=p["corrupted"]["text"]) for p in pairs]
    fixed_max_len = max(len(tok(t, return_token_type_ids=False)["input_ids"])
                        for t in prompts)

    print("\nbaseline (no patching)...", flush=True)
    base = gen(prompts, fixed_max_len, -1, -1)
    n_corr = sum(1 for i, p in enumerate(pairs) if base[i] == p["corrupted"]["answer"])
    print(f"  corrupted baseline correct: {n_corr}/{len(pairs)}")

    grid = np.zeros((n_blocks, n_heads), dtype=np.float32)
    print(f"\nzero head h at all K=1..{NUM_STEPS} for {n_blocks*n_heads} cells", flush=True)
    for b in range(n_blocks):
        for h in range(n_heads):
            preds = gen(prompts, fixed_max_len, b, h)
            n_changed = sum(1 for i in range(len(pairs)) if preds[i] != base[i])
            grid[b, h] = n_changed / len(pairs)
        top = np.argsort(-grid[b])[:5]
        print(f"  block {b}: max change {grid[b].max()*100:.0f}%  top heads: " +
              "  ".join(f"h{int(h)}={grid[b, h]*100:.0f}%" for h in top), flush=True)

    for h in handles: h.remove()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "patching_per_head_all_K.json"
    out_json.write_text(json.dumps({
        "n_pairs": len(pairs), "n_corr_correct": n_corr,
        "grid_change_rate": grid.tolist(),
    }, indent=2))
    print(f"\nsaved {out_json}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(15, 3.2))
    im = ax.imshow(grid * 100, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([str(h) for h in range(n_heads)], fontsize=6)
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels([f"block {b}" for b in range(n_blocks)])
    ax.set_xlabel("head index (0..54)")
    ax.set_title(
        f"Huginn  ·  zero head h at ALL K=1..32, % outputs changed from baseline  "
        f"(N={len(pairs)} pairs)\n"
        f"high change ⇒ that single head is doing irreplaceable work across the recurrence")
    cb = plt.colorbar(im, ax=ax, label="% outputs changed")
    fig.tight_layout()
    out_png = OUT_DIR / "patching_per_head_all_K.png"
    fig.savefig(out_png, dpi=140)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
