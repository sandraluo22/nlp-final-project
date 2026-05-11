"""Per-head per-core-block activation patching on Huginn-3.5B.

For each (core_block, head) cell at a fixed recurrence step K, patch that
head's contribution to the attention output (i.e. the d_head-wide slice that
feeds attn.proj) from a CLEAN forward into a CORRUPTED forward, and measure
the flip rate to the clean answer.

Architecture facts (huginn-0125):
  hidden_dim   = 5280
  n_heads      = 55
  d_head       = 96  (5280 / 55)
  attn.Wqkv    : Linear(5280 -> 15840)   # fused QKV
  attn.proj    : Linear(5280 -> 5280)    # output W_O
  4 core blocks, looped K times during the prompt forward

We hook with `forward_pre_hook` on each core_block[i].attn.proj. The pre-hook
input is shape (B, T, 5280) which is the concatenated per-head contributions
in head-major order: [h0 | h1 | ... | h54]. To patch head h we replace the
slice [..., h*d_head : (h+1)*d_head] with the cached clean tensor.

By default we only inject at the LAST prompt-token position and at recurrence
step --K_inject. Earlier steps are not patched.

Usage:
  python huginn-work/huginn/patching_per_head.py \\
      --pairs cf-datasets/numeral_pairs_b1_sub.json \\
      --K_inject 14 --num_steps 32 --batch 16 --num 60 \\
      --out huginn-work/visualizations/probes/patching_per_head_b1_sub.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset


MODEL_NAME = "tomg-group-umd/huginn-0125"
PROMPT_TEMPLATE = (
    "You are a careful step-by-step math problem solver. Solve the problem "
    "and end with 'The answer is <number>.'\n\n"
    "Question: {q}\nAnswer:"
)


def extract_answer(text: str) -> float:
    s = text.replace(",", "")
    for stop in ("\n\nQuestion:", "\nQuestion:", "Question:"):
        idx = s.find(stop)
        if idx > 0:
            s = s[:idx]
            break
    m = re.search(r"answer is\s*\$?\s*(-?\d+\.?\d*)", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    pred = re.findall(r"-?\d+\.?\d*", s)
    if not pred:
        return float("inf")
    return float(pred[-1])


# ---------------------------------------------------------------------------
# Hook state machine
# ---------------------------------------------------------------------------

class HookState:
    def __init__(self, n_blocks: int):
        self.n_blocks = n_blocks
        self.iter_per_block = [0] * n_blocks
        self.do_cache = False                # cache pre-attn.proj input at K_inject
        self.cache: dict[int, torch.Tensor] = {}   # block_idx -> (B, T, 5280)
        self.K_inject = 1
        self.patch_block: int | None = None
        self.patch_head: int | None = None
        self.d_head = 96


def make_pre_hook(state: HookState, block_idx: int):
    def _hook(_mod, inputs):
        # Track which iteration of this block we're on.
        state.iter_per_block[block_idx] += 1
        k = state.iter_per_block[block_idx]
        if k != state.K_inject:
            return None
        x = inputs[0]                                # (B, T, 5280)
        # Caching pass: store the full last-token slice.
        if state.do_cache:
            state.cache[block_idx] = x[:, -1, :].detach().clone()
            return None
        # Patching pass: if this is the targeted block + head, replace that slice.
        if state.patch_block == block_idx and state.patch_head is not None:
            h = state.patch_head
            d = state.d_head
            cached = state.cache.get(block_idx)
            if cached is None:
                return None
            # Patch the LAST token, head-h slice only.
            x_new = x.clone()
            x_new[:, -1, h*d:(h+1)*d] = cached[:, h*d:(h+1)*d]
            return (x_new,)
        return None
    return _hook


# ---------------------------------------------------------------------------
# Run a batch through Huginn with the hooks active.
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_batch(model, tok, texts, state, num_steps, max_new_tokens, fixed_max_len):
    inp = tok(texts, return_tensors="pt", padding="max_length",
              max_length=fixed_max_len, truncation=True,
              return_token_type_ids=False).to("cuda")
    # Reset per-block iteration counters.
    state.iter_per_block = [0] * state.n_blocks
    # Single forward to populate cache or apply patch.
    _ = model(**inp, num_steps=num_steps, use_cache=False)
    # Now generate; hooks will fire again during generation but we DON'T patch
    # there (state.K_inject is set; iteration counters reset on each forward).
    state.iter_per_block = [0] * state.n_blocks
    saved_K_inject = state.K_inject
    saved_pblock = state.patch_block
    state.K_inject = -1                     # disable injection during decode
    gen = model.generate(
        **inp, max_new_tokens=max_new_tokens, num_steps=num_steps,
        do_sample=False, pad_token_id=tok.pad_token_id,
    )
    state.K_inject = saved_K_inject
    state.patch_block = saved_pblock
    decoded = tok.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                               skip_special_tokens=True)
    return [extract_answer(t) for t in decoded]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--num", type=int, default=60)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--K_inject", type=int, default=14, help="recurrence step (1..K) at which to inject")
    p.add_argument("--num_steps", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pairs = json.load(open(args.pairs))[: args.num]
    print(f"loaded {len(pairs)} pairs", flush=True)

    print("loading Huginn", flush=True)
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda",
    )
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    H = model.config.n_embd
    n_heads = model.config.n_heads
    d_head = H // n_heads
    n_blocks = len(model.transformer.core_block)
    print(f"  H={H}  n_heads={n_heads}  d_head={d_head}  core_blocks={n_blocks}", flush=True)

    state = HookState(n_blocks)
    state.d_head = d_head
    state.K_inject = args.K_inject

    handles = []
    for i, blk in enumerate(model.transformer.core_block):
        handles.append(blk.attn.proj.register_forward_pre_hook(make_pre_hook(state, i)))

    try:
        # Tokenize once to find a global max prompt length.
        prompts_clean = [PROMPT_TEMPLATE.format(q=p["clean"]["text"]) for p in pairs]
        prompts_corr = [PROMPT_TEMPLATE.format(q=p["corrupted"]["text"]) for p in pairs]
        all_texts = prompts_clean + prompts_corr
        lens = [len(tok(t, add_special_tokens=True, return_token_type_ids=False)["input_ids"]) for t in all_texts]
        fixed_max_len = max(lens)
        print(f"  fixed prompt max_len = {fixed_max_len}", flush=True)

        # ---- Caching pass: per pair, cache CLEAN attn-proj inputs at K_inject for all 4 blocks.
        clean_cache = {}                     # (pair_idx, block) -> (1, 5280) tensor
        clean_preds, corr_preds = {}, {}
        print("\ncaching CLEAN forward + recording answers...", flush=True)
        state.do_cache = True
        state.patch_block = None; state.patch_head = None
        for st in range(0, len(pairs), args.batch):
            batch_pairs = pairs[st: st+args.batch]
            texts = [PROMPT_TEMPLATE.format(q=p["clean"]["text"]) for p in batch_pairs]
            preds = run_batch(model, tok, texts, state, args.num_steps,
                              args.max_new_tokens, fixed_max_len)
            for j, pred in enumerate(preds):
                clean_preds[st+j] = pred
            # state.cache now has block -> (B, 5280)
            for b_idx, t in state.cache.items():
                for j in range(t.shape[0]):
                    clean_cache[(st+j, b_idx)] = t[j:j+1].clone()
            state.cache = {}
        print("recording CORRUPTED baseline answers...", flush=True)
        state.do_cache = False
        for st in range(0, len(pairs), args.batch):
            batch_pairs = pairs[st: st+args.batch]
            texts = [PROMPT_TEMPLATE.format(q=p["corrupted"]["text"]) for p in batch_pairs]
            preds = run_batch(model, tok, texts, state, args.num_steps,
                              args.max_new_tokens, fixed_max_len)
            for j, pred in enumerate(preds):
                corr_preds[st+j] = pred

        # Filter to pairs where BOTH baselines are correct (clean=clean, corrupted=corrupted)
        keep = [
            i for i in range(len(pairs))
            if clean_preds[i] == pairs[i]["clean"]["answer"]
            and corr_preds[i] == pairs[i]["corrupted"]["answer"]
        ]
        print(f"\nbaseline clean correct: {sum(1 for i in range(len(pairs)) if clean_preds[i] == pairs[i]['clean']['answer'])}/{len(pairs)}")
        print(f"baseline corr correct:  {sum(1 for i in range(len(pairs)) if corr_preds[i] == pairs[i]['corrupted']['answer'])}/{len(pairs)}")
        print(f"kept (both correct):    {len(keep)}/{len(pairs)}")
        if not keep:
            print("no valid pairs; aborting"); return

        # ---- Patching sweep: for each (block, head) cell, run all kept pairs, measure flip.
        recovery = np.zeros((n_blocks, n_heads), dtype=np.float32)
        print(f"\npatching {n_blocks * n_heads} (block, head) cells at K={args.K_inject}", flush=True)
        for blk in range(n_blocks):
            for h in range(n_heads):
                state.patch_block = blk
                state.patch_head = h
                state.do_cache = False
                hits = 0
                for st in range(0, len(keep), args.batch):
                    batch_indices = keep[st: st+args.batch]
                    batch_pairs = [pairs[i] for i in batch_indices]
                    # Stash patches into a per-batch tensor: (B, 5280)
                    cached = torch.cat(
                        [clean_cache[(i, blk)] for i in batch_indices], dim=0
                    ).to("cuda")
                    # Inject this into the cache so the hook can read it. The hook
                    # uses state.cache[blk]; treat it as the per-batch clean tensor.
                    state.cache = {blk: cached}
                    texts = [PROMPT_TEMPLATE.format(q=p["corrupted"]["text"]) for p in batch_pairs]
                    preds = run_batch(model, tok, texts, state, args.num_steps,
                                      args.max_new_tokens, fixed_max_len)
                    for j, pred in enumerate(preds):
                        i = batch_indices[j]
                        if pred == pairs[i]["clean"]["answer"]:
                            hits += 1
                    state.cache = {}
                recovery[blk, h] = hits / len(keep)
            print(f"  block {blk}/{n_blocks-1}: heads top-3 by recovery = "
                  f"{[(int(h_), float(recovery[blk, h_])) for h_ in np.argsort(-recovery[blk])[:3]]}",
                  flush=True)

    finally:
        for h in handles:
            h.remove()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "K_inject": args.K_inject,
        "num_steps": args.num_steps,
        "n_blocks": n_blocks,
        "n_heads": n_heads,
        "n_pairs": len(pairs),
        "n_kept": len(keep),
        "recovery": recovery.tolist(),
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}")
    print("\nrecovery (% of kept that flipped to clean) per (block, head):")
    for blk in range(n_blocks):
        top = np.argsort(-recovery[blk])[:5]
        print(f"  block {blk}: max={recovery[blk].max()*100:.1f}%  "
              f"top heads: " +
              "  ".join(f"h{int(h)}={recovery[blk, h]*100:.0f}%" for h in top))


if __name__ == "__main__":
    main()
