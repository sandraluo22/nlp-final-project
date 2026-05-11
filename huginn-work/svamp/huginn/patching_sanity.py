"""Sanity check v2 — hook patches DURING the prompt forward inside generate.

We rely on input.shape[1] > 1 to identify the prompt forward (vs subsequent
single-token decode forwards). This way the modification is in the KV cache
that's used for the rest of the generation.

Three tests at (block 0, K=14):
  (1) BASELINE        : no hook intervention.
  (2) ZERO-PATCH      : zero out block 0 attn output at K=14, last prompt token.
  (3) FULL-CLEAN-PATCH: replay the whole 5280-d attn output (all 55 heads) from
                        the cached CLEAN prompt forward, into the corrupted run.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

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

K_INJECT = 14
NUM_STEPS = 32
N_PAIRS = 10
TARGET_BLOCK = 0


def extract_answer(text):
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
    return float(pred[-1]) if pred else float("inf")


def main():
    pairs = json.load(open(PAIRS_PATH))[:N_PAIRS]
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

    state = {
        "iter": 0, "K_inject": K_INJECT,
        "mode": "off", "cache": None, "fired_count": 0,
    }

    def pre_hook(_mod, inputs):
        x = inputs[0]
        T = x.shape[1]
        if T < 2:
            return None                              # decode-time forward, skip
        # New prompt forward — reset iteration counter the first time we see it
        # (the recurrent core loops 32 times; we count them).
        if state["iter"] >= NUM_STEPS:
            state["iter"] = 0
        state["iter"] += 1
        k = state["iter"]
        if k != state["K_inject"] or state["mode"] == "off":
            return None
        state["fired_count"] += 1
        if state["mode"] == "cache":
            state["cache"] = x[:, -1, :].detach().clone()
            return None
        elif state["mode"] == "zero":
            x_new = x.clone(); x_new[:, -1, :] = 0.0
            return (x_new,)
        elif state["mode"] == "full_clean":
            if state["cache"] is None: return None
            x_new = x.clone()
            x_new[:, -1, :] = state["cache"]
            return (x_new,)
        return None

    handle = model.transformer.core_block[TARGET_BLOCK].attn.proj.register_forward_pre_hook(pre_hook)

    @torch.no_grad()
    def gen(texts, fixed_max_len, mode):
        inp = tok(texts, return_tensors="pt", padding="max_length",
                  max_length=fixed_max_len, truncation=True,
                  return_token_type_ids=False).to("cuda")
        state["iter"] = 0
        state["mode"] = mode
        state["fired_count"] = 0
        gen_out = model.generate(
            **inp, max_new_tokens=64, num_steps=NUM_STEPS, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        decoded = tok.batch_decode(gen_out[:, inp["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
        return [extract_answer(t) for t in decoded], state["fired_count"]

    prompts_clean = [PROMPT_TEMPLATE.format(q=p["clean"]["text"]) for p in pairs]
    prompts_corr = [PROMPT_TEMPLATE.format(q=p["corrupted"]["text"]) for p in pairs]
    fixed_max_len = max(len(tok(t, return_token_type_ids=False)["input_ids"])
                        for t in prompts_clean + prompts_corr)
    print(f"  fixed_max_len = {fixed_max_len}\n", flush=True)

    # ---- (1) BASELINE ----
    print("=== (1) BASELINE: no hook intervention ===")
    clean_baseline, fired = gen(prompts_clean, fixed_max_len, "off")
    print(f"  hook fired (should be 0): {fired}")
    n_clean = sum(1 for i, p in enumerate(pairs) if clean_baseline[i] == p["clean"]["answer"])
    print(f"  clean baseline correct: {n_clean}/{len(pairs)}")
    corr_baseline, fired = gen(prompts_corr, fixed_max_len, "off")
    n_corr = sum(1 for i, p in enumerate(pairs) if corr_baseline[i] == p["corrupted"]["answer"])
    print(f"  corr  baseline correct: {n_corr}/{len(pairs)}")
    print(f"  sample corr preds[:5]: {corr_baseline[:5]}")

    # ---- (2) ZERO-PATCH ----
    print("\n=== (2) ZERO-PATCH: zero out block 0 attn output at K=14 ===")
    state["cache"] = None
    zero_corr, fired = gen(prompts_corr, fixed_max_len, "zero")
    print(f"  hook fired (should be 1, the prompt forward): {fired}")
    n_unchanged = sum(1 for i in range(len(pairs)) if zero_corr[i] == corr_baseline[i])
    print(f"  outputs unchanged from corr-baseline: {n_unchanged}/{len(pairs)}")
    print(f"  sample zero-patched preds[:5]:    {zero_corr[:5]}")

    # ---- (3) FULL-CLEAN-PATCH ----
    print("\n=== (3) FULL-CLEAN-PATCH: ALL 55 heads, clean→corr at (block 0, K=14) ===")
    state["cache"] = None
    _, _ = gen(prompts_clean, fixed_max_len, "cache")
    print(f"  cache populated: {state['cache'] is not None}, "
          f"shape={tuple(state['cache'].shape) if state['cache'] is not None else None}")
    fc_corr, fired = gen(prompts_corr, fixed_max_len, "full_clean")
    print(f"  hook fired (should be 1): {fired}")
    n_flip = sum(1 for i, p in enumerate(pairs) if fc_corr[i] == p["clean"]["answer"])
    n_unchanged = sum(1 for i in range(len(pairs)) if fc_corr[i] == corr_baseline[i])
    print(f"  flipped to clean: {n_flip}/{len(pairs)}")
    print(f"  unchanged from corr-baseline: {n_unchanged}/{len(pairs)}")
    print(f"  sample full-patch preds[:5]:    {fc_corr[:5]}")

    handle.remove()
    print("\n=== INTERPRETATION ===")
    if n_unchanged == len(pairs):
        print("  → Even zeroing/full-patching attn at this cell doesn't change a single output.")
        print("    Attention output at (block 0, K=14) is genuinely not causal here.")
    elif n_flip > 0:
        print(f"  → Full-attn patching flipped {n_flip} pairs. Per-head sweep should be redone with this hook.")
    else:
        print("  → Hook works (some outputs change), full-attn flips 0 pairs anyway. Attn not causal.")


if __name__ == "__main__":
    main()
