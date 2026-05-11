"""Sanity check v3 — aggressive destructive tests on Huginn attn output.

Tests on 5 corrupted-pair prompts:
  (a) print the L2 norm of the attn-proj input at (block 0, K=14, last token)
      — confirms the activation is non-trivial.
  (b) zero attn-proj input at ALL 4 blocks × K=14 (full last-token slice).
  (c) zero attn-proj input at ALL 4 blocks × ALL 32 recurrence steps for the
      last token (extreme — should completely break the model if attn matters).
  (d) zero the FULL last-token row of the residual (across whole sequence —
      via inserting random noise into the embedding) — guaranteed-destructive
      control, confirms generate path is reachable.
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

NUM_STEPS = 32
N_PAIRS = 5


def extract_answer(text):
    s = text.replace(",", "")
    for stop in ("\n\nQuestion:", "\nQuestion:"):
        idx = s.find(stop)
        if idx > 0:
            s = s[:idx]; break
    m = re.search(r"answer is\s*\$?\s*(-?\d+\.?\d*)", s, re.IGNORECASE)
    if m: return float(m.group(1))
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

    n_blocks = len(model.transformer.core_block)

    # iter counter shared across all 4 hooks (incremented on the first block per K)
    state = {"iter": 0, "K_target": [], "mode": "off",
             "norms_logged": False, "block_iter": [0]*n_blocks}

    def make_hook(block_idx):
        def pre_hook(_mod, inputs):
            x = inputs[0]
            T = x.shape[1]
            if T < 2:
                return None
            # Each block's hook fires once per recurrence step. Use per-block counter.
            if state["block_iter"][block_idx] >= NUM_STEPS:
                state["block_iter"][block_idx] = 0
            state["block_iter"][block_idx] += 1
            k = state["block_iter"][block_idx]

            if state["mode"] == "off":
                return None
            if k not in state["K_target"]:
                return None

            # log L2 norm at (block 0, K=14) once for visibility
            if state["mode"] == "log_norms":
                if not state["norms_logged"]:
                    print(f"    [log] block {block_idx} K={k}  ‖x[:,-1,:]‖₂ = "
                          f"{torch.linalg.vector_norm(x[:, -1, :], dim=-1).tolist()}",
                          flush=True)
                return None

            if state["mode"] == "zero":
                x_new = x.clone()
                x_new[:, -1, :] = 0.0
                return (x_new,)
            return None
        return pre_hook

    handles = []
    for i, blk in enumerate(model.transformer.core_block):
        handles.append(blk.attn.proj.register_forward_pre_hook(make_hook(i)))

    @torch.no_grad()
    def gen(texts, fixed_max_len, mode, K_target):
        inp = tok(texts, return_tensors="pt", padding="max_length",
                  max_length=fixed_max_len, truncation=True,
                  return_token_type_ids=False).to("cuda")
        for i in range(n_blocks): state["block_iter"][i] = 0
        state["mode"] = mode
        state["K_target"] = K_target
        state["norms_logged"] = False
        gen_out = model.generate(
            **inp, max_new_tokens=64, num_steps=NUM_STEPS, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        return [extract_answer(t) for t in
                tok.batch_decode(gen_out[:, inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True)]

    prompts_corr = [PROMPT_TEMPLATE.format(q=p["corrupted"]["text"]) for p in pairs]
    fixed_max_len = max(len(tok(t, return_token_type_ids=False)["input_ids"])
                        for t in prompts_corr)
    print(f"  fixed_max_len = {fixed_max_len}\n", flush=True)

    # Baseline
    print("=== BASELINE corrupted ===")
    base = gen(prompts_corr, fixed_max_len, "off", [])
    print(f"  preds: {base}")

    # (a) Log norms at (any block, K=14, K=1)
    print("\n=== (a) Log L2 norms at K=1 and K=14, all blocks ===")
    state["mode"] = "log_norms"
    _ = gen(prompts_corr, fixed_max_len, "log_norms", [1, 14])

    # (b) Zero attn at all 4 blocks × K=14
    print("\n=== (b) ZERO attn output at all 4 blocks × K=14 ===")
    out = gen(prompts_corr, fixed_max_len, "zero", [14])
    n_unchanged = sum(1 for i in range(len(pairs)) if out[i] == base[i])
    print(f"  preds: {out}")
    print(f"  unchanged from baseline: {n_unchanged}/{len(pairs)}")

    # (c) Zero attn at ALL 4 blocks × ALL 32 K
    print("\n=== (c) ZERO attn output at all 4 blocks × ALL K=1..32 ===")
    out = gen(prompts_corr, fixed_max_len, "zero", list(range(1, NUM_STEPS+1)))
    n_unchanged = sum(1 for i in range(len(pairs)) if out[i] == base[i])
    print(f"  preds: {out}")
    print(f"  unchanged from baseline: {n_unchanged}/{len(pairs)}")
    print("  (if still 10/10 unchanged → hook truly isn't taking effect)")
    print("  (if outputs become broken/random → attn IS being patched, just not at K=14)")

    for h in handles: h.remove()


if __name__ == "__main__":
    main()
