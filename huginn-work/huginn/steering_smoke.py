"""Minimal smoke test: does additive steering at the peak operator-LDA point
flip Huginn's predicted operator on SVAMP?

Plan:
  - Load Huginn.
  - Pick 10 SVAMP Subtraction problems where K=32 baseline gives a sensible
    numeric answer (we'll grade liberally — just want to detect a *change*).
  - At intervention point (core_block=3, recurrence_step=K_INTERVENE) inject
    `alpha * (lda_means[Addition] - lda_means[Subtraction])` into the residual
    via a forward hook, for the FULL prompt forward (last token at every
    application of core_block[3] during recurrence step K_INTERVENE).
  - Compare baseline vs. intervened answers across α ∈ {0, 1, 5, 20, 50}.

Outputs to stdout: per-α flip-rate-vs-baseline, per-α "answer matches add(a, b)"
when we can extract two numerals from the prompt.

Usage on GPU:
  python huginn-work/huginn/steering_smoke.py \
      --src Subtraction --tgt Addition --K 14 --block 3 --n 10
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset


HUGINN_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HUGINN_ROOT.parent
LDA_CACHE = HUGINN_ROOT / "visualizations" / "cf_lda_80_20_slideshow_cache.pkl"

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


def load_svamp_subset(src: str, n: int):
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    items = []
    for ex in full:
        ptype = ex["Type"].replace("Common-Divison", "Common-Division")
        if ptype != src:
            continue
        nums = re.findall(r"\d+\.?\d*", ex["Equation"])
        if len(nums) != 2:
            continue
        a, b = float(nums[0]), float(nums[1])
        items.append({
            "q": ex["question_concat"].strip().replace("  ", " "),
            "src_answer": float(str(ex["Answer"]).replace(",", "")),
            "a": a, "b": b,
            "tgt_answer_add": a + b,
            "tgt_answer_mul": a * b,
        })
        if len(items) >= n:
            break
    return items


def steering_vector(lda_cache: dict, src: str, tgt: str, k: int, layer: int, step: int) -> np.ndarray:
    """Return lda_means[tgt] - lda_means[src] at (layer, step) for the slideshow
    cache entry of dim=k. Identical across k since lda_means is from the
    underlying full LDA fit before projection."""
    res_k = lda_cache[k]
    classes = list(res_k["lda_classes"])
    src_i = classes.index(src)
    tgt_i = classes.index(tgt)
    vec = res_k["lda_means"][layer, step, tgt_i] - res_k["lda_means"][layer, step, src_i]
    return vec.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="Subtraction")
    p.add_argument("--tgt", default="Addition")
    p.add_argument("--K", type=int, default=14, help="recurrence step (1-indexed) at which to inject")
    p.add_argument("--block", type=int, default=3, help="core block index (0..3) to hook")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 5.0, 20.0, 50.0])
    p.add_argument("--lda_dim", type=int, default=3, help="which dim slice of slideshow cache to use (1/2/3)")
    args = p.parse_args()

    print(f"loading LDA cache: {LDA_CACHE}", flush=True)
    with open(LDA_CACHE, "rb") as f:
        lda_cache = pickle.load(f)
    vec = steering_vector(lda_cache, args.src, args.tgt, args.lda_dim, args.block, args.K - 1)
    print(f"steering vector ‖v‖={np.linalg.norm(vec):.2f}  H={vec.shape[0]}  "
          f"({args.src}→{args.tgt} at block={args.block} K={args.K})", flush=True)

    print(f"loading Huginn", flush=True)
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(
        "tomg-group-umd/huginn-0125", trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cuda",
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Steering hook on core_block[args.block]. We track the call count so we
    # only inject during recurrence step args.K.
    target_block = model.transformer.core_block[args.block]
    v_t = torch.tensor(vec, dtype=torch.bfloat16, device="cuda")
    state = {"call": 0, "alpha": 0.0, "fired": False}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        state["call"] += 1
        if state["call"] == args.K and state["alpha"] != 0.0:
            # Add steering vec to LAST token only (matches activation extractor).
            h[:, -1, :] = h[:, -1, :] + state["alpha"] * v_t
            state["fired"] = True
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return out

    handle = target_block.register_forward_hook(hook)

    items = load_svamp_subset(args.src, args.n)
    print(f"  {len(items)} {args.src} examples loaded", flush=True)

    # Per-alpha results
    results = []
    for alpha in args.alphas:
        print(f"\n=== alpha = {alpha:>5.1f} ===", flush=True)
        per_q = []
        for i, it in enumerate(items):
            state["call"] = 0
            state["alpha"] = alpha
            state["fired"] = False
            prompt = PROMPT_TEMPLATE.format(q=it["q"])
            inp = tok(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **inp, max_new_tokens=args.max_new_tokens,
                    num_steps=args.num_steps, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            text = tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(text)
            per_q.append({
                "idx": i,
                "src_correct": it["src_answer"],
                "tgt_add": it["tgt_answer_add"],
                "tgt_mul": it["tgt_answer_mul"],
                "pred": pred,
                "gen_snippet": text[:80],
                "fired": state["fired"],
            })
            tag_src = "S" if pred == it["src_answer"] else "."
            tag_tgt = "T" if pred == it["tgt_answer_add"] else "."
            print(f"  ex{i}: pred={pred!s:>10}  src={it['src_answer']:.0f} tgt_add={it['tgt_answer_add']:.0f}  "
                  f"[{tag_src}{tag_tgt}]  {text[:60]!r}",
                  flush=True)
        n_src = sum(1 for r in per_q if r["pred"] == r["src_correct"])
        n_tgt = sum(1 for r in per_q if r["pred"] == r["tgt_add"])
        n_other = len(per_q) - n_src - n_tgt
        print(f"  alpha={alpha:>5.1f}  =src: {n_src}/{len(per_q)}   =tgt(add): {n_tgt}/{len(per_q)}   other: {n_other}",
              flush=True)
        results.append({"alpha": alpha, "n_src": n_src, "n_tgt": n_tgt, "n_other": n_other,
                        "per_q": per_q})

    handle.remove()
    out_path = HUGINN_ROOT / "visualizations" / "probes" / "steering_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")
    print("\n=== summary ===")
    print(f" alpha | =src(stay) | =tgt(flip) | other")
    for r in results:
        print(f"  {r['alpha']:>5.1f} |   {r['n_src']:>4d}    |   {r['n_tgt']:>4d}    | {r['n_other']:>4d}")


if __name__ == "__main__":
    main()
