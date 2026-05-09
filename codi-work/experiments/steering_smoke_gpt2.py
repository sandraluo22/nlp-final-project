"""Steering smoke test on CODI-GPT2 — analog of huginn-work/huginn/steering_smoke.py.

For each (src_op, tgt_op) pair, inject `alpha * (centroid[tgt] - centroid[src])`
into the residual at one chosen (layer, latent_step) during the prompt forward,
via a forward hook on the GPT-2 transformer block.

Centroids come from codi-work/experiments/gpt2_cf_centroids.pkl (per-class
means of cf_balanced activations).

Usage:
  python codi-work/experiments/steering_smoke_gpt2.py \
      --src Subtraction --tgt Addition --layer 10 --step 4 --n 10
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# Allow importing codi.src.model
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "codi"))

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file


CENTROIDS = REPO / "experiments" / "gpt2_cf_centroids.pkl"


def extract_answer(text: str) -> float:
    """CODI-GPT2 emits 'The answer is: <num>' style. Take the last number."""
    s = text.replace(",", "")
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
            "tgt_add": a + b,
            "tgt_mul": a * b,
        })
        if len(items) >= n:
            break
    return items


def load_codi_gpt2(ckpt_dir: str, base_model: str = "gpt2"):
    """Force-load CODI-GPT2 the same way run_eval_with_hooks.py does."""
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore

    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=128, lora_alpha=32, lora_dropout=0.1,
        target_modules=target_modules, init_lora_weights=True,
    )
    model_args = ModelArguments(
        model_name_or_path=base_model, full_precision=True, train=False,
        lora_init=True, ckpt_dir=str(Path(ckpt_dir).expanduser()),
    )
    training_args = TrainingArguments(
        output_dir="/tmp/codi-gpt2-steering", bf16=True,
        use_lora=True, use_prj=True, prj_dim=768, prj_no_ln=False, prj_dropout=0.0,
        num_latent=6, inf_latent_iterations=6, remove_eos=True, greedy=True,
        model_max_length=512, seed=11,
    )
    model = CODI(model_args, training_args, lora_config)
    sd_safe = Path(ckpt_dir).expanduser() / "model.safetensors"
    sd_bin = Path(ckpt_dir).expanduser() / "pytorch_model.bin"
    if sd_safe.exists():
        sd = load_file(str(sd_safe))
    else:
        sd = torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained(
        base_model, model_max_length=512, padding_side="left", use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    return model, tok, training_args


def find_layers(model):
    """Find the per-block ModuleList in CODI-GPT2."""
    # CODI-GPT2 path: model.codi.base_model.model.transformer.h
    for path in [
        ("codi", "base_model", "model", "transformer", "h"),
        ("codi", "transformer", "h"),
    ]:
        cur = model
        try:
            for p in path:
                cur = getattr(cur, p)
            if isinstance(cur, torch.nn.ModuleList) and len(cur) > 4:
                print(f"[smoke] found layers at model.{'.'.join(path)} ({len(cur)} blocks)", flush=True)
                return cur
        except AttributeError:
            continue
    raise RuntimeError("could not find transformer block ModuleList")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="Subtraction")
    p.add_argument("--tgt", default="Addition")
    p.add_argument("--layer", type=int, default=10,
                   help="hidden_states layer index (0=embed, 1..12=block outputs)")
    p.add_argument("--step", type=int, default=4,
                   help="latent step (0..5) at which to inject")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 5.0, 20.0, 50.0])
    p.add_argument("--ckpt_dir", default=os.path.expanduser("~/codi_ckpt/CODI-gpt2"))
    args = p.parse_args()

    print(f"loading centroids: {CENTROIDS}", flush=True)
    cache = pickle.load(open(CENTROIDS, "rb"))
    classes = list(cache["classes"])
    means = cache["means"]   # (n_classes, S=6, L=13, H=768)
    src_i, tgt_i = classes.index(args.src), classes.index(args.tgt)
    # Hook fires on the OUTPUT of block (layer-1) (0-indexed). The activation
    # tensor's "layer" axis is 0=embed, 1..L=block outputs. So hooking on
    # transformer.h[layer-1] gives layer-output residual, matching the L=layer
    # slice. But we want to ADD to the residual that block produces, which IS
    # transformer.h[layer-1].forward output.
    block_idx = args.layer - 1
    if block_idx < 0:
        raise ValueError("--layer must be 1..12 (0=embedding, can't hook embedding)")
    vec_np = means[tgt_i, args.step, args.layer] - means[src_i, args.step, args.layer]
    print(f"steering {args.src}→{args.tgt} at layer={args.layer} step={args.step}+1: "
          f"‖v‖={np.linalg.norm(vec_np):.2f}  H={vec_np.shape[0]}", flush=True)

    print(f"loading CODI-GPT2 from {args.ckpt_dir}", flush=True)
    t0 = time.time()
    model, tok, training_args = load_codi_gpt2(args.ckpt_dir)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Activation extractor target
    layers = find_layers(model)
    target_block = layers[block_idx]
    v_t = torch.tensor(vec_np, dtype=torch.bfloat16, device="cuda")
    state = {"latent_step": -1, "alpha": 0.0, "fired": False}

    def hook(_mod, _inp, out):
        # GPT-2 transformer.h[i] returns a tuple (hidden_states, presents, ...).
        h = out[0] if isinstance(out, tuple) else out
        # Inject only when the latent loop is at the target step.
        if state["latent_step"] == args.step and state["alpha"] != 0.0:
            h[:, -1, :] = h[:, -1, :] + state["alpha"] * v_t
            state["fired"] = True
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h
        return out

    handle = target_block.register_forward_hook(hook)

    items = load_svamp_subset(args.src, args.n)
    print(f"  {len(items)} {args.src} examples loaded", flush=True)

    # CODI-GPT2 inference loop (mirrors run_eval_with_hooks.run_student) —
    # we run it manually so we can control state["latent_step"] per iteration.
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id

    results_all = []
    for alpha in args.alphas:
        print(f"\n=== alpha = {alpha:>5.1f} ===", flush=True)
        per_q = []
        for i, it in enumerate(items):
            state["latent_step"] = -1
            state["alpha"] = alpha
            state["fired"] = False
            inp = tok(it["q"], return_tensors="pt").to("cuda")
            B = 1
            bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
            input_ids = torch.cat([inp["input_ids"], bot], dim=1)
            attn_mask = torch.cat([inp["attention_mask"], torch.ones_like(bot)], dim=1)
            with torch.no_grad():
                out = model.codi(input_ids=input_ids, attention_mask=attn_mask,
                                 use_cache=True, output_hidden_states=True)
                past_kv = out.past_key_values
                latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
                if training_args.use_prj:
                    latent = model.prj(latent)
                # 6 latent iterations
                for s in range(training_args.inf_latent_iterations):
                    state["latent_step"] = s
                    attn_mask = torch.cat(
                        [attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device="cuda")], dim=1)
                    out = model.codi(inputs_embeds=latent, attention_mask=attn_mask,
                                     use_cache=True, output_hidden_states=True,
                                     past_key_values=past_kv)
                    past_kv = out.past_key_values
                    latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
                    if training_args.use_prj:
                        latent = model.prj(latent)
                state["latent_step"] = -1  # disable hook during decode
                # Decode answer
                eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
                output = eot_emb.unsqueeze(0).expand(B, -1, -1)
                pred_tokens = []
                for _ in range(args.max_new_tokens):
                    attn_mask = torch.cat(
                        [attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device="cuda")], dim=1)
                    step_out = model.codi(inputs_embeds=output, attention_mask=attn_mask,
                                          use_cache=True, output_hidden_states=False,
                                          output_attentions=False, past_key_values=past_kv)
                    past_kv = step_out.past_key_values
                    logits = step_out.logits[:, -1, :model.codi.config.vocab_size - 1]
                    next_id = int(torch.argmax(logits, dim=-1).item())
                    pred_tokens.append(next_id)
                    if next_id == eos_id:
                        break
                    output = embed_fn(torch.tensor([[next_id]], device="cuda"))
            text = tok.decode(pred_tokens, skip_special_tokens=True)
            pred = extract_answer(text)
            per_q.append({
                "idx": i,
                "src_correct": it["src_answer"],
                "tgt_add": it["tgt_add"],
                "tgt_mul": it["tgt_mul"],
                "pred": pred,
                "gen_snippet": text[:80],
                "fired": state["fired"],
            })
            tag_s = "S" if pred == it["src_answer"] else "."
            tag_t = "T" if pred == it["tgt_add"] else "."
            tag_m = "M" if pred == it["tgt_mul"] else "."
            print(f"  ex{i}: pred={pred!s:>10}  src={it['src_answer']:.0f} "
                  f"tgt_add={it['tgt_add']:.0f} tgt_mul={it['tgt_mul']:.0f}  "
                  f"[{tag_s}{tag_t}{tag_m}]  {text[:60]!r}", flush=True)
        n_src = sum(1 for r in per_q if r["pred"] == r["src_correct"])
        n_tgt = sum(1 for r in per_q if r["pred"] == r["tgt_add"])
        n_mul = sum(1 for r in per_q if r["pred"] == r["tgt_mul"])
        n_other = len(per_q) - n_src - n_tgt - n_mul
        print(f"  alpha={alpha:>5.1f}  =src: {n_src}/{len(per_q)}   "
              f"=tgt(add): {n_tgt}/{len(per_q)}   =mul: {n_mul}/{len(per_q)}   "
              f"other: {n_other}", flush=True)
        results_all.append({"alpha": alpha, "n_src": n_src, "n_tgt_add": n_tgt,
                            "n_tgt_mul": n_mul, "n_other": n_other,
                            "per_q": per_q})

    handle.remove()
    out_path = REPO / "experiments" / "steering_smoke_gpt2.json"
    out_path.write_text(json.dumps(results_all, indent=2))
    print(f"\nsaved -> {out_path}")
    print("\n=== summary ===")
    print(f" alpha | =src | =tgt(add) | =mul | other")
    for r in results_all:
        print(f"  {r['alpha']:>5.1f} |  {r['n_src']:>2d}  |    {r['n_tgt_add']:>2d}     |  {r['n_tgt_mul']:>2d}  |  {r['n_other']:>2d}")


if __name__ == "__main__":
    main()
