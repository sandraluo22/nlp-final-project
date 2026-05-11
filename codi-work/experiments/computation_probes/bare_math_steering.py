"""Use the bare-math operator-direction vectors as steering on SVAMP. Tests
whether the clean (no-prose) operator direction is CAUSAL on SVAMP outputs.

We pick three (pos, layer) cells and try the add->sub direction at each:
  CELL_HIGH:  highest cos_sim with SVAMP add->sub direction (aligned cell)
  CELL_LOW :  lowest cos_sim (volatile / context-flipped cell)
  CELL_REF :  pos 1 layer 7 — the cell that responded best in our previous
              SVAMP-derived steering at orthogonal directions

For each cell, sweep alpha (steering scale, with the direction normalized to
unit length) and measure:
  - n_changed_int: outputs different from baseline
  - bucket distribution shift
  - median |int| shift
  - count of outputs whose magnitude DECREASED (sub-shaped) vs INCREASED
"""

from __future__ import annotations
import json, os, re, sys
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))


def parse_int(s):
    m = re.search(r"answer is\s*:\s*(-?\d+)", s)
    return int(m.group(1)) if m else None


def main():
    print("loading bare-math op direction + SVAMP cos-sim...")
    bare = np.load(PD / "bare_math_op_dirs.npz")
    bare_diff = bare["addition->subtraction"]   # (P_b, L+1, H)
    print(f"  bare add->sub shape: {bare_diff.shape}")
    cos = np.load(PD / "bare_vs_svamp_op_dir.npz")
    cs = cos["add->sub"]                        # (P, L+1)
    print(f"  cos sim grid shape: {cs.shape}")
    # The cs grid is min(P_b, P_s)=12 by min(Lp1)=13
    P, Lp1 = cs.shape
    H = bare_diff.shape[2]

    # Pick cells (P_b is 12 — bare prompts have P_DECODE=12 vs SVAMP's 16)
    high_idx = np.unravel_index(int(np.argmax(cs[:P, :Lp1])), (P, Lp1))
    low_idx  = np.unravel_index(int(np.argmin(cs[:P, :Lp1])), (P, Lp1))
    ref_idx  = (1, 7)
    print(f"  CELL_HIGH (highest cos): pos {high_idx[0]} L{high_idx[1]} cos={cs[high_idx]:+.3f}")
    print(f"  CELL_LOW  (lowest cos):  pos {low_idx[0]} L{low_idx[1]} cos={cs[low_idx]:+.3f}")
    print(f"  CELL_REF  (pos1 L7):     cos={cs[ref_idx]:+.3f}")

    # build unit-vec steering directions per cell
    def unit_vec(p, l):
        v = bare_diff[p, l]
        n = np.linalg.norm(v)
        return torch.tensor(v / max(n, 1e-9), dtype=torch.float32), float(n)

    # ---- Load CODI ----
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"\nloading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_bsteer", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=768,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=6,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)

    HOOK = {"step": -1, "active": False, "vec": None, "p_target": None,
            "layer": None, "alpha": 0.0}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"] or HOOK["layer"] is None: return output
            if block_idx != HOOK["layer"] - 1: return output
            if HOOK["step"] != HOOK["p_target"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, vec, alpha, layer, p_target):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"vec": vec, "p_target": p_target, "layer": layer,
                     "alpha": alpha, "active": True, "step": -1})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for _ in range(targs.inf_latent_iterations):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        for step in range(12):
            HOOK["step"] = step
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B): tokens[b].append(int(next_ids[b].item()))
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK.update({"active": False, "step": -1})
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    # SVAMP eval
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    answers = [int(round(float(str(ex["Answer"]).replace(",", "")))) for ex in full]
    N = len(questions)
    np.random.seed(0)
    eval_idx = np.random.choice(N, size=200, replace=False)
    eval_qs = [questions[i] for i in eval_idx]
    BS = 16

    def run_full(vec, alpha, layer, p_target):
        out = []
        for s in range(0, len(eval_qs), BS):
            out += run_batch(eval_qs[s:s+BS], vec=vec, alpha=alpha,
                              layer=layer, p_target=p_target)
        return out

    print("\n=== Baseline ===", flush=True)
    base_strs = run_full(torch.zeros(H), 0.0, 1, 0)
    base_ints = [parse_int(s) for s in base_strs]
    base_abs = [abs(v) for v in base_ints if v is not None]
    base_med = int(np.median(base_abs)) if base_abs else 0
    print(f"  baseline median |int|: {base_med}", flush=True)

    summary = {"baseline_median_abs": base_med, "baseline_n_valid": int(sum(v is not None for v in base_ints))}

    # Sweep cells & alphas
    cells = {
        "HIGH": (high_idx[0], high_idx[1], cs[high_idx]),
        "LOW":  (low_idx[0], low_idx[1], cs[low_idx]),
        "REF":  (ref_idx[0], ref_idx[1], cs[ref_idx]),
    }
    alphas = [50.0, 100.0, 200.0, 400.0, 800.0]   # bare unit vec; large alphas needed

    for cell_name, (p, l, cossim) in cells.items():
        v_bare, n_bare = unit_vec(p, l)
        print(f"\n=== Cell {cell_name}: pos {p} L{l}  cos_sim={cossim:+.3f}  bare_vec_norm={n_bare:.2f} ===", flush=True)
        for a in alphas:
            strs = run_full(v_bare, a, l, p)
            ints = [parse_int(s) for s in strs]
            abs_vals = [abs(v) for v in ints if v is not None]
            n_changed = sum(1 for x, y in zip(ints, base_ints) if x != y)
            med = int(np.median(abs_vals)) if abs_vals else 0
            n_smaller = sum(1 for x, y in zip(ints, base_ints)
                             if x is not None and y is not None and abs(x) < abs(y))
            n_larger = sum(1 for x, y in zip(ints, base_ints)
                            if x is not None and y is not None and abs(x) > abs(y))
            print(f"  alpha={a:>6.1f}: changed={n_changed}/200  median|int|={med}  "
                  f"smaller={n_smaller} larger={n_larger}", flush=True)
            summary[f"{cell_name}|alpha={a}"] = {
                "n_changed": n_changed, "median_abs": med,
                "n_smaller": n_smaller, "n_larger": n_larger,
                "preds": strs[:30],
            }

    out = PD / "bare_math_steering.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
