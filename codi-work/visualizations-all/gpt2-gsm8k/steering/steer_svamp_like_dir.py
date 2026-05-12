"""Steering with the CLEAN op-direction (derived from correct-subset SVAMP
activations) on the FIXED SVAMP pipeline (max_new=256, last-number parser).

For add->sub steering at the (pos, layer) cell where the clean direction
is most distinct from digit/context noise (we use orthogonal-PC cell from
earlier analysis; or the cell with highest |op direction|/|grand mean|).

Test:
  - baseline: how many examples get correct answer
  - steered (add subjects -> sub direction): does the answer flip to the
    actual subtraction result (clean operator flip!)?
"""

from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))


def codi_extract(sentence: str):
    sentence = sentence.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', sentence)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def parse_two_operands(equation):
    nums = re.findall(r"-?\d+\.?\d*", equation)
    if len(nums) < 2: return None
    try: return float(nums[0]), float(nums[1])
    except: return None


def main():
    print("loading svamp_like op directions + activations...")
    npz = np.load(PD / "svamp_like_op_dir.npz")
    means = npz["correct_op_means"]   # (4, P=16, L+1=13, H=768)
    print(f"  op means shape (svamp_like): {means.shape}")

    # Compute add->sub direction
    add_to_sub = means[1] - means[0]   # (P, L+1, H)
    sub_to_add = means[0] - means[1]
    add_to_mul = means[2] - means[0]
    P, Lp1, H = add_to_sub.shape

    # Find the cell with strongest direction relative to grand mean noise
    acts = torch.load(PD / "svamp_fixed_acts.pt", map_location="cpu").to(torch.float32).numpy()
    print(f"  acts shape: {acts.shape}")
    grand_mean = acts.mean(axis=0)               # (P, L+1, H)
    centered_acts = acts - grand_mean[None]
    var_per_cell = (centered_acts ** 2).mean(axis=(0, 3))   # (P, L+1)
    # SNR = ||add->sub||^2 / mean variance per cell
    diff_sq = (add_to_sub ** 2).sum(axis=2)              # (P, L+1)
    snr = diff_sq / np.sqrt(var_per_cell + 1e-9)
    # constrain to early cells (pos 1..7 — early in CoT)
    snr_early = snr.copy(); snr_early[8:] = 0
    p_best, l_best = np.unravel_index(int(np.argmax(snr_early)), snr.shape)
    print(f"  best add->sub cell (pos<=7): pos {p_best} L{l_best}  ||diff||^2={diff_sq[p_best, l_best]:.1f}")
    # Also a couple of fallback cells
    cells = {
        "BEST_SNR": (p_best, l_best),
        "POS_5_L0": (5, 0),
        "POS_1_L7": (1, 7),
    }

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
    targs = TrainingArguments(output_dir="/tmp/_csteer", bf16=True,
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
    eos_id = tok.eos_token_id

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
    def run_batch(qs, *, vec, alpha, layer, p_target, max_new=256):
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
        done = [False] * B
        for step in range(max_new):
            HOOK["step"] = step
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if not done[b]:
                    tokens[b].append(int(next_ids[b].item()))
                    if int(next_ids[b].item()) == eos_id:
                        done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK.update({"active": False, "step": -1})
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    # SVAMP eval set: filter to ADDITION and SUBTRACTION problems with parseable operands
    ds = load_dataset("gsm8k", "main")
    full = concatenate_datasets([ds["train"], ds["test"]])
    add_idx, sub_idx = [], []
    for i, ex in enumerate(full):
        t = ex["Type"].lower()
        if t == "addition":      add_idx.append(i)
        elif t == "subtraction": sub_idx.append(i)
    print(f"  SVAMP: addition={len(add_idx)}  subtraction={len(sub_idx)}")
    # take 100 of each
    np.random.seed(0)
    add_eval = np.random.choice(add_idx, size=min(100, len(add_idx)), replace=False).tolist()
    sub_eval = np.random.choice(sub_idx, size=min(100, len(sub_idx)), replace=False).tolist()

    def gen_eval(idx_list, vec, alpha, layer, p_target):
        qs = [full[int(i)]["question_concat"].strip().replace("  ", " ") for i in idx_list]
        results = []
        for s in range(0, len(qs), 16):
            results += run_batch(qs[s:s+16], vec=vec, alpha=alpha,
                                  layer=layer, p_target=p_target, max_new=256)
        return results

    # Compute candidate answers (a+b, a-b) for each ex
    def candidates(idx_list):
        cs = []
        for i in idx_list:
            ex = full[int(i)]
            ab = parse_two_operands(ex["Equation"])
            if ab is None: cs.append(None); continue
            a, b = ab
            cs.append({"add": round(a+b), "sub": round(a-b),
                       "mul": round(a*b), "div": round(a/b) if b != 0 else None})
        return cs

    add_cs = candidates(add_eval)
    sub_cs = candidates(sub_eval)

    # === BASELINE ===
    print("\n=== Baseline (no steering) ===")
    base_add = gen_eval(add_eval, torch.zeros(H), 0.0, 0, 0)
    base_sub = gen_eval(sub_eval, torch.zeros(H), 0.0, 0, 0)
    def score(strs, idx_list, cs):
        out = {"add_match": 0, "sub_match": 0, "mul_match": 0, "div_match": 0,
               "any_correct": 0, "n_valid": 0}
        for s, i, c in zip(strs, idx_list, cs):
            v = codi_extract(s)
            if v is None: continue
            out["n_valid"] += 1
            if c is None: continue
            if v == c["add"]: out["add_match"] += 1
            if v == c["sub"]: out["sub_match"] += 1
            if c["mul"] is not None and v == c["mul"]: out["mul_match"] += 1
            if c["div"] is not None and v == c["div"]: out["div_match"] += 1
        gold = full[int(idx_list[0])]["Type"].lower()
        return out
    print(f"  ADD problems baseline: {score(base_add, add_eval, add_cs)}")
    print(f"  SUB problems baseline: {score(base_sub, sub_eval, sub_cs)}")

    # ---- Sweep clean steering ----
    summary = {"add_baseline_score": score(base_add, add_eval, add_cs),
               "sub_baseline_score": score(base_sub, sub_eval, sub_cs)}

    for cell_name, (p_t, l_t) in cells.items():
        v_unit = add_to_sub[p_t, l_t]
        v_unit = v_unit / max(np.linalg.norm(v_unit), 1e-9)
        v_unit_torch = torch.tensor(v_unit, dtype=torch.float32)
        for alpha in [50.0, 100.0, 200.0, 400.0]:
            print(f"\n=== Cell {cell_name} pos {p_t} L{l_t}  alpha={alpha} ===")
            # Steer ADD problems with add->sub direction (should flip toward sub answer)
            strs_add = gen_eval(add_eval, v_unit_torch, alpha, l_t, p_t)
            sc_add = score(strs_add, add_eval, add_cs)
            print(f"  ADD problems (add->sub steered): {sc_add}")
            # Steer SUB problems with sub->add direction (negative of add->sub)
            strs_sub = gen_eval(sub_eval, v_unit_torch, -alpha, l_t, p_t)
            sc_sub = score(strs_sub, sub_eval, sub_cs)
            print(f"  SUB problems (sub->add steered): {sc_sub}")
            summary[f"{cell_name}|alpha={alpha}|add"] = sc_add
            summary[f"{cell_name}|alpha={alpha}|sub"] = sc_sub

    out = PD / "steer_svamp_like_dir_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
