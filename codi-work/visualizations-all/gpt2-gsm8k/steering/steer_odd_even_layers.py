"""Steering at two cell groups:
  ODD STEPS x LAYER 12: (step=1, L=12), (step=3, L=12), (step=5, L=12)
  EVEN STEPS x LAYER 0: (step=0, L=0), (step=2, L=0), (step=4, L=0)

For each cell: steer ADD problems with add->sub direction, SUB problems
with sub->add (negated alpha). Score per-operator match counts on real SVAMP.

Operator direction = mean(cf_balanced correct add) - mean(cf_balanced correct sub)
at each (latent_step, layer) cell.
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


def codi_extract(s: str):
    s = s.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def parse_two_operands(equation):
    nums = re.findall(r"-?\d+\.?\d*", equation)
    if len(nums) < 2: return None
    try: return float(nums[0]), float(nums[1])
    except: return None


def main():
    print("loading cf_balanced activations + correctness...")
    cf_acts = torch.load(REPO / "inference/runs/cf_balanced_student_gpt2/activations.pt",
                          map_location="cpu").to(torch.float32).numpy()  # (676, 6, 13, 768)
    cf_results = json.load(open(REPO / "inference/runs/cf_balanced_student_gpt2/results.json"))
    cf_correct = np.array([r.get("correct", False) for r in cf_results])
    cf_data = json.load(open(REPO.parent / "cf-datasets" / "cf_balanced.json"))
    cf_op = np.array([{"Addition": 0, "Subtraction": 1, "Multiplication": 2,
                        "Common-Division": 3}.get(d["type"], -1) for d in cf_data])[:cf_acts.shape[0]]

    add_mask = cf_correct & (cf_op == 0)
    sub_mask = cf_correct & (cf_op == 1)
    print(f"  CF correct: add={add_mask.sum()}  sub={sub_mask.sum()}")

    N, S, Lp1, H = cf_acts.shape
    op_dir = np.zeros((S, Lp1, H), dtype=np.float32)
    for s in range(S):
        for l in range(Lp1):
            op_dir[s, l] = cf_acts[add_mask, s, l].mean(0) - cf_acts[sub_mask, s, l].mean(0)

    # Cell groups
    odd_step_L12 = [(1, 12), (3, 12), (5, 12)]
    even_step_L0 = [(0, 0), (2, 0), (4, 0)]

    # ---- Load CODI ----
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_oe", bf16=True,
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

    HOOK = {"active": False, "vec": None, "alpha": 0.0,
            "tgt_step": -1, "tgt_layer": -1, "cur_step": -99}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if block_idx != HOOK["tgt_layer"] - 1: return output
            if HOOK["cur_step"] != HOOK["tgt_step"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn
    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, vec, alpha, layer, latent_step):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"vec": vec, "alpha": alpha, "tgt_step": latent_step,
                     "tgt_layer": layer, "cur_step": -99, "active": True})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for s in range(targs.inf_latent_iterations):
            HOOK["cur_step"] = s
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        HOOK["active"] = False
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(256):
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
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    ds = load_dataset("gsm8k", "main")
    full = concatenate_datasets([ds["train"], ds["test"]])
    add_idx = [i for i, ex in enumerate(full) if ex["Type"] == "Addition"]
    sub_idx = [i for i, ex in enumerate(full) if ex["Type"] == "Subtraction"]
    np.random.seed(0)
    add_eval = np.random.choice(add_idx, size=100, replace=False).tolist()
    sub_eval = np.random.choice(sub_idx, size=100, replace=False).tolist()

    def candidates(idx_list):
        cs = []
        for i in idx_list:
            ab = parse_two_operands(full[int(i)]["Equation"])
            if ab is None: cs.append(None); continue
            a, b = ab
            cs.append({"add": round(a+b), "sub": round(a-b),
                        "mul": round(a*b), "div": round(a/b) if b else None})
        return cs
    add_cs = candidates(add_eval); sub_cs = candidates(sub_eval)

    def gen(idx_list, vec, alpha, layer, latent_step):
        qs = [full[int(i)]["question_concat"].strip().replace("  ", " ") for i in idx_list]
        out = []
        for s in range(0, len(qs), 16):
            out += run_batch(qs[s:s+16], vec=vec, alpha=alpha, layer=layer, latent_step=latent_step)
        return out

    def score(strs, cs):
        o = {"add": 0, "sub": 0, "mul": 0, "div": 0, "n_valid": 0}
        for s, c in zip(strs, cs):
            v = codi_extract(s)
            if v is None: continue
            o["n_valid"] += 1
            if c is None: continue
            if v == c["add"]: o["add"] += 1
            if v == c["sub"]: o["sub"] += 1
            if c["mul"] is not None and v == c["mul"]: o["mul"] += 1
            if c["div"] is not None and v == c["div"]: o["div"] += 1
        return o

    print("\n=== Baseline ===")
    base_a = gen(add_eval, torch.zeros(H), 0.0, 0, 0)
    base_s = gen(sub_eval, torch.zeros(H), 0.0, 0, 0)
    print(f"  ADD baseline: {score(base_a, add_cs)}")
    print(f"  SUB baseline: {score(base_s, sub_cs)}")
    summary = {"add_baseline": score(base_a, add_cs),
                "sub_baseline": score(base_s, sub_cs)}

    for label, cells in [("ODD_STEP_L12", odd_step_L12), ("EVEN_STEP_L0", even_step_L0)]:
        for (cs_step, cs_layer) in cells:
            v_op = op_dir[cs_step, cs_layer]
            n = max(np.linalg.norm(v_op), 1e-9)
            v_op = v_op / n
            v_op_t = torch.tensor(v_op, dtype=torch.float32)
            for alpha in [50.0, 100.0, 200.0, 400.0]:
                tag = f"{label}_step{cs_step}_L{cs_layer}_a{alpha}"
                print(f"\n=== {tag} ===")
                sa = gen(add_eval, v_op_t, alpha, cs_layer, cs_step)
                ss = gen(sub_eval, v_op_t, -alpha, cs_layer, cs_step)
                sc_a = score(sa, add_cs); sc_s = score(ss, sub_cs)
                print(f"  ADD steered (add->sub): {sc_a}")
                print(f"  SUB steered (sub->add): {sc_s}")
                summary[f"{tag}|add"] = sc_a
                summary[f"{tag}|sub"] = sc_s

    out = PD / "steer_odd_even_layers.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
