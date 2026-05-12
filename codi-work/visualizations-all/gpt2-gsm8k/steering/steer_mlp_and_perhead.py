"""Two new steering experiments at the cell that gave the strongest ablation
effect (step=2, L=0):

  1. PER-HEAD STEERING: project the operator direction onto each head's
     64-dim subspace and add to that head's slice of the c_proj input.
     12 heads x 4 alphas.

  2. MLP-INPUT STEERING: add the operator direction to the MLP's input
     (post-layernorm residual). This is nonlinearly transformed by the MLP,
     so it differs from residual-stream steering at the same layer.
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

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

HEAD_DIM = 64
N_HEADS = 12

CELL_STEP = 2
CELL_LAYER = 0


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
                          map_location="cpu").to(torch.float32).numpy()
    cf_results = json.load(open(REPO / "inference/runs/cf_balanced_student_gpt2/results.json"))
    cf_correct = np.array([r.get("correct", False) for r in cf_results])
    cf_data = json.load(open(REPO.parent / "cf-datasets" / "cf_balanced.json"))
    cf_op = np.array([{"Addition": 0, "Subtraction": 1, "Multiplication": 2,
                        "Common-Division": 3}.get(d["type"], -1) for d in cf_data])[:cf_acts.shape[0]]
    add_mask = cf_correct & (cf_op == 0); sub_mask = cf_correct & (cf_op == 1)
    print(f"  CF correct: add={add_mask.sum()}  sub={sub_mask.sum()}")
    # Op direction at chosen cell
    op_dir = cf_acts[add_mask, CELL_STEP, CELL_LAYER].mean(0) - cf_acts[sub_mask, CELL_STEP, CELL_LAYER].mean(0)
    print(f"  op_dir at (step={CELL_STEP}, L={CELL_LAYER}) || ||={np.linalg.norm(op_dir):.2f}")
    op_dir_unit = op_dir / max(np.linalg.norm(op_dir), 1e-9)

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
    targs = TrainingArguments(output_dir="/tmp/_smp", bf16=True,
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

    # We hook on:
    #  - block CELL_LAYER's c_proj (forward_pre_hook): for per-head, ADD
    #    a scaled head-slice of op_dir to one head's slot at the last token.
    #  - block CELL_LAYER's mlp (forward_pre_hook): add op_dir to the MLP input
    HOOK = {"mode": "off",       # 'head' or 'mlp_in'
            "head": -1, "alpha": 0.0,
            "vec_full": None,    # (768,) for mlp_in
            "vec_head": None,    # (64,)  for per-head
            "active": False, "cur_step": -99}

    def make_attn_pre_hook(block_idx):
        def fn(module, inputs):
            if not HOOK["active"]: return None
            if HOOK["mode"] != "head": return None
            if HOOK["cur_step"] != CELL_STEP: return None
            if block_idx != CELL_LAYER: return None
            x = inputs[0].clone()
            h = HOOK["head"]
            v = HOOK["vec_head"].to(x.device, dtype=x.dtype)
            x[:, -1, h*HEAD_DIM:(h+1)*HEAD_DIM] += HOOK["alpha"] * v
            return (x,) + inputs[1:]
        return fn

    def make_mlp_pre_hook(block_idx):
        def fn(module, inputs):
            if not HOOK["active"]: return None
            if HOOK["mode"] != "mlp_in": return None
            if HOOK["cur_step"] != CELL_STEP: return None
            if block_idx != CELL_LAYER: return None
            x = inputs[0].clone()
            v = HOOK["vec_full"].to(x.device, dtype=x.dtype)
            x[:, -1, :] += HOOK["alpha"] * v
            return (x,) + inputs[1:]
        return fn

    handles = []
    for L, blk in enumerate(transformer.h):
        attn_mod = getattr(blk, "self_attn", None) or getattr(blk, "attn", None)
        handles.append(attn_mod.c_proj.register_forward_pre_hook(make_attn_pre_hook(L)))
        if hasattr(blk, "mlp"):
            handles.append(blk.mlp.register_forward_pre_hook(make_mlp_pre_hook(L)))

    @torch.no_grad()
    def run_batch(qs):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
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

    # SVAMP eval
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

    def gen(idx_list):
        qs = [full[int(i)]["question_concat"].strip().replace("  ", " ") for i in idx_list]
        out = []
        for s in range(0, len(qs), 16):
            out += run_batch(qs[s:s+16])
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
    HOOK.update({"active": False})
    base_a = gen(add_eval); base_s = gen(sub_eval)
    print(f"  ADD baseline: {score(base_a, add_cs)}")
    print(f"  SUB baseline: {score(base_s, sub_cs)}")
    summary = {"cell": [CELL_STEP, CELL_LAYER],
                "add_baseline": score(base_a, add_cs),
                "sub_baseline": score(base_s, sub_cs)}

    # === EXPERIMENT 1: Per-head steering at (step=2, L=0) ===
    print(f"\n=== Per-head steering at (step={CELL_STEP}, L={CELL_LAYER}) ===")
    HOOK.update({"mode": "head"})
    # For each head, project op_dir onto its 64-dim slot
    for h in range(N_HEADS):
        slot = op_dir_unit[h*HEAD_DIM:(h+1)*HEAD_DIM]
        slot_unit = slot / max(np.linalg.norm(slot), 1e-9)
        HOOK["vec_head"] = torch.tensor(slot_unit, dtype=torch.float32)
        HOOK["head"] = h
        for alpha in [50.0, 200.0]:
            HOOK["alpha"] = alpha; HOOK["active"] = True
            sa = gen(add_eval); ss = gen(sub_eval)
            HOOK["alpha"] = -alpha
            ss_neg = gen(sub_eval)
            HOOK["alpha"] = alpha
            sc_a = score(sa, add_cs); sc_s = score(ss_neg, sub_cs)
            print(f"  head H{h:02d} α={alpha}: ADD={sc_a}  SUB(neg α)={sc_s}", flush=True)
            summary[f"head_H{h:02d}_alpha={alpha}|add"] = sc_a
            summary[f"head_H{h:02d}_alpha={alpha}|sub_neg"] = sc_s
    HOOK["active"] = False

    # === EXPERIMENT 2: MLP-input steering at (step=2, L=0) ===
    print(f"\n=== MLP-input steering at (step={CELL_STEP}, L={CELL_LAYER}) ===")
    HOOK.update({"mode": "mlp_in", "vec_full": torch.tensor(op_dir_unit, dtype=torch.float32)})
    for alpha in [50.0, 100.0, 200.0, 400.0, 800.0]:
        HOOK["alpha"] = alpha; HOOK["active"] = True
        sa = gen(add_eval)
        HOOK["alpha"] = -alpha
        ss = gen(sub_eval)
        sc_a = score(sa, add_cs); sc_s = score(ss, sub_cs)
        print(f"  MLP-in α=±{alpha}: ADD={sc_a}  SUB={sc_s}", flush=True)
        summary[f"mlp_in_alpha={alpha}|add"] = sc_a
        summary[f"mlp_in_alpha={alpha}|sub_neg"] = sc_s
    HOOK["active"] = False

    out = PD / "steer_mlp_and_perhead.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
