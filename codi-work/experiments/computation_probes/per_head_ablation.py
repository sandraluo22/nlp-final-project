"""Per-head zero-ablation on CODI-GPT-2 attention. For each (layer, head)
cell, zero that head's contribution at the prompt-end position (last token of
the prompt forward), then run inference and score per-operator correctness on
real SVAMP add and sub problems.

If operator info is encoded in attention, we should see specific heads whose
ablation crashes operator-conditional accuracy.

GPT-2 architecture: 12 layers × 12 heads × 64 head_dim = 768 hidden.
The c_proj weight is (768, 768); its INPUT is concat(head_0 .. head_11).
We hook on c_proj as a forward_pre_hook and zero the slice for head h.
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

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64


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
    targs = TrainingArguments(output_dir="/tmp/_ph", bf16=True,
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

    HOOK = {"phase": "off", "layer": -1, "head": -1,
            "active": False}
    # Hooks on attn.c_proj (forward_pre_hook): zero out the slice for HOOK['head']
    # at the LAST token only, when we're in 'prompt' phase.

    def make_pre_hook(layer_idx):
        def fn(module, inputs):
            if not HOOK["active"]: return None
            if HOOK["phase"] != "prompt_end": return None
            if HOOK["layer"] != layer_idx: return None
            x = inputs[0]   # (B, T, 768)
            x = x.clone()
            h = HOOK["head"]
            x[:, -1, h*HEAD_DIM:(h+1)*HEAD_DIM] = 0
            return (x,) + inputs[1:]
        return fn

    handles = []
    for L, blk in enumerate(transformer.h):
        attn_mod = getattr(blk, "self_attn", None) or getattr(blk, "attn", None)
        c_proj = attn_mod.c_proj
        handles.append(c_proj.register_forward_pre_hook(make_pre_hook(L)))

    @torch.no_grad()
    def run_batch(qs, *, layer, head, max_new=256):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)

        # Activate hook ONLY for the prompt forward
        HOOK.update({"phase": "prompt_end", "layer": layer, "head": head, "active": layer >= 0})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        # disable for latent loop & decode (the KV cache already has the modified prompt-end)
        HOOK["active"] = False
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
        for _ in range(max_new):
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

    # SVAMP eval: 100 add + 100 sub
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    add_idx, sub_idx = [], []
    for i, ex in enumerate(full):
        t = ex["Type"].lower()
        if t == "addition":      add_idx.append(i)
        elif t == "subtraction": sub_idx.append(i)
    np.random.seed(0)
    add_eval = np.random.choice(add_idx, size=100, replace=False).tolist()
    sub_eval = np.random.choice(sub_idx, size=100, replace=False).tolist()

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
    add_cs = candidates(add_eval); sub_cs = candidates(sub_eval)

    def gen_eval(idx_list, *, layer, head):
        qs = [full[int(i)]["question_concat"].strip().replace("  ", " ") for i in idx_list]
        out = []
        for s in range(0, len(qs), 16):
            out += run_batch(qs[s:s+16], layer=layer, head=head, max_new=256)
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

    # Baseline
    print("\n=== Baseline (no ablation) ===")
    base_a = gen_eval(add_eval, layer=-1, head=-1)
    base_s = gen_eval(sub_eval, layer=-1, head=-1)
    sc_a_b = score(base_a, add_cs); sc_s_b = score(base_s, sub_cs)
    print(f"  ADD baseline: {sc_a_b}")
    print(f"  SUB baseline: {sc_s_b}")

    grid_add_correct = np.zeros((N_LAYERS, N_HEADS), dtype=int)
    grid_sub_correct = np.zeros((N_LAYERS, N_HEADS), dtype=int)
    grid_add_to_sub  = np.zeros((N_LAYERS, N_HEADS), dtype=int)
    grid_sub_to_add  = np.zeros((N_LAYERS, N_HEADS), dtype=int)

    summary = {"add_baseline": sc_a_b, "sub_baseline": sc_s_b}

    t0 = time.time()
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            sa = gen_eval(add_eval, layer=L, head=H)
            ss = gen_eval(sub_eval, layer=L, head=H)
            sc_a = score(sa, add_cs); sc_s = score(ss, sub_cs)
            grid_add_correct[L, H] = sc_a["add"]
            grid_sub_correct[L, H] = sc_s["sub"]
            grid_add_to_sub[L, H]  = sc_a["sub"]
            grid_sub_to_add[L, H]  = sc_s["add"]
            print(f"  L{L:2d} H{H:2d}: add_correct={sc_a['add']}/100  sub_correct={sc_s['sub']}/100  "
                  f"add->sub={sc_a['sub']}  sub->add={sc_s['add']}  ({time.time()-t0:.0f}s)", flush=True)

    out = PD / "per_head_ablation.json"
    out.write_text(json.dumps({
        "add_baseline": sc_a_b, "sub_baseline": sc_s_b,
        "grid_add_correct": grid_add_correct.tolist(),
        "grid_sub_correct": grid_sub_correct.tolist(),
        "grid_add_to_sub":  grid_add_to_sub.tolist(),
        "grid_sub_to_add":  grid_sub_to_add.tolist(),
    }, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
