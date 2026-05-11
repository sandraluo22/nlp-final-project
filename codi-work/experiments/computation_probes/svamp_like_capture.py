"""Capture activations on the svamp_like prose-template prompts where CODI
gets 84% accuracy. Use the FIXED pipeline (max_new=256, EOS stop)."""

from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
from itertools import product

import numpy as np
import torch
import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

P_CAPTURE = 16
MAX_NEW = 256

TEMPLATES = [
    ("addition",       "Sandy has {a} apples. She gets {b} more. How many apples does Sandy have?",
     lambda a,b: a + b),
    ("subtraction",    "Sandy has {a} apples. She gives away {b}. How many apples does Sandy have left?",
     lambda a,b: a - b),
    ("multiplication", "Sandy has {a} bags. Each bag has {b} apples. How many apples does Sandy have?",
     lambda a,b: a * b),
    ("division",       "Sandy has {a} apples. She splits them into groups of {b}. How many groups?",
     lambda a,b: a // b if b != 0 and a % b == 0 else None),
]
OP_NAME_TO_IDX = {"addition": 0, "subtraction": 1, "multiplication": 2, "division": 3}


def codi_extract(s: str):
    s = s.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
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
    targs = TrainingArguments(output_dir="/tmp/_svl", bf16=True,
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
    eos_id = tok.eos_token_id

    # Build prompts: for each op, all (a, b) in [1..20]
    prompts, meta = [], []
    for op_name, tmpl, fn in TEMPLATES:
        for a, b in product(range(1, 21), range(1, 21)):
            gold = fn(a, b)
            if gold is None: continue
            prompts.append(tmpl.format(a=a, b=b))
            meta.append({"a": a, "b": b, "op": op_name, "op_idx": OP_NAME_TO_IDX[op_name],
                          "gold": int(gold)})
    N = len(prompts)
    print(f"  N={N} svamp-like prompts (per-op: add=400 sub=400 mul=400 div=66 = 1266)", flush=True)

    bs = 16
    multi_acts = []
    pred_strings = []
    t0 = time.time()
    for i in range(0, N, bs):
        batch_q = prompts[i:i+bs]
        B = len(batch_q)
        batch = tok(batch_q, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        with torch.no_grad():
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
            pos_acts = []
            for p in range(MAX_NEW):
                step_out = model.codi(inputs_embeds=output, attention_mask=attn,
                                      use_cache=True,
                                      output_hidden_states=(p < P_CAPTURE),
                                      past_key_values=past)
                past = step_out.past_key_values
                if p < P_CAPTURE:
                    pos_acts.append(torch.stack([h[:, -1, :] for h in step_out.hidden_states], dim=1))
                logits = step_out.logits[:, -1, :model.codi.config.vocab_size - 1]
                next_ids = torch.argmax(logits, dim=-1)
                for b in range(B):
                    if not done[b]:
                        tokens[b].append(int(next_ids[b].item()))
                        if int(next_ids[b].item()) == eos_id:
                            done[b] = True
                if all(done): break
                attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
                output = embed_fn(next_ids).unsqueeze(1)
            while len(pos_acts) < P_CAPTURE:
                pos_acts.append(pos_acts[-1].clone())
            pos_acts_t = torch.stack(pos_acts[:P_CAPTURE], dim=1).to(torch.float32).cpu()
            multi_acts.append(pos_acts_t)
            for b in range(B):
                pred_strings.append(tok.decode(tokens[b], skip_special_tokens=True))
        done_n = i + B
        if done_n % 64 == 0 or done_n == N:
            print(f"  {done_n}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    multi_t = torch.cat(multi_acts, dim=0)[:N]
    print(f"\nshape: {tuple(multi_t.shape)}", flush=True)
    out_pt = PD / "svamp_like_acts.pt"
    torch.save(multi_t.to(torch.bfloat16), out_pt)
    print(f"saved {out_pt}  ({out_pt.stat().st_size/1e6:.1f} MB)")

    # parse + score
    correct = []
    pred_floats = []
    for m, s in zip(meta, pred_strings):
        v = codi_extract(s)
        pred_floats.append(None if v is None else float(v))
        correct.append(v is not None and abs(v - m["gold"]) < 1e-3)
    correct = np.array(correct)
    print(f"  total accuracy: {correct.sum()}/{N} = {100*correct.mean():.1f}%")
    op_idx_arr = np.array([m["op_idx"] for m in meta])
    for c, name in [(0, "addition"), (1, "subtraction"), (2, "multiplication"), (3, "division")]:
        m = op_idx_arr == c
        if m.sum() == 0: continue
        n_c = int(correct[m].sum()); n_t = int(m.sum())
        print(f"    {name:>15s}: {n_c}/{n_t} = {100*n_c/n_t:.1f}%")

    out_meta = PD / "svamp_like_meta.json"
    out_meta.write_text(json.dumps({
        "meta": meta, "preds": pred_strings,
        "pred_floats": pred_floats,
        "correct": [bool(c) for c in correct],
        "n_correct": int(correct.sum()), "n_total": int(N),
        "p_capture": P_CAPTURE, "max_new": MAX_NEW,
    }, indent=2))
    print(f"saved {out_meta}")


if __name__ == "__main__":
    main()
