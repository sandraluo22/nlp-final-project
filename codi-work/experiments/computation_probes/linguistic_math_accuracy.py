"""Test CODI-GPT-2 accuracy on linguistic arithmetic prompts (vs bare-symbol).
Try several phrasings to find which one CODI handles best."""

from __future__ import annotations
import os, re, sys, time
from pathlib import Path
from itertools import product

import torch
import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "codi"))


def parse_first_int(s):
    """LEGACY parser - kept for backward compat."""
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def codi_extract(s):
    """CODI's official last-number parser."""
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
    targs = TrainingArguments(output_dir="/tmp/_ling", bf16=True,
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

    # Templates to test (4 styles × 4 ops)
    TEMPLATES = {
        "verbose_question": [
            ("addition",       "What is {a} plus {b}?"),
            ("subtraction",    "What is {a} minus {b}?"),
            ("multiplication", "What is {a} times {b}?"),
            ("division",       "What is {a} divided by {b}?"),
        ],
        "added_to_phrase": [
            ("addition",       "{a} added to {b} ="),
            ("subtraction",    "{a} subtracted by {b} ="),
            ("multiplication", "{a} multiplied by {b} ="),
            ("division",       "{a} divided by {b} ="),
        ],
        "svamp_like": [
            ("addition",       "Sandy has {a} apples. She gets {b} more. How many apples does Sandy have?"),
            ("subtraction",    "Sandy has {a} apples. She gives away {b}. How many apples does Sandy have left?"),
            ("multiplication", "Sandy has {a} bags. Each bag has {b} apples. How many apples does Sandy have?"),
            ("division",       "Sandy has {a} apples. She splits them into groups of {b}. How many groups?"),
        ],
        "compute_only": [
            ("addition",       "Compute {a} + {b}."),
            ("subtraction",    "Compute {a} - {b}."),
            ("multiplication", "Compute {a} * {b}."),
            ("division",       "Compute {a} / {b}."),
        ],
    }

    eos_id = tok.eos_token_id

    @torch.no_grad()
    def run_batch(qs, max_new=256):
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

    BS = 32
    AB_PAIRS = list(product(range(1, 21), range(1, 21)))   # 400 pairs

    print("\n=== Per-template accuracy ===")
    for tname, ops in TEMPLATES.items():
        print(f"\n--- Template: {tname} ---")
        # Show one example
        a_ex, b_ex = 5, 3
        for op_name, tmpl in ops:
            print(f"  example for {op_name}: {tmpl.format(a=a_ex, b=b_ex)!r}")
        # Build prompts and run
        all_prompts, all_meta = [], []
        for op_name, tmpl in ops:
            for a, b in AB_PAIRS:
                all_prompts.append(tmpl.format(a=a, b=b))
                all_meta.append((op_name, a, b))
        # batched inference
        preds = []
        t0 = time.time()
        for s in range(0, len(all_prompts), BS):
            preds += run_batch(all_prompts[s:s+BS], max_new=256)
            done = s + BS
            if done % 320 == 0:
                print(f"    {min(done,len(all_prompts))}/{len(all_prompts)}  ({time.time()-t0:.0f}s)", flush=True)
        # compute accuracy
        per_op = {"addition": [0, 0], "subtraction": [0, 0],
                  "multiplication": [0, 0], "division": [0, 0]}
        for (op_name, a, b), pred in zip(all_meta, preds):
            if op_name == "addition":       gold = a + b
            elif op_name == "subtraction":  gold = a - b
            elif op_name == "multiplication": gold = a * b
            else:
                if b == 0: continue
                if a % b != 0: continue
                gold = a // b
            per_op[op_name][1] += 1
            p_val = codi_extract(pred)
            if p_val is not None and abs(p_val - gold) < 1e-3:
                per_op[op_name][0] += 1
        total_c, total_n = 0, 0
        for op, (c, n) in per_op.items():
            total_c += c; total_n += n
            print(f"    {op:>15s}: {c}/{n}  ({100*c/max(n,1):.1f}%)")
        print(f"    {'TOTAL':>15s}: {total_c}/{total_n}  ({100*total_c/max(total_n,1):.1f}%)")
        # Show 5 example predictions
        for i, (m, p) in enumerate(zip(all_meta, preds)):
            if i % 80 == 0:
                op_name, a, b = m
                print(f"    sample: ({op_name} {a},{b}) -> {p[:60]!r}")


if __name__ == "__main__":
    main()
