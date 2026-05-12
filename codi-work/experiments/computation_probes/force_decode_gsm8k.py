"""Force-decode CODI-GPT-2's chain-of-thought on GSM8K test.

For each example, captures:
  - The greedy-decoded prediction (full emission).
  - For each of the 6 latent steps, the SINGLE TOKEN that the LM head would
    emit if we read off the last-layer residual at that step's last position.
  - Joined into a "CODI internal CoT" string for use by the faithfulness
    judge.

Output: gsm8k_codi_cot.json
   [{idx, question, gold, codi_pred_int, codi_full_emission,
     codi_step_tokens: [t1,...,t6]}, ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

OUT_PATH = CF_DIR / "gsm8k_codi_cot.json"


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    questions, golds = [], []
    for ex in ds:
        m = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if m is None: continue
        questions.append(ex["question"].strip().replace("  ", " "))
        golds.append(float(m.group(1)))
    N = len(questions)

    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=["c_attn", "c_proj", "c_fc"],
                          init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_fd", bf16=True,
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
    model = model.to("cuda").to(torch.bfloat16); model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id

    # LM head: weight tied to embeddings
    lm_head = model.codi.get_output_embeddings()   # nn.Linear

    BS = args.bs
    rows = []
    t0 = time.time()

    @torch.no_grad()
    def run_batch(qs, indices):
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
        # Per-step: capture the last-layer hidden state at the new position
        # and force-decode it through lm_head.
        step_tokens = [[] for _ in range(B)]
        for step in range(6):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            o = model.codi(inputs_embeds=latent, attention_mask=attn,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            h_last = o.hidden_states[-1][:, -1, :]    # (B, 768)
            logits = lm_head(h_last)                  # (B, vocab)
            tok_ids = torch.argmax(logits[:, :model.codi.config.vocab_size - 1], dim=-1)
            for b in range(B):
                step_tokens[b].append(int(tok_ids[b].item()))
            latent = h_last.unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        # Emission phase
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]; done = [False] * B
        for _ in range(64):
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past)
            past = s.past_key_values
            logits = s.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if done[b]: continue
                tid = int(next_ids[b].item()); tokens[b].append(tid)
                if tid == eos_id: done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        for b in range(B):
            emit_text = tok.decode(tokens[b], skip_special_tokens=True)
            step_text = [tok.decode([t], skip_special_tokens=True) for t in step_tokens[b]]
            pred_int = codi_extract(emit_text)
            rows.append({
                "idx": int(indices[b]),
                "question": qs[b],
                "gold": float(golds[indices[b]]),
                "codi_emit": emit_text,
                "codi_pred_int": None if pred_int is None else float(pred_int),
                "codi_step_tokens": step_text,
            })

    for s in range(0, N, BS):
        run_batch(questions[s:s+BS], list(range(s, min(s+BS, N))))
        if (s + BS) % 128 == 0 or s + BS >= N:
            elapsed = time.time() - t0
            print(f"  {min(s+BS, N)}/{N}  ({elapsed:.0f}s)")

    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"saved {len(rows)} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
