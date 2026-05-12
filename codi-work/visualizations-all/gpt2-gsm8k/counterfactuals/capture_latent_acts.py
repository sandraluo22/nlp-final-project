"""Capture CODI-GPT-2 latent-loop activations for any dataset.

Mirrors capture_colon_acts.py but saves the LATENT-LOOP residuals (one per
latent step, all 13 layers) instead of the ':' residual.

Output: {out}_latent_acts.pt of shape (N, 6, 13, 768) bf16.
"""
from __future__ import annotations

import argparse
import json
import os
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


def load_dataset_questions(name: str):
    if name == "svamp":
        from datasets import load_dataset, concatenate_datasets
        ds = load_dataset("gsm8k", "main")
        full = concatenate_datasets([ds["train"], ds["test"]])
        qs = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
        golds = [float(str(ex["Answer"]).replace(",", "")) for ex in full]
        types = [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
        return qs, golds, types
    rows = json.load(open(CF_DIR / f"{name}.json"))
    qs, golds, types = [], [], []
    if name in ("cf_balanced", "cf_magmatched", "cf_under99", "cf_under99_b"):
        for r in rows:
            qs.append(r["cf_question_concat"].strip().replace("  ", " "))
            golds.append(float(r.get("cf_answer", float("nan"))))
            types.append(r.get("type", ""))
    elif name.startswith("vary_"):
        for r in rows:
            qs.append(r["question_concat"].strip().replace("  ", " "))
            golds.append(float(r["answer"]))
            types.append(r["type"])
    elif name.startswith("numeral_pairs_"):
        for r in rows:
            text = r["clean"]["text"]
            qs.append(text.strip().replace("  ", " "))
            golds.append(float(r["clean"]["answer"]))
            types.append(r.get("type", ""))
    else:
        raise SystemExit(f"unknown dataset {name}")
    return qs, golds, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    out_tag = args.out or args.dataset

    questions, golds, types = load_dataset_questions(args.dataset)
    N = len(questions)
    print(f"dataset={args.dataset}  N={N}", flush=True)

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
    targs = TrainingArguments(output_dir="/tmp/_cl", bf16=True,
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

    L_plus1 = model.codi.config.n_layer + 1
    HID = model.codi.config.n_embd
    out = torch.zeros(N, 6, L_plus1, HID, dtype=torch.bfloat16)

    @torch.no_grad()
    def run_batch(start):
        qs = questions[start:start + args.bs]
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out_p = model.codi(input_ids=input_ids, attention_mask=attn,
                           use_cache=True, output_hidden_states=True)
        past = out_p.past_key_values
        latent = out_p.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for step in range(6):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            o = model.codi(inputs_embeds=latent, attention_mask=attn,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            hs = torch.stack([h[:, -1, :] for h in o.hidden_states], dim=1)  # (B, L+1, H)
            out[start:start + B, step] = hs.to(torch.bfloat16).cpu()
            latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)

    t0 = time.time()
    for s in range(0, N, args.bs):
        run_batch(s)
        done = s + args.bs
        if done % 64 == 0 or done >= N:
            print(f"  {min(done, N)}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    out_pt = PD / f"{out_tag}_latent_acts.pt"
    torch.save(out, out_pt)
    print(f"saved {out_pt}  ({out_pt.stat().st_size / 1e6:.1f} MB)  shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
