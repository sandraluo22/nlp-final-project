"""Capture activations at each of CODI's 6 LATENT LOOP steps + prompt_end
position on REAL SVAMP (1000 examples). Same setup as svamp_fixed_acts so
the probe fitting on real SVAMP is consistent across stages."""

from __future__ import annotations
import json, os, sys, time
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
    targs = TrainingArguments(output_dir="/tmp/_svll", bf16=True,
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

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    N = len(questions)
    print(f"  N={N}", flush=True)

    bs = 16
    latent_acts = []
    pe_acts = []
    t0 = time.time()
    for i in range(0, N, bs):
        batch_q = questions[i:i+bs]
        B = len(batch_q)
        batch = tok(batch_q, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        with torch.no_grad():
            out = model.codi(input_ids=input_ids, attention_mask=attn,
                             use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            pe_step = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
            pe_acts.append(pe_step.to(torch.float32).cpu())
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
            steps = []
            for s in range(targs.inf_latent_iterations):
                attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
                out = model.codi(inputs_embeds=latent, attention_mask=attn,
                                 use_cache=True, output_hidden_states=True,
                                 past_key_values=past)
                past = out.past_key_values
                steps.append(torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1))
                latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
                if targs.use_prj: latent = model.prj(latent)
            latent_acts.append(torch.stack(steps, dim=1).to(torch.float32).cpu())
        d = i + B
        if d % 64 == 0 or d == N:
            print(f"  {d}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    latent_t = torch.cat(latent_acts, dim=0)[:N]
    pe_t = torch.cat(pe_acts, dim=0)[:N]
    print(f"\n  latent: {tuple(latent_t.shape)}  prompt_end: {tuple(pe_t.shape)}")
    out_pt = PD / "svamp_real_latent_acts.pt"
    torch.save({"latent": latent_t.to(torch.bfloat16),
                "prompt_end": pe_t.to(torch.bfloat16)}, out_pt)
    print(f"saved {out_pt}  ({out_pt.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
