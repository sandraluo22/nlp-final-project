"""Modified inference for CODI-GPT-2: in addition to the last-prompt-token
activations we already saved, capture the residual stream at the FIRST
DECODE step (i.e., when the model emits the first token of its answer).

This is the activation at the position the model is committing to an answer,
which is what (2) "per-token PCA on the answer position" needs.

Saves: codi-work/experiments/computation_probes/svamp_decode_acts.pt
       (N, layers+1=13, hidden=768) bf16
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# bridge to codi-work/codi/src
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "codi"))

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file


def main():
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    base_model = "gpt2"
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
    margs = ModelArguments(model_name_or_path=base_model, full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_decode", bf16=True,
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
    tok = transformers.AutoTokenizer.from_pretrained(base_model, model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()

    # SVAMP loader matching the original svamp_student run
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    golds = [float(str(ex["Answer"]).replace(",", "")) for ex in full]
    N = len(questions)
    print(f"  N={N}", flush=True)

    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id

    decode_acts = []   # per question (layers+1, H)
    pred_strings = []
    bs = 32
    t0 = time.time()
    for i in range(0, N, bs):
        batch_q = questions[i:i+bs]
        batch_g = golds[i:i+bs]
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
            # FIRST decode forward — this is where we capture activations
            eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
            output = eot_emb.unsqueeze(0).expand(B, -1, -1)
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            step_out = model.codi(inputs_embeds=output, attention_mask=attn,
                                  use_cache=True, output_hidden_states=True,
                                  past_key_values=past)
            past = step_out.past_key_values
            # capture: shape (layers+1, H) for the LAST (and only) token in step
            hs = torch.stack([h[:, -1, :] for h in step_out.hidden_states], dim=1)  # (B, layers+1, H)
            decode_acts.append(hs.to(torch.float32).cpu())
            # continue greedy decoding for the answer string (no further capture)
            logits = step_out.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            tokens_per = [[int(t.item())] for t in next_ids]
            output = embed_fn(next_ids).unsqueeze(1)
            for _ in range(64 - 1):
                attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
                step_out = model.codi(inputs_embeds=output, attention_mask=attn,
                                      use_cache=True, output_hidden_states=False,
                                      output_attentions=False, past_key_values=past)
                past = step_out.past_key_values
                logits = step_out.logits[:, -1, :model.codi.config.vocab_size - 1]
                next_ids = torch.argmax(logits, dim=-1)
                for b in range(B):
                    tokens_per[b].append(int(next_ids[b].item()))
                output = embed_fn(next_ids).unsqueeze(1)
            for b in range(B):
                pred_strings.append(tok.decode(tokens_per[b], skip_special_tokens=True))
        done = i + B
        if done % 96 == 0 or done == N:
            print(f"  {done}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    decode_acts_t = torch.cat(decode_acts, dim=0)  # (N, layers+1, H)
    print(f"\ndecode_acts shape: {tuple(decode_acts_t.shape)}", flush=True)
    out = REPO / "experiments" / "computation_probes" / "svamp_decode_acts.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decode_acts_t.to(torch.bfloat16), out)
    print(f"saved {out}  ({out.stat().st_size/1e6:.1f} MB)")
    pred_path = REPO / "experiments" / "computation_probes" / "svamp_decode_preds.json"
    pred_path.write_text(json.dumps({"preds": pred_strings, "golds": golds}, indent=2))
    print(f"saved {pred_path}")


if __name__ == "__main__":
    main()
