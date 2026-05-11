"""Extract CODI-GPT-2's actual lm_head matrix (with the added BOT/EOT/PAD
tokens) from the checkpoint, save as a numpy .npy. This fixes the logit-lens
projection that previously used vanilla GPT-2's unembedding."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "codi"))

import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file


def main():
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    out_W = REPO / "experiments" / "computation_probes" / "codi_gpt2_lm_head.npy"
    out_meta = REPO / "experiments" / "computation_probes" / "codi_gpt2_lm_head_meta.json"

    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore

    base_model = "gpt2"
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path=base_model, full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_lm_head", bf16=True,
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
    tok = transformers.AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")

    W = model.codi.get_output_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    print(f"  CODI lm_head W shape: {W.shape}", flush=True)
    np.save(out_W, W)
    print(f"  saved {out_W}")

    # Save metadata: bot_id, eot_id, pad_id, vocab size
    import json
    meta = {
        "vocab_size": int(W.shape[0]),
        "hidden_size": int(W.shape[1]),
        "bot_id": int(model.bot_id) if model.bot_id is not None else None,
        "eot_id": int(model.eot_id) if model.eot_id is not None else None,
        "pad_id": int(model.pad_token_id) if model.pad_token_id is not None else None,
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"  saved {out_meta}: {meta}")


if __name__ == "__main__":
    main()
