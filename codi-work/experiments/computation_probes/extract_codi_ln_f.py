"""Extract CODI-GPT-2's final LayerNorm (ln_f) weight, bias, eps so we can
apply it locally before the logit-lens projection."""

from __future__ import annotations
import json, os, sys
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
    targs = TrainingArguments(output_dir="/tmp/_lnf", bf16=True,
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

    # Find ln_f. CODI-GPT2's underlying GPT2 model has it at `transformer.ln_f`.
    candidates = [
        ("codi", "transformer", "ln_f"),
        ("codi", "base_model", "model", "transformer", "ln_f"),
    ]
    ln_f = None
    for path in candidates:
        cur = model
        ok = True
        for p in path:
            if hasattr(cur, p): cur = getattr(cur, p)
            else: ok = False; break
        if ok and isinstance(cur, torch.nn.LayerNorm):
            ln_f = cur; break
    if ln_f is None:
        # walk all modules
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.LayerNorm) and name.endswith("ln_f"):
                ln_f = m; break
    print(f"  found ln_f: weight shape {tuple(ln_f.weight.shape)}, "
          f"bias shape {tuple(ln_f.bias.shape)}, eps {ln_f.eps}", flush=True)
    out = REPO / "experiments" / "computation_probes" / "codi_gpt2_ln_f.npz"
    np.savez(out,
             weight=ln_f.weight.detach().to(torch.float32).cpu().numpy(),
             bias=ln_f.bias.detach().to(torch.float32).cpu().numpy(),
             eps=np.array(float(ln_f.eps)))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
