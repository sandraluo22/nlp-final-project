"""Per-head attention ablation on GSM8K test (zero-out a single head at a
specific (step, layer) and measure accuracy).

For each (latent_step, layer, head_idx):
  - Zero that head's value vector at the last-token position during step k's
    forward pass.
  - Continue model normally through remaining steps + emission.
  - Record the emitted prediction.

GPT-2-small has 12 heads × 12 layers × 6 steps × 1 (attn-only) = 864 cells.
Too many for a full sweep with default batches; we run a strategic subset:
  - Each step × layer × all 12 heads at THE LAST EMISSION (where we showed
    L2-L11 attention is the answer-read site on SVAMP).
  - Or: per-(step, layer) zero ALL HEADS (= zero whole attn block) for
    comparison baseline.

By default we sweep the FULL grid but limit N_EXAMPLES to 200 for speed.

Output:
  per_head_ablation_gsm8k.json:
    {
      "baseline_acc": float,
      "ablate_per_cell": {
        "step_S_L_LH_": {acc, delta_acc, n_changed, ...},
        ...
      }
    }
"""
from __future__ import annotations

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
sys.path.insert(0, str(REPO / "codi"))

N_EXAMPLES = 200
OUT_PATH = PD / "per_head_ablation_gsm8k.json"


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    BS = 1
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    questions, golds = [], []
    for ex in ds:
        m = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if m is None: continue
        questions.append(ex["question"].strip().replace("  ", " "))
        golds.append(float(m.group(1)))
    questions = questions[:N_EXAMPLES]; golds = golds[:N_EXAMPLES]
    N = len(questions); golds_arr = np.array(golds)

    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI from {ckpt}", flush=True)
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
    targs = TrainingArguments(output_dir="/tmp/_pha", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=768,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=6,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False); model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16); model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id
    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    N_LAYERS = model.codi.config.n_layer
    N_HEADS = model.codi.config.n_head
    HID = model.codi.config.n_embd
    D_HEAD = HID // N_HEADS
    N_LAT = 6
    print(f"  N_LAYERS={N_LAYERS} N_HEADS={N_HEADS} D_HEAD={D_HEAD} N_LAT={N_LAT}")

    CAP = {"step": -1, "ablate_step": -1, "ablate_layer": -1, "ablate_head": -1}

    def make_attn_hook(idx):
        def fn(_m, _i, output):
            if (CAP["step"] == CAP["ablate_step"] and idx == CAP["ablate_layer"]
                    and CAP["ablate_head"] >= 0):
                a = output[0].clone()
                # a: (B, T, H). Zero head 'ablate_head' at the LAST position.
                h = CAP["ablate_head"]
                a[:, -1, h*D_HEAD:(h+1)*D_HEAD] = 0
                return (a,) + output[1:]
            return output
        return fn

    handles = [transformer.h[i].attn.register_forward_hook(make_attn_hook(i))
               for i in range(N_LAYERS)]

    @torch.no_grad()
    def run_one(q):
        CAP["step"] = -1
        batch = tok([q], return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((1, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for step in range(N_LAT):
            CAP["step"] = step
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            o = model.codi(inputs_embeds=latent, attention_mask=attn,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        CAP["step"] = -1
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda")).unsqueeze(0)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
        output = eot_emb; tokens = []
        for _ in range(48):
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past)
            past = s.past_key_values
            nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
            tokens.append(int(nid.item()))
            if tokens[-1] == eos_id: break
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(nid).unsqueeze(1)
        return tok.decode(tokens, skip_special_tokens=True)

    # Baseline
    CAP["ablate_step"] = -1; CAP["ablate_layer"] = -1; CAP["ablate_head"] = -1
    t0 = time.time()
    base_strs = [run_one(q) for q in questions]
    base_ints = [codi_extract(s) for s in base_strs]
    base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                              for i, v in enumerate(base_ints)])
    base_acc = float(base_correct.mean())
    print(f"baseline acc={base_acc:.2f}  ({time.time()-t0:.0f}s for {N} examples)")

    # Per-head ablation grid
    grid = {}
    for step in range(N_LAT):
        for layer in range(N_LAYERS):
            for head in range(N_HEADS):
                CAP["ablate_step"] = step; CAP["ablate_layer"] = layer; CAP["ablate_head"] = head
                strs = [run_one(q) for q in questions]
                ints = [codi_extract(s) for s in strs]
                correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                    for i, v in enumerate(ints)])
                n_changed = int(sum(1 for i in range(N)
                                    if (ints[i] is None) != (base_ints[i] is None)
                                    or (ints[i] is not None and base_ints[i] is not None
                                        and abs(ints[i] - base_ints[i]) > 1e-3)))
                acc = float(correct.mean())
                wr = int(((~base_correct) & correct).sum())
                rw = int((base_correct & ~correct).sum())
                grid[f"step{step+1}_L{layer}_H{head}"] = {
                    "acc": acc, "delta_acc": acc - base_acc,
                    "n_changed": n_changed,
                    "wrong_to_right": wr, "right_to_wrong": rw,
                }
            print(f"  step {step+1} L{layer} done; current min delta_acc = "
                  f"{min(grid[f'step{step+1}_L{layer}_H{h}']['delta_acc'] for h in range(N_HEADS)):+.3f}",
                  flush=True)

    OUT_PATH.write_text(json.dumps({
        "N": N, "baseline_acc": base_acc, "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS,
        "N_LAT": N_LAT,
        "ablate_per_cell": grid,
    }, indent=2))
    print(f"saved {OUT_PATH}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
