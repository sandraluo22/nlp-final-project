"""Sweep number of latent loop steps from 1 to 20. For each K:
  - Run inference on real SVAMP (200 examples)
  - Score CODI's accuracy
  - Capture residual at canonical cell (layer 8) at every latent step
  - Compute cos similarity between consecutive step residuals (mean over examples)
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

PROBE_LAYER = 8


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
    from src.model import CODI, ModelArguments, TrainingArguments
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_lk", bf16=True,
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

    @torch.no_grad()
    def run_batch(qs, *, K):
        """Run inference with K latent loop steps. Capture residual at
        (PROBE_LAYER) at every step. Return predictions + activations
        (B, K, hidden)."""
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

        cap_per_step = []
        for s in range(K):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            cap_per_step.append(out.hidden_states[PROBE_LAYER][:, -1, :].to(torch.float32).cpu().numpy())
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        cap_arr = np.stack(cap_per_step, axis=1)  # (B, K, H)

        # Decode
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(256):
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
        strs = [tok.decode(t, skip_special_tokens=True) for t in tokens]
        return cap_arr, strs

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    np.random.seed(0)
    eval_idx = np.random.choice(len(full), size=200, replace=False)
    eval_qs = [questions[i] for i in eval_idx]
    eval_gold = golds[eval_idx]
    BS = 16

    K_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]
    accuracy_per_K = {}
    cos_sim_per_K = {}   # K -> (K-1,) array of mean cos sim between consecutive steps

    for K in K_values:
        # Update model to use K latent iterations
        targs.inf_latent_iterations = K
        # Run
        all_caps = []
        all_strs = []
        t0 = time.time()
        for s in range(0, len(eval_qs), BS):
            cap, strs = run_batch(eval_qs[s:s+BS], K=K)
            all_caps.append(cap); all_strs += strs
        all_caps = np.concatenate(all_caps, axis=0)  # (200, K, H)
        # Accuracy
        n_correct = 0
        for s, g in zip(all_strs, eval_gold):
            v = codi_extract(s)
            if v is not None and abs(v - g) < 1e-3:
                n_correct += 1
        acc = n_correct / len(eval_qs)
        accuracy_per_K[K] = acc
        # Cos sim between consecutive steps (per example, then mean)
        if K > 1:
            cs = []
            for k in range(K - 1):
                a = all_caps[:, k]; b = all_caps[:, k+1]
                cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
                cs.append(float(cos.mean()))
            cos_sim_per_K[K] = cs
        else:
            cos_sim_per_K[K] = []
        print(f"  K={K:>2d}: acc={n_correct}/200={acc*100:.1f}%  ({time.time()-t0:.0f}s)  "
              f"cos_sim_consec_first5={cos_sim_per_K[K][:5]}", flush=True)

    out = PD / "latent_steps_sweep.json"
    out.write_text(json.dumps({
        "K_values": K_values,
        "accuracy": {str(k): v for k, v in accuracy_per_K.items()},
        "cos_sim_consec_per_K": {str(k): v for k, v in cos_sim_per_K.items()},
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
