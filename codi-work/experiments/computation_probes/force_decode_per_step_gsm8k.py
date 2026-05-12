"""Force-decode CODI's answer AT EACH LATENT STEP on GSM8K test.

For each example: run CODI normally up through latent step k (k = 1..6),
then jump to EOT and decode auto-regressively to get the FULL emitted
answer. This gives a (N, 6) array of "what answer would CODI give if we
cut off the latent loop at step k".

Use case: step1to2_deep_dive and related cohort analyses (wr / rw /
same_right / same_wrong cohorts based on per-step correctness).

Output:
  force_decode_per_step_gsm8k.json:
    {
      "N": int,
      "gold": [...],
      "baseline_emit": [...],         # full 6-step emit
      "step_predictions": [[...]],    # (N, 6) int predictions per step
      "step_correct": [[...]],        # (N, 6) bool per step
      "correct_per_step": [[...]],    # (6, N) transpose for legacy compat
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
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

OUT_PATH = PD / "force_decode_per_step_gsm8k.json"


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    BS = 16
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    questions, golds = [], []
    for ex in ds:
        m = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if m is None: continue
        questions.append(ex["question"].strip().replace("  ", " "))
        golds.append(float(m.group(1)))
    N = len(questions); golds_arr = np.array(golds)
    print(f"GSM8K test: N={N}")

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
    targs = TrainingArguments(output_dir="/tmp/_fdps", bf16=True,
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
    N_LAT = 6

    @torch.no_grad()
    def run_batch_with_force_decodes(qs):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        prompt_past = out.past_key_values
        prompt_attn = attn
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        # Save (past, attn, latent) snapshot after each latent step.
        # Branch: at step k, copy past + attn and finish via EOT decode.
        step_emits = [[None] * B for _ in range(N_LAT)]
        baseline_emit = [None] * B

        past = prompt_past; attn_run = prompt_attn
        for step in range(N_LAT):
            attn_run = torch.cat([attn_run,
                                   torch.ones((B, 1), dtype=attn_run.dtype, device="cuda")], dim=1)
            o = model.codi(inputs_embeds=latent, attention_mask=attn_run,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
            # Branch: decode from end-of-step-k by jumping to EOT.
            branch_past = past
            branch_attn = torch.cat([attn_run,
                                      torch.ones((B, 1), dtype=attn_run.dtype, device="cuda")], dim=1)
            eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda")).unsqueeze(0).expand(B, -1, -1)
            tokens = [[] for _ in range(B)]; done = [False] * B
            output = eot_emb
            for _ in range(64):
                s = model.codi(inputs_embeds=output, attention_mask=branch_attn,
                               use_cache=True, past_key_values=branch_past)
                branch_past = s.past_key_values
                nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
                for b in range(B):
                    if done[b]: continue
                    tid = int(nid[b].item()); tokens[b].append(tid)
                    if tid == eos_id: done[b] = True
                if all(done): break
                branch_attn = torch.cat([branch_attn,
                                          torch.ones((B, 1), dtype=branch_attn.dtype, device="cuda")], dim=1)
                output = embed_fn(nid).unsqueeze(1)
            for b in range(B):
                step_emits[step][b] = tok.decode(tokens[b], skip_special_tokens=True)

        for b in range(B):
            baseline_emit[b] = step_emits[N_LAT - 1][b]
        return baseline_emit, step_emits

    rows = []
    step_preds = np.full((N, N_LAT), np.nan)
    t0 = time.time()
    for s in range(0, N, BS):
        qs_batch = questions[s:s+BS]
        idx_batch = list(range(s, s+len(qs_batch)))
        baseline_emit, step_emits = run_batch_with_force_decodes(qs_batch)
        for bi, gi in enumerate(idx_batch):
            for k in range(N_LAT):
                v = codi_extract(step_emits[k][bi])
                step_preds[gi, k] = v if v is not None else np.nan
            rows.append({
                "idx": int(gi), "gold": float(golds_arr[gi]),
                "baseline_emit": baseline_emit[bi],
                "step_emits": [step_emits[k][bi] for k in range(N_LAT)],
            })
        if (s + BS) % 128 == 0 or s + BS >= N:
            print(f"  {min(s+BS, N)}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    correct = np.zeros((N, N_LAT), dtype=bool)
    for i in range(N):
        for k in range(N_LAT):
            p = step_preds[i, k]
            correct[i, k] = (not np.isnan(p)) and abs(p - golds_arr[i]) < 1e-3

    OUT_PATH.write_text(json.dumps({
        "N": N, "gold": golds_arr.tolist(),
        "step_predictions": step_preds.tolist(),
        "step_correct": correct.tolist(),
        "correct_per_step": correct.T.astype(int).tolist(),
        "rows": rows,
    }, indent=2))
    acc_per_step = correct.mean(axis=0)
    print(f"\nGSM8K force-decode-per-step accuracy:")
    for k in range(N_LAT):
        print(f"  step {k+1}: {acc_per_step[k]*100:.1f}%")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
