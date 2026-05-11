"""Force-decode per latent step on CODI-Llama-3.2-1B-Instruct.

Same methodology as force_decode_per_step.py:
  K_max = 10 (trained K=6 + 4 off-distribution steps)
  Run latent iteration k=1..K_max, force-decode after each, score per-example
  correctness, count right<->wrong transitions.
"""

from __future__ import annotations
import argparse, json, os, re, sys, time
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

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
CKPT_DIR_NAME = "CODI-llama3.2-1b-Instruct"


def codi_extract(s: str):
    s = s.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="number of SVAMP examples (0 = use all 1000)")
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", type=str, default="force_decode_per_step_llama.json")
    args = ap.parse_args()
    K_MAX = args.k_max

    ckpt = os.path.expanduser(f"~/codi_ckpt/{CKPT_DIR_NAME}")
    print(f"loading CODI-Llama from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    # Llama target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path=BASE_MODEL, full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_fdl", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=2048,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=K_MAX,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained(BASE_MODEL, model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id

    def clone_past(past):
        return tuple((k.clone(), v.clone()) for k, v in past)

    @torch.no_grad()
    def decode_from_past(past, attn, B, max_new=96):
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn_local = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_new):
            sout = model.codi(inputs_embeds=output, attention_mask=attn_local,
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
            attn_local = torch.cat([attn_local, torch.ones((B, 1), dtype=attn_local.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    @torch.no_grad()
    def run_batch(qs):
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

        preds_per_step = []
        for k in range(K_MAX):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            past_clone = clone_past(past)
            strs = decode_from_past(past_clone, attn, B, max_new=96)
            preds_per_step.append(strs)
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        return preds_per_step

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    if args.n and args.n < len(full):
        np.random.seed(0)
        eval_idx = np.random.choice(len(full), size=args.n, replace=False)
        eval_qs = [questions[i] for i in eval_idx]
        eval_gold = golds[eval_idx]
    else:
        eval_qs = questions
        eval_gold = golds

    BS = args.bs
    correct_per_step = np.zeros((K_MAX, len(eval_qs)), dtype=bool)
    t0 = time.time()
    for s in range(0, len(eval_qs), BS):
        preds_per_step = run_batch(eval_qs[s:s+BS])
        for k in range(K_MAX):
            for j, pred in enumerate(preds_per_step[k]):
                v = codi_extract(pred)
                correct_per_step[k, s + j] = (v is not None and abs(v - eval_gold[s + j]) < 1e-3)
        done = s + BS
        if done % 16 == 0 or done >= len(eval_qs):
            print(f"  {min(done, len(eval_qs))}/{len(eval_qs)}  ({time.time()-t0:.0f}s)", flush=True)

    acc_per_step = correct_per_step.mean(axis=1)
    print(f"\n=== Accuracy per step (CODI-Llama-1B) ===")
    for k in range(K_MAX):
        print(f"  step {k+1:2d}: {int(correct_per_step[k].sum()):3d}/{len(eval_qs)} = {acc_per_step[k]*100:.1f}%")

    print(f"\n=== Transitions k -> k+1 ===")
    transitions = []
    for k in range(K_MAX - 1):
        c_k = correct_per_step[k]; c_k1 = correct_per_step[k+1]
        n_rtw = int((c_k & ~c_k1).sum()); n_wtr = int((~c_k & c_k1).sum())
        transitions.append({"from_step": k+1, "to_step": k+2,
                            "right_to_wrong": n_rtw, "wrong_to_right": n_wtr,
                            "stable_right": int((c_k & c_k1).sum()),
                            "stable_wrong": int((~c_k & ~c_k1).sum())})
        print(f"  step {k+1:2d} -> {k+2:2d}: right->wrong={n_rtw:3d}  wrong->right={n_wtr:3d}  net={n_wtr-n_rtw:+d}")

    out = PD / args.out
    out.write_text(json.dumps({
        "K_max": K_MAX, "n_eval": len(eval_qs),
        "accuracy_per_step": acc_per_step.tolist(),
        "n_correct_per_step": correct_per_step.sum(axis=1).tolist(),
        "transitions": transitions,
        "correct_per_step": correct_per_step.astype(int).tolist(),
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
