"""Capture CODI-GPT-2 residuals AT the ":" token position for any dataset.

The ":" residual is the canonical "right before answer emission" cell —
the model has just been fed ":" and the next output is the answer digit(s).

Usage:
  python capture_colon_acts.py --dataset svamp --out svamp
  python capture_colon_acts.py --dataset cf_balanced --out cf_balanced
  ...

Output:
  {out}_colon_acts.pt        (N, layers+1=13, hidden=768) bf16
  {out}_colon_acts_meta.json examples skipped, gold answers, types, etc.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
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

MAX_DECODE = 16


def load_dataset_questions(name: str):
    """Return (questions, golds, types, schema_note)."""
    if name == "svamp":
        from datasets import load_dataset, concatenate_datasets
        ds = load_dataset("ChilleD/SVAMP")
        full = concatenate_datasets([ds["train"], ds["test"]])
        qs = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
        golds = [float(str(ex["Answer"]).replace(",", "")) for ex in full]
        types = [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
        return qs, golds, types, "SVAMP train+test"
    p = CF_DIR / f"{name}.json"
    rows = json.load(open(p))
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
            # use the "clean" version as the question
            text = r["clean"]["text"]
            qs.append(text.strip().replace("  ", " "))
            golds.append(float(r["clean"]["answer"]))
            types.append(r.get("type", ""))
    elif name == "cf_gpt_transformed":
        raise SystemExit(f"cf_gpt_transformed: cf_question field is a placeholder; text not stored. Regenerate first.")
    else:
        raise SystemExit(f"unknown dataset {name}")
    return qs, golds, types, p.name


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="svamp | cf_balanced | cf_magmatched | cf_under99 | cf_under99_b "
                         "| vary_a | vary_a_2digit | vary_b | vary_b_2digit "
                         "| vary_both_2digit | vary_numerals | vary_operator "
                         "| numeral_pairs_a1_mul | numeral_pairs_b1_sub")
    ap.add_argument("--out", default=None, help="output prefix (default = dataset)")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    out_tag = args.out or args.dataset

    questions, golds, types, schema_note = load_dataset_questions(args.dataset)
    N = len(questions)
    print(f"dataset={args.dataset}  N={N}  ({schema_note})", flush=True)

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
    targs = TrainingArguments(output_dir="/tmp/_cc", bf16=True,
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

    # All token-id variants of ":"
    colon_ids = set()
    for s_ in (":", " :"):
        for tid in tok.encode(s_, add_special_tokens=False):
            colon_ids.add(int(tid))
    print(f"colon token ids = {colon_ids}", flush=True)

    L_plus1 = model.codi.config.n_layer + 1
    HID = model.codi.config.n_embd
    colon_acts = torch.zeros(N, L_plus1, HID, dtype=torch.bfloat16)
    emit_pos_of_colon = np.full(N, -1, dtype=int)
    pred_strs = [None] * N
    pred_int_extracted = [None] * N

    @torch.no_grad()
    def run_batch(start):
        qs = questions[start:start + args.bs]
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
        for _ in range(targs.inf_latent_iterations):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        # Decode loop with capture per step + EOS early-exit per example.
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        prev_token_id = [model.eot_id] * B
        captured_acts = [None] * B
        captured_pos = [-1] * B
        tokens = [[] for _ in range(B)]
        done = [False] * B   # per-example EOS reached
        for step in range(MAX_DECODE):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=True,
                              past_key_values=past)
            past = sout.past_key_values
            hs = torch.stack([h[:, -1, :] for h in sout.hidden_states], dim=1)  # (B, L+1, H)
            for b in range(B):
                if prev_token_id[b] in colon_ids and captured_acts[b] is None:
                    captured_acts[b] = hs[b].to(torch.bfloat16).cpu()
                    captured_pos[b] = step
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if done[b]: continue
                tid = int(next_ids[b].item())
                tokens[b].append(tid)
                if tid == eos_id: done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
            for b in range(B):
                prev_token_id[b] = int(next_ids[b].item())
        for j in range(B):
            i = start + j
            if captured_acts[j] is not None:
                colon_acts[i] = captured_acts[j]
            emit_pos_of_colon[i] = captured_pos[j]
            pred_strs[i] = tok.decode(tokens[j], skip_special_tokens=True)
            pred_int_extracted[i] = codi_extract(pred_strs[i])

    import time
    t0 = time.time()
    for s in range(0, N, args.bs):
        run_batch(s)
        done = s + args.bs
        if done % 64 == 0 or done >= N:
            print(f"  {min(done, N)}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    n_captured = int((emit_pos_of_colon >= 0).sum())
    print(f"\ncaptured ':' residual for {n_captured}/{N} examples")
    if n_captured < N:
        print(f"  positions: {dict(zip(*np.unique(emit_pos_of_colon, return_counts=True)))}")

    out_pt = PD / f"{out_tag}_colon_acts.pt"
    out_meta = PD / f"{out_tag}_colon_acts_meta.json"
    torch.save(colon_acts, out_pt)
    meta = {
        "dataset": args.dataset, "N": int(N), "n_captured": n_captured,
        "shape": list(colon_acts.shape), "dtype": "bfloat16",
        "emit_pos_of_colon": emit_pos_of_colon.tolist(),
        "preds": pred_strs, "pred_int_extracted": pred_int_extracted,
        "gold": [None if (g is None or (isinstance(g, float) and np.isnan(g))) else float(g) for g in golds],
        "types": types,
        "notes": ("residual at the step whose INPUT is ':' (i.e., the residual produced when "
                  "the model has just been fed ':' and is about to emit the answer)."),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"saved {out_pt}  ({out_pt.stat().st_size / 1e6:.1f} MB)")
    print(f"saved {out_meta}")


if __name__ == "__main__":
    main()
