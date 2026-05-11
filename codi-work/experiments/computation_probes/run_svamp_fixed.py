"""Fixed SVAMP eval matching CODI's official test.py:
  - max_new_tokens = 256
  - per-example EOS stopping
  - parser = LAST number in the output

Captures multi-position residual activations during the FIRST 16 decode steps
(same as before so the probe pipeline is consistent), then continues to 256
tokens for the final answer-extraction. Uses longest-prefix-correct gen
limit to avoid wasted compute on examples that are clearly done.

Outputs:
  svamp_fixed_acts.pt    — (N, 16, L+1, H) residual at first 16 decode positions
  svamp_fixed_preds.json — full preds + per-example pred_int + correctness
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

P_CAPTURE = 16   # capture residual at first 16 decode positions (same as before)
MAX_NEW = 256    # but keep generating up to 256 for the official answer parse


def codi_extract(sentence: str):
    """CODI's official last-number parser."""
    sentence = sentence.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', sentence)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


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
    targs = TrainingArguments(output_dir="/tmp/_fix", bf16=True,
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

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    golds = [float(str(ex["Answer"]).replace(",", "")) for ex in full]
    N = len(questions)
    print(f"  N={N}  P_CAPTURE={P_CAPTURE}  MAX_NEW={MAX_NEW}", flush=True)

    multi_acts = []
    pred_strings = []
    pred_ids_per_pos = []
    t0 = time.time()
    bs = 16
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
            # decode pos 0: feed EOT, capture
            eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
            output = eot_emb.unsqueeze(0).expand(B, -1, -1)
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            tokens = [[] for _ in range(B)]
            done = [False] * B
            pos_acts = []
            for p in range(MAX_NEW):
                step_out = model.codi(inputs_embeds=output, attention_mask=attn,
                                      use_cache=True,
                                      output_hidden_states=(p < P_CAPTURE),
                                      past_key_values=past)
                past = step_out.past_key_values
                if p < P_CAPTURE:
                    pos_acts.append(torch.stack([h[:, -1, :] for h in step_out.hidden_states], dim=1))
                logits = step_out.logits[:, -1, :model.codi.config.vocab_size - 1]
                next_ids = torch.argmax(logits, dim=-1)
                # stop tracking per-example after EOS
                for b in range(B):
                    if not done[b]:
                        tokens[b].append(int(next_ids[b].item()))
                        if int(next_ids[b].item()) == eos_id:
                            done[b] = True
                if all(done): break
                attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
                output = embed_fn(next_ids).unsqueeze(1)
            # stack per-position acts: pad if we exited early before P_CAPTURE
            while len(pos_acts) < P_CAPTURE:
                # pad last (shouldn't happen unless every example finishes in <16 tokens)
                pos_acts.append(pos_acts[-1].clone())
            pos_acts_t = torch.stack(pos_acts[:P_CAPTURE], dim=1).to(torch.float32).cpu()
            multi_acts.append(pos_acts_t)
            for b in range(B):
                pred_strings.append(tok.decode(tokens[b], skip_special_tokens=True))
                pad_p = list(tokens[b][:P_CAPTURE]) + [eos_id] * max(0, P_CAPTURE - len(tokens[b]))
                pred_ids_per_pos.append(pad_p[:P_CAPTURE])
        done_n = i + B
        if done_n % 64 == 0 or done_n == N:
            print(f"  {done_n}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    multi_t = torch.cat(multi_acts, dim=0)
    print(f"\nshape: {tuple(multi_t.shape)}", flush=True)
    out_pt = PD / "svamp_fixed_acts.pt"
    torch.save(multi_t.to(torch.bfloat16), out_pt)
    print(f"saved {out_pt}  ({out_pt.stat().st_size/1e6:.1f} MB)")

    # parse predictions with CODI's official extractor
    pred_floats = [codi_extract(s) for s in pred_strings]
    correct = [pf is not None and abs(pf - g) < 1e-3 for pf, g in zip(pred_floats, golds)]
    n_correct = sum(correct)
    print(f"\n=== Accuracy: {n_correct}/{N} = {100*n_correct/N:.2f}% ===")

    # save
    out_json = PD / "svamp_fixed_preds.json"
    out_json.write_text(json.dumps({
        "preds": pred_strings,
        "pred_floats": [None if pf is None else float(pf) for pf in pred_floats],
        "golds": golds,
        "correct": correct,
        "pred_ids_per_pos": pred_ids_per_pos,
        "p_capture": P_CAPTURE,
        "max_new": MAX_NEW,
        "n_correct": int(n_correct),
        "n": int(N),
    }, indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
