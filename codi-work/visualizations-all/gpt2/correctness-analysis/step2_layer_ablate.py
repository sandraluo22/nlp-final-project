"""Zero-ablate one layer of step 2 at a time to find which layer is doing
the 1->2 work.

Procedure:
  1. Baseline force-decode at step 1 and step 2 (no intervention) on all 1000
     SVAMP. Compute the 1->2 transition stats (wrong->right and right->wrong).
  2. For each layer L in 0..11:
     - Zero-ablate BOTH attn_out and mlp_out at step 2, layer L only.
     - Force-decode at step 2.
     - Compute new w->r and r->w from step 1 baseline to step 2 ablated.
     - Recovery rate vs baseline step-2 predictions.

Output: step2_layer_ablate.json with per-layer flip counts and accuracy delta.
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
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    BS = 16
    OUT_JSON = PD / "step2_layer_ablate.json"

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
    targs = TrainingArguments(output_dir="/tmp/_s2", bf16=True,
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

    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    N_LAYERS = model.codi.config.n_layer
    HID = model.codi.config.n_embd

    CAP = {"mode": "off", "step": -1, "force_decode_step": -1,
           "ablate_layer": -1}  # -1 means no ablation

    def make_attn_hook(idx):
        def fn(_module, _inputs, output):
            if (CAP["mode"] == "ablate" and CAP["step"] == 1
                and idx == CAP["ablate_layer"]):
                a = output[0].clone()
                a[:, -1, :] = 0
                return (a,) + output[1:]
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_module, _inputs, output):
            if (CAP["mode"] == "ablate" and CAP["step"] == 1
                and idx == CAP["ablate_layer"]):
                o = output.clone()
                o[:, -1, :] = 0
                return o
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))

    def clone_past(past):
        return tuple((k.clone(), v.clone()) for k, v in past)

    @torch.no_grad()
    def decode_from_past(past, attn, B, max_new=64):
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
                if done[b]: continue
                tid = int(next_ids[b].item()); tokens[b].append(tid)
                if tid == eos_id: done[b] = True
            if all(done): break
            attn_local = torch.cat([attn_local, torch.ones((B, 1), dtype=attn_local.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    @torch.no_grad()
    def run_batch(qs, ablate_layer=-1):
        """Returns (step1_preds, step2_preds_with_ablation_at_step2_layer_L)."""
        CAP["ablate_layer"] = ablate_layer
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
        # Step 1 (no ablation)
        CAP["mode"] = "off"; CAP["step"] = 0
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        o = model.codi(inputs_embeds=latent, attention_mask=attn,
                       use_cache=True, output_hidden_states=True, past_key_values=past)
        past_after_s1 = o.past_key_values
        latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        # Force-decode after step 1
        s1_strs = decode_from_past(clone_past(past_after_s1), attn, B)

        # Step 2 (ablation if requested)
        CAP["mode"] = "ablate" if ablate_layer >= 0 else "off"
        CAP["step"] = 1
        attn2 = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        o2 = model.codi(inputs_embeds=latent, attention_mask=attn2,
                        use_cache=True, output_hidden_states=True,
                        past_key_values=past_after_s1)
        past_after_s2 = o2.past_key_values
        # Force-decode after step 2 (no ablation during decode)
        CAP["mode"] = "off"; CAP["step"] = -1
        s2_strs = decode_from_past(clone_past(past_after_s2), attn2, B)
        return s1_strs, s2_strs

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    N = len(questions)

    print(f"\nN={N}.  Baseline force-decode at step 1 and step 2...")
    t0 = time.time()
    base_s1 = []
    base_s2 = []
    for s in range(0, N, BS):
        s1, s2 = run_batch(questions[s:s+BS], ablate_layer=-1)
        base_s1 += s1; base_s2 += s2
    base_s1_int = [codi_extract(s) for s in base_s1]
    base_s2_int = [codi_extract(s) for s in base_s2]
    base_s1_correct = np.array([v is not None and abs(v - golds[i]) < 1e-3
                                 for i, v in enumerate(base_s1_int)])
    base_s2_correct = np.array([v is not None and abs(v - golds[i]) < 1e-3
                                 for i, v in enumerate(base_s2_int)])
    print(f"  baseline step-1 acc: {base_s1_correct.mean()*100:.1f}%  "
          f"step-2 acc: {base_s2_correct.mean()*100:.1f}%  "
          f"({time.time()-t0:.0f}s)")
    base_wr = int(((~base_s1_correct) & base_s2_correct).sum())
    base_rw = int((base_s1_correct & (~base_s2_correct)).sum())
    base_net = base_wr - base_rw
    print(f"  baseline 1->2: w->r={base_wr}  r->w={base_rw}  net={base_net:+d}")

    results = {
        "N": N, "baseline_step1_acc": float(base_s1_correct.mean()),
        "baseline_step2_acc": float(base_s2_correct.mean()),
        "baseline_wr": base_wr, "baseline_rw": base_rw, "baseline_net": base_net,
        "per_layer": {},
    }

    for L_ab in range(N_LAYERS):
        print(f"\n--- ablating step-2, layer {L_ab} (attn+mlp -> zero) ---")
        t1 = time.time()
        abl_s1 = []; abl_s2 = []
        for s in range(0, N, BS):
            s1, s2 = run_batch(questions[s:s+BS], ablate_layer=L_ab)
            abl_s1 += s1; abl_s2 += s2
        abl_s2_int = [codi_extract(s) for s in abl_s2]
        abl_s2_correct = np.array([v is not None and abs(v - golds[i]) < 1e-3
                                    for i, v in enumerate(abl_s2_int)])
        wr = int(((~base_s1_correct) & abl_s2_correct).sum())
        rw = int((base_s1_correct & (~abl_s2_correct)).sum())
        net = wr - rw
        n_changed_vs_base_s2 = int(sum(1 for i in range(N) if abl_s2_int[i] != base_s2_int[i]))
        results["per_layer"][L_ab] = {
            "step2_acc": float(abl_s2_correct.mean()),
            "delta_acc_vs_baseline_s2": float(abl_s2_correct.mean() - base_s2_correct.mean()),
            "wr_from_s1": wr, "rw_from_s1": rw, "net": net,
            "delta_net_vs_baseline": net - base_net,
            "n_changed_from_baseline_s2": n_changed_vs_base_s2,
        }
        print(f"  step-2 acc: {abl_s2_correct.mean()*100:.1f}% "
              f"(Δ {abl_s2_correct.mean()*100 - base_s2_correct.mean()*100:+.1f}pp)  "
              f"1->2 w->r={wr}  r->w={rw}  net={net:+d}  "
              f"(Δnet {net - base_net:+d})  changed_vs_base_s2={n_changed_vs_base_s2}  "
              f"({time.time()-t1:.0f}s)")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
