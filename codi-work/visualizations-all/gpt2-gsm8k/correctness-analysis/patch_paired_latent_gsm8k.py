"""GSM8K paired-CF latent patching.

Replicates patch_paired_perex.py on GSM8K. Key question: on real multi-step
problems where the prompt KV does NOT linearly encode the gold answer,
does patching the latent loop's residuals between paired examples now
transfer the answer (as it would if latents are doing real work)?

Tests (matching the SVAMP version's sweep D):
   - patch all 12 layers' residual stream at step 1
   - patch all 12 layers at steps 1+2, 1+2+3, ..., 1..6
   - plus: patch step 1's L0 resid only (most disruptive on SVAMP)
   - plus: patch every layer's resid at every step (the strongest possible).

If on GSM8K we see real transfer (or large disruption) where on SVAMP we
saw 0%, that's evidence the latents are doing computation on GSM8K that
isn't redundant with prompt KV.
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

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "codi"))

N_LIMIT = 200
SEED = 0
OUT_JSON = PD / "patch_paired_latent_gsm8k.json"


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def derangement(n, rng, max_tries=100):
    for _ in range(max_tries):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p
    return np.r_[np.arange(1, n), 0]


def main():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    questions, golds = [], []
    for ex in ds:
        m = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if m is None: continue
        questions.append(ex["question"].strip().replace("  ", " "))
        golds.append(float(m.group(1)))
    questions = questions[:N_LIMIT]; golds = golds[:N_LIMIT]
    N = len(questions); golds_arr = np.array(golds)

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
    targs = TrainingArguments(output_dir="/tmp/_plg", bf16=True,
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
    model = model.to("cuda").to(torch.bfloat16); model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id

    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    N_LAYERS = model.codi.config.n_layer; HID = model.codi.config.n_embd; N_LAT = 6

    CAP = {
        "mode": "off", "step": -1, "ex_idx": 0,
        "patch_steps": set(), "patch_layers": set(),
        "cap_resid": None, "patch_resid": None,
    }

    def make_block_hook(idx):
        def fn(_m, _i, output):
            m = CAP["mode"]
            if m == "capture" and CAP["step"] >= 0:
                h = output[0] if isinstance(output, tuple) else output
                CAP["cap_resid"][CAP["ex_idx"], CAP["step"], idx, :] = (
                    h[0, -1, :].detach().to(torch.bfloat16).cpu())
            elif (m == "patch" and CAP["step"] in CAP["patch_steps"]
                  and idx in CAP["patch_layers"]):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                src = CAP["patch_resid"][CAP["ex_idx"], CAP["step"], idx, :].to(
                    h.device, dtype=h.dtype)
                h[:, -1, :] = src
                if isinstance(output, tuple): return (h,) + output[1:]
                return h
            return output
        return fn

    handles = [transformer.h[i].register_forward_hook(make_block_hook(i))
               for i in range(N_LAYERS)]

    @torch.no_grad()
    def run_one(q, ex_idx):
        CAP["ex_idx"] = ex_idx
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
        output = eot_emb; emitted = []
        for _ in range(48):
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past)
            past = s.past_key_values
            nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
            emitted.append(int(nid.item()))
            if emitted[-1] == eos_id: break
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(nid).unsqueeze(1)
        return tok.decode(emitted, skip_special_tokens=True)

    # Capture
    CAP["cap_resid"] = torch.zeros((N, N_LAT, N_LAYERS, HID), dtype=torch.bfloat16)
    CAP["mode"] = "capture"
    t0 = time.time()
    base_strs = [run_one(q, i) for i, q in enumerate(questions)]
    base_ints = [codi_extract(s) for s in base_strs]
    base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                              for i, v in enumerate(base_ints)])
    base_acc = float(base_correct.mean())
    print(f"baseline acc={base_acc:.2f}  capture in {time.time()-t0:.0f}s")

    rng = np.random.default_rng(SEED)
    pi = derangement(N, rng)
    CAP["patch_resid"] = CAP["cap_resid"][pi].clone()
    source_ints = [base_ints[pi[i]] for i in range(N)]

    def score(strs):
        ints = [codi_extract(s) for s in strs]
        n_src = n_tgt = n_oth = n_unp = 0
        eq = lambda x, y: x is not None and y is not None and abs(x - y) < 1e-3
        for i in range(N):
            v = ints[i]; si = source_ints[i]; ti = base_ints[i]
            if v is None: n_unp += 1
            elif eq(v, si): n_src += 1
            elif eq(v, ti): n_tgt += 1
            else: n_oth += 1
        return {"transfer_rate": n_src / N, "n_followed_source": n_src,
                "n_followed_target": n_tgt, "n_other": n_oth, "n_unparseable": n_unp}

    CAP["mode"] = "patch"
    cf_out = {"N": N, "baseline_accuracy": base_acc, "pairing": pi.tolist(),
              "base_preds": [None if v is None else float(v) for v in base_ints],
              "source_preds": [None if v is None else float(v) for v in source_ints],
              "golds": golds, "conditions": {}}

    def run_variant(steps, layers, key):
        CAP["patch_steps"] = set(steps); CAP["patch_layers"] = set(layers)
        t1 = time.time()
        strs = [run_one(q, i) for i, q in enumerate(questions)]
        r = score(strs); cf_out["conditions"][key] = r
        print(f"  {key:35s} transfer={r['transfer_rate']:.2f}  src={r['n_followed_source']:3d}  "
              f"tgt={r['n_followed_target']:3d}  other={r['n_other']:3d}  unp={r['n_unparseable']:3d}  "
              f"({time.time()-t1:.0f}s)")

    L_ALL = list(range(N_LAYERS))
    for K in range(1, N_LAT + 1):
        run_variant(list(range(K)), L_ALL, f"steps_1to{K}_all_layers")
    run_variant([0], [0], "step1_L0_only")
    run_variant(list(range(N_LAT)), L_ALL, "ALL_steps_ALL_layers")

    OUT_JSON.write_text(json.dumps(cf_out, indent=2))
    print(f"saved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
