"""GSM8K version of emission-attention patching.

Captures emission attention output at every layer × every emission position
for N=200 GSM8K problems. Pairs via derangement and tests transfer rate at
various (layer set × emission step set) combinations.

Same conditions as the SVAMP version, plus we capture more emission steps
(up to 30) since GSM8K answers can be multi-digit and the "digit emission
step" varies more.
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

SEED = 0
N_LIMIT = 200
K_EMIT = 30
OUT_JSON = PD / "patch_emission_attn_gsm8k.json"


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
    targs = TrainingArguments(output_dir="/tmp/_emg", bf16=True,
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
    N_LAYERS = model.codi.config.n_layer; HID = model.codi.config.n_embd

    CAP = {
        "mode": "off", "phase": "off", "emit_step": -1, "ex_idx": 0,
        "patch_layers": set(), "patch_emit_steps": set(),
        "cap_emit_attn": None, "patch_emit_attn": None,
    }
    def make_attn_hook(idx):
        def fn(_m, _i, output):
            mode = CAP["mode"]
            if mode == "capture_emission" and CAP["phase"] == "emission" and 0 <= CAP["emit_step"] < K_EMIT:
                a = output[0]
                CAP["cap_emit_attn"][CAP["ex_idx"], CAP["emit_step"], idx, :] = a[0, -1, :].detach().to(torch.bfloat16).cpu()
            elif (mode == "patch_emission" and CAP["phase"] == "emission"
                  and 0 <= CAP["emit_step"] < K_EMIT
                  and CAP["emit_step"] in CAP["patch_emit_steps"]
                  and idx in CAP["patch_layers"]):
                a = output[0].clone()
                src = CAP["patch_emit_attn"][CAP["ex_idx"], CAP["emit_step"], idx, :].to(a.device, dtype=a.dtype)
                a[:, -1, :] = src
                return (a,) + output[1:]
            return output
        return fn

    handles = [transformer.h[i].attn.register_forward_hook(make_attn_hook(i))
               for i in range(N_LAYERS)]

    @torch.no_grad()
    def run_one(q):
        CAP["phase"] = "prompt"; CAP["emit_step"] = -1
        batch = tok([q], return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((1, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        CAP["phase"] = "latent"
        for step in range(6):
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            o = model.codi(inputs_embeds=latent, attention_mask=attn,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        CAP["phase"] = "emission"; CAP["emit_step"] = 0
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda")).unsqueeze(0)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
        output = eot_emb; emitted = []
        for _ in range(48):
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past)
            past = s.past_key_values
            nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
            emitted.append(int(nid.item()))
            CAP["emit_step"] += 1
            if emitted[-1] == eos_id: break
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(nid).unsqueeze(1)
        return tok.decode(emitted, skip_special_tokens=True)

    # Pass 1: capture
    CAP["cap_emit_attn"] = torch.zeros((N, K_EMIT, N_LAYERS, HID), dtype=torch.bfloat16)
    CAP["mode"] = "capture_emission"
    t0 = time.time()
    base_strs = []
    for i in range(N):
        CAP["ex_idx"] = i
        base_strs.append(run_one(questions[i]))
    base_ints = [codi_extract(s) for s in base_strs]
    base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                              for i, v in enumerate(base_ints)])
    base_acc = float(base_correct.mean())
    print(f"GSM8K: baseline acc={base_acc:.2f}  capture in {time.time()-t0:.0f}s")

    # Build pairing
    rng = np.random.default_rng(SEED)
    pi = derangement(N, rng)
    CAP["patch_emit_attn"] = CAP["cap_emit_attn"][pi].clone()
    source_ints = [base_ints[pi[i]] for i in range(N)]

    def score(strs, key):
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

    CAP["mode"] = "patch_emission"
    cf_out = {"N": N, "baseline_accuracy": base_acc,
              "K_EMIT": K_EMIT,
              "pairing": pi.tolist(),
              "base_preds": [None if v is None else float(v) for v in base_ints],
              "source_preds": [None if v is None else float(v) for v in source_ints],
              "golds": golds, "conditions": {}}

    def run_variant(layers, emit_steps, key):
        CAP["patch_layers"] = set(layers); CAP["patch_emit_steps"] = set(emit_steps)
        t1 = time.time()
        strs = [run_one(q) for q in questions]
        r = score(strs, key)
        cf_out["conditions"][key] = r
        print(f"  {key:35s} transfer={r['transfer_rate']:.2f}  src={r['n_followed_source']:3d}  "
              f"tgt={r['n_followed_target']:3d}  other={r['n_other']:3d}  unp={r['n_unparseable']:3d}  "
              f"({time.time()-t1:.0f}s)")

    ALL_STEPS = list(range(K_EMIT))
    run_variant([0], ALL_STEPS, "L0_all_emit")
    run_variant([1], ALL_STEPS, "L1_all_emit")
    run_variant([0, 1], ALL_STEPS, "L0_L1_all_emit")
    run_variant(list(range(N_LAYERS)), ALL_STEPS, "ALL_layers_all_emit")
    run_variant(list(range(2, N_LAYERS)), ALL_STEPS, "L2_to_L11_all_emit")

    OUT_JSON.write_text(json.dumps(cf_out, indent=2))
    print(f"saved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
