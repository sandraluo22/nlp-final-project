"""Patch attention block output AT THE EMISSION POSITION at specific layers.

Earlier results: paired-CF residual patching during the latent loop has
zero transfer. The attention-at-emission analysis showed that operand-
reading is concentrated at L0.h5 and L1.h7, and latent-reading at L0.h0-
h11. These heads fire at the emission position (the first token after EOT,
where the answer is decoded). This script tests:

  If we replace what THE ATTENTION BLOCK at L0/L1 wrote into the residual
  stream at the emission position with B's contribution, does the answer
  transfer from A to B?

This is a much narrower intervention than:
  - prompt switching (changes every head at every position).
  - paired-CF on latent residuals (zero transfer; we already tested).

If transfer happens, the operand-reading mechanism is essentially these
early-layer heads at the emission position. If not, the operand info has
been baked into the residual stream PRIOR to emission via the latent loop's
own attention reads, and patching emission alone can't move it.

Three variants per CF set:
   A) patch L0 attn output at emission position with B's.
   B) patch L1 attn output at emission position with B's.
   C) patch L0 + L1 attn outputs at emission position with B's.

Output: patch_emission_attn.{json}.
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
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

CF_SETS = ["vary_numerals", "vary_both_2digit"]
SEED = 0
OUT_JSON = PD / "patch_emission_attn.json"


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def load_cf(name):
    rows = json.load(open(CF_DIR / f"{name}.json"))
    qs = [r["question_concat"].strip().replace("  ", " ") for r in rows]
    golds = [float(r["answer"]) for r in rows]
    return qs, golds


def derangement(n, rng, max_tries=100):
    for _ in range(max_tries):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p
    return np.r_[np.arange(1, n), 0]


def main():
    BS = 1
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
    targs = TrainingArguments(output_dir="/tmp/_em", bf16=True,
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
    N_LAYERS = model.codi.config.n_layer
    HID = model.codi.config.n_embd

    K_EMIT = 16   # capture/patch up to 16 emission positions
    CAP = {
        "mode": "off",            # "capture_emission" | "patch_emission"
        "phase": "prompt",        # "prompt" | "latent" | "emission"
        "emit_step": -1,          # index within the emission sequence (0..K_EMIT-1)
        "ex_idx": 0,
        "patch_layers": set(),
        "patch_emit_steps": set(),  # which emission steps to patch (set of int)
        "cap_emit_attn": None,    # (N, K_EMIT, N_LAYERS, HID) bf16 CPU
        "patch_emit_attn": None,
    }

    def make_attn_hook(idx):
        def fn(_m, _i, output):
            mode = CAP["mode"]
            if (mode == "capture_emission" and CAP["phase"] == "emission"
                  and 0 <= CAP["emit_step"] < K_EMIT):
                a = output[0]
                CAP["cap_emit_attn"][CAP["ex_idx"], CAP["emit_step"], idx, :] = (
                    a[0, -1, :].detach().to(torch.bfloat16).cpu())
            elif (mode == "patch_emission" and CAP["phase"] == "emission"
                  and 0 <= CAP["emit_step"] < K_EMIT
                  and CAP["emit_step"] in CAP["patch_emit_steps"]
                  and idx in CAP["patch_layers"]):
                a = output[0].clone()
                src = CAP["patch_emit_attn"][CAP["ex_idx"], CAP["emit_step"], idx, :].to(
                    a.device, dtype=a.dtype)
                a[:, -1, :] = src
                return (a,) + output[1:]
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))

    @torch.no_grad()
    def run_one(q, ex_idx):
        """One pass through CODI; hooks fire based on CAP['phase']."""
        CAP["ex_idx"] = ex_idx
        CAP["phase"] = "prompt"
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
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda")).unsqueeze(0)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
        # Emission phase. emit_step counts up across all emission forward calls.
        CAP["phase"] = "emission"
        CAP["emit_step"] = 0
        s = model.codi(inputs_embeds=eot_emb, attention_mask=attn,
                       use_cache=True, past_key_values=past)
        past = s.past_key_values
        nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
        emitted = [int(nid.item())]
        for _ in range(48):
            if emitted[-1] == eos_id: break
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(nid).unsqueeze(1)
            CAP["emit_step"] += 1
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past)
            past = s.past_key_values
            nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
            emitted.append(int(nid.item()))
        CAP["emit_step"] = -1
        return tok.decode(emitted, skip_special_tokens=True)

    results = {}
    for cf_name in CF_SETS:
        print(f"\n=== {cf_name} ===", flush=True)
        qs, golds = load_cf(cf_name)
        N = len(qs)
        golds_arr = np.array(golds)

        # PASS 1: capture each example's emission attn output at every (step, layer).
        CAP["cap_emit_attn"] = torch.zeros((N, K_EMIT, N_LAYERS, HID), dtype=torch.bfloat16)
        CAP["mode"] = "capture_emission"
        t0 = time.time()
        base_strs = []
        for i in range(N):
            base_strs.append(run_one(qs[i], i))
        base_ints = [codi_extract(s) for s in base_strs]
        base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                  for i, v in enumerate(base_ints)])
        base_acc = float(base_correct.mean())
        print(f"  N={N}, baseline acc={base_acc:.2f}  (capture in {time.time()-t0:.0f}s)")

        # Build pairing: derangement, prefer A.pred != B.pred.
        rng = np.random.default_rng(SEED)
        for _ in range(50):
            pi = derangement(N, rng)
            coll = sum(1 for i in range(N) if base_ints[i] is not None
                       and base_ints[pi[i]] is not None
                       and base_ints[i] == base_ints[pi[i]])
            if coll <= N // 20: break
        CAP["patch_emit_attn"] = CAP["cap_emit_attn"][pi].clone()
        source_ints = [base_ints[pi[i]] for i in range(N)]
        print(f"  pairing derangement seed={SEED}, prediction-collisions={coll}/{N}")

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
            return {
                "key": key,
                "n_followed_source": n_src, "n_followed_target": n_tgt,
                "n_other": n_oth, "n_unparseable": n_unp,
                "transfer_rate": n_src / N,
            }

        CAP["mode"] = "patch_emission"
        cf_out = {"N": N, "baseline_accuracy": base_acc,
                  "K_EMIT": K_EMIT,
                  "pairing": pi.tolist(),
                  "base_preds": [None if v is None else float(v) for v in base_ints],
                  "source_preds": [None if v is None else float(v) for v in source_ints],
                  "golds": golds, "conditions": {}}

        def run_variant(layers, emit_steps, key):
            CAP["patch_layers"] = set(layers)
            CAP["patch_emit_steps"] = set(emit_steps)
            t1 = time.time()
            strs = [run_one(qs[i], i) for i in range(N)]
            r = score(strs, key)
            cf_out["conditions"][key] = r
            print(f"  {key:40s} transfer={r['transfer_rate']:.2f}  "
                  f"src={r['n_followed_source']:3d}  tgt={r['n_followed_target']:3d}  "
                  f"other={r['n_other']:3d}  unp={r['n_unparseable']:3d}  "
                  f"({time.time()-t1:.0f}s)")

        ALL_STEPS = list(range(K_EMIT))
        # First-emission only (the previous, weak intervention) — kept for
        # comparison.
        run_variant([0], [0], "L0_emit0_only")
        run_variant([1], [0], "L1_emit0_only")
        # Every emission step, single early layer.
        run_variant([0], ALL_STEPS, "L0_all_emit")
        run_variant([1], ALL_STEPS, "L1_all_emit")
        run_variant([0, 1], ALL_STEPS, "L0_L1_all_emit")
        # Every layer at every emission step.
        run_variant(list(range(N_LAYERS)), ALL_STEPS, "ALL_layers_all_emit")
        # Late-layer only (no L0/L1) — the heads that mostly attend to template.
        run_variant(list(range(2, N_LAYERS)), ALL_STEPS, "L2_to_L11_all_emit")
        # Only the explicit numeric-emission step (typically emit step 4-5, the
        # actual digit).  We'll mark it specifically.
        run_variant(list(range(N_LAYERS)), [3, 4, 5], "ALL_layers_emit3to5")

        results[cf_name] = cf_out

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
