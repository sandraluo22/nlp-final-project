"""What are the latent steps doing?

Compares four ablation modes for step 1's residual stream output (at every
layer's block output, at the last-token position of step 1):
   zero        : replace with zeros
   mean        : replace with the mean residual across examples
   random      : replace with N(0, σ²) of matched per-dim variance
   paired_cf   : replace with a different example's residual (derangement)

For each mode we record:
   - accuracy and prediction distribution
   - emission attention pattern at emit step 4 (the digit-emission step)

If all four modes produce the same accuracy as the paired_cf null (~baseline),
the latents are pure structural padding — any on-manifold filling works.
If only paired_cf is harmless, the latent CONTENT carries information
redundant with the prompt but the model still routes through latent
positions.
If only zero crashes (other three preserved), latents need to be non-degenerate
but content doesn't matter.

Also captures emission attention over (template, operand_a, operand_b, bot,
latent_1..6, eot) at emit step 4 to see if the pattern shifts under ablation.
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
OUT_JSON = PD / "latent_ablation_modes_gsm8k.json"

ABLATE_STEP = 0   # zero-indexed; ablate step 1
EMIT_TARGET = 4   # emit step where digit is produced ("The answer is: <DIGIT>")


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
    return qs, golds, rows


def derangement(n, rng, max_tries=100):
    for _ in range(max_tries):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p
    return np.r_[np.arange(1, n), 0]


def find_operand_positions(input_ids, tok, a_value, b_value):
    text_ids = input_ids.tolist()
    def find_sub(needle):
        out = []
        for s in range(len(text_ids) - len(needle) + 1):
            if text_ids[s:s+len(needle)] == needle:
                out.append(list(range(s, s+len(needle))))
        return out
    candidates_a, candidates_b = [], []
    for prefix in (" ", ""):
        ids_a = tok.encode(f"{prefix}{int(a_value)}", add_special_tokens=False)
        ids_b = tok.encode(f"{prefix}{int(b_value)}", add_special_tokens=False)
        if (m := find_sub(ids_a)): candidates_a.extend(m)
        if (m := find_sub(ids_b)): candidates_b.extend(m)
    pos_a = candidates_a[0] if candidates_a else []
    pos_b = []
    for cb in candidates_b:
        if not set(cb) & set(pos_a):
            pos_b = cb; break
    if not pos_b and candidates_b:
        pos_b = candidates_b[0]
    return pos_a, pos_b


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
    targs = TrainingArguments(output_dir="/tmp/_lm", bf16=True,
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
    N_HEADS = model.codi.config.n_head
    HID = model.codi.config.n_embd

    CAP = {
        "phase": "off",            # "prompt" | "latent" | "emission"
        "step": -1,                # current latent step
        "emit_step": -1,
        "mode": "baseline",        # "baseline" | "capture" | "patch"
        "ablation": "none",        # "none" | "zero" | "mean" | "random" | "paired_cf"
        "cap_resid": None,         # (N, N_LAYERS, HID) on CPU for step ABLATE_STEP
        "patch_value": None,       # (N_LAYERS, HID) on GPU — what to replace with
        "ex_idx": 0,
        "emit_attn": None,         # (N_LAYERS, N_HEADS, prefix_len) for emit step 4
    }

    def make_block_hook(idx):
        def fn(_m, _i, output):
            if CAP["mode"] == "capture" and CAP["phase"] == "latent" \
                    and CAP["step"] == ABLATE_STEP:
                h = output[0] if isinstance(output, tuple) else output
                CAP["cap_resid"][CAP["ex_idx"], idx, :] = h[0, -1, :].detach().to(
                    torch.bfloat16).cpu()
            elif CAP["mode"] == "patch" and CAP["phase"] == "latent" \
                    and CAP["step"] == ABLATE_STEP and CAP["patch_value"] is not None:
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                v = CAP["patch_value"][idx].to(h.device, dtype=h.dtype)
                h[:, -1, :] = v
                if isinstance(output, tuple): return (h,) + output[1:]
                return h
            return output
        return fn

    handles = [transformer.h[idx].register_forward_hook(make_block_hook(idx))
               for idx in range(N_LAYERS)]

    @torch.no_grad()
    def run_one(q, capture_attn_at_emit=-1, return_attn=False):
        CAP["phase"] = "prompt"; CAP["step"] = -1; CAP["emit_step"] = -1
        batch = tok([q], return_tensors="pt", padding="longest").to("cuda")
        prompt_ids = batch["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])
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
            CAP["step"] = step
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            o = model.codi(inputs_embeds=latent, attention_mask=attn,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        CAP["phase"] = "emission"; CAP["step"] = -1; CAP["emit_step"] = 0
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda")).unsqueeze(0)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
        output = eot_emb
        captured_attn = None
        emitted = []
        for emit_i in range(16):
            # Capture attn only at the target emit step.
            wants_attn = (emit_i == capture_attn_at_emit) and return_attn
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past,
                           output_attentions=wants_attn)
            past = s.past_key_values
            if wants_attn:
                # attentions: tuple of (1, n_heads, q_len=1, k_len) per layer
                arr = np.zeros((N_LAYERS, N_HEADS, s.attentions[0].shape[-1]),
                               dtype=np.float32)
                for li, a in enumerate(s.attentions):
                    arr[li] = a[0, :, 0, :].float().cpu().numpy()
                captured_attn = arr
            nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
            emitted.append(int(nid.item()))
            CAP["emit_step"] += 1
            if emitted[-1] == eos_id: break
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(nid).unsqueeze(1)
        decoded = tok.decode(emitted, skip_special_tokens=True)
        return decoded, prompt_len, captured_attn

    results = {}
    for cf_name in CF_SETS:
        print(f"\n=== {cf_name} ===", flush=True)
        qs, golds, rows = load_cf(cf_name)
        N = len(qs); golds_arr = np.array(golds)

        # PASS 1: capture all examples' step-1 residual at every layer.
        CAP["mode"] = "capture"
        CAP["cap_resid"] = torch.zeros((N, N_LAYERS, HID), dtype=torch.bfloat16)
        t0 = time.time()
        base_strs, base_attns, prompt_lens, op_positions = [], {}, [], []
        for i in range(N):
            CAP["ex_idx"] = i
            wants_emit_attn = i < 30  # capture emission attn for the first 30 only
            decoded, plen, attn_arr = run_one(qs[i],
                                              capture_attn_at_emit=EMIT_TARGET if wants_emit_attn else -1,
                                              return_attn=wants_emit_attn)
            base_strs.append(decoded)
            prompt_lens.append(plen)
            if wants_emit_attn and attn_arr is not None:
                base_attns[i] = attn_arr
            a_val = rows[i].get("a"); b_val = rows[i].get("b")
            pa, pb = find_operand_positions(
                tok([qs[i]], return_tensors="pt")["input_ids"][0], tok, a_val, b_val)
            op_positions.append({"pos_a": pa, "pos_b": pb, "prompt_len": plen})
        base_ints = [codi_extract(s) for s in base_strs]
        base_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                  for i, v in enumerate(base_ints)])
        base_acc = float(base_correct.mean())
        print(f"  baseline acc={base_acc:.2f}  (capture in {time.time()-t0:.0f}s)")

        # Build pairing for paired_cf.
        rng = np.random.default_rng(SEED)
        pi = derangement(N, rng)
        # Build mean residual at step 1 across examples.
        cap_resid_np = CAP["cap_resid"].float().numpy()
        mean_resid = torch.from_numpy(cap_resid_np.mean(axis=0)).to(
            torch.bfloat16).to("cuda")          # (N_LAYERS, HID)
        zero_resid = torch.zeros((N_LAYERS, HID), dtype=torch.bfloat16, device="cuda")

        # PASS 2: each ablation mode.
        cf_out = {"N": N, "baseline_accuracy": base_acc,
                  "base_strs": base_strs, "base_ints": [None if v is None else float(v) for v in base_ints],
                  "op_positions": op_positions,
                  "pairing": pi.tolist(),
                  "modes": {}}
        for mode in ["zero", "mean", "random", "paired_cf"]:
            CAP["mode"] = "patch"
            ablate_strs = []
            ablate_attns = {}
            t1 = time.time()
            for i in range(N):
                CAP["ex_idx"] = i
                # Build patch_value per example
                if mode == "zero":
                    CAP["patch_value"] = zero_resid
                elif mode == "mean":
                    CAP["patch_value"] = mean_resid
                elif mode == "random":
                    # Random Gaussian with same per-dim std as the captured residual.
                    std = torch.from_numpy(cap_resid_np.std(axis=0)).float()  # (N_LAYERS, HID)
                    g = torch.randn((N_LAYERS, HID), generator=torch.Generator().manual_seed(SEED + i)) * std
                    CAP["patch_value"] = g.to(torch.bfloat16).to("cuda")
                elif mode == "paired_cf":
                    src = CAP["cap_resid"][pi[i]]   # (N_LAYERS, HID)
                    CAP["patch_value"] = src.to("cuda")
                wants_emit_attn = i < 30
                decoded, _, attn_arr = run_one(qs[i],
                                               capture_attn_at_emit=EMIT_TARGET if wants_emit_attn else -1,
                                               return_attn=wants_emit_attn)
                ablate_strs.append(decoded)
                if wants_emit_attn and attn_arr is not None:
                    ablate_attns[i] = attn_arr
            ablate_ints = [codi_extract(s) for s in ablate_strs]
            ablate_correct = np.array([v is not None and abs(v - golds_arr[i]) < 1e-3
                                       for i, v in enumerate(ablate_ints)])
            ablate_acc = float(ablate_correct.mean())
            # Compare to baseline predictions
            n_same_pred = sum(1 for i in range(N)
                              if ablate_ints[i] is not None and base_ints[i] is not None
                              and abs(ablate_ints[i] - base_ints[i]) < 1e-3)
            n_unparseable = sum(1 for v in ablate_ints if v is None)
            # Attention shift: average absolute difference between baseline and
            # ablated attention pattern, per layer averaged over heads, summed
            # over prefix positions.  Lower = pattern preserved; higher = shifted.
            attn_shifts = []
            for i in base_attns:
                if i not in ablate_attns: continue
                ba, aa = base_attns[i], ablate_attns[i]
                # truncate to common length
                kl = min(ba.shape[-1], aa.shape[-1])
                diff = np.abs(ba[..., :kl] - aa[..., :kl]).sum(axis=-1).mean(axis=-1)  # (N_LAYERS,)
                attn_shifts.append(diff)
            attn_shift_per_layer = np.stack(attn_shifts).mean(axis=0).tolist() if attn_shifts else []
            cf_out["modes"][mode] = {
                "accuracy": ablate_acc, "n_same_as_baseline": n_same_pred,
                "n_unparseable": n_unparseable,
                "ablate_ints": [None if v is None else float(v) for v in ablate_ints],
                "attn_shift_per_layer": attn_shift_per_layer,
            }
            print(f"  {mode:11s}: acc={ablate_acc:.2f}  same-as-base={n_same_pred}/{N}  "
                  f"unparseable={n_unparseable}  attn_shift_mean={np.mean(attn_shift_per_layer):.3f}  "
                  f"({time.time()-t1:.0f}s)")

        results[cf_name] = cf_out

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
