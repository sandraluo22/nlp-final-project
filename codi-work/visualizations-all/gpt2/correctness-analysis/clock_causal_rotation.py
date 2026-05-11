"""Causal rotation test: if numbers really live on a helix, rotating the
2D projection by 2π·Δ/T should make the model emit an answer corresponding
to operand a shifted by Δ.

Setup:
   1. Train cos/sin probe on residual at (step*, L*) -> [cos(2πa/T), sin(2πa/T)].
      Get 2D probe weights W ∈ ℝ^{768 × 2}.
   2. Build the 768-D "helix subspace" basis Q ∈ ℝ^{768 × 2} via QR of W.
   3. For each example i:
        a. Run CODI normally, capture residual at (step*, L*) just before
           the patch point.
        b. Compute current 2D projection p = X · Q  (shape (2,)).
        c. Rotate p by angle 2π·Δ/T → p'.
        d. Update residual: X' = X + Q @ (p' - p).
        e. Continue forward; read out the answer.
   4. Compare emitted answer to baseline answer and to (a + Δ - b) "target".
      If the model is truly Clock-like, emitted answer should track (a + Δ).

We sweep Δ ∈ {-50, -20, -10, +10, +20, +50}. We do this on vary_numerals
at T=100, step=3, L=11 (the strongest closure cell).

Reports:
   - baseline answer per example
   - rotated answer per Δ
   - fraction that match  (a + Δ - b)  (the Clock prediction)
   - fraction that match  a - b        (no transfer)
   - fraction that match neither       (broken)
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
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
LAT_DIR = REPO / "visualizations-all" / "gpt2" / "counterfactuals"
sys.path.insert(0, str(REPO / "codi"))

OUT_JSON = PD / "clock_causal_rotation.json"

# Use vary_numerals at (step=3, L=11), T=100, which had strong probe alignment.
CF_NAME = "vary_numerals"
TARGET_STEP = 3   # 1-indexed; index 2 in zero-indexed
TARGET_LAYER = 11
PERIOD_T = 100
DELTAS = [-50, -20, -10, +10, +20, +50]


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def fourier_target(n, T):
    return np.stack([np.cos(2 * np.pi * n / T), np.sin(2 * np.pi * n / T)], axis=-1)


def main():
    rows = json.load(open(CF_DIR / f"{CF_NAME}.json"))
    qs = [r["question_concat"].strip().replace("  ", " ") for r in rows]
    golds = np.array([float(r["answer"]) for r in rows])
    a_vals = np.array([float(r["a"]) for r in rows])
    b_vals = np.array([float(r["b"]) for r in rows])
    N = len(qs)

    # ---- Train the cos/sin probe on the captured latent acts. ----
    acts = torch.load(LAT_DIR / f"{CF_NAME}_latent_acts.pt", map_location="cpu",
                      weights_only=True).float().numpy()
    X = acts[:, TARGET_STEP - 1, TARGET_LAYER, :]   # (N, 768)
    Y = fourier_target(a_vals, PERIOD_T)            # (N, 2)
    print(f"training cos/sin probe on residual at step={TARGET_STEP}, L={TARGET_LAYER}, T={PERIOD_T}")
    clf = Ridge(alpha=1.0).fit(X, Y)
    W = clf.coef_.T   # (768, 2)
    # Orthonormalize via QR to get a clean 2D basis Q (the patch subspace).
    Q, R = np.linalg.qr(W)   # Q: (768, 2)
    # The PROBE in the orthonormal basis: pred ≈ X·Q·R + intercept.  When we
    # rotate, we want to act in the orthonormal coords.  For each example:
    #     pred (cos,sin) = X @ W = X @ Q @ R
    # The "natural" 2D coords for rotation are: c = X @ Q, transformed by R.
    # But rotation by angle Δθ is just a 2D rotation in (cos, sin) space; the
    # cleanest way to "rotate the residual" is to (i) compute the current
    # prediction p = X @ W, (ii) compute the target p' = rotate(p, Δθ), and
    # (iii) substitute residual increment that achieves the change in
    # prediction.  Since pred = X @ W, the minimal residual change to update
    # pred by Δp is  Δresid = Δp @ pinv(W).T.  Equivalently:
    #     Δresid = Q @ (Δp ... in Q's basis).
    # Easiest: solve W^T Δresid = Δp for Δresid living in span(Q).  In Q's
    # basis: Δresid = Q @ Q.T @ W @ inv(W.T @ Q @ Q.T @ W) @ Δp.
    # Since Q is orthonormal and W = Q R, we get W.T @ Q = R.T, so the
    # update is Δresid = Q @ inv(R.T) @ Δp.
    R_inv = np.linalg.inv(R.T)        # (2, 2)
    update_basis = Q @ R_inv          # (768, 2): right-multiplied by Δp (2,) gives Δresid (768,)

    # ---- Load CODI, set up hooks. ----
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
    targs = TrainingArguments(output_dir="/tmp/_cr", bf16=True,
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

    W_gpu = torch.from_numpy(W).to("cuda", dtype=torch.float32)
    update_basis_gpu = torch.from_numpy(update_basis).to("cuda", dtype=torch.float32)

    # CAP controls hook behavior.
    CAP = {
        "phase": "off",
        "delta_theta": 0.0,
        "patched_pred_a": [None],
        "patched_pred_a_after": [None],
    }

    def make_block_hook(idx):
        def fn(_m, _i, output):
            if CAP["phase"] != "patching" or idx != TARGET_LAYER:
                return output
            h = output[0] if isinstance(output, tuple) else output
            # h shape: (1, T_new, hidden).  Patch only the last position (the
            # newly produced latent token at step TARGET_STEP).
            last = h[:, -1, :].float()    # (1, 768)
            # Current prediction (cos, sin):
            p = last @ W_gpu              # (1, 2)
            # Rotation matrix
            c = np.cos(CAP["delta_theta"]); s = np.sin(CAP["delta_theta"])
            R_rot = torch.tensor([[c, -s], [s, c]], device="cuda", dtype=torch.float32)
            p_new = p @ R_rot.T           # (1, 2)
            dp = (p_new - p).squeeze(0)   # (2,)
            dres = update_basis_gpu @ dp  # (768,)
            h_new = h.clone()
            h_new[:, -1, :] = h[:, -1, :].to(torch.float32) + dres
            h_new = h_new.to(h.dtype)
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        return fn

    handles = [transformer.h[idx].register_forward_hook(make_block_hook(idx))
               for idx in range(N_LAYERS)]

    @torch.no_grad()
    def run_one(q):
        batch = tok([q], return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((1, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        # Step-by-step: patch only at TARGET_STEP, layer TARGET_LAYER.
        for step in range(6):
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            CAP["phase"] = "patching" if (step + 1) == TARGET_STEP else "off"
            o = model.codi(inputs_embeds=latent, attention_mask=attn,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        CAP["phase"] = "off"
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda")).unsqueeze(0)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
        output = eot_emb
        tokens = []
        for _ in range(48):
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past)
            past = s.past_key_values
            nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
            tokens.append(int(nid.item()))
            if tokens[-1] == eos_id: break
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(nid).unsqueeze(1)
        return tok.decode(tokens, skip_special_tokens=True)

    # ---- Baseline: no rotation (Δ=0) ----
    print("\nBaseline (Δ=0)")
    CAP["delta_theta"] = 0.0
    base_strs = []
    for q in qs:
        base_strs.append(run_one(q))
    base_ints = [codi_extract(s) for s in base_strs]
    base_correct = sum(1 for i, v in enumerate(base_ints)
                       if v is not None and abs(v - golds[i]) < 1e-3)
    print(f"  baseline acc = {base_correct/N:.2f}")

    # ---- Sweep deltas ----
    results = {
        "cf_name": CF_NAME, "target_step_1indexed": TARGET_STEP,
        "target_layer": TARGET_LAYER, "period_T": PERIOD_T,
        "N": N, "deltas": DELTAS, "baseline_strs": base_strs,
        "baseline_ints": [None if v is None else float(v) for v in base_ints],
        "a": a_vals.tolist(), "b": b_vals.tolist(), "gold": golds.tolist(),
        "per_delta": {},
    }
    for delta in DELTAS:
        # Rotation angle: 2π · delta / T
        theta = 2 * np.pi * delta / PERIOD_T
        CAP["delta_theta"] = theta
        print(f"\nΔ={delta:+d}  (θ={theta:+.3f} rad)")
        strs = []
        for q in qs:
            strs.append(run_one(q))
        ints = [codi_extract(s) for s in strs]
        # Clock predictions: emit answer matches (a + Δ - b)?
        clock_targets = a_vals + delta - b_vals
        eq = lambda x, y: x is not None and abs(x - y) < 1e-3
        n_clock = sum(1 for i, v in enumerate(ints) if eq(v, clock_targets[i]))
        n_base = sum(1 for i, v in enumerate(ints) if eq(v, base_ints[i]) if base_ints[i] is not None)
        n_change_to_a_minus_b_plus_delta = sum(1 for i, v in enumerate(ints) if eq(v, a_vals[i] + delta - b_vals[i]))
        n_unparseable = sum(1 for v in ints if v is None)
        print(f"  followed Clock target (a+Δ-b): {n_clock}/{N}")
        print(f"  followed baseline (no change): {n_base}/{N}")
        print(f"  unparseable: {n_unparseable}/{N}")
        # Show a few examples
        for i in range(5):
            print(f"    ex{i}: a={a_vals[i]:.0f} b={b_vals[i]:.0f} gold={golds[i]:.0f} "
                  f"base={base_ints[i]} rotated={ints[i]}  "
                  f"clock_target={clock_targets[i]:.0f}")
        results["per_delta"][str(delta)] = {
            "delta": delta, "theta": theta,
            "n_followed_clock_target": n_clock,
            "n_followed_baseline": n_base,
            "n_unparseable": n_unparseable,
            "outputs_int": [None if v is None else float(v) for v in ints],
            "strs": strs,
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
