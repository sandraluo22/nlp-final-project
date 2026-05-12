"""Causal test of the correctness direction at (latent step 2, layer 6).

Falsification setup:
- Fit logreg probe on activations[:, step=1, layer=6, :] -> student_correct
  using the same 80/20 split as correctness_probe.py.
- Take direction v = sigma * w (where sigma is StandardScaler scale_ to put
  the unit-norm probe weight back into raw-residual space).
- For each alpha in {-16, -8, -4, -2, -1, +1, +2, +4, +8, +16}, hook the
  GPT-2 student forward pass at block layer=6 (HOOK["layer"]=6 -> block_idx=5)
  during latent step p_target=1 (0-indexed = step 2 in 1-indexed display) and
  add alpha * v to the last-token residual. Then let the model finish the
  latent loop, emit EOT, and decode the answer.

Outcomes measured (vs baseline / alpha=0):
- n_changed_int: how many of N_eval examples changed their parsed answer.
- n_correct_after: matches gold.
- delta_correct = n_correct_after - n_correct_base.
- For originally-wrong: how many flipped to correct (+).
- For originally-right: how many flipped to wrong (-).

Baselines:
- random_unit: same-norm random Gaussian direction.
- permuted_labels: refit probe on shuffled student_correct.

If the correctness direction is *causal*, +alpha should make originally-wrong
examples flip to correct (and -alpha vice versa), more than the random and
permuted baselines.

If steering produces no controlled flips (vs baselines), the direction is an
*effect* of successful computation, not its cause.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
STUDENT_RES = REPO / "inference" / "runs" / "svamp_student_gpt2" / "results.json"

# Best (step, layer) from correctness_probe.py (1-indexed step in display).
# step 2 1-indexed -> 0-indexed s=1; layer 6 (out of 13 layers 0..12).
TARGET_STEP_0IDX = 1
TARGET_LAYER = 6


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def fit_correctness_probe(acts_cell, y, seed=0):
    """Return direction v (length H) in raw-residual space."""
    idx_tr, _ = train_test_split(np.arange(len(y)), test_size=0.2,
                                 random_state=seed, stratify=y)
    sc = StandardScaler().fit(acts_cell[idx_tr])
    Xtr = sc.transform(acts_cell[idx_tr])
    clf = LogisticRegression(max_iter=2000, C=0.1,
                             solver="lbfgs").fit(Xtr, y[idx_tr])
    # w is logistic regression coef in scaled space.
    # In raw space, adding v to x is equivalent to adding scale_ * w / 1 in
    # scaled space. So v_raw = scale_ * w yields a 1-unit-in-scaled-space step.
    sigma = sc.scale_
    w = clf.coef_[0]
    v = sigma * w
    v = v / np.linalg.norm(v) * np.linalg.norm(sigma)  # match scale_ norm
    return torch.tensor(v, dtype=torch.float32)


def random_direction(H, sigma, rng):
    g = rng.standard_normal(H)
    g = g / np.linalg.norm(g) * np.linalg.norm(sigma)
    return torch.tensor(g, dtype=torch.float32)


def main():
    print(f"loading activations {ACTS}", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    print(f"  shape={a.shape}")

    student = json.load(open(STUDENT_RES))
    y = np.array([bool(s["correct"]) for s in student], dtype=int)
    assert len(y) == N

    # Direction: correctness probe at (TARGET_STEP_0IDX, TARGET_LAYER).
    Xcell = a[:, TARGET_STEP_0IDX, TARGET_LAYER, :]
    print(f"fitting correctness probe on step={TARGET_STEP_0IDX+1}, layer={TARGET_LAYER}")
    v_corr = fit_correctness_probe(Xcell, y, seed=0)
    print(f"  direction norm: {float(torch.linalg.vector_norm(v_corr)):.3f}")

    sigma_cell = StandardScaler().fit(Xcell).scale_
    rng = np.random.default_rng(0)
    v_rand = random_direction(H, sigma_cell, rng)

    # Permuted-labels baseline: refit on shuffled y.
    y_perm = y.copy(); rng.shuffle(y_perm)
    v_perm = fit_correctness_probe(Xcell, y_perm, seed=0)
    print(f"  permuted-label probe norm: {float(torch.linalg.vector_norm(v_perm)):.3f}")

    # ---- Load CODI-GPT-2 ----
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"\nloading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *args, **k: _orig(*args, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=["c_attn", "c_proj", "c_fc"],
                          init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_sc", bf16=True,
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
    HOOK = {"step": -1, "active": False, "vec": None, "p_target": None,
            "layer": None, "alpha": 0.0}

    def make_hook(block_idx):
        def fn(_module, _inputs, output):
            if not HOOK["active"] or HOOK["layer"] is None: return output
            if block_idx != HOOK["layer"] - 1: return output
            if HOOK["step"] != HOOK["p_target"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i))
               for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, p_target, layer, vec, alpha, max_new=96):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"vec": vec, "p_target": p_target, "layer": layer,
                     "alpha": alpha, "active": True, "step": -1})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)

        for step in range(targs.inf_latent_iterations):
            HOOK["step"] = step
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)

        # Decode: hook is inactive during decode (step != p_target after loop).
        HOOK["step"] = -1
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_new):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
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
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK.update({"active": False, "step": -1})
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    # ---- Eval set ----
    ds = load_dataset("gsm8k")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    N_eval = N
    eval_qs = questions
    eval_gold = golds
    print(f"\nN_eval = {N_eval}")

    BS = 16

    def run_full(p, l, vec, alpha):
        strs = []
        for s in range(0, N_eval, BS):
            strs += run_batch(eval_qs[s:s+BS], p_target=p, layer=l,
                              vec=vec, alpha=alpha)
        return strs

    # Baseline
    print("\n=== Baseline (alpha=0) ===", flush=True)
    base_strs = run_full(0, 0, torch.zeros(H), 0.0)
    base_int = [codi_extract(s) for s in base_strs]
    base_correct = np.array([v is not None and abs(v - eval_gold[i]) < 1e-3
                              for i, v in enumerate(base_int)])
    base_acc = float(base_correct.mean())
    print(f"  baseline accuracy: {base_acc*100:.1f}% ({base_correct.sum()}/{N_eval})")

    summary = {
        "target_step_1indexed": TARGET_STEP_0IDX + 1,
        "target_layer": TARGET_LAYER,
        "N_eval": N_eval,
        "baseline_accuracy": base_acc,
        "baseline_n_correct": int(base_correct.sum()),
        "alphas": [],
        "directions": ["correctness", "permuted", "random"],
    }

    alphas = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
    p, l = TARGET_STEP_0IDX, TARGET_LAYER
    for dname, vec in [("correctness", v_corr), ("permuted", v_perm), ("random", v_rand)]:
        print(f"\n=== Direction: {dname} ===", flush=True)
        for a in alphas:
            strs = run_full(p, l, vec, float(a))
            ints = [codi_extract(s) for s in strs]
            correct = np.array([v is not None and abs(v - eval_gold[i]) < 1e-3
                                 for i, v in enumerate(ints)])
            n_changed = int(sum(1 for i in range(N_eval) if ints[i] != base_int[i]))
            n_correct = int(correct.sum())
            wrong_to_right = int(((~base_correct) & correct).sum())
            right_to_wrong = int((base_correct & ~correct).sum())
            print(f"  alpha={a:+d}:  acc={n_correct/N_eval*100:5.1f}%  "
                  f"changed={n_changed}/{N_eval}  w->r={wrong_to_right:3d}  r->w={right_to_wrong:3d}  "
                  f"delta_acc={(n_correct - base_correct.sum())/N_eval*100:+5.1f}pp")
            summary["alphas"].append({
                "direction": dname, "alpha": a,
                "acc": n_correct / N_eval,
                "n_correct": n_correct,
                "n_changed": n_changed,
                "wrong_to_right": wrong_to_right,
                "right_to_wrong": right_to_wrong,
                "delta_acc": (n_correct - int(base_correct.sum())) / N_eval,
            })

    out = PD / "steering_correctness.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
