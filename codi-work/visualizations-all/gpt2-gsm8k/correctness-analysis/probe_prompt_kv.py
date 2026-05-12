"""Probe the prompt KV cache for operand encoding.

For each CF example, capture K and V vectors at every layer at the prompt
positions of operand a and operand b. Then fit linear regression probes:
   K[L, pos_a] -> operand_a value      (per layer)
   V[L, pos_a] -> operand_a value      (per layer)
   K[L, pos_b] -> operand_b value
   V[L, pos_b] -> operand_b value
   K[L, pos_a] -> gold answer           (does operand-a's K already "know" the answer?)
   V[L, pos_b] -> gold answer

This tells us:
   - Whether operand identity is in K (the "what queries should retrieve me"
     vector) or V (the "what content I provide" vector) or both.
   - At which layers the operand information is most concentrated.
   - Whether the answer is implicit in operand K/V already, vs only computable
     by combining K's of both operands at emission time.

Setup: capture K, V via a forward hook on the c_attn module of each layer.
c_attn outputs (Q, K, V) concatenated; we split.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

CF_SETS = ["vary_numerals", "vary_both_2digit"]
OUT_JSON = PD / "probe_prompt_kv_gsm8k.json"
OUT_PDF = PD / "probe_prompt_kv_gsm8k.pdf"
RIDGE_ALPHA = 1.0
N_FOLDS = 5
SEED = 0


def load_cf(name):
    rows = json.load(open(CF_DIR / f"{name}.json"))
    qs = [r["question_concat"].strip().replace("  ", " ") for r in rows]
    golds = [float(r["answer"]) for r in rows]
    return qs, golds, rows


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


def cv_r2(X, y):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    preds = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        clf = Ridge(alpha=RIDGE_ALPHA).fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main():
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
    targs = TrainingArguments(output_dir="/tmp/_kv", bf16=True,
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

    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    N_LAYERS = model.codi.config.n_layer
    HID = model.codi.config.n_embd

    # Storage for hook: per-call we capture K, V (B, T, hidden) and store last.
    cap = {"K": None, "V": None}

    def make_attn_hook(idx):
        def fn(_m, _i, output):
            return output
        return fn

    def make_c_attn_hook(idx):
        # c_attn's output is (B, T, 3*hidden) -> split into Q, K, V.
        def fn(_m, _i, output):
            o = output  # tensor (B, T, 3H)
            q, k, v = o.split(HID, dim=-1)
            cap_K = cap.setdefault("K_full", [None] * N_LAYERS)
            cap_V = cap.setdefault("V_full", [None] * N_LAYERS)
            cap_K[idx] = k.detach().float().cpu().numpy()
            cap_V[idx] = v.detach().float().cpu().numpy()
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        # In GPT2 the attention sublayer has c_attn (fused QKV projection).
        attn = blk.attn
        if hasattr(attn, "c_attn"):
            handles.append(attn.c_attn.register_forward_hook(make_c_attn_hook(i)))
        else:
            print(f"  WARN: layer {i} has no c_attn — peft wrapping likely. Trying base_layer.")
            handles.append(attn.c_attn.base_layer.register_forward_hook(make_c_attn_hook(i)))

    @torch.no_grad()
    def run_prompt(q):
        """Run just the prompt + BOT forward pass; capture K,V at all positions."""
        cap.pop("K_full", None); cap.pop("V_full", None)
        batch = tok([q], return_tensors="pt", padding="longest").to("cuda")
        prompt_ids = batch["input_ids"][0]
        bot = torch.full((1, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        model.codi(input_ids=input_ids, attention_mask=attn,
                   use_cache=True, output_hidden_states=False)
        K_full = cap["K_full"]; V_full = cap["V_full"]
        return prompt_ids, K_full, V_full

    results = {}
    for cf_name in CF_SETS:
        print(f"\n=== {cf_name} ===", flush=True)
        qs, golds, rows = load_cf(cf_name)
        N = len(qs)
        a_vals = np.array([float(r["a"]) for r in rows])
        b_vals = np.array([float(r["b"]) for r in rows])
        gold_vals = np.array(golds)
        # Capture K, V at operand positions per example.
        K_at_a = np.zeros((N, N_LAYERS, HID), dtype=np.float32)
        K_at_b = np.zeros((N, N_LAYERS, HID), dtype=np.float32)
        V_at_a = np.zeros((N, N_LAYERS, HID), dtype=np.float32)
        V_at_b = np.zeros((N, N_LAYERS, HID), dtype=np.float32)
        n_skipped = 0
        t0 = time.time()
        for i, q in enumerate(qs):
            prompt_ids, K_full, V_full = run_prompt(q)
            pos_a, pos_b = find_operand_positions(prompt_ids, tok, a_vals[i], b_vals[i])
            if not pos_a or not pos_b:
                n_skipped += 1; continue
            for L in range(N_LAYERS):
                K_at_a[i, L] = K_full[L][0, pos_a[0], :]
                K_at_b[i, L] = K_full[L][0, pos_b[0], :]
                V_at_a[i, L] = V_full[L][0, pos_a[0], :]
                V_at_b[i, L] = V_full[L][0, pos_b[0], :]
        print(f"  captured K,V for N={N - n_skipped}/{N}  ({time.time()-t0:.0f}s)")

        # Probes: per layer, for K[a] and V[a] -> operand_a; same for b; cross-product
        # for K[a] -> gold; V[a] -> gold; etc.
        layers = list(range(N_LAYERS))
        r2 = {key: [] for key in [
            "K_a->a", "V_a->a", "K_b->b", "V_b->b",
            "K_a->gold", "V_a->gold", "K_b->gold", "V_b->gold",
            "K_a->b", "V_a->b",
        ]}
        for L in layers:
            r2["K_a->a"].append(cv_r2(K_at_a[:, L], a_vals))
            r2["V_a->a"].append(cv_r2(V_at_a[:, L], a_vals))
            r2["K_b->b"].append(cv_r2(K_at_b[:, L], b_vals))
            r2["V_b->b"].append(cv_r2(V_at_b[:, L], b_vals))
            r2["K_a->gold"].append(cv_r2(K_at_a[:, L], gold_vals))
            r2["V_a->gold"].append(cv_r2(V_at_a[:, L], gold_vals))
            r2["K_b->gold"].append(cv_r2(K_at_b[:, L], gold_vals))
            r2["V_b->gold"].append(cv_r2(V_at_b[:, L], gold_vals))
            r2["K_a->b"].append(cv_r2(K_at_a[:, L], b_vals))
            r2["V_a->b"].append(cv_r2(V_at_a[:, L], b_vals))
        for k, vs in r2.items():
            best_L = int(np.argmax(vs))
            print(f"  {k:14s}  best L{best_L:2d} R²={vs[best_L]:+.3f}  "
                  f"(per-layer: {','.join(f'{v:+.2f}' for v in vs)})")
        results[cf_name] = {"N": int(N - n_skipped), "N_LAYERS": N_LAYERS, "r2_per_layer": r2}

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()

    # PDF: per-CF line plot of R² vs layer for each probe target
    with PdfPages(OUT_PDF) as pdf:
        for cf, r in results.items():
            r2 = r["r2_per_layer"]
            xs = np.arange(r["N_LAYERS"])
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            # Panel 1: K vs V for predicting own operand
            ax = axes[0]
            ax.plot(xs, r2["K_a->a"], "o-", color="#1f77b4", lw=2, label="K[operand_a] -> a")
            ax.plot(xs, r2["V_a->a"], "s-", color="#1f77b4", lw=2, ls=":", label="V[operand_a] -> a")
            ax.plot(xs, r2["K_b->b"], "o-", color="#ff7f0e", lw=2, label="K[operand_b] -> b")
            ax.plot(xs, r2["V_b->b"], "s-", color="#ff7f0e", lw=2, ls=":", label="V[operand_b] -> b")
            ax.axhline(0, color="black", lw=0.3)
            ax.set_xlabel("layer"); ax.set_ylabel("R²")
            ax.set_title(f"{cf} — operand identity in K vs V", fontsize=10, fontweight="bold")
            ax.set_ylim(-0.2, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            # Panel 2: K/V at a position predicting gold (and b)
            ax = axes[1]
            ax.plot(xs, r2["K_a->gold"], "o-", color="#2ca02c", lw=2, label="K[a] -> gold")
            ax.plot(xs, r2["V_a->gold"], "s-", color="#2ca02c", lw=2, ls=":", label="V[a] -> gold")
            ax.plot(xs, r2["K_a->b"], "o-", color="#d62728", lw=2, label="K[a] -> b")
            ax.plot(xs, r2["V_a->b"], "s-", color="#d62728", lw=2, ls=":", label="V[a] -> b")
            ax.axhline(0, color="black", lw=0.3)
            ax.set_xlabel("layer"); ax.set_ylabel("R²")
            ax.set_title(f"{cf} — does a's K/V already know gold / b?",
                         fontsize=10, fontweight="bold")
            ax.set_ylim(-0.2, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            # Panel 3: K/V at b predicting gold
            ax = axes[2]
            ax.plot(xs, r2["K_b->gold"], "o-", color="#2ca02c", lw=2, label="K[b] -> gold")
            ax.plot(xs, r2["V_b->gold"], "s-", color="#2ca02c", lw=2, ls=":", label="V[b] -> gold")
            ax.axhline(0, color="black", lw=0.3)
            ax.set_xlabel("layer"); ax.set_ylabel("R²")
            ax.set_title(f"{cf} — does b's K/V already know gold?",
                         fontsize=10, fontweight="bold")
            ax.set_ylim(-0.2, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            fig.suptitle(f"Probe the prompt KV cache for operand info — {cf}",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
