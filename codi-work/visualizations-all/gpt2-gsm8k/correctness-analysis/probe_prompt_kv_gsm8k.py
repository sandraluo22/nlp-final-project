"""Probe GSM8K prompt KV cache for answer encoding at every NUMERIC token.

GSM8K problems are multi-step; we identify ALL numeric tokens in each
question, capture K and V at each at every layer, and check which numeric
token position holds the gold answer at which layer.

For each example: numeric_token_positions = list of (token_idx, value).
For each numeric token in the prompt: probe K[L, pos] and V[L, pos] -> gold.

Aggregate: per-layer R²(gold) when probing K and V at the FIRST / LAST /
ALL numeric tokens.

This is the analog of probe_prompt_kv.py but for multi-step problems where
"operand a / operand b" isn't well-defined.
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
sys.path.insert(0, str(REPO / "codi"))

OUT_JSON = PD / "probe_prompt_kv_gsm8k.json"
OUT_PDF = PD / "probe_prompt_kv_gsm8k.pdf"

N_LIMIT = 300        # cap to keep runtime reasonable
RIDGE_ALPHA = 1.0
SEED = 0


def cv_r2(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        clf = Ridge(alpha=RIDGE_ALPHA).fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def find_numeric_token_positions(input_ids, tok):
    """Return list of (token_index, numeric_value).  Greedy: tokenize as
    decoded substrings, find digit runs in the decoded text, and back-map to
    token positions.
    """
    decoded_tokens = [tok.decode([tid]) for tid in input_ids.tolist()]
    out = []
    for ti, t in enumerate(decoded_tokens):
        stripped = t.strip()
        if re.match(r"^-?\d+\.?\d*$", stripped):
            try:
                out.append((ti, float(stripped)))
            except ValueError:
                pass
    return out


def main():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    questions, golds = [], []
    for ex in ds:
        q = ex["question"].strip().replace("  ", " ")
        m = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if m is None: continue
        questions.append(q); golds.append(float(m.group(1)))
    questions = questions[:N_LIMIT]; golds = golds[:N_LIMIT]
    N_qs = len(questions)
    print(f"GSM8K test (capped): N={N_qs}")

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
    targs = TrainingArguments(output_dir="/tmp/_kvg", bf16=True,
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
    N_LAYERS = model.codi.config.n_layer; HID = model.codi.config.n_embd

    cap = {}
    def make_c_attn_hook(idx):
        def fn(_m, _i, output):
            o = output
            q, k, v = o.split(HID, dim=-1)
            cap.setdefault("K", [None] * N_LAYERS)[idx] = k.detach().float().cpu().numpy()
            cap.setdefault("V", [None] * N_LAYERS)[idx] = v.detach().float().cpu().numpy()
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        attn = blk.attn
        target = attn.c_attn.base_layer if hasattr(attn.c_attn, "base_layer") else attn.c_attn
        handles.append(target.register_forward_hook(make_c_attn_hook(i)))

    @torch.no_grad()
    def run_prompt(q):
        cap.pop("K", None); cap.pop("V", None)
        batch = tok([q], return_tensors="pt", padding="longest").to("cuda")
        prompt_ids = batch["input_ids"][0]
        bot = torch.full((1, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        model.codi(input_ids=input_ids, attention_mask=attn,
                   use_cache=True, output_hidden_states=False)
        return prompt_ids, cap["K"], cap["V"]

    # Capture: for each example, K, V at each numeric token position at every layer.
    # Aggregate into three datasets: first-numeric, last-numeric, all-numerics-stacked.
    first_K = np.zeros((N_qs, N_LAYERS, HID), dtype=np.float32)
    first_V = np.zeros((N_qs, N_LAYERS, HID), dtype=np.float32)
    last_K  = np.zeros((N_qs, N_LAYERS, HID), dtype=np.float32)
    last_V  = np.zeros((N_qs, N_LAYERS, HID), dtype=np.float32)
    last_val = np.zeros(N_qs, dtype=np.float64)
    first_val = np.zeros(N_qs, dtype=np.float64)
    n_num_per_ex = np.zeros(N_qs, dtype=np.int64)
    t0 = time.time()
    for i, q in enumerate(questions):
        ids, K_full, V_full = run_prompt(q)
        positions = find_numeric_token_positions(ids, tok)
        if len(positions) == 0:
            continue
        n_num_per_ex[i] = len(positions)
        pos_first, val_first = positions[0]
        pos_last,  val_last  = positions[-1]
        first_val[i] = val_first; last_val[i] = val_last
        for L in range(N_LAYERS):
            first_K[i, L] = K_full[L][0, pos_first, :]
            first_V[i, L] = V_full[L][0, pos_first, :]
            last_K[i, L]  = K_full[L][0, pos_last, :]
            last_V[i, L]  = V_full[L][0, pos_last, :]
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_qs}  ({time.time()-t0:.0f}s)")
    print(f"avg numeric tokens per question: {n_num_per_ex.mean():.1f}")

    valid = n_num_per_ex > 0
    g = np.array(golds)[valid]
    K1 = first_K[valid]; V1 = first_V[valid]
    K2 = last_K[valid];  V2 = last_V[valid]
    fv = first_val[valid]; lv = last_val[valid]

    r2 = {key: [] for key in [
        "K_first->first_val", "V_first->first_val",
        "K_last->last_val",   "V_last->last_val",
        "K_first->gold", "V_first->gold",
        "K_last->gold",  "V_last->gold",
    ]}
    print("\nFitting per-layer probes...")
    for L in range(N_LAYERS):
        r2["K_first->first_val"].append(cv_r2(K1[:, L], fv))
        r2["V_first->first_val"].append(cv_r2(V1[:, L], fv))
        r2["K_last->last_val"].append(cv_r2(K2[:, L], lv))
        r2["V_last->last_val"].append(cv_r2(V2[:, L], lv))
        r2["K_first->gold"].append(cv_r2(K1[:, L], g))
        r2["V_first->gold"].append(cv_r2(V1[:, L], g))
        r2["K_last->gold"].append(cv_r2(K2[:, L], g))
        r2["V_last->gold"].append(cv_r2(V2[:, L], g))
    for k, vs in r2.items():
        bl = int(np.argmax(vs))
        print(f"  {k:25s} best L{bl:2d} R²={vs[bl]:+.3f}  per-layer: "
              f"{','.join(f'{v:+.2f}' for v in vs)}")

    out = {"N": int(valid.sum()), "N_LAYERS": N_LAYERS,
           "avg_numeric_tokens": float(n_num_per_ex.mean()),
           "r2_per_layer": r2}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")
    for h in handles: h.remove()

    with PdfPages(OUT_PDF) as pdf:
        xs = np.arange(N_LAYERS)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        ax = axes[0]
        ax.plot(xs, r2["K_first->first_val"], "o-", color="#1f77b4", label="K[first] -> first_val")
        ax.plot(xs, r2["V_first->first_val"], "s-", color="#1f77b4", ls=":", label="V[first] -> first_val")
        ax.plot(xs, r2["K_last->last_val"],   "o-", color="#ff7f0e", label="K[last] -> last_val")
        ax.plot(xs, r2["V_last->last_val"],   "s-", color="#ff7f0e", ls=":", label="V[last] -> last_val")
        ax.axhline(0, color="black", lw=0.3)
        ax.set_xlabel("layer"); ax.set_ylabel("R²"); ax.set_ylim(-0.2, 1.05)
        ax.set_title("GSM8K — numeric-token K/V → own value",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axes[1]
        ax.plot(xs, r2["K_first->gold"], "o-", color="#2ca02c", label="K[first] -> gold")
        ax.plot(xs, r2["V_first->gold"], "s-", color="#2ca02c", ls=":", label="V[first] -> gold")
        ax.plot(xs, r2["K_last->gold"],  "o-", color="#d62728", label="K[last] -> gold")
        ax.plot(xs, r2["V_last->gold"],  "s-", color="#d62728", ls=":", label="V[last] -> gold")
        ax.axhline(0, color="black", lw=0.3)
        ax.set_xlabel("layer"); ax.set_ylabel("R²"); ax.set_ylim(-0.2, 1.05)
        ax.set_title("GSM8K — numeric-token K/V → gold answer",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle(f"GSM8K prompt KV probing — first/last numeric token (N={int(valid.sum())})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
