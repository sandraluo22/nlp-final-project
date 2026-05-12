"""SVAMP→GSM8K TRANSFER per-head ablation scored with the SVAMP-fit CANONICAL
probe (LDA fit on SVAMP cf_balanced at layer=8, latent_step=3).

For each (layer, head) cell:
  1. Zero head h's contribution at the prompt-end position
  2. Run the latent loop on a GSM8K test subset, capture residual at
     (layer=8, latent_step=3)
  3. Apply the SAVED SVAMP LDA probe → operator accuracy where the GSM8K
     label is the FIRST marker's operator from the gold chain.
"""

from __future__ import annotations
import json, os, pickle, re, sys, time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

N_LAYERS_GPT2 = 12
N_HEADS = 12
HEAD_DIM = 64
PROBE_LAYER = 8        # canonical probe layer (CODI-GPT-2 has L+1=13 hidden states; we capture L=8)
PROBE_LATENT_STEP = 3  # canonical probe latent step

CLASSES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
CL2IDX = {c: i for i, c in enumerate(CLASSES)}


def main():
    print("loading canonical probe...")
    with open(PD / "canonical_probe.pkl", "rb") as f:
        probe = pickle.load(f)
    sc = probe["scaler"]; clf = probe["clf"]
    print(f"  probe at (layer={probe['layer']}, latent_step={probe['latent_step']}) loaded")

    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_phc", bf16=True,
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
    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)

    HOOK = {"phase": "off", "layer": -1, "head": -1, "active": False}

    def make_pre_hook(layer_idx):
        def fn(module, inputs):
            if not HOOK["active"]: return None
            if HOOK["phase"] != "prompt_end": return None
            if HOOK["layer"] != layer_idx: return None
            x = inputs[0].clone()
            h = HOOK["head"]
            x[:, -1, h*HEAD_DIM:(h+1)*HEAD_DIM] = 0
            return (x,) + inputs[1:]
        return fn

    handles = []
    for L, blk in enumerate(transformer.h):
        attn_mod = getattr(blk, "self_attn", None) or getattr(blk, "attn", None)
        c_proj = attn_mod.c_proj
        handles.append(c_proj.register_forward_pre_hook(make_pre_hook(L)))

    @torch.no_grad()
    def run_batch(qs, *, layer, head):
        """Run inference with attn-head ablated at prompt-end, capture residual
        at (layer=PROBE_LAYER, latent_step=PROBE_LATENT_STEP)."""
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"phase": "prompt_end", "layer": layer, "head": head, "active": layer >= 0})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        HOOK["active"] = False

        captured = None
        for s in range(targs.inf_latent_iterations):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            if s == PROBE_LATENT_STEP:
                # Capture all-layer hidden states at this latent step's last token
                captured = out.hidden_states[PROBE_LAYER][:, -1, :].to(torch.float32).cpu().numpy()
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        return captured

    # GSM8K eval — first-marker operator label
    op_char_to_idx = {"+": 0, "-": 1, "*": 2, "/": 3}
    ds = load_dataset("gsm8k", "main")["test"]
    qs_all, op_all = [], []
    for ex in ds:
        ans = ex["answer"].replace(",", "")
        m = re.search(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=", ans)
        if m is None:
            continue
        qs_all.append(ex["question"].strip().replace("  ", " "))
        op_all.append(op_char_to_idx.get(m.group(2), -1))
    op_labels = np.array(op_all)
    # subsample for speed
    np.random.seed(0)
    eval_idx = np.random.choice(len(qs_all), size=200, replace=False)
    eval_qs = [qs_all[i] for i in eval_idx]
    eval_y = op_labels[eval_idx]
    BS = 16
    mask_y = eval_y >= 0

    def run_full(layer, head):
        all_caps = []
        for s in range(0, len(eval_qs), BS):
            cap = run_batch(eval_qs[s:s+BS], layer=layer, head=head)
            all_caps.append(cap)
        return np.concatenate(all_caps, axis=0)

    print("\n=== Baseline (no ablation) ===")
    base_cap = run_full(-1, -1)
    base_pred = clf.predict(sc.transform(base_cap))
    base_acc = float(np.mean(base_pred[mask_y] == eval_y[mask_y]))
    print(f"  baseline SVAMP-fit LDA probe acc on 200 GSM8K examples: {base_acc*100:.1f}%")

    grid = np.zeros((N_LAYERS_GPT2, N_HEADS), dtype=float)
    t0 = time.time()
    for L in range(N_LAYERS_GPT2):
        for H in range(N_HEADS):
            cap = run_full(L, H)
            pred = clf.predict(sc.transform(cap))
            acc = float(np.mean(pred[mask_y] == eval_y[mask_y]))
            grid[L, H] = acc
            print(f"  L{L:2d} H{H:2d}: probe_acc={acc*100:5.1f}%  "
                  f"(Δ {(acc - base_acc)*100:+.1f}pp)  ({time.time()-t0:.0f}s)", flush=True)

    out = Path(__file__).resolve().parent / "per_head_ablation_svamp2gsm8k_canonical.json"
    out.write_text(json.dumps({
        "base_acc": float(base_acc),
        "grid": grid.tolist(),
        "probe_layer": PROBE_LAYER,
        "probe_latent_step": PROBE_LATENT_STEP,
    }, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
