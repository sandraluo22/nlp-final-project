"""Per-(MLP layer) and per-(residual layer) ablation at prompt-end, scored
with the canonical LDA probe (cf_balanced-trained, layer=8, latent_step=3).

For each layer L in 0..11:
  (a) MLP ablation: zero the MLP block's output at the prompt-end token
  (b) RESID ablation: zero the entire decoder block's output (= cumulative
      residual stream after that block) at the prompt-end token
"""

from __future__ import annotations
import json, os, pickle, sys, time
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

PROBE_LAYER = 8
PROBE_LATENT_STEP = 3
N_LAYERS_GPT2 = 12

CLASSES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
CL2IDX = {c: i for i, c in enumerate(CLASSES)}


def main():
    print("loading canonical probe...")
    with open(PD / "canonical_probe.pkl", "rb") as f:
        probe = pickle.load(f)
    sc = probe["scaler"]; clf = probe["clf"]

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
    targs = TrainingArguments(output_dir="/tmp/_pmr", bf16=True,
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

    HOOK = {"phase": "off", "layer": -1, "kind": "off", "active": False}

    def make_mlp_hook(layer_idx):
        # forward_hook on mlp module: zero its output at last token during prompt
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if HOOK["phase"] != "prompt_end": return output
            if HOOK["kind"] != "mlp": return output
            if HOOK["layer"] != layer_idx: return output
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, -1, :] = 0
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    def make_resid_hook(layer_idx):
        # forward_hook on the BLOCK: zero its output at last token during prompt
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if HOOK["phase"] != "prompt_end": return output
            if HOOK["kind"] != "resid": return output
            if HOOK["layer"] != layer_idx: return output
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, -1, :] = 0
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = []
    for L, blk in enumerate(transformer.h):
        handles.append(blk.register_forward_hook(make_resid_hook(L)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(L)))

    @torch.no_grad()
    def run_batch(qs, *, kind, layer):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"phase": "prompt_end", "kind": kind, "layer": layer, "active": kind != "off"})
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
                captured = out.hidden_states[PROBE_LAYER][:, -1, :].to(torch.float32).cpu().numpy()
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        return captured

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_labels = np.array([CL2IDX.get(ex["Type"], -1) for ex in full])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    np.random.seed(0)
    eval_idx = np.random.choice(len(full), size=200, replace=False)
    eval_qs = [questions[i] for i in eval_idx]
    eval_y = op_labels[eval_idx]
    BS = 16
    mask_y = eval_y >= 0

    def run_full(kind, layer):
        all_caps = []
        for s in range(0, len(eval_qs), BS):
            cap = run_batch(eval_qs[s:s+BS], kind=kind, layer=layer)
            all_caps.append(cap)
        return np.concatenate(all_caps, axis=0)

    print("\n=== Baseline ===")
    base_cap = run_full("off", -1)
    base_pred = clf.predict(sc.transform(base_cap))
    base_acc = float(np.mean(base_pred[mask_y] == eval_y[mask_y]))
    print(f"  baseline LDA probe acc: {base_acc*100:.1f}%")

    print("\n=== MLP ablation per layer ===")
    mlp_grid = np.zeros(N_LAYERS_GPT2)
    t0 = time.time()
    for L in range(N_LAYERS_GPT2):
        cap = run_full("mlp", L)
        pred = clf.predict(sc.transform(cap))
        mlp_grid[L] = float(np.mean(pred[mask_y] == eval_y[mask_y]))
        print(f"  L{L:2d}: MLP ablated → probe_acc={mlp_grid[L]*100:5.1f}%  "
              f"(Δ {(mlp_grid[L] - base_acc)*100:+.1f}pp)  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== Residual stream (block output) ablation per layer ===")
    resid_grid = np.zeros(N_LAYERS_GPT2)
    t0 = time.time()
    for L in range(N_LAYERS_GPT2):
        cap = run_full("resid", L)
        pred = clf.predict(sc.transform(cap))
        resid_grid[L] = float(np.mean(pred[mask_y] == eval_y[mask_y]))
        print(f"  L{L:2d}: RESID ablated → probe_acc={resid_grid[L]*100:5.1f}%  "
              f"(Δ {(resid_grid[L] - base_acc)*100:+.1f}pp)  ({time.time()-t0:.0f}s)", flush=True)

    out = PD / "per_mlp_resid_ablation_canonical.json"
    out.write_text(json.dumps({
        "base_acc": float(base_acc),
        "mlp_grid": mlp_grid.tolist(),
        "resid_grid": resid_grid.tolist(),
        "probe_layer": PROBE_LAYER,
        "probe_latent_step": PROBE_LATENT_STEP,
    }, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
