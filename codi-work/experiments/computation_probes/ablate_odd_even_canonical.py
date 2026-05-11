"""Ablation at the specific (latent_step, layer) cells the user requested:
  ODD STEPS x LAYER 12: (step=1, L=12), (step=3, L=12), (step=5, L=12)
  EVEN STEPS x LAYER 0: (step=0, L=0), (step=2, L=0), (step=4, L=0)

Three modes: attention-only, MLP-only, both. Canonical LDA probe (trained on
cf_balanced correct-only at layer=8, latent_step=3) measures operator-probe
accuracy on real SVAMP held-out 200 after each intervention.

Note: ablations at (step >= 4) cannot affect the canonical capture at
step 3, so those cells should show ~0pp by construction (sanity check).
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

CLASSES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
CL2IDX = {c: i for i, c in enumerate(CLASSES)}


def main():
    print("loading canonical probe...")
    with open(PD / "canonical_probe.pkl", "rb") as f:
        probe = pickle.load(f)
    sc = probe["scaler"]; clf = probe["clf"]
    print(f"  probe at (layer={probe['layer']}, latent_step={probe['latent_step']})")

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
    targs = TrainingArguments(output_dir="/tmp/_aoe", bf16=True,
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

    HOOK = {"active": False, "tgt_step": -1, "tgt_layer": -1,
            "kill_attn": False, "kill_mlp": False, "cur_step": -99}

    def make_attn_hook(layer_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if HOOK["cur_step"] != HOOK["tgt_step"]: return output
            if HOOK["tgt_layer"] != layer_idx: return output
            if not HOOK["kill_attn"]: return output
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, -1, :] = 0
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    def make_mlp_hook(layer_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if HOOK["cur_step"] != HOOK["tgt_step"]: return output
            if HOOK["tgt_layer"] != layer_idx: return output
            if not HOOK["kill_mlp"]: return output
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, -1, :] = 0
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = []
    for L, blk in enumerate(transformer.h):
        attn_mod = getattr(blk, "self_attn", None) or getattr(blk, "attn", None)
        if attn_mod is not None:
            handles.append(attn_mod.register_forward_hook(make_attn_hook(L)))
        if hasattr(blk, "mlp"):
            handles.append(blk.mlp.register_forward_hook(make_mlp_hook(L)))

    @torch.no_grad()
    def run_batch(qs, *, tgt_step, tgt_layer, kill_attn, kill_mlp):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        # First do prompt forward (no ablation)
        HOOK.update({"active": False})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)

        # Latent loop with ablation at the specified step
        captured = None
        HOOK.update({"active": True, "tgt_step": tgt_step, "tgt_layer": tgt_layer + 0,  # 0-indexed block
                     "kill_attn": kill_attn, "kill_mlp": kill_mlp})
        for s in range(targs.inf_latent_iterations):
            HOOK["cur_step"] = s
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            if s == PROBE_LATENT_STEP:
                captured = out.hidden_states[PROBE_LAYER][:, -1, :].to(torch.float32).cpu().numpy()
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        HOOK["active"] = False
        return captured

    # Real SVAMP eval
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

    def run_full(tgt_step, tgt_layer, kill_attn, kill_mlp):
        # Note: layer index here is the 0-indexed block. The user said L0/L12 in
        # hidden-state convention (L=0 is embedding, L=12 is post-block-11).
        # Block index for hidden-state layer L is (L - 1) — but L=0 means
        # ablation before any block, which is the embedding (we can't ablate it
        # via attn/mlp hooks). For L=12, the ablation should be on block 11.
        # We treat the user's L as block index (so L=0 means block 0, L=12 is
        # ambiguous since there are only 12 blocks 0..11). Convention: L=12
        # refers to block 11 (last block).
        block_idx = min(tgt_layer, 11)
        all_caps = []
        for s in range(0, len(eval_qs), BS):
            cap = run_batch(eval_qs[s:s+BS], tgt_step=tgt_step, tgt_layer=block_idx,
                             kill_attn=kill_attn, kill_mlp=kill_mlp)
            all_caps.append(cap)
        return np.concatenate(all_caps, axis=0)

    print("\n=== Baseline (no ablation) ===")
    base_cap = run_full(-99, -1, False, False)
    base_pred = clf.predict(sc.transform(base_cap))
    base_acc = float(np.mean(base_pred[mask_y] == eval_y[mask_y]))
    print(f"  baseline LDA probe acc: {base_acc*100:.1f}%")

    odd_step_L12 = [(1, 12), (3, 12), (5, 12)]
    even_step_L0 = [(0, 0), (2, 0), (4, 0)]
    summary = {"baseline_acc": float(base_acc)}

    for label, cells in [("ODD_STEP_L12", odd_step_L12),
                          ("EVEN_STEP_L0", even_step_L0)]:
        for (step, layer) in cells:
            for mode_name, kill_attn, kill_mlp in [
                ("attn", True, False),
                ("mlp",  False, True),
                ("both", True, True),
            ]:
                cap = run_full(step, layer, kill_attn, kill_mlp)
                pred = clf.predict(sc.transform(cap))
                acc = float(np.mean(pred[mask_y] == eval_y[mask_y]))
                tag = f"{label}_step{step}_L{layer}_{mode_name}"
                print(f"  {tag}: probe_acc={acc*100:5.1f}%  (Δ {(acc - base_acc)*100:+.1f}pp)", flush=True)
                summary[tag] = float(acc)

    out = PD / "ablate_odd_even_canonical.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
