"""Per-(stage, layer, component) zero-ablation on CODI-GPT-2 over SVAMP.

For each cell (stage in {0=prompt, 1..6 = latent step}) × (layer 0..11) ×
(component in {resid, attn, mlp}), run inference on N SVAMP examples with that
component's output at that cell zeroed at the LAST POSITION of that stage.
Record:
  - n_changed_int: count where parsed first-int answer differs from baseline
  - n_first_token_changed: count where the first decode token differs
  - mean change in median |output|

That gives a 7 × 12 × 3 grid of "how causal is this cell".
"""

from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]   # codi-work
sys.path.insert(0, str(REPO / "codi"))

N_LATENT = 6
N_LAYERS = 12
COMPS = ["resid", "attn", "mlp"]


def parse_int(s):
    m = re.search(r"answer is\s*:\s*(-?\d+)", s)
    return int(m.group(1)) if m else None


def main():
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_abl", bf16=True,
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

    HOOK = {"current_stage": -1, "active": False,
            "tgt_stage": -1, "tgt_layer": -1, "tgt_comp": ""}

    def make_hook(component, layer_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if HOOK["current_stage"] != HOOK["tgt_stage"]: return output
            if HOOK["tgt_layer"] != layer_idx: return output
            if HOOK["tgt_comp"] != component: return output
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, -1, :] = 0  # zero the last-token contribution
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = []
    for L, blk in enumerate(transformer.h):
        handles.append(blk.register_forward_hook(make_hook("resid", L)))
        attn_mod = getattr(blk, "self_attn", None) or getattr(blk, "attn", None)
        if attn_mod is not None:
            handles.append(attn_mod.register_forward_hook(make_hook("attn", L)))
        if hasattr(blk, "mlp"):
            handles.append(blk.mlp.register_forward_hook(make_hook("mlp", L)))

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    answers = [int(round(float(str(ex["Answer"]).replace(",", "")))) for ex in full]

    @torch.no_grad()
    def run_batch(qs, *, tgt_stage=-1, tgt_layer=-1, tgt_comp=""):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"active": tgt_layer >= 0,
                     "tgt_stage": tgt_stage, "tgt_layer": tgt_layer,
                     "tgt_comp": tgt_comp, "current_stage": 0})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for s in range(targs.inf_latent_iterations):
            HOOK["current_stage"] = s + 1
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        # decode WITHOUT ablation (set stage to -1 so hook doesn't fire)
        HOOK["current_stage"] = -1
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        for _ in range(12):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B): tokens[b].append(int(next_ids[b].item()))
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK["active"] = False
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    N = 200
    np.random.seed(0)
    eval_idx = np.random.choice(len(questions), size=N, replace=False)
    eval_qs = [questions[i] for i in eval_idx]
    eval_gold = [answers[i] for i in eval_idx]
    BS = 16

    def run_full(**kw):
        out = []
        for s in range(0, N, BS):
            out += run_batch(eval_qs[s:s+BS], **kw)
        return out

    print(f"\n=== Baseline (no ablation) ===", flush=True)
    base_strs = run_full()
    base_ints = [parse_int(s) for s in base_strs]
    base_first_tok = [s[len("The answer is: "):][:1] if "answer is:" in s else "" for s in base_strs]
    base_correct = sum(1 for i, g in zip(base_ints, eval_gold) if i == g)
    print(f"  baseline correct: {base_correct}/{N}", flush=True)
    print(f"  baseline median |int|: {int(np.median([abs(v) for v in base_ints if v is not None]))}", flush=True)

    n_stages = N_LATENT + 1
    grid_changed = {c: np.zeros((n_stages, N_LAYERS), dtype=int) for c in COMPS}
    grid_correct = {c: np.zeros((n_stages, N_LAYERS), dtype=int) for c in COMPS}
    grid_med = {c: np.zeros((n_stages, N_LAYERS), dtype=int) for c in COMPS}
    # NEW: save per-example output ints for resid only (smaller footprint)
    per_ex_resid = np.zeros((n_stages, N_LAYERS, N), dtype=np.int64)
    base_ints_arr = np.array([v if v is not None else 0 for v in base_ints], dtype=np.int64)

    t0 = time.time()
    for stage in range(n_stages):
        for layer in range(N_LAYERS):
            for comp in COMPS:
                strs = run_full(tgt_stage=stage, tgt_layer=layer, tgt_comp=comp)
                ints = [parse_int(s) for s in strs]
                n_chg = sum(1 for a, b in zip(ints, base_ints) if a != b)
                n_cor = sum(1 for a, g in zip(ints, eval_gold) if a == g)
                med = int(np.median([abs(v) for v in ints if v is not None])) if any(v is not None for v in ints) else 0
                grid_changed[comp][stage, layer] = n_chg
                grid_correct[comp][stage, layer] = n_cor
                grid_med[comp][stage, layer] = med
                if comp == "resid":
                    per_ex_resid[stage, layer] = np.array([v if v is not None else 0 for v in ints], dtype=np.int64)
            print(f"  stage {stage} L{layer:2d}: resid={grid_changed['resid'][stage,layer]}/200  "
                  f"attn={grid_changed['attn'][stage,layer]}/200  mlp={grid_changed['mlp'][stage,layer]}/200  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = REPO / "head-patching" / "ablation_codi_gpt2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_eval": N,
        "baseline_correct": int(base_correct),
        "baseline_median_abs": int(np.median([abs(v) for v in base_ints if v is not None])),
        "grid_changed": {c: grid_changed[c].tolist() for c in COMPS},
        "grid_correct": {c: grid_correct[c].tolist() for c in COMPS},
        "grid_median_abs": {c: grid_med[c].tolist() for c in COMPS},
    }, indent=2))
    print(f"\nsaved {out}")
    # save per-example resid outputs separately as .npz (small but n_stages*N_LAYERS*N ints)
    npz_path = out.with_suffix(".perex.npz")
    np.savez(npz_path, base_ints=base_ints_arr, per_ex_resid=per_ex_resid)
    print(f"saved {npz_path}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
