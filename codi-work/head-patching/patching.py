"""Activation patching: clean Mul into corrupted Add CODI student forward pass.

For each (clean, corrupted) pair (same numerals, swapped operator):

  1. Run student on CLEAN (Mul).  Cache the residual stream output of every
     decoder block at the LAST POSITION of each forward pass:
        - prompt+bot   : single position = bot token
        - latent thoughts (6 forward passes) : single position each
     Total cached: 16 layers × 7 stages.
  2. Run student on CORRUPTED (Add).  Cache same.
  3. For each (layer L, stage S), do a PATCHED corrupted run: at stage S,
     layer L, replace corrupted's last-position residual output with clean's
     cached value at that cell. Continue forward, generate answer.
  4. Score the patched answer:
        - matches CLEAN (a*b)      → patching recovered Mul. Cell is causal for
                                       the operator decision.
        - matches CORRUPTED (a+b)  → no recovery (Add stuck).
        - other                    → confused.

Aggregates per (layer, stage) flip-rate over 100ish pairs and saves a heatmap.

Usage:
  python patching.py --num 100 --batch 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "codi"))

import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file


DEFAULT_BASE = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_CKPT = "~/codi_ckpt/CODI-llama3.2-1b-Instruct"
N_LATENT_STEPS = 6
N_LAYERS = 16  # decoder blocks (0..15 in module list)


def load_codi(base_model: str, ckpt_dir: str, device: str = "cuda"):
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128, lora_alpha=32,
        lora_dropout=0.1, target_modules=target_modules, init_lora_weights=True,
    )
    margs = ModelArguments(
        model_name_or_path=base_model, full_precision=True, train=False,
        lora_init=True, ckpt_dir=str(Path(ckpt_dir).expanduser()),
    )
    targs = TrainingArguments(
        output_dir="/tmp/_codi_patch", bf16=True, use_lora=True,
        use_prj=True, prj_dim=2048, prj_no_ln=False, prj_dropout=0.0,
        num_latent=6, inf_latent_iterations=6, remove_eos=True, greedy=True,
        model_max_length=512, seed=11,
    )
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt_dir).expanduser() / "model.safetensors"
    sd_bin = Path(ckpt_dir).expanduser() / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained(
        base_model, model_max_length=512, padding_side="left", use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model, tok


def find_layer_modules(model):
    for path in [
        ("codi", "model", "layers"),
        ("codi", "base_model", "model", "model", "layers"),
        ("codi", "base_model", "model", "layers"),
    ]:
        cur = model
        ok = True
        for p in path:
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok:
            return list(cur)
    raise RuntimeError("could not locate decoder layers")


def extract_answer_number(text: str) -> float:
    text = text.replace(",", "")
    m = re.findall(r"-?\d+\.?\d*", text)
    return float(m[-1]) if m else float("inf")


# -----------------------------------------------------------------------------
# Hook state: per-(stage, layer) cache + (optional) one patch cell.
# -----------------------------------------------------------------------------

class HookState:
    def __init__(self, n_layers: int, n_stages: int):
        # Cache key: (stage, layer, component) where component ∈ {"resid","attn","mlp"}.
        self.cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self.current_stage: int = -1   # 0 = prompt+bot, 1..6 = latent steps 1..6
        self.do_cache: bool = False
        self.patch: dict[tuple[int, int, str], torch.Tensor] = {}


def _apply_patch(tensor: torch.Tensor, v: torch.Tensor, stage: int) -> torch.Tensor:
    """In-place patch the given tensor with v.
       stage 0 → full slice; stages 1..6 → last position only."""
    if stage == 0:
        if v.shape == tensor.shape:
            tensor.copy_(v)
        else:
            tensor[:, -1, :] = v[:, -1, :]
    else:
        tensor[:, -1, :] = v.squeeze(1) if v.dim() == 3 else v
    return tensor


def _do_cache(stage: int, t: torch.Tensor) -> torch.Tensor:
    if stage == 0:
        return t.detach().clone()
    else:
        return t[:, -1, :].detach().clone()


def make_residual_hook(state: HookState, layer_idx: int):
    def _hook(_m, _i, output):
        s = state.current_stage
        if s < 0:
            return output
        residual = output[0] if isinstance(output, tuple) else output
        cell = (s, layer_idx, "resid")
        if cell in state.patch:
            _apply_patch(residual, state.patch[cell], s)
        if state.do_cache:
            state.cache[cell] = _do_cache(s, residual)
        return (residual,) + output[1:] if isinstance(output, tuple) else residual
    return _hook


def make_attn_hook(state: HookState, layer_idx: int):
    """Hook on self_attn module: output = (attn_out, attn_weights, past_kv)."""
    def _hook(_m, _i, output):
        s = state.current_stage
        if s < 0:
            return output
        attn_out = output[0] if isinstance(output, tuple) else output
        cell = (s, layer_idx, "attn")
        if cell in state.patch:
            _apply_patch(attn_out, state.patch[cell], s)
        if state.do_cache:
            state.cache[cell] = _do_cache(s, attn_out)
        return (attn_out,) + output[1:] if isinstance(output, tuple) else attn_out
    return _hook


def make_mlp_hook(state: HookState, layer_idx: int):
    """Hook on mlp module: output is a single tensor."""
    def _hook(_m, _i, output):
        s = state.current_stage
        if s < 0:
            return output
        cell = (s, layer_idx, "mlp")
        if cell in state.patch:
            _apply_patch(output, state.patch[cell], s)
        if state.do_cache:
            state.cache[cell] = _do_cache(s, output)
        return output
    return _hook


# -----------------------------------------------------------------------------
# Run a batch through CODI, with hooks managing cache/patch.
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_batch(model, tok, texts: list[str], state: HookState,
              max_new_tokens: int = 64, device: str = "cuda",
              fixed_max_len: int | None = None) -> list[str]:
    embed_fn = model.get_embd(model.codi, model.model_name)
    B = len(texts)
    if fixed_max_len is not None:
        batch = tok(texts, return_tensors="pt", padding="max_length",
                    max_length=fixed_max_len, truncation=True).to(device)
    else:
        batch = tok(texts, return_tensors="pt", padding="longest").to(device)
    bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device=device)
    input_ids = torch.cat([batch["input_ids"], bot], dim=1)
    attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)

    # STAGE 0: prompt + bot.
    state.current_stage = 0
    out = model.codi(input_ids=input_ids, attention_mask=attn,
                     use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)

    # STAGES 1..N_LATENT_STEPS: latent thoughts.
    for s in range(N_LATENT_STEPS):
        state.current_stage = s + 1
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=device)], dim=1)
        out = model.codi(inputs_embeds=latent, attention_mask=attn,
                         use_cache=True, output_hidden_states=True, past_key_values=past_kv)
        past_kv = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent = model.prj(latent)
    state.current_stage = -1   # disables hooks during decode

    # DECODE.
    eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device=device))
    output = eot_emb.unsqueeze(0).expand(B, -1, -1)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    pred_tokens = [[] for _ in range(B)]
    for _ in range(max_new_tokens):
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=device)], dim=1)
        step_out = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              output_attentions=False, past_key_values=past_kv)
        past_kv = step_out.past_key_values
        logits = step_out.logits[:, -1, : model.codi.config.vocab_size - 1]
        next_ids = torch.argmax(logits, dim=-1)
        for b in range(B):
            if not finished[b]:
                tid = int(next_ids[b].item())
                pred_tokens[b].append(tid)
                if tid == tok.eos_token_id:
                    finished[b] = True
        if bool(finished.all()):
            break
        output = embed_fn(next_ids).unsqueeze(1)
    return [tok.decode(p, skip_special_tokens=True) for p in pred_tokens]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def bucket(pred: float, clean_ans: int, corrupted_ans: int) -> str:
    if pred == float(clean_ans) and pred != float(corrupted_ans):
        return "clean"  # patched answer matches Mul → recovery
    if pred == float(corrupted_ans) and pred != float(clean_ans):
        return "corrupted"  # patched answer matches Add → no recovery
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_path", required=True,
                   help="Path to pairs.json (each row needs clean.text/answer + corrupted.text/answer)")
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--base_model", default=DEFAULT_BASE)
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    p.add_argument("--out", required=True,
                   help="Where to save the per-cell recovery stats JSON")
    args = p.parse_args()

    pairs = json.load(open(args.pairs_path))[: args.num]
    print(f"loaded {len(pairs)} pairs")

    # Determine a global fixed max-length so clean and corrupted residuals have
    # matching tensor shapes for full-prompt patching at stage 0.

    print("loading CODI student...")
    model, tok = load_codi(args.base_model, args.ckpt_dir, device=args.device)
    # Compute global max prompt length (in tokens) over all texts; +1 for bot.
    all_texts = [p["clean"]["text"] for p in pairs] + [p["corrupted"]["text"] for p in pairs]
    lens = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in all_texts]
    fixed_max_len = max(lens)
    print(f"  fixed prompt max_len = {fixed_max_len} tokens (largest in dataset)")
    layer_modules = find_layer_modules(model)
    n_layers = len(layer_modules)
    n_stages = N_LATENT_STEPS + 1  # 0=prompt, 1..6=latent
    state = HookState(n_layers, n_stages)
    handles = []
    for L, block in enumerate(layer_modules):
        handles.append(block.register_forward_hook(make_residual_hook(state, L)))
        if hasattr(block, "self_attn"):
            handles.append(block.self_attn.register_forward_hook(make_attn_hook(state, L)))
        if hasattr(block, "mlp"):
            handles.append(block.mlp.register_forward_hook(make_mlp_hook(state, L)))
    print(f"  hooked {n_layers} decoder blocks (resid+attn+mlp); stages = {n_stages}")

    try:
        # ---- Cache CLEAN and CORRUPTED activations + record baseline answers.
        clean_caches = {}    # (pair_idx, stage, layer) -> tensor (1, H)
        corr_caches = {}
        clean_answers = {}   # pair_idx -> str answer
        corr_answers = {}

        clean_texts = [p["clean"]["text"] for p in pairs]
        corr_texts = [p["corrupted"]["text"] for p in pairs]

        # Cache by running batches with do_cache=True (no patch).
        def run_and_cache(texts, target):
            for st in range(0, len(texts), args.batch):
                batch_texts = texts[st : st + args.batch]
                state.cache = {}
                state.patch = {}
                state.do_cache = True
                preds = run_batch(model, tok, batch_texts, state, device=args.device)
                state.do_cache = False
                # Grab the cache (it has tensors of shape (B, H) per cell).
                for cell, t in state.cache.items():
                    for b in range(t.shape[0]):
                        target[(st + b, cell[0], cell[1], cell[2])] = t[b:b+1].clone()
                yield st, preds

        print("\ncaching CLEAN activations + answers...")
        for st, preds in run_and_cache(clean_texts, clean_caches):
            for j, txt in enumerate(preds):
                clean_answers[st + j] = txt
        print("caching CORRUPTED activations + answers...")
        for st, preds in run_and_cache(corr_texts, corr_answers_target := {}):
            for j, txt in enumerate(preds):
                corr_answers[st + j] = txt
        # corr_caches via second pass restricted to caching only (already done).
        # Actually we need corrupted cache for downstream sanity / source patches.
        # Recompute corrupted caches in a separate pass (cheap enough).
        for st, _ in run_and_cache(corr_texts, corr_caches):
            pass

        # Per-pair baseline buckets.
        baseline_clean = []
        baseline_corr = []
        for i, p in enumerate(pairs):
            cp = extract_answer_number(clean_answers[i])
            ap = extract_answer_number(corr_answers[i])
            baseline_clean.append({"pred": cp, "match_clean": cp == p["clean"]["answer"],
                                   "match_corr": cp == p["corrupted"]["answer"]})
            baseline_corr.append({"pred": ap, "match_clean": ap == p["clean"]["answer"],
                                  "match_corr": ap == p["corrupted"]["answer"]})
        n_clean_correct = sum(1 for r in baseline_clean if r["match_clean"])
        n_corr_correct = sum(1 for r in baseline_corr if r["match_corr"])
        print(f"\nbaseline clean (Mul) correctness: {n_clean_correct}/{len(pairs)}")
        print(f"baseline corrupted (Add) correctness: {n_corr_correct}/{len(pairs)}")

        # Restrict to pairs where BOTH baselines are correct (cleanest test set).
        keep_idx = [i for i in range(len(pairs))
                    if baseline_clean[i]["match_clean"] and baseline_corr[i]["match_corr"]]
        print(f"  kept {len(keep_idx)} pairs where both baselines are correct")
        if len(keep_idx) == 0:
            print("no valid pairs; aborting"); return

        # ---- Patching sweep over (component, stage, layer) ----
        components = ["resid", "attn", "mlp"]
        recovery = {c: np.zeros((n_stages, n_layers), dtype=np.float32) for c in components}
        per_cell_buckets: dict = {}
        for comp in components:
            for stage in range(n_stages):
                for layer in range(n_layers):
                    cnt = Counter()
                    for st in range(0, len(keep_idx), args.batch):
                        batch_indices = keep_idx[st : st + args.batch]
                        batch_texts = [pairs[i]["corrupted"]["text"] for i in batch_indices]
                        patches = torch.cat(
                            [clean_caches[(i, stage, layer, comp)] for i in batch_indices], dim=0,
                        ).to(args.device)
                        state.cache = {}
                        state.patch = {(stage, layer, comp): patches}
                        state.do_cache = False
                        preds = run_batch(model, tok, batch_texts, state, device=args.device,
                                           fixed_max_len=fixed_max_len)
                        state.patch = {}
                        for j, txt in enumerate(preds):
                            i = batch_indices[j]
                            pp = extract_answer_number(txt)
                            b = bucket(pp, pairs[i]["clean"]["answer"], pairs[i]["corrupted"]["answer"])
                            cnt[b] += 1
                    per_cell_buckets[(comp, stage, layer)] = dict(cnt)
                    n_kept = len(keep_idx)
                    recovery[comp][stage, layer] = cnt["clean"] / max(n_kept, 1)
                print(f"  {comp:<5s} stage {stage:>1d}  recovery = {recovery[comp][stage].mean()*100:.1f}%",
                      flush=True)

    finally:
        for h in handles:
            h.remove()

    # ---- Save + print ----
    stats = {
        "n_pairs": len(pairs),
        "n_kept_pairs": len(keep_idx),
        "baseline_clean_correct": n_clean_correct,
        "baseline_corr_correct": n_corr_correct,
        "recovery": {c: recovery[c].tolist() for c in components},
        "per_cell_buckets": {f"{c}_stage{s}_layer{l}": d
                             for (c, s, l), d in per_cell_buckets.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, indent=2))
    for comp in components:
        print(f"\n=== {comp.upper()} per-(stage, layer) recovery (Mul-flip rate) ===")
        print("       " + "  ".join(f"L{L:02d}" for L in range(n_layers)))
        for s in range(n_stages):
            label = "prompt" if s == 0 else f" lat{s} "
            row = "  ".join(f"{recovery[comp][s, L]*100:>4.0f}%"
                            for L in range(n_layers))
            print(f" {label}  {row}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
