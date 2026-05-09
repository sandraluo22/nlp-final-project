"""Activation-patching / steering on the CODI student.

Sweep of patch protocols for a (src_op → tgt_op) pair:
  1. Single-position sweep  (17 layers × 6 latent steps = 102 protocols)
  2. All thoughts at peak layer
  3. Even thoughts (0, 2, 4) at peak layer
  4. Odd thoughts  (1, 3, 5) at peak layer
  5. Baseline (no patch)              ← reference
  6. All layers at peak step          ← extreme intervention

Per-problem outcomes are bucketed into {=src, =tgt, other} where the gold
reference for "successful steering" is the target operator's answer
  tgt_ans = OP_target(cf_subs).

Centroids are precomputed (experiments/operator_centroids_layer10_step4.json)
but for the per-(layer, step) sweep we recompute them on the fly because the
LDA basis differs per (layer, step). For multi-position protocols the layer is
fixed (peak layer) and we use that one set.

Usage:
  python steering.py --src Subtraction --tgt Addition --num 100
  python steering.py --src Multiplication --tgt Common-Division --num 100 --batch 64
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "codi"))

import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file


CENTROIDS_PATH = REPO / "experiments" / "operator_centroids_layer10_step4.json"
CF_DATA_PATH = REPO.parent / "cf-datasets" / "cf_balanced.json"
CF_ACTS_PATH = REPO / "inference" / "runs" / "cf_balanced_student" / "activations.pt"

DEFAULT_BASE = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_CKPT = "~/codi_ckpt/CODI-llama3.2-1b-Instruct"
PEAK_LAYER = 10
PEAK_STEP_IDX = 3   # latent step 4 (0-indexed)
N_LATENT_STEPS = 6
N_LAYERS_PLUS_EMB = 17  # layer 0 = embedding, 1..16 = decoder blocks


# ---------------------------------------------------------------------------
# Per-(layer, step) centroid table — needed for the single-position sweep.
# ---------------------------------------------------------------------------

def compute_all_centroids():
    """Return centroids (L, S, n_classes, H), LDA-direction tensor
    (L, S, n_classes, n_classes, H) where lda[..., ti, si] = coef_[ti] - coef_[si],
    a class->idx mapping, and the raw acts + types arrays (for latent-space
    centroid recomputation at runtime through model.prj).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    print(f"computing centroids + LDAs from {CF_ACTS_PATH}", flush=True)
    acts = torch.load(CF_ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(CF_DATA_PATH))
    types = np.array([r["type"] for r in rows])
    classes = sorted(set(types))
    cidx = {c: i for i, c in enumerate(classes)}
    N, S, L, H = acts.shape
    centroids = np.zeros((L, S, len(classes), H), dtype=np.float32)
    for c in classes:
        mask = types == c
        m = acts[mask].mean(axis=0)
        centroids[:, :, cidx[c], :] = m.transpose(1, 0, 2)

    print(f"  fitting {L*S} per-(layer, step) LDAs", flush=True)
    lda_dirs = np.zeros((L, S, len(classes), len(classes), H), dtype=np.float32)
    for layer in range(L):
        for step in range(S):
            X = acts[:, step, layer, :]
            try:
                lda = LinearDiscriminantAnalysis(solver="svd")
                lda.fit(X, types)
                for ti in range(len(classes)):
                    for si in range(len(classes)):
                        if ti != si:
                            lda_dirs[layer, step, ti, si] = lda.coef_[ti] - lda.coef_[si]
            except Exception:
                # Fallback to centroid diff if LDA can't be fit at this layer/step
                for ti in range(len(classes)):
                    for si in range(len(classes)):
                        if ti != si:
                            lda_dirs[layer, step, ti, si] = (
                                centroids[layer, step, ti] - centroids[layer, step, si]
                            )
    print(f"  centroids={centroids.shape}  lda_dirs={lda_dirs.shape}", flush=True)
    return centroids, lda_dirs, cidx, acts, types


# ---------------------------------------------------------------------------
# Model loading (mirrors run_eval_with_hooks.py)
# ---------------------------------------------------------------------------

def load_codi(base_model: str, ckpt_dir: str, device: str = "cuda"):
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = lambda *a, **k: _orig(*a, **{**k, "use_fast": True})

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
        output_dir="/tmp/_codi_steering", bf16=True, use_lora=True,
        use_prj=True, prj_dim=2048, prj_no_ln=False, prj_dropout=0.0,
        num_latent=6, inf_latent_iterations=6, remove_eos=True, greedy=True,
        model_max_length=512, seed=11,
    )
    model = CODI(margs, targs, lora_cfg)
    sd_path_safe = Path(ckpt_dir).expanduser() / "model.safetensors"
    sd_path_bin = Path(ckpt_dir).expanduser() / "pytorch_model.bin"
    if sd_path_safe.exists():
        sd = load_file(str(sd_path_safe))
    else:
        sd = torch.load(str(sd_path_bin), map_location="cpu")
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
    """Return list of N_LAYERS_PLUS_EMB - 1 (=16) decoder block modules. Index i
    of the returned list corresponds to layer (i+1) in our hidden-state
    indexing (since output_hidden_states[0] is the embedding, and
    output_hidden_states[L] for L in 1..16 = output of decoder block L-1)."""
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
    matches = re.findall(r"-?\d+\.?\d*", text)
    if not matches:
        return float("inf")
    return float(matches[-1])


# ---------------------------------------------------------------------------
# Patch state + hooks: per-layer hook reads global step counter.
# ---------------------------------------------------------------------------

class PatchPlan:
    """Holds the steering plan and current step counter for a single
    forward run. The hook reads this state."""
    def __init__(self):
        # Residual-mode plan: layer (1..16) -> set of step indices to patch.
        self.plan: dict[int, set[int]] = {}
        # Per-(layer, step) steering vector pre-built (depends on src/tgt + mode).
        self.vec_per_layer_step: dict[tuple[int, int], torch.Tensor] = {}
        # Latent-mode plan: set of step indices (0..5) to intervene AFTER.
        self.latent_steps_to_patch: set[int] = set()
        # Pre-built latent-thought target / src centroids (in latent-projected space).
        self.latent_tgt_per_step: dict[int, torch.Tensor] = {}
        self.latent_src_per_step: dict[int, torch.Tensor] = {}
        self.latent_mode: str = "residual"  # "residual" | "latent_replace" | "latent_shift"
        self.current_step: int = -1
        self.scale: float = 1.0

    def reset(self):
        self.current_step = -1


def make_layer_hook(plan: PatchPlan, layer_idx: int):
    def _hook(_module, _input, output):
        s = plan.current_step
        if s < 0:
            return output
        if layer_idx not in plan.plan or s not in plan.plan[layer_idx]:
            return output
        residual = output[0] if isinstance(output, tuple) else output
        # Add the precomputed steering vector (already encodes either
        # tgt-src centroid diff or LDA's coef[tgt]-coef[src]).
        vec = plan.vec_per_layer_step[(layer_idx, s)]
        residual[:, -1, :] = residual[:, -1, :] + plan.scale * vec
        if isinstance(output, tuple):
            return (residual,) + output[1:]
        return residual
    return _hook


# ---------------------------------------------------------------------------
# Batched CODI inference with steering.
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_batch(
    model, tok, questions: list[str],
    plan: PatchPlan, max_new_tokens: int, device: str,
) -> list[str]:
    """Run CODI inference on a batch of questions. The plan controls patching."""
    embed_fn = model.get_embd(model.codi, model.model_name)
    B = len(questions)

    # Tokenize batch, left-padded.
    batch = tok(questions, return_tensors="pt", padding="longest").to(device)
    bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device=device)
    input_ids = torch.cat([batch["input_ids"], bot], dim=1)
    attn_mask = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)

    plan.reset()  # current_step = -1 during prompt encode (no patching)
    out = model.codi(
        input_ids=input_ids, attention_mask=attn_mask,
        use_cache=True, output_hidden_states=True,
    )
    past_kv = out.past_key_values
    latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)

    # 6 latent forward passes; hooks may fire on each.
    for s in range(N_LATENT_STEPS):
        plan.current_step = s
        attn_mask = torch.cat(
            [attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=device)],
            dim=1,
        )
        out = model.codi(
            inputs_embeds=latent, attention_mask=attn_mask,
            use_cache=True, output_hidden_states=True, past_key_values=past_kv,
        )
        past_kv = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent = model.prj(latent)
        # Optional latent-level intervention: AFTER computing the next latent
        # thought, replace or shift it.
        if plan.latent_mode != "residual" and s in plan.latent_steps_to_patch:
            tgt_c = plan.latent_tgt_per_step[s]  # shape (1, 1, H) on device
            tgt_b = tgt_c.expand(B, 1, -1)
            if plan.latent_mode == "latent_replace":
                latent = tgt_b
            elif plan.latent_mode == "latent_shift":
                src_c = plan.latent_src_per_step[s]
                src_b = src_c.expand(B, 1, -1)
                latent = latent + plan.scale * (tgt_b - src_b)
    plan.reset()

    # Decode answer per row.
    eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device=device))
    output = eot_emb.unsqueeze(0).expand(B, -1, -1)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    pred_tokens: list[list[int]] = [[] for _ in range(B)]
    for _ in range(max_new_tokens):
        attn_mask = torch.cat(
            [attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=device)],
            dim=1,
        )
        step_out = model.codi(
            inputs_embeds=output, attention_mask=attn_mask,
            use_cache=True, output_hidden_states=False,
            output_attentions=False, past_key_values=past_kv,
        )
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


# ---------------------------------------------------------------------------
# Protocol enumeration.
# ---------------------------------------------------------------------------

def enumerate_residual_protocols(peak_layer: int, peak_step: int):
    """Residual-mode protocols: (name, plan_dict {layer -> set of step_idx})."""
    yield ("baseline", {})
    for layer in range(1, N_LAYERS_PLUS_EMB):
        for step in range(N_LATENT_STEPS):
            yield (f"single_layer{layer}_step{step+1}", {layer: {step}})
    yield (f"all_thoughts_layer{peak_layer}",
           {peak_layer: set(range(N_LATENT_STEPS))})
    yield (f"even_thoughts_layer{peak_layer}", {peak_layer: {0, 2, 4}})
    yield (f"odd_thoughts_layer{peak_layer}", {peak_layer: {1, 3, 5}})
    yield (f"all_layers_step{peak_step+1}",
           {layer: {peak_step} for layer in range(1, N_LAYERS_PLUS_EMB)})


def enumerate_latent_protocols():
    """Latent-mode protocols: (name, set of latent step indices to patch)."""
    yield ("baseline", set())
    for step in range(N_LATENT_STEPS):
        yield (f"single_step{step+1}", {step})
    yield ("all_thoughts", set(range(N_LATENT_STEPS)))
    yield ("even_thoughts", {0, 2, 4})
    yield ("odd_thoughts", {1, 3, 5})


# ---------------------------------------------------------------------------
# Hypothetical target answer.
# ---------------------------------------------------------------------------

def target_answer(cf_subs: list[int], tgt: str) -> int | None:
    if len(cf_subs) != 2:
        return None
    a, b = cf_subs
    if tgt == "Addition":
        return a + b
    if tgt == "Subtraction":
        if a < b:
            return None  # avoid negatives
        return a - b
    if tgt == "Multiplication":
        return a * b
    if tgt == "Common-Division":
        if b == 0 or a % b != 0:
            return None
        return a // b
    raise ValueError(tgt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   choices=["Addition", "Subtraction", "Multiplication", "Common-Division"])
    p.add_argument("--tgt", required=True,
                   choices=["Addition", "Subtraction", "Multiplication", "Common-Division"])
    p.add_argument("--num", type=int, default=100, help="problems to evaluate per protocol")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--peak_layer", type=int, default=PEAK_LAYER)
    p.add_argument("--peak_step", type=int, default=PEAK_STEP_IDX,
                   help="0-indexed peak latent step (default 3 = step 4)")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiplier on the steering direction. >1 amplifies it.")
    p.add_argument("--filter_baseline_correct", action="store_true",
                   help="Pre-run baseline; keep only candidates where model gets src answer right.")
    p.add_argument(
        "--mode",
        choices=["residual_centroid", "residual_lda", "latent_replace", "latent_shift"],
        default="residual_centroid",
        help=(
            "residual_centroid: hook adds (mu_tgt - mu_src) at layer L, step s. "
            "residual_lda: hook adds (lda_coef[tgt] - lda_coef[src]). "
            "latent_replace: replace projected latent thought with target's projected centroid. "
            "latent_shift: shift projected latent thought by (mu_tgt - mu_src) in latent space."
        ),
    )
    p.add_argument("--base_model", default=DEFAULT_BASE)
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(REPO / "experiments" / "steering_results.json"))
    args = p.parse_args()

    if args.src == args.tgt:
        raise ValueError("src and tgt must differ")

    centroids, lda_dirs, cidx, raw_acts, raw_types = compute_all_centroids()
    src_idx, tgt_idx = cidx[args.src], cidx[args.tgt]

    # Pick test problems.
    rows = json.load(open(CF_DATA_PATH))
    candidates = []
    for r in rows:
        if r["type"] != args.src:
            continue
        ta = target_answer(r["cf_subs"], args.tgt)
        if ta is None:
            continue
        if ta == r["cf_answer"]:
            continue  # degenerate cases where src and tgt answers coincide
        candidates.append({
            "src_idx": r["src_idx"], "subs": r["cf_subs"],
            "question": r["cf_question_concat"], "src_ans": r["cf_answer"],
            "tgt_ans": ta,
        })
    print(f"candidates with well-defined {args.src}→{args.tgt}: {len(candidates)}")
    candidates = candidates[: args.num]
    n = len(candidates)
    print(f"evaluating {n}")

    print("loading CODI student...")
    model, tok = load_codi(args.base_model, args.ckpt_dir, device=args.device)
    layer_modules = find_layer_modules(model)  # 16 modules; layer i+1 -> layer_modules[i]
    print(f"  hooked {len(layer_modules)} decoder blocks")

    plan = PatchPlan()
    plan.scale = args.scale
    plan.latent_mode = args.mode if args.mode.startswith("latent") else "residual"

    # Build per-(layer, step) steering vectors for residual modes.
    if args.mode == "residual_centroid":
        for layer in range(1, N_LAYERS_PLUS_EMB):
            for step in range(N_LATENT_STEPS):
                v = centroids[layer, step, tgt_idx] - centroids[layer, step, src_idx]
                plan.vec_per_layer_step[(layer, step)] = torch.tensor(
                    v, dtype=torch.bfloat16, device=args.device,
                )
    elif args.mode == "residual_lda":
        for layer in range(1, N_LAYERS_PLUS_EMB):
            for step in range(N_LATENT_STEPS):
                v = lda_dirs[layer, step, tgt_idx, src_idx]
                plan.vec_per_layer_step[(layer, step)] = torch.tensor(
                    v, dtype=torch.bfloat16, device=args.device,
                )
    else:
        # latent modes: vec_per_layer_step unused; build latent-space centroids
        # below after we have the model loaded.
        pass

    # Register hooks (always — they no-op if plan.plan is empty or current_step < 0).
    handles = [
        layer_modules[layer - 1].register_forward_hook(make_layer_hook(plan, layer))
        for layer in range(1, N_LAYERS_PLUS_EMB)
    ]

    # For latent modes, project the LAST-LAYER per-step class centroids through
    # model.prj to get centroids in the latent-thought space.
    if args.mode in ("latent_replace", "latent_shift"):
        with torch.no_grad():
            for s in range(N_LATENT_STEPS):
                src_h16 = torch.tensor(centroids[16, s, src_idx],
                                       dtype=torch.bfloat16, device=args.device)
                tgt_h16 = torch.tensor(centroids[16, s, tgt_idx],
                                       dtype=torch.bfloat16, device=args.device)
                src_proj = model.prj(src_h16.view(1, 1, -1))
                tgt_proj = model.prj(tgt_h16.view(1, 1, -1))
                plan.latent_src_per_step[s] = src_proj
                plan.latent_tgt_per_step[s] = tgt_proj
        print(f"  built latent-space centroids for {args.mode}", flush=True)

    # Optional pre-filter: run baseline on all candidates, keep only those the
    # student gets correct (matches src_ans). Lets us measure flip rate against
    # a clean reference rather than against confused baseline outputs.
    if args.filter_baseline_correct:
        plan.plan = {}
        kept = []
        preds: list[float] = []
        for start in range(0, n, args.batch):
            batch_q = [c["question"] for c in candidates[start : start + args.batch]]
            texts = run_batch(model, tok, batch_q, plan, args.max_new_tokens, args.device)
            preds.extend(extract_answer_number(t) for t in texts)
        for c, pr in zip(candidates, preds):
            if pr == c["src_ans"]:
                kept.append(c)
        print(f"  baseline-correct filter: {len(kept)}/{len(candidates)}")
        candidates = kept
        n = len(candidates)

    # Iterate protocols.
    if args.mode.startswith("residual"):
        protocols = list(enumerate_residual_protocols(args.peak_layer, args.peak_step))
    else:
        protocols = list(enumerate_latent_protocols())
    print(f"running {len(protocols)} protocols", flush=True)
    summary: dict[str, dict] = {}
    per_problem: dict[str, list[dict]] = {}
    try:
        for pi, (name, plan_arg) in enumerate(protocols):
            if args.mode.startswith("residual"):
                plan.plan = plan_arg
                plan.latent_steps_to_patch = set()
            else:
                plan.plan = {}
                plan.latent_steps_to_patch = plan_arg
            preds: list[float] = []
            for start in range(0, n, args.batch):
                batch_q = [c["question"] for c in candidates[start : start + args.batch]]
                texts = run_batch(model, tok, batch_q, plan, args.max_new_tokens, args.device)
                preds.extend(extract_answer_number(t) for t in texts)
            counts = Counter()
            rows_p = []
            for c, p_pred in zip(candidates, preds):
                eq_src = p_pred == c["src_ans"]
                eq_tgt = p_pred == c["tgt_ans"]
                bucket = (
                    "tgt"   if eq_tgt and not eq_src
                    else "src" if eq_src and not eq_tgt
                    else "either" if eq_tgt and eq_src
                    else "other"
                )
                counts[bucket] += 1
                rows_p.append({"src_idx": c["src_idx"], "pred": p_pred, "bucket": bucket})
            summary[name] = {
                "n": n, "tgt": counts["tgt"], "src": counts["src"],
                "either": counts["either"], "other": counts["other"],
                "tgt_rate": counts["tgt"] / n,
                "src_rate": counts["src"] / n,
            }
            per_problem[name] = rows_p
            if pi < 6 or pi % 20 == 0 or pi == len(protocols) - 1:
                print(
                    f"  [{pi+1:>3d}/{len(protocols)}]  {name:<35s}  "
                    f"tgt={counts['tgt']:>3d}/{n} ({counts['tgt']/n*100:>5.1f}%)  "
                    f"src={counts['src']:>3d}/{n}  other={counts['other']:>3d}",
                    flush=True,
                )
    finally:
        for h in handles:
            h.remove()

    # Save full results.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "n_candidates": n,
        "summary": summary,
        "per_problem": per_problem,
    }, indent=2))
    print(f"\nsaved -> {out_path}")

    # Print top-10 most-effective single-position protocols.
    sp = [(name, s["tgt_rate"]) for name, s in summary.items() if name.startswith("single_")]
    sp.sort(key=lambda x: -x[1])
    print("\ntop-10 most-effective single-position patches:")
    for name, rate in sp[:10]:
        print(f"  {name:<35s}  tgt_rate={rate*100:.1f}%")

    print("\nmulti-position protocols:")
    if args.mode.startswith("residual"):
        keys = [
            "baseline",
            f"all_thoughts_layer{args.peak_layer}",
            f"even_thoughts_layer{args.peak_layer}",
            f"odd_thoughts_layer{args.peak_layer}",
            f"all_layers_step{args.peak_step+1}",
        ]
    else:
        keys = ["baseline", "all_thoughts", "even_thoughts", "odd_thoughts"] + [
            f"single_step{s+1}" for s in range(N_LATENT_STEPS)
        ]
    for nm in keys:
        if nm in summary:
            s = summary[nm]
            print(f"  {nm:<35s}  tgt={s['tgt_rate']*100:>5.1f}%  src={s['src_rate']*100:>5.1f}%  other={s['other']/n*100:>5.1f}%")


if __name__ == "__main__":
    main()
