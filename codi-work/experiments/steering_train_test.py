"""Train/test split steering experiment.

Procedure:
  1. Take a labeled dataset of CODI-student activations (original SVAMP or
     cf_balanced).
  2. Stratified 50/50 split by operator.
  3. On the TRAIN half, compute per-class centroids at peak layer × peak step.
  4. Build a steering vector v = mu[tgt] - mu[src].
  5. On the held-out TEST half, run the student twice per problem:
        baseline   (no patch)
        steered    (residual at layer L step S += scale * v at last token)
  6. Filter to baseline-correct candidates, then report what each output
     matches against ALL FOUR operator gold answers (Add/Sub/Mul/Div).

Run both directions in one invocation: src→tgt AND tgt→src. Saves a single
JSON.

Usage:
  python steering_train_test.py --dataset svamp --src Subtraction --tgt Addition
  python steering_train_test.py --dataset cf_balanced --src Multiplication --tgt Common-Division
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
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "codi"))

import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

DEFAULT_BASE = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_CKPT = "~/codi_ckpt/CODI-llama3.2-1b-Instruct"
PEAK_LAYER = 10
PEAK_STEP_IDX = 3   # latent step 4 (0-indexed)
N_LATENT_STEPS = 6
N_LAYERS_PLUS_EMB = 17
OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(name: str):
    """Returns (acts, types, problems) where:
      acts: np.array (N, S, L, H)
      types: np.array (N,) of operator names
      problems: list of dicts with keys src_idx, type, question, numerals, answer
    """
    if name == "svamp":
        from datasets import concatenate_datasets, load_dataset as hf_load
        acts_path = REPO / "inference/runs/svamp_student/activations.pt"
        ds = hf_load("ChilleD/SVAMP")
        full = concatenate_datasets([ds["train"], ds["test"]])
        acts = torch.load(acts_path, map_location="cpu", weights_only=True).float().numpy()
        types = np.array(
            [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
        )
        problems = []
        for i, ex in enumerate(full):
            t = types[i]
            nums = [float(x) for x in re.findall(r"\d+\.?\d*", ex["Equation"])]
            problems.append({
                "src_idx": i, "type": t,
                "question": ex["question_concat"].strip().replace("  ", " "),
                "numerals": nums,
                "answer": float(str(ex["Answer"]).replace(",", "")),
            })
    elif name == "cf_balanced":
        acts_path = REPO / "inference/runs/cf_balanced_student/activations.pt"
        rows = json.load(open(REPO / "unfaithful/cf_balanced.json"))
        acts = torch.load(acts_path, map_location="cpu", weights_only=True).float().numpy()
        types = np.array([r["type"] for r in rows])
        problems = []
        for i, r in enumerate(rows):
            problems.append({
                "src_idx": r["src_idx"], "type": r["type"],
                "question": r["cf_question_concat"].strip().replace("  ", " "),
                "numerals": [float(x) for x in r["cf_subs"]],
                "answer": float(r["cf_answer"]),
            })
    else:
        raise ValueError(name)
    return acts, types, problems


def gold_for_op(numerals: list[float], op: str) -> float | None:
    if len(numerals) < 2:
        return None
    a, b = int(numerals[0]), int(numerals[1])
    if op == "Addition":
        return float(a + b)
    if op == "Subtraction":
        return float(a - b) if a >= b else None
    if op == "Multiplication":
        return float(a * b)
    if op == "Common-Division":
        return float(a // b) if b != 0 and a % b == 0 else None
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Model loading & batched inference (mostly mirrors steering.py)
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
        output_dir="/tmp/_codi_steer_tt", bf16=True, use_lora=True,
        use_prj=True, prj_dim=2048, prj_no_ln=False, prj_dropout=0.0,
        num_latent=6, inf_latent_iterations=6, remove_eos=True, greedy=True,
        model_max_length=512, seed=11,
    )
    model = CODI(margs, targs, lora_cfg)
    sd_path_safe = Path(ckpt_dir).expanduser() / "model.safetensors"
    sd_path_bin = Path(ckpt_dir).expanduser() / "pytorch_model.bin"
    sd = load_file(str(sd_path_safe)) if sd_path_safe.exists() else torch.load(str(sd_path_bin), map_location="cpu")
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
    matches = re.findall(r"-?\d+\.?\d*", text)
    return float(matches[-1]) if matches else float("inf")


class State:
    def __init__(self):
        # Active cells: set of (layer, step) tuples that should fire.
        self.cells: set[tuple[int, int]] = set()
        # Per-(layer, step) steering vector (precomputed centroids).
        self.vec_per_cell: dict[tuple[int, int], torch.Tensor] = {}
        self.scale = 1.0
        self.current_step = -1


def make_hook(state: State, layer_idx: int):
    def _hook(_m, _i, output):
        s = state.current_step
        if (layer_idx, s) not in state.cells:
            return output
        residual = output[0] if isinstance(output, tuple) else output
        v = state.vec_per_cell[(layer_idx, s)]
        residual[:, -1, :] = residual[:, -1, :] + state.scale * v
        return (residual,) + output[1:] if isinstance(output, tuple) else residual
    return _hook


@torch.no_grad()
def run_batch(model, tok, questions: list[str], state: State, max_new_tokens: int, device: str) -> list[str]:
    embed_fn = model.get_embd(model.codi, model.model_name)
    B = len(questions)
    batch = tok(questions, return_tensors="pt", padding="longest").to(device)
    bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device=device)
    input_ids = torch.cat([batch["input_ids"], bot], dim=1)
    attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)

    state.current_step = -1
    out = model.codi(
        input_ids=input_ids, attention_mask=attn,
        use_cache=True, output_hidden_states=True,
    )
    past_kv = out.past_key_values
    latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)
    for s in range(N_LATENT_STEPS):
        state.current_step = s
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=device)], dim=1)
        out = model.codi(
            inputs_embeds=latent, attention_mask=attn,
            use_cache=True, output_hidden_states=True, past_key_values=past_kv,
        )
        past_kv = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent = model.prj(latent)
    state.current_step = -1

    eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device=device))
    output = eot_emb.unsqueeze(0).expand(B, -1, -1)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    pred_tokens = [[] for _ in range(B)]
    for _ in range(max_new_tokens):
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=device)], dim=1)
        step_out = model.codi(
            inputs_embeds=output, attention_mask=attn,
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
# Main
# ---------------------------------------------------------------------------

def bucket_pred(pred: float, problem: dict) -> str:
    matches = [op for op in OPS
               if (g := gold_for_op(problem["numerals"], op)) is not None and pred == g]
    if not matches:
        return "unmatched"
    if len(matches) > 1:
        return "tied"  # rare: e.g., a+b == a-b only when b=0
    return matches[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["svamp", "cf_balanced"], default="svamp")
    p.add_argument("--src", required=True, choices=OPS)
    p.add_argument("--tgt", required=True, choices=OPS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--peak_layer", type=int, default=PEAK_LAYER)
    p.add_argument("--peak_step", type=int, default=PEAK_STEP_IDX)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--base_model", default=DEFAULT_BASE)
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument("--mode", choices=["centroid", "lda"], default="centroid",
                   help="centroid = mu_tgt - mu_src; lda = lda.coef_[tgt] - lda.coef_[src]")
    args = p.parse_args()

    if args.src == args.tgt:
        raise ValueError("src and tgt must differ")

    print(f"loading dataset: {args.dataset}")
    acts, types, problems = load_dataset(args.dataset)
    print(f"  N={len(types)}  acts={acts.shape}")

    # Stratified 50/50 split by operator type.
    train_idx, test_idx = train_test_split(
        np.arange(len(types)), test_size=0.5, random_state=args.seed,
        stratify=types,
    )
    print(f"  train={len(train_idx)}  test={len(test_idx)}")

    # Per-(layer, step) TRAIN-side directions per (src, tgt) op pair.
    # mode='centroid'  → vec = mu_tgt - mu_src
    # mode='lda'       → vec = lda.coef_[tgt] - lda.coef_[src]  (one-vs-rest LR weight diff)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    types_train = types[train_idx]
    L_total = acts.shape[2]
    S_total = acts.shape[1]
    centroids_per_cell = {}  # (layer, step) -> {op: vec or None}
    for L in range(1, L_total):
        for S in range(S_total):
            X = acts[train_idx, S, L, :]
            if args.mode == "centroid":
                mu = {}
                for op in OPS:
                    m = types_train == op
                    if m.sum() == 0:
                        continue
                    mu[op] = X[m].mean(axis=0)
                centroids_per_cell[(L, S)] = mu
            else:  # lda
                lda = LinearDiscriminantAnalysis(solver="svd")
                lda.fit(X, types_train)
                # store per-op LDA classifier weights (one-vs-rest)
                op_to_weight = {}
                for i, cls in enumerate(lda.classes_):
                    op_to_weight[cls] = lda.coef_[i]
                centroids_per_cell[(L, S)] = op_to_weight
    L = args.peak_layer
    S = args.peak_step
    centroids = centroids_per_cell[(L, S)]
    print(f"\nTRAIN centroids at PEAK (layer={L}, step={S+1}):")
    for op in OPS:
        if op in centroids:
            print(f"  {op:<18s}  n={int((types_train==op).sum()):>4d}  ‖μ‖={np.linalg.norm(centroids[op]):.2f}")
    v_src_to_tgt = centroids[args.tgt] - centroids[args.src]
    v_tgt_to_src = centroids[args.src] - centroids[args.tgt]
    print(f"  ‖v_{args.src}→{args.tgt}‖ = {np.linalg.norm(v_src_to_tgt):.2f}")

    # Pick TEST candidates per direction (where target_answer is well-defined and != src).
    def candidates_for(direction_src, direction_tgt):
        out = []
        for i in test_idx:
            pr = problems[i]
            if pr["type"] != direction_src:
                continue
            ta = gold_for_op(pr["numerals"], direction_tgt)
            sa = gold_for_op(pr["numerals"], direction_src)
            if ta is None or sa is None or ta == sa:
                continue
            out.append({**pr, "src_ans": sa, "tgt_ans": ta})
        return out

    cand_fwd = candidates_for(args.src, args.tgt)
    cand_rev = candidates_for(args.tgt, args.src)
    print(f"\nTEST candidates:  {args.src}→{args.tgt}={len(cand_fwd)}   "
          f"{args.tgt}→{args.src}={len(cand_rev)}")

    print("\nloading model...")
    model, tok = load_codi(args.base_model, args.ckpt_dir, device=args.device)
    layer_modules = find_layer_modules(model)

    state = State()
    # Register hooks on every decoder layer; they no-op unless the (layer, step)
    # is in state.cells.
    handles = [
        layer_modules[layer - 1].register_forward_hook(make_hook(state, layer))
        for layer in range(1, L_total)
    ]

    out_results: dict = {"config": vars(args), "directions": {}}

    def build_protocols(direction_label: str):
        """Yield (proto_name, cells_dict) where cells_dict has the steering
        vector for each (layer, step) cell to patch under direction_label."""
        src_op_, tgt_op_ = direction_label.split("->")
        # Peak only.
        peak_v = (centroids_per_cell[(L, S)][tgt_op_]
                  - centroids_per_cell[(L, S)][src_op_])
        yield ("peak_layer{}_step{}".format(L, S+1), {(L, S): peak_v})
        # All cells (every layer x every step) — vector specific per cell.
        all_cells = {}
        for layer in range(1, L_total):
            for step in range(S_total):
                mu = centroids_per_cell.get((layer, step), {})
                if src_op_ in mu and tgt_op_ in mu:
                    all_cells[(layer, step)] = mu[tgt_op_] - mu[src_op_]
        yield ("all_layers_all_steps", all_cells)

    try:
        for direction_label, candidates in [
            (f"{args.src}->{args.tgt}", cand_fwd),
            (f"{args.tgt}->{args.src}", cand_rev),
        ]:
            if not candidates:
                continue
            print(f"\n=== Direction: {direction_label}  (n_test={len(candidates)}) ===")

            # Baseline.
            state.cells = set()
            state.vec_per_cell = {}
            state.scale = 0.0
            preds_base = []
            for st in range(0, len(candidates), args.batch):
                qs = [c["question"] for c in candidates[st:st + args.batch]]
                preds_base.extend(extract_answer_number(t) for t in run_batch(
                    model, tok, qs, state, args.max_new_tokens, args.device,
                ))
            base_buckets = [bucket_pred(p, c) for p, c in zip(preds_base, candidates)]
            src_op = direction_label.split("->")[0]
            tgt_op = direction_label.split("->")[1]
            correct_idx = [i for i, b in enumerate(base_buckets) if b == src_op]
            cands_correct = [candidates[i] for i in correct_idx]
            n_c = len(cands_correct)
            print(f"  baseline-correct: {n_c}/{len(candidates)} ({n_c/len(candidates)*100:.1f}%)")

            dir_results = {
                "n_test_candidates": len(candidates),
                "n_baseline_correct": n_c,
                "protocols": {},
            }
            for proto_name, cells_vec in build_protocols(direction_label):
                # Move vectors to device.
                vec_per_cell = {
                    cell: torch.tensor(v, dtype=torch.bfloat16, device=args.device)
                    for cell, v in cells_vec.items()
                }
                state.vec_per_cell = vec_per_cell
                state.cells = set(vec_per_cell.keys())
                results_per_scale = []
                for scale in args.scales:
                    state.scale = float(scale)
                    preds = []
                    for st in range(0, n_c, args.batch):
                        qs = [c["question"] for c in cands_correct[st:st + args.batch]]
                        preds.extend(extract_answer_number(t) for t in run_batch(
                            model, tok, qs, state, args.max_new_tokens, args.device,
                        ))
                    buckets = [bucket_pred(p, c) for p, c in zip(preds, cands_correct)]
                    cnt = Counter(buckets)
                    results_per_scale.append({
                        "scale": scale,
                        "n": n_c,
                        "counts": {k: cnt.get(k, 0) for k in OPS + ["unmatched", "tied"]},
                    })
                    row = "  ".join(f"{op[:5]}={cnt.get(op,0):>3d}" for op in OPS)
                    print(f"    [{proto_name}]  scale={scale:>5.1f}  {row}  "
                          f"unm={cnt.get('unmatched',0):>3d}   "
                          f"flip_to_tgt={cnt.get(tgt_op,0)}/{n_c} "
                          f"({cnt.get(tgt_op,0)/max(n_c,1)*100:.1f}%)")
                dir_results["protocols"][proto_name] = {
                    "n_cells_patched": len(cells_vec),
                    "scales": results_per_scale,
                }
            out_results["directions"][direction_label] = dir_results
    finally:
        for h in handles:
            h.remove()

    out_path = Path(args.out) if args.out else (
        REPO / "experiments" / f"steering_tt_{args.dataset}_{args.src[:3]}_{args.tgt[:3]}.json"
    )
    out_path.write_text(json.dumps(out_results, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
