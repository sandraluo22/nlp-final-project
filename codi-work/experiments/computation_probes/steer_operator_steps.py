"""Operator-steering sweep across all 6 latent steps at layer 10, three methods.

Cells: (latent step k in {1..6}, layer 10) — six cells total.
For each cell × each method × each src->tgt operator pair:
  A.  CENTROID patching   — replace residual with per-op mean.
  C.  CROSS patching       — replace residual with a real partner activation
                              from a target-op example (matched on magnitude).
  DAS.SUBSPACE patching   — replace ONLY the projection onto the operator
                              subspace, preserving the rest of the source
                              residual. The subspace is the orthonormalized
                              row-space of a 4-class operator logreg fit on
                              the cell's activations.

Operator-flip metric (per src->tgt): on src-op problems with single-op
Equation '(a op b)', fraction of new predictions equal to target_op(a, b).
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
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"

TARGET_LAYER = 10
TARGET_STEPS_0IDX = list(range(6))  # all 6 latent steps
DAS_RANK = 4  # subspace dim for DAS
OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
OP_TO_SYMBOL = {"Addition": "+", "Subtraction": "-",
                "Multiplication": "*", "Common-Division": "/"}


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def parse_equation(eq):
    nums = re.findall(r"-?\d+\.?\d*", eq)
    ops = [c for c in eq if c in "+-*/"]
    if len(nums) != 2 or len(ops) != 1: return None
    try: return float(nums[0]), float(nums[1]), ops[0]
    except: return None


def apply_op(a, b, op):
    try:
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return a / b if b != 0 else None
    except Exception:
        return None
    return None


def fit_das_subspace(X, y_op, rank=DAS_RANK, seed=0):
    """Return Q (rank x H) orthonormal rows spanning the operator subspace,
    and projection matrix P = Q^T Q (H x H)."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs",
                             multi_class="multinomial",
                             random_state=seed).fit(Xs, y_op)
    # coef in scaled space: (4, H). Convert back to raw via division by scale.
    sigma = sc.scale_
    coef_raw = clf.coef_ / sigma   # (4, H), each row is a direction in raw acts
    # Orthonormalize: take top-`rank` left singular vectors of coef_raw.
    U, S, Vt = np.linalg.svd(coef_raw, full_matrices=False)
    Q = Vt[:rank]                  # (rank, H), orthonormal rows
    return Q, clf.score(Xs, y_op)


def main():
    print(f"loading activations {ACTS}", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    print(f"  shape={a.shape}")

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types_full = np.array([t.replace("Common-Divison", "Common-Division") for t in full["Type"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    equations = [ex.get("Equation", "") for ex in full]
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    parsed = [parse_equation(eq) for eq in equations]
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y_op_full = np.array([op_to_idx.get(t, -1) for t in types_full])

    # Centroids per step at layer=10
    print(f"computing per-step centroids at layer={TARGET_LAYER}", flush=True)
    centroids = {}  # (step, op) -> (H,)
    for s_idx in TARGET_STEPS_0IDX:
        for op in OPS:
            mask = types_full == op
            centroids[(s_idx, op)] = a[mask, s_idx, TARGET_LAYER, :].mean(axis=0)

    # DAS subspaces per step
    print(f"fitting DAS subspaces (rank={DAS_RANK}) per step", flush=True)
    Q_per_step = {}  # step -> (rank, H)
    for s_idx in TARGET_STEPS_0IDX:
        valid = y_op_full >= 0
        Xc = a[valid, s_idx, TARGET_LAYER, :]
        yc = y_op_full[valid]
        Q, acc = fit_das_subspace(Xc, yc, rank=DAS_RANK)
        Q_per_step[s_idx] = Q
        print(f"  step {s_idx+1}: 4-class probe acc={acc*100:.1f}%, ||Q||={np.linalg.norm(Q):.2f}")

    # Donor matching
    rng = np.random.default_rng(42)
    def mag_bucket(v):
        v = abs(v)
        if v < 10: return 0
        if v < 100: return 1
        if v < 1000: return 2
        return 3
    buckets = np.array([mag_bucket(g) for g in golds])
    donor_idx = {}
    for src in OPS:
        for tgt in OPS:
            if src == tgt: continue
            assign = np.full(N, -1, dtype=int)
            for b in range(4):
                src_idx = np.where((types_full == src) & (buckets == b))[0]
                tgt_idx = np.where((types_full == tgt) & (buckets == b))[0]
                if len(tgt_idx) == 0:
                    tgt_idx = np.where(types_full == tgt)[0]
                for i in src_idx:
                    assign[i] = int(rng.choice(tgt_idx))
            donor_idx[(src, tgt)] = assign

    # Load model
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
    targs = TrainingArguments(output_dir="/tmp/_sos", bf16=True,
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

    HOOK = {"step": -1, "active": False, "vec": None, "layer": None,
            "p_target": None, "per_batch_vec": None}

    def make_hook(block_idx):
        def fn(_module, _inputs, output):
            if not HOOK["active"] or HOOK["layer"] is None: return output
            if block_idx != HOOK["layer"] - 1: return output
            if HOOK["step"] != HOOK["p_target"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["per_batch_vec"]
            if v is None:
                v = HOOK["vec"]
            v = v.to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i))
               for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, per_batch_vec=None, *, p_target, layer, vec=None,
                   max_new=64):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        if per_batch_vec is not None:
            HOOK["per_batch_vec"] = per_batch_vec.to("cuda", dtype=torch.bfloat16)
        else:
            HOOK["per_batch_vec"] = None
        HOOK.update({"vec": vec, "p_target": p_target, "layer": layer,
                     "active": True, "step": -1})
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
                    if int(next_ids[b].item()) == eos_id: done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK.update({"active": False, "step": -1, "per_batch_vec": None})
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    BS = 16
    def run_full(per_batch_vec_fn=None, *, p_target, layer, vec=None):
        outputs = []
        for s in range(0, N, BS):
            qs = questions[s:s+BS]
            pbv = None
            if per_batch_vec_fn is not None:
                pbv = per_batch_vec_fn(s, s + len(qs))
            strs = run_batch(qs, per_batch_vec=pbv, p_target=p_target,
                              layer=layer, vec=vec)
            outputs += strs
        ints = [codi_extract(s) for s in outputs]
        return outputs, ints

    # Baseline
    print("\n=== Baseline (no patching) ===", flush=True)
    HOOK["active"] = False
    base_strs = []
    for s in range(0, N, BS):
        base_strs += run_batch(questions[s:s+BS], p_target=-99, layer=0,
                                vec=torch.zeros(H))
    base_ints = [codi_extract(s) for s in base_strs]
    base_correct = np.array([
        v is not None and abs(v - golds[i]) < 1e-3 for i, v in enumerate(base_ints)
    ])
    print(f"  baseline accuracy: {base_correct.mean()*100:.1f}% ({base_correct.sum()}/{N})")

    def measure_flip(src_op, tgt_op, ints):
        src_sym = OP_TO_SYMBOL[src_op]
        tgt_sym = OP_TO_SYMBOL[tgt_op]
        n_in = 0; n_tgt = 0; n_src = 0; n_neither = 0
        n_changed = 0; n_correct = 0
        for i in range(N):
            if types_full[i] != src_op: continue
            p = parsed[i]
            if p is None: continue
            a_, b_, op_eq = p
            if op_eq != src_sym: continue
            n_in += 1
            v_tgt = apply_op(a_, b_, tgt_sym)
            v_src = apply_op(a_, b_, src_sym)
            v_pred = ints[i]
            if v_pred is None: n_neither += 1; continue
            tol = 0.5
            hit_tgt = v_tgt is not None and abs(v_pred - v_tgt) < tol
            hit_src = v_src is not None and abs(v_pred - v_src) < tol
            if hit_tgt: n_tgt += 1
            elif hit_src: n_src += 1
            else: n_neither += 1
            if v_pred != base_ints[i]: n_changed += 1
            if v_pred == golds[i]: n_correct += 1
        return {"n": n_in, "n_tgt": n_tgt, "n_src": n_src,
                "n_neither": n_neither, "n_changed": n_changed,
                "n_correct": n_correct,
                "frac_tgt": n_tgt / max(1, n_in),
                "frac_src": n_src / max(1, n_in)}

    summary = {
        "target_layer": TARGET_LAYER, "das_rank": DAS_RANK, "N": N,
        "baseline_accuracy": float(base_correct.mean()),
        "baseline_n_correct": int(base_correct.sum()),
        "steps": [s + 1 for s in TARGET_STEPS_0IDX],
        "A_centroid_patch": {}, "C_cross_patch": {}, "DAS_subspace_patch": {},
    }

    for s_idx in TARGET_STEPS_0IDX:
        s_label = s_idx + 1
        print(f"\n===== Step {s_label} (layer {TARGET_LAYER}) =====", flush=True)

        # A — centroid
        print(" -- A: centroid --", flush=True)
        for src in OPS:
            for tgt in OPS:
                if src == tgt: continue
                vec = torch.tensor(centroids[(s_idx, tgt)], dtype=torch.float32)
                _, ints = run_full(p_target=s_idx, layer=TARGET_LAYER, vec=vec)
                stats = measure_flip(src, tgt, ints)
                print(f"   {src}->{tgt}: n={stats['n']:4d}  "
                      f"to_tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)  "
                      f"to_src={stats['n_src']:3d}  changed={stats['n_changed']:3d}")
                summary["A_centroid_patch"][f"s{s_label}|{src}->{tgt}"] = stats

        # C — cross-patch
        print(" -- C: cross-patch --", flush=True)
        for src in OPS:
            for tgt in OPS:
                if src == tgt: continue
                assign = donor_idx[(src, tgt)]
                def pbv_fn(s_, e_, _assign=assign, _step=s_idx):
                    vecs = a[_assign[s_:e_], _step, TARGET_LAYER, :]
                    return torch.tensor(vecs, dtype=torch.float32)
                _, ints = run_full(per_batch_vec_fn=pbv_fn, p_target=s_idx,
                                    layer=TARGET_LAYER, vec=torch.zeros(H))
                stats = measure_flip(src, tgt, ints)
                print(f"   {src}->{tgt}: n={stats['n']:4d}  "
                      f"to_tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)  "
                      f"to_src={stats['n_src']:3d}  changed={stats['n_changed']:3d}")
                summary["C_cross_patch"][f"s{s_label}|{src}->{tgt}"] = stats

        # DAS — subspace swap
        Q = Q_per_step[s_idx]  # (rank, H)
        print(" -- DAS (rank=%d): subspace swap --" % DAS_RANK, flush=True)
        for src in OPS:
            for tgt in OPS:
                if src == tgt: continue
                assign = donor_idx[(src, tgt)]
                def pbv_fn(s_, e_, _assign=assign, _step=s_idx, _Q=Q):
                    x_src = a[s_:e_, _step, TARGET_LAYER, :]  # (B, H)
                    x_tgt = a[_assign[s_:e_], _step, TARGET_LAYER, :]
                    # patched = x_src - Q^T Q x_src + Q^T Q x_tgt
                    proj_src = x_src @ _Q.T @ _Q
                    proj_tgt = x_tgt @ _Q.T @ _Q
                    patched = x_src - proj_src + proj_tgt
                    return torch.tensor(patched, dtype=torch.float32)
                _, ints = run_full(per_batch_vec_fn=pbv_fn, p_target=s_idx,
                                    layer=TARGET_LAYER, vec=torch.zeros(H))
                stats = measure_flip(src, tgt, ints)
                print(f"   {src}->{tgt}: n={stats['n']:4d}  "
                      f"to_tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)  "
                      f"to_src={stats['n_src']:3d}  changed={stats['n_changed']:3d}")
                summary["DAS_subspace_patch"][f"s{s_label}|{src}->{tgt}"] = stats

    out = PD / "steering_operator_steps.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
