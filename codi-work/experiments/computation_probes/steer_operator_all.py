"""Operator steering — three interventions + attention capture.

A. Centroid patching at the latent cell (step_idx=3 -> step 4 in 1-indexed,
   layer=10). Replaces the last-token residual with the target-operator centroid.

C. Cross-patching at the same latent cell. Pairs source-op example with a
   target-op example matched in magnitude bucket and replaces source's
   residual with the partner's at the same cell.

ATTN. Captures attention from the latent token at (step=4, layer=10) to all
   preceding tokens for the first 200 examples. Saves the mean attention
   to each (token, head) so we can identify which heads attend to operator
   cue words.

Outputs: steering_operator_all.json, attention_operator.npz, attention_operator_meta.json

Operator-flip metric: for SVAMP examples with single-op Equation '( a op b )',
compute expected answers under each operator and report:
  - pred_eq_target: predicted_int equals target_op(a, b)
  - pred_eq_source: predicted_int equals source_op(a, b) (baseline-like)
  - pred_unchanged: predicted_int == baseline_int
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

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
CENTROIDS = REPO / "experiments" / "operator_centroids_layer10_step4.json"

TARGET_STEP_0IDX = 3   # latent step 4 in 1-indexed
TARGET_LAYER = 10
DECODE_POS = 1         # decode position 1 (first emitted token, usually "The")
DECODE_LAYER = 8
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
    """Return (a, b, op_symbol) for single-op '( a op b )' equations, else None."""
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
    print(f"  N={len(questions)}  types: {dict(zip(*np.unique(types_full, return_counts=True)))}")

    # Parse equations
    parsed = [parse_equation(eq) for eq in equations]
    has_2op = np.array([p is not None for p in parsed])
    print(f"  parseable 2-operand equations: {int(has_2op.sum())}/{N}")

    # ---- Compute centroids from activations.pt ----
    print(f"\ncomputing centroids at (step_idx={TARGET_STEP_0IDX}, layer={TARGET_LAYER})")
    centroids_latent = {}  # op -> (H,)
    for op in OPS:
        mask = types_full == op
        cen = a[mask, TARGET_STEP_0IDX, TARGET_LAYER, :].mean(axis=0)
        centroids_latent[op] = cen
        print(f"  {op}: n={int(mask.sum())}, ||c||={np.linalg.norm(cen):.2f}")

    # ---- Match donors for cross-patching (within magnitude bucket) ----
    print("\nbuilding cross-patch donor index")
    def mag_bucket(v):
        v = abs(v)
        if v < 10: return 0
        if v < 100: return 1
        if v < 1000: return 2
        return 3
    buckets = np.array([mag_bucket(g) for g in golds])
    rng = np.random.default_rng(42)
    donor_idx = {}  # (src_op, tgt_op) -> array of donor indices for each example
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

    # ---- Load CODI-GPT-2 ----
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
    targs = TrainingArguments(output_dir="/tmp/_so", bf16=True,
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

    # ---- Patching hook (replace, not add). Supports latent or decode phase. ----
    HOOK = {"phase": "off", "step": -1, "active": False, "vec": None,
            "layer": None, "target_phase": None, "p_target": None,
            "per_batch_vec": None}

    def make_hook(block_idx):
        def fn(_module, _inputs, output):
            if not HOOK["active"] or HOOK["layer"] is None: return output
            if block_idx != HOOK["layer"] - 1: return output
            if HOOK["phase"] != HOOK["target_phase"]: return output
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
    def run_batch(qs, per_batch_vec=None, *, target_phase, p_target, layer,
                   vec=None, capture_attn_at=None, max_new=64):
        """
        target_phase: "latent" or "decode" — which loop to apply patching at.
        p_target: position within that phase (latent step 0..5, or decode iter 0..N).
        capture_attn_at: tuple (phase, step, layer) for attention capture.
        per_batch_vec: tensor (B, H) — if set, used instead of vec (cross-patch).
        """
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        if per_batch_vec is not None:
            HOOK["per_batch_vec"] = per_batch_vec.to("cuda", dtype=torch.bfloat16)
        else:
            HOOK["per_batch_vec"] = None
        HOOK.update({"vec": vec, "target_phase": target_phase, "p_target": p_target,
                     "layer": layer, "active": True, "step": -1, "phase": "off"})

        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        attn_captured = None
        HOOK["phase"] = "latent"
        for step in range(targs.inf_latent_iterations):
            HOOK["step"] = step
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            need_attn = (capture_attn_at is not None
                         and capture_attn_at[0] == "latent"
                         and step == capture_attn_at[1])
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             output_attentions=need_attn,
                             past_key_values=past)
            past = out.past_key_values
            if need_attn:
                blk_attn = out.attentions[capture_attn_at[2] - 1]
                attn_captured = blk_attn[:, :, -1, :].cpu().float().numpy()
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)

        HOOK["phase"] = "decode"
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for dec_step in range(max_new):
            HOOK["step"] = dec_step
            need_attn = (capture_attn_at is not None
                         and capture_attn_at[0] == "decode"
                         and dec_step == capture_attn_at[1])
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              output_attentions=need_attn,
                              past_key_values=past)
            past = sout.past_key_values
            if need_attn:
                blk_attn = sout.attentions[capture_attn_at[2] - 1]
                attn_captured = blk_attn[:, :, -1, :].cpu().float().numpy()
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if not done[b]:
                    tokens[b].append(int(next_ids[b].item()))
                    if int(next_ids[b].item()) == eos_id:
                        done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK.update({"active": False, "step": -1, "phase": "off", "per_batch_vec": None})
        strs = [tok.decode(t, skip_special_tokens=True) for t in tokens]
        return strs, attn_captured

    def run_full(per_batch_vec_fn=None, *, target_phase, p_target, layer,
                 vec=None, BS=16, N_eval=None):
        if N_eval is None: N_eval = len(questions)
        outputs = []
        for s in range(0, N_eval, BS):
            qs = questions[s:s+BS]
            pbv = None
            if per_batch_vec_fn is not None:
                pbv = per_batch_vec_fn(s, s + len(qs))
            strs, _ = run_batch(qs, per_batch_vec=pbv, target_phase=target_phase,
                                 p_target=p_target, layer=layer, vec=vec)
            outputs += strs
        ints = [codi_extract(s) for s in outputs]
        return outputs, ints

    # ---- Baseline ----
    print("\n=== Baseline (no patching) ===", flush=True)
    HOOK["active"] = False
    base_strs, base_ints = run_full(target_phase="off", p_target=-99,
                                     layer=0, vec=torch.zeros(H))
    base_correct = np.array([
        v is not None and abs(v - golds[i]) < 1e-3 for i, v in enumerate(base_ints)
    ])
    print(f"  baseline accuracy: {base_correct.mean()*100:.1f}% ({base_correct.sum()}/{N})")

    # Flip-metric helper
    def measure_flip(src_op, tgt_op, ints):
        """For each parseable single-op example with type==src_op, check if
        prediction matches target_op(a, b) or source_op(a, b)."""
        src_sym = OP_TO_SYMBOL[src_op]
        tgt_sym = OP_TO_SYMBOL[tgt_op]
        n_in = 0
        n_tgt = 0  # pred == tgt_op(a, b)
        n_src = 0  # pred == src_op(a, b) (preserved)
        n_neither = 0
        n_changed = 0
        n_correct = 0
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
                "n_correct": n_correct}

    summary = {
        "target_step_1indexed": TARGET_STEP_0IDX + 1,
        "target_layer": TARGET_LAYER,
        "N": N,
        "baseline_accuracy": float(base_correct.mean()),
        "baseline_n_correct": int(base_correct.sum()),
        "baseline_ints": base_ints,
        "A_centroid_patch": {},
        "C_cross_patch": {},
    }

    summary["A_centroid_patch_decode"] = {}
    summary["C_cross_patch_decode"] = {}

    # ---- A_latent: Centroid patching at LATENT cell ----
    print(f"\n=== A: Centroid patching at latent (step={TARGET_STEP_0IDX+1}, "
          f"layer={TARGET_LAYER}) ===", flush=True)
    for src in OPS:
        for tgt in OPS:
            if src == tgt: continue
            vec = torch.tensor(centroids_latent[tgt], dtype=torch.float32)
            strs, ints = run_full(target_phase="latent",
                                   p_target=TARGET_STEP_0IDX,
                                   layer=TARGET_LAYER, vec=vec)
            stats = measure_flip(src, tgt, ints)
            stats["frac_tgt"] = stats["n_tgt"] / max(1, stats["n"])
            stats["frac_src"] = stats["n_src"] / max(1, stats["n"])
            print(f"  {src} -> {tgt}: n={stats['n']:4d}  "
                  f"to_tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)  "
                  f"to_src={stats['n_src']:3d} ({stats['frac_src']*100:.1f}%)  "
                  f"neither={stats['n_neither']:3d}  changed={stats['n_changed']:3d}")
            summary["A_centroid_patch"][f"{src}->{tgt}"] = stats

    # ---- C_latent: Cross-patching at LATENT cell ----
    print(f"\n=== C: Cross-patching at latent (step={TARGET_STEP_0IDX+1}, "
          f"layer={TARGET_LAYER}) ===", flush=True)
    for src in OPS:
        for tgt in OPS:
            if src == tgt: continue
            assign = donor_idx[(src, tgt)]
            def pbv_fn(s_, e_, _assign=assign):
                vecs = a[_assign[s_:e_], TARGET_STEP_0IDX, TARGET_LAYER, :]
                return torch.tensor(vecs, dtype=torch.float32)
            strs, ints = run_full(per_batch_vec_fn=pbv_fn,
                                   target_phase="latent",
                                   p_target=TARGET_STEP_0IDX, layer=TARGET_LAYER,
                                   vec=torch.zeros(H))
            stats = measure_flip(src, tgt, ints)
            stats["frac_tgt"] = stats["n_tgt"] / max(1, stats["n"])
            stats["frac_src"] = stats["n_src"] / max(1, stats["n"])
            print(f"  {src} -> {tgt}: n={stats['n']:4d}  "
                  f"to_tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)  "
                  f"to_src={stats['n_src']:3d} ({stats['frac_src']*100:.1f}%)  "
                  f"neither={stats['n_neither']:3d}  changed={stats['n_changed']:3d}")
            summary["C_cross_patch"][f"{src}->{tgt}"] = stats

    # ---- Decode-cell centroids: need svamp_multipos_decode_acts.pt ----
    multipos_path = PD / "svamp_multipos_decode_acts.pt"
    if multipos_path.exists():
        print(f"\nloading multipos decode activations {multipos_path}")
        mp_acts = torch.load(multipos_path, map_location="cpu",
                             weights_only=True).float().numpy()
        # shape (N, P, L+1, H)
        print(f"  shape={mp_acts.shape}")
        centroids_decode = {}
        for op in OPS:
            mask = types_full == op
            cen = mp_acts[mask, DECODE_POS, DECODE_LAYER, :].mean(axis=0)
            centroids_decode[op] = cen
            print(f"  centroid {op}: n={int(mask.sum())}, ||c||={np.linalg.norm(cen):.2f}")

        # A_decode: centroid replace at decode iter=DECODE_POS, layer=DECODE_LAYER
        print(f"\n=== A_decode: Centroid patching at decode (pos={DECODE_POS}, "
              f"layer={DECODE_LAYER}) ===", flush=True)
        for src in OPS:
            for tgt in OPS:
                if src == tgt: continue
                vec = torch.tensor(centroids_decode[tgt], dtype=torch.float32)
                strs, ints = run_full(target_phase="decode",
                                       p_target=DECODE_POS,
                                       layer=DECODE_LAYER, vec=vec)
                stats = measure_flip(src, tgt, ints)
                stats["frac_tgt"] = stats["n_tgt"] / max(1, stats["n"])
                stats["frac_src"] = stats["n_src"] / max(1, stats["n"])
                print(f"  {src} -> {tgt}: n={stats['n']:4d}  "
                      f"to_tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)  "
                      f"to_src={stats['n_src']:3d} ({stats['frac_src']*100:.1f}%)  "
                      f"neither={stats['n_neither']:3d}  changed={stats['n_changed']:3d}")
                summary["A_centroid_patch_decode"][f"{src}->{tgt}"] = stats

        # C_decode: cross-patch using multipos acts as donor pool
        print(f"\n=== C_decode: Cross-patching at decode (pos={DECODE_POS}, "
              f"layer={DECODE_LAYER}) ===", flush=True)
        for src in OPS:
            for tgt in OPS:
                if src == tgt: continue
                assign = donor_idx[(src, tgt)]
                def pbv_fn(s_, e_, _assign=assign):
                    vecs = mp_acts[_assign[s_:e_], DECODE_POS, DECODE_LAYER, :]
                    return torch.tensor(vecs, dtype=torch.float32)
                strs, ints = run_full(per_batch_vec_fn=pbv_fn,
                                       target_phase="decode",
                                       p_target=DECODE_POS, layer=DECODE_LAYER,
                                       vec=torch.zeros(H))
                stats = measure_flip(src, tgt, ints)
                stats["frac_tgt"] = stats["n_tgt"] / max(1, stats["n"])
                stats["frac_src"] = stats["n_src"] / max(1, stats["n"])
                print(f"  {src} -> {tgt}: n={stats['n']:4d}  "
                      f"to_tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)  "
                      f"to_src={stats['n_src']:3d} ({stats['frac_src']*100:.1f}%)  "
                      f"neither={stats['n_neither']:3d}  changed={stats['n_changed']:3d}")
                summary["C_cross_patch_decode"][f"{src}->{tgt}"] = stats
    else:
        print(f"\nNOTE: {multipos_path} not found — skipping decode-cell interventions.")
        print("Run run_multipos_decode_acts_gpt2.py first to generate it.")

    out = PD / "steering_operator_all.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}", flush=True)

    # ---- Attention capture: latent + decode cells ----
    def capture_attention(phase, step, layer, N_ATTN=200, BS=8, out_tag=""):
        print(f"\n=== Attention capture at {phase} step={step}, layer={layer} ===", flush=True)
        all_attn = []
        all_tokens = []
        max_k = 0
        for s in range(0, N_ATTN, BS):
            qs = questions[s:s+BS]
            HOOK["active"] = False
            strs_, attn_arr = run_batch(qs, target_phase="off",
                                          p_target=-99, layer=0, vec=torch.zeros(H),
                                          capture_attn_at=(phase, step, layer))
            if attn_arr is None:
                print("WARNING: attention not captured")
                return None
            for q in qs:
                tids = tok(q, return_tensors="pt").input_ids[0].tolist()
                tids += [model.bot_id]
                all_tokens.append(tids)
            all_attn.append(attn_arr)
            max_k = max(max_k, attn_arr.shape[-1])
        padded = np.zeros((sum(x.shape[0] for x in all_attn),
                           all_attn[0].shape[1], max_k), dtype=np.float32)
        i = 0
        for arr in all_attn:
            padded[i:i+arr.shape[0], :, -arr.shape[-1]:] = arr
            i += arr.shape[0]
        np.savez(PD / f"attention_operator_{out_tag}.npz", attn=padded)
        meta = {"phase": phase, "step": step, "layer": layer,
                "n": padded.shape[0], "heads": padded.shape[1], "k_len": padded.shape[2],
                "tokens": all_tokens[:padded.shape[0]],
                "decoded_tokens": [[tok.decode([t]) for t in ts]
                                   for ts in all_tokens[:padded.shape[0]]],
                "types": types_full[:padded.shape[0]].tolist()}
        (PD / f"attention_operator_{out_tag}_meta.json").write_text(json.dumps(meta))
        print(f"saved attention_operator_{out_tag}.npz  shape={padded.shape}")
        return padded

    capture_attention("latent", TARGET_STEP_0IDX, TARGET_LAYER, out_tag="latent")
    if multipos_path.exists():
        capture_attention("decode", DECODE_POS, DECODE_LAYER, out_tag="decode")

    for h in handles: h.remove()


if __name__ == "__main__":
    main()
