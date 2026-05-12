"""Operator-steering at the ':' residual (decode position 4, where ':' is the
input being processed). Three methods: A (centroid), C (cross-patch),
DAS (rank-4 operator subspace).
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

REPO = Path(__file__).resolve().parents[3]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

COLON_ACTS = PD / "gsm8k_colon_acts.pt"
COLON_META = PD / "gsm8k_colon_acts_meta.json"

DECODE_POS = 4   # position where ':' is the input
TARGET_LAYER = 9 # operator-probe peak at colon (layer 9, acc 93.0%)
DAS_RANK = 4
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


def main():
    print(f"loading colon acts {COLON_ACTS}", flush=True)
    a = torch.load(COLON_ACTS, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(COLON_META))
    N, L, H = a.shape
    print(f"  shape={a.shape}")

    ds = load_dataset("gsm8k", "main")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array([t.replace("Common-Divison", "Common-Division") for t in full["Type"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    equations = [ex.get("Equation", "") for ex in full]
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    parsed = [parse_equation(eq) for eq in equations]
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y_op = np.array([op_to_idx.get(t, -1) for t in types])

    # Per-op centroids at (layer=TARGET_LAYER) from colon acts
    print(f"computing centroids at layer={TARGET_LAYER}", flush=True)
    centroids = {}
    for op in OPS:
        mask = types == op
        centroids[op] = a[mask, TARGET_LAYER, :].mean(axis=0)

    # DAS subspace
    valid = y_op >= 0
    sc = StandardScaler().fit(a[valid, TARGET_LAYER, :])
    Xs = sc.transform(a[valid, TARGET_LAYER, :])
    clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs").fit(Xs, y_op[valid])
    sigma = sc.scale_
    coef_raw = clf.coef_ / sigma
    U, S, Vt = np.linalg.svd(coef_raw, full_matrices=False)
    Q = Vt[:DAS_RANK]
    print(f"  DAS subspace shape={Q.shape}; probe acc={clf.score(Xs, y_op[valid])*100:.1f}%")

    # Donor index (match by magnitude bucket)
    def mag_bucket(v):
        v = abs(v)
        if v < 10: return 0
        if v < 100: return 1
        if v < 1000: return 2
        return 3
    buckets = np.array([mag_bucket(g) for g in golds])
    rng = np.random.default_rng(42)
    donor_idx = {}
    for src in OPS:
        for tgt in OPS:
            if src == tgt: continue
            assign = np.full(N, -1, dtype=int)
            for b in range(4):
                tgt_idx = np.where((types == tgt) & (buckets == b))[0]
                if len(tgt_idx) == 0: tgt_idx = np.where(types == tgt)[0]
                src_idx = np.where((types == src) & (buckets == b))[0]
                for i in src_idx: assign[i] = int(rng.choice(tgt_idx))
            donor_idx[(src, tgt)] = assign

    # Load model
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"\nloading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *args, **k: _orig(*args, **{**k, "use_fast": True}))
    from src.model import CODI, ModelArguments, TrainingArguments
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=["c_attn", "c_proj", "c_fc"],
                          init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_sc", bf16=True,
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

    HOOK = {"phase": "off", "step": -1, "active": False, "vec": None, "layer": None,
            "target_phase": None, "p_target": None, "per_batch_vec": None}

    def make_hook(block_idx):
        def fn(_module, _inputs, output):
            if not HOOK["active"] or HOOK["layer"] is None: return output
            if block_idx != HOOK["layer"] - 1: return output
            if HOOK["phase"] != HOOK["target_phase"]: return output
            if HOOK["step"] != HOOK["p_target"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["per_batch_vec"] if HOOK["per_batch_vec"] is not None else HOOK["vec"]
            v = v.to(h.device, dtype=h.dtype)
            h = h.clone(); h[:, -1, :] = v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, per_batch_vec=None, *, p_target, layer, vec=None, max_new=24):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK["per_batch_vec"] = per_batch_vec.to("cuda", dtype=torch.bfloat16) if per_batch_vec is not None else None
        HOOK.update({"vec": vec, "target_phase": "decode", "p_target": p_target,
                     "layer": layer, "active": True, "step": -1, "phase": "off"})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        HOOK["phase"] = "latent"
        for step in range(targs.inf_latent_iterations):
            HOOK["step"] = step
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        HOOK["phase"] = "decode"
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]; done = [False] * B
        for dec_step in range(max_new):
            HOOK["step"] = dec_step
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if done[b]: continue
                tid = int(next_ids[b].item()); tokens[b].append(tid)
                if tid == eos_id: done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK.update({"active": False, "phase": "off", "per_batch_vec": None})
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    BS = 16
    def run_full(per_batch_vec_fn=None, *, p_target, layer, vec=None):
        outs = []
        for s in range(0, N, BS):
            qs = questions[s:s+BS]
            pbv = per_batch_vec_fn(s, s + len(qs)) if per_batch_vec_fn else None
            outs += run_batch(qs, per_batch_vec=pbv, p_target=p_target, layer=layer, vec=vec)
        return outs, [codi_extract(s) for s in outs]

    print("\n=== Baseline (no patching) ===", flush=True)
    HOOK["active"] = False
    base_strs, base_ints = run_full(p_target=-99, layer=0, vec=torch.zeros(H))
    base_correct = np.array([v is not None and abs(v - golds[i]) < 1e-3
                              for i, v in enumerate(base_ints)])
    print(f"  baseline accuracy: {base_correct.mean()*100:.1f}%")

    def measure_flip(src, tgt, ints):
        src_sym = OP_TO_SYMBOL[src]; tgt_sym = OP_TO_SYMBOL[tgt]
        n_in = n_tgt = n_src = n_neither = 0
        for i in range(N):
            if types[i] != src: continue
            p = parsed[i]
            if p is None or p[2] != src_sym: continue
            n_in += 1
            v_tgt = apply_op(p[0], p[1], tgt_sym)
            v_src = apply_op(p[0], p[1], src_sym)
            v_pred = ints[i]
            if v_pred is None: n_neither += 1; continue
            tol = 0.5
            if v_tgt is not None and abs(v_pred - v_tgt) < tol: n_tgt += 1
            elif v_src is not None and abs(v_pred - v_src) < tol: n_src += 1
            else: n_neither += 1
        return {"n": n_in, "n_tgt": n_tgt, "n_src": n_src, "n_neither": n_neither,
                "frac_tgt": n_tgt / max(1, n_in), "frac_src": n_src / max(1, n_in)}

    summary = {"target_decode_pos": DECODE_POS, "target_layer": TARGET_LAYER,
               "das_rank": DAS_RANK, "N": N,
               "baseline_accuracy": float(base_correct.mean()),
               "A_centroid_patch": {}, "C_cross_patch": {}, "DAS_subspace_patch": {}}

    print(f"\n=== A: Centroid patching at ':' (decode pos {DECODE_POS}, layer {TARGET_LAYER}) ===")
    for src in OPS:
        for tgt in OPS:
            if src == tgt: continue
            vec = torch.tensor(centroids[tgt], dtype=torch.float32)
            _, ints = run_full(p_target=DECODE_POS, layer=TARGET_LAYER, vec=vec)
            stats = measure_flip(src, tgt, ints)
            print(f"  {src}->{tgt}: n={stats['n']:4d}  tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)  "
                  f"src={stats['n_src']:3d} neither={stats['n_neither']:3d}")
            summary["A_centroid_patch"][f"{src}->{tgt}"] = stats

    print(f"\n=== C: Cross-patching at ':' ===")
    for src in OPS:
        for tgt in OPS:
            if src == tgt: continue
            assign = donor_idx[(src, tgt)]
            def pbv_fn(s_, e_, _assign=assign):
                vecs = a[_assign[s_:e_], TARGET_LAYER, :]
                return torch.tensor(vecs, dtype=torch.float32)
            _, ints = run_full(per_batch_vec_fn=pbv_fn, p_target=DECODE_POS,
                                layer=TARGET_LAYER, vec=torch.zeros(H))
            stats = measure_flip(src, tgt, ints)
            print(f"  {src}->{tgt}: n={stats['n']:4d}  tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)")
            summary["C_cross_patch"][f"{src}->{tgt}"] = stats

    print(f"\n=== DAS (rank={DAS_RANK}): subspace swap at ':' ===")
    for src in OPS:
        for tgt in OPS:
            if src == tgt: continue
            assign = donor_idx[(src, tgt)]
            def pbv_fn(s_, e_, _assign=assign):
                x_src = a[s_:e_, TARGET_LAYER, :]
                x_tgt = a[_assign[s_:e_], TARGET_LAYER, :]
                proj_src = x_src @ Q.T @ Q
                proj_tgt = x_tgt @ Q.T @ Q
                return torch.tensor(x_src - proj_src + proj_tgt, dtype=torch.float32)
            _, ints = run_full(per_batch_vec_fn=pbv_fn, p_target=DECODE_POS,
                                layer=TARGET_LAYER, vec=torch.zeros(H))
            stats = measure_flip(src, tgt, ints)
            print(f"  {src}->{tgt}: n={stats['n']:4d}  tgt={stats['n_tgt']:3d} ({stats['frac_tgt']*100:.1f}%)")
            summary["DAS_subspace_patch"][f"{src}->{tgt}"] = stats

    out = PD / "steering_operator_colon.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
