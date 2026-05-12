"""Causality test for operator steering: does adding the steering vector
make the model's emitted answer match a DIFFERENT operator?

Method:
- For each SVAMP example, parse two operands a, b from the Equation field
  and compute add(a,b), sub(a,b), mul(a,b), div(a,b).
- Run baseline + each operator-steering sweep (add->sub, sub->add, add->mul, etc.)
  at alpha=8 on 200 examples, capture the integer the model emits.
- Classify each output as which operator's result it matches (or 'other').
- Report transition matrix: baseline_op_match -> steered_op_match.
"""

from __future__ import annotations
import json, os, re, sys
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


def parse_int(s):
    m = re.search(r"answer is\s*:\s*(-?\d+)", s)
    if not m: return None
    try: return int(m.group(1))
    except: return None


def parse_two_operands(equation):
    """SVAMP Equation looks like '( 67.0 + 96.0 )' or 'a / b'. Pull out two
    numeric operands. If we can't find exactly two, return None."""
    nums = re.findall(r"-?\d+\.?\d*", equation)
    if len(nums) < 2: return None
    try:
        a, b = float(nums[0]), float(nums[1])
        return a, b
    except: return None


def candidate_results(a, b):
    """Return dict of operator -> integer-rounded result (None if undefined)."""
    out = {}
    out["add"] = round(a + b)
    out["sub"] = round(a - b)
    out["mul"] = round(a * b)
    out["div"] = round(a / b) if b != 0 else None
    return out


def classify_match(emitted, cands, tol=0):
    """Return list of operators whose result the emitted int matches."""
    if emitted is None: return []
    ms = []
    for op, r in cands.items():
        if r is None: continue
        if abs(emitted - r) <= tol: ms.append(op)
    return ms


def fit_full_probe(X, y):
    mask = y >= 0
    X = X[mask]; y = y[mask]
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=4000, C=1.0,
                                  solver="lbfgs").fit(sc.transform(X), y)


def main():
    print("loading multi-pos decode activations + preds...", flush=True)
    acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N, P, Lp1, H = acts.shape

    ds = load_dataset("gsm8k")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    equations = [ex["Equation"] for ex in full]
    answers = [int(round(float(str(ex["Answer"]).replace(",", "")))) for ex in full]

    # Pick best operator cell from JSON
    pj = json.load(open(PD / "gpt2_multipos_probes_modelans.json"))
    op_acc = np.array(pj["operator_acc"])
    op_pos, op_lay = np.unravel_index(int(np.argmax(op_acc)), op_acc.shape)
    print(f"  op cell: pos {op_pos}, layer {op_lay}, acc {op_acc[op_pos, op_lay]*100:.1f}%")

    sc_op, clf_op = fit_full_probe(acts[:, op_pos, op_lay, :], operators)

    def get_steer_vec(c_target, c_source):
        sigma = sc_op.scale_; W = clf_op.coef_
        return torch.tensor(sigma * (W[c_target] - W[c_source]), dtype=torch.float32)

    # ---- Load CODI ----
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
    targs = TrainingArguments(output_dir="/tmp/_opana", bf16=True,
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
    HOOK = {"step": -1, "active": False, "vec": None, "p_target": None,
            "layer": None, "alpha": 0.0}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"] or HOOK["layer"] is None: return output
            if block_idx != HOOK["layer"] - 1: return output
            if HOOK["step"] != HOOK["p_target"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, vec, alpha, layer, p_target):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"vec": vec, "p_target": p_target, "layer": layer,
                     "alpha": alpha, "active": True, "step": -1})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for _ in range(targs.inf_latent_iterations):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        for step in range(12):
            HOOK["step"] = step
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B): tokens[b].append(int(next_ids[b].item()))
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK.update({"active": False, "step": -1})
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    # filter to examples with parseable two-operand equations
    keep_idx = []
    cands_per = []
    for i in range(N):
        ab = parse_two_operands(equations[i])
        if ab is None: continue
        a, b = ab
        c = candidate_results(a, b)
        cands_per.append(c)
        keep_idx.append(i)
    print(f"  parseable two-operand examples: {len(keep_idx)}/{N}")
    eval_idx = np.random.RandomState(0).choice(len(keep_idx), size=min(200, len(keep_idx)), replace=False)
    eval_qs = [questions[keep_idx[i]] for i in eval_idx]
    eval_cands = [cands_per[i] for i in eval_idx]
    eval_gold_op = [operators[keep_idx[i]] for i in eval_idx]
    eval_gold_ans = [answers[keep_idx[i]] for i in eval_idx]
    OPS = ["add", "sub", "mul", "div"]
    print(f"  eval set N={len(eval_qs)}, gold-op dist:")
    for op_name, op_idx in [("add", 0), ("sub", 1), ("mul", 2), ("div", 3)]:
        n = sum(1 for o in eval_gold_op if o == op_idx)
        print(f"    {op_name}: {n}")

    BS = 16
    def run_full(vec, alpha):
        strs = []
        for s in range(0, len(eval_qs), BS):
            strs += run_batch(eval_qs[s:s+BS], vec=vec, alpha=alpha,
                              layer=op_lay, p_target=op_pos)
        return strs

    # baseline
    print("\n=== Baseline ===", flush=True)
    base_strs = run_full(torch.zeros(H), 0.0)
    base_ints = [parse_int(s) for s in base_strs]
    base_match = [classify_match(b, c) for b, c in zip(base_ints, eval_cands)]
    n_match_baseline = sum(1 for ms in base_match if ms)
    print(f"  baseline: {n_match_baseline}/{len(eval_qs)} match SOME operator")
    for op_name in OPS:
        n = sum(1 for ms in base_match if op_name in ms)
        print(f"    matches {op_name}: {n}")

    # operator sweeps
    SWEEPS = [
        ("add->sub", 1, 0),
        ("sub->add", 0, 1),
        ("add->mul", 2, 0),
        ("sub->mul", 2, 1),
        ("add->div", 3, 0),
    ]
    out_summary = {"baseline_match_counts": {op: sum(1 for ms in base_match if op in ms) for op in OPS},
                   "n_eval": len(eval_qs)}

    for name, tgt, src in SWEEPS:
        v = get_steer_vec(tgt, src)
        for alpha in [4.0, 8.0, 12.0]:
            strs = run_full(v, alpha)
            ints = [parse_int(s) for s in strs]
            match = [classify_match(i, c) for i, c in zip(ints, eval_cands)]
            print(f"\n=== {name}  alpha={alpha} ===", flush=True)
            print(f"  total match-some: {sum(1 for m in match if m)}/{len(eval_qs)}")
            for op in OPS:
                base_n = sum(1 for ms in base_match if op in ms)
                steer_n = sum(1 for ms in match if op in ms)
                # transition: was matching X in baseline, now matching tgt
                base_only_op = [i for i, ms in enumerate(base_match) if op in ms]
                base_only_op_to_tgt = sum(1 for i in base_only_op if OPS[tgt] in match[i])
                arrow = "->" + OPS[tgt] if op == OPS[src] else ""
                print(f"  matches {op}: {base_n} -> {steer_n}  "
                      f"(of those originally matching {op}, {base_only_op_to_tgt} now match {OPS[tgt]})")
            out_summary[f"{name}|alpha={alpha}"] = {
                "match_total": sum(1 for m in match if m),
                "match_per_op": {op: sum(1 for ms in match if op in ms) for op in OPS},
                "preds": strs[:30],
                "ints": ints[:30],
            }

    out = PD / "steering_operator_causality.json"
    out.write_text(json.dumps(out_summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
