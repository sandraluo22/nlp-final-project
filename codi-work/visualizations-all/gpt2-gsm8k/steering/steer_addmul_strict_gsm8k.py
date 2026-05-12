"""Causal add↔mul steering on gsm8k_cf_op_strict.

Unlike lda_probe_and_steer_addmul_gsm8k.py which re-classifies shifted activations
through the LDA, this script:
  1. Computes the LDA add↔mul direction at the operator-probe-winning cell
     (latent_step=1, layer=11) from gsm8k_latent_acts.pt.
  2. Filters gsm8k_cf_op_strict to Addition and Multiplication examples.
  3. For each example, computes candidate answers:
       answer_add = reduce_+(operands)
       answer_mul = reduce_*(operands)
       answer_sub = operands[0] - sum(operands[1:])
       answer_div = operands[0] / reduce_*(operands[1:])
  4. For each α, hooks layer 11 during latent_step 1's forward pass and adds
     α·v to the last-token residual. Then runs the remaining latent steps,
     decodes the answer, parses the emitted numeric answer, and scores it
     against the candidates.

Output: steer_addmul_strict_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, math, os, re, sys, time
from collections import Counter
from pathlib import Path
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
sys.path.insert(0, str(REPO / "codi"))

# Winner cell from lda_probe_and_steer_addmul_gsm8k.json: step=1, layer=11
PROBE_LATENT_STEP_1INDEXED = 1   # 1-indexed; means index 0 in 0-indexed loop
PROBE_LAYER = 11

OUT_JSON = PD / "steer_addmul_strict_gsm8k.json"
OUT_PDF = PD / "steer_addmul_strict_gsm8k.pdf"


def codi_extract(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def candidate_answers(operands):
    """Compute answer if the chain were applied with each operator."""
    try:
        operands = [float(x) for x in operands]
    except Exception:
        return None
    if len(operands) < 2: return None
    add = reduce(lambda a, b: a + b, operands)
    mul = reduce(lambda a, b: a * b, operands)
    sub = operands[0] - sum(operands[1:])
    div_rest = reduce(lambda a, b: a * b, operands[1:])
    div = operands[0] / div_rest if div_rest != 0 else None
    return {"add": add, "mul": mul, "sub": sub, "div": div}


def parse_first_op(answer_trace: str):
    """Extract the first operator from a <<a op b = c>> chain."""
    s = answer_trace.replace(",", "")
    m = re.search(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=", s)
    return m.group(2) if m else None


def fit_lda_addmul():
    """Compute add↔mul LDA direction at (PROBE_LATENT_STEP, PROBE_LAYER) using GSM8K test."""
    print("computing LDA add↔mul direction...", flush=True)
    ds = load_dataset("gsm8k", "main")["test"]
    types = []
    for ex in ds:
        m = re.search(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=", ex["answer"])
        if m is None: types.append(None); continue
        op = m.group(2)
        types.append({"+": "Addition", "-": "Subtraction",
                      "*": "Multiplication", "/": "Common-Division"}.get(op))
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = acts.shape
    X = acts[:, PROBE_LATENT_STEP_1INDEXED - 1, PROBE_LAYER, :]
    y = np.array([t if t else "" for t in types])
    mask = (y != "")
    sc = StandardScaler().fit(X[mask])
    Xs = sc.transform(X[mask])
    lda = LinearDiscriminantAnalysis(solver="svd").fit(Xs, y[mask])
    # The "add → mul" direction: difference between class means projected back.
    classes = list(lda.classes_)
    add_idx = classes.index("Addition")
    mul_idx = classes.index("Multiplication")
    add_mean = lda.means_[add_idx]
    mul_mean = lda.means_[mul_idx]
    v_std = mul_mean - add_mean   # direction in standardized space
    v_orig = sc.scale_ * v_std    # unscale
    v_unit = v_orig / np.linalg.norm(v_orig)
    print(f"  LDA fit on {mask.sum()} GSM8K test problems; "
          f"add↔mul direction norm={np.linalg.norm(v_orig):.3f}, "
          f"unit-norm direction extracted.")
    return v_unit, lda, sc


def main():
    BS = 8

    # 1) Compute steering direction
    v_unit, lda, sc = fit_lda_addmul()
    v_torch = torch.tensor(v_unit, dtype=torch.float32)

    # 2) Load CF + filter Add/Mul + candidate answers
    rows = json.load(open(CF_DIR / "gsm8k_cf_op_strict.json"))
    add_rows = [r for r in rows if r["type"] == "Addition"]
    mul_rows = [r for r in rows if r["type"] == "Multiplication"]
    print(f"cf_op_strict: addition={len(add_rows)}, multiplication={len(mul_rows)}")

    def prep(rows):
        out = []
        for r in rows:
            c = candidate_answers(r["operands"])
            if c is None: continue
            out.append({
                "q": r["question_concat"].strip().replace("  ", " "),
                "operands": r["operands"],
                "type": r["type"],
                "gold": float(r["answer"]),
                "cand": c,
                "n_steps": len(r["operands"]) - 1,
            })
        # Cap to 100 each for tractability
        return out[:100]

    add_eval = prep(add_rows)
    mul_eval = prep(mul_rows)
    print(f"  evaluating: ADD={len(add_eval)}, MUL={len(mul_eval)}")

    # 3) Load CODI
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=["c_attn", "c_proj", "c_fc"],
                          init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_steerstrict", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=768,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=6,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False); model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16); model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id
    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)

    # Hook intervenes at (latent_step == PROBE_LATENT_STEP_1INDEXED - 1, layer = PROBE_LAYER).
    # We index latent_step from 0 (first latent step is 0).
    HOOK = {"active": False, "latent_step": -1, "alpha": 0.0}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if block_idx != PROBE_LAYER: return output
            if HOOK["latent_step"] != PROBE_LATENT_STEP_1INDEXED - 1: return output
            h = output[0] if isinstance(output, tuple) else output
            v = v_torch.to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, alpha, max_new=128):
        B = len(qs)
        HOOK["alpha"] = float(alpha); HOOK["active"] = True
        HOOK["latent_step"] = -1
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for s in range(targs.inf_latent_iterations):
            HOOK["latent_step"] = s
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        HOOK["latent_step"] = -1
        HOOK["active"] = False
        # decode (no steering during decode)
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]; done = [False] * B
        for _ in range(max_new):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, past_key_values=past)
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
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    def eval_set(eval_data, alpha):
        qs = [d["q"] for d in eval_data]
        strs = []
        for s in range(0, len(qs), BS):
            strs += run_batch(qs[s:s+BS], alpha)
        return strs

    def score(strs, eval_data):
        n = len(strs); ok = 0
        out = {"n": n, "n_parseable": 0,
               "match_add": 0, "match_mul": 0, "match_sub": 0, "match_div": 0,
               "match_gold": 0, "match_none": 0,
               "match_add_only": 0, "match_mul_only": 0}
        for s, d in zip(strs, eval_data):
            v = codi_extract(s)
            if v is None: continue
            out["n_parseable"] += 1
            c = d["cand"]; gold = d["gold"]; eps = 1e-3
            ma = abs(v - c["add"]) < eps
            mm = abs(v - c["mul"]) < eps
            ms = abs(v - c["sub"]) < eps
            md = c["div"] is not None and abs(v - c["div"]) < eps
            mg = abs(v - gold) < eps
            if ma: out["match_add"] += 1
            if mm: out["match_mul"] += 1
            if ms: out["match_sub"] += 1
            if md: out["match_div"] += 1
            if mg: out["match_gold"] += 1
            if not (ma or mm or ms or md): out["match_none"] += 1
            if ma and not mm: out["match_add_only"] += 1
            if mm and not ma: out["match_mul_only"] += 1
        return out

    alphas = [-100, -50, -25, -10, 0, 10, 25, 50, 100]   # in units of unit-vector length

    results = {
        "winner": {"latent_step_1indexed": PROBE_LATENT_STEP_1INDEXED, "layer": PROBE_LAYER},
        "v_norm": float(np.linalg.norm(v_unit)),
        "n_add": len(add_eval), "n_mul": len(mul_eval),
        "by_alpha": {},
        "sample_outputs": {},
    }

    t0 = time.time()
    for alpha in alphas:
        print(f"\n=== α = {alpha} ===", flush=True)
        s_add = eval_set(add_eval, alpha)
        s_mul = eval_set(mul_eval, alpha)
        sc_add = score(s_add, add_eval)
        sc_mul = score(s_mul, mul_eval)
        results["by_alpha"][str(alpha)] = {"add": sc_add, "mul": sc_mul}
        results["sample_outputs"][str(alpha)] = {
            "add": [{"gold_type": "Addition", "operands": add_eval[i]["operands"],
                     "cand": add_eval[i]["cand"], "gold": add_eval[i]["gold"],
                     "emit": s_add[i], "parsed": codi_extract(s_add[i])}
                    for i in range(min(5, len(add_eval)))],
            "mul": [{"gold_type": "Multiplication", "operands": mul_eval[i]["operands"],
                     "cand": mul_eval[i]["cand"], "gold": mul_eval[i]["gold"],
                     "emit": s_mul[i], "parsed": codi_extract(s_mul[i])}
                    for i in range(min(5, len(mul_eval)))],
        }
        print(f"  ADD problems: kept_add={sc_add['match_add']}/{sc_add['n']}  "
              f"flipped_to_mul={sc_add['match_mul']}/{sc_add['n']}  "
              f"none={sc_add['match_none']}/{sc_add['n']}  ({time.time()-t0:.0f}s)")
        print(f"  MUL problems: kept_mul={sc_mul['match_mul']}/{sc_mul['n']}  "
              f"flipped_to_add={sc_mul['match_add']}/{sc_mul['n']}  "
              f"none={sc_mul['match_none']}/{sc_mul['n']}")
        OUT_JSON.write_text(json.dumps(results, indent=2))

    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    by_alpha = results["by_alpha"]
    a_keys = sorted(by_alpha.keys(), key=lambda x: float(x))
    a_vals = [float(a) for a in a_keys]
    add_kept = [by_alpha[a]["add"]["match_add"] / by_alpha[a]["add"]["n"] for a in a_keys]
    add_flip = [by_alpha[a]["add"]["match_mul"] / by_alpha[a]["add"]["n"] for a in a_keys]
    mul_kept = [by_alpha[a]["mul"]["match_mul"] / by_alpha[a]["mul"]["n"] for a in a_keys]
    mul_flip = [by_alpha[a]["mul"]["match_add"] / by_alpha[a]["mul"]["n"] for a in a_keys]
    axes[0].plot(a_vals, add_kept, "-o", color="C0", label="match ADD answer (kept)")
    axes[0].plot(a_vals, add_flip, "-s", color="C3", label="match MUL answer (flipped)")
    axes[0].axvline(0, color="gray", ls="--", alpha=0.5)
    axes[0].set_xlabel("α (× unit-norm direction)"); axes[0].set_ylabel("fraction of N")
    axes[0].set_title("ADD problems → steered toward MUL", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(a_vals, mul_kept, "-o", color="C1", label="match MUL answer (kept)")
    axes[1].plot(a_vals, mul_flip, "-s", color="C2", label="match ADD answer (flipped)")
    axes[1].axvline(0, color="gray", ls="--", alpha=0.5)
    axes[1].set_xlabel("α (× unit-norm direction)"); axes[1].set_ylabel("fraction of N")
    axes[1].set_title("MUL problems → steered toward ADD (negative α)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig.suptitle(f"Causal add↔mul steering on gsm8k_cf_op_strict — "
                 f"intervene at (latent step {PROBE_LATENT_STEP_1INDEXED}, layer {PROBE_LAYER})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
