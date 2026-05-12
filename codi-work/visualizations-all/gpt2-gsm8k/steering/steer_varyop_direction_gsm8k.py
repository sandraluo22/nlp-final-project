"""Causal steering using a template-controlled "add↔mul direction".

Instead of LDA on heterogeneous problems, compute the add→mul direction at
each (step, layer) cell as the MEAN ACTIVATION DIFFERENCE between Mul-version
and Add-version of the SAME template in gsm8k_vary_operator. Each template
fixes (a, b); only the operator varies. So the activation difference is
PURELY due to the operation, not confounded by operand magnitudes, narrative,
or chain length.

Procedure:
  1. Collect activations on the 320-problem gsm8k_vary_operator CF set
     (80 templates × 4 ops) at every (latent_step, layer, last_token).
     Use the pre-saved gsm8k_vary_operator_latent_acts.pt if available;
     else run CODI fresh.
  2. For each cell C = (step, layer):
       v_addmul[C] = mean_t (acts[t, Mul] − acts[t, Add])
     averaged over 80 templates.
  3. Use refined-probe cells (per marker m) as intervention points. At each
     marker's cell, hook in α · v_addmul[cell_m] during the matching latent
     step. Decode and score against per-(marker_steered, target_op) chain
     alternatives, as in steer_per_marker_natural_gsm8k.py.

This direction is CLEANER than the LDA direction because:
  - LDA's mul_mean − add_mean was computed across heterogeneous GSM8K problems
    with different operands and different chain lengths.
  - This direction is computed on matched-operand templates: the only thing
    that differs between Mul and Add is the operator.

Outputs: steer_varyop_direction_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, os, re, sys, time
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
ACTS_VARYOP_PATH = REPO / "experiments" / "computation_probes" / "gsm8k_vary_operator_latent_acts.pt"
REFINED_JSON = REPO / "visualizations-all" / "gpt2-gsm8k" / "operator-probe" / "multi_op_probe_refined_gsm8k.json"
sys.path.insert(0, str(REPO / "codi"))

OUT_JSON = PD / "steer_varyop_direction_gsm8k.json"
OUT_PDF = PD / "steer_varyop_direction_gsm8k.pdf"

M_MAX = 4
OPS = ["+", "-", "*", "/"]
OP_NAMES = {"Addition": "+", "Subtraction": "-", "Multiplication": "*", "Common-Division": "/"}
NAME_TO_IDX = {"Addition": 0, "Subtraction": 1, "Multiplication": 2, "Common-Division": 3}


def parse_markers(s):
    s = s.replace(",", "")
    ms = re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)
    return [(float(a), op, float(b), float(c)) for a, op, b, c in ms]


def codi_extract(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def apply_op(a, op, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b if b != 0 else None
    return None


def chain_answer(markers, ops_override=None):
    if not markers: return None
    cur = markers[0][0]
    for k, (a, op, b, c) in enumerate(markers):
        use_op = (ops_override[k] if ops_override else op)
        cur = apply_op(cur, use_op, b)
        if cur is None: return None
    if abs(cur - round(cur)) < 1e-6: cur = round(cur)
    return float(cur)


def compute_varyop_directions(acts, rows):
    """acts: (320, S, L, H). rows: list of 320 cf rows.
    For each cell (step, layer), compute v_addmul = mean_template(mul - add).
    Returns dict[(step, layer)] = unit-norm direction, plus norms."""
    N, S, L, H = acts.shape
    # Build per-template, per-op index map
    tmpl = {}
    for i, r in enumerate(rows):
        tmpl.setdefault(r["template_id"], {})[r["type"]] = i
    valid = [t for t, d in tmpl.items() if len(d) == 4]
    print(f"  templates with all 4 op variants: {len(valid)}")

    add_idx = np.array([tmpl[t]["Addition"] for t in valid])
    mul_idx = np.array([tmpl[t]["Multiplication"] for t in valid])
    # acts[mul] - acts[add] averaged over templates
    diff_addmul = acts[mul_idx] - acts[add_idx]   # (T, S, L, H)
    v_addmul = diff_addmul.mean(axis=0)            # (S, L, H)
    norms = np.linalg.norm(v_addmul, axis=-1)      # (S, L)
    return v_addmul, norms, len(valid)


def main():
    BS = 8

    # ---- Load refined probe cells ----
    refined = json.load(open(REFINED_JSON))
    pref_filter = "correct_and_length_matched" if "correct_and_length_matched" in refined else \
                  "correct_only" if "correct_only" in refined else "length_matched"
    cells = {}
    for m in range(1, M_MAX + 1):
        info = refined[pref_filter].get("op", {}).get(str(m))
        if info is None or info["best"] is None:
            info = refined.get("original", {}).get("op", {}).get(str(m))
        if info is None or info["best"] is None: continue
        cells[m] = {"step_1idx": info["best"]["step"], "layer": info["best"]["layer"],
                    "probe_acc": info["best"]["acc"]}
        print(f"  m={m}: cell = step{cells[m]['step_1idx']} L{cells[m]['layer']}  "
              f"probe_acc={cells[m]['probe_acc']:.3f}")

    # ---- Load activations for vary_operator ----
    if not ACTS_VARYOP_PATH.exists():
        raise FileNotFoundError(f"missing {ACTS_VARYOP_PATH}")
    print(f"loading vary_operator activations from {ACTS_VARYOP_PATH}", flush=True)
    acts = torch.load(ACTS_VARYOP_PATH, map_location="cpu", weights_only=True).float().numpy()
    print(f"  shape={acts.shape}")
    rows = json.load(open(CF_DIR / "gsm8k_vary_operator.json"))
    assert len(rows) == acts.shape[0], f"row count mismatch {len(rows)} vs {acts.shape[0]}"

    # ---- Compute add↔mul direction per cell from template differences ----
    v_addmul, dnorm, n_templates = compute_varyop_directions(acts, rows)
    print(f"  template-controlled add↔mul direction norms (per cell):")
    print(f"    max norm = {dnorm.max():.2f}  at (step{np.argmax(dnorm.max(axis=1)) + 1}, "
          f"L{np.argmax(dnorm[np.argmax(dnorm.max(axis=1))])})")
    print(f"    mean norm across cells = {dnorm.mean():.2f}")
    for m, c in cells.items():
        s_idx = c["step_1idx"] - 1
        ly = c["layer"]
        print(f"    m={m} cell norm = {dnorm[s_idx, ly]:.2f}")

    # ---- Load cf_natural eval set ----
    cf_rows = json.load(open(CF_DIR / "gsm8k_cf_natural.json"))
    eval_data = []
    for r in cf_rows:
        ms_tr = parse_markers(r["answer_trace"])
        if not (2 <= len(ms_tr) <= 4): continue
        if len(set(op for _, op, _, _ in ms_tr)) < 2: continue
        base = chain_answer(ms_tr)
        if base is None or abs(base - float(r["answer"])) > 1e-3: continue
        K = len(ms_tr)
        orig_ops = [op for _, op, _, _ in ms_tr]
        alt = {}
        for j in range(K):
            for new_op in OPS:
                if new_op == orig_ops[j]: continue
                ovr = list(orig_ops); ovr[j] = new_op
                v = chain_answer(ms_tr, ovr)
                if v is None: continue
                alt[(j + 1, new_op)] = v
        eval_data.append({"q": r["question_concat"].strip().replace("  ", " "),
                          "orig_ops": orig_ops, "gold": float(r["answer"]),
                          "K": K, "alt": alt})
    eval_data = eval_data[:80]
    print(f"  cf_natural mixed-op eval problems: {len(eval_data)}")

    # ---- Load CODI ----
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
    targs = TrainingArguments(output_dir="/tmp/_steervo", bf16=True,
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

    HOOK = {"active": False, "latent_step": -1, "target_step": -1,
            "target_layer": -1, "alpha": 0.0, "vec": None}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if block_idx != HOOK["target_layer"]: return output
            if HOOK["latent_step"] != HOOK["target_step"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, target_step, target_layer, alpha, vec, max_new=128):
        B = len(qs)
        HOOK["target_step"] = target_step; HOOK["target_layer"] = target_layer
        HOOK["alpha"] = float(alpha); HOOK["vec"] = vec
        HOOK["active"] = True; HOOK["latent_step"] = -1
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
        HOOK["active"] = False; HOOK["latent_step"] = -1
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

    def classify(emit_val, ex_d, m_steered, target_op):
        if emit_val is None: return "unparsed"
        if abs(emit_val - ex_d["gold"]) < 1e-3: return "baseline"
        v_t = ex_d["alt"].get((m_steered, target_op))
        if v_t is not None and abs(emit_val - v_t) < 1e-3: return "target"
        for (j, op), v in ex_d["alt"].items():
            if op == target_op and j != m_steered and abs(emit_val - v) < 1e-3: return "off_target"
        for (j, op), v in ex_d["alt"].items():
            if abs(emit_val - v) < 1e-3: return "other_alt"
        return "scrambled"

    # ---- Sweep at each m's cell using the vary_operator direction ----
    qs = [d["q"] for d in eval_data]
    # alphas chosen relative to the direction's own norm; scale to e.g. 0.5x, 1x, 2x cell-norm
    alphas_rel = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]

    results = {"cells": cells, "n_templates_used": int(n_templates),
               "v_addmul_norms": {f"step{s+1}_L{l}": float(dnorm[s, l])
                                  for s in range(dnorm.shape[0]) for l in range(dnorm.shape[1])},
               "n_problems": len(eval_data),
               "per_m": {}}

    # Baseline
    print("\n=== baseline (α=0) ===", flush=True)
    dummy_v = torch.zeros(768, dtype=torch.float32)
    base_strs = []
    for s in range(0, len(qs), BS):
        base_strs += run_batch(qs[s:s+BS], target_step=-99, target_layer=-99, alpha=0.0, vec=dummy_v)
    base_vals = [codi_extract(s) for s in base_strs]
    n_match_gold = sum(1 for v, d in zip(base_vals, eval_data) if v is not None and abs(v - d["gold"]) < 1e-3)
    print(f"  baseline acc: {n_match_gold}/{len(eval_data)}")
    results["baseline"] = {"n_match_gold": n_match_gold, "n": len(eval_data),
                            "emits": base_strs[:8]}

    t0 = time.time()
    for m_steered in cells:
        c = cells[m_steered]
        s_idx = c["step_1idx"] - 1
        ly = c["layer"]
        v_raw = torch.tensor(v_addmul[s_idx, ly], dtype=torch.float32)
        cell_norm = float(dnorm[s_idx, ly])
        # Use unit direction; α expressed as multiplier on the cell's own diff-norm.
        v_unit = v_raw / max(cell_norm, 1e-9)
        results["per_m"][m_steered] = {"cell_norm": cell_norm}
        print(f"\n=== m={m_steered} cell step{c['step_1idx']} L{ly}  diff_norm={cell_norm:.2f} ===")
        for alpha_rel in alphas_rel:
            alpha_eff = alpha_rel * cell_norm   # ranges roughly ±3 × |add→mul direction|
            target_op = "*" if alpha_eff > 0 else "+"
            strs = []
            for s in range(0, len(qs), BS):
                strs += run_batch(qs[s:s+BS], target_step=s_idx, target_layer=ly,
                                  alpha=alpha_eff, vec=v_unit)
            vals = [codi_extract(s) for s in strs]
            counts = {"baseline": 0, "target": 0, "off_target": 0,
                      "other_alt": 0, "scrambled": 0, "unparsed": 0}
            for v, d in zip(vals, eval_data):
                counts[classify(v, d, m_steered, target_op)] += 1
            results["per_m"][m_steered][f"α_rel={alpha_rel}"] = {
                "alpha_eff": alpha_eff, "target_op": target_op,
                "counts": counts, "emits": strs[:5],
            }
            print(f"  α_rel={alpha_rel:+.2f} (α_eff={alpha_eff:+.1f}, target={target_op}):  "
                  f"target={counts['target']}  off={counts['off_target']}  "
                  f"other_alt={counts['other_alt']}  base={counts['baseline']}  "
                  f"scram={counts['scrambled']}  ({time.time()-t0:.0f}s)")
            OUT_JSON.write_text(json.dumps(results, indent=2))

    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()

    # ---- Plot ----
    with PdfPages(OUT_PDF) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = ("Causal steering via template-controlled add↔mul direction\n\n"
                f"  Direction source: gsm8k_vary_operator  (80 templates × 4 ops)\n"
                f"  v_addmul[cell] = mean_template (acts[Mul] − acts[Add])\n"
                f"  → controlled for (a, b), narrative, chain length\n\n"
                f"  Intervention cells (from refined probe, "
                f"correct_and_length_matched filter):\n")
        for m, c in cells.items():
            key = f"step{c['step_1idx']}_L{c['layer']}"
            vnorm = results["v_addmul_norms"][key]
            body += (f"    m={m}: step{c['step_1idx']} L{c['layer']}  "
                     f"probe_acc={c['probe_acc']:.3f}  "
                     f"|v_addmul|={vnorm:.2f}\n")
        body += f"\n  Eval: {len(eval_data)} mixed-op cf_natural problems\n"
        body += f"  Baseline (α=0) acc: {n_match_gold}/{len(eval_data)}\n\n"
        body += "α scaled relative to cell's own direction norm.  α_rel=1 ≈ full add→mul shift.\n"
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # per-m panels
        nm = len(cells)
        fig, axes = plt.subplots(1, max(1, nm), figsize=(5 * max(1, nm), 5))
        if nm == 1: axes = [axes]
        for col, m in enumerate(cells):
            data = results["per_m"].get(m, {})
            ax = axes[col]
            keys = sorted([k for k in data.keys() if k.startswith("α_rel=")],
                          key=lambda x: float(x.split("=")[1]))
            xs = [float(k.split("=")[1]) for k in keys]
            n = len(eval_data)
            for label, color in [("baseline", "C0"), ("target", "C2"),
                                  ("off_target", "C3"), ("scrambled", "C7")]:
                ys = [data[k]["counts"][label] / n for k in keys]
                ax.plot(xs, ys, "-o", color=color, label=label)
            ax.axvline(0, color="gray", ls="--", alpha=0.5)
            ax.set_xlabel("α (× |v_addmul|)"); ax.set_ylabel("fraction")
            ax.set_title(f"m={m}: cell step{cells[m]['step_1idx']} L{cells[m]['layer']}",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
        fig.suptitle("Vary-operator-direction steering: did the model's emit follow?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
