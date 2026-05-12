"""Multi-cell joint steering using the template-controlled add↔mul direction.

Single-cell interventions at probe-winning cells produced complete null
results. This script tests whether JOINT intervention at multiple cells
simultaneously flips the model's output. Uses the same vary_operator-derived
add↔mul direction (mean of acts[Mul] − acts[Add] across matched-operand
templates) computed per cell.

Tests four configurations on cf_natural mixed-op problems:
  cells_1 : {m=1's cell}                            (single cell, baseline)
  cells_2 : {m=1, m=2 cells}                         (two cells joint)
  cells_3 : {m=1, m=2, m=3 cells}                    (three cells joint)
  cells_4 : {m=1, m=2, m=3, m=4 cells}               (all four joint)

For each config, sweep α and measure flip rate per (cell_m → target_op).

If joint interventions show effects where single-cell ones don't, the
operator decision is distributed across cells (consistent with the probe
distribution).

Outputs: steer_multicell_varyop_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, os, re, sys, time
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
ACTS_VARYOP = REPO / "experiments" / "computation_probes" / "gsm8k_vary_operator_latent_acts.pt"
REFINED_JSON = REPO / "visualizations-all" / "gpt2-gsm8k" / "operator-probe" / "multi_op_probe_refined_gsm8k.json"
sys.path.insert(0, str(REPO / "codi"))

OUT_JSON = PD / "steer_multicell_varyop_gsm8k.json"
OUT_PDF = PD / "steer_multicell_varyop_gsm8k.pdf"

M_MAX = 4
OPS = ["+", "-", "*", "/"]


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


def main():
    BS = 8

    # Load refined cells
    refined = json.load(open(REFINED_JSON))
    pref = "correct_and_length_matched" if "correct_and_length_matched" in refined else "length_matched"
    cells = {}
    for m in range(1, M_MAX + 1):
        info = refined[pref].get("op", {}).get(str(m))
        if info is None or info["best"] is None:
            info = refined.get("original", {}).get("op", {}).get(str(m))
        if info is None or info["best"] is None: continue
        cells[m] = (info["best"]["step"] - 1, info["best"]["layer"])

    # Vary-op direction per cell
    print(f"loading {ACTS_VARYOP}", flush=True)
    acts = torch.load(ACTS_VARYOP, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(CF_DIR / "gsm8k_vary_operator.json"))
    tmpl = {}
    for i, r in enumerate(rows):
        tmpl.setdefault(r["template_id"], {})[r["type"]] = i
    valid = [t for t, d in tmpl.items() if len(d) == 4]
    add_idx = np.array([tmpl[t]["Addition"] for t in valid])
    mul_idx = np.array([tmpl[t]["Multiplication"] for t in valid])
    v_addmul = (acts[mul_idx] - acts[add_idx]).mean(axis=0)   # (S, L, H)
    norms = np.linalg.norm(v_addmul, axis=-1)
    print(f"  templates used: {len(valid)}; "
          f"per-cell add→mul norm: mean={norms.mean():.2f}, max={norms.max():.2f}")
    print("  cells under intervention:")
    for m in cells:
        s, l = cells[m]
        print(f"    m={m}: step{s+1} L{l}  |v_addmul|={norms[s, l]:.2f}")

    # cf_natural eval
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
    print(f"cf_natural eval problems: {len(eval_data)}")

    # Load CODI
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
    targs = TrainingArguments(output_dir="/tmp/_steermulti", bf16=True,
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

    # HOOK: holds a dict of (block_idx, latent_step) -> direction (tensor) and alpha
    HOOK = {"active": False, "latent_step": -1,
            "plan": {},   # dict {(block_idx, target_step): unit-vec tensor}
            "alpha": 0.0}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            key = (block_idx, HOOK["latent_step"])
            v = HOOK["plan"].get(key)
            if v is None: return output
            h = output[0] if isinstance(output, tuple) else output
            v_dev = v.to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v_dev
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, plan, alpha, max_new=128):
        """plan: dict {(block_idx, latent_step): unit_vec_tensor}. α scalar."""
        B = len(qs)
        HOOK["plan"] = plan; HOOK["alpha"] = float(alpha); HOOK["active"] = True
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

    def classify_any_flip(emit_val, ex_d, target_op):
        """For multi-cell intervention, we don't tie the flip to a specific m.
        Count: matches gold (no flip), matches alt with target_op for some j,
        matches some other alt, or scrambled."""
        if emit_val is None: return "unparsed"
        if abs(emit_val - ex_d["gold"]) < 1e-3: return "baseline"
        for (j, op), v in ex_d["alt"].items():
            if op == target_op and abs(emit_val - v) < 1e-3:
                return f"flip_to_{target_op}_at_m{j}"
        for (j, op), v in ex_d["alt"].items():
            if abs(emit_val - v) < 1e-3: return "other_alt"
        return "scrambled"

    # Build plans: cell_subsets of size 1, 2, 3, 4
    cell_list = sorted(cells.keys())
    # All single-cell plans + cumulative additions
    subsets = []
    for k in range(1, len(cell_list) + 1):
        sub = cell_list[:k]
        subsets.append({"name": f"{k}-cell", "ms": sub})

    qs = [d["q"] for d in eval_data]
    alphas_rel = [-3.0, -1.5, 0, 1.5, 3.0]

    results = {"cells": {m: {"step_1idx": cells[m][0] + 1, "layer": cells[m][1]}
                          for m in cells},
               "v_addmul_norms": {f"m={m}": float(norms[cells[m]])
                                   for m in cells},
               "n_problems": len(eval_data),
               "subsets": [],
               "baseline": None}

    # Baseline
    print("\n=== baseline (no intervention) ===", flush=True)
    base_strs = []
    for s in range(0, len(qs), BS):
        base_strs += run_batch(qs[s:s+BS], plan={}, alpha=0.0)
    base_vals = [codi_extract(s) for s in base_strs]
    n_match = sum(1 for v, d in zip(base_vals, eval_data) if v is not None and abs(v - d["gold"]) < 1e-3)
    print(f"  baseline acc: {n_match}/{len(eval_data)}")
    results["baseline"] = {"n_match_gold": n_match, "n": len(eval_data)}

    t0 = time.time()
    for sub in subsets:
        # Build plan dict: for each m in sub, add direction at (block=L_m, step=s_m).
        plan = {}
        sub_norms = []
        for m in sub["ms"]:
            s_idx, ly = cells[m]
            v = v_addmul[s_idx, ly]
            v_unit = v / max(np.linalg.norm(v), 1e-9)
            plan[(ly, s_idx)] = torch.tensor(v_unit, dtype=torch.float32)
            sub_norms.append(float(norms[s_idx, ly]))
        # α scales: use the MAX cell norm in the subset as reference
        ref_norm = max(sub_norms)
        print(f"\n=== subset {sub['name']}: ms={sub['ms']}  ref_norm={ref_norm:.1f} ===", flush=True)
        sub_res = {"name": sub["name"], "ms": sub["ms"],
                   "cell_norms_in_subset": sub_norms, "ref_norm": ref_norm,
                   "by_alpha_rel": {}}
        for a_rel in alphas_rel:
            alpha_eff = a_rel * ref_norm
            target_op = "*" if alpha_eff > 0 else "+" if alpha_eff < 0 else "*"
            strs = []
            for s in range(0, len(qs), BS):
                strs += run_batch(qs[s:s+BS], plan=plan, alpha=alpha_eff)
            vals = [codi_extract(s) for s in strs]
            cnt = {"baseline": 0, "scrambled": 0, "unparsed": 0, "other_alt": 0}
            flip_per_m = {f"flip_to_{target_op}_at_m{m}": 0 for m in range(1, M_MAX + 1)}
            for v, d in zip(vals, eval_data):
                c = classify_any_flip(v, d, target_op)
                if c in flip_per_m: flip_per_m[c] += 1
                elif c in cnt: cnt[c] += 1
                else: cnt[c] = cnt.get(c, 0) + 1
            cnt.update(flip_per_m)
            cnt["total_flips_to_target"] = sum(flip_per_m.values())
            sub_res["by_alpha_rel"][f"α_rel={a_rel}"] = {
                "alpha_eff": alpha_eff, "target_op": target_op,
                "counts": cnt, "emits": strs[:5],
            }
            tflips = cnt["total_flips_to_target"]
            print(f"  α_rel={a_rel:+.1f}  α_eff={alpha_eff:+.0f}  target={target_op}  "
                  f"baseline={cnt['baseline']}  total_flips_to_{target_op}={tflips}  "
                  f"scram={cnt['scrambled']}  ({time.time()-t0:.0f}s)")
        results["subsets"].append(sub_res)
        OUT_JSON.write_text(json.dumps(results, indent=2))

    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()

    # ---- Plot ----
    with PdfPages(OUT_PDF) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = ("Multi-cell joint add↔mul steering on cf_natural\n\n"
                "Tests whether the operator decision is distributed across multiple\n"
                "probe-winning cells. Single-cell interventions produced null results;\n"
                "this script intervenes at 1, 2, 3, 4 cells SIMULTANEOUSLY.\n\n"
                "Cells (from refined probe, correct+length_matched filter):\n")
        for m in cells:
            s, l = cells[m]
            body += f"  m={m}: step{s+1} L{l}  |v_addmul|={norms[s, l]:.2f}\n"
        body += f"\nEval: {len(eval_data)} mixed-op cf_natural problems\n"
        body += f"Baseline acc: {n_match}/{len(eval_data)}\n\n"
        body += "α scaled by max cell-norm in the subset.\n"
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Multi-cell joint steering setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # One panel: total_flips_to_target vs α_rel for each subset size
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for sub_res in results["subsets"]:
            keys = sorted(sub_res["by_alpha_rel"].keys(), key=lambda k: float(k.split("=")[1]))
            xs = [float(k.split("=")[1]) for k in keys]
            ys = [sub_res["by_alpha_rel"][k]["counts"]["total_flips_to_target"] / len(eval_data)
                  for k in keys]
            ax.plot(xs, ys, "-o", label=sub_res["name"], lw=2)
        ax.axvline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("α (× max cell-norm in subset)")
        ax.set_ylabel("fraction with target-op flip (any marker)")
        ax.set_title("Multi-cell joint steering — total flips to target operator",
                     fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
