"""Probe-direction steering using MODEL-EMITTED-LABEL probes.

Differences from steer_codi_gpt2.py:
- Probes are trained against the model's emitted digit (not gold).
- Digit-steering is constrained to (pos in {1..4}, any layer) so the
  intervention happens BEFORE the answer digit is emitted at pos 4. Steering
  at pos>=5 cannot change pos 4's output.
- Operator steering still uses the global best cell (not constrained).
- We measure effect at decode pos 4 (the first-digit-token) AND on the parsed
  full integer the model emits.
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


def parse_model_answer(s):
    m = re.search(r"answer is\s*:\s*(-?\d+)", s)
    if not m: return None
    try: return int(m.group(1))
    except: return None


def fit_full_probe(X, y):
    mask = y >= 0
    X = X[mask]; y = y[mask]
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=4000, C=1.0,
                                  solver="lbfgs").fit(sc.transform(X), y)


def get_steering_vec(sc, clf, c_target, c_original=None):
    sigma = sc.scale_
    W = clf.coef_
    if c_original is None:
        return torch.tensor(sigma * W[c_target], dtype=torch.float32)
    return torch.tensor(sigma * (W[c_target] - W[c_original]), dtype=torch.float32)


def main():
    print("loading multi-pos decode activations + preds...", flush=True)
    acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N, P, Lp1, H = acts.shape
    print(f"  acts shape {acts.shape}", flush=True)
    preds = json.load(open(PD / "svamp_multipos_decode_preds.json"))

    # model labels
    model_ans_py = [parse_model_answer(s) for s in preds["preds"]]
    model_units = np.array([(abs(v) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)
    model_tens  = np.array([((abs(v)//10) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)

    ds = load_dataset("gsm8k", "main")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    answers = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])

    # Pick best cell per variable.
    pj = json.load(open(PD / "gpt2_multipos_probes_modelans.json"))
    units_acc = np.array(pj["model_units_acc"])
    tens_acc = np.array(pj["model_tens_acc"])
    op_acc = np.array(pj["operator_acc"])

    # operator: best anywhere
    op_pos, op_lay = np.unravel_index(int(np.argmax(op_acc)), op_acc.shape)
    # digits: best in pos 1..4 (steering must be pre-emission)
    M_units = units_acc.copy(); M_units[5:] = 0
    M_tens  = tens_acc.copy();  M_tens[5:]  = 0
    units_pos, units_lay = np.unravel_index(int(np.argmax(M_units)), M_units.shape)
    tens_pos,  tens_lay  = np.unravel_index(int(np.argmax(M_tens)),  M_tens.shape)
    print(f"  op   cell: pos {op_pos} layer {op_lay}  ({op_acc[op_pos, op_lay]*100:.1f}%)")
    print(f"  units cell (pos<=4): pos {units_pos} layer {units_lay}  ({units_acc[units_pos, units_lay]*100:.1f}%)")
    print(f"  tens  cell (pos<=4): pos {tens_pos} layer {tens_lay}  ({tens_acc[tens_pos, tens_lay]*100:.1f}%)")

    sc_op,    clf_op    = fit_full_probe(acts[:, op_pos, op_lay, :], operators)
    sc_units, clf_units = fit_full_probe(acts[:, units_pos, units_lay, :], model_units)
    sc_tens,  clf_tens  = fit_full_probe(acts[:, tens_pos, tens_lay, :], model_tens)

    # ---- Load CODI-GPT-2 ----
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_steer2", bf16=True,
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
    def run_batch(batch_q, *, p_target, layer, vec, alpha):
        B = len(batch_q)
        batch = tok(batch_q, return_tensors="pt", padding="longest").to("cuda")
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
        return tokens

    N_eval = 100
    eval_idx = np.random.RandomState(0).choice(N, size=N_eval, replace=False)
    eval_qs = [questions[i] for i in eval_idx]
    eval_gold = [int(round(answers[i])) for i in eval_idx]
    BS = 16

    def run_full(p, l, vec, alpha):
        toks = []
        for s in range(0, N_eval, BS):
            toks += run_batch(eval_qs[s:s+BS], p_target=p, layer=l, vec=vec, alpha=alpha)
        strs = [tok.decode(t, skip_special_tokens=True) for t in toks]
        digits_at_pos4 = [int(t[4]) if 4 < len(t) else None for t in toks]
        ints = [parse_model_answer(s) for s in strs]
        return strs, digits_at_pos4, ints

    print("\n=== Baseline ===", flush=True)
    base_strs, base_d4, base_int = run_full(0, 0, torch.zeros(H), 0.0)
    base_units = [(abs(v) % 10) if v is not None else None for v in base_int]
    base_tens = [((abs(v)//10) % 10) if v is not None else None for v in base_int]
    print(f"  baseline first 5 outputs:")
    for i in range(5):
        print(f"    gold={eval_gold[i]:>5}  pred={base_strs[i][:40]!r}  int={base_int[i]}")

    sweeps = [
        ("operator_add->sub", op_pos, op_lay, sc_op, clf_op, 1, 0),
        ("operator_sub->add", op_pos, op_lay, sc_op, clf_op, 0, 1),
        ("operator_add->mul", op_pos, op_lay, sc_op, clf_op, 2, 0),
        ("tens_to_5",  tens_pos,  tens_lay,  sc_tens,  clf_tens,  5, None),
        ("tens_to_8",  tens_pos,  tens_lay,  sc_tens,  clf_tens,  8, None),
        ("units_to_3", units_pos, units_lay, sc_units, clf_units, 3, None),
        ("units_to_7", units_pos, units_lay, sc_units, clf_units, 7, None),
        ("units_to_9", units_pos, units_lay, sc_units, clf_units, 9, None),
    ]
    alphas = [1.0, 2.0, 4.0, 8.0]
    summary = {"baseline_units_dist": {int(k): int(v) for k, v in zip(*np.unique([u for u in base_units if u is not None], return_counts=True))},
               "baseline_tens_dist":  {int(k): int(v) for k, v in zip(*np.unique([t for t in base_tens if t is not None], return_counts=True))}}

    for name, p, l, sc, clf, tgt, src in sweeps:
        v = get_steering_vec(sc, clf, tgt, src)
        print(f"\n=== {name}  (pos {p}, layer {l}, target={tgt}, source={src}) ===", flush=True)
        for a in alphas:
            strs, d4, ints = run_full(p, l, v, a)
            new_units = [(abs(v) % 10) if v is not None else None for v in ints]
            new_tens = [((abs(v)//10) % 10) if v is not None else None for v in ints]
            n_changed_int = sum(1 for i in range(N_eval) if ints[i] != base_int[i])
            n_changed_d4 = sum(1 for i in range(N_eval) if d4[i] != base_d4[i])
            n_units_to_target = sum(1 for u in new_units if u == tgt) if name.startswith("units") else None
            n_tens_to_target = sum(1 for t in new_tens if t == tgt) if name.startswith("tens") else None
            print(f"  alpha={a}:  changed_full={n_changed_int}/{N_eval}  changed_d4={n_changed_d4}/{N_eval}", end="")
            if n_units_to_target is not None:
                print(f"  units==target({tgt}): {n_units_to_target}/{N_eval}", end="")
            if n_tens_to_target is not None:
                print(f"  tens==target({tgt}): {n_tens_to_target}/{N_eval}", end="")
            print(flush=True)
            print(f"    examples: gold={eval_gold[0]} base_int={base_int[0]} steered_int={ints[0]} | "
                  f"gold={eval_gold[1]} base_int={base_int[1]} steered_int={ints[1]} | "
                  f"gold={eval_gold[2]} base_int={base_int[2]} steered_int={ints[2]}")
            summary[f"{name}|alpha={a}"] = {
                "n_changed_full": n_changed_int,
                "n_changed_d4": n_changed_d4,
                "n_units_to_target": n_units_to_target,
                "n_tens_to_target": n_tens_to_target,
                "preds": strs[:20],
            }

    out = PD / "steering_modellabel_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
