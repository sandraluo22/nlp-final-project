"""Steering experiment for CODI-GPT-2.

Pipeline:
1. Load svamp_multipos_decode_acts.pt (1000, 16, 13, 768) and labels.
2. For each variable in {operator, tens, units}, fit a multinomial logistic-
   regression at the (best position, best layer) cell. The fitted probe
   coefficient W has shape (n_classes, 768) — each row is a direction in
   residual space that *increases* the log-prob of that class.
3. Define steering vector v = W[c_target] - W[c_original] (or, for the
   "free" version, just W[c_target] which pushes toward c_target).
4. Run CODI-GPT-2 inference on N_eval examples WITH a forward hook that adds
   alpha * v to the residual at (chosen layer, chosen position). Record the
   model's emitted token at the answer position and check if it changed in
   the predicted direction.
5. Sweep alpha for each variable + report results.
"""

from __future__ import annotations
import json, os, sys, time
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


def fit_full_probe(X, y):
    """Fit on all data, return (StandardScaler, LogisticRegression)."""
    mask = y >= 0
    X = X[mask]; y = y[mask]
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs").fit(Xs, y)
    return sc, clf


def best_cell(M):
    """Return (pos, layer) of the max value in a (P, L+1) accuracy matrix."""
    p, l = np.unravel_index(int(np.argmax(M)), M.shape)
    return int(p), int(l)


def get_steering_vec(sc, clf, c_target, c_original=None):
    """
    Steering vector in raw residual space.
    The probe operates on standardized features:
      logit_c = w_c · ((x - mu) / sigma) + b_c
    A unit step in standardized space corresponds to (sigma * w_c) in raw x.
    So a steering direction in raw space is: sigma * (w_target - w_original).
    If c_original is None, just push toward target.
    """
    sigma = sc.scale_  # (768,)
    W = clf.coef_      # (n_classes, 768)
    if c_original is None:
        v = sigma * W[c_target]
    else:
        v = sigma * (W[c_target] - W[c_original])
    return torch.tensor(v, dtype=torch.float32)


def main():
    # ---- 1. Load activations + labels --------------------------------
    print("loading multi-pos decode activations...", flush=True)
    acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N, P, Lp1, H = acts.shape
    print(f"  acts shape {acts.shape}", flush=True)

    ds = load_dataset("gsm8k")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2, "common-division": 3}
    operators = np.array([op_map.get(ex["Type"].lower().replace("divison","division"), -1) for ex in full])
    answers = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    units = np.array([int(abs(int(round(a))) % 10) for a in answers])
    tens = np.array([int((abs(int(round(a))) // 10) % 10) for a in answers])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]

    # ---- 2. Pick best (pos, layer) per variable ---------------------
    probes_json = json.load(open(PD / "gpt2_multipos_probes.json"))
    op_acc = np.array(probes_json["operator_acc"])
    tens_acc = np.array(probes_json["tens_acc"])
    units_acc = np.array(probes_json["units_acc"])
    op_pos, op_lay = best_cell(op_acc)
    tens_pos, tens_lay = best_cell(tens_acc)
    units_pos, units_lay = best_cell(units_acc)
    print(f"  op   best cell: pos {op_pos}, layer {op_lay}, acc {op_acc[op_pos, op_lay]*100:.1f}%")
    print(f"  tens best cell: pos {tens_pos}, layer {tens_lay}, acc {tens_acc[tens_pos, tens_lay]*100:.1f}%")
    print(f"  units best cell: pos {units_pos}, layer {units_lay}, acc {units_acc[units_pos, units_lay]*100:.1f}%")

    # ---- 3. Fit full probes (no held-out) and compute steering dirs --
    sc_op,    clf_op    = fit_full_probe(acts[:, op_pos, op_lay, :], operators)
    sc_tens,  clf_tens  = fit_full_probe(acts[:, tens_pos, tens_lay, :], tens)
    sc_units, clf_units = fit_full_probe(acts[:, units_pos, units_lay, :], units)

    # ---- 4. Load CODI-GPT-2 ------------------------------------------
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
    targs = TrainingArguments(output_dir="/tmp/_steer", bf16=True,
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

    # Set up forward hook on the right layer's residual.
    # We add `alpha * v` at decode position p_target.
    # Layer indexing: L=0 means token embedding, L=1..12 is after blocks 1..12.
    # Need to hook BEFORE the next block reads, i.e., on output of block (L-1).
    # We'll attach a hook to gpt2.transformer.h[L-1] that adds v to the output's
    # last token if and only if step counter == p_target.
    transformer = model.codi.transformer if hasattr(model.codi, "transformer") else model.codi.base_model.model.transformer
    n_blocks = len(transformer.h)
    print(f"  GPT-2 has {n_blocks} blocks; L+1=13 hidden states (L=0 is embed).")

    # State container shared across hook + step counter.
    HOOK_STATE = {"step": -1, "active": False, "vec": None, "p_target": None,
                  "layer": None, "alpha": 0.0}

    def make_hook(block_idx):
        def hook_fn(module, inputs, output):
            if not HOOK_STATE["active"]: return output
            if HOOK_STATE["layer"] is None: return output
            # output is (hidden_states,) tuple OR a tensor depending on impl
            h = output[0] if isinstance(output, tuple) else output
            # we want hook for layer L means after block L-1 → block_idx == L-1
            if block_idx != HOOK_STATE["layer"] - 1: return output
            if HOOK_STATE["step"] != HOOK_STATE["p_target"]: return output
            v = HOOK_STATE["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK_STATE["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return hook_fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.register_forward_hook(make_hook(i)))

    @torch.no_grad()
    def run_and_get_first_digit_token(batch_q, *, p_target, layer, vec, alpha):
        """Run inference; return the token id emitted at decode pos 4 (where the
        first answer-digit token sits) and the full predicted string."""
        B = len(batch_q)
        batch = tok(batch_q, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)

        # arm the hook
        HOOK_STATE.update({"vec": vec, "p_target": p_target, "layer": layer,
                           "alpha": alpha, "active": True, "step": -1})

        # prompt forward (latent loop). step stays at -1 (steering only fires
        # at decode positions, which start at step=0).
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

        # decode greedy 12 tokens; HOOK_STATE.step increments each step
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        for step in range(12):
            HOOK_STATE["step"] = step
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B): tokens[b].append(int(next_ids[b].item()))
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK_STATE.update({"active": False, "step": -1})

        out_strs = [tok.decode(t, skip_special_tokens=True) for t in tokens]
        # token at decode position 4 is the first answer digit
        digit_tokens = [t[4] if len(t) > 4 else None for t in tokens]
        return digit_tokens, out_strs

    # ---- 5. Sweep -------------------------------------------------
    N_eval = 100
    eval_idx = np.random.RandomState(0).choice(N, size=N_eval, replace=False)
    eval_qs = [questions[i] for i in eval_idx]
    eval_ans = [int(round(answers[i])) for i in eval_idx]
    eval_units = [units[i] for i in eval_idx]
    eval_tens = [tens[i] for i in eval_idx]
    eval_op = [operators[i] for i in eval_idx]

    BS = 16

    def run_full(p, l, vec, alpha, label):
        all_dig, all_strs = [], []
        for s in range(0, N_eval, BS):
            dig, strs = run_and_get_first_digit_token(
                eval_qs[s:s+BS], p_target=p, layer=l, vec=vec, alpha=alpha)
            all_dig += dig; all_strs += strs
        return all_dig, all_strs

    # baseline (alpha=0)
    print("\n=== Baseline (no steering) ===", flush=True)
    base_dig, base_strs = run_full(0, 0, torch.zeros(H), 0.0, "baseline")
    base_units = []
    for s in base_strs:
        # parse "The answer is: <num>..."
        import re
        m = re.search(r"answer is\s*:\s*([\-]?\d+)", s)
        base_units.append(int(m.group(1)) if m else None)
    print(f"  baseline first 8 outputs:")
    for i in range(8):
        print(f"    gold={eval_ans[i]:>5}  pred={base_strs[i][:40]!r}")

    # Per-variable steering sweeps
    results = {"baseline": [{"gold": int(eval_ans[i]),
                              "pred": base_strs[i],
                              "extracted": base_units[i]} for i in range(N_eval)]}

    sweeps = [
        ("operator_add_to_sub", op_pos, op_lay, sc_op, clf_op, 1, 0),  # tgt=sub from add
        ("operator_sub_to_add", op_pos, op_lay, sc_op, clf_op, 0, 1),
        ("tens_to_5",            tens_pos, tens_lay, sc_tens, clf_tens, 5, None),
        ("tens_to_8",            tens_pos, tens_lay, sc_tens, clf_tens, 8, None),
        ("units_to_3",           units_pos, units_lay, sc_units, clf_units, 3, None),
        ("units_to_7",           units_pos, units_lay, sc_units, clf_units, 7, None),
    ]
    alphas = [1.0, 2.0, 4.0, 8.0]

    for name, p, l, sc, clf, tgt, src in sweeps:
        if src is not None:
            v = get_steering_vec(sc, clf, tgt, src)
        else:
            v = get_steering_vec(sc, clf, tgt, None)
        print(f"\n=== {name}  (pos {p}, layer {l}) ===", flush=True)
        for a in alphas:
            dig, strs = run_full(p, l, v, a, name)
            results[f"{name}|alpha={a}"] = [{"gold": int(eval_ans[i]),
                                              "pred": strs[i]} for i in range(N_eval)]
            # quick on-line summary
            import re
            extracted = []
            for s in strs:
                m = re.search(r"answer is\s*:\s*([\-]?\d+)", s)
                extracted.append(int(m.group(1)) if m else None)
            n_changed = sum(1 for i in range(N_eval) if extracted[i] != base_units[i])
            print(f"  alpha={a}: {n_changed}/{N_eval} answers changed vs baseline")
            print(f"    examples: gold={eval_ans[0]} base={base_units[0]} steered={extracted[0]} | "
                  f"gold={eval_ans[1]} base={base_units[1]} steered={extracted[1]}")

    out = PD / "steering_results.json"
    out.write_text(json.dumps({"results": results,
                                "config": {
                                    "op_cell": [op_pos, op_lay],
                                    "tens_cell": [tens_pos, tens_lay],
                                    "units_cell": [units_pos, units_lay],
                                    "alphas": alphas,
                                    "N_eval": N_eval,
                                }}, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
