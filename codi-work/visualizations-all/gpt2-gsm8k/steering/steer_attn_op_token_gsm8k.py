"""Last-attempt steering: redirect the OPERATOR TOKEN at step 1, layer 10
attention output.

Finding from logit lens: at step 1, layer 10's attention output's modal top-1
token is ' *' with 0.58 confidence — the model is *attention-writing the
multiplication operator token directly into the residual at step 1*. This is
the only cell in the latent loop where a specific operator token clearly
emerges as the attention-output top-1 modal token.

Hypothesis: if we directly modify this attention output to push the model
toward a DIFFERENT operator (e.g., ' +'), the model's subsequent latent
processing might propagate this alternate operator into the emit.

Direction = W_LM[" +"] - W_LM[" *"], i.e., the LM-head direction that would
shift logits from ' *' toward ' +'. We add α·this_direction to the L10
attention output at step 1's last-token position.

Test set: gsm8k_cf_op_strict, Multiplication problems (the model gets ~95%
of these right at baseline so we have full headroom to FLIP them to
'addition' answers).

For each α: hook step1/L10 attn_out, decode, parse emit, check if
  - emit equals MUL answer (kept) vs ADD answer (flipped to add) vs other

Output: steer_attn_op_token_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, os, re, sys, time
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

TARGET_LATENT_STEP = 0  # 0-indexed; latent step 1
TARGET_LAYER = 10        # the L10 attention output where modal is ' *'

OUT_JSON = PD / "steer_attn_op_token_gsm8k.json"
OUT_PDF = PD / "steer_attn_op_token_gsm8k.pdf"


def codi_extract(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    BS = 8
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
    targs = TrainingArguments(output_dir="/tmp/_steerop", bf16=True,
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
    H = model.codi.config.n_embd
    final_ln = transformer.ln_f
    lm_head = model.codi.get_output_embeddings()

    # Get LM-head row directions for the operator tokens. We want the
    # "direction that pushes residual toward ' +'" — that's the row of the
    # LM head matrix for the ' +' token id (with bias against ' *').
    # The LM head is a tied-weight Linear; its weight has shape (V, H).
    W_LM = lm_head.weight.detach().to(torch.float32).cpu().numpy()  # (V, H)
    print(f"  LM head weight shape: {W_LM.shape}")

    op_ids = {}
    for op in ["+", "-", "*", "/"]:
        # Encode the space-prefixed token (' +' is one token in GPT-2)
        ids = tok.encode(" " + op, add_special_tokens=False)
        op_ids[op] = ids[0] if ids else None
        bare = tok.encode(op, add_special_tokens=False)
        print(f"  op {op!r}: ' {op}' -> id {op_ids[op]} ('{tok.decode([op_ids[op]])}'), "
              f"bare '{op}' -> {bare}")

    # Compute the direction
    def op_dir(from_op, to_op):
        """Direction to push residual from from_op toward to_op."""
        return W_LM[op_ids[to_op]] - W_LM[op_ids[from_op]]

    # Hook: at (target_step, target_layer)'s ATTN OUTPUT, add α·v
    HOOK = {"active": False, "latent_step": -1, "alpha": 0.0, "v": None}

    def make_attn_hook(idx):
        def fn(_m, _i, output):
            if not HOOK["active"]: return output
            if idx != TARGET_LAYER: return output
            if HOOK["latent_step"] != TARGET_LATENT_STEP: return output
            a = output[0] if isinstance(output, tuple) else output
            v = HOOK["v"].to(a.device, dtype=a.dtype)
            a = a.clone()
            a[:, -1, :] = a[:, -1, :] + HOOK["alpha"] * v
            return (a,) + output[1:] if isinstance(output, tuple) else a
        return fn

    handles = [blk.attn.register_forward_hook(make_attn_hook(i))
               for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, alpha, v_torch, max_new=64):
        B = len(qs)
        HOOK["alpha"] = float(alpha); HOOK["v"] = v_torch
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

    # Eval set: cf_op_strict, Multiplication problems (high baseline acc → headroom)
    rows = json.load(open(CF_DIR / "gsm8k_cf_op_strict.json"))
    mul_rows = [r for r in rows if r["type"] == "Multiplication"]
    prepped = []
    for r in mul_rows:
        try:
            ops = [float(x) for x in r["operands"]]
        except Exception:
            continue
        if len(ops) < 2: continue
        mul_ans = reduce(lambda a, b: a * b, ops)
        add_ans = sum(ops)
        sub_ans = ops[0] - sum(ops[1:])
        prepped.append({"q": r["question_concat"].strip().replace("  ", " "),
                         "operands": ops, "mul_ans": float(mul_ans),
                         "add_ans": float(add_ans), "sub_ans": float(sub_ans),
                         "gold": float(r["answer"])})
    eval_set = prepped[:80]
    qs = [d["q"] for d in eval_set]
    print(f"  eval N={len(eval_set)}")

    # Sweep multiple direction choices + alphas
    direction_choices = [
        ("*→+",  op_dir("*", "+")),
        ("*→-",  op_dir("*", "-")),
        ("*→/",  op_dir("*", "/")),
    ]
    alphas = [-3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 12.0]

    results = {"target_step_1idx": TARGET_LATENT_STEP + 1,
               "target_layer": TARGET_LAYER, "n_eval": len(eval_set),
               "directions": {}}

    def score(strs):
        out = {"n": len(strs), "match_mul": 0, "match_add": 0, "match_sub": 0,
               "match_other": 0, "unparsed": 0}
        for s, d in zip(strs, eval_set):
            v = codi_extract(s)
            if v is None: out["unparsed"] += 1; continue
            if abs(v - d["mul_ans"]) < 1e-3: out["match_mul"] += 1
            elif abs(v - d["add_ans"]) < 1e-3: out["match_add"] += 1
            elif abs(v - d["sub_ans"]) < 1e-3: out["match_sub"] += 1
            else: out["match_other"] += 1
        return out

    t0 = time.time()
    for dir_name, v_np in direction_choices:
        v_torch = torch.tensor(v_np, dtype=torch.float32)
        v_norm = float(np.linalg.norm(v_np))
        print(f"\n=== direction {dir_name}  |v|={v_norm:.2f} ===", flush=True)
        results["directions"][dir_name] = {"v_norm": v_norm, "by_alpha": {}}
        for alpha in alphas:
            strs = []
            for s in range(0, len(qs), BS):
                strs += run_batch(qs[s:s+BS], alpha=alpha, v_torch=v_torch)
            sc = score(strs)
            results["directions"][dir_name]["by_alpha"][str(alpha)] = {
                "score": sc, "sample": strs[:3]}
            print(f"  α={alpha:+.1f}: kept_mul={sc['match_mul']}  "
                  f"→add={sc['match_add']}  →sub={sc['match_sub']}  "
                  f"other={sc['match_other']}  unparsed={sc['unparsed']}  "
                  f"({time.time()-t0:.0f}s)")
        OUT_JSON.write_text(json.dumps(results, indent=2))

    for h in handles: h.remove()
    print(f"\nsaved {OUT_JSON}")

    # Plot
    with PdfPages(OUT_PDF) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")
        body = (f"Last-attempt steering: redirect operator-token at "
                f"step {TARGET_LATENT_STEP+1} layer {TARGET_LAYER} ATTENTION output.\n\n"
                f"  Hook applied at (step {TARGET_LATENT_STEP+1}, L{TARGET_LAYER})'s\n"
                f"  attention-block output, last-token position. Added α × v where:\n"
                f"    v = W_LM[' to_op'] - W_LM[' from_op']  (LM-head direction)\n\n"
                f"  Eval: {len(eval_set)} Multiplication problems from cf_op_strict\n"
                f"  Tracking the EMITTED FINAL ANSWER vs:\n"
                f"    mul_ans (original answer)\n"
                f"    add_ans (if all operations were +)\n"
                f"    sub_ans (if all were -)\n")
        ax.text(0.04, 0.95, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for dir_name, info in results["directions"].items():
            fig, ax = plt.subplots(figsize=(12, 5.5))
            by_a = info["by_alpha"]
            a_keys = sorted(by_a.keys(), key=lambda k: float(k))
            xs = np.arange(len(a_keys))
            sc = lambda key: [by_a[a]["score"][key] / len(eval_set) for a in a_keys]
            ax.plot(xs, sc("match_mul"), "-o", color="#4c72b0",
                    label="kept multiplication")
            ax.plot(xs, sc("match_add"), "-s", color="#2ca02c",
                    label="flipped to addition")
            ax.plot(xs, sc("match_sub"), "-^", color="#d62728",
                    label="flipped to subtraction")
            ax.plot(xs, sc("match_other"), "-x", color="#7f7f7f",
                    label="other/scrambled")
            ax.set_xticks(xs); ax.set_xticklabels(a_keys)
            ax.set_xlabel("α (scalar on op-direction)")
            ax.set_ylabel("fraction of eval N")
            ax.set_title(f"Direction: {dir_name}  |v|={info['v_norm']:.2f}",
                         fontsize=11, fontweight="bold")
            ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
