"""Trained steering vector at a single (latent_step, layer) cell on GSM8K.

The previous LDA-direction and vary-op-direction interventions produced null
emit-flip rates. This script tests whether *any* learnable direction at a
chosen cell can causally flip the model's emitted answer — i.e., is the cell
genuinely inert (operator decision lives elsewhere) or just misaligned with
the directions we computed?

Procedure:
  1. Pick a target cell (default: step 5, L 11 — m=4's refined-probe winner,
     probe acc 0.90).
  2. Pick a target operator (default: '*'). For Addition-chain problems in
     gsm8k_cf_op_strict, the alt-mul answer is product(operands).
  3. Initialize a learnable v ∈ R^H with requires_grad=True (fp32).
  4. For each training problem:
       - Run CODI with v added at (step, layer) during latent loop
       - Get logits at the FIRST decode token (post-EOT)
       - Loss = -log P(target_first_digit_token), where target_first_digit
         is the first digit of alt_answer (mul-version)
       - Backprop into v
  5. After K steps of Adam, freeze v and test causal flip rate on held-out
     problems by hooking v during forward + decoding the chain + parsing
     emitted final answer.

If the trained v flips emit operators with small α → the cell IS causally
addressable; our analytic directions just missed the right subspace. If no
trained v achieves flips → the cell is genuinely inert.

Output: steer_trained_vector_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, os, re, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

TARGET_STEP = 4   # 0-indexed step 5
TARGET_LAYER = 11  # m=4's cell
TARGET_OP = "*"   # we'll train v to flip ADD→MUL
N_TRAIN = 60
N_EVAL = 40
N_ITERS = 80
LR = 5e-2

OUT_JSON = PD / "steer_trained_vector_gsm8k.json"
OUT_PDF = PD / "steer_trained_vector_gsm8k.pdf"


def codi_extract(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def apply_op_all(operands, op):
    if op == "+": return sum(operands)
    if op == "-": return operands[0] - sum(operands[1:])
    if op == "*":
        r = 1.0
        for x in operands: r *= x
        return r
    if op == "/":
        r = float(operands[0])
        for x in operands[1:]:
            if x == 0: return None
            r /= x
        return r
    return None


def first_digit_token_id(tok, val):
    """Get the token id for the answer's first digit when emitted AFTER
    ' The answer is:'. GPT-2 tokenizes ' 240' as [' 240'] or [' 2', '40'] —
    we want the leading space-prefixed first digit."""
    s = str(int(val)) if val == int(val) else str(val)
    # The position right after ':' produces a token starting with ' '.
    # Prefer space-prefixed single-digit (most common at low magnitudes).
    candidates = [" " + s[0], " " + s, s[0], s]
    for c in candidates:
        ids = tok.encode(c, add_special_tokens=False)
        if ids: return ids[0]
    return None


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
    targs = TrainingArguments(output_dir="/tmp/_trainvec", bf16=True,
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
    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    H = model.codi.config.n_embd
    eos_id = tok.eos_token_id

    # Freeze all model parameters
    for p in model.parameters(): p.requires_grad_(False)

    # Trainable steering vector
    v = torch.zeros(H, device="cuda", dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=LR)

    HOOK = {"active": False, "latent_step": -1, "v_param": None}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if block_idx != TARGET_LAYER: return output
            if HOOK["latent_step"] != TARGET_STEP: return output
            h = output[0] if isinstance(output, tuple) else output
            v_use = HOOK["v_param"].to(h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + v_use   # alpha=1; v's magnitude is learned
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    # Load Addition problems from cf_op_strict
    rows = json.load(open(CF_DIR / "gsm8k_cf_op_strict.json"))
    add_rows = [r for r in rows if r["type"] == "Addition"][:N_TRAIN + N_EVAL]
    print(f"  loaded {len(add_rows)} Addition problems from cf_op_strict")

    # Prep per-problem target token id (the FIRST digit of the mul-version answer)
    prepped = []
    for r in add_rows:
        try:
            ops = [float(x) for x in r["operands"]]
        except Exception:
            continue
        if len(ops) < 2: continue
        mul_ans = apply_op_all(ops, "*")
        if mul_ans is None or mul_ans <= 0: continue
        # first digit token id of mul_ans (the model usually emits "The answer is: N";
        # we score against the LAST digit of N's leading character — closer to what
        # model would emit after "is:")
        target_id = first_digit_token_id(tok, mul_ans)
        if target_id is None: continue
        prepped.append({"q": r["question_concat"].strip().replace("  ", " "),
                         "operands": ops, "mul_ans": float(mul_ans),
                         "add_ans": float(sum(ops)), "target_id": int(target_id)})
    if len(prepped) < N_TRAIN + 10:
        print(f"  too few prepped ({len(prepped)}); aborting"); return
    train_set = prepped[:N_TRAIN]; eval_set = prepped[N_TRAIN:N_TRAIN + N_EVAL]
    print(f"  train={len(train_set)} eval={len(eval_set)}")

    # Force-decode prefix " The answer is: " so the NEXT predicted token is
    # the answer's first digit. We then optimize the digit-position logits.
    PREFIX = " The answer is:"
    prefix_ids = tok.encode(PREFIX, add_special_tokens=False)
    print(f"  prefix '{PREFIX}' -> token ids {prefix_ids}")

    def forward_digit_logits(qs, v_param):
        """Run CODI through latent loop with steering, then force-decode the
        prefix tokens (no steering), and return logits at the position
        immediately AFTER the prefix (where the digit appears)."""
        B = len(qs)
        HOOK["v_param"] = v_param; HOOK["active"] = True; HOOK["latent_step"] = -1
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
        # Feed EOT then the prefix tokens in sequence (no steering here)
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        sout = model.codi(inputs_embeds=output, attention_mask=attn,
                          use_cache=True, past_key_values=past)
        past = sout.past_key_values
        for t_id in prefix_ids:
            t_emb = embed_fn(torch.tensor([t_id], dtype=torch.long, device="cuda")).unsqueeze(0).expand(B, -1, -1)
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            sout = model.codi(inputs_embeds=t_emb, attention_mask=attn,
                              use_cache=True, past_key_values=past)
            past = sout.past_key_values
        # logits at position AFTER the last prefix token = where digit emerges
        return sout.logits[:, -1, :model.codi.config.vocab_size - 1]

    forward_first_emit_logits = forward_digit_logits   # alias for old name

    loss_history = []
    t0 = time.time()
    for it in range(N_ITERS):
        # Mini-batch
        idx = np.random.choice(len(train_set), size=min(BS, len(train_set)), replace=False)
        batch = [train_set[i] for i in idx]
        qs = [b["q"] for b in batch]
        target_ids = torch.tensor([b["target_id"] for b in batch], device="cuda", dtype=torch.long)
        optimizer.zero_grad()
        logits = forward_first_emit_logits(qs, v)
        logits_f = logits.float()
        loss = F.cross_entropy(logits_f, target_ids)
        loss.backward()
        # clip v gradient to avoid blowup
        torch.nn.utils.clip_grad_norm_([v], max_norm=10.0)
        optimizer.step()
        loss_history.append(float(loss.item()))
        if it % 10 == 0 or it == N_ITERS - 1:
            print(f"  iter {it}/{N_ITERS}  loss={loss.item():.3f}  "
                  f"|v|={v.detach().norm().item():.2f}  ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"\nfinal |v| = {v.detach().norm().item():.3f}")

    # ---------- Eval ----------
    @torch.no_grad()
    def decode_with_v(qs, v_param, max_new=64):
        B = len(qs)
        HOOK["v_param"] = v_param; HOOK["active"] = True; HOOK["latent_step"] = -1
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

    eval_qs = [d["q"] for d in eval_set]

    # ---------- EVAL PROTOCOL: same as training. ----------
    # Force-decode " The answer is:" then check the next-token argmax.
    # Score: argmax token id matches mul-digit-id (target flip) vs add-digit-id
    # (kept) vs neither.
    @torch.no_grad()
    def force_decode_next_token(qs, v_param):
        """Same as training forward but returns argmax next-token (B,)."""
        logits = forward_digit_logits(qs, v_param)   # (B, V)
        return torch.argmax(logits, dim=-1)

    # Per-problem add/mul first-digit-after-prefix target ids
    eval_target_mul = []
    eval_target_add = []
    for d in eval_set:
        eval_target_mul.append(first_digit_token_id(tok, d["mul_ans"]))
        eval_target_add.append(first_digit_token_id(tok, d["add_ans"]))

    def score_force(next_ids):
        out = {"n": len(next_ids), "match_add": 0, "match_mul": 0,
                "match_other": 0}
        for i, nid in enumerate(next_ids):
            nid = int(nid)
            if nid == eval_target_mul[i]: out["match_mul"] += 1
            elif nid == eval_target_add[i]: out["match_add"] += 1
            else: out["match_other"] += 1
        return out

    print("\n=== EVAL (force-decode-prefix protocol, matches training) ===")
    sweeps = {}
    for alpha in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        v_scaled = (alpha * v.detach()).to(torch.float32)
        all_next = []
        for s in range(0, len(eval_qs), BS):
            ids = force_decode_next_token(eval_qs[s:s+BS], v_scaled)
            all_next += ids.cpu().tolist()
        sweeps[alpha] = {"score": score_force(all_next), "sample_ids": all_next[:5]}
        sc = sweeps[alpha]["score"]
        print(f"  α={alpha}: match_add={sc['match_add']}  match_mul={sc['match_mul']}  "
              f"other={sc['match_other']} / {sc['n']}")

    base_score = sweeps[0.0]["score"]
    learned_score = sweeps[1.0]["score"]
    # Also keep a natural-decode comparison at α=0 vs learned for transparency
    z = torch.zeros_like(v.detach())
    nat_base_strs = []
    nat_learned_strs = []
    for s in range(0, len(eval_qs), BS):
        nat_base_strs += decode_with_v(eval_qs[s:s+BS], z)
        nat_learned_strs += decode_with_v(eval_qs[s:s+BS], v.detach())
    print("\n  natural-decode samples (α=0 vs learned v at α=1):")
    for i in range(min(3, len(nat_base_strs))):
        print(f"    base: {nat_base_strs[i][:60]!r}")
        print(f"    lrn:  {nat_learned_strs[i][:60]!r}")

    # Compare learned v to LDA add↔mul direction at the same cell
    # (load lda_addmul direction from refined probe / vary-op data if available)
    # Here just compute cosine similarity to a random direction for sanity
    np.random.seed(0)
    rand_v = np.random.randn(H); rand_v /= np.linalg.norm(rand_v)
    v_np = v.detach().cpu().numpy()
    cos_to_rand = float(v_np @ rand_v / (np.linalg.norm(v_np) + 1e-9))

    results = {
        "target_step_1idx": TARGET_STEP + 1, "target_layer": TARGET_LAYER,
        "target_op": TARGET_OP, "N_train": len(train_set), "N_eval": len(eval_set),
        "N_iters": N_ITERS, "lr": LR,
        "final_v_norm": float(v.detach().norm().item()),
        "loss_history": loss_history,
        "baseline_score": base_score, "learned_score": learned_score,
        "alpha_sweep": {str(a): {"score": sweeps[a]["score"]}
                         for a in sweeps},
        "cos_to_random_v": cos_to_rand,
        "v_vector_l2": float(np.linalg.norm(v_np)),
    }
    np.save(PD / "steer_trained_v_gsm8k.npy", v_np)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON} (and steer_trained_v_gsm8k.npy with the learned vector)")

    # Plot
    with PdfPages(OUT_PDF) as pdf:
        # Loss curve
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(loss_history, "-o", lw=1, ms=2)
        ax.set_xlabel("training iter"); ax.set_ylabel("cross-entropy loss")
        ax.set_title(f"Training loss for steering vector at "
                     f"(step {TARGET_STEP+1}, L {TARGET_LAYER})  target='{TARGET_OP}'",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Bar chart of scores by alpha
        fig, ax = plt.subplots(figsize=(11, 5))
        alphas = sorted(sweeps.keys())
        n = sweeps[alphas[0]]["score"]["n"]
        ax.bar(np.arange(len(alphas)) - 0.2, [sweeps[a]["score"]["match_add"] / n for a in alphas],
               0.4, label="match ADD answer", color="C0")
        ax.bar(np.arange(len(alphas)) + 0.2, [sweeps[a]["score"]["match_mul"] / n for a in alphas],
               0.4, label="match MUL answer (target)", color="C3")
        ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"α={a}" for a in alphas])
        ax.set_ylabel("fraction of eval N"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Trained v at (step {TARGET_STEP+1}, L {TARGET_LAYER}) — eval on cf_op_strict ADD",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
