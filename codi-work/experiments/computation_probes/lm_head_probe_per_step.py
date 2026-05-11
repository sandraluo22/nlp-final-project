"""LM-head probe per latent step for CODI-GPT-2 OR CODI-Llama.

At each latent step k=1..K, take the final-layer residual, apply ln_f + lm_head,
extract:
  - top-1 token (and its prob)
  - softmax distribution (truncated to top-1000 to save space)
Report:
  - Top-1 stability across steps (per example)
  - Cos sim between softmax distributions at consecutive steps
  - JS divergence between distributions
"""

from __future__ import annotations
import json, os, re, sys, time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))


def codi_extract(s):
    s = s.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt2", "llama"], required=True)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--N", type=int, default=0,
                    help="number of SVAMP examples (0 = use all 1000)")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    if args.model == "gpt2":
        BASE_MODEL = "gpt2"
        ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
        target_modules = ["c_attn", "c_proj", "c_fc"]
        prj_dim = 768
        out_name = "lm_head_probe_gpt2.json"
    else:
        BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
        ckpt = os.path.expanduser("~/codi_ckpt/CODI-llama3.2-1b-Instruct")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        prj_dim = 2048
        out_name = "lm_head_probe_llama.json"

    print(f"loading {args.model.upper()} CODI from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path=BASE_MODEL, full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_lmh", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=prj_dim,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=args.K,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained(BASE_MODEL, model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()

    # Apply lm_head: GPT-2 has codi.transformer.ln_f; Llama has codi.model.norm
    inner = None
    for attr in ("transformer", "model"):
        if hasattr(model.codi, attr):
            inner = getattr(model.codi, attr); break
        # PEFT wrapper path
        bm = getattr(model.codi, "base_model", None)
        if bm is not None:
            for sub in ("transformer", "model"):
                if hasattr(bm, sub):
                    inner = getattr(bm, sub); break
                bm2 = getattr(bm, "model", None)
                if bm2 is not None and hasattr(bm2, sub):
                    inner = getattr(bm2, sub); break
            if inner: break
    if inner is None:
        raise RuntimeError("couldn't locate transformer/model attribute")
    ln_f = getattr(inner, "ln_f", None) or getattr(inner, "norm", None)
    lm_head = model.codi.get_output_embeddings()
    print(f"  inner: {type(inner).__name__}  ln_f: {type(ln_f).__name__}  lm_head: {type(lm_head).__name__}")

    K = args.K
    EPS = 1e-12

    def gold_to_str(g):
        return str(int(g)) if float(g).is_integer() else str(g)

    def target_first_token(gold_val):
        """First *informative* (non-whitespace-only) BPE token of " {gold}".
        For GPT-2 the leading-space number is usually a single token (" 145"), so this
        returns that. For Llama tokenizers a leading space is its own token (220) and the
        digits split into separate tokens (" ", "1", "4", "5") — we skip the space and
        return the first digit token so the target is discriminating across examples.
        """
        ids = tok.encode(" " + gold_to_str(gold_val), add_special_tokens=False)
        for tid in ids:
            piece = tok.decode([tid])
            if piece.strip():
                return tid
        return ids[0] if ids else -1

    @torch.no_grad()
    def run_batch(qs, target_ids_batch):
        """Run K latent steps; return per-example top-1 ids (B, K), top-1 probs (B, K),
        per-step correctness masks for top-1 / top-5 / top-10, and CPU-side accumulators
        of cos-sim / KL / top-1 stability per consecutive transition.
        Probs are kept only on-GPU; only summary stats and small per-example arrays leave the loop.
        """
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)

        targets_t = torch.tensor(target_ids_batch, device="cuda", dtype=torch.long)

        top1_per_step = np.zeros((B, K), dtype=np.int64)
        top1_prob_per_step = np.zeros((B, K), dtype=np.float32)
        correct1 = np.zeros((B, K), dtype=bool)
        correct5 = np.zeros((B, K), dtype=bool)
        correct10 = np.zeros((B, K), dtype=bool)
        # per-transition sums (size K-1)
        cos_sum = np.zeros(K - 1, dtype=np.float64)
        kl_sum = np.zeros(K - 1, dtype=np.float64)
        top1_same = np.zeros(K - 1, dtype=np.int64)

        prev_probs = None
        prev_top1 = None
        for k in range(K):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            final = out.hidden_states[-1][:, -1, :]  # (B, H)
            normed = ln_f(final.to(ln_f.weight.dtype)) if ln_f is not None else final
            logits = lm_head(normed)  # (B, vocab_size)
            vocab_size = model.codi.config.vocab_size - 1
            logits = logits[:, :vocab_size]
            probs = F.softmax(logits.float(), dim=-1)  # (B, vocab), on GPU
            top10_vals, top10_ids = torch.topk(probs, k=10, dim=-1)  # (B, 10)
            top1_ids = top10_ids[:, 0]
            top1_probs = top10_vals[:, 0]
            top1_per_step[:, k] = top1_ids.cpu().numpy()
            top1_prob_per_step[:, k] = top1_probs.cpu().float().numpy()
            # correctness: is target token in top-1 / top-5 / top-10?
            matches10 = (top10_ids == targets_t.unsqueeze(1))  # (B, 10)
            correct1[:, k] = matches10[:, 0].cpu().numpy()
            correct5[:, k] = matches10[:, :5].any(dim=1).cpu().numpy()
            correct10[:, k] = matches10.any(dim=1).cpu().numpy()

            if prev_probs is not None:
                cos = F.cosine_similarity(prev_probs, probs, dim=-1)
                a = prev_probs + EPS
                b = probs + EPS
                kl = (a * (a.log() - b.log())).sum(dim=-1)
                cos_sum[k - 1] = cos.sum().item()
                kl_sum[k - 1] = kl.sum().item()
                top1_same[k - 1] = (prev_top1 == top1_ids).sum().item()

            prev_probs = probs
            prev_top1 = top1_ids
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)

        return (top1_per_step, top1_prob_per_step,
                correct1, correct5, correct10,
                cos_sum, kl_sum, top1_same)

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    if args.N and args.N < len(full):
        np.random.seed(0)
        eval_idx = np.random.choice(len(full), size=args.N, replace=False)
        eval_qs = [questions[i] for i in eval_idx]
        eval_golds = golds[eval_idx]
    else:
        eval_qs = questions
        eval_golds = golds

    target_ids_all = np.array([target_first_token(g) for g in eval_golds], dtype=np.int64)
    print(f"  N={len(eval_qs)}, sample target tokens: "
          f"{[(gold_to_str(eval_golds[i]), int(target_ids_all[i]), tok.decode([int(target_ids_all[i])])) for i in range(min(5, len(eval_qs)))]}")

    N_total = len(eval_qs)
    all_top1 = np.zeros((N_total, K), dtype=np.int64)
    all_top1_prob = np.zeros((N_total, K), dtype=np.float32)
    all_correct1 = np.zeros((N_total, K), dtype=bool)
    all_correct5 = np.zeros((N_total, K), dtype=bool)
    all_correct10 = np.zeros((N_total, K), dtype=bool)
    cos_running = np.zeros(K - 1, dtype=np.float64)
    kl_running = np.zeros(K - 1, dtype=np.float64)
    top1_same_running = np.zeros(K - 1, dtype=np.int64)

    t0 = time.time()
    for s in range(0, N_total, args.bs):
        e = min(s + args.bs, N_total)
        t1, p1, c1, c5, c10, cos_sum, kl_sum, top1_same = run_batch(
            eval_qs[s:e], target_ids_all[s:e].tolist())
        all_top1[s:e] = t1
        all_top1_prob[s:e] = p1
        all_correct1[s:e] = c1
        all_correct5[s:e] = c5
        all_correct10[s:e] = c10
        cos_running += cos_sum
        kl_running += kl_sum
        top1_same_running += top1_same
        done = s + args.bs
        if done % 16 == 0 or done >= N_total:
            print(f"  {min(done, N_total)}/{N_total}  ({time.time()-t0:.0f}s)", flush=True)

    top1_stab = (top1_same_running / N_total).tolist()
    cos_sims = (cos_running / N_total).tolist()
    # KL in nats (matches earlier JSONs and slideshow scale)
    kls = (kl_running / N_total).tolist()

    def transitions_for(cmask):
        """cmask: (N, K) bool. Return per-K accuracy + per-(k,k+1) transitions."""
        acc = cmask.mean(axis=0).tolist()
        n_corr = cmask.sum(axis=0).tolist()
        trs = []
        for k in range(K - 1):
            ck = cmask[:, k]
            ck1 = cmask[:, k + 1]
            trs.append({
                "from_step": k + 1, "to_step": k + 2,
                "right_to_wrong": int((ck & ~ck1).sum()),
                "wrong_to_right": int((~ck & ck1).sum()),
                "stable_right": int((ck & ck1).sum()),
                "stable_wrong": int((~ck & ~ck1).sum()),
            })
        return acc, [int(n) for n in n_corr], trs

    acc1, ncorr1, trs1 = transitions_for(all_correct1)
    acc5, ncorr5, trs5 = transitions_for(all_correct5)
    acc10, ncorr10, trs10 = transitions_for(all_correct10)

    print(f"\n=== LM-head target-token accuracy per step (top-1 / top-5 / top-10) ===")
    for k in range(K):
        print(f"  step {k+1}: top1={acc1[k]*100:.1f}%  top5={acc5[k]*100:.1f}%  top10={acc10[k]*100:.1f}%")
    print(f"\n=== LM-head right<->wrong transitions (top-1) ===")
    for t in trs1:
        net = t["wrong_to_right"] - t["right_to_wrong"]
        print(f"  step {t['from_step']}->{t['to_step']}: w->r={t['wrong_to_right']:3d}  r->w={t['right_to_wrong']:3d}  net={net:+d}")

    print(f"\n=== Top-1 token stability per step transition ===")
    for k in range(K - 1):
        print(f"  step {k+1} -> {k+2}: top-1 same in {top1_stab[k]*100:.1f}% of examples")

    print(f"\n=== Cos sim between softmax distributions, consecutive steps ===")
    for k in range(K - 1):
        print(f"  step {k+1} -> {k+2}: mean cos sim = {cos_sims[k]:.4f}")

    print(f"\n=== KL divergence (nats) between consecutive step distributions ===")
    for k in range(K - 1):
        print(f"  step {k+1} -> {k+2}: mean KL(step_k || step_k+1) = {kls[k]:.4f} nats")

    print(f"\n=== Most common top-1 tokens at each step ===")
    from collections import Counter
    for k in range(K):
        c = Counter(all_top1[:, k].tolist()).most_common(5)
        decoded = [(tok.decode([t]), n) for t, n in c]
        print(f"  step {k+1}: {decoded}")

    out = PD / out_name
    n_show = min(50, N_total)
    out.write_text(json.dumps({
        "model": args.model, "K": K, "N": N_total,
        "top1_stability": top1_stab,
        "cos_sim_consec": cos_sims,
        "kl_consec": kls,
        "kl_unit": "nats",
        "accuracy_per_step_top1": acc1,
        "accuracy_per_step_top5": acc5,
        "accuracy_per_step_top10": acc10,
        "n_correct_per_step_top1": ncorr1,
        "n_correct_per_step_top5": ncorr5,
        "n_correct_per_step_top10": ncorr10,
        "transitions_top1": trs1,
        "transitions_top5": trs5,
        "transitions_top10": trs10,
        "top1_per_step_examples": all_top1[:n_show].tolist(),
        "top1_decoded_per_step": [[tok.decode([int(t)]) for t in all_top1[i]] for i in range(n_show)],
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
