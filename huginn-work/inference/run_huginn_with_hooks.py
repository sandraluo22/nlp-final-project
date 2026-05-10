"""Run Huginn-0125 (recurrent-depth latent reasoning) with hooks to save
per-iteration, per-layer residual stream activations.

Huginn-0125 is structured as: prelude P -> core block R iterated num_steps
times -> coda C -> lm_head. We hook every layer of the core block, capture the
last-prompt-token residual on the prefill pass, and reshape to
    (N, num_steps, L_core, H)
matching the CODI student tensor shape so downstream scripts (logit_lens.py,
aggregate_logit_lens.py, probe_accuracy.py) work with no further changes.

Usage:
  python huginn-work/inference/run_huginn_with_hooks.py \\
      --dataset svamp \\
      --num_steps 32 \\
      --num_examples 1000 \\
      --out_dir huginn-work/inference/runs/svamp_huginn

Notes:
  - Loads model with trust_remote_code=True (Huginn ships custom code).
  - Two passes per question: (1) prefill forward with hooks active to capture
    activations, (2) generate to score correctness. Roughly 1.5x compute vs
    minimal eval; worth the simplicity.
  - DEFAULT batch_size=1 because Huginn's released main forward
    (raven_modeling_minimal.py) sets prepared_attn_mask = None and ignores
    attention_mask, so batched left-padded input would attend across pad
    tokens. If you batch, all rows in a batch must be the same length or you
    will silently corrupt the activations.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# load_data() lives in codi-work/inference/run_eval_with_hooks.py.
sys.path.insert(0, str(_PROJECT_ROOT / "codi-work" / "inference"))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from run_eval_with_hooks import load_data  # type: ignore[import-not-found]


def find_core_block(model):
    """Locate the recurrent core block ModuleList.

    Huginn-0125's released code lives at `model.transformer.core_block`. We
    probe at runtime to remain robust to upstream refactors and surface a
    clear error pointing at the actual module tree if not found.
    """
    candidates = [
        ("transformer", "core_block"),
        ("model", "core_block"),
        ("core_block",),
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj) > 0:
            return obj, ".".join(path)

    visible = [n for n, _ in model.named_modules()][:60]
    raise RuntimeError(
        "could not find core_block on the Huginn model. First 60 module "
        "names:\n  " + "\n  ".join(visible)
    )


def build_chat_prompts(tokenizer, questions, sys_msg):
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]


def parse_pred(text: str, dataset: str, scoring_fn):
    """Mirror the parsing logic in run_eval_with_hooks.run_teacher."""
    numeric = (
        "svamp", "gsm-hard", "cf_magmatched", "cf_balanced", "cf_under99",
        "cf_under99_b", "cf_gpt_transformed", "logic701_numeric",
        "mathqa_numeric",
    )
    if dataset in numeric:
        m = re.search(r"Answer\s*[:=]\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return scoring_fn(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", default="svamp",
        choices=[
            "svamp", "logic701", "logic701_numeric",
            "mathqa_numeric", "gsm-hard",
            "cf_magmatched", "cf_balanced", "cf_under99",
            "cf_under99_b", "cf_gpt_transformed",
        ],
    )
    p.add_argument("--num_examples", type=int, default=10**9)
    p.add_argument("--num_steps", type=int, default=32,
                   help="recurrent iterations through the core block (4-64; <4 is coarse)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="default 1 because Huginn's main forward ignores attention_mask")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--base_model", default="tomg-group-umd/huginn-0125")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[huginn] WARNING: no CUDA detected; this will be very slow.",
              flush=True)

    print(f"[huginn] loading {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    core_block, core_path = find_core_block(model)
    L_core = len(core_block)
    H = getattr(model.config, "n_embd", None) or getattr(model.config, "hidden_size", None)
    print(
        f"[huginn] core block at model.{core_path}: L_core={L_core}, hidden={H}, "
        f"num_steps={args.num_steps}",
        flush=True,
    )

    questions, golds, sys_msg, scoring_fn = load_data(args.dataset, args.num_examples)
    prompts = build_chat_prompts(tokenizer, questions, sys_msg)

    state = {"active": False, "captured": []}

    def hook(_mod, _inp, out):
        if not state["active"]:
            return
        h = out[0] if isinstance(out, tuple) else out
        state["captured"].append(h[:, -1, :].detach().to(torch.bfloat16).cpu())

    handles = [layer.register_forward_hook(hook) for layer in core_block]

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        stop_strings=["<|end_text|>", "<|end_turn|>"],
        use_cache=True,
        do_sample=False,
        temperature=None, top_k=None, top_p=None, min_p=None,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    results = []
    activations = []  # list of per-question (num_steps, L_core, H)

    N = len(prompts)
    bs = args.batch_size
    if bs > 1:
        print("[huginn] WARNING: batch_size > 1 with Huginn's released main "
              "forward will silently mis-attend pad tokens unless every prompt "
              "in a batch tokenizes to the same length.", flush=True)

    try:
        for i in range(0, N, bs):
            batch_p = prompts[i:i + bs]
            batch_q = questions[i:i + bs]
            batch_g = golds[i:i + bs]
            B = len(batch_p)

            # The chat template already injects bos_token; do not re-add.
            inputs = tokenizer(
                batch_p,
                return_tensors="pt",
                padding="longest" if B > 1 else False,
                add_special_tokens=False,
            ).to(device)

            # ---- Pass 1: prefill with hooks active to capture activations ----
            state["captured"].clear()
            state["active"] = True
            with torch.no_grad():
                _ = model(
                    input_ids=inputs["input_ids"],
                    num_steps=args.num_steps,
                    use_cache=False,
                )
            state["active"] = False

            # captured is K * L_core tensors of shape (B, H), in execution order:
            #   iter1_layer0, ..., iter1_layer{L-1}, iter2_layer0, ...
            expected = args.num_steps * L_core
            assert len(state["captured"]) == expected, (
                f"hook count mismatch: got {len(state['captured'])}, expected "
                f"{expected} (num_steps={args.num_steps} * L_core={L_core})"
            )
            stacked = torch.stack(state["captured"], dim=1)  # (B, K*L, H)
            stacked = stacked.view(B, args.num_steps, L_core, -1).contiguous()
            for b in range(B):
                activations.append(stacked[b])

            # ---- Pass 2: generate to score correctness ----
            with torch.no_grad():
                gen_out = model.generate(
                    inputs["input_ids"],
                    gen_cfg,
                    tokenizer=tokenizer,
                    num_steps=args.num_steps,
                )
            seqs = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = seqs[:, prompt_len:]
            for b, (q, gold) in enumerate(zip(batch_q, batch_g)):
                text = tokenizer.decode(gen_ids[b], skip_special_tokens=True)
                pred = parse_pred(text, args.dataset, scoring_fn)
                correct = (pred == gold)
                results.append({
                    "idx": i + b,
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "response": text,
                })
            done = min(i + bs, N)
            recent_acc = sum(r["correct"] for r in results[-B:]) / B
            print(f"[huginn] {done}/{N} batch_acc={recent_acc:.2f}", flush=True)
    finally:
        for h in handles:
            h.remove()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    acts = torch.stack(activations, dim=0)
    torch.save(acts, out_dir / "activations.pt")
    meta = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "num_steps": args.num_steps,
        "L_core": L_core,
        "hidden_dim": int(acts.shape[-1]),
        "n_examples": len(results),
        "tensor_shape": list(acts.shape),
        "tensor_layout": "(N, num_steps, L_core, H)",
        "note": (
            "axis 1 (num_steps) corresponds to the recurrence iteration index "
            "i in {1..num_steps}; axis 2 (L_core) is the layer index within "
            "one core_block call. Identical shape semantics to "
            "inference/runs/svamp_student/activations.pt for CODI student."
        ),
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"[huginn] accuracy = {acc:.3f} on {len(results)} examples", flush=True)
    print(f"[huginn] activations: {tuple(acts.shape)} -> {out_dir/'activations.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
