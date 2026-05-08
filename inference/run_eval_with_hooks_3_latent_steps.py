"""Run paired teacher/student SVAMP eval and save per-layer activations.

Teacher: base Llama-3.2-1B-Instruct with explicit chain-of-thought prompting.
  - Hooks: residual stream at the last prompt token, every layer.
  - Shape per question: (num_layers + 1, hidden_dim) bf16.

Student: CODI-llama3.2-1b-Instruct (LoRA + projection on top of the base) with
latent thoughts. Mirrors test.py's inference loop.
  - Hooks: residual stream at the last token of every latent step, every layer.
  - Shape per question: (num_latent_steps, num_layers + 1, hidden_dim) bf16.

Run one mode at a time:
  python run_eval_with_hooks.py --mode teacher --num_examples 50 --out_dir runs/teacher
  python run_eval_with_hooks.py --mode student --num_examples 50 --out_dir runs/student \
      --base_model unsloth/Llama-3.2-1B-Instruct \
      --ckpt_dir ~/codi_ckpt/CODI-llama3.2-1b-Instruct
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Make `codi/src/...` importable when this script lives in `inference/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "codi"))

import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", sentence)
    if not pred:
        return float("inf")
    return float(pred[-1])


def extract_option_number(sentence: str) -> int:
    """Extract the LOGIC-701 option choice (1..5). Prefer 'Answer: <N>'; else
    pick the last standalone 1..5 digit; else return -1 (unparseable)."""
    m = re.search(r"Answer\s*[:=]\s*([1-5])", sentence, re.IGNORECASE)
    if m:
        return int(m.group(1))
    digits = re.findall(r"(?<!\d)([1-5])(?!\d)", sentence)
    if digits:
        return int(digits[-1])
    return -1


def _format_logic701_question(ex: dict) -> str:
    opts = "\n".join(
        f"{i}. {ex[f'answer_option_{i}']}" for i in range(1, 6)
    )
    return f"{ex['problem_statement'].strip()}\n\nOptions:\n{opts}"


# -----------------------------------------------------------------------------
# Dataset loaders. Each returns (questions, golds, sys_msg, scoring_fn) where
# scoring_fn(decoded_text) -> parsed_prediction comparable to gold via ==.
# -----------------------------------------------------------------------------

def load_svamp(num_examples: int):
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions, golds = [], []
    for ex in full:
        q = ex["question_concat"].strip().replace("  ", " ")
        a = float(str(ex["Answer"]).replace(",", ""))
        questions.append(q)
        golds.append(a)
    sys_msg = (
        "You are a helpful math tutor. Solve the problem step by step, then on "
        "the final line write 'Answer: <number>' with only the numeric answer."
    )
    return (
        questions[:num_examples],
        golds[:num_examples],
        sys_msg,
        extract_answer_number,
    )


def load_logic701(num_examples: int):
    ds = load_dataset("hivaze/LOGIC-701", "en")
    split = ds["train"]
    questions, golds = [], []
    for ex in split:
        questions.append(_format_logic701_question(ex))
        golds.append(int(ex["correct_option_number"]))
    sys_msg = (
        "You are a careful logical reasoner. The user will give you a multiple "
        "choice problem with five options labeled 1 through 5. Think step by "
        "step, then on the final line write 'Answer: <N>' where N is the option "
        "number you choose (a single digit 1, 2, 3, 4, or 5)."
    )
    return (
        questions[:num_examples],
        golds[:num_examples],
        sys_msg,
        extract_option_number,
    )


def load_data(dataset: str, num_examples: int):
    if dataset == "svamp":
        return load_svamp(num_examples)
    if dataset == "logic701":
        return load_logic701(num_examples)
    raise ValueError(f"unknown dataset: {dataset}")


def run_teacher(args):
    device = "cuda"
    print(f"[teacher] loading {args.base_model}", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.padding_side = "left"  # required for batched left-padded generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"[teacher] num_layers={num_layers} hidden_dim={hidden_dim}", flush=True)

    questions, golds, sys_msg, scoring_fn = load_data(args.dataset, args.num_examples)

    results = []
    activations = []  # list of tensors, each (num_layers+1, hidden_dim)

    N = len(questions)
    bs = args.batch_size
    for i in range(0, N, bs):
        batch_q = questions[i : i + bs]
        batch_g = golds[i : i + bs]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch_q
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding="longest").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        # out.hidden_states[0] is the prefill: tuple of (num_layers+1) of (B, P, H).
        # With left-padding, the last *real* prompt token is at position -1 for
        # every row, so a single [:, -1, :] slice is correct.
        prefill = out.hidden_states[0]
        last_tok = torch.stack([h[:, -1, :] for h in prefill], dim=1)  # (B, layers+1, H)
        last_tok = last_tok.to(torch.bfloat16).cpu()
        gen_ids = out.sequences[:, prompt_len:]

        for b, (q, gold) in enumerate(zip(batch_q, batch_g)):
            activations.append(last_tok[b])
            text = tokenizer.decode(gen_ids[b], skip_special_tokens=True)
            if args.dataset == "svamp":
                m = re.search(r"Answer\s*[:=]\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
                pred = float(m.group(1)) if m else scoring_fn(text)
            else:
                pred = scoring_fn(text)
            correct = pred == gold
            results.append(
                {
                    "idx": i + b,
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "response": text,
                }
            )
        done = min(i + bs, N)
        print(
            f"[teacher] {done}/{N} batch_acc={sum(r['correct'] for r in results[-len(batch_q):])/len(batch_q):.2f}",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    acts = torch.stack(activations, dim=0)
    torch.save(acts, out_dir / "activations.pt")
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"[teacher] accuracy = {acc:.3f} on {len(results)} examples", flush=True)
    print(f"[teacher] activations: {tuple(acts.shape)} -> {out_dir/'activations.pt'}", flush=True)


def run_student(args):
    # CODI internally calls AutoTokenizer.from_pretrained(..., use_fast=False),
    # which silently returns False on Llama-3.2 (no slow tokenizer ships). Force
    # fast everywhere so the CODI module's internal tokenizer loads correctly.
    _orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def _force_fast(*a, **kw):
        kw["use_fast"] = True
        return _orig_from_pretrained(*a, **kw)

    transformers.AutoTokenizer.from_pretrained = _force_fast  # type: ignore[assignment]

    # Lazy imports to keep teacher mode independent of CODI src.
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore

    device = "cuda"
    print(f"[student] base={args.base_model} ckpt={args.ckpt_dir}")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    model_args = ModelArguments(
        model_name_or_path=args.base_model,
        full_precision=True,
        train=False,
        lora_init=True,
        ckpt_dir=args.ckpt_dir,
    )
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        bf16=True,
        use_lora=True,
        use_prj=True,
        prj_dim=2048,
        prj_no_ln=False,
        prj_dropout=0.0,
        num_latent=3,
        inf_latent_iterations=3,
        remove_eos=True,
        greedy=True,
        model_max_length=512,
        seed=11,
    )

    model = CODI(model_args, training_args, lora_config)
    sd_path_safe = os.path.join(args.ckpt_dir, "model.safetensors")
    sd_path_bin = os.path.join(args.ckpt_dir, "pytorch_model.bin")
    if os.path.exists(sd_path_safe):
        state_dict = load_file(sd_path_safe)
    else:
        state_dict = torch.load(sd_path_bin, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, model_max_length=512, padding_side="left", use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    model = model.to(device).to(torch.bfloat16)
    model.eval()
    num_layers = model.codi.config.num_hidden_layers
    hidden_dim = model.codi.config.hidden_size
    inf_latent_iterations = training_args.inf_latent_iterations
    print(
        f"[student] num_layers={num_layers} hidden_dim={hidden_dim} "
        f"latent_steps={inf_latent_iterations}"
    )

    questions, golds, _sys_msg, scoring_fn = load_data(args.dataset, args.num_examples)

    results = []
    activations = []  # each: (num_latent_steps, num_layers+1, hidden_dim)

    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tokenizer.eos_token_id

    N = len(questions)
    bs = args.batch_size
    for i in range(0, N, bs):
        batch_q = questions[i : i + bs]
        batch_g = golds[i : i + bs]
        B = len(batch_q)

        batch = tokenizer(batch_q, return_tensors="pt", padding="longest").to(device)
        bot = torch.full(
            (B, 1), model.bot_id, dtype=torch.long, device=device
        )
        batch["input_ids"] = torch.cat([batch["input_ids"], bot], dim=1)
        attn_mask = torch.cat(
            [batch["attention_mask"], torch.ones_like(bot)], dim=1
        )

        with torch.no_grad():
            out = model.codi(
                input_ids=batch["input_ids"],
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv = out.past_key_values
            latent_embd = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            per_step = []  # each: (B, num_layers+1, hidden)
            for _ in range(inf_latent_iterations):
                attn_mask = torch.cat(
                    [attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=device)],
                    dim=1,
                )
                out = model.codi(
                    inputs_embeds=latent_embd,
                    attention_mask=attn_mask,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_kv,
                )
                past_kv = out.past_key_values
                hs = torch.stack(
                    [h[:, -1, :] for h in out.hidden_states], dim=1
                )  # (B, layers+1, H)
                per_step.append(hs.to(torch.bfloat16).cpu())
                latent_embd = out.hidden_states[-1][:, -1, :].unsqueeze(1)
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)
            stacked = torch.stack(per_step, dim=1)  # (B, steps, layers+1, H)
            for b in range(B):
                activations.append(stacked[b])

            # eot_emb shape (1, H); broadcast to (B, 1, H) for batched decode start.
            eot_emb = embed_fn(
                torch.tensor([model.eot_id], dtype=torch.long, device=device)
            )
            output = eot_emb.unsqueeze(0).expand(B, -1, -1)

            finished = torch.zeros(B, dtype=torch.bool, device=device)
            pred_tokens: list[list[int]] = [[] for _ in range(B)]
            for _ in range(args.max_new_tokens):
                attn_mask = torch.cat(
                    [attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=device)],
                    dim=1,
                )
                step_out = model.codi(
                    inputs_embeds=output,
                    attention_mask=attn_mask,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    past_key_values=past_kv,
                )
                past_kv = step_out.past_key_values
                logits = step_out.logits[:, -1, : model.codi.config.vocab_size - 1]
                next_ids = torch.argmax(logits, dim=-1)
                for b in range(B):
                    if not finished[b]:
                        tid = int(next_ids[b].item())
                        pred_tokens[b].append(tid)
                        if tid == eos_id:
                            finished[b] = True
                if bool(finished.all()):
                    break
                output = embed_fn(next_ids).unsqueeze(1)

        for b, (q, gold) in enumerate(zip(batch_q, batch_g)):
            text = tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
            pred = scoring_fn(text)
            correct = pred == gold
            results.append(
                {
                    "idx": i + b,
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "response": text,
                }
            )
        done = min(i + bs, N)
        print(
            f"[student] {done}/{N} batch_acc={sum(r['correct'] for r in results[-B:])/B:.2f}",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    acts = torch.stack(activations, dim=0)
    torch.save(acts, out_dir / "activations.pt")
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"[student] accuracy = {acc:.3f} on {len(results)} examples")
    print(f"[student] activations: {tuple(acts.shape)} -> {out_dir/'activations.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["teacher", "student"], required=True)
    p.add_argument("--dataset", choices=["svamp", "logic701"], default="svamp")
    p.add_argument("--num_examples", type=int, default=10**9)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--ckpt_dir", default=os.path.expanduser("~/codi_ckpt/CODI-llama3.2-1b-Instruct"))
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    if args.mode == "teacher":
        run_teacher(args)
    else:
        run_student(args)


if __name__ == "__main__":
    main()
