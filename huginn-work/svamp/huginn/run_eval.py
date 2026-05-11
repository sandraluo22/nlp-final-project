"""Run SVAMP / GSM-Hard / LOGIC-701-numeric / MathQA-numeric eval on Huginn
(tomg-group-umd/huginn-0125, the recurrent-depth latent-reasoning model from
Geiping et al. 2025) and optionally save per-(recurrence-step, core-block)
activations.

Architecture (config: n_embd=5280):
  prelude  : 2 SandwichBlocks (run once)
  core     : 4 SandwichBlocks (looped `num_steps` times per forward)
  coda     : 2 SandwichBlocks (run once)

We hook only the 4 core_block modules. With recurrence depth K, each block
fires K times during the prompt forward; we capture the residual at the LAST
prompt token at each call, giving an activation tensor per question of shape
(K, 4, H=5280) that mirrors CODI's (latent_steps, layers+1, H) layout.

Usage:
  python huginn/run_eval.py --dataset svamp --num_steps 32 --num_examples 1000 \
      --batch_size 4 --out_dir runs/huginn_svamp_K32

  python huginn/run_eval.py --dataset gsm-hard --num_steps 8 --num_examples 200 \
      --batch_size 4 --out_dir runs/huginn_gsmhard_K8 --no_save_activations
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import transformers
from datasets import concatenate_datasets, load_dataset


MODEL_NAME = "tomg-group-umd/huginn-0125"


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #

def extract_answer_number(sentence: str) -> float:
    """Truncate at the first sign Huginn has wandered into a new fake Q/A
    block, then prefer 'The answer is X' if present, else fall back to the
    last numeric token in the truncated text."""
    s = sentence.replace(",", "")
    for stop in ("\n\nQuestion:", "\nQuestion:", "Question:"):
        idx = s.find(stop)
        if idx > 0:
            s = s[:idx]
            break
    m = re.search(r"answer is\s*\$?\s*(-?\d+\.?\d*)", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    pred = re.findall(r"-?\d+\.?\d*", s)
    if not pred:
        return float("inf")
    return float(pred[-1])


_LOGIC701_NUM_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z%][^\d]*)?\s*$")


def _logic701_clean_num(s):
    s = str(s).replace(",", "")
    m = _LOGIC701_NUM_RE.match(s)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Prompt format. Huginn was trained largely on GSM8k-style "Question: ...
# Answer:" prompts; the paper reports best results when the answer line is
# parsed for a trailing number.
# --------------------------------------------------------------------------- #

PROMPT_TEMPLATE = (
    "You are a careful step-by-step math problem solver. Solve the problem "
    "and end with 'The answer is <number>.'\n\n"
    "Question: {q}\nAnswer:"
)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

def load_svamp(num_examples):
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    qs, gs = [], []
    for ex in full:
        qs.append(ex["question_concat"].strip().replace("  ", " "))
        gs.append(float(str(ex["Answer"]).replace(",", "")))
    return qs[:num_examples], gs[:num_examples], extract_answer_number


def load_gsm_hard(num_examples):
    ds = load_dataset("juyoung-trl/gsm-hard")["train"]
    qs, gs = [], []
    for ex in ds:
        q = ex["instruction"].strip().replace("  ", " ")
        ans_raw = ex["response"]
        ans_str = ans_raw.split("####")[-1] if "####" in ans_raw else ans_raw
        try:
            a = float(ans_str.replace(",", "").strip())
        except (ValueError, AttributeError):
            a = float("inf")
        qs.append(q)
        gs.append(a)
    return qs[:num_examples], gs[:num_examples], extract_answer_number


def _extract_choice_12(text):
    """Extract a 1 or 2 (PiQA solution choice) from generation."""
    m = re.search(r"answer is\s*\$?\s*([12])\b", text, re.IGNORECASE)
    if m: return int(m.group(1))
    m = re.search(r"\b(?:solution|choice|option)\s*([12])\b", text, re.IGNORECASE)
    if m: return int(m.group(1))
    digits = re.findall(r"(?<!\d)([12])(?!\d)", text)
    return int(digits[-1]) if digits else -1


def load_piqa(num_examples):
    """PiQA: physical commonsense binary multiple-choice. Downloads the
    official JSONL directly (HF dropped script datasets)."""
    import json, os, urllib.request
    cache_dir = os.path.expanduser("~/.cache/piqa_raw")
    os.makedirs(cache_dir, exist_ok=True)
    urls = {
        "valid.jsonl": "https://yonatanbisk.com/piqa/data/valid.jsonl",
        "valid-labels.lst": "https://yonatanbisk.com/piqa/data/valid-labels.lst",
    }
    for fname, url in urls.items():
        p = os.path.join(cache_dir, fname)
        if not os.path.exists(p):
            urllib.request.urlretrieve(url, p)
    examples = []
    with open(os.path.join(cache_dir, "valid.jsonl")) as f:
        for ln in f: examples.append(json.loads(ln))
    with open(os.path.join(cache_dir, "valid-labels.lst")) as f:
        labels = [int(x.strip()) for x in f if x.strip()]
    qs, gs = [], []
    for ex, lab in zip(examples, labels):
        q = (
            f"Goal: {ex['goal']}\n"
            f"Solution 1: {ex['sol1']}\n"
            f"Solution 2: {ex['sol2']}\n"
            f"Which solution is better? Answer with just '1' or '2'."
        )
        qs.append(q)
        gs.append(lab + 1)        # 0/1 -> 1/2
    return qs[:num_examples], gs[:num_examples], _extract_choice_12


def _extract_choice_15(text):
    """Extract a 1..5 from the model's output (LOGIC-701 multiple choice)."""
    m = re.search(r"answer is\s*\$?\s*([1-5])\b", text, re.IGNORECASE)
    if m: return int(m.group(1))
    m = re.search(r"\b(?:option|choice)\s*([1-5])\b", text, re.IGNORECASE)
    if m: return int(m.group(1))
    digits = re.findall(r"(?<!\d)([1-5])(?!\d)", text)
    return int(digits[-1]) if digits else -1


def load_logic701_mc(num_examples):
    """LOGIC-701 multiple-choice (5-way): score by option index 1..5."""
    ds = load_dataset("hivaze/LOGIC-701", "en")["train"]
    qs, gs = [], []
    for ex in ds:
        opts = "\n".join(f"{i}. {ex[f'answer_option_{i}']}" for i in range(1, 6))
        q = (
            f"{ex['problem_statement'].strip()}\n\nOptions:\n{opts}\n\n"
            f"Answer with just the option number (1, 2, 3, 4, or 5)."
        )
        qs.append(q)
        gs.append(int(ex["correct_option_number"]))
    return qs[:num_examples], gs[:num_examples], _extract_choice_15


def load_logic701_numeric(num_examples):
    ds = load_dataset("hivaze/LOGIC-701", "en")["train"]
    qs, gs = [], []
    for ex in ds:
        c = int(ex["correct_option_number"])
        cv = str(ex[f"answer_option_{c}"]).strip()
        if cv.lower() == "another answer":
            continue
        cn = _logic701_clean_num(cv)
        if cn is None:
            continue
        others = [_logic701_clean_num(str(ex[f"answer_option_{j}"])) for j in range(1, 6) if j != c]
        if any(o is not None and o == cn for o in others):
            continue
        qs.append(ex["problem_statement"].strip().replace("  ", " "))
        gs.append(cn)
    return qs[:num_examples], gs[:num_examples], extract_answer_number


def load_mathqa_numeric(num_examples):
    ds = load_dataset("allenai/math_qa", trust_remote_code=True)["test"]
    qs, gs = [], []
    for ex in ds:
        q = ex["Problem"].strip()
        rationale = ex.get("Rationale", "")
        ans_letter = ex["correct"].strip().lower()
        # Parse "a ) 4 , b ) 8 , c ) 12 , d ) 16 , e ) 20" → dict
        opt_str = ex["options"]
        opts = re.findall(r"([a-e])\s*\)\s*([^,]+?)(?=,\s*[a-e]\s*\)|$)", opt_str)
        opt_map = {k: v.strip() for k, v in opts}
        if ans_letter not in opt_map:
            continue
        cv = opt_map[ans_letter]
        cn = _logic701_clean_num(cv)
        if cn is None:
            continue
        others = [_logic701_clean_num(v) for k, v in opt_map.items() if k != ans_letter]
        if any(o is not None and o == cn for o in others):
            continue
        qs.append(q)
        gs.append(cn)
    return qs[:num_examples], gs[:num_examples], extract_answer_number


def load_cf_balanced(num_examples):
    """Magnitude-balanced counterfactual SVAMP set generated for CODI work."""
    cf_path = Path(__file__).resolve().parent.parent.parent / "cf-datasets" / "cf_balanced.json"
    rows = json.load(open(cf_path))
    qs = [r["cf_question_concat"].strip().replace("  ", " ") for r in rows]
    gs = [float(r["cf_answer"]) for r in rows]
    return qs[:num_examples], gs[:num_examples], extract_answer_number


def _load_simple_cf(filename, num_examples):
    """Loader for simple cf-datasets/{filename}.json with question_concat/answer fields."""
    cf_path = Path(__file__).resolve().parent.parent.parent / "cf-datasets" / filename
    rows = json.load(open(cf_path))
    qs = [r["question_concat"].strip().replace("  ", " ") for r in rows]
    gs = [float(r["answer"]) for r in rows]
    return qs[:num_examples], gs[:num_examples], extract_answer_number


def load_data(dataset, num_examples):
    if dataset == "svamp":
        return load_svamp(num_examples)
    if dataset == "gsm-hard":
        return load_gsm_hard(num_examples)
    if dataset == "cf_balanced":
        return load_cf_balanced(num_examples)
    if dataset == "vary_numerals":
        return _load_simple_cf("vary_numerals.json", num_examples)
    if dataset == "vary_operator":
        return _load_simple_cf("vary_operator.json", num_examples)
    if dataset == "vary_a":
        return _load_simple_cf("vary_a.json", num_examples)
    if dataset == "vary_b":
        return _load_simple_cf("vary_b.json", num_examples)
    if dataset == "vary_a_2digit":
        return _load_simple_cf("vary_a_2digit.json", num_examples)
    if dataset == "vary_b_2digit":
        return _load_simple_cf("vary_b_2digit.json", num_examples)
    if dataset == "vary_both_2digit":
        return _load_simple_cf("vary_both_2digit.json", num_examples)
    if dataset == "logic701_numeric":
        return load_logic701_numeric(num_examples)
    if dataset == "logic701":
        return load_logic701_mc(num_examples)
    if dataset == "piqa":
        return load_piqa(num_examples)
    if dataset == "mathqa_numeric":
        return load_mathqa_numeric(num_examples)
    raise ValueError(f"unknown dataset: {dataset}")


# --------------------------------------------------------------------------- #
# Activation hooks on Huginn's 4 recurrent core blocks. Each block fires once
# per recurrence step during the prompt forward; we collect the residual at
# the last prompt token at every call, in order.
# --------------------------------------------------------------------------- #

class CoreBlockCapture:
    def __init__(self, model):
        self.model = model
        self.core_blocks = list(model.transformer.core_block)
        self.handles = []
        # buf[i] = list of (B, H) tensors, one per call to core_block[i]
        self.buf = [[] for _ in self.core_blocks]
        self._capture_last_token_only = True

    def _hook(self, idx):
        def fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Capture last-token only (matches what the CODI pipeline saved).
            self.buf[idx].append(h[:, -1, :].detach().to(torch.float32).cpu())
        return fn

    def attach(self):
        self.detach()
        self.handles = [b.register_forward_hook(self._hook(i)) for i, b in enumerate(self.core_blocks)]

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        for b in self.buf:
            b.clear()

    def stack(self):
        """After a single forward, return tensor (S, L, B, H) where S = number
        of recurrence steps (= len(buf[0])) and L = num core blocks."""
        S = len(self.buf[0])
        if S == 0:
            return None
        for i in range(1, len(self.buf)):
            assert len(self.buf[i]) == S, (
                f"core_block call counts disagree: {[len(x) for x in self.buf]}"
            )
        # buf[i][s]: (B, H)
        per_step = []
        for s in range(S):
            per_step.append(torch.stack([self.buf[i][s] for i in range(len(self.buf))], dim=0))
            # shape (L, B, H)
        out = torch.stack(per_step, dim=0)  # (S, L, B, H)
        return out


# --------------------------------------------------------------------------- #
# Main eval loop
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["svamp", "gsm-hard", "cf_balanced", "vary_numerals", "vary_operator", "vary_a", "vary_b", "vary_a_2digit", "vary_b_2digit", "vary_both_2digit", "logic701_numeric", "logic701", "piqa", "mathqa_numeric"], required=True)
    p.add_argument("--num_examples", type=int, default=10**9)
    p.add_argument("--num_steps", type=int, default=32, help="Huginn recurrence depth.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_save_activations", action="store_true",
                   help="Don't save (potentially huge) activation tensors.")
    p.add_argument("--model_name", default=MODEL_NAME)
    args = p.parse_args()

    device = "cuda"
    print(f"[huginn] loading {args.model_name}", flush=True)
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device,
    )
    model.eval()
    H = model.config.n_embd
    L = len(model.transformer.core_block)
    print(f"[huginn] loaded in {time.time()-t0:.1f}s  n_embd={H} core_blocks={L} num_steps={args.num_steps}", flush=True)

    questions, golds, scoring_fn = load_data(args.dataset, args.num_examples)
    N = len(questions)
    print(f"[huginn] dataset={args.dataset} N={N}", flush=True)

    cap = CoreBlockCapture(model) if not args.no_save_activations else None
    results = []
    activations = []  # each: (S, L, H)

    bs = args.batch_size
    t_eval = time.time()
    for i in range(0, N, bs):
        batch_q = questions[i : i + bs]
        batch_g = golds[i : i + bs]
        prompts = [PROMPT_TEMPLATE.format(q=q) for q in batch_q]
        inp = tok(prompts, return_tensors="pt", padding="longest",
                  return_token_type_ids=False).to(device)
        prompt_len = inp["input_ids"].shape[1]

        if cap is not None:
            cap.clear()
            cap.attach()

        # 1) Prompt forward to capture activations at last prompt token
        with torch.no_grad():
            _ = model(**inp, num_steps=args.num_steps, use_cache=True)

        if cap is not None:
            stacked = cap.stack()  # (S, L, B, H)
            cap.detach()
            # split per-question
            for b in range(stacked.shape[2]):
                activations.append(stacked[:, :, b, :])  # (S, L, H)

        # 2) Generate the answer (no need to capture activations during decode)
        with torch.no_grad():
            gen = model.generate(
                **inp, max_new_tokens=args.max_new_tokens,
                num_steps=args.num_steps, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        decoded = tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        for b, (q, gold, text) in enumerate(zip(batch_q, batch_g, decoded)):
            pred = scoring_fn(text)
            correct = pred == gold
            results.append({
                "idx": i + b,
                "question": q,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "response": text,
            })
        done = min(i + bs, N)
        elapsed = time.time() - t_eval
        rate = done / max(elapsed, 1e-9)
        eta = (N - done) / max(rate, 1e-9)
        batch_acc = sum(r["correct"] for r in results[-len(batch_q):]) / len(batch_q)
        print(
            f"[huginn] {done}/{N}  batch_acc={batch_acc:.2f}  "
            f"rate={rate:.2f} q/s  eta={eta:.0f}s",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    acc = sum(r["correct"] for r in results) / max(len(results), 1)
    print(f"[huginn] accuracy = {acc:.3f} on {len(results)} examples", flush=True)
    if activations:
        # (N, S, L, H) cast to bf16 to halve disk
        acts = torch.stack(activations, dim=0).to(torch.bfloat16)
        torch.save(acts, out_dir / "activations.pt")
        print(f"[huginn] activations: {tuple(acts.shape)} -> {out_dir/'activations.pt'}", flush=True)


if __name__ == "__main__":
    main()
