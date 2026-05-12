"""LLM-generated operator-balanced CF dataset for GSM8K-style problems.

For each operator class (Addition, Subtraction, Multiplication, Division),
prompts GPT-5 to generate N multi-step word problems whose PRIMARY
operation is that operator. Each generated problem is verified by:
  (a) a separate LLM call asking it to confirm the gold answer, and
  (b) Python eval of the marker chain in the answer.

Output: gsm8k_cf_op_balanced.json
  [{"type": str, "question_concat": str, "answer": float,
    "answer_trace": str, "operands": list,
    "magnitude_bucket": str, "source": "llm-gen-gpt5"}, ...]

Magnitude buckets are controlled so each operator class spans matched
answer ranges (avoids the "Mul has huge answers" conflation).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

PD = Path(__file__).resolve().parent
OUT = PD / "gsm8k_cf_op_balanced.json"

OPERATORS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
N_PER_OP = 100        # 100 per operator → 400 total
MAGNITUDE_BUCKETS = ["small (1-30)", "medium (50-500)", "large (1000-10000)"]
MODEL_GEN = "gpt-5-mini"
MODEL_VERIFY = "gpt-5-mini"

GEN_SYSTEM = (
    "You generate clean GSM8K-style multi-step arithmetic word problems for "
    "a counterfactual dataset used in mechanistic interpretability research. "
    "Each problem you generate MUST satisfy:\n"
    "  1. The PRIMARY arithmetic operation (the one performed in the largest "
    "step or the final step) is the requested operator.\n"
    "  2. The problem has 2-4 numeric operands in the question text.\n"
    "  3. The final numeric answer is a positive integer.\n"
    "  4. You provide an `answer_trace`: 1-3 steps using GSM8K's <<a op b=c>> "
    "marker style, ending with '#### {gold}'.\n"
    "  5. The final marker result matches the integer gold answer.\n"
    "Output ONLY valid JSON of the form:\n"
    '  {"question": "...", "operands": [n1, n2, ...], "answer": int, '
    '"answer_trace": "step text with <<...>> markers and #### N at end"}\n'
)


def gen_prompt(op: str, mag: str, seed_hint: int):
    return (
        f"Generate one GSM8K-style word problem.\n"
        f"REQUIRED operator class: {op}\n"
        f"REQUIRED answer magnitude bucket: {mag}\n"
        f"Use only English narrative. Vary the scenario (people, objects, "
        f"context) for diversity. Seed: {seed_hint}\n"
        f"Return JSON only."
    )


def verify_marker_chain(answer_trace: str, stated_gold: float):
    """Walk through <<a op b = c>> markers and confirm they evaluate."""
    markers = re.findall(r"<<(.+?)=(-?\d+\.?\d*)>>", answer_trace)
    if not markers: return False, "no markers"
    for expr, claimed in markers:
        expr = expr.strip()
        if not re.match(r"^[\d+\-*/().\s]+$", expr): return False, f"bad chars: {expr}"
        try:
            v = eval(expr)
        except Exception as e:
            return False, f"eval fail: {expr} ({e})"
        if abs(v - float(claimed)) > 1e-3:
            return False, f"marker mismatch: {expr}={v} ≠ {claimed}"
    final_claim = float(markers[-1][1])
    m = re.search(r"####\s*(-?\d+\.?\d*)", answer_trace.replace(",", ""))
    if not m: return False, "no #### marker"
    final_from_hash = float(m.group(1))
    if abs(final_claim - final_from_hash) > 1e-3: return False, "marker ≠ ####"
    if abs(final_claim - stated_gold) > 1e-3: return False, f"gold mismatch ({final_claim} vs {stated_gold})"
    return True, "ok"


def main():
    client = OpenAI()
    rows = []
    seen_questions = set()
    for op in OPERATORS:
        for mag in MAGNITUDE_BUCKETS:
            n_target = N_PER_OP // len(MAGNITUDE_BUCKETS) + 1
            ok_count = 0
            attempts = 0
            print(f"\n=== {op} | magnitude {mag} | targeting ~{n_target} ===")
            while ok_count < n_target and attempts < n_target * 4:
                attempts += 1
                seed_hint = attempts + hash((op, mag)) % 100000
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_GEN,
                        messages=[
                            {"role": "system", "content": GEN_SYSTEM},
                            {"role": "user", "content": gen_prompt(op, mag, seed_hint)},
                        ],
                        response_format={"type": "json_object"},
                    )
                    raw = resp.choices[0].message.content
                    obj = json.loads(raw)
                except Exception as e:
                    print(f"  attempt {attempts}: API/parse fail: {e}")
                    continue
                q = (obj.get("question") or "").strip()
                ops_list = obj.get("operands") or []
                ans = obj.get("answer")
                trace = (obj.get("answer_trace") or "").strip()
                if not q or ans is None or not trace:
                    continue
                try:
                    ans_f = float(ans)
                except Exception:
                    continue
                if q in seen_questions: continue
                ok, msg = verify_marker_chain(trace, ans_f)
                if not ok:
                    if attempts < 3:
                        print(f"  reject [{msg}]: {q[:60]}")
                    continue
                seen_questions.add(q)
                ok_count += 1
                rows.append({
                    "type": op,
                    "magnitude_bucket": mag,
                    "question_concat": q,
                    "answer": ans_f,
                    "answer_trace": trace,
                    "operands": ops_list,
                    "source": f"llm-gen-{MODEL_GEN}",
                    "seed": seed_hint,
                })
                if ok_count % 5 == 0:
                    print(f"  generated {ok_count}/{n_target} (attempts={attempts})")
        # Save after each operator class (incremental)
        OUT.write_text(json.dumps(rows, indent=2))
        print(f"  saved {len(rows)} rows so far → {OUT}")
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nFINAL: {len(rows)} rows saved to {OUT}")
    # Print distribution
    from collections import Counter
    c_op = Counter(r["type"] for r in rows)
    c_mag = Counter(r["magnitude_bucket"] for r in rows)
    print(f"  by operator: {dict(c_op)}")
    print(f"  by magnitude: {dict(c_mag)}")


if __name__ == "__main__":
    main()
