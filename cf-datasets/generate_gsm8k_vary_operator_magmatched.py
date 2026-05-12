"""LLM-generated MAGNITUDE-MATCHED vary_operator CF.

Unlike gsm8k_vary_operator.json (where (a, b) is identical across the 4
operator variants — but the gold magnitude differs wildly because mul/div
have different scales than add/sub), this dataset CONTROLS GOLD ANSWER
MAGNITUDE.

For each template (scenario_id × magnitude_bucket):
  All four operator variants share:
    - Same scenario / characters / story setting
    - Same magnitude bucket for the final gold answer
  Each variant has its OWN (a, b) chosen so that a op b lands in the
  bucket and is a positive integer.

Magnitude buckets: small (10-50), medium (50-500), large (500-5000).

This trades the operand-uniformity guarantee for a GOLD-MAGNITUDE-
UNIFORMITY guarantee, which is the right control for testing whether
the operator probe was reading gold magnitude rather than operator.

Output: gsm8k_vary_operator_magmatched.json
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

PD = Path(__file__).resolve().parent
OUT = PD / "gsm8k_vary_operator_magmatched.json"

N_TEMPLATES = 60      # 60 × 4 = 240 problems
MAGNITUDE_BUCKETS = [
    ("small",  (10, 50)),
    ("medium", (50, 500)),
    ("large",  (500, 5000)),
]
MODEL = "gpt-5-mini"
N_WORKERS = 12
OPERATORS = {"Addition": "+", "Subtraction": "-",
             "Multiplication": "*", "Common-Division": "/"}


SYSTEM = (
    "You generate four GSM8K-style word problems that share the SAME scenario "
    "(same characters, items, setting) but use FOUR DIFFERENT operators.\n"
    "STRICT requirements:\n"
    "  1. The four variants are labeled Addition / Subtraction / Multiplication "
    "/ Common-Division.\n"
    "  2. Each variant uses its own choice of operands (a, b) — they are NOT "
    "required to share operands across variants.\n"
    "  3. Each variant's gold answer (a op b) must be a positive integer in "
    "the requested magnitude range.\n"
    "  4. For Common-Division: a must be divisible by b.\n"
    "  5. For Subtraction: a > b.\n"
    "  6. Each variant has a single <<a op b = gold>> marker followed by "
    "'#### gold' on its own line.\n"
    "Output ONLY valid JSON of the form:\n"
    '  {"scenario_seed": "...", "magnitude_bucket": "small|medium|large", '
    '"variants": [\n'
    '    {"type": "Addition",        "a": <int>, "b": <int>, "question": "...", "answer_trace": "..."},\n'
    '    {"type": "Subtraction",     "a": <int>, "b": <int>, "question": "...", "answer_trace": "..."},\n'
    '    {"type": "Multiplication",  "a": <int>, "b": <int>, "question": "...", "answer_trace": "..."},\n'
    '    {"type": "Common-Division", "a": <int>, "b": <int>, "question": "...", "answer_trace": "..."}\n'
    '  ]}'
)


def prompt_for(bucket_name, lo, hi, seed):
    return (
        f"Generate one operator-magnitude-matched template.\n"
        f"Magnitude bucket: {bucket_name} — all four variants' gold answers "
        f"must be positive integers in the range [{lo}, {hi}].\n"
        f"For each operator, choose (a, b) so a op b falls in the range. "
        f"You may use different (a, b) for different operators.\n"
        f"Vary the scenario. Seed: {seed}.\n"
        f"Return JSON only."
    )


def verify_variant(v, op_sym, lo, hi):
    a = v.get("a"); b = v.get("b")
    if a is None or b is None: return False, "missing a or b"
    try: a = int(a); b = int(b)
    except Exception: return False, "non-int operand"
    if a <= 0 or b <= 0: return False, "non-positive operand"
    if op_sym == "/":
        if b == 0 or a % b != 0: return False, "non-exact division"
    if op_sym == "-" and a <= b: return False, "a<=b for subtraction"
    gold_arith = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op_sym]
    if not (lo <= gold_arith <= hi):
        return False, f"gold {gold_arith} not in [{lo}, {hi}]"
    trace = v.get("answer_trace", "").strip()
    markers = re.findall(r"<<(.+?)=(-?\d+\.?\d*)>>", trace)
    if len(markers) != 1: return False, f"expected 1 marker, got {len(markers)}"
    expr, claimed = markers[0]
    expr = expr.strip()
    toks = re.findall(r"[+\-*/]", expr)
    if expr.startswith("-") and toks and toks[0] == "-": toks = toks[1:]
    if len(toks) != 1 or toks[0] != op_sym: return False, f"wrong op in {expr}"
    try: v_eval = eval(expr)
    except Exception: return False, f"eval fail: {expr}"
    if abs(v_eval - gold_arith) > 1e-3: return False, f"marker={v_eval}≠expected={gold_arith}"
    m = re.search(r"####\s*(-?\d+\.?\d*)", trace.replace(",", ""))
    if not m: return False, "no ####"
    if abs(float(m.group(1)) - gold_arith) > 1e-3: return False, "#### mismatch"
    return True, gold_arith


def gen_one_template(client, t_id, bucket_name, lo, hi, seed):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_for(bucket_name, lo, hi, seed)},
            ],
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return None, f"api: {e}"
    variants = obj.get("variants", [])
    if len(variants) != 4: return None, f"got {len(variants)} variants"
    out_rows = []
    seen_types = set()
    for v in variants:
        t = v.get("type", "")
        if t not in OPERATORS or t in seen_types: continue
        ok, gold_or_msg = verify_variant(v, OPERATORS[t], lo, hi)
        if not ok: return None, f"{t}: {gold_or_msg}"
        seen_types.add(t)
        out_rows.append({
            "template_id": t_id, "type": t,
            "magnitude_bucket": bucket_name,
            "magnitude_range": [lo, hi],
            "a": int(v["a"]), "b": int(v["b"]),
            "question_concat": v["question"].strip().replace("  ", " "),
            "answer": float(gold_or_msg),
            "answer_trace": v["answer_trace"].strip(),
            "source": f"llm-gen-vary_operator_magmatched-{MODEL}",
            "scenario_seed": obj.get("scenario_seed", ""),
        })
    if len(out_rows) != 4: return None, f"only {len(out_rows)} valid types"
    return out_rows, "ok"


def main():
    client = OpenAI()
    rows = []
    failures = Counter()
    t0 = time.time()
    # Distribute N_TEMPLATES across 3 buckets
    n_per_bucket = N_TEMPLATES // len(MAGNITUDE_BUCKETS)
    futures = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for bi, (bname, (lo, hi)) in enumerate(MAGNITUDE_BUCKETS):
            for i in range(n_per_bucket * 3):    # over-submit
                t_id = bi * 1000 + i
                seed = t_id * 17 + 31337
                futures[pool.submit(gen_one_template, client, t_id, bname, lo, hi, seed)] = (t_id, bname)
        by_bucket = Counter()
        for fut in as_completed(futures):
            res, msg = fut.result()
            if res is None:
                failures[msg.split(":")[0]] += 1; continue
            bname = res[0]["magnitude_bucket"]
            if by_bucket[bname] >= n_per_bucket: continue
            rows.extend(res)
            by_bucket[bname] += 1
            if sum(by_bucket.values()) % 10 == 0:
                print(f"  kept {sum(by_bucket.values())}/{N_TEMPLATES} templates  "
                      f"({len(rows)} rows)  by_bucket={dict(by_bucket)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if sum(by_bucket.values()) >= N_TEMPLATES: break
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nDONE: {len(rows)} rows from {len(set(r['template_id'] for r in rows))} templates")
    print(f"by type: {dict(Counter(r['type'] for r in rows))}")
    print(f"by bucket: {dict(Counter(r['magnitude_bucket'] for r in rows))}")
    print(f"failures: {failures.most_common(8)}")


if __name__ == "__main__":
    main()
