"""Magnitude-matched operator dataset for GSM8K.

For each (operator × magnitude bucket), generate ~60 single-step word
problems whose gold answer falls in the bucket.  This decouples operator
from gold-magnitude:
   - WITHIN a magnitude bucket, all 4 operators have overlapping gold ranges.
   - WITHIN-bucket operator probing tests whether operator is encoded
     independently of gold magnitude.

Buckets:  small (10-50)  medium (50-500)  large (500-5000)
Per (op × bucket): 60 problems.  Total: 4 × 3 × 60 = 720 problems.

This is the magnitude-controlled control the user asked for.
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
OUT = PD / "gsm8k_op_magmatched.json"

N_PER_CELL = 60
MAGNITUDE_BUCKETS = [
    ("small",  10, 50),
    ("medium", 50, 500),
    ("large",  500, 5000),
]
OPERATORS = {"Addition": "+", "Subtraction": "-",
             "Multiplication": "*", "Common-Division": "/"}
MODEL = "gpt-5-mini"
N_WORKERS = 24


SYSTEM = (
    "You generate single-step GSM8K-style word problems for a magnitude-"
    "controlled operator dataset.  STRICT requirements:\n"
    "  1. Each problem uses EXACTLY ONE arithmetic operator (the requested "
    "one).  Single-step computation: a OP b = gold.\n"
    "  2. The gold answer (a OP b) must be a positive integer in the requested "
    "magnitude range.\n"
    "  3. For Common-Division: a must be exactly divisible by b.\n"
    "  4. For Subtraction: a > b.\n"
    "  5. Vary scenario: people, items, contexts.\n"
    "  6. Answer trace: single <<a op b = gold>> marker, then '#### gold' "
    "on next line.\n"
    "Output ONLY valid JSON:\n"
    '  {"a": <int>, "b": <int>, "question": "...", "answer_trace": "..."}'
)


def prompt_for(op_name, op_sym, lo, hi, seed):
    extra = ""
    if op_sym == "*":
        # Mul tends to overshoot — help with operand bounds
        max_a = int((hi / 2) ** 0.5 + 5)
        extra = (f"  Hint for Multiplication: pick a, b in [2, {max_a}] so a*b "
                 f"lands in [{lo}, {hi}].\n")
    if op_sym == "/":
        # Div needs a much larger than b
        extra = (f"  Hint for Division: pick b in [2, 20] and a so that a/b is "
                 f"an integer in [{lo}, {hi}] (so a in [b*{lo}, b*{hi}]).\n")
    return (
        f"Generate ONE {op_name} problem.\n"
        f"Operator symbol: {op_sym}\n"
        f"Required gold range: [{lo}, {hi}] (a {op_sym} b must be a "
        f"positive integer in this range).\n"
        + extra +
        f"  Vary scenario.  Seed: {seed}.\n"
        f"Return JSON only."
    )


def verify(v, op_sym, lo, hi):
    a = v.get("a"); b = v.get("b")
    if a is None or b is None: return False, "missing operands"
    try: a = int(a); b = int(b)
    except Exception: return False, "non-int"
    if a <= 0 or b <= 0: return False, "non-pos operand"
    if op_sym == "-" and a <= b: return False, "a<=b"
    if op_sym == "/" and (b == 0 or a % b != 0): return False, "non-exact div"
    g = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op_sym]
    if not (lo <= g <= hi): return False, f"gold {g} out of [{lo},{hi}]"
    trace = v.get("answer_trace", "").strip()
    markers = re.findall(r"<<(.+?)=(-?\d+\.?\d*)>>", trace)
    if len(markers) != 1: return False, "wrong marker count"
    expr, claimed = markers[0]; expr = expr.strip()
    toks = re.findall(r"[+\-*/]", expr)
    if expr.startswith("-") and toks and toks[0] == "-": toks = toks[1:]
    if len(toks) != 1 or toks[0] != op_sym: return False, "wrong op symbol"
    try: ev = eval(expr)
    except Exception: return False, "eval fail"
    if abs(ev - g) > 1e-3: return False, "marker mismatch"
    m = re.search(r"####\s*(-?\d+\.?\d*)", trace.replace(",", ""))
    if not m: return False, "no ####"
    if abs(float(m.group(1)) - g) > 1e-3: return False, "#### mismatch"
    return True, g


def gen_one(client, op_name, op_sym, lo, hi, seed):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_for(op_name, op_sym, lo, hi, seed)},
            ],
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return None, f"api: {e}"
    ok, gold_or_msg = verify(obj, op_sym, lo, hi)
    if not ok: return None, gold_or_msg
    return {
        "type": op_name, "magnitude_bucket": f"{lo}-{hi}",
        "a": int(obj["a"]), "b": int(obj["b"]),
        "question_concat": obj["question"].strip().replace("  ", " "),
        "answer": float(gold_or_msg),
        "answer_trace": obj["answer_trace"].strip(),
        "source": f"llm-gen-magmatched-{MODEL}",
        "seed": seed,
    }, "ok"


def main():
    client = OpenAI()
    rows = []; failures = Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for op_name, op_sym in OPERATORS.items():
            for bname, lo, hi in MAGNITUDE_BUCKETS:
                ok = 0; attempts = 0
                while ok < N_PER_CELL and attempts < N_PER_CELL * 5:
                    batch_size = min(N_WORKERS, (N_PER_CELL - ok) * 2)
                    futs = [pool.submit(gen_one, client, op_name, op_sym, lo, hi,
                                        attempts * 100 + i + hash((op_name, bname)) % 99999)
                            for i in range(batch_size)]
                    for fut in as_completed(futs):
                        attempts += 1
                        r, msg = fut.result()
                        if r is None:
                            failures[msg.split(":")[0]] += 1; continue
                        rows.append(r); ok += 1
                        if ok >= N_PER_CELL: break
                print(f"  {op_name} {bname} [{lo}-{hi}]: kept {ok} "
                      f"(attempts={attempts}, total={len(rows)})  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                OUT.write_text(json.dumps(rows, indent=2))
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nDONE: {len(rows)} rows")
    print(f"by op: {dict(Counter(r['type'] for r in rows))}")
    print(f"by bucket: {dict(Counter(r['magnitude_bucket'] for r in rows))}")
    print(f"failures: {failures.most_common(8)}")


if __name__ == "__main__":
    main()
