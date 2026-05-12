"""LLM-generated NATURAL multi-step GSM8K-style CF dataset.

Mixed-operator chains are ALLOWED — we just verify the marker chain is
internally consistent and parse the operator-presence labels from it.

For each (magnitude bucket × step count), prompts GPT-5-mini to generate
a natural multi-step word problem.  Each problem is verified by:
   1. Every <<expr=result>> marker evaluates correctly.
   2. The final marker value equals the #### gold value.
   3. The gold is a positive integer.
After verification, we PARSE the operator symbols used in each marker to
derive multi-label operator-presence labels {has_add, has_sub, has_mul,
has_div, n_distinct_ops}.

Output: gsm8k_cf_natural.json
   [{"question_concat": str, "answer": float, "answer_trace": str,
     "operands": list, "magnitude_bucket": str, "n_steps": int,
     "has_add": bool, "has_sub": bool, "has_mul": bool, "has_div": bool,
     "n_distinct_ops": int, "operator_string": "add+mul+sub"}, ...]
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

PD = Path(__file__).resolve().parent
OUT = PD / "gsm8k_cf_natural.json"

MAGNITUDE_BUCKETS = ["small (1-50)", "medium (50-500)", "large (500-5000)"]
STEP_COUNTS = [2, 3, 4]      # number of <<...>> marker steps
N_PER_CELL = 70              # 3 buckets × 3 step-counts × 70 ≈ 630 total
MODEL = "gpt-5-mini"
N_WORKERS = 24

SYSTEM = (
    "You generate GSM8K-style multi-step arithmetic word problems for a "
    "counterfactual mechanistic-interpretability dataset.  Each problem must:\n"
    "  1. Be a coherent natural-language word problem (people, items, "
    "scenarios). Vary scenarios.\n"
    "  2. Have EXACTLY the requested number of computation steps, each shown "
    "as a <<expr=result>> marker, mirroring the GSM8K answer format.\n"
    "  3. End with the line '#### N' where N is the final integer answer.\n"
    "  4. Use a MIX of arithmetic operators when natural (e.g., +, *, -). "
    "Do not artificially force just one operator.\n"
    "  5. Result of every step is a positive integer; final answer is a "
    "positive integer in the requested magnitude bucket.\n"
    "  6. For division, use ONLY exact integer division.\n"
    "Output ONLY valid JSON of the form:\n"
    '  {"question": "...", "operands": [n1, n2, ...], "answer": int, '
    '"answer_trace": "step text with <<...>> markers and final \'#### N\' line"}\n'
)


def gen_prompt(mag: str, n_steps: int, seed: int):
    return (
        f"Generate ONE natural multi-step word problem.\n"
        f"Required magnitude bucket for final answer: {mag}\n"
        f"Required number of <<expr=result>> computation steps: EXACTLY {n_steps}.\n"
        f"Mix operators naturally as needed (+, -, *, /).\n"
        f"Vary the scenario. Seed: {seed}.\n"
        f"Return JSON only."
    )


def parse_operator_presence(answer_trace: str):
    """Return ({has_add, has_sub, has_mul, has_div, n_distinct_ops},
    operator_string).  Ignores leading minus on first number in an expression.
    """
    has = {"add": False, "sub": False, "mul": False, "div": False}
    for m in re.finditer(r"<<(.+?)=(-?\d+\.?\d*)>>", answer_trace):
        expr = m.group(1).strip()
        toks = re.findall(r"[+\-*/]", expr)
        if expr.startswith("-") and toks and toks[0] == "-":
            toks = toks[1:]
        for t in toks:
            if t == "+": has["add"] = True
            elif t == "-": has["sub"] = True
            elif t == "*": has["mul"] = True
            elif t == "/": has["div"] = True
    n_distinct = sum(has.values())
    op_str = "+".join(k for k in ["add", "sub", "mul", "div"] if has[k]) or "none"
    return has, n_distinct, op_str


def verify(answer_trace: str, gold: float, expected_steps: int):
    markers = re.findall(r"<<(.+?)=(-?\d+\.?\d*)>>", answer_trace)
    if not markers: return False, "no markers"
    if len(markers) != expected_steps: return False, f"wrong step count {len(markers)} vs {expected_steps}"
    for expr, claimed in markers:
        expr = expr.strip()
        if not re.match(r"^[\d+\-*/().\s]+$", expr): return False, f"bad chars: {expr}"
        try: v = eval(expr)
        except Exception: return False, f"eval fail: {expr}"
        if abs(float(v) - float(claimed)) > 1e-3: return False, f"marker mismatch {expr}={v}≠{claimed}"
        if v != int(v) or v < 0: return False, f"non-positive-int step: {expr}={v}"
    m = re.search(r"####\s*(-?\d+\.?\d*)", answer_trace.replace(",", ""))
    if not m: return False, "no ####"
    final = float(m.group(1))
    if abs(final - float(markers[-1][1])) > 1e-3: return False, "marker ≠ ####"
    if abs(final - gold) > 1e-3: return False, f"gold {gold} ≠ {final}"
    if final != int(final) or final <= 0: return False, "non-positive-int gold"
    return True, "ok"


def gen_one(client: OpenAI, mag: str, n_steps: int, seed: int):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": gen_prompt(mag, n_steps, seed)},
            ],
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return None, f"api: {e}"
    q = (obj.get("question") or "").strip()
    ans = obj.get("answer"); trace = (obj.get("answer_trace") or "").strip()
    ops_list = obj.get("operands") or []
    if not q or ans is None or not trace: return None, "missing fields"
    try: ans_f = float(ans)
    except Exception: return None, "bad ans"
    ok, msg = verify(trace, ans_f, n_steps)
    if not ok: return None, msg
    has, n_distinct, op_str = parse_operator_presence(trace)
    return {
        "magnitude_bucket": mag, "n_steps": n_steps,
        "question_concat": q, "answer": ans_f,
        "answer_trace": trace, "operands": ops_list,
        "has_add": has["add"], "has_sub": has["sub"],
        "has_mul": has["mul"], "has_div": has["div"],
        "n_distinct_ops": n_distinct, "operator_string": op_str,
        "source": f"llm-gen-natural-{MODEL}", "seed": seed,
    }, "ok"


def main():
    client = OpenAI()
    rows = []; seen = set(); fail_reasons = Counter()
    for mag in MAGNITUDE_BUCKETS:
        for n_steps in STEP_COUNTS:
            ok = 0; attempts = 0
            print(f"\n=== {mag} | n_steps={n_steps} | target ~{N_PER_CELL} ===", flush=True)
            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                while ok < N_PER_CELL and attempts < N_PER_CELL * 4:
                    batch = [pool.submit(gen_one, client, mag, n_steps,
                                          attempts * 200 + i + hash((mag, n_steps)) % 99999)
                             for i in range(min(N_WORKERS, N_PER_CELL * 4 - attempts))]
                    for fut in as_completed(batch):
                        attempts += 1
                        r, msg = fut.result()
                        if r is None:
                            fail_reasons[msg.split(":")[0]] += 1
                            continue
                        if r["question_concat"] in seen: continue
                        seen.add(r["question_concat"])
                        rows.append(r); ok += 1
                        if ok >= N_PER_CELL: break
            print(f"  kept {ok} (attempts={attempts}, total={len(rows)})", flush=True)
            OUT.write_text(json.dumps(rows, indent=2))
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nDONE: {len(rows)} rows → {OUT}")
    print(f"by magnitude: {Counter(r['magnitude_bucket'] for r in rows)}")
    print(f"by n_steps: {Counter(r['n_steps'] for r in rows)}")
    print(f"by n_distinct_ops: {Counter(r['n_distinct_ops'] for r in rows)}")
    for op in ("add", "sub", "mul", "div"):
        c = sum(1 for r in rows if r[f"has_{op}"])
        print(f"  has_{op}: {c}/{len(rows)} ({c/max(len(rows),1)*100:.1f}%)")
    print(f"fail reasons: {fail_reasons.most_common(8)}")


if __name__ == "__main__":
    main()
