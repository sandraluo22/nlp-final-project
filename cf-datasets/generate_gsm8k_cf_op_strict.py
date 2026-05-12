"""LLM-generated operator-PURE multi-step CF dataset for GSM8K-style probing.

Each problem uses EVERY step's operator from a single class:
  Addition       — every <<a op b=c>> step uses `+` (no other operators)
  Subtraction    — every step uses `-`
  Multiplication — every step uses `*`
  Common-Division — every step uses `/` (and divides cleanly)

This removes the operator-conflation that arises when a "primary operator"
problem still contains stray operators from other classes.

We further control magnitude via 3 buckets so each operator class spans
matched answer ranges.

Output: gsm8k_cf_op_strict.json
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
OUT = PD / "gsm8k_cf_op_strict.json"

OPERATORS = {"Addition": "+", "Subtraction": "-",
             "Multiplication": "*", "Common-Division": "/"}
N_PER_OP = 80     # 80 × 4 = 320 total
MAGNITUDE_BUCKETS = ["small (1-50)", "medium (50-500)", "large (500-5000)"]
MODEL = "gpt-5-mini"
N_WORKERS = 16

SYSTEM = (
    "You generate GSM8K-style multi-step arithmetic word problems for a "
    "counterfactual operator-isolation dataset. STRICT requirements:\n"
    "  1. Every <<expr=result>> marker in answer_trace MUST use ONLY the "
    "requested operator symbol. NO mixed operators allowed.\n"
    "  2. The problem must have 2-4 steps in the chain (chained applications "
    "of the same operator, like a-b-c or a*b*c).\n"
    "  3. The final integer answer must fall in the requested magnitude bucket.\n"
    "  4. For Common-Division: every division must be exact (integer result).\n"
    "  5. Answer is positive integer; no fractions.\n"
    "Output ONLY valid JSON of the form:\n"
    '  {"question": "...", "operands": [n1, n2, ...], "answer": int, '
    '"answer_trace": "step text with <<...>> markers, final line #### N"}\n'
)


def gen_prompt(op_name: str, op_sym: str, mag: str, seed: int):
    return (
        f"Generate ONE strictly-{op_name} word problem.\n"
        f"Required operator symbol in EVERY <<...>> marker: `{op_sym}`\n"
        f"Required magnitude bucket: {mag}\n"
        f"Number of chained steps: 2-4 applications of `{op_sym}`.\n"
        f"Vary the scenario (people, items, contexts). Seed: {seed}.\n"
        f"Return JSON only."
    )


def verify(answer_trace: str, gold: float, target_sym: str):
    markers = re.findall(r"<<(.+?)=(-?\d+\.?\d*)>>", answer_trace)
    if not markers: return False, "no markers"
    # Every marker must contain ONLY target_sym (no other op symbols).
    other = set("+-*/") - {target_sym}
    for expr, claimed in markers:
        # Strip leading minus on first number? For Subtraction we expect `-` so be careful.
        # Check that operator symbols used outside of "leading minus" all equal target.
        # Simple check: split by space-delimited tokens, look at operators between numbers.
        toks = re.findall(r"[+\-*/]", expr)
        # First token of expr might be a leading negative; ignore if expr starts with '-'.
        if expr.lstrip().startswith("-") and toks and toks[0] == "-":
            toks = toks[1:]
        if any(t != target_sym for t in toks):
            return False, f"non-target op in {expr}"
        # Evaluate
        if not re.match(r"^[\d+\-*/().\s]+$", expr): return False, f"bad chars: {expr}"
        try: v = eval(expr)
        except Exception: return False, f"eval fail: {expr}"
        if abs(float(v) - float(claimed)) > 1e-3: return False, f"marker {expr}={v}≠{claimed}"
    m = re.search(r"####\s*(-?\d+\.?\d*)", answer_trace.replace(",", ""))
    if not m: return False, "no ####"
    final = float(m.group(1))
    if abs(final - float(markers[-1][1])) > 1e-3: return False, "marker≠####"
    if abs(final - gold) > 1e-3: return False, f"gold {gold}≠{final}"
    if final != int(final) or final < 0: return False, "non-positive-int gold"
    return True, "ok"


def gen_one(client: OpenAI, op_name: str, op_sym: str, mag: str, seed: int):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": gen_prompt(op_name, op_sym, mag, seed)},
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
    ok, msg = verify(trace, ans_f, op_sym)
    if not ok: return None, msg
    return {"type": op_name, "magnitude_bucket": mag,
            "question_concat": q, "answer": ans_f,
            "answer_trace": trace, "operands": ops_list,
            "source": f"llm-gen-strict-{MODEL}", "seed": seed}, "ok"


def main():
    client = OpenAI()
    rows = []
    seen = set()
    fail_reasons = Counter()
    for op_name, op_sym in OPERATORS.items():
        for mag in MAGNITUDE_BUCKETS:
            n_target = N_PER_OP // len(MAGNITUDE_BUCKETS) + 1
            ok = 0; attempts = 0
            print(f"\n=== {op_name} [{op_sym}] | {mag} | target ~{n_target} ===", flush=True)
            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                while ok < n_target and attempts < n_target * 5:
                    batch = [pool.submit(gen_one, client, op_name, op_sym, mag,
                                          attempts * 100 + i + hash((op_name, mag)) % 99999)
                             for i in range(min(N_WORKERS, n_target * 5 - attempts))]
                    for fut in as_completed(batch):
                        attempts += 1
                        r, msg = fut.result()
                        if r is None:
                            fail_reasons[msg.split(":")[0]] += 1
                            continue
                        if r["question_concat"] in seen: continue
                        seen.add(r["question_concat"])
                        rows.append(r); ok += 1
                        if ok >= n_target: break
            print(f"  kept {ok} (attempts={attempts}, total={len(rows)})", flush=True)
        OUT.write_text(json.dumps(rows, indent=2))
        print(f"  intermediate save: {len(rows)} → {OUT}")
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nDONE: {len(rows)} rows → {OUT}")
    print(f"by op: {Counter(r['type'] for r in rows)}")
    print(f"by mag: {Counter(r['magnitude_bucket'] for r in rows)}")
    print(f"fail reasons: {fail_reasons.most_common(8)}")


if __name__ == "__main__":
    main()
