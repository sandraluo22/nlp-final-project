"""LLM-generated GSM8K-style vary_operator CF dataset.

For each template_id we generate FOUR variants — one per operator
{Addition, Subtraction, Multiplication, Common-Division} — with:
   - The same scenario context (people, items, story).
   - The same two operands (a, b).
   - The operator and result vary.

This is the operator-paired analog of SVAMP's vary_operator.json: it
isolates operator identity from operand magnitudes by holding (a, b)
fixed within each template.

Output:
   gsm8k_vary_operator.json
   each row: {template_id, type, a, b, question_concat, answer,
              answer_trace, source}
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
OUT = PD / "gsm8k_vary_operator.json"

N_TEMPLATES = 80
MODEL = "gpt-5-mini"
N_WORKERS = 8
OPERATORS = {"Addition": "+", "Subtraction": "-",
             "Multiplication": "*", "Common-Division": "/"}
OPERAND_RANGES = [
    (10, 99, 10, 99),    # 2-digit + 2-digit, large enough for div to be meaningful
    (5, 50, 2, 20),      # smaller numbers
]


SYSTEM = (
    "You generate FOUR paired GSM8K-style word problems that share the "
    "SAME scenario and operands, but apply a different arithmetic operator "
    "in each variant. STRICT requirements:\n"
    "  1. All four variants use the SAME numerical operands {a, b} and the "
    "same characters/objects/setting.\n"
    "  2. Variant types: Addition, Subtraction, Multiplication, Common-Division.\n"
    "  3. For Common-Division, choose (a, b) such that a / b is a positive "
    "integer.\n"
    "  4. For Subtraction, choose (a, b) such that a - b is a positive integer.\n"
    "  5. Each variant has a single-step <<a op b = gold>> marker followed "
    "by '#### gold' on its own line. (Single-step keeps the operator clean.)\n"
    "  6. Vary the scenario across templates: shops, animals, baking, sports, "
    "schools, factories, gardens, etc.\n"
    "Output ONLY valid JSON of the form:\n"
    '  {"a": <int>, "b": <int>, "scenario_seed": "...", '
    '"variants": [\n'
    '    {"type": "Addition",       "question": "...", "answer_trace": "Anna had <<a+b=g>>g cookies.\\n#### g"},\n'
    '    {"type": "Subtraction",    "question": "...", "answer_trace": "..."},\n'
    '    {"type": "Multiplication", "question": "...", "answer_trace": "..."},\n'
    '    {"type": "Common-Division","question": "...", "answer_trace": "..."}\n'
    '  ]}'
)


def prompt_for(a_lo, a_hi, b_lo, b_hi, seed):
    return (
        f"Generate one operator-paired template with FOUR variants.\n"
        f"Operand a in range [{a_lo}, {a_hi}], operand b in range [{b_lo}, {b_hi}].\n"
        f"All four variants must use the SAME (a, b) integers.\n"
        f"For Subtraction: ensure a > b.\n"
        f"For Common-Division: ensure a is exactly divisible by b.\n"
        f"Scenario seed: {seed}.\n"
        f"Return JSON only."
    )


def verify_variant(v, a, b, op_sym):
    expected = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op_sym]
    if expected is None or expected < 0: return False, "non-positive gold"
    if op_sym == "/" and (a % b != 0): return False, "non-integer division"
    expected_int = int(expected)
    trace = v.get("answer_trace", "").strip()
    markers = re.findall(r"<<(.+?)=(-?\d+\.?\d*)>>", trace)
    if len(markers) != 1: return False, f"expected 1 marker, got {len(markers)}"
    expr, claimed = markers[0]
    expr = expr.strip()
    toks = re.findall(r"[+\-*/]", expr)
    if expr.startswith("-") and toks and toks[0] == "-": toks = toks[1:]
    if len(toks) != 1 or toks[0] != op_sym: return False, f"wrong operator in {expr}"
    try: v_eval = eval(expr)
    except Exception: return False, f"eval fail: {expr}"
    if abs(v_eval - expected_int) > 1e-3: return False, f"marker={v_eval}≠expected={expected_int}"
    m = re.search(r"####\s*(-?\d+\.?\d*)", trace.replace(",", ""))
    if not m: return False, "no ####"
    if abs(float(m.group(1)) - expected_int) > 1e-3: return False, "#### mismatch"
    return True, expected_int


def gen_one_template(client, t_id, a_range, seed):
    a_lo, a_hi, b_lo, b_hi = a_range
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_for(a_lo, a_hi, b_lo, b_hi, seed)},
            ],
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return None, f"api: {e}"
    a = int(obj.get("a", -1)); b = int(obj.get("b", -1))
    variants = obj.get("variants", [])
    if len(variants) != 4: return None, f"expected 4 variants got {len(variants)}"
    if b == 0: return None, "b=0"
    if a <= 0 or b <= 0: return None, "non-positive operand"
    if a <= b: return None, "a<=b (sub would be non-positive)"
    if a % b != 0: return None, "a not divisible by b (div would be non-integer)"
    out_rows = []
    seen_types = set()
    for v in variants:
        t = v.get("type", "")
        if t not in OPERATORS: continue
        if t in seen_types: continue
        sym = OPERATORS[t]
        ok, gold_or_msg = verify_variant(v, a, b, sym)
        if not ok: return None, f"{t}: {gold_or_msg}"
        seen_types.add(t)
        out_rows.append({
            "template_id": t_id,
            "type": t, "a": a, "b": b,
            "question_concat": v["question"].strip().replace("  ", " "),
            "answer": float(gold_or_msg),
            "answer_trace": v["answer_trace"].strip(),
            "source": f"llm-gen-vary_operator-{MODEL}",
            "scenario_seed": obj.get("scenario_seed", ""),
        })
    if len(out_rows) != 4: return None, f"only {len(out_rows)} valid types"
    return out_rows, "ok"


def main():
    client = OpenAI()
    rows = []
    failures = Counter()
    t0 = time.time()
    futures = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for t_id in range(N_TEMPLATES * 3):     # over-submit
            a_range = OPERAND_RANGES[t_id % len(OPERAND_RANGES)]
            seed = t_id * 17 + 31337
            futures[pool.submit(gen_one_template, client, t_id, a_range, seed)] = t_id
        n_kept = 0
        for fut in as_completed(futures):
            res, msg = fut.result()
            if res is None:
                failures[msg.split(":")[0]] += 1
                continue
            rows.extend(res)
            n_kept += 1
            if n_kept % 5 == 0:
                print(f"  kept {n_kept} templates  ({len(rows)} rows)  "
                      f"({time.time()-t0:.0f}s)")
            if n_kept >= N_TEMPLATES: break
    # Trim to N_TEMPLATES templates
    by_tid = {}
    for r in rows:
        by_tid.setdefault(r["template_id"], []).append(r)
    final_rows = []
    for tid in sorted(by_tid)[:N_TEMPLATES]:
        final_rows.extend(by_tid[tid])
    OUT.write_text(json.dumps(final_rows, indent=2))
    print(f"\nDONE: {len(final_rows)} rows from {len(set(r['template_id'] for r in final_rows))} templates")
    print(f"by type: {Counter(r['type'] for r in final_rows)}")
    print(f"failures: {failures.most_common(8)}")


if __name__ == "__main__":
    main()
