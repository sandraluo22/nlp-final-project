"""Generate GSM8K-compatible CF datasets by parsing the <<expr=result>>
markers in the GSM8K chain-of-thought.

For each GSM8K problem, the answer field looks like:
   "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
    She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
    #### 18"

Strategy:
  1. Extract all numbers from the question text and their positions.
  2. Extract all <<expr=result>> markers from the answer.
  3. Map numbers used in the markers to numbers in the question.
  4. For each CF variant, substitute new numbers for the question numbers
     (within configured ranges), then re-evaluate every <<expr=result>>
     marker step-by-step.
  5. Compute new final answer.
  6. Skip problems where:
     - Marker math doesn't actually evaluate to the stated result (parse failures).
     - Question numbers don't all show up in markers (we can't track substitution).
     - Resulting answer is negative or fractional (out of distribution).

Output: one JSON file per CF variant.
   gsm8k_cf_simple.json    — vary all numbers in the question with the same
                             multiplicative scale (1.5-3x)
   gsm8k_cf_balanced.json  — vary all numbers with random new values in
                             matched ranges
   gsm8k_cf_under99.json   — vary numbers such that all stay ≤ 99
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset


def parse_markers(answer_text: str):
    """Return list of {expr, result_stated, position} for each <<...>> marker."""
    out = []
    for m in re.finditer(r"<<(.+?)=(-?\d+\.?\d*)>>", answer_text):
        expr = m.group(1).strip()
        result_stated = float(m.group(2))
        out.append({"expr": expr, "result_stated": result_stated,
                    "span": (m.start(), m.end())})
    return out


def extract_numbers_in_order(text: str):
    """Return list of (start_pos, end_pos, value, raw_str) for each number."""
    out = []
    for m in re.finditer(r"\b-?\d+\.?\d*\b", text):
        try:
            v = float(m.group(0))
            out.append((m.start(), m.end(), v, m.group(0)))
        except ValueError:
            pass
    return out


def safe_eval_expr(expr: str):
    """Evaluate an arithmetic expression like '16 - 3 - 4' or '9 * 2'.
    Returns float or None.  Uses Python eval after a strict whitelist check.
    """
    if not re.match(r"^[\d+\-*/().\s]+$", expr): return None
    try:
        v = eval(expr)
        return float(v)
    except Exception:
        return None


def substitute_in_marker(expr: str, num_subs: dict, used_values: dict):
    """num_subs maps OLD raw number string -> NEW number value. Substitute
    each old number in the expression with its new value (where it appears).
    Track which values were used via used_values dict (set).

    Returns new expression string."""
    # Tokenize expression by numbers.
    parts = re.split(r"(-?\d+\.?\d*)", expr)
    out_parts = []
    for p in parts:
        m = re.match(r"^(-?\d+\.?\d*)$", p)
        if m:
            old = m.group(1)
            if old in num_subs:
                new_val = num_subs[old]
                used_values[old] = used_values.get(old, 0) + 1
                out_parts.append(str(new_val) if isinstance(new_val, int)
                                  else f"{new_val:g}")
            else:
                out_parts.append(p)   # keep literal
        else:
            out_parts.append(p)
    return "".join(out_parts)


def recompute_chain(answer_text: str, num_subs: dict):
    """Walk through the markers in order, evaluating each with substitutions.
    Returns:  (success, new_final_value, recomputed_markers, why_failed)
    """
    markers = parse_markers(answer_text)
    if not markers: return False, None, [], "no_markers"
    used = {}
    recomputed = []
    last_val = None
    for mk in markers:
        new_expr = substitute_in_marker(mk["expr"], num_subs, used)
        v = safe_eval_expr(new_expr)
        if v is None: return False, None, recomputed, f"eval_failed: {new_expr}"
        if v != int(v) and abs(v - round(v, 4)) > 1e-6:
            # Allow simple non-integer expressions but flag fractional results.
            return False, None, recomputed, f"fractional_result: {new_expr}={v}"
        recomputed.append({"orig_expr": mk["expr"], "new_expr": new_expr, "value": v})
        last_val = v
    # Confirm the final marker result is the #### answer.  Also confirm we
    # used every question-number at least once (so all substitutions matter).
    return True, last_val, recomputed, None


def question_numbers_to_sub(q_nums, mode: str, rng):
    """Build substitution map: old raw string -> new value.

    Modes:
        simple: multiply by random scale in [1.5, 3.0]
        balanced: replace each number with a random number in [10, 99] (or
                  matching magnitude bucket)
        under99: replace each number with a random number ≤ 99, similar mag.
    """
    subs = {}
    for _, _, v, raw in q_nums:
        if raw in subs: continue
        if mode == "simple":
            scale = rng.uniform(1.5, 3.0)
            new = int(v * scale) if v == int(v) else round(v * scale, 2)
            if new == int(new): new = int(new)
            subs[raw] = new
        elif mode == "balanced":
            # Map magnitude bucket to a new value in the same bucket
            if v < 10:        new = int(rng.integers(2, 10))
            elif v < 100:     new = int(rng.integers(10, 100))
            elif v < 1000:    new = int(rng.integers(100, 1000))
            else:             new = int(rng.integers(1000, 10000))
            subs[raw] = new
        elif mode == "under99":
            new = int(rng.integers(2, 100))
            subs[raw] = new
    return subs


def build_cf_dataset(mode: str, n_keep: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    ds = load_dataset("gsm8k", "main")["test"]
    out_rows = []
    n_attempted = 0
    n_kept = 0
    for ex in ds:
        if n_kept >= n_keep: break
        q = ex["question"].strip().replace("  ", " ")
        ans_text = ex["answer"]
        # Verify the natural problem first.
        m_final = re.search(r"####\s*(-?\d+\.?\d*)", ans_text.replace(",", ""))
        if m_final is None: continue
        orig_gold = float(m_final.group(1))
        markers = parse_markers(ans_text)
        if not markers: continue
        # Marker check: do markers actually evaluate?
        ok = True
        for mk in markers:
            v = safe_eval_expr(mk["expr"])
            if v is None or abs(v - mk["result_stated"]) > 1e-3:
                ok = False; break
        if not ok: continue
        # Final marker result should equal orig_gold.
        if abs(markers[-1]["result_stated"] - orig_gold) > 1e-3: continue
        n_attempted += 1
        q_nums = extract_numbers_in_order(q)
        if not q_nums: continue
        # Try up to 5 CF substitutions per problem.
        for trial in range(5):
            subs = question_numbers_to_sub(q_nums, mode, rng)
            ok2, new_gold, recomp, why = recompute_chain(ans_text, subs)
            if not ok2: continue
            # Skip if recompute didn't use all question numbers (some were never
            # referenced in markers).
            used_in_markers = set()
            for r in recomp:
                used_in_markers.update(re.findall(r"-?\d+\.?\d*", r["new_expr"]))
            # Substitute into the question.
            new_q = q
            for (start, end, v, raw) in sorted(q_nums, key=lambda x: -x[0]):
                if raw in subs:
                    new_val = subs[raw]
                    new_q = new_q[:start] + str(new_val) + new_q[end:]
            out_rows.append({
                "src_idx": int(n_attempted - 1),
                "orig_question": q, "orig_gold": orig_gold,
                "cf_question": new_q,
                "cf_question_concat": new_q,
                "cf_gold": new_gold,
                "cf_answer": new_gold,
                "answer": new_gold,    # alias for compat
                "subs": {k: (v if isinstance(v, (int, float)) else float(v))
                         for k, v in subs.items()},
                "type": "multi-step",
                "n_steps": len(recomp),
            })
            n_kept += 1
            break
    return out_rows


def main():
    out_dir = Path(__file__).resolve().parent
    for mode, name in [("balanced", "gsm8k_cf_balanced"),
                        ("under99",  "gsm8k_cf_under99"),
                        ("simple",   "gsm8k_cf_simple")]:
        rows = build_cf_dataset(mode, n_keep=400)
        out_path = out_dir / f"{name}.json"
        out_path.write_text(json.dumps(rows, indent=2))
        # Stats
        if rows:
            golds = [r["cf_gold"] for r in rows]
            n_steps = [r["n_steps"] for r in rows]
            print(f"{name}: N={len(rows)}  gold range [{min(golds):.0f},{max(golds):.0f}]  "
                  f"median n_steps={int(np.median(n_steps))}  "
                  f"saved {out_path}")
        else:
            print(f"{name}: NO rows generated")


if __name__ == "__main__":
    main()
