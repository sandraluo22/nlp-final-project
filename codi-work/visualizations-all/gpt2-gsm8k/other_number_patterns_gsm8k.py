"""For 'other_number' wrong emits (the 88% of pre-rescue emits that are
neither markers nor zero nor unparsed), categorize what they actually are.

Categories tested for each wrong-emit value v vs gold g and question numbers
ops = [...]:
  - 'operand'       : v equals any number that appears in the question
  - 'sum_of_ops'    : v == sum(ops)
  - 'prod_of_ops'   : v == product(ops)
  - 'diff_of_ops'   : v == ops[0] − sum(ops[1:])
  - 'op_pair'       : v == any op_i ± op_j or op_i × op_j or op_i / op_j  (for any i ≠ j)
  - 'gold_off_by_1' : |v − g| ∈ {1}
  - 'gold_off_by_10': |v − g| ∈ {10, 100, 1000} or by-factor 10
  - 'small_const'   : v ∈ {0, 1, 2, 3, 5, 10, 100, 1000}
  - 'same_magnitude': same number of integer digits as g
  - 'other'         : none of the above

Output: other_number_patterns_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, re
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
FD_JSON = REPO / "experiments" / "computation_probes" / "force_decode_per_step_gsm8k.json"
OUT_JSON = PD / "other_number_patterns_gsm8k.json"
OUT_PDF = PD / "other_number_patterns_gsm8k.pdf"


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def emit_final(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def extract_question_numbers(q):
    """Find all numeric values mentioned in the question text."""
    s = q.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    out = []
    for n in nums:
        try: out.append(float(n))
        except: pass
    return out


def safe_div(a, b):
    if b == 0: return None
    return a / b


def n_digits(x):
    if x is None: return -1
    return len(str(int(abs(x)))) if x == int(x) else len(str(int(abs(x))))


def categorize(v, gold, ops, markers):
    """Return a SET of categories matching this wrong-emit value."""
    cats = set()
    if v is None:
        return {"unparsed"}
    # Markers (any intermediate marker result)
    for m in markers:
        if abs(v - m[3]) < 1e-3:
            cats.add("matches_marker"); break
    # Direct operand match
    if any(abs(v - x) < 1e-3 for x in ops):
        cats.add("matches_operand")
    # Full-chain alternative ops
    if ops and abs(v - sum(ops)) < 1e-3:
        cats.add("sum_of_ops")
    if ops:
        prod = 1.0
        for x in ops: prod *= x
        if abs(v - prod) < 1e-3:
            cats.add("prod_of_ops")
    if ops and len(ops) >= 2 and abs(v - (ops[0] - sum(ops[1:]))) < 1e-3:
        cats.add("diff_of_ops")
    # Any pair-op
    for i in range(len(ops)):
        for j in range(len(ops)):
            if i == j: continue
            a, b = ops[i], ops[j]
            for r in [a + b, a - b, a * b, safe_div(a, b)]:
                if r is None: continue
                if abs(v - r) < 1e-3:
                    cats.add("matches_op_pair"); break
            if "matches_op_pair" in cats: break
        if "matches_op_pair" in cats: break
    # Off-by-small
    if abs(v - gold) <= 1:
        cats.add("near_gold")  # ±1
    if 0 < abs(v - gold) <= 5:
        cats.add("near_gold_within_5")
    # Off by factor of 10
    if gold != 0 and v != 0:
        ratio = v / gold
        for f in [10, 0.1, 100, 0.01, 1000, 0.001]:
            if abs(ratio - f) < 1e-3:
                cats.add("off_by_factor10"); break
    # Magnitude match
    if abs(v) > 0 and abs(gold) > 0:
        if int(np.log10(abs(v))) == int(np.log10(abs(gold))):
            cats.add("same_magnitude")
    # Small constant
    if v in [0, 1, 2, 3, 4, 5, 10, 100, 1000]:
        cats.add("small_const")
    # If nothing matched
    if not cats:
        cats.add("truly_other")
    return cats


def main():
    fd = json.load(open(FD_JSON))
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]

    rows = fd["rows"]
    n_steps = len(rows[0]["step_emits"])

    # Collect all "other_number" wrong emits from CORRECT-AT-STEP-6 problems
    # (those that needed rescue), so we focus on the pre-rescue cohort.
    wrong_emits = []   # list of dicts: {idx, step, v, gold, ops, markers, q}
    for r in rows:
        idx = r["idx"]
        ex = ds[idx]
        gold_match = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if gold_match is None: continue
        gold = float(gold_match.group(1))
        markers = [(float(a), op, float(b), float(c))
                   for a, op, b, c in parse_markers(ex["answer"])]
        emit_vals = [emit_final(e) for e in r["step_emits"]]
        step_correct = [v is not None and abs(v - gold) < 1e-3 for v in emit_vals]
        if not step_correct[-1]: continue   # focus on correct-at-step-6 only
        first_correct = next((k for k, c in enumerate(step_correct) if c), -1)
        if first_correct < 1: continue   # skip 'already-correct' problems
        ops = extract_question_numbers(ex["question"])
        for k in range(first_correct):
            v = emit_vals[k]
            if v is None or v == 0: continue
            # check it's "other_number" — i.e., NOT a marker match
            is_marker = markers and any(abs(v - m[3]) < 1e-3 for m in markers)
            if is_marker: continue
            wrong_emits.append({
                "idx": idx, "step": k + 1, "v": v, "gold": gold,
                "ops": ops, "markers": markers,
                "first_correct_step": first_correct + 1,
                "q": ex["question"].strip().replace("  ", " "),
                "emit_text": r["step_emits"][k],
            })

    print(f"  N other-number wrong emits to categorize: {len(wrong_emits)}")

    # Categorize each
    counts = Counter()
    by_first = defaultdict(Counter)
    examples_per_cat = defaultdict(list)
    for w in wrong_emits:
        cats = categorize(w["v"], w["gold"], w["ops"], w["markers"])
        for c in cats: counts[c] += 1
        for c in cats: by_first[w["first_correct_step"]][c] += 1
        # collect samples
        primary_cat = "truly_other"
        for c in ["matches_operand", "matches_op_pair", "sum_of_ops",
                  "prod_of_ops", "diff_of_ops", "small_const",
                  "off_by_factor10", "near_gold", "near_gold_within_5",
                  "same_magnitude"]:
            if c in cats:
                primary_cat = c; break
        if len(examples_per_cat[primary_cat]) < 5:
            examples_per_cat[primary_cat].append({
                "idx": w["idx"], "step": w["step"], "v": w["v"], "gold": w["gold"],
                "ops_in_q": w["ops"][:6],
                "emit_text": w["emit_text"][:80],
            })

    total = len(wrong_emits)
    print(f"\nCategory counts (multi-label; each emit can hit multiple):")
    for c, n in counts.most_common():
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {c:<22} {n:>4}  ({pct:5.1f}%)  {bar}")

    out = {
        "n_other_number_wrong_emits": total,
        "category_counts": dict(counts),
        "by_first_correct_step": {k: dict(v) for k, v in by_first.items()},
        "examples_per_category": {c: l for c, l in examples_per_cat.items()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # Plots
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: bar of category counts
        fig, ax = plt.subplots(figsize=(13, 6.5))
        labels = [c for c, _ in counts.most_common()]
        vals = [counts[c] / total * 100 for c in labels]
        bars = ax.barh(range(len(labels)), vals, color="#4c72b0",
                       edgecolor="black")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for i, v in enumerate(vals):
            ax.text(v + 0.5, i, f"{v:.1f}%  (n={counts[labels[i]]})",
                    va="center", fontsize=9)
        ax.set_xlabel("% of N other-number wrong emits matching this pattern")
        ax.set_title(f"Patterns in 'other_number' wrong emits (N={total})\n"
                     "(multi-label: each emit can satisfy multiple)",
                     fontsize=11, fontweight="bold")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(vals) * 1.18)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: text summary + interpretation
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        ax.axis("off")
        body = (f"'Other-number' wrong-emit analysis  (N = {total} entries)\n\n"
                "Categories (each emit can satisfy multiple):\n\n")
        for c in labels:
            n = counts[c]; pct = n / total * 100
            body += f"  {c:<26} {n:>4}  ({pct:5.1f}%)\n"
        body += (f"\nInterpretation:\n"
                 f"  - 'matches_operand': model emits a number that LITERALLY APPEARS in Q\n"
                 f"  - 'matches_op_pair': model emits a sum/product/diff of TWO operands\n"
                 f"  - 'sum_of_ops/prod_of_ops/diff_of_ops': full-chain alternative answers\n"
                 f"  - 'near_gold' / 'near_gold_within_5': off-by-1 or off-by-≤5\n"
                 f"  - 'same_magnitude': same number-of-digits as gold (right magnitude, wrong value)\n"
                 f"  - 'off_by_factor10': order-of-magnitude error\n"
                 f"  - 'truly_other': none of the above\n")
        ax.text(0.04, 0.95, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Other-number patterns — interpretation",
                     fontsize=12, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: sample emits per category
        for cat in ["matches_operand", "matches_op_pair", "sum_of_ops", "prod_of_ops",
                    "diff_of_ops", "small_const", "off_by_factor10", "near_gold",
                    "same_magnitude", "truly_other"]:
            exs = examples_per_cat.get(cat, [])
            if not exs: continue
            fig, ax = plt.subplots(figsize=(11.5, 6))
            ax.axis("off")
            body = f"Category: {cat}\n\n"
            for e in exs:
                body += (f"  idx={e['idx']}  step={e['step']}  emit={e['v']:g}  gold={e['gold']:g}\n"
                         f"    operands in Q: {e['ops_in_q']}\n"
                         f"    emit text: {e['emit_text']!r}\n\n")
            ax.text(0.02, 0.97, body, va="top", ha="left",
                    family="monospace", fontsize=9)
            ax.set_title(f"Samples — '{cat}'",
                         fontsize=12, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
