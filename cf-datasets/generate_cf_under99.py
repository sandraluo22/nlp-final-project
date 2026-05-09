"""Counterfactual SVAMP generator restricted to numerals < 100.

Like cf_balanced, but with all numerals (and answers) constrained to [1, 99].
This pulls the dataset into a regime the small student handles much better,
so 'student-correct' filtering doesn't drop most of the dataset.

Buckets are 2-way: <10 and 10-99. We aim to fill (operator × input_bucket ×
output_bucket) cells uniformly.

Per-operator reachability under the [1, 99] constraint:
  Add: (<10,<10), (<10,10-99), (10-99,10-99) — output 10-99 needs sum<100
  Sub: (<10,<10), (10-99,<10), (10-99,10-99)
  Mul: (<10,<10), (<10,10-99), (10-99,<10) — output <100 forces small inputs
  Div: (<10,<10), (10-99,<10), (10-99,10-99) — output 10-99 hard with int div

So at most 3 cells per operator are reachable (12 total of 16 logical cells).

Output: cf-datasets/cf_under99.json with one CF instance per source SVAMP problem
(no duplicates).
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


REPO = Path(__file__).resolve().parent.parent
OUT = REPO.parent / "cf-datasets" / "cf_under99.json"

# Two buckets only.
BUCKETS = [(1, 9), (10, 99)]
BUCKET_NAMES = ["<10", "10-99"]
SEED = 23
TARGET_PER_CELL = 60         # higher target since we have fewer cells
MAX_VARIANTS_PER_PROBLEM = 1
MAX_REJECTION_TRIES = 400
MAX_ROUNDS = 1


def parse_equation_numerals(equation: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+\.?\d*", equation)]


def evaluate_with_subs(equation: str, subs: list[float]) -> float:
    parts = re.split(r"(\d+\.?\d*)", equation)
    out: list[str] = []
    i = 0
    for p in parts:
        if re.fullmatch(r"\d+\.?\d*", p):
            out.append(repr(float(subs[i])))
            i += 1
        else:
            out.append(p)
    return eval("".join(out), {"__builtins__": {}}, {})


def find_numeral_positions(text: str, originals: list[float]) -> list[tuple[int, int]] | None:
    nums = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\d+", text)]
    used: set[int] = set()
    positions = []
    for orig in originals:
        target = int(orig)
        for i, (s, e, t) in enumerate(nums):
            if i in used:
                continue
            if int(t) == target:
                positions.append((s, e))
                used.add(i)
                break
        else:
            return None
    return positions


def substitute_numerals(text: str, positions: list[tuple[int, int]], subs: list[int]) -> str:
    paired = sorted(zip(positions, subs), key=lambda x: x[0][0], reverse=True)
    out = text
    for (s, e), v in paired:
        out = out[:s] + str(v) + out[e:]
    return out


def output_in_bucket(answer: float, bucket_idx: int) -> bool:
    if answer is None or answer < 1:
        return False
    if abs(answer - round(answer)) > 1e-9:
        return False
    a = int(round(answer))
    lo, hi = BUCKETS[bucket_idx]
    return lo <= a <= hi


def all_in_bucket(subs: list[int], bucket_idx: int) -> bool:
    lo, hi = BUCKETS[bucket_idx]
    return all(lo <= s <= hi for s in subs)


def try_one_cell(rng, equation: str, n_numerals: int,
                 in_bucket: int, out_bucket: int):
    in_lo, in_hi = BUCKETS[in_bucket]
    for _ in range(MAX_REJECTION_TRIES):
        subs = [rng.randint(in_lo, in_hi) for _ in range(n_numerals)]
        try:
            ans = evaluate_with_subs(equation, [float(x) for x in subs])
        except ZeroDivisionError:
            continue
        if output_in_bucket(ans, out_bucket):
            return subs, int(round(ans))
    return None


def main():
    rng = random.Random(SEED)
    print("loading SVAMP")
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])

    problems = []
    for idx, ex in enumerate(full):
        op = ex["Type"].replace("Common-Divison", "Common-Division")
        eq = ex["Equation"]
        body = ex["Body"]
        question = ex["Question"]
        full_text = body + " " + question
        originals = parse_equation_numerals(eq)
        if find_numeral_positions(full_text, originals) is None:
            continue
        problems.append({
            "idx": idx, "type": op, "equation": eq,
            "body": body, "question": question,
            "n_numerals": len(originals), "originals": originals,
        })
    print(f"  {len(problems)} usable problems  (skipped {len(full) - len(problems)})")

    cell_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    variants_per_problem: dict[int, int] = defaultdict(int)
    infeasible_cells: set[tuple[str, int, int]] = set()
    emitted: list[dict] = []
    operators = sorted({p["type"] for p in problems})

    for round_idx in range(MAX_ROUNDS):
        progress = 0
        rng.shuffle(problems)
        for p in problems:
            if variants_per_problem[p["idx"]] >= MAX_VARIANTS_PER_PROBLEM:
                continue
            op = p["type"]
            under = [
                (in_b, out_b)
                for in_b in range(len(BUCKETS))
                for out_b in range(len(BUCKETS))
                if cell_counts[(op, in_b, out_b)] < TARGET_PER_CELL
                and (op, in_b, out_b) not in infeasible_cells
            ]
            if not under:
                continue
            # Prefer least-filled cells.
            under.sort(key=lambda c: (cell_counts[(op, c[0], c[1])], rng.random()))
            placed = False
            for (in_b, out_b) in under:
                got = try_one_cell(rng, p["equation"], p["n_numerals"], in_b, out_b)
                if got is None:
                    continue
                subs, ans = got
                pos_body = find_numeral_positions(p["body"], p["originals"])
                pos_q = find_numeral_positions(p["question"], p["originals"])
                if pos_body is None and pos_q is None:
                    continue
                cf_body = (substitute_numerals(p["body"], pos_body, subs)
                           if pos_body is not None else p["body"])
                cf_question = (substitute_numerals(p["question"], pos_q, subs)
                               if pos_q is not None else p["question"])
                cf_concat = cf_body + (" " if cf_body and cf_question else "") + cf_question
                emitted.append({
                    "idx": len(emitted), "src_idx": p["idx"], "type": op,
                    "orig_equation": p["equation"], "orig_body": p["body"],
                    "orig_question": p["question"],
                    "cf_subs": subs, "cf_body": cf_body, "cf_question": cf_question,
                    "cf_question_concat": cf_concat,
                    "input_bucket_idx": in_b, "input_bucket": BUCKET_NAMES[in_b],
                    "output_bucket_idx": out_b, "output_bucket": BUCKET_NAMES[out_b],
                    "cf_answer": ans,
                })
                cell_counts[(op, in_b, out_b)] += 1
                variants_per_problem[p["idx"]] += 1
                placed = True
                progress += 1
                break
        # Mark cells empty after this round as infeasible (heuristic).
        for op in operators:
            for in_b in range(len(BUCKETS)):
                for out_b in range(len(BUCKETS)):
                    cell = (op, in_b, out_b)
                    if cell_counts[cell] == 0 and cell not in infeasible_cells:
                        infeasible_cells.add(cell)
        n_total = len(operators) * len(BUCKETS) ** 2
        n_filled = sum(1 for v in cell_counts.values() if v >= TARGET_PER_CELL)
        print(f"  round {round_idx+1}: emitted={len(emitted)}  "
              f"filled={n_filled}/{n_total}  infeasible={len(infeasible_cells)}")
        if progress == 0:
            break

    print(f"\nemitted {len(emitted)} from {len(set(r['src_idx'] for r in emitted))} unique source problems")
    print("\ncell counts (operator × in_bucket × out_bucket):")
    for op in operators:
        print(f"  {op}")
        print(f"    in \\ out      <10    10-99")
        for in_idx, in_name in enumerate(BUCKET_NAMES):
            row = [f"    {in_name:<10s}"]
            for out_idx in range(len(BUCKETS)):
                c = cell_counts[(op, in_idx, out_idx)]
                marker = "" if (op, in_idx, out_idx) not in infeasible_cells else "X"
                row.append(f"{c:>6d}{marker}")
            print(" ".join(row))

    OUT.write_text(json.dumps(emitted, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
