"""Iterative-balanced counterfactual SVAMP generator.

For each (operator, input_bucket, output_bucket) cell we try to fill it with
roughly TARGET_PER_CELL counterfactual instances. Each original SVAMP problem
can produce multiple variants — one per cell it can fit. Adaptive: cells that
are physically infeasible (e.g. Subtraction with both inputs <10 producing
output 1000+) are simply left empty.

By construction, within each operator the joint distribution over
(input_bucket × output_bucket) is uniform up to feasibility, so both input
magnitudes and output magnitudes are roughly uniform per operator and across
operators. Cramér's V (operator ↔ bucket) should drop substantially compared
with the previous cf_magmatched dataset.

Output: cf-datasets/cf_balanced.json with fields per row:
    idx, src_idx, type, orig_equation, orig_body, orig_question, orig_answer,
    cf_subs, cf_body, cf_question, cf_question_concat,
    input_bucket, output_bucket, cf_answer
"""

import json
import random
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


REPO = Path(__file__).resolve().parent.parent
OUT = REPO.parent / "cf-datasets" / "cf_balanced.json"

BUCKETS = [(1, 9), (10, 99), (100, 999), (1000, 9999)]
BUCKET_NAMES = ["<10", "10-99", "100-999", "1000+"]
SEED = 17
TARGET_PER_CELL = 25           # target CF instances per (op, in_b, out_b) cell
MAX_VARIANTS_PER_PROBLEM = 1   # one CF variant per source problem (no dupes)
MAX_REJECTION_TRIES = 300      # samples per (problem, cell) attempt
MAX_ROUNDS = 1                 # one pass; each problem is placed at most once


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
    nums_in_text = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\d+", text)]
    used: set[int] = set()
    positions: list[tuple[int, int]] = []
    for orig in originals:
        target = int(orig)
        for i, (s, e, t) in enumerate(nums_in_text):
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


def all_inputs_in_bucket(subs: list[int], bucket_idx: int) -> bool:
    lo, hi = BUCKETS[bucket_idx]
    return all(lo <= int(s) <= hi for s in subs)


def output_in_bucket(answer: float, bucket_idx: int) -> bool:
    if answer is None:
        return False
    if answer < 0:
        return False
    if abs(answer - round(answer)) > 1e-9:
        return False
    a = int(round(answer))
    lo, hi = BUCKETS[bucket_idx]
    return lo <= a <= hi


def try_one_cell(
    rng: random.Random,
    equation: str,
    n_numerals: int,
    in_bucket: int,
    out_bucket: int,
) -> tuple[list[int], int] | None:
    """Try to draw n_numerals integers all inside `in_bucket` such that the
    equation evaluates inside `out_bucket`. Returns (subs, answer) or None."""
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

    problems: list[dict] = []
    for idx, ex in enumerate(full):
        op_type = ex["Type"].replace("Common-Divison", "Common-Division")
        equation = ex["Equation"]
        body = ex["Body"]
        question = ex["Question"]
        full_text = body + " " + question
        originals = parse_equation_numerals(equation)
        # Skip problems whose original numerals can't be located in the text;
        # they would also fail substitution.
        if find_numeral_positions(full_text, originals) is None:
            continue
        problems.append(
            {
                "idx": idx,
                "type": op_type,
                "equation": equation,
                "body": body,
                "question": question,
                "n_numerals": len(originals),
                "originals": originals,
            }
        )
    print(f"  {len(problems)} usable problems  (skipped {len(full) - len(problems)} unmatchable)")

    # Cell counts and per-problem variant counts.
    cell_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    variants_per_problem: dict[int, int] = defaultdict(int)
    # Cells we've tried and failed for this (operator, in_b, out_b) globally —
    # mark as infeasible and stop trying.
    infeasible_cells: set[tuple[str, int, int]] = set()

    emitted: list[dict] = []
    operators = sorted({p["type"] for p in problems})
    print(f"  operators: {operators}")
    print(f"  target per cell: {TARGET_PER_CELL}")

    for round_idx in range(MAX_ROUNDS):
        progress = 0
        rng.shuffle(problems)
        for p in problems:
            if variants_per_problem[p["idx"]] >= MAX_VARIANTS_PER_PROBLEM:
                continue
            op = p["type"]
            # Cells under target for this operator, not yet flagged infeasible.
            under = [
                (in_b, out_b)
                for in_b in range(len(BUCKETS))
                for out_b in range(len(BUCKETS))
                if cell_counts[(op, in_b, out_b)] < TARGET_PER_CELL
                and (op, in_b, out_b) not in infeasible_cells
            ]
            if not under:
                continue
            # Prioritize the least-filled cells so placements spread evenly
            # rather than piling into easy cells.
            under.sort(
                key=lambda c: (cell_counts[(op, c[0], c[1])], rng.random())
            )
            placed = False
            for (in_b, out_b) in under:
                got = try_one_cell(rng, p["equation"], p["n_numerals"], in_b, out_b)
                if got is None:
                    continue
                subs, ans = got
                # Substitute back into body + question.
                pos_body = find_numeral_positions(p["body"], p["originals"])
                pos_q = find_numeral_positions(p["question"], p["originals"])
                if pos_body is None and pos_q is None:
                    continue
                cf_body = (
                    substitute_numerals(p["body"], pos_body, subs)
                    if pos_body is not None
                    else p["body"]
                )
                cf_question = (
                    substitute_numerals(p["question"], pos_q, subs)
                    if pos_q is not None
                    else p["question"]
                )
                cf_concat = (
                    cf_body
                    + (" " if cf_body and cf_question else "")
                    + cf_question
                )
                emitted.append(
                    {
                        "idx": len(emitted),
                        "src_idx": p["idx"],
                        "type": op,
                        "orig_equation": p["equation"],
                        "orig_body": p["body"],
                        "orig_question": p["question"],
                        "cf_subs": subs,
                        "cf_body": cf_body,
                        "cf_question": cf_question,
                        "cf_question_concat": cf_concat,
                        "input_bucket_idx": in_b,
                        "input_bucket": BUCKET_NAMES[in_b],
                        "output_bucket_idx": out_b,
                        "output_bucket": BUCKET_NAMES[out_b],
                        "cf_answer": ans,
                    }
                )
                cell_counts[(op, in_b, out_b)] += 1
                variants_per_problem[p["idx"]] += 1
                placed = True
                progress += 1
                break
            if not placed:
                # Flag the cells we tried but couldn't fill *for this problem*.
                # (We don't flag globally — a different problem's equation might
                # fill the cell. Global infeasibility is detected later.)
                pass
        # Detect globally-infeasible cells: cells still empty after a round
        # where we attempted to fill them and all attempts failed.
        for op in operators:
            for in_b in range(len(BUCKETS)):
                for out_b in range(len(BUCKETS)):
                    cell = (op, in_b, out_b)
                    if cell in infeasible_cells:
                        continue
                    if cell_counts[cell] >= TARGET_PER_CELL:
                        continue
                    # Heuristic infeasibility: after 2 rounds with 0 placements
                    # in this cell, mark infeasible.
                    if round_idx >= 1 and cell_counts[cell] == 0:
                        infeasible_cells.add(cell)
        n_filled = sum(1 for v in cell_counts.values() if v >= TARGET_PER_CELL)
        n_total = len(operators) * len(BUCKETS) * len(BUCKETS)
        n_inf = len(infeasible_cells)
        print(
            f"  round {round_idx+1}: emitted_total={len(emitted)}  "
            f"filled={n_filled}/{n_total}  infeasible={n_inf}  "
            f"placed_this_round={progress}"
        )
        if progress == 0:
            break

    # Report
    print(f"\nemitted {len(emitted)} CF instances "
          f"from {len(set(r['src_idx'] for r in emitted))} unique source problems")
    print(f"variants per source problem: "
          f"min={min(variants_per_problem.values()) if variants_per_problem else 0}  "
          f"max={max(variants_per_problem.values()) if variants_per_problem else 0}  "
          f"mean={sum(variants_per_problem.values())/max(len(variants_per_problem),1):.1f}")

    print("\ncell counts by (operator, in_bucket, out_bucket):")
    for op in operators:
        print(f"\n  {op}")
        print(f"    {'in \\ out':<10s}" + "".join(f" {b:>10s}" for b in BUCKET_NAMES))
        for in_idx, in_name in enumerate(BUCKET_NAMES):
            row = [f"    {in_name:<10s}"]
            for out_idx in range(len(BUCKETS)):
                c = cell_counts[(op, in_idx, out_idx)]
                marker = "" if (op, in_idx, out_idx) not in infeasible_cells else "X"
                row.append(f"{c:>9d}{marker}")
            print(" ".join(row))

    OUT.write_text(json.dumps(emitted, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
