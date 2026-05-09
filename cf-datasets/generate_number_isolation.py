"""Two synthetic CF datasets for isolating number vs operator representations.

A) vary_numerals.json
   Fixed: scenario, wording, operator (Subtraction).
   Varies: the two numerals (a, b) across the prompt.
   PCA on activations → directions that encode numerical inputs (and the
   subtraction answer a-b, which varies with the inputs).

B) vary_operator.json
   Fixed: the two numerals (a, b) = same across every prompt.
   Varies: operator and scenario phrasing.
   PCA on activations → directions that encode operator/wording structure.
   Number-encoding directions should NOT show up here since the numbers don't
   vary, so they contribute zero variance.

Comparing the principal subspaces of A and B tells us whether number and
operator are linearly disentangled in the residual stream.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
OUT_A = OUT_DIR / "vary_numerals.json"
OUT_B = OUT_DIR / "vary_operator.json"


# ---------------------------------------------------------------------------
# Dataset A: vary_numerals
# ---------------------------------------------------------------------------

# A SVAMP-shaped Subtraction scenario. The two numerals a and b appear once
# each in the body; the answer is a - b. Keep the body identical so anything
# that varies in the residual stream is downstream of the numeral encoding.
A_TEMPLATE = (
    "John has {a} apples in his basket. He gives {b} apples to his sister. "
    "How many apples does John have left?"
)
A_OPERATOR = "Subtraction"


def make_dataset_a(n: int = 80, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    used = set()
    while len(rows) < n:
        a = rng.randint(5, 200)
        b = rng.randint(1, a - 1)        # ensures a-b > 0 and a > b
        if (a, b) in used:
            continue
        used.add((a, b))
        rows.append({
            "idx": len(rows),
            "a": a,
            "b": b,
            "type": A_OPERATOR,
            "answer": a - b,
            "question_concat": A_TEMPLATE.format(a=a, b=b),
        })
    return rows


# ---------------------------------------------------------------------------
# Dataset B: vary_operator
# ---------------------------------------------------------------------------

# Fix: a, b. Vary: operator + scenario. Each (operator, scenario) combination
# uses the SAME two numerals.
B_A = 12
B_B = 4   # picked so a*b=48, a+b=16, a-b=8, a/b=3 are all clean integers

def b_templates(a: int, b: int):
    """All entries answer to a single integer using (a, b). Each operator gets
    multiple scenario variants for diversity within operator."""
    add_ans = a + b
    sub_ans = a - b
    mul_ans = a * b
    div_ans = a // b
    rows = []

    # Addition (a + b)
    for body in [
        f"Maya has {a} stickers. Her friend gives her {b} more stickers. How many stickers does Maya have now?",
        f"There are {a} cars in the parking lot. {b} more cars arrive. How many cars are now in the lot?",
        f"A baker made {a} cookies in the morning and {b} cookies in the afternoon. How many cookies in total?",
        f"Tom collected {a} seashells on Monday and {b} seashells on Tuesday. How many seashells does he have?",
        f"A library had {a} books on the shelf. {b} new books were added. How many books are on the shelf now?",
        f"A garden has {a} red roses and {b} white roses. How many roses are in the garden total?",
    ]:
        rows.append({"type": "Addition", "answer": add_ans, "question_concat": body})

    # Subtraction (a - b)
    for body in [
        f"Maya has {a} stickers. She gives {b} stickers to her brother. How many stickers does Maya have left?",
        f"There are {a} birds on a branch. {b} birds fly away. How many birds are still on the branch?",
        f"A baker made {a} cookies. {b} cookies were eaten. How many cookies are left?",
        f"Tom had {a} marbles. He lost {b} marbles. How many marbles does he have now?",
        f"A library had {a} books on the shelf. {b} books were borrowed. How many books are on the shelf now?",
        f"There were {a} candies in a jar. {b} candies were taken out. How many candies remain?",
    ]:
        rows.append({"type": "Subtraction", "answer": sub_ans, "question_concat": body})

    # Multiplication (a * b)
    for body in [
        f"Maya has {a} sticker books. Each book contains {b} stickers. How many stickers does Maya have in total?",
        f"There are {a} parking lots. Each lot holds {b} cars. How many cars can be parked in total?",
        f"A baker made {a} batches of cookies. Each batch has {b} cookies. How many cookies were made in total?",
        f"Tom buys {a} bags of marbles. Each bag has {b} marbles. How many marbles does Tom have?",
        f"A library has {a} shelves. Each shelf holds {b} books. How many books does the library have in total?",
        f"There are {a} jars of candy. Each jar contains {b} candies. How many candies are there altogether?",
    ]:
        rows.append({"type": "Multiplication", "answer": mul_ans, "question_concat": body})

    # Common-Division (a / b — choose so the answer is an integer)
    for body in [
        f"Maya has {a} stickers and wants to split them equally among {b} friends. How many stickers does each friend get?",
        f"There are {a} cars to be parked equally in {b} parking lots. How many cars per lot?",
        f"A baker made {a} cookies and packed them equally into {b} boxes. How many cookies in each box?",
        f"Tom has {a} marbles to share equally among {b} bags. How many marbles per bag?",
        f"A library has {a} books to put equally on {b} shelves. How many books per shelf?",
        f"There are {a} candies to be split equally into {b} bowls. How many candies in each bowl?",
    ]:
        rows.append({"type": "Common-Division", "answer": div_ans, "question_concat": body})

    return rows


def make_dataset_b():
    rows = b_templates(B_A, B_B)
    for i, r in enumerate(rows):
        r["idx"] = i
        r["a"] = B_A
        r["b"] = B_B
    return rows


OUT_VARY_A = OUT_DIR / "vary_a.json"
OUT_VARY_B = OUT_DIR / "vary_b.json"


def make_vary_a_only(b_fixed: int = 4, n: int = 80, seed: int = 2):
    """Subtraction, fixed b, varied a. Used to isolate the encoding of `a`."""
    rng = random.Random(seed)
    rows, used = [], set()
    a_min = b_fixed + 1
    while len(rows) < n:
        a = rng.randint(a_min, 200)
        if a in used: continue
        used.add(a)
        rows.append({
            "idx": len(rows), "a": a, "b": b_fixed,
            "type": A_OPERATOR, "answer": a - b_fixed,
            "question_concat": A_TEMPLATE.format(a=a, b=b_fixed),
        })
    return rows


def make_vary_b_only(a_fixed: int = 200, n: int = 80, seed: int = 3):
    """Subtraction, fixed a, varied b. Used to isolate the encoding of `b`."""
    rng = random.Random(seed)
    rows, used = [], set()
    while len(rows) < n:
        b = rng.randint(1, a_fixed - 1)
        if b in used: continue
        used.add(b)
        rows.append({
            "idx": len(rows), "a": a_fixed, "b": b,
            "type": A_OPERATOR, "answer": a_fixed - b,
            "question_concat": A_TEMPLATE.format(a=a_fixed, b=b),
        })
    return rows


def main():
    rows_a = make_dataset_a(n=80)
    rows_b = make_dataset_b()
    rows_va = make_vary_a_only(b_fixed=4, n=80)
    rows_vb = make_vary_b_only(a_fixed=200, n=80)
    OUT_A.write_text(json.dumps(rows_a, indent=2))
    OUT_B.write_text(json.dumps(rows_b, indent=2))
    OUT_VARY_A.write_text(json.dumps(rows_va, indent=2))
    OUT_VARY_B.write_text(json.dumps(rows_vb, indent=2))
    print(f"vary_numerals: {len(rows_a)} rows  ->  {OUT_A}")
    print(f"vary_operator: {len(rows_b)} rows  ->  {OUT_B}")
    print(f"  (numerals fixed at a={B_A}, b={B_B})")
    print(f"vary_a       : {len(rows_va)} rows  ->  {OUT_VARY_A}  (b fixed=4)")
    print(f"vary_b       : {len(rows_vb)} rows  ->  {OUT_VARY_B}  (a fixed=200)")
    from collections import Counter
    print(f"  operator counts in vary_operator: {dict(Counter(r['type'] for r in rows_b))}")


if __name__ == "__main__":
    main()
