"""Numeral-corruption pairs for head patching.

Same operation (Subtraction), same scenario template, only the second
numeral b differs:
  clean    : "John has {a} apples in his basket. He gives {b} apples to his sister. ..."
  corrupted: "John has {a} apples in his basket. He gives 1 apple to his sister. ..."

clean   answer = a - b
corrupt answer = a - 1

We additionally generate `a=1` Multiplication pairs as a control where the
corruption sits in the FIRST numeral and the answer remains positive:
  clean    : "{name} has {a} packs. Each pack has {b} cookies. How many cookies?"
  corrupted: "{name} has 1 pack.  Each pack has {b} cookies. How many cookies?"
clean   answer = a*b
corrupt answer = b

Output:
  cf-datasets/numeral_pairs_b1_sub.json   # b=1 Sub corruption
  cf-datasets/numeral_pairs_a1_mul.json   # a=1 Mul corruption
"""

import json
import random
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
OUT_B1 = OUT_DIR / "numeral_pairs_b1_sub.json"
OUT_A1 = OUT_DIR / "numeral_pairs_a1_mul.json"


SUB_TEMPLATE = (
    "John has {a} apples in his basket. He gives {b} {plural} to his sister. "
    "How many apples does John have left?"
)


def make_b1_sub(n: int = 60, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    seen = set()
    while len(rows) < n:
        a = rng.randint(5, 30)
        b = rng.randint(2, a - 1)         # b >= 2 so corrupted (b=1) differs
        if a - b == a - 1:                # ensures clean and corrupted answers differ
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        clean_text = SUB_TEMPLATE.format(a=a, b=b, plural="apples")
        corr_text = SUB_TEMPLATE.format(a=a, b=1, plural="apple")
        rows.append({
            "src_idx": len(rows),
            "a": a, "b_clean": b, "b_corr": 1,
            "type": "Subtraction",
            "clean":     {"text": clean_text, "answer": a - b},
            "corrupted": {"text": corr_text, "answer": a - 1},
        })
    return rows


MUL_TEMPLATE_CLEAN = (
    "Maya has {a} packs of stickers. Each pack contains {b} stickers. "
    "How many stickers does Maya have in total?"
)
MUL_TEMPLATE_CORR = (
    "Maya has 1 pack of stickers. Each pack contains {b} stickers. "
    "How many stickers does Maya have in total?"
)


def make_a1_mul(n: int = 60, seed: int = 1):
    rng = random.Random(seed)
    rows = []
    seen = set()
    while len(rows) < n:
        a = rng.randint(2, 12)            # keep small so a*b doesn't blow up
        b = rng.randint(2, 12)
        if a * b == b:                    # would make clean == corrupted (a=1 ⇒ b)
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append({
            "src_idx": len(rows),
            "a_clean": a, "a_corr": 1, "b": b,
            "type": "Multiplication",
            "clean":     {"text": MUL_TEMPLATE_CLEAN.format(a=a, b=b), "answer": a * b},
            "corrupted": {"text": MUL_TEMPLATE_CORR.format(b=b),       "answer": b},
        })
    return rows


def main():
    rows_b = make_b1_sub(60)
    rows_a = make_a1_mul(60)
    OUT_B1.write_text(json.dumps(rows_b, indent=2))
    OUT_A1.write_text(json.dumps(rows_a, indent=2))
    print(f"b=1 Sub pairs   : {len(rows_b)} -> {OUT_B1}")
    print(f"a=1 Mul pairs   : {len(rows_a)} -> {OUT_A1}")
    print("\nexample b=1 Sub:")
    print(json.dumps(rows_b[0], indent=2))
    print("\nexample a=1 Mul:")
    print(json.dumps(rows_a[0], indent=2))


if __name__ == "__main__":
    main()
