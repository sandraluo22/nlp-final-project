"""Generate a magnitude-matched counterfactual SVAMP dataset.

For each SVAMP problem:
  1. Sample a target answer-magnitude bucket uniformly at random from
     {1-9, 10-99, 100-999, 1000+}.
  2. Rejection-sample new integer numerals for the equation until the
     evaluated answer falls in the target bucket and is integer-valued.
  3. Substitute those numerals into the body+question text via greedy
     left-to-right matching against the original numerals.
  4. Emit (cf_question, cf_gold, target_bucket, ...) per problem.

By construction, output magnitude is uncorrelated with operator class — so any
operator-discriminative direction in the activations cannot be a magnitude
direction in disguise.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


REPO = Path(__file__).resolve().parent.parent
OUT = REPO.parent / "cf-datasets" / "cf_magmatched.json"

BUCKETS = [(1, 9), (10, 99), (100, 999), (1000, 9999)]
BUCKET_NAMES = ["<10", "10-99", "100-999", "1000+"]
SEED = 13


def parse_equation_numerals(equation: str) -> list[float]:
    """Numerals from the equation in left-to-right order."""
    return [float(x) for x in re.findall(r"\d+\.?\d*", equation)]


def evaluate_with_subs(equation: str, subs: list[float]) -> float:
    """Replace numerals in equation by `subs` (left-to-right) and eval()."""
    parts = re.split(r"(\d+\.?\d*)", equation)
    out: list[str] = []
    i = 0
    for p in parts:
        if re.fullmatch(r"\d+\.?\d*", p):
            out.append(repr(float(subs[i])))
            i += 1
        else:
            out.append(p)
    expr = "".join(out)
    # equation strings are well-formed and only contain digits/operators/parens/spaces
    return eval(expr, {"__builtins__": {}}, {})


def sample_subs(n: int, op_type: str, rng: random.Random) -> list[int]:
    """Sample n positive integer numerals biased to the operator's typical
    range, but kept small enough that products stay reasonable."""
    if op_type == "Multiplication":
        # Keep individual factors modest so 2- and 3-factor products span buckets.
        return [rng.randint(2, 30) for _ in range(n)]
    if op_type == "Common-Division":
        # Will retry until division is integer-valued.
        return [rng.randint(2, 50) for _ in range(n)]
    # Addition / Subtraction: any small-medium integers.
    return [rng.randint(1, 200) for _ in range(n)]


def fits_bucket(answer: float, bucket_idx: int) -> bool:
    if not (isinstance(answer, (int, float)) and answer >= 0):
        return False
    if abs(answer - round(answer)) > 1e-9:
        return False
    a = int(round(answer))
    lo, hi = BUCKETS[bucket_idx]
    return lo <= a <= hi


def find_numeral_positions(body_question: str, originals: list[float]) -> list[tuple[int, int]] | None:
    """Greedy left-to-right match of equation numerals against integer numerals
    in body+question. Returns list of (start, end) per match, or None if a
    numeral cannot be located."""
    nums_in_text = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\d+", body_question)]
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
    """Replace the (start,end) spans in `text` with subs, in reverse order."""
    paired = sorted(zip(positions, subs), key=lambda x: x[0][0], reverse=True)
    out = text
    for (s, e), v in paired:
        out = out[:s] + str(v) + out[e:]
    return out


def main():
    rng = random.Random(SEED)
    print("loading SVAMP")
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])

    out: list[dict] = []
    failures = Counter()
    bucket_assigned = Counter()

    for idx, ex in enumerate(full):
        op_type = ex["Type"].replace("Common-Divison", "Common-Division")
        equation = ex["Equation"]
        body = ex["Body"]
        question = ex["Question"]
        full_text = body + " " + question
        originals = parse_equation_numerals(equation)

        # Try buckets in random order; assign to whichever produces a valid match.
        bucket_order = list(range(len(BUCKETS)))
        rng.shuffle(bucket_order)

        emitted = False
        for bucket_idx in bucket_order:
            for _ in range(120):  # rejection samples
                subs = sample_subs(len(originals), op_type, rng)
                try:
                    ans = evaluate_with_subs(equation, [float(x) for x in subs])
                except ZeroDivisionError:
                    continue
                if not fits_bucket(ans, bucket_idx):
                    continue
                positions = find_numeral_positions(full_text, originals)
                if positions is None:
                    failures["no-numeral-match"] += 1
                    break  # can't do anything; try a different bucket won't help
                cf_text = substitute_numerals(full_text, positions, subs)
                # Split the substituted text back into body / question on the
                # boundary we used to join.
                # Use the original lengths to find the cut: rebuild from body+space+question.
                cf_body = substitute_numerals(body, positions, subs[: sum(1 for p in positions if p[0] < len(body))])
                # Easier: rebuild by separately substituting body and question with the
                # numerals that fall in each. Recompute positions per part:
                pos_body = find_numeral_positions(body, originals)
                pos_question = None
                if pos_body is None:
                    # fallback: keep cf_text as-is and split by " "
                    cf_body_part, cf_question_part = cf_text, ""
                else:
                    cf_body_part = substitute_numerals(body, pos_body, subs)
                    cf_question_part = question  # numerals are typically only in body
                    pos_question = find_numeral_positions(question, originals)
                    if pos_question is not None and any(pos_question):
                        cf_question_part = substitute_numerals(question, pos_question, subs)

                cf_concat = cf_body_part + (" " if cf_body_part and cf_question_part else "") + cf_question_part
                out.append(
                    {
                        "idx": idx,
                        "type": op_type,
                        "orig_equation": equation,
                        "orig_body": body,
                        "orig_question": question,
                        "orig_answer": float(ex["Answer"]),
                        "cf_subs": subs,
                        "cf_body": cf_body_part,
                        "cf_question": cf_question_part,
                        "cf_question_concat": cf_concat,
                        "cf_answer": int(round(ans)),
                        "target_bucket_idx": bucket_idx,
                        "target_bucket": BUCKET_NAMES[bucket_idx],
                    }
                )
                bucket_assigned[BUCKET_NAMES[bucket_idx]] += 1
                emitted = True
                break
            if emitted:
                break

        if not emitted:
            failures["all-buckets-rejected"] += 1

    print(f"emitted {len(out)} / {len(full)} counterfactuals")
    print("bucket assignment:", dict(bucket_assigned))
    print("failures:", dict(failures))
    by_type_bucket = Counter()
    for r in out:
        by_type_bucket[(r["type"], r["target_bucket"])] += 1
    print("by type x bucket:")
    for (t, b), n in sorted(by_type_bucket.items()):
        print(f"  {t:>18s} {b:>8s}: {n}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
