"""GPT-transformed CF dataset.

For each Mul/Div SVAMP problem:
  - Extract the top-2 numerals (a, b with a >= b) from the equation.
  - Use GPT-5 to generate an Addition variant and a Subtraction variant of the
    problem text, using the same two numerals.
  - Add answer  = a + b
  - Sub answer  = a - b   (we swap if needed so a >= b, ensuring positive)

Per source problem we emit 2 CF rows (one Add, one Sub). Total dataset size
≈ 274 × 2 = 548.

Output: cf-datasets/cf_gpt_transformed.json.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from datasets import concatenate_datasets, load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel


REPO = Path(__file__).resolve().parent.parent
OUT = REPO.parent / "cf-datasets" / "cf_gpt_transformed.json"


class Variant(BaseModel):
    problem: str
    answer: int


class TransformedFull(BaseModel):
    addition: Variant
    subtraction: Variant
    multiplication: Variant
    division: Variant


class TransformedNoDiv(BaseModel):
    addition: Variant
    subtraction: Variant
    multiplication: Variant


SYSTEM_MSG = (
    "You create counterfactual math word problems. "
    "Given an existing word problem and two specific numerals (a, b), you write "
    "NEW word problems using EXACTLY those two numerals (a and b) — preserving the "
    "scenario/objects/characters from the original where possible. Each new problem "
    "must read fluently in natural English and be unambiguous about which operation "
    "is being asked."
)


def user_msg(orig_op: str, orig_text: str, a: int, b: int, include_div: bool) -> str:
    parts = [
        f"Original {orig_op} problem:",
        f"\"{orig_text}\"",
        "",
        f"Numerals to reuse: a = {a}, b = {b}.",
        "",
        f"Write NEW problems using exactly these two numerals:",
        f"  1. ADDITION variant       — natural answer is a + b = {a + b}",
        f"  2. SUBTRACTION variant    — natural answer is a - b = {a - b}",
        f"  3. MULTIPLICATION variant — natural answer is a * b = {a * b}",
    ]
    if include_div:
        parts.append(f"  4. DIVISION variant       — natural answer is a / b = {a // b}")
    parts += [
        "",
        "Use the same scenario/characters/objects from the original where possible.",
        "Each problem MUST read as a clear single-question word problem.",
        "Return only the JSON; no explanation.",
    ]
    return "\n".join(parts)


def parse_numerals(equation: str) -> list[int]:
    return [int(float(x)) for x in re.findall(r"\d+\.?\d*", equation)]


async def transform_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    item: dict,
    retries: int = 3,
) -> dict | None:
    a, b = item["a"], item["b"]
    include_div = (b != 0) and (a % b == 0) and (a // b > 0)
    schema = TransformedFull if include_div else TransformedNoDiv
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": user_msg(
                            item["type"], item["question_concat"],
                            a, b, include_div,
                        )},
                    ],
                    response_format=schema,
                )
                t = resp.choices[0].message.parsed
                out = {
                    "src_idx": item["src_idx"],
                    "src_type": item["type"],
                    "src_question": item["question_concat"],
                    "src_answer": item["answer"],
                    "a": a, "b": b,
                    "addition": t.addition.model_dump(),
                    "subtraction": t.subtraction.model_dump(),
                    "multiplication": t.multiplication.model_dump(),
                    "model": model,
                }
                if include_div:
                    out["division"] = t.division.model_dump()
                return out
            except Exception as e:
                if attempt == retries - 1:
                    return {
                        "src_idx": item["src_idx"],
                        "error": f"{type(e).__name__}: {e}",
                    }
                await asyncio.sleep(2 ** attempt)
        return None


async def main_async(args):
    print("loading SVAMP")
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])

    items = []
    for i, ex in enumerate(full):
        op = ex["Type"].replace("Common-Divison", "Common-Division")
        if op not in ("Multiplication", "Common-Division"):
            continue
        nums = parse_numerals(ex["Equation"])
        if len(nums) < 2:
            continue
        nums_sorted = sorted(set(nums), reverse=True)
        if len(nums_sorted) < 2:
            continue
        a, b = nums_sorted[0], nums_sorted[1]
        if a < b:
            a, b = b, a  # ensure a >= b
        items.append({
            "src_idx": i, "type": op,
            "question_concat": ex["question_concat"].strip().replace("  ", " "),
            "answer": float(str(ex["Answer"]).replace(",", "")),
            "a": a, "b": b,
        })
    if args.limit:
        items = items[: args.limit]
    print(f"transforming {len(items)} Mul+Div problems with {args.model}")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    transformed: list[dict] = []

    tasks = [asyncio.create_task(transform_one(client, sem, args.model, it))
             for it in items]
    for fut in asyncio.as_completed(tasks):
        r = await fut
        if r is not None:
            transformed.append(r)
        done += 1
        if done % 20 == 0 or done == len(items):
            n_err = sum(1 for x in transformed if "error" in x)
            print(f"  [{done}/{len(items)}] errors={n_err}", flush=True)

    transformed.sort(key=lambda x: x["src_idx"])

    # Validate that the GPT-claimed answer matches a+b / a-b. If GPT returned
    # something different (e.g., paraphrased the problem in a way that changed
    # the computation), drop the row and log it.
    rows: list[dict] = []
    drops = 0
    for r in transformed:
        if "error" in r:
            drops += 1
            continue
        a, b = r["a"], r["b"]
        # Validate per-variant answers.
        targets = {
            "Addition": (r["addition"], a + b),
            "Subtraction": (r["subtraction"], a - b),
            "Multiplication": (r["multiplication"], a * b),
        }
        if "division" in r:
            targets["Common-Division"] = (r["division"], a // b)
        bad = [op for op, (v, want) in targets.items() if v["answer"] != want]
        if bad:
            drops += 1
            continue
        for op, (v, want) in targets.items():
            rows.append({
                "idx": len(rows), "src_idx": r["src_idx"],
                "src_type": r["src_type"], "type": op,
                "src_question": r["src_question"],
                "cf_subs": [a, b], "cf_question": v["problem"],
                "cf_answer": want, "src_answer": r["src_answer"],
                "model": r["model"],
            })

    print(f"\n{len(transformed)} GPT calls; dropped {drops} for errors / answer mismatch")
    print(f"emitted {len(rows)} CF rows")
    by_type = {"Addition": 0, "Subtraction": 0}
    by_src = {"Multiplication": 0, "Common-Division": 0}
    for r in rows:
        by_type[r["type"]] += 1
        by_src[r["src_type"]] += 1
    print(f"  by surface type: {by_type}")
    print(f"  by src type:     {by_src}")
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"saved -> {OUT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
