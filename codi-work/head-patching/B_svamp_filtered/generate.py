"""SVAMP-derived paired prompts for activation patching.

Filter SVAMP to genuinely 2-numeral Multiplication problems (74 such), then
ask GPT-5 to write an Addition variant with the SAME numerals, the SAME
characters/objects/scenario, and as few token changes as possible. Keep only
pairs whose token Jaccard with the original is high.

Output: head-patching/B_svamp_filtered/pairs.json
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel


OUT = Path(__file__).resolve().parent / "pairs.json"


class Variant(BaseModel):
    problem: str
    answer: int


class Corruption(BaseModel):
    addition: Variant


SYSTEM_MSG = (
    "You are creating tightly paired math word problems for an activation-"
    "patching experiment. The user gives you a Multiplication problem and two "
    "numerals (a, b). Your job is to rewrite the problem so it asks for "
    "Addition (a + b) with as few word changes as possible. STRICT RULES:\n"
    "  - Use the EXACT SAME numerals (a and b) — do not introduce new numbers.\n"
    "  - Keep ALL character names, object names, scenario nouns, and connective "
    "phrasing identical wherever possible.\n"
    "  - Change ONLY the operation-implying phrasing (e.g., 'each / per / "
    "times / for each / multiplied by') to additive phrasing (e.g., 'and / "
    "plus / together with / in total').\n"
    "  - Do NOT introduce new entities, new relationships, or new direction-of-"
    "comparison flips (no 'shorter→taller' or 'less→more' swaps).\n"
    "  - The new problem should read naturally and clearly ask for Addition.\n"
    "Return JSON with the new problem and its answer (= a + b)."
)


def user_msg(orig_text: str, a: int, b: int) -> str:
    return (
        f"Original Multiplication problem:\n\"{orig_text}\"\n\n"
        f"Numerals: a = {a}, b = {b}.  Mul answer = {a*b}.  Add target = {a+b}.\n\n"
        f"Rewrite as an Addition problem with the SAME numerals and minimum "
        f"token edits. Only swap operation-implying words.\n\n"
        f'Return JSON: {{"addition": {{"problem": "...", "answer": {a+b}}}}}'
    )


def jaccard_tokens(s1: str, s2: str) -> float:
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    return len(t1 & t2) / max(len(t1 | t2), 1)


async def transform_one(client, sem, model, item, retries=3):
    a, b = item["a"], item["b"]
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": user_msg(item["text"], a, b)},
                    ],
                    response_format=Corruption,
                )
                t = resp.choices[0].message.parsed.addition
                return {
                    "src_idx": item["src_idx"],
                    "a": a, "b": b,
                    "clean": {"text": item["text"], "answer": a * b},
                    "corrupted": {"text": t.problem, "answer": t.answer},
                }
            except Exception as e:
                if attempt == retries - 1:
                    return {"src_idx": item["src_idx"], "error": f"{type(e).__name__}: {e}"}
                await asyncio.sleep(2 ** attempt)


async def main_async(args):
    print("loading SVAMP")
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])

    items = []
    for i, ex in enumerate(full):
        op = ex["Type"].replace("Common-Divison", "Common-Division")
        if op != "Multiplication":
            continue
        nums = [int(float(x)) for x in re.findall(r"\d+\.?\d*", ex["Equation"])]
        if len(nums) != 2:
            continue
        a, b = max(nums), min(nums)
        # Skip degenerate cases where a*b == a+b (e.g., 2*2 == 2+2 == 4) and
        # cases where the answer is non-integer.
        if a * b == a + b:
            continue
        items.append({
            "src_idx": i,
            "text": ex["question_concat"].strip().replace("  ", " "),
            "a": a, "b": b,
        })
    if args.limit:
        items = items[: args.limit]
    print(f"transforming {len(items)} 2-numeral Mul problems with {args.model}")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    transformed = []
    tasks = [asyncio.create_task(transform_one(client, sem, args.model, it)) for it in items]
    for fut in asyncio.as_completed(tasks):
        r = await fut
        transformed.append(r)
        done += 1
        if done % 10 == 0 or done == len(items):
            n_err = sum(1 for x in transformed if "error" in x)
            print(f"  [{done}/{len(items)}] errors={n_err}", flush=True)
    transformed.sort(key=lambda x: x["src_idx"])

    rows = []
    drops_err = drops_ans = drops_jacc = 0
    for r in transformed:
        if "error" in r:
            drops_err += 1; continue
        if r["corrupted"]["answer"] != r["a"] + r["b"]:
            drops_ans += 1; continue
        jacc = jaccard_tokens(r["clean"]["text"], r["corrupted"]["text"])
        if jacc < args.min_jaccard:
            drops_jacc += 1; continue
        r["jaccard"] = round(jacc, 3)
        rows.append(r)

    print(f"\nemitted {len(rows)} pairs (drops: error={drops_err}, "
          f"answer_mismatch={drops_ans}, low_jaccard={drops_jacc})")
    if rows:
        avg = sum(r["jaccard"] for r in rows) / len(rows)
        print(f"  avg token Jaccard: {avg:.3f}")
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"saved -> {OUT}")
    if rows:
        print("\n--- example pair ---")
        print(json.dumps(rows[0], indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min_jaccard", type=float, default=0.65,
                   help="Min token Jaccard between clean/corrupted to keep")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
