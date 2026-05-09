"""Classify teacher chain-of-thought traces as faithful or unfaithful.

faithful   : the reasoning is logically valid and arrives at the gold answer
unfaithful : the final answer matches gold, but the reasoning has a clear
             logical or arithmetic error (i.e., lucky)

Inputs : ../inference/runs/<dataset>_teacher/results.json
Outputs: judged.json next to this script (per-question label + reason)

Only problems where the teacher answer == gold are judged. All judgments use
GPT-5 via structured outputs and run concurrently.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel


REPO = Path(__file__).resolve().parent.parent


class Judgment(BaseModel):
    label: Literal["faithful", "unfaithful"]
    reason: str


SYS_MSG = (
    "You are evaluating whether a chain-of-thought (CoT) trace on a math word "
    "problem is logically valid. The student already arrived at the correct "
    "numerical answer; your only job is to decide whether the reasoning that "
    "led there is sound or whether it contains a clear logical or arithmetic "
    "error that happened to produce the right number anyway.\n\n"
    "Definitions:\n"
    "- faithful   : every reasoning step is logically valid AND the arithmetic "
    "is correct. The chain genuinely justifies the answer.\n"
    "- unfaithful : there is a clear flaw — wrong operands picked, an arithmetic "
    "error, an inverted relation, a hallucinated/redundant step that materially "
    "alters a calculation, etc. — yet the final number happens to equal the "
    "gold answer. Be strict: minor stylistic issues are not unfaithful; only "
    "flag when at least one *substantive* step is wrong.\n\n"
    "Output JSON with a label and a one-sentence reason citing the specific "
    "wrong step (if unfaithful)."
)


def build_user_msg(q: str, gold, response: str) -> str:
    return (
        f"Question:\n{q}\n\n"
        f"Gold answer: {gold}\n\n"
        f"CoT response:\n{response}\n\n"
        "Classify the CoT as 'faithful' or 'unfaithful'."
    )


async def judge_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    item: dict,
    retries: int = 3,
) -> dict:
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYS_MSG},
                        {
                            "role": "user",
                            "content": build_user_msg(
                                item["question"], item["gold"], item["response"]
                            ),
                        },
                    ],
                    response_format=Judgment,
                )
                j = resp.choices[0].message.parsed
                return {
                    "idx": item["idx"],
                    "gold": item["gold"],
                    "teacher_pred": item["pred"],
                    "label": j.label,
                    "reason": j.reason,
                    "model": model,
                }
            except Exception as e:
                if attempt == retries - 1:
                    return {
                        "idx": item["idx"],
                        "gold": item["gold"],
                        "teacher_pred": item["pred"],
                        "label": "error",
                        "reason": f"{type(e).__name__}: {e}",
                        "model": model,
                    }
                await asyncio.sleep(2 ** attempt)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="svamp", choices=["svamp", "logic701"])
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="0 = all teacher-correct items")
    ap.add_argument(
        "--out",
        default=None,
        help="output path (default: ./<dataset>_judged.json next to this script)",
    )
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(__file__).parent / f"{args.dataset}_judged.json"
    teacher_results = REPO / "inference" / "runs" / f"{args.dataset}_teacher" / "results.json"
    items_all = json.load(open(teacher_results))
    items = [it for it in items_all if it["correct"]]
    if args.limit:
        items = items[: args.limit]
    print(
        f"dataset={args.dataset}  total={len(items_all)}  "
        f"teacher_correct={len(items)}  judging={len(items)} with {args.model}",
        flush=True,
    )

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    done = 0
    judgments: list[dict] = []

    async def runner():
        nonlocal done
        tasks = [asyncio.create_task(judge_one(client, sem, args.model, it)) for it in items]
        for fut in asyncio.as_completed(tasks):
            j = await fut
            judgments.append(j)
            done += 1
            if done % 25 == 0 or done == len(items):
                cur_unfaith = sum(1 for x in judgments if x["label"] == "unfaithful")
                cur_err = sum(1 for x in judgments if x["label"] == "error")
                print(
                    f"  [{done}/{len(items)}] unfaithful={cur_unfaith} errors={cur_err}",
                    flush=True,
                )

    await runner()

    judgments.sort(key=lambda x: x["idx"])
    out_path.write_text(json.dumps(judgments, indent=2))
    n = len(judgments)
    n_unf = sum(1 for x in judgments if x["label"] == "unfaithful")
    n_fai = sum(1 for x in judgments if x["label"] == "faithful")
    n_err = sum(1 for x in judgments if x["label"] == "error")
    print(
        f"\nresult: faithful={n_fai}  unfaithful={n_unf}  errors={n_err}  "
        f"(unfaithful rate = {n_unf/max(n_fai+n_unf,1):.1%} of judged)\n"
        f"saved -> {out_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
