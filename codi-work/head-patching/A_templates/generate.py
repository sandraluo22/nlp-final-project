"""Synthetic templated paired Mul/Add prompts for activation patching.

Each template defines a (mul_text, add_text) skeleton with the SAME slots
({a}, {b}, plus character/object placeholders). The two skeletons share most
tokens; they differ only in the operator-specific connective phrasing.

We sample numerals (a, b) and slot fillers, instantiate, and emit pairs whose
clean answer = a * b and corrupted answer = a + b.

Output: head-patching/A_templates/pairs.json
"""

import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "pairs.json"


# A list of paired Mul/Add templates. Both versions share most lexical content;
# only operator-implying words/phrases differ.
TEMPLATES = [
    {
        "mul": "{name} has {a} {container}s of {item}. Each {container} holds {b} {item}. How many {item} does {name} have in total?",
        "add": "{name} has {a} {item} in one {container} and {b} {item} in another {container}. How many {item} does {name} have in total?",
    },
    {
        "mul": "{name} packs {a} {item} into each {container}. {pron_subj} fills {b} {container}s. How many {item} does {pron_subj} pack?",
        "add": "{name} packs {a} {item} into the morning {container} and {b} {item} into the afternoon {container}. How many {item} does {pron_subj} pack?",
    },
    {
        "mul": "There are {a} shelves in the room. Each shelf holds {b} {item}. How many {item} are in the room?",
        "add": "There are {a} {item} on the top shelf and {b} {item} on the bottom shelf. How many {item} are in the room?",
    },
    {
        "mul": "{name} bought {a} {container}s. Each {container} contains {b} {item}. How many {item} did {name} buy?",
        "add": "{name} bought {a} {item} from the first store and {b} {item} from the second store. How many {item} did {name} buy?",
    },
    {
        "mul": "A factory produces {a} {item} every minute for {b} minutes. How many {item} did the factory produce?",
        "add": "A factory produced {a} {item} in the morning and {b} {item} in the afternoon. How many {item} did the factory produce?",
    },
    {
        "mul": "{name} reads {a} pages each day for {b} days. How many pages does {name} read?",
        "add": "{name} read {a} pages on Monday and {b} pages on Tuesday. How many pages does {name} read?",
    },
    {
        "mul": "Each room has {a} {item}, and there are {b} rooms. How many {item} are there in all the rooms?",
        "add": "One room has {a} {item}, and another room has {b} {item}. How many {item} are there in all the rooms?",
    },
    {
        "mul": "{name} earns {a} dollars per hour and works for {b} hours. How much money does {name} earn?",
        "add": "{name} earned {a} dollars on Monday and {b} dollars on Tuesday. How much money does {name} earn?",
    },
    {
        "mul": "{name} has {a} {container}s, with {b} {item} inside each one. How many {item} does {name} have?",
        "add": "{name} has {a} {item} in one {container} and {b} {item} in another. How many {item} does {name} have?",
    },
    {
        "mul": "A garden has {a} rows of {item}. Each row has {b} {item}. How many {item} are in the garden?",
        "add": "A garden has {a} {item} in the front and {b} {item} in the back. How many {item} are in the garden?",
    },
    {
        "mul": "{name} prints {a} copies for each of the {b} students. How many copies does {name} print?",
        "add": "{name} printed {a} copies for the morning class and {b} copies for the afternoon class. How many copies does {name} print?",
    },
    {
        "mul": "Each box weighs {a} pounds, and there are {b} boxes. What is the total weight?",
        "add": "One box weighs {a} pounds and another box weighs {b} pounds. What is the total weight?",
    },
]

VARS = {
    "name": ["Tom", "Sarah", "Ben", "Lily", "Max", "Anna", "Zoe", "Kai"],
    "pron_subj": ["he", "she"],
    "container": ["box", "bag", "basket", "crate"],
    "item": ["apples", "marbles", "books", "pencils", "candies", "cookies", "toys", "stickers"],
}


def sample_filler(rng: random.Random):
    return {k: rng.choice(v) for k, v in VARS.items()}


def jaccard_tokens(s1: str, s2: str) -> float:
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    return len(t1 & t2) / max(len(t1 | t2), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per_template", type=int, default=15,
                   help="Numeral instantiations per template")
    p.add_argument("--min_jaccard", type=float, default=0.4,
                   help="Min token Jaccard between mul/add to keep a pair")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for t_idx, tpl in enumerate(TEMPLATES):
        for _ in range(args.per_template):
            # Sample numerals: small enough so a*b is reasonable for a 1B model.
            a = rng.randint(2, 9)
            b = rng.randint(2, 9)
            if a + b == a * b:
                continue  # avoid degenerate (e.g., 2+2 vs 2*2 both = 4)
            fill = sample_filler(rng)
            mul_text = tpl["mul"].format(a=a, b=b, **fill)
            add_text = tpl["add"].format(a=a, b=b, **fill)
            jacc = jaccard_tokens(mul_text, add_text)
            if jacc < args.min_jaccard:
                continue
            rows.append({
                "src_idx": len(rows),
                "template_idx": t_idx,
                "a": a, "b": b,
                "jaccard": round(jacc, 3),
                "clean":     {"text": mul_text, "answer": a * b},
                "corrupted": {"text": add_text, "answer": a + b},
            })

    print(f"emitted {len(rows)} pairs across {len(TEMPLATES)} templates")
    print(f"  avg token Jaccard: {sum(r['jaccard'] for r in rows) / max(len(rows),1):.3f}")
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"saved -> {OUT}")
    print("\n--- example pair ---")
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
