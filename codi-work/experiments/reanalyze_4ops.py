"""Re-bucket steering predictions against ALL FOUR operator gold answers.

The original steering.py classified each output as {src, tgt, other}. But
"other" might actually be the answer that *another* operator would produce
(e.g. Sub→Add steering could push the output to a*b instead of a+b).

This script reuses the saved predictions, joins on cf_balanced.json to
recover the numerals, and computes the full breakdown:
  Addition / Subtraction / Multiplication / Common-Division / unmatched.
"""

import json
import math
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
CF_DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"

OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


def gold_for_op(a: int, b: int, op: str) -> float | None:
    if op == "Addition":
        return float(a + b)
    if op == "Subtraction":
        return float(a - b)
    if op == "Multiplication":
        return float(a * b)
    if op == "Common-Division":
        if b == 0 or a % b != 0:
            return None
        return float(a // b)
    raise ValueError(op)


def bucket_pred(pred: float, golds: dict[str, float | None]) -> str:
    matched = [op for op, g in golds.items() if g is not None and pred == g]
    if not matched:
        return "unmatched"
    if len(matched) > 1:
        return "tied:" + ",".join(matched)
    return matched[0]


def main():
    cf_rows = json.load(open(CF_DATA))
    by_src = {r["src_idx"]: r for r in cf_rows}

    files = sorted(EXP.glob("runs_*.json"))
    print(f"reanalyzing {len(files)} run files\n")

    for f in files:
        d = json.load(open(f))
        cfg = d["config"]
        src_op = cfg["src"]
        tgt_op = cfg["tgt"]
        mode = cfg.get("mode") or "residual_centroid"
        scale = cfg.get("scale", 1.0)
        per_prob = d["per_problem"]

        # We need numerals; only proceed if predictions joinable to CF.
        # Pick the most-effective protocol (highest tgt_rate) for headline.
        proto_summaries = []
        for proto, rows in per_prob.items():
            counts = Counter()
            for r in rows:
                src_idx = r["src_idx"]
                cf = by_src.get(src_idx)
                if cf is None:
                    continue
                a, b = cf["cf_subs"][:2] if len(cf["cf_subs"]) >= 2 else (None, None)
                if a is None or b is None:
                    continue
                golds = {op: gold_for_op(a, b, op) for op in OPS}
                bucket = bucket_pred(r["pred"], golds)
                counts[bucket] += 1
            proto_summaries.append((proto, counts))

        # Aggregate across all (proto, problem) — but we want PER PROTOCOL view.
        # Print headline + a few key protocols.
        n = len(per_prob.get(list(per_prob.keys())[0], []))
        print(f"=== {f.name}  ({src_op} -> {tgt_op}, mode={mode}, scale={scale}, n={n}) ===")
        # Choose a meaningful subset of protocols to print
        keep = []
        for proto, counts in proto_summaries:
            if proto == "baseline":
                keep.append((proto, counts))
        for proto, counts in proto_summaries:
            if proto.startswith(("all_", "even_", "odd_", "single_step")):
                keep.append((proto, counts))
        # Plus best 3 single-position protocols by tgt-class match.
        single_pos = [(p, c) for p, c in proto_summaries if p.startswith("single_layer") or p.startswith("single_step")]
        single_pos.sort(key=lambda x: -x[1].get(tgt_op, 0))
        for p, c in single_pos[:3]:
            if (p, c) not in keep:
                keep.append((p, c))

        header = f"{'protocol':<35s}  " + "  ".join(f"{op[:5]:>6s}" for op in OPS) + f"  {'unm':>5s}"
        print(header)
        print("-" * len(header))
        for proto, counts in keep:
            row = [counts.get(op, 0) for op in OPS]
            unm = counts.get("unmatched", 0)
            row_str = "  ".join(f"{c:>6d}" for c in row)
            print(f"{proto:<35s}  {row_str}  {unm:>5d}")
        print()


if __name__ == "__main__":
    main()
