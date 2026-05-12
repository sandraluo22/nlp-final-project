"""Silver-trace analyzer for CODI on GSM8K.

When CODI emits an answer that doesn't match the gold, does the prediction
nonetheless match SOME plausible arithmetic combination of the numbers in
the prompt? If so, the model is "trying" arithmetic — just executing the
wrong operator chain — rather than producing nonsense.

Define:
   GOLD match     — CODI's prediction == gold (correct).
   SILVER 1-step  — CODI's prediction equals a op b for some pair (a, b)
                    in the prompt and op ∈ {+, -, *, /} (exact integer).
   SILVER 2-step  — equals (a op1 b) op2 c for some triple and op pair.
   SILVER 3-step  — same, 3 chained ops.
   NONSENSE       — none of the above; CODI emitted an unrelated integer.

For each wrong prediction, report:
   - shallowest silver depth (1, 2, or 3)
   - the matching expression (one example, if multiple)
   - whether the silver depth matches GSM8K's own chain length

Aggregate stats:
   - % of all examples that are gold-correct
   - % of wrong-predictions classified as silver-1, silver-2, silver-3, nonsense
   - cross-tab: silver depth × gold's actual chain length
   - patterns of which operations CODI substituted (if 1-step silver, which op)

Output: silver_traces_gsm8k.{json, pdf}
"""
from __future__ import annotations

import json
import re
from collections import Counter
from itertools import permutations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
COT_PATH = REPO.parent / "cf-datasets" / "gsm8k_codi_cot.json"
OUT_JSON = PD / "silver_traces_gsm8k.json"
OUT_PDF = PD / "silver_traces_gsm8k.pdf"

OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
       "*": lambda a, b: a * b, "/": lambda a, b: a / b if b != 0 else None}


def parse_question_numbers(q: str):
    """Extract integers from question text."""
    nums = re.findall(r"\b-?\d+\b", q)
    out = []
    for n in nums:
        try:
            v = int(n)
            if v != 0: out.append(v)
        except ValueError:
            pass
    # Dedupe but keep order
    seen = set(); kept = []
    for v in out:
        if v not in seen: kept.append(v); seen.add(v)
    return kept[:8]  # cap to first 8 distinct numbers


def gsm_chain_length(answer_text: str) -> int:
    return len(re.findall(r"<<(.+?)=(-?\d+\.?\d*)>>", answer_text))


def search_silver(pred: float, nums: list, max_depth: int = 3):
    """Return (depth, expression_string) if pred matches some combination
    of nums up to max_depth ops; (None, None) otherwise.
    Combinations are commutative pairs/triples at each step.
    """
    if pred != int(pred): return None, None
    pred_int = int(pred)
    nums_set = set(nums)
    # Depth 1: any pair
    for a, b in permutations(nums, 2):
        for sym, fn in OPS.items():
            v = fn(a, b)
            if v is None: continue
            if v == int(v) and int(v) == pred_int:
                return 1, f"{a}{sym}{b}={int(v)}"
    if max_depth < 2: return None, None
    # Depth 2: ((a op b) op c) for ordered triples
    triples = []
    for a, b, c in permutations(nums, 3):
        triples.append((a, b, c))
    for a, b, c in triples:
        for s1, f1 in OPS.items():
            inter = f1(a, b)
            if inter is None or inter != int(inter): continue
            inter = int(inter)
            for s2, f2 in OPS.items():
                v = f2(inter, c)
                if v is None or v != int(v): continue
                if int(v) == pred_int:
                    return 2, f"({a}{s1}{b}){s2}{c}={int(v)}"
    if max_depth < 3: return None, None
    # Depth 3: ((a op b) op c) op d
    for combo in permutations(nums, 4):
        a, b, c, d = combo
        for s1, f1 in OPS.items():
            i1 = f1(a, b)
            if i1 is None or i1 != int(i1): continue
            i1 = int(i1)
            for s2, f2 in OPS.items():
                i2 = f2(i1, c)
                if i2 is None or i2 != int(i2): continue
                i2 = int(i2)
                for s3, f3 in OPS.items():
                    v = f3(i2, d)
                    if v is None or v != int(v): continue
                    if int(v) == pred_int:
                        return 3, f"(({a}{s1}{b}){s2}{c}){s3}{d}={int(v)}"
    return None, None


def main():
    cot = json.load(open(COT_PATH))
    by_idx = {r["idx"]: r for r in cot}
    ds = load_dataset("gsm8k", "main")["test"]
    rows = []
    for i, ex in enumerate(ds):
        if i not in by_idx: continue
        rec = by_idx[i]
        question = ex["question"].strip()
        answer_text = ex["answer"]
        gold = rec["gold"]
        pred = rec.get("codi_pred_int")
        chain_len = gsm_chain_length(answer_text)
        nums = parse_question_numbers(question)
        category = "?"
        silver_depth = None
        silver_expr = None
        op_substituted_for = None
        if pred is None:
            category = "unparseable"
        elif abs(pred - gold) < 1e-3:
            category = "gold"
        else:
            d, expr = search_silver(pred, nums, max_depth=3)
            if d is not None:
                category = f"silver{d}"
                silver_depth = d; silver_expr = expr
            else:
                category = "nonsense"
        rows.append({
            "idx": i, "gold": gold, "pred": pred, "chain_len": chain_len,
            "n_numbers_in_q": len(nums),
            "category": category, "silver_depth": silver_depth,
            "silver_expression": silver_expr,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds)}  current dist: "
                  f"{dict(Counter(r['category'] for r in rows))}")

    OUT_JSON.write_text(json.dumps(rows, indent=2))

    # Stats
    N = len(rows)
    cats = Counter(r["category"] for r in rows)
    print(f"\n=== GSM8K silver-trace breakdown (N={N}) ===")
    for c in ["gold", "silver1", "silver2", "silver3", "nonsense", "unparseable"]:
        v = cats.get(c, 0)
        print(f"  {c:14s} {v:5d}  ({v/N*100:.1f}%)")

    # Wrong-prediction breakdown (excluding unparseable)
    wrong = [r for r in rows if r["category"] in {"silver1", "silver2", "silver3", "nonsense"}]
    Nw = len(wrong)
    print(f"\n=== Of the {Nw} wrong predictions: ===")
    cw = Counter(r["category"] for r in wrong)
    for c in ["silver1", "silver2", "silver3", "nonsense"]:
        v = cw.get(c, 0)
        print(f"  {c:14s} {v:5d}  ({v/max(Nw,1)*100:.1f}%)")

    # Cross-tab: silver depth × actual chain length
    print("\n=== Cross-tab: silver depth × actual chain length ===")
    chain_lens = sorted({r["chain_len"] for r in rows})
    silver_levels = ["gold", "silver1", "silver2", "silver3", "nonsense"]
    print(f"  chain_len  " + "  ".join([f"{c:>10s}" for c in silver_levels]))
    for cl in chain_lens:
        row = [r for r in rows if r["chain_len"] == cl]
        counts = Counter(r["category"] for r in row)
        line = f"  cl={cl} (N={len(row):4d})  " + "  ".join(
            [f"{counts.get(s, 0)/max(len(row),1)*100:>6.1f}%   " for s in silver_levels])
        print(line)

    # Among silver-1 cases, which op did CODI substitute?
    silver1 = [r for r in rows if r["category"] == "silver1" and r["silver_expression"]]
    op_used = Counter()
    for r in silver1:
        m = re.search(r"[+\-*/]", r["silver_expression"].replace("=", " ").split(" ")[0])
        if m: op_used[m.group(0)] += 1
    print(f"\n=== Of {len(silver1)} silver-1 wrong predictions, op used by CODI: ===")
    for op, n in op_used.most_common():
        print(f"  {op}: {n} ({n/max(len(silver1),1)*100:.1f}%)")

    # Sample silver-1 / silver-2 cases for the report
    examples = {}
    for c in ["silver1", "silver2", "silver3"]:
        examples[c] = [r for r in rows if r["category"] == c][:3]

    summary = {
        "N": N,
        "category_counts": dict(cats),
        "wrong_breakdown": dict(cw),
        "silver1_op_distribution": dict(op_used),
        "by_chain_length": {str(cl): {
            "N": len([r for r in rows if r["chain_len"] == cl]),
            "by_category": {c: int(sum(1 for r in rows
                                        if r["chain_len"] == cl and r["category"] == c))
                            for c in silver_levels},
        } for cl in chain_lens},
        "examples": examples,
    }
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # PDF
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: overall pie + wrong-prediction breakdown
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Overall
        labels = ["gold", "silver1", "silver2", "silver3", "nonsense", "unparseable"]
        vals = [cats.get(l, 0) for l in labels]
        colors = ["#2ca02c", "#1f77b4", "#aec7e8", "#dbe9f4", "#d62728", "#7f7f7f"]
        axes[0].pie(vals, labels=[f"{l}\n{v}" for l, v in zip(labels, vals)],
                     colors=colors, autopct="%1.1f%%", startangle=90)
        axes[0].set_title(f"Overall: GOLD vs SILVER vs nonsense (N={N})",
                          fontsize=11, fontweight="bold")
        # Wrong-only
        wlabels = ["silver1", "silver2", "silver3", "nonsense"]
        wvals = [cw.get(l, 0) for l in wlabels]
        axes[1].pie(wvals, labels=[f"{l}\n{v}" for l, v in zip(wlabels, wvals)],
                     colors=["#1f77b4", "#aec7e8", "#dbe9f4", "#d62728"],
                     autopct="%1.1f%%", startangle=90)
        axes[1].set_title(f"Of {Nw} wrong predictions: how silver?",
                          fontsize=11, fontweight="bold")
        fig.suptitle("CODI-GPT-2 on GSM8K — silver-trace analysis",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 2: by-chain-length stacked bars
        fig, ax = plt.subplots(figsize=(13, 5))
        cl_keys = sorted({r["chain_len"] for r in rows})
        cl_keys = [k for k in cl_keys if k <= 8]
        cat_list = ["gold", "silver1", "silver2", "silver3", "nonsense"]
        bottoms = np.zeros(len(cl_keys))
        for c, color in zip(cat_list, colors[:5]):
            counts = [sum(1 for r in rows if r["chain_len"] == cl and r["category"] == c) for cl in cl_keys]
            totals = [sum(1 for r in rows if r["chain_len"] == cl) for cl in cl_keys]
            fracs = np.array([c2 / max(t, 1) for c2, t in zip(counts, totals)])
            ax.bar(np.arange(len(cl_keys)), fracs, bottom=bottoms, color=color, label=c)
            bottoms += fracs
        ax.set_xticks(np.arange(len(cl_keys)))
        ax.set_xticklabels([f"cl={cl}\nN={sum(1 for r in rows if r['chain_len']==cl)}"
                            for cl in cl_keys])
        ax.set_ylabel("fraction"); ax.set_ylim(0, 1.05)
        ax.set_title("Silver-trace category by gold-chain length",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 3: op substitution distribution (silver-1 only)
        fig, ax = plt.subplots(figsize=(10, 5))
        ops_seen = ["+", "-", "*", "/"]
        op_counts = [op_used.get(o, 0) for o in ops_seen]
        ax.bar(ops_seen, op_counts, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        for o, v in zip(ops_seen, op_counts):
            ax.text(o, v + max(op_counts)*0.01, str(v), ha="center", fontsize=10)
        ax.set_ylabel("count")
        ax.set_title(f"Of {len(silver1)} silver-1 wrong predictions, "
                     f"which operator CODI used",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # Page 4: example silver traces
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.axis("off")
        ax.set_title("Sample silver-trace examples", fontsize=13, fontweight="bold", loc="left")
        ytxt = 0.95
        for c in ["silver1", "silver2", "silver3"]:
            ax.text(0.02, ytxt, f"=== {c.upper()} ===", fontsize=11, fontweight="bold")
            ytxt -= 0.04
            for r in examples.get(c, []):
                ax.text(0.04, ytxt,
                        f"idx={r['idx']}: gold={r['gold']:.0f}, "
                        f"codi_pred={r['pred']:.0f}, expr: {r['silver_expression']}, "
                        f"actual chain_len={r['chain_len']}",
                        fontsize=8.5, family="monospace")
                ytxt -= 0.04
            ytxt -= 0.02
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
