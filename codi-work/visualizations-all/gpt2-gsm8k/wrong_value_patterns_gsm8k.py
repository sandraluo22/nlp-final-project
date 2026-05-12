"""Value-level patterns in the 'other_number' wrong emits.

For the 534 wrong emits before rescue, examine the NUMBERS themselves:
  - Most-common specific values (attractor numbers)
  - First-digit and last-digit distributions
  - Roundness (multiples of 5 / 10 / 25 / 50 / 100)
  - Digit count distribution vs gold
  - Distribution of the values (histogram, log scale)

Output: wrong_value_patterns_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
FD_JSON = REPO / "experiments" / "computation_probes" / "force_decode_per_step_gsm8k.json"
OUT_JSON = PD / "wrong_value_patterns_gsm8k.json"
OUT_PDF = PD / "wrong_value_patterns_gsm8k.pdf"


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def emit_final(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def first_digit(x):
    s = str(abs(int(x))) if x == int(x) else str(abs(x)).split(".")[0]
    s = s.lstrip("0")
    return int(s[0]) if s else 0


def last_digit(x):
    return int(abs(int(x))) % 10


def main():
    fd = json.load(open(FD_JSON))
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]

    rows = fd["rows"]
    n_steps = len(rows[0]["step_emits"])

    # Collect the same set: 'other_number' wrong emits from problems
    # eventually correct
    wrong_vals = []
    gold_vals = []
    for r in rows:
        idx = r["idx"]
        ex = ds[idx]
        gold_match = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if gold_match is None: continue
        gold = float(gold_match.group(1))
        markers = [(float(a), op, float(b), float(c))
                   for a, op, b, c in parse_markers(ex["answer"])]
        emit_vals = [emit_final(e) for e in r["step_emits"]]
        step_correct = [v is not None and abs(v - gold) < 1e-3 for v in emit_vals]
        if not step_correct[-1]: continue
        first_correct = next((k for k, c in enumerate(step_correct) if c), -1)
        if first_correct < 1: continue
        for k in range(first_correct):
            v = emit_vals[k]
            if v is None or v == 0: continue
            # exclude marker hits
            if markers and any(abs(v - m[3]) < 1e-3 for m in markers): continue
            wrong_vals.append(v)
            gold_vals.append(gold)

    wrong_vals = np.array(wrong_vals)
    gold_vals = np.array(gold_vals)
    N = len(wrong_vals)
    print(f"  N other-number wrong emits: {N}")

    # === Value-level analyses ===

    # 1. Most-common specific values
    val_counter = Counter()
    for v in wrong_vals:
        # round to int if whole
        if v == int(v):
            val_counter[int(v)] += 1
        else:
            val_counter[round(v, 2)] += 1
    top_values = val_counter.most_common(30)
    print(f"\nTop 30 specific wrong-emit values:")
    for v, c in top_values:
        print(f"  {v}: {c}  ({c/N*100:.1f}%)")

    # 2. First-digit distribution
    first_digits = [first_digit(v) for v in wrong_vals if v != 0]
    fd_counts = Counter(first_digits)
    fd_dist = np.array([fd_counts.get(d, 0) / len(first_digits) * 100
                        for d in range(1, 10)])
    benford = np.array([np.log10(1 + 1 / d) * 100 for d in range(1, 10)])
    print(f"\nFirst-digit distribution (vs Benford's law):")
    for d in range(1, 10):
        bar = "█" * int(fd_dist[d-1] / 2)
        print(f"  {d}: emit {fd_dist[d-1]:5.1f}%  | benford {benford[d-1]:5.1f}%  {bar}")

    # 3. Last-digit distribution
    last_digits = [last_digit(v) for v in wrong_vals]
    ld_counts = Counter(last_digits)
    ld_dist = np.array([ld_counts.get(d, 0) / len(last_digits) * 100 for d in range(10)])
    print(f"\nLast-digit distribution (uniform = 10%):")
    for d in range(10):
        bar = "█" * int(ld_dist[d] / 2)
        print(f"  {d}: {ld_dist[d]:5.1f}%  {bar}")

    # 4. Roundness
    n_mult10 = int(np.sum(wrong_vals % 10 == 0))
    n_mult5 = int(np.sum(wrong_vals % 5 == 0))
    n_mult100 = int(np.sum(wrong_vals % 100 == 0))
    print(f"\nRoundness (vs N={N}):")
    print(f"  multiple of 5:   {n_mult5}  ({n_mult5/N*100:.1f}%)")
    print(f"  multiple of 10:  {n_mult10}  ({n_mult10/N*100:.1f}%)")
    print(f"  multiple of 100: {n_mult100}  ({n_mult100/N*100:.1f}%)")
    gold_mult5 = int(np.sum(gold_vals % 5 == 0))
    gold_mult10 = int(np.sum(gold_vals % 10 == 0))
    print(f"  (for comparison, gold values: mult5={gold_mult5/N*100:.1f}%  "
          f"mult10={gold_mult10/N*100:.1f}%)")

    # 5. Digit count distribution vs gold
    emit_digs = [len(str(int(abs(v)))) for v in wrong_vals if v != 0]
    gold_digs = [len(str(int(abs(v)))) for v in gold_vals if v != 0]
    emit_dig_dist = Counter(emit_digs)
    gold_dig_dist = Counter(gold_digs)
    print(f"\nDigit-count distribution (emit vs gold):")
    for d in sorted(set(list(emit_dig_dist.keys()) + list(gold_dig_dist.keys()))):
        ec = emit_dig_dist.get(d, 0)
        gc = gold_dig_dist.get(d, 0)
        print(f"  {d}-digit numbers: emit {ec:>4} ({ec/len(emit_digs)*100:.1f}%) | "
              f"gold {gc:>4} ({gc/len(gold_digs)*100:.1f}%)")

    # 6. Sign + range
    n_neg = int(np.sum(wrong_vals < 0))
    n_zero_emit = int(np.sum(wrong_vals == 0))  # already excluded
    n_pos = int(np.sum(wrong_vals > 0))
    print(f"\nSign: {n_neg} negative, {n_pos} positive ({n_neg/N*100:.1f}% / {n_pos/N*100:.1f}%)")
    print(f"  emit value range: [{wrong_vals.min():g}, {wrong_vals.max():g}]")
    print(f"  emit value median: {np.median(wrong_vals):g}")
    print(f"  gold value range: [{gold_vals.min():g}, {gold_vals.max():g}]")
    print(f"  gold value median: {np.median(gold_vals):g}")

    out = {
        "N": N,
        "top_values": [{"value": float(v), "count": c, "pct": c/N*100}
                       for v, c in top_values],
        "first_digit_dist_pct": fd_dist.tolist(),
        "benford_pct": benford.tolist(),
        "last_digit_dist_pct": ld_dist.tolist(),
        "roundness": {"mult5_pct": n_mult5/N*100, "mult10_pct": n_mult10/N*100,
                       "mult100_pct": n_mult100/N*100,
                       "gold_mult5_pct": gold_mult5/N*100,
                       "gold_mult10_pct": gold_mult10/N*100},
        "emit_digit_count_dist": dict(emit_dig_dist),
        "gold_digit_count_dist": dict(gold_dig_dist),
        "sign": {"neg_pct": n_neg/N*100, "pos_pct": n_pos/N*100},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # Plots
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: top values bar chart
        fig, ax = plt.subplots(figsize=(13, 6))
        top_for_plot = top_values[:25]
        labels = [str(v) for v, _ in top_for_plot]
        cnts = [c for _, c in top_for_plot]
        ax.bar(range(len(labels)), cnts, color="#dd8452", edgecolor="black")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("# wrong emits with this value")
        for i, c in enumerate(cnts):
            ax.text(i, c + 0.3, str(c), ha="center", fontsize=8)
        ax.set_title(f"Most-common specific wrong-emit values  (N={N} total)\n"
                     "'Attractor numbers' the model emits when it's wrong",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: first-digit distribution vs Benford
        fig, ax = plt.subplots(figsize=(11, 5.5))
        w = 0.4
        ax.bar(np.arange(1, 10) - w/2, fd_dist, w, color="#4c72b0",
               edgecolor="black", label="emit first digit")
        ax.bar(np.arange(1, 10) + w/2, benford, w, color="#2ca02c",
               edgecolor="black", label="Benford's law")
        ax.set_xticks(range(1, 10)); ax.set_xlabel("first digit")
        ax.set_ylabel("% of N")
        ax.set_title("First-digit distribution of wrong emits vs Benford's law",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: last-digit distribution
        fig, ax = plt.subplots(figsize=(11, 5.5))
        bars = ax.bar(range(10), ld_dist, color="#9467bd", edgecolor="black")
        for b, v in zip(bars, ld_dist):
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}",
                    ha="center", fontsize=9)
        ax.axhline(10, color="black", ls="--", alpha=0.5, label="uniform (10%)")
        ax.set_xticks(range(10)); ax.set_xlabel("last digit")
        ax.set_ylabel("% of N")
        ax.set_title("Last-digit distribution of wrong emits (uniform expectation = 10%)",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: digit count + roundness
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        dlist = sorted(set(list(emit_dig_dist.keys()) + list(gold_dig_dist.keys())))
        e_pcts = [emit_dig_dist.get(d, 0) / len(emit_digs) * 100 for d in dlist]
        g_pcts = [gold_dig_dist.get(d, 0) / len(gold_digs) * 100 for d in dlist]
        w = 0.4
        ax.bar(np.arange(len(dlist)) - w/2, e_pcts, w, color="#dd8452",
               edgecolor="black", label="wrong emit")
        ax.bar(np.arange(len(dlist)) + w/2, g_pcts, w, color="#2ca02c",
               edgecolor="black", label="gold")
        ax.set_xticks(range(len(dlist))); ax.set_xticklabels(dlist)
        ax.set_xlabel("# digits")
        ax.set_ylabel("% of N")
        ax.set_title("Digit-count distribution: wrong emits vs gold",
                     fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

        ax = axes[1]
        labels = ["mult of 5", "mult of 10", "mult of 100"]
        emit_pcts = [n_mult5/N*100, n_mult10/N*100, n_mult100/N*100]
        gold_pcts = [gold_mult5/N*100, gold_mult10/N*100,
                     int(np.sum(gold_vals % 100 == 0))/N*100]
        w = 0.4
        xs = np.arange(len(labels))
        ax.bar(xs - w/2, emit_pcts, w, color="#dd8452", edgecolor="black",
               label="wrong emit")
        ax.bar(xs + w/2, gold_pcts, w, color="#2ca02c", edgecolor="black",
               label="gold")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_ylabel("% of N")
        ax.set_title("Roundness (multiples) — wrong emits vs gold",
                     fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        for x, e, g in zip(xs, emit_pcts, gold_pcts):
            ax.text(x - w/2, e + 0.5, f"{e:.0f}%", ha="center", fontsize=8)
            ax.text(x + w/2, g + 0.5, f"{g:.0f}%", ha="center", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 5: text summary
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        ax.axis("off")
        body = (f"Wrong-emit value-level summary  (N={N})\n\n"
                f"Most-common attractor values:\n")
        for v, c in top_values[:10]:
            body += f"  {v!s:<8} {c}  ({c/N*100:.1f}%)\n"
        body += f"\nFirst-digit distribution:\n"
        body += f"  {'digit':<7} {'emit':<8} {'Benford':<8}\n"
        for d in range(1, 10):
            body += f"  {d}       {fd_dist[d-1]:5.1f}%  {benford[d-1]:5.1f}%\n"
        body += f"\nLast-digit distribution (uniform = 10%):\n"
        body += f"  {'digit':<7} {'pct':<8}\n"
        for d in range(10):
            mark = "  ← peak" if ld_dist[d] == ld_dist.max() else ""
            body += f"  {d}       {ld_dist[d]:5.1f}%{mark}\n"
        body += f"\nRoundness:\n"
        body += f"  mult of 5:   {n_mult5/N*100:.1f}% (gold {gold_mult5/N*100:.1f}%)\n"
        body += f"  mult of 10:  {n_mult10/N*100:.1f}% (gold {gold_mult10/N*100:.1f}%)\n"
        body += f"  mult of 100: {n_mult100/N*100:.1f}%\n"
        ax.text(0.04, 0.97, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Wrong-emit value-level summary",
                     fontsize=12, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
