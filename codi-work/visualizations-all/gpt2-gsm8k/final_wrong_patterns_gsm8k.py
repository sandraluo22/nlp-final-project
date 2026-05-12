"""Value-level patterns in the FINAL wrong emits for the 768 GSM8K problems
CODI never gets right (wrong at step 6 force-decode).

Compare against:
  - the gold final answers (what should have been emitted)
  - the 'rescue cohort' wrong-before-right pattern we analyzed earlier
  - the question's mentioned operands

Patterns examined:
  - Most-common attractor values
  - First-digit distribution (Benford check)
  - Last-digit distribution (round-number bias)
  - Roundness multiples
  - Sign
  - Digit count vs gold
  - Distance to gold (|emit - gold| distribution)
  - Wrong emit equals an operand-from-question?
  - Wrong emit equals a pairwise op result?
  - Wrong emit equals a marker's c-value?

Output: final_wrong_patterns_gsm8k.{json,pdf}
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
OUT_JSON = PD / "final_wrong_patterns_gsm8k.json"
OUT_PDF = PD / "final_wrong_patterns_gsm8k.pdf"


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def emit_final(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def extract_question_numbers(q):
    s = q.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    out = []
    for n in nums:
        try: out.append(float(n))
        except: pass
    return out


def first_digit(x):
    if x == 0: return None
    s = str(abs(int(x))) if x == int(x) else str(abs(x)).split(".")[0]
    s = s.lstrip("0")
    return int(s[0]) if s else None


def last_digit(x):
    return int(abs(int(x))) % 10


def main():
    fd = json.load(open(FD_JSON))
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]

    rows = fd["rows"]
    wrong_finals = []  # list of dicts for problems wrong-at-step-6
    for r in rows:
        idx = r["idx"]
        ex = ds[idx]
        gold_match = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if gold_match is None: continue
        gold = float(gold_match.group(1))
        markers = [(float(a), op, float(b), float(c))
                   for a, op, b, c in parse_markers(ex["answer"])]
        emit_vals = [emit_final(e) for e in r["step_emits"]]
        final_emit = emit_vals[-1]
        if final_emit is not None and abs(final_emit - gold) < 1e-3:
            continue   # exclude correct
        wrong_finals.append({
            "idx": idx,
            "q": ex["question"].strip().replace("  ", " "),
            "gold": gold,
            "markers": markers,
            "n_markers": len(markers),
            "ops_in_q": extract_question_numbers(ex["question"]),
            "final_emit": final_emit,
            "final_emit_text": r["step_emits"][-1],
        })

    parseable = [p for p in wrong_finals if p["final_emit"] is not None]
    unparseable = len(wrong_finals) - len(parseable)
    print(f"  Wrong-at-step-6 problems: {len(wrong_finals)}  "
          f"({len(parseable)} parseable, {unparseable} unparseable)")

    vals = np.array([p["final_emit"] for p in parseable])
    golds = np.array([p["gold"] for p in parseable])
    N = len(vals)

    # ===== Top attractor values =====
    val_counter = Counter()
    for v in vals:
        if v == int(v):
            val_counter[int(v)] += 1
        else:
            val_counter[round(v, 2)] += 1
    top_values = val_counter.most_common(20)
    print(f"\nTop-15 wrong-final values:")
    for v, c in top_values[:15]:
        print(f"  {v:<10}{c:>4}  ({c/N*100:.1f}%)")

    # ===== First-digit (Benford) =====
    first_digits = [first_digit(v) for v in vals if first_digit(v) is not None]
    fd_counts = Counter(first_digits)
    fd_dist = np.array([fd_counts.get(d, 0) / len(first_digits) * 100
                        for d in range(1, 10)])
    benford = np.array([np.log10(1 + 1 / d) * 100 for d in range(1, 10)])

    # ===== Last-digit =====
    last_digits = [last_digit(v) for v in vals]
    ld_counts = Counter(last_digits)
    ld_dist = np.array([ld_counts.get(d, 0) / len(last_digits) * 100 for d in range(10)])

    # ===== Roundness =====
    n_mult5 = int(np.sum(vals % 5 == 0))
    n_mult10 = int(np.sum(vals % 10 == 0))
    n_mult100 = int(np.sum(vals % 100 == 0))
    gold_mult5 = int(np.sum(golds % 5 == 0))
    gold_mult10 = int(np.sum(golds % 10 == 0))

    # ===== Sign =====
    n_neg = int(np.sum(vals < 0))
    n_zero = int(np.sum(vals == 0))
    n_pos = int(np.sum(vals > 0))

    # ===== Distance to gold =====
    diffs = vals - golds
    abs_diffs = np.abs(diffs)
    rel_diffs = np.where(golds != 0, abs_diffs / np.abs(golds), np.nan)

    # ===== Relation to question / markers =====
    n_match_op = 0
    n_match_op_pair = 0
    n_match_marker = 0
    for p in parseable:
        v = p["final_emit"]
        ops = p["ops_in_q"]; markers = p["markers"]
        if any(abs(v - x) < 1e-3 for x in ops):
            n_match_op += 1
        # pair op
        for i in range(len(ops)):
            for j in range(len(ops)):
                if i == j: continue
                a, b = ops[i], ops[j]
                rs = [a + b, a - b, a * b]
                if b != 0: rs.append(a / b)
                if any(abs(v - r) < 1e-3 for r in rs):
                    n_match_op_pair += 1; break
            else: continue
            break
        if markers and any(abs(v - m[3]) < 1e-3 for m in markers):
            n_match_marker += 1

    print(f"\nFirst-digit distribution (vs Benford):")
    for d in range(1, 10):
        bar = "█" * int(fd_dist[d-1] / 2)
        print(f"  {d}: emit {fd_dist[d-1]:5.1f}%  benford {benford[d-1]:5.1f}%  {bar}")
    print(f"\nLast-digit distribution:")
    for d in range(10):
        bar = "█" * int(ld_dist[d] / 2)
        print(f"  {d}: {ld_dist[d]:5.1f}%  {bar}")
    print(f"\nRoundness: mult5={n_mult5/N*100:.1f}%  mult10={n_mult10/N*100:.1f}%  "
          f"mult100={n_mult100/N*100:.1f}%  (gold: mult5={gold_mult5/N*100:.1f}%  "
          f"mult10={gold_mult10/N*100:.1f}%)")
    print(f"Sign: {n_neg/N*100:.1f}% neg, {n_zero/N*100:.1f}% zero, {n_pos/N*100:.1f}% pos")
    print(f"Distance to gold:  median={np.median(abs_diffs):g}, "
          f"mean={np.mean(abs_diffs):g}, max={abs_diffs.max():g}")
    print(f"Relative distance:  median={np.nanmedian(rel_diffs):g}, "
          f"mean={np.nanmean(rel_diffs):g}")
    print(f"Match question content:")
    print(f"  = a question operand: {n_match_op} ({n_match_op/N*100:.1f}%)")
    print(f"  = a pair-op of two operands: {n_match_op_pair} ({n_match_op_pair/N*100:.1f}%)")
    print(f"  = an intermediate marker value: {n_match_marker} ({n_match_marker/N*100:.1f}%)")

    out = {
        "N_wrong_problems": len(wrong_finals),
        "N_parseable_finals": N,
        "N_unparseable_finals": unparseable,
        "top_values": [{"value": float(v), "count": c, "pct": c/N*100}
                       for v, c in top_values],
        "first_digit_dist_pct": fd_dist.tolist(),
        "benford_pct": benford.tolist(),
        "last_digit_dist_pct": ld_dist.tolist(),
        "roundness": {"mult5_pct": n_mult5/N*100, "mult10_pct": n_mult10/N*100,
                       "mult100_pct": n_mult100/N*100,
                       "gold_mult5_pct": gold_mult5/N*100,
                       "gold_mult10_pct": gold_mult10/N*100},
        "sign": {"neg_pct": n_neg/N*100, "zero_pct": n_zero/N*100,
                  "pos_pct": n_pos/N*100},
        "distance_to_gold": {"median": float(np.median(abs_diffs)),
                              "mean": float(np.mean(abs_diffs)),
                              "max": float(abs_diffs.max()),
                              "median_rel": float(np.nanmedian(rel_diffs))},
        "match_question_content": {"operand_pct": n_match_op/N*100,
                                    "op_pair_pct": n_match_op_pair/N*100,
                                    "marker_pct": n_match_marker/N*100},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # Plots
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: top values
        fig, ax = plt.subplots(figsize=(13, 6))
        top = top_values[:25]
        labels = [str(v) for v, _ in top]
        cnts = [c for _, c in top]
        ax.bar(range(len(labels)), cnts, color="#d62728", edgecolor="black")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("# wrong final emits with this value")
        for i, c in enumerate(cnts):
            ax.text(i, c + 0.3, str(c), ha="center", fontsize=8)
        ax.set_title(f"Most-common WRONG FINAL answers  (N={N} of {len(wrong_finals)} wrong problems)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: first-digit vs Benford
        fig, ax = plt.subplots(figsize=(11, 5.5))
        w = 0.4
        ax.bar(np.arange(1, 10) - w/2, fd_dist, w, color="#d62728",
               edgecolor="black", label="wrong final emit")
        ax.bar(np.arange(1, 10) + w/2, benford, w, color="#2ca02c",
               edgecolor="black", label="Benford's law")
        ax.set_xticks(range(1, 10)); ax.set_xlabel("first digit")
        ax.set_ylabel("% of N")
        ax.set_title("First-digit distribution of WRONG FINAL emits vs Benford",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: last-digit
        fig, ax = plt.subplots(figsize=(11, 5.5))
        bars = ax.bar(range(10), ld_dist, color="#9467bd", edgecolor="black")
        for b, v in zip(bars, ld_dist):
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}",
                    ha="center", fontsize=9)
        ax.axhline(10, color="black", ls="--", alpha=0.5, label="uniform (10%)")
        ax.set_xticks(range(10)); ax.set_xlabel("last digit")
        ax.set_ylabel("% of N")
        ax.set_title("Last-digit distribution of WRONG FINAL emits",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: distance to gold histogram (capped)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        # absolute diff capped at percentile
        cap = float(np.percentile(abs_diffs, 95))
        ad_clip = np.clip(abs_diffs, 0, cap)
        ax.hist(ad_clip, bins=50, color="#d62728", edgecolor="black")
        ax.axvline(np.median(abs_diffs), color="black", ls="--",
                   label=f"median = {np.median(abs_diffs):g}")
        ax.set_xlabel(f"|wrong emit − gold|  (clipped at p95={cap:g})")
        ax.set_ylabel("# problems")
        ax.set_title("Absolute distance from gold", fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1]
        # relative diff capped at 5
        rd_clip = np.clip(rel_diffs[~np.isnan(rel_diffs)], 0, 5)
        ax.hist(rd_clip, bins=40, color="#dd8452", edgecolor="black")
        ax.axvline(np.nanmedian(rel_diffs), color="black", ls="--",
                   label=f"median = {np.nanmedian(rel_diffs):.2f}")
        ax.set_xlabel("|wrong emit − gold| / |gold|  (clipped at 5)")
        ax.set_ylabel("# problems")
        ax.set_title("Relative distance from gold", fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 5: question-content matches + roundness
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        labels = ["= operand", "= pair-op", "= a marker"]
        pcts = [n_match_op/N*100, n_match_op_pair/N*100, n_match_marker/N*100]
        ax.bar(labels, pcts, color=["#4c72b0", "#dd8452", "#2ca02c"],
               edgecolor="black")
        for i, p in enumerate(pcts):
            ax.text(i, p + 0.5, f"{p:.1f}%", ha="center", fontsize=10)
        ax.set_ylabel("% of N")
        ax.set_title("Wrong final emit matches…?", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax = axes[1]
        labels = ["mult 5", "mult 10", "mult 100"]
        e_pcts = [n_mult5/N*100, n_mult10/N*100, n_mult100/N*100]
        g_pcts = [gold_mult5/N*100, gold_mult10/N*100,
                   int(np.sum(golds % 100 == 0))/N*100]
        w = 0.4; xs = np.arange(len(labels))
        ax.bar(xs - w/2, e_pcts, w, color="#d62728", edgecolor="black", label="wrong emit")
        ax.bar(xs + w/2, g_pcts, w, color="#2ca02c", edgecolor="black", label="gold")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_ylabel("% of N")
        ax.set_title("Roundness comparison", fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 6: text summary
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        ax.axis("off")
        body = (f"Wrong-final-answer summary  (N={N} of {len(wrong_finals)} wrong-at-step-6 problems)\n\n"
                f"Top-12 most-common wrong values:\n")
        for v, c in top_values[:12]:
            body += f"  {v!s:<10} {c:>4}  ({c/N*100:.1f}%)\n"
        body += f"\nFirst-digit distribution (~ Benford? yes):\n"
        body += "  " + "  ".join(f"{d}:{fd_dist[d-1]:.1f}%" for d in range(1, 10)) + "\n"
        body += f"\nLast-digit distribution (uniform = 10%):\n"
        body += "  " + "  ".join(f"{d}:{ld_dist[d]:.1f}%" for d in range(10)) + "\n"
        body += f"\nRoundness:  mult5 = {n_mult5/N*100:.1f}%  mult10 = {n_mult10/N*100:.1f}%  "
        body += f"mult100 = {n_mult100/N*100:.1f}%\n"
        body += f"\nSign: {n_pos/N*100:.1f}% positive, {n_neg/N*100:.1f}% negative, {n_zero/N*100:.1f}% zero\n"
        body += f"\nDistance to gold: median {np.median(abs_diffs):g}, "
        body += f"mean {np.mean(abs_diffs):g}, median-relative {np.nanmedian(rel_diffs):.2f}x\n"
        body += f"\nWrong emit matches…\n"
        body += f"  a question operand:    {n_match_op}  ({n_match_op/N*100:.1f}%)\n"
        body += f"  a pair-op of operands: {n_match_op_pair}  ({n_match_op_pair/N*100:.1f}%)\n"
        body += f"  an intermediate marker:{n_match_marker}  ({n_match_marker/N*100:.1f}%)\n"
        ax.text(0.04, 0.97, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Wrong-final-answer summary",
                     fontsize=12, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
