"""Analyze CODI-correct problems by when they first became correct,
and what they were emitting at earlier (still-wrong) steps.

For each CORRECT-AT-STEP-6 problem, find the first step it became correct.
Then look at patterns:
  - distribution of first-correct-step (k=1..6)
  - chain length (n_markers) vs first-correct-step
  - for problems first-correct at step k>1, did the earlier emits hit any
    gold marker's intermediate value? Or were they totally off?
  - sample problems per first-correct-step bucket

Output: correct_trace_patterns_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
FD_JSON = REPO / "experiments" / "computation_probes" / "force_decode_per_step_gsm8k.json"
OUT_JSON = PD / "correct_trace_patterns_gsm8k.json"
OUT_PDF = PD / "correct_trace_patterns_gsm8k.pdf"


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def emit_final(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    fd = json.load(open(FD_JSON))
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]

    rows = fd["rows"]
    n_steps = len(rows[0]["step_emits"])

    correct_problems = []
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
        if not step_correct[-1]: continue  # only correct-at-step-6 problems
        first_correct = next((k for k, c in enumerate(step_correct) if c), -1)
        # Earlier emits' relation to gold markers
        early_emit_categories = []   # one entry per step before first_correct
        for k in range(first_correct):
            v = emit_vals[k]
            cat = "no_match"
            if v is None: cat = "unparsed"
            elif markers and any(abs(v - m[3]) < 1e-3 for m in markers):
                cat = "matches_marker"
            elif v == 0: cat = "zero"
            else: cat = "other_number"
            early_emit_categories.append(cat)
        correct_problems.append({
            "idx": idx, "q": ex["question"].strip().replace("  ", " "),
            "gold": gold, "n_markers": len(markers), "markers": markers,
            "first_correct_step": first_correct + 1,  # 1-indexed
            "step_emits": r["step_emits"],
            "emit_vals": emit_vals,
            "step_correct": step_correct,
            "early_emit_categories": early_emit_categories,
        })

    print(f"  total correct-at-step-6 problems: {len(correct_problems)}")

    # ===== Analysis 1: distribution of first-correct-step =====
    fcs_counter = Counter(p["first_correct_step"] for p in correct_problems)
    print(f"\nFirst-correct-step distribution (among correct problems):")
    for k in range(1, n_steps + 1):
        n = fcs_counter[k]
        print(f"  step {k}: {n}  ({n/len(correct_problems)*100:.1f}%)")

    # ===== Analysis 2: chain length vs first-correct-step =====
    by_nm = defaultdict(lambda: Counter())
    for p in correct_problems:
        by_nm[p["n_markers"]][p["first_correct_step"]] += 1
    print(f"\nChain length × first-correct-step:")
    print(f"  {'n_markers':<10} " + " ".join(f"k={k}" for k in range(1, n_steps + 1)))
    for nm in sorted(by_nm.keys()):
        row = [by_nm[nm][k] for k in range(1, n_steps + 1)]
        total = sum(row)
        print(f"  {nm:<10} {total:>4} total: " +
              " ".join(f"{c:>3}" for c in row))

    # ===== Analysis 3: for problems first-correct at step ≥ 2, what did earlier emits look like? =====
    by_first = defaultdict(lambda: Counter())
    for p in correct_problems:
        if p["first_correct_step"] < 2: continue
        for cat in p["early_emit_categories"]:
            by_first[p["first_correct_step"]][cat] += 1
    print(f"\nFor problems first-correct at step k≥2: distribution of EARLIER emit categories:")
    print(f"  {'first_k':<8} {'no_match':<10} {'matches_marker':<16} {'zero':<6} {'unparsed':<10} {'other_number':<14}")
    for k in sorted(by_first.keys()):
        c = by_first[k]
        total = sum(c.values())
        if total == 0: continue
        print(f"  k={k}      {c['no_match']:<10} {c['matches_marker']:<16} "
              f"{c['zero']:<6} {c['unparsed']:<10} {c['other_number']:<14}")

    # ===== Analysis 4: among 'matches_marker' early emits, WHICH marker was matched? =====
    marker_idx_hits = Counter()
    for p in correct_problems:
        if p["first_correct_step"] < 2: continue
        markers = p["markers"]
        for k in range(p["first_correct_step"] - 1):  # steps before first_correct
            v = p["emit_vals"][k]
            if v is None: continue
            for mi, m in enumerate(markers, 1):
                if abs(v - m[3]) < 1e-3 and abs(m[3] - p["gold"]) > 1e-3:
                    # An intermediate-marker hit, not the gold final
                    marker_idx_hits[mi] += 1
                    break
    print(f"\nWhen earlier emit matches a non-gold marker, which one?")
    for mi in sorted(marker_idx_hits.keys()):
        print(f"  marker {mi}: {marker_idx_hits[mi]}")

    # ===== Analysis 5: 'rescue at step 3' pattern — were they wrong at step 1 and 2? =====
    rescue_step3 = [p for p in correct_problems
                    if p["first_correct_step"] == 3 and not p["step_correct"][0] and not p["step_correct"][1]]
    print(f"\n'Rescue at step 3' (wrong at 1+2, right at 3): {len(rescue_step3)} problems")

    # Save full result
    out = {
        "n_correct_total": len(correct_problems),
        "first_correct_distribution": dict(fcs_counter),
        "by_nmarkers_x_firstk": {nm: dict(by_first) for nm, by_first in by_nm.items()},
        "early_emit_categories_by_firstk": {k: dict(c) for k, c in by_first.items()},
        "marker_idx_hits": dict(marker_idx_hits),
        "n_rescue_at_step3": len(rescue_step3),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # ===== Plots =====
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: text summary
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        ax.axis("off")
        body = (
            f"Patterns in {len(correct_problems)} CODI-correct GSM8K problems "
            f"(correct at step 6).\n\n"
            f"Distribution of FIRST-correct step:\n"
        )
        for k in range(1, n_steps + 1):
            n = fcs_counter[k]; pct = n / len(correct_problems) * 100
            bar = "█" * int(pct / 2)
            body += f"  step {k}: {n:>4}  ({pct:5.1f}%)  {bar}\n"
        body += f"\nMost-common patterns:\n"
        body += f"  - {fcs_counter[1]} ({fcs_counter[1]/len(correct_problems)*100:.0f}%) correct from step 1 (no recovery needed)\n"
        body += f"  - {len(rescue_step3)} ({len(rescue_step3)/len(correct_problems)*100:.0f}%) RESCUED AT STEP 3 (wrong at 1 AND 2)\n"
        body += f"\nWhen the model is wrong before becoming correct, what is it emitting?\n"
        body += f"  {'first_k':<8} {'matches_a_marker':<20} {'other_number':<15} {'no_match':<10} {'zero':<6}\n"
        for k in sorted(by_first.keys()):
            c = by_first[k]
            total = sum(c.values())
            if total == 0: continue
            body += (f"  k={k}      "
                     f"{c['matches_marker']} ({c['matches_marker']/total*100:.0f}%)        "
                     f"{c['other_number']} ({c['other_number']/total*100:.0f}%)    "
                     f"{c['no_match']} ({c['no_match']/total*100:.0f}%)   "
                     f"{c['zero']} ({c['zero']/total*100:.0f}%)\n")
        body += f"\nAmong intermediate-marker hits, marker index distribution:\n"
        for mi in sorted(marker_idx_hits.keys()):
            body += f"  marker {mi}: {marker_idx_hits[mi]} hits\n"
        ax.text(0.04, 0.97, body, va="top", ha="left",
                family="monospace", fontsize=10)
        ax.set_title("Patterns in correct traces — text summary",
                     fontsize=13, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: distribution of first-correct-step + chain length breakdown
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        ax = axes[0]
        ks = list(range(1, n_steps + 1))
        counts = [fcs_counter[k] for k in ks]
        ax.bar(ks, counts, color="#2ca02c", edgecolor="black")
        for k, c in zip(ks, counts):
            ax.text(k, c + 5, str(c), ha="center", fontsize=9)
        ax.set_xticks(ks); ax.set_xlabel("first step correct")
        ax.set_ylabel("# correct problems")
        ax.set_title(f"First-correct-step distribution\n(N={len(correct_problems)})",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1]
        nm_max = max(by_nm.keys())
        stack_data = np.zeros((nm_max, n_steps))
        for nm in range(1, nm_max + 1):
            for k in range(1, n_steps + 1):
                stack_data[nm - 1, k - 1] = by_nm[nm][k]
        bot = np.zeros(n_steps)
        colors = plt.cm.viridis(np.linspace(0, 1, nm_max))
        for nm in range(nm_max):
            ax.bar(ks, stack_data[nm], bottom=bot, color=colors[nm],
                   edgecolor="white", linewidth=0.3,
                   label=f"{nm+1} marker{'s' if nm > 0 else ''}")
            bot += stack_data[nm]
        ax.set_xticks(ks); ax.set_xlabel("first step correct")
        ax.set_ylabel("# correct problems")
        ax.set_title("First-correct-step × chain length (n_markers)",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, title="n_markers")
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Where in the loop does each correct problem first become correct?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: early-emit category by first-correct-step
        fig, ax = plt.subplots(figsize=(13, 5.5))
        cats = ["matches_marker", "other_number", "no_match", "zero", "unparsed"]
        cat_colors = {"matches_marker": "#dd8452", "other_number": "#4c72b0",
                       "no_match": "#7f7f7f", "zero": "#d62728", "unparsed": "#cccccc"}
        ks_with_data = sorted(by_first.keys())
        bot = np.zeros(len(ks_with_data))
        for cat in cats:
            vals = np.array([by_first[k][cat] for k in ks_with_data])
            ax.bar(ks_with_data, vals, bottom=bot, color=cat_colors[cat],
                   edgecolor="white", linewidth=0.3, label=cat)
            bot += vals
        ax.set_xticks(ks_with_data)
        ax.set_xlabel("step at which the problem first becomes correct")
        ax.set_ylabel("# (step, problem) entries among earlier wrong steps")
        ax.set_title("For problems first-correct at step k: what were the earlier wrong steps emitting?",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: sample problems per first-correct-step bucket
        for k in sorted(fcs_counter.keys()):
            if k == 1: continue  # skip "correct from start" (boring)
            samples = [p for p in correct_problems if p["first_correct_step"] == k][:4]
            if not samples: continue
            fig, ax = plt.subplots(figsize=(11.5, 8.5))
            ax.axis("off")
            body = f"Sample 'first-correct at step {k}' problems (4 of {fcs_counter[k]} total):\n\n"
            for p in samples:
                body += f"--- idx={p['idx']}  gold={p['gold']}  ({p['n_markers']} markers)\n"
                body += f"Q: {p['q'][:120]}{'...' if len(p['q'])>120 else ''}\n"
                body += f"Markers: " + " ".join(f"<<{m[0]:g}{m[1]}{m[2]:g}={m[3]:g}>>" for m in p['markers']) + "\n"
                body += "  Per-step emits:\n"
                for j in range(n_steps):
                    v = p["emit_vals"][j]
                    mk = "✓" if p["step_correct"][j] else "·"
                    body += f"    step{j+1} [{mk}] {v!s:<10} {p['step_emits'][j][:60]!r}\n"
                body += "\n"
            ax.text(0.02, 0.99, body, va="top", ha="left",
                    family="monospace", fontsize=8)
            ax.set_title(f"Sample problems: first correct at step {k}",
                         fontsize=12, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
