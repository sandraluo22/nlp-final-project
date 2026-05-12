"""Among Mode 1 (loop-rescued) correct problems, does the emit pass through
markers in chain order?

Mode 1 = correct at step 6 AND first-correct at step ≥ 2 (the loop did work).
For each such problem with ≥ 2 markers:
  - For each gold marker m (1..n), find the FIRST step where emit final
    equals marker m's gold c-value (or never).
  - Test: is first_step(m=1) ≤ first_step(m=2) ≤ ... ≤ first_step(m=n)?
    (chaining order preserved)
  - Or are markers hit out of order, or skipped?

Compare to a null: if emit values were random over the steps, what fraction
would have monotone ordering by chance?

Output: mode1_chaining_gsm8k.{json,pdf}
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
OUT_JSON = PD / "mode1_chaining_gsm8k.json"
OUT_PDF = PD / "mode1_chaining_gsm8k.pdf"


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
    enriched = []
    for r in rows:
        idx = r["idx"]
        ex = ds[idx]
        gm = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if gm is None: continue
        gold = float(gm.group(1))
        markers = [(float(a), op, float(b), float(c))
                    for a, op, b, c in parse_markers(ex["answer"])]
        if len(markers) < 2: continue   # need ≥2 markers for chaining
        emit_vals = [emit_final(e) for e in r["step_emits"]]
        step_correct = [v is not None and abs(v - gold) < 1e-3 for v in emit_vals]
        if not step_correct[-1]: continue  # only correct
        fc = next((k+1 for k, c in enumerate(step_correct) if c), -1)
        if fc < 2: continue   # only Mode 1 (rescued)
        enriched.append({
            "idx": idx, "gold": gold, "n_markers": len(markers),
            "markers": markers, "emit_vals": emit_vals,
            "first_correct": fc, "step_emits": r["step_emits"],
        })
    print(f"  Mode 1 problems (loop-rescued, ≥2 markers): {len(enriched)}")

    # For each problem, find first-step each marker's c appears in emit_final
    # AND also find which step the FINAL emit appears.
    n_steps = 6
    per_problem = []
    for p in enriched:
        markers = p["markers"]
        emit_vals = p["emit_vals"]
        # For each marker, first step where emit final == marker_c
        first_step_per_marker = []
        for m in markers:
            c = m[3]
            fs = -1
            for k, v in enumerate(emit_vals):
                if v is not None and abs(v - c) < 1e-3:
                    fs = k + 1; break
            first_step_per_marker.append(fs)
        per_problem.append({
            "idx": p["idx"], "n_markers": p["n_markers"],
            "first_correct": p["first_correct"],
            "first_step_per_marker": first_step_per_marker,
            "gold": p["gold"], "markers": markers,
        })

    # ===== Analysis 1: how many markers' values appear in emits at all? =====
    n_hit_count = Counter()
    for p in per_problem:
        n_hit = sum(1 for x in p["first_step_per_marker"] if x > 0)
        n_hit_count[n_hit] += 1
    print(f"\nA1: how many markers' gold-c values appear in emit at any step?")
    total = len(per_problem)
    for k in sorted(n_hit_count.keys()):
        n = n_hit_count[k]
        print(f"  {k} markers hit: {n} ({n/total*100:.1f}%)")

    # ===== Analysis 2: among problems where ALL markers' c appear, is order preserved? =====
    fully_traced = [p for p in per_problem if all(x > 0 for x in p["first_step_per_marker"])]
    print(f"\nA2: among {len(fully_traced)} problems with ALL markers' c-values hit:")
    monotone = []
    weakly_monotone = []   # ties allowed
    for p in fully_traced:
        steps = p["first_step_per_marker"]
        is_monotone = all(steps[i] < steps[i+1] for i in range(len(steps)-1))
        is_weak_mono = all(steps[i] <= steps[i+1] for i in range(len(steps)-1))
        if is_monotone: monotone.append(p)
        if is_weak_mono: weakly_monotone.append(p)
    print(f"  strictly monotone (m1 first-step < m2 first-step < ...): "
          f"{len(monotone)}/{len(fully_traced)} ({len(monotone)/max(1,len(fully_traced))*100:.1f}%)")
    print(f"  weakly monotone (≤): {len(weakly_monotone)}/{len(fully_traced)} "
          f"({len(weakly_monotone)/max(1,len(fully_traced))*100:.1f}%)")

    # Stratify by chain length
    by_nm = Counter()
    for p in fully_traced:
        by_nm[p["n_markers"]] += 1
    print(f"\n  Fully-traced count by chain length:")
    for nm in sorted(by_nm):
        n = by_nm[nm]
        # of these, how many monotone?
        nm_subset = [p for p in fully_traced if p["n_markers"] == nm]
        nm_mono = sum(1 for p in nm_subset
                       if all(p["first_step_per_marker"][i] < p["first_step_per_marker"][i+1]
                              for i in range(len(p["first_step_per_marker"])-1)))
        print(f"    {nm} markers: {n} fully-traced, {nm_mono} strictly monotone "
              f"({nm_mono/max(1,n)*100:.0f}%)")

    # ===== Analysis 3: average first-step per marker position =====
    # For problems with at least m markers, what's the average first-step
    # for marker m (when it appears at all)?
    avg_first_step_per_m = {}
    n_problems_per_m = {}
    for m_pos in range(1, 5):
        firsts = [p["first_step_per_marker"][m_pos - 1]
                  for p in per_problem
                  if p["n_markers"] >= m_pos and p["first_step_per_marker"][m_pos - 1] > 0]
        if firsts:
            avg_first_step_per_m[m_pos] = float(np.mean(firsts))
            n_problems_per_m[m_pos] = len(firsts)
    print(f"\nA3: average first-step at which marker m's c-value first appears:")
    for m_pos in sorted(avg_first_step_per_m):
        avg = avg_first_step_per_m[m_pos]
        n = n_problems_per_m[m_pos]
        print(f"  m={m_pos}: avg first-step = {avg:.2f}  (n={n} problems)")

    # ===== Analysis 4: detailed pattern for fully-traced 2- and 3-marker problems =====
    samples_2 = [p for p in fully_traced if p["n_markers"] == 2][:6]
    samples_3 = [p for p in fully_traced if p["n_markers"] == 3][:6]

    out = {
        "n_mode1_problems": len(enriched),
        "n_hit_count": dict(n_hit_count),
        "n_fully_traced": len(fully_traced),
        "n_strictly_monotone": len(monotone),
        "n_weakly_monotone": len(weakly_monotone),
        "pct_monotone_among_fully_traced": len(monotone) / max(1, len(fully_traced)) * 100,
        "by_nm_fully_traced": dict(by_nm),
        "avg_first_step_per_m": avg_first_step_per_m,
        "n_problems_per_m": n_problems_per_m,
        "samples_2_marker": [
            {"idx": p["idx"], "first_steps": p["first_step_per_marker"],
             "marker_cs": [m[3] for m in p["markers"]],
             "gold": p["gold"]} for p in samples_2],
        "samples_3_marker": [
            {"idx": p["idx"], "first_steps": p["first_step_per_marker"],
             "marker_cs": [m[3] for m in p["markers"]],
             "gold": p["gold"]} for p in samples_3],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # ===== Plots =====
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: hit count distribution
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ks = sorted(n_hit_count.keys())
        vals = [n_hit_count[k] for k in ks]
        ax.bar(ks, vals, color="#4c72b0", edgecolor="black")
        for k, v in zip(ks, vals):
            ax.text(k, v + 0.5, f"{v}\n({v/total*100:.0f}%)",
                    ha="center", fontsize=9)
        ax.set_xlabel("# of gold markers whose c-value appears in any emit")
        ax.set_ylabel("# Mode 1 problems")
        ax.set_xticks(ks)
        ax.set_title("How many markers leave a fingerprint in emits?\n"
                     f"(Mode 1, ≥2 markers, N={total})",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax = axes[1]
        labels = ["strictly\nmonotone\n(m1<m2<m3<...)", "weakly\nmonotone\n(m1≤m2≤...)"]
        ax_vals = [len(monotone), len(weakly_monotone)]
        ax_pcts = [v / max(1, len(fully_traced)) * 100 for v in ax_vals]
        bars = ax.bar(labels, ax_pcts, color="#2ca02c", edgecolor="black")
        for b, n, p in zip(bars, ax_vals, ax_pcts):
            ax.text(b.get_x() + b.get_width()/2, p + 1.5,
                    f"{n}\n({p:.1f}%)", ha="center", fontsize=9)
        ax.set_ylabel(f"% of {len(fully_traced)} fully-traced problems")
        ax.set_title("Is the emit order chained (m1 → m2 → m3)?",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(ax_pcts) * 1.3 if ax_pcts else 100)
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Chaining order in Mode 1 (loop-rescued) correct problems",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: average first-step per marker position
        fig, ax = plt.subplots(figsize=(11, 5))
        ms = sorted(avg_first_step_per_m.keys())
        avgs = [avg_first_step_per_m[m] for m in ms]
        ns = [n_problems_per_m[m] for m in ms]
        bars = ax.bar(ms, avgs, color="#dd8452", edgecolor="black")
        for b, m, a, n in zip(bars, ms, avgs, ns):
            ax.text(b.get_x() + b.get_width()/2, a + 0.05,
                    f"{a:.2f}\n(n={n})", ha="center", fontsize=9)
        # Reference: if chaining were perfect, m=1 first at step ≈1, m=2 at step ≈3, m=3 at step ≈5
        for m in ms:
            ax.scatter(m, 2*m - 1, marker="*", s=200, color="black",
                       zorder=5, label="predicted: 2m−1" if m == 1 else None)
        ax.set_xlabel("marker position m")
        ax.set_ylabel("average first-step where marker m's c appears in emit")
        ax.set_xticks(ms)
        ax.set_title("Average first-emit step per marker (among problems that hit it)\n"
                     "Stars: '2m−1' prediction (sequential chaining)",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: sample 2-marker and 3-marker problems
        for samples, title in [(samples_2, "2-marker"), (samples_3, "3-marker")]:
            if not samples: continue
            fig, ax = plt.subplots(figsize=(11.5, 6.5))
            ax.axis("off")
            body = f"Sample fully-traced {title} problems:\n\n"
            for s in samples:
                body += f"  idx={s['idx']}  gold={s['gold']}\n"
                cs = [m[3] for m in s["markers"]]
                fs = s["first_step_per_marker"]
                body += "  Markers: " + " → ".join(
                    f"m{i+1}=c{c:g}@step{fs[i]}" if fs[i] > 0 else f"m{i+1}=c{c:g}@never"
                    for i, c in enumerate(cs)) + "\n\n"
            ax.text(0.02, 0.97, body, va="top", ha="left",
                    family="monospace", fontsize=10)
            ax.set_title(f"Sample {title} Mode 1 problems", fontsize=12, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: text summary
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        ax.axis("off")
        body = (
            f"Chaining order among {len(enriched)} Mode 1 (loop-rescued) correct problems.\n\n"
            f"A1: How many markers' c-values appear in any emit?\n"
        )
        for k in sorted(n_hit_count.keys()):
            n = n_hit_count[k]
            body += f"  {k} markers traced: {n} ({n/total*100:.1f}%)\n"
        body += (f"\nA2: Among {len(fully_traced)} problems with ALL markers traced:\n"
                 f"  strictly monotone: {len(monotone)} ({len(monotone)/max(1,len(fully_traced))*100:.1f}%)\n"
                 f"  weakly monotone:   {len(weakly_monotone)} "
                 f"({len(weakly_monotone)/max(1,len(fully_traced))*100:.1f}%)\n\n"
                 f"A3: Average first-step per marker position (when it appears at all):\n")
        for m_pos in sorted(avg_first_step_per_m):
            avg = avg_first_step_per_m[m_pos]; n = n_problems_per_m[m_pos]
            body += f"  m={m_pos}: avg = {avg:.2f}  (n={n})\n"
        body += ("\nInterpretation:\n"
                 "  - If chaining were perfect, ALL markers should appear in order:\n"
                 "      m1 at low step, m2 at higher step, etc.\n"
                 "  - And the average first-step should grow with marker position.\n"
                 "  - The data tells us whether the model truly walks the chain\n"
                 "    or just lands on the final answer (skipping intermediates).\n")
        ax.text(0.04, 0.97, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Mode 1 chaining-order summary",
                     fontsize=12, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
