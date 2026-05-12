"""Per-marker emergence test: at each force-decode step, what fraction of
problems' emitted text contains marker m's gold result number?

If the model computes the chain sequentially m=1, m=2, m=3, then:
  - marker 1's gold result should first appear in many emits around step 3
  - marker 2's gold result around step 5
  - marker 3's gold result around step 6+
  - marker k's gold result should NEVER appear at steps before its 'completion'

Inputs:
  experiments/computation_probes/force_decode_per_step_gsm8k.json
    (has step_emits per problem)
  gold marker chains from datasets.load_dataset("gsm8k","main")["test"]

Output: chain_emergence_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
IN_JSON = REPO / "experiments" / "computation_probes" / "force_decode_per_step_gsm8k.json"
OUT_JSON = PD / "chain_emergence_gsm8k.json"
OUT_PDF = PD / "chain_emergence_gsm8k.pdf"


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def number_appears_in_text(num, text):
    """Whether num appears as an exact-token number in the text (any context)."""
    if num is None: return False
    # match either integer or with optional .0
    s_int = str(int(num)) if num == int(num) else None
    s_float = str(num)
    nums_in_text = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if s_int is not None and any(abs(float(x) - num) < 1e-3 for x in nums_in_text):
        return True
    return any(abs(float(x) - num) < 1e-3 for x in nums_in_text)


def main():
    d = json.load(open(IN_JSON))
    rows = d["rows"]   # list of dicts with idx, gold, baseline_emit, step_emits
    N = d["N"]
    n_steps = len(rows[0]["step_emits"])

    # Parse gold markers from baseline_emit fallback OR from each row's gold
    # Actually we need GSM8K test answers. Load via datasets.
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    gold_markers = []
    for ex in ds:
        gms = parse_markers(ex["answer"])
        gold_markers.append([(float(a), op, float(b), float(c)) for a, op, b, c in gms])

    # STRICT TEST: at each step k, the model's EMITTED FINAL ANSWER (the last
    # number in the emit) equals gold marker m's result value. This says
    # "the model treated marker m as the final answer at step k" — i.e., it
    # has committed to marker m's value but isn't going further.
    M_MAX = 4
    presence = np.zeros((n_steps, M_MAX), dtype=np.float64)
    cnt_per_m = np.zeros(M_MAX, dtype=np.int64)

    def emit_final(s):
        s = s.replace(",", "")
        nums = re.findall(r"-?\d+\.?\d*", s)
        if not nums: return None
        try: return float(nums[-1])
        except: return None

    for r in rows:
        idx = r["idx"]
        gms = gold_markers[idx]
        if not gms: continue
        for m in range(1, min(len(gms), M_MAX) + 1):
            gold_c = gms[m - 1][3]
            cnt_per_m[m - 1] += 1
            for k in range(n_steps):
                emit_v = emit_final(r["step_emits"][k])
                if emit_v is not None and abs(emit_v - gold_c) < 1e-3:
                    presence[k, m - 1] += 1

    frac = presence / np.where(cnt_per_m == 0, 1, cnt_per_m)

    # Also: first step at which the emit's final-answer equals marker m's gold.
    first_step = np.full((N, M_MAX), -1, dtype=int)
    for r in rows:
        idx = r["idx"]
        gms = gold_markers[idx]
        if not gms: continue
        for m in range(1, min(len(gms), M_MAX) + 1):
            gold_c = gms[m - 1][3]
            for k in range(n_steps):
                emit_v = emit_final(r["step_emits"][k])
                if emit_v is not None and abs(emit_v - gold_c) < 1e-3:
                    first_step[idx, m - 1] = k + 1
                    break

    first_step_dist = {m: Counter(first_step[:, m - 1].tolist()) for m in range(1, M_MAX + 1)}

    out = {
        "N": int(N),
        "n_steps": int(n_steps),
        "cnt_per_m": cnt_per_m.tolist(),
        "frac_present_per_step_marker": frac.tolist(),
        "first_step_distribution": {str(m): dict(first_step_dist[m]) for m in range(1, M_MAX + 1)},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    # Plot
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: per-marker fraction-present-in-emit per step
        fig, ax = plt.subplots(figsize=(12, 5.5))
        steps_x = np.arange(1, n_steps + 1)
        for m in range(1, M_MAX + 1):
            ys = [frac[k, m - 1] * 100 for k in range(n_steps)]
            n_m = cnt_per_m[m - 1]
            ax.plot(steps_x, ys, "-o", lw=2,
                    label=f"m={m} (N={n_m})", color=plt.cm.tab10(m / 5))
        ax.set_xlabel("force-decode step")
        ax.set_ylabel("% of problems whose emit contains marker m's gold result")
        ax.set_xticks(steps_x)
        ax.set_title("Per-marker gold-result emergence in force-decoded emits\n"
                     "(if marker m completes at step 2m+1, expect a jump there)",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        # Vertical lines at predicted completion steps 2m+1
        for m in range(1, M_MAX + 1):
            predicted = min(n_steps, 2 * m - 1)
            ax.axvline(predicted, ls="--", lw=0.5, alpha=0.4,
                       color=plt.cm.tab10(m / 5))
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: first-step distribution per marker
        fig, axes = plt.subplots(1, M_MAX, figsize=(4 * M_MAX, 5), sharey=True)
        for col, m in enumerate(range(1, M_MAX + 1)):
            ax = axes[col]
            dist = first_step_dist[m]
            xs_ = list(range(1, n_steps + 1)) + [n_steps + 1]
            heights = [dist.get(k, 0) for k in range(1, n_steps + 1)] + [dist.get(-1, 0)]
            colors_ = ["#4c72b0"] * n_steps + ["#d62728"]
            ax.bar(range(len(xs_)), heights, color=colors_, edgecolor="black")
            ax.set_xticks(range(len(xs_)))
            ax.set_xticklabels([str(k) for k in range(1, n_steps + 1)] + ["never"],
                               fontsize=8)
            ax.set_xlabel("first step marker m appeared in emit")
            if col == 0: ax.set_ylabel("# problems")
            ax.set_title(f"m={m} (N applicable={cnt_per_m[m-1]})",
                         fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            for i, h in enumerate(heights):
                if h > 0:
                    ax.text(i, h + 5, str(h), ha="center", fontsize=7)
        fig.suptitle("First force-decode step at which marker m's gold result appears in emit",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")
    print()
    print("Fraction of problems where marker m's gold result appears in emit:")
    print(f"  {'step':<5} m=1     m=2     m=3     m=4")
    for k in range(n_steps):
        print(f"  {k+1}    {frac[k, 0]*100:5.1f}%  {frac[k, 1]*100:5.1f}%  {frac[k, 2]*100:5.1f}%  {frac[k, 3]*100:5.1f}%")


if __name__ == "__main__":
    main()
