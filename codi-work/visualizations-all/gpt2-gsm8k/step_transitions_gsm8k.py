"""Per-example correctness transitions across consecutive force-decode steps.

For each transition k → k+1 (k = 1..5):
  w→r : example was WRONG at step k, becomes RIGHT at step k+1 (gained)
  r→w : example was RIGHT at step k, becomes WRONG at step k+1 (lost)
  r→r : stayed right
  w→w : stayed wrong
  net = w→r − r→w   (the net accuracy change)

Built from force_decode_per_step_gsm8k.json which has per-example
correctness per step (correct_per_step: list of N_LAT lists, each length N).

Output: step_transitions_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
IN_JSON = REPO / "experiments" / "computation_probes" / "force_decode_per_step_gsm8k.json"
OUT_JSON = PD / "step_transitions_gsm8k.json"
OUT_PDF = PD / "step_transitions_gsm8k.pdf"


def main():
    d = json.load(open(IN_JSON))
    cps = d["correct_per_step"]   # list of 6 lists, each length N
    N = d["N"]
    n_steps = len(cps)
    correct = np.array(cps, dtype=bool)  # (6, N)

    results = []
    for k in range(n_steps - 1):
        a = correct[k]
        b = correct[k + 1]
        wr = int(((~a) & b).sum())
        rw = int((a & (~b)).sum())
        rr = int((a & b).sum())
        ww = int(((~a) & (~b)).sum())
        results.append({
            "from": k + 1, "to": k + 2,
            "wr": wr, "rw": rw, "rr": rr, "ww": ww,
            "net": wr - rw,
            "acc_from": int(a.sum()), "acc_to": int(b.sum()),
        })

    OUT_JSON.write_text(json.dumps({"N": N, "transitions": results}, indent=2))

    # Plot
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: stacked bars of all 4 categories per transition
        fig, ax = plt.subplots(figsize=(13, 5.5))
        labels = [f"{r['from']}→{r['to']}" for r in results]
        xs = np.arange(len(labels))
        rrs = np.array([r["rr"] for r in results]) / N * 100
        wrs = np.array([r["wr"] for r in results]) / N * 100
        rws = np.array([r["rw"] for r in results]) / N * 100
        wws = np.array([r["ww"] for r in results]) / N * 100
        ax.bar(xs, rrs,         color="#2ca02c", label="r→r (stayed right)")
        ax.bar(xs, wrs, bottom=rrs,                       color="#a8d8a8", label="w→r (recovered)")
        ax.bar(xs, rws, bottom=rrs + wrs,                 color="#f4b6b6", label="r→w (regressed)")
        ax.bar(xs, wws, bottom=rrs + wrs + rws,           color="#d62728", label="w→w (stayed wrong)")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("transition (step k → step k+1)")
        ax.set_ylabel("% of N")
        ax.set_title("Per-example correctness transitions across consecutive force-decode steps  "
                     f"(N={N})",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 105)
        for x, r in zip(xs, results):
            ax.text(x, 102, f"net={r['net']:+d}", ha="center", fontsize=9, fontweight="bold")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: focused w→r and r→w side-by-side
        fig, ax = plt.subplots(figsize=(13, 5.5))
        w = 0.35
        ax.bar(xs - w/2, [r["wr"] for r in results], w,
               color="#2ca02c", edgecolor="black", label="w→r (recovered)")
        ax.bar(xs + w/2, [r["rw"] for r in results], w,
               color="#d62728", edgecolor="black", label="r→w (regressed)")
        for i, r in enumerate(results):
            ax.text(i - w/2, r["wr"] + 5, str(r["wr"]),
                    ha="center", fontsize=9, color="#2ca02c", fontweight="bold")
            ax.text(i + w/2, r["rw"] + 5, str(r["rw"]),
                    ha="center", fontsize=9, color="#d62728", fontweight="bold")
            ax.text(i, max(r["wr"], r["rw"]) + 60, f"net={r['net']:+d}",
                    ha="center", fontsize=10, fontweight="bold",
                    color="#2ca02c" if r["net"] > 0 else "#d62728")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("transition")
        ax.set_ylabel("# of examples")
        ax.set_title("Recovered (w→r) vs regressed (r→w) per consecutive-step transition\n"
                     "Big positive net = a 'compute-and-propagate' step pays off",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: cumulative correctness trajectory
        fig, ax = plt.subplots(figsize=(11, 5))
        accs = [int(c.sum()) / N * 100 for c in correct]
        xs2 = np.arange(1, n_steps + 1)
        ax.bar(xs2, accs, color="#4c72b0", edgecolor="black")
        for s, a in zip(xs2, accs):
            ax.text(s, a + 0.3, f"{a:.1f}%", ha="center", fontsize=9)
        ax.set_xticks(xs2); ax.set_xlabel("force-decode at step k")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Cumulative: force-decode accuracy at each latent step",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_JSON} and {OUT_PDF}")
    print()
    print(f"{'transition':<12} {'w→r':<6} {'r→w':<6} {'r→r':<6} {'w→w':<6} {'net':<6}")
    for r in results:
        print(f"  {r['from']}→{r['to']:<8}  {r['wr']:<6} {r['rw']:<6} {r['rr']:<6} {r['ww']:<6} {r['net']:+d}")


if __name__ == "__main__":
    main()
