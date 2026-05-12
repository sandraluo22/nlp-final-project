"""Decompose existing patch_paired_cf results by step to test the
'store-and-read pair' hypothesis (steps 2→3 and 4→5).

No new compute — pulls the 'step{k}_ALL' conditions from
patch_paired_cf_gsm8k.json (which patches ALL layers at a single latent step
k) and plots per-step transfer/broken rates.

If steps 2 and 4 are 'store' steps that step 3/5 read from, patching ONLY at
step 2 (or step 4) should produce larger transfer than patching at step 1, 3,
5, or 6. The flow_map showed step 3 attends to L2 at 13% and step 5 attends
to L4 at 12% — patching step 2's or step 4's output should propagate through.

Output: step_patch_decomposition_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
PP_JSON = PD / "correctness-analysis" / "patch_paired_cf_gsm8k.json"
OUT_JSON = PD / "step_patch_decomposition_gsm8k.json"
OUT_PDF = PD / "step_patch_decomposition_gsm8k.pdf"


def main():
    d = json.load(open(PP_JSON))
    results = {}
    for cf, sub in d["cf_sets"].items():
        N = sub["N"]
        N_LAT = d["N_LAT"]
        per_step = {"step": [], "source": [], "target": [], "other": []}
        for k in range(1, N_LAT + 1):
            key = f"step{k}_ALL"
            c = sub["conditions"].get(key)
            if c is None: continue
            per_step["step"].append(k)
            per_step["source"].append(c["n_followed_source"] / N)
            per_step["target"].append(c["n_followed_target"] / N)
            per_step["other"].append(c["n_other"] / N)
        results[cf] = {"N": N, "baseline_acc": sub["baseline_accuracy"], "per_step": per_step}
    OUT_JSON.write_text(json.dumps(results, indent=2))

    with PdfPages(OUT_PDF) as pdf:
        for cf, r in results.items():
            ps = r["per_step"]
            steps = ps["step"]
            fig, ax = plt.subplots(figsize=(11, 5.5))
            w = 0.27; xs = np.arange(len(steps))
            ax.bar(xs - w, [v * 100 for v in ps["source"]], w,
                   color="#2ca02c", edgecolor="black", label="followed source (transferred)")
            ax.bar(xs,     [v * 100 for v in ps["other"]],  w,
                   color="#d62728", edgecolor="black", label="other (broken)")
            ax.bar(xs + w, [v * 100 for v in ps["target"]], w,
                   color="#cccccc", edgecolor="black", label="followed target (no effect)")
            ax.set_xticks(xs); ax.set_xticklabels([f"step {s}" for s in steps])
            ax.set_xlabel("which latent step was paired-patched (all layers at that step)")
            ax.set_ylabel("% of N")
            ax.set_title(f"{cf} — whole-step paired patching  (N={r['N']}, base={r['baseline_acc']*100:.0f}%)",
                         fontsize=11, fontweight="bold")
            ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 105)
            for x, v in zip(xs - w, ps["source"]):
                ax.text(x, v*100 + 1, f"{v*100:.0f}", ha="center", fontsize=7)
            for x, v in zip(xs, ps["other"]):
                ax.text(x, v*100 + 1, f"{v*100:.0f}", ha="center", fontsize=7)
            for x, v in zip(xs + w, ps["target"]):
                ax.text(x, v*100 + 1, f"{v*100:.0f}", ha="center", fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Combined per-CF: source rate vs step
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for cf, r in results.items():
            ps = r["per_step"]
            ax.plot(ps["step"], [v * 100 for v in ps["source"]], "-o",
                    label=f"{cf} (base {r['baseline_acc']*100:.0f}%)", lw=2)
        ax.set_xlabel("latent step patched (all layers)")
        ax.set_ylabel("% transfer (followed source's answer)")
        ax.set_title("Whole-step paired-CF transfer rate per step  "
                     "(higher = step k carries the answer forward to emit)",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        # mark step-3 and step-5 with dashed verticals (the 'read' steps)
        for s_read in [3, 5]:
            ax.axvline(s_read, ls="--", color="gray", alpha=0.4)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_JSON} and {OUT_PDF}")


if __name__ == "__main__":
    main()
