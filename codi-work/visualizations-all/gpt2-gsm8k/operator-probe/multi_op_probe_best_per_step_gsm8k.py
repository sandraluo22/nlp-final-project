"""Best-decodability-per-step trajectory.

Re-analyzes multi_op_probe_gsm8k.json (no new compute) to answer:
  - At each latent step k, what's the BEST any-layer accuracy at predicting
    each marker m's label? (max_l acc[k, l])
  - When does marker m FIRST become decodable somewhere (any layer up to and
    including step k)? cum_max_l_j acc[j, l] for j ≤ k.
  - Where (which layer) is marker m best read at each step?

This is a cleaner trajectory than restricting to the best layer overall —
it lets the best layer move with the step, which is more honest to the data.

Outputs: multi_op_probe_best_per_step_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
IN_JSON = PD / "multi_op_probe_gsm8k.json"
OUT_JSON = PD / "multi_op_probe_best_per_step_gsm8k.json"
OUT_PDF = PD / "multi_op_probe_best_per_step_gsm8k.pdf"

PROBES = ["op", "a_ld", "c_ld"]
M_MAX = 4
CHANCE = {"op": 0.25, "a_ld": 0.10, "c_ld": 0.10}


def main():
    D = json.load(open(IN_JSON))
    acc = {p: {m: np.array(D["acc"][p][str(m)]) for m in range(1, M_MAX + 1)} for p in PROBES}
    S, L = acc["op"][1].shape

    # Compute: for each (probe, marker, step): max_l acc[step, l] + argmax layer
    best_at_step = {p: {m: np.full(S, np.nan) for m in range(1, M_MAX + 1)} for p in PROBES}
    best_at_step_layer = {p: {m: np.full(S, -1, dtype=int) for m in range(1, M_MAX + 1)} for p in PROBES}
    cum_max = {p: {m: np.full(S, np.nan) for m in range(1, M_MAX + 1)} for p in PROBES}
    for p in PROBES:
        for m in range(1, M_MAX + 1):
            G = acc[p][m]
            for k in range(S):
                row = G[k, :]
                if np.all(np.isnan(row)): continue
                best_at_step[p][m][k] = float(np.nanmax(row))
                best_at_step_layer[p][m][k] = int(np.nanargmax(row))
            # cumulative max over steps 0..k
            run = -np.inf
            for k in range(S):
                cell = np.nanmax(G[:k+1, :])
                cum_max[p][m][k] = float(cell)

    out = {
        "best_at_step": {p: {m: best_at_step[p][m].tolist() for m in range(1, M_MAX + 1)} for p in PROBES},
        "best_at_step_layer": {p: {m: best_at_step_layer[p][m].tolist() for m in range(1, M_MAX + 1)} for p in PROBES},
        "cum_max": {p: {m: cum_max[p][m].tolist() for m in range(1, M_MAX + 1)} for p in PROBES},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    with PdfPages(OUT_PDF) as pdf:
        # Page 1: setup
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = ("multi_op_probe — BEST-anywhere-per-step trajectory\n\n"
                "Re-uses the saved (step, layer) accuracy grid from the original\n"
                "multi_op_probe sweep. For each step k:\n"
                "  best_at_step[k]  = max over the 13 layers of test acc at (k, layer)\n"
                "  cum_max[k]       = max over all (step ≤ k, layer)  — 'first-becomes\n"
                "                     -decodable anywhere by step k', non-decreasing\n"
                "  best_layer[k]    = which layer wins at step k (annotated on plot)\n\n"
                "Compare best_at_step to chance line for emergence; compare\n"
                "best_at_step at step k+1 vs step k for 'persistence' or 'fade'.\n")
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: best_at_step curves per probe
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for col, p in enumerate(PROBES):
            ax = axes[col]
            steps = np.arange(1, S + 1)
            for m in range(1, M_MAX + 1):
                ax.plot(steps, best_at_step[p][m], "-o",
                        label=f"m={m}", color=f"C{m-1}", lw=2)
                # annotate best layer at each step
                for k, st in enumerate(steps):
                    ly = best_at_step_layer[p][m][k]
                    if ly >= 0:
                        ax.annotate(f"L{ly}", (st, best_at_step[p][m][k] + 0.005),
                                    ha="center", fontsize=6, color=f"C{m-1}", alpha=0.7)
            ax.axhline(CHANCE[p], color="black", ls=":", alpha=0.5, label=f"chance={CHANCE[p]:.2f}")
            ax.set_xlabel("latent step"); ax.set_ylabel("max-over-layer test accuracy")
            ax.set_xticks(steps)
            ax.set_title(f"{p}: best-anywhere per step (L# = best layer at that step)",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ymax = max(best_at_step[p][m].max() for m in range(1, M_MAX + 1))
            ax.set_ylim(0, ymax + 0.07)
        fig.suptitle("Best-anywhere-per-step trajectory — at step k, "
                     "how decodable is marker m from the BEST layer at that step?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: cumulative max — when does marker m first become decodable?
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for col, p in enumerate(PROBES):
            ax = axes[col]
            steps = np.arange(1, S + 1)
            for m in range(1, M_MAX + 1):
                ax.plot(steps, cum_max[p][m], "-s",
                        label=f"m={m}", color=f"C{m-1}", lw=2)
            ax.axhline(CHANCE[p], color="black", ls=":", alpha=0.5, label=f"chance={CHANCE[p]:.2f}")
            ax.set_xlabel("latent step ≤ k"); ax.set_ylabel("cumulative max acc (any earlier step)")
            ax.set_xticks(steps)
            ax.set_title(f"{p}: cumulative max  (when does marker m\nfirst become decodable anywhere?)",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ymax = max(cum_max[p][m].max() for m in range(1, M_MAX + 1))
            ax.set_ylim(0, ymax + 0.05)
        fig.suptitle("Cumulative max accuracy — non-decreasing across steps "
                     "(once decodable, stays decodable here)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: persistence — best_at_step vs cum_max gap per marker per probe
        # Gap = cum_max[k] - best_at_step[k]. If marker m peaked earlier, gap > 0
        # at step k (info available somewhere but not here).
        fig, axes = plt.subplots(M_MAX, 3, figsize=(15, 11))
        for row_m, m in enumerate(range(1, M_MAX + 1)):
            for col_p, p in enumerate(PROBES):
                ax = axes[row_m, col_p]
                steps = np.arange(1, S + 1)
                ax.plot(steps, best_at_step[p][m], "-o", label="best at step k", color="C0")
                ax.plot(steps, cum_max[p][m], "--s", label="cumulative max (≤k)", color="C3", alpha=0.7)
                ax.axhline(CHANCE[p], color="black", ls=":", alpha=0.5)
                ax.set_xticks(steps); ax.set_xlabel("step", fontsize=8)
                ax.set_ylabel("acc", fontsize=8)
                # gap shading
                ax.fill_between(steps, best_at_step[p][m], cum_max[p][m],
                                where=(cum_max[p][m] > best_at_step[p][m] + 1e-6),
                                color="red", alpha=0.15, label="gap = info faded")
                ax.set_title(f"{p}, m={m}", fontsize=9, fontweight="bold")
                if row_m == 0 and col_p == 0:
                    ax.legend(fontsize=7, loc="lower right")
                ax.grid(alpha=0.3)
        fig.suptitle("Persistence vs fade — is marker m's signal as strong at step k "
                     "as the best earlier step?  (red = faded)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
