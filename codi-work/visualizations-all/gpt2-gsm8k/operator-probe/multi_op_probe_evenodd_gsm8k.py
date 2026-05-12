"""Re-analyzes multi_op_probe_gsm8k.json (no new compute) to answer:

  Q1: For each (step, layer) cell, which marker does it predict best, and how
      selective is it for that marker vs others? (cleaner chain-tracing signal)
  Q2: Does the operator/last-digit probe accuracy differ between ODD steps
      (1, 3, 5) and EVEN steps (2, 4, 6)? Per marker, per probe type.
  Q3: If we restrict the best-cell search to odd steps only vs even steps only,
      how does the (step, layer) assignment per marker change?

Reads:  multi_op_probe_gsm8k.json
Writes: multi_op_probe_evenodd_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
IN_JSON = PD / "multi_op_probe_gsm8k.json"
OUT_JSON = PD / "multi_op_probe_evenodd_gsm8k.json"
OUT_PDF = PD / "multi_op_probe_evenodd_gsm8k.pdf"

PROBES = ["op", "a_ld", "c_ld"]
M_MAX = 4
CHANCE = {"op": 0.25, "a_ld": 0.10, "c_ld": 0.10}


def main():
    D = json.load(open(IN_JSON))
    # acc[probe][m] is a (S, L) list
    acc = {p: {m: np.array(D["acc"][p][str(m)]) for m in range(1, M_MAX + 1)}
           for p in PROBES}
    S, L = acc["op"][1].shape
    print(f"S={S} L={L}")

    odd_steps = [s for s in range(S) if (s + 1) % 2 == 1]   # 1-indexed odd
    even_steps = [s for s in range(S) if (s + 1) % 2 == 0]

    # ---- Q1: selectivity per cell ----
    # For each cell (s, l) and each probe type, find argmax m_best(s,l)
    # and selectivity = acc_max - mean(other markers' acc at this cell)
    selectivity = {p: np.full((S, L), np.nan) for p in PROBES}
    argmax_marker = {p: np.full((S, L), -1, dtype=int) for p in PROBES}
    for p in PROBES:
        # stack (S, L, M)
        accM = np.stack([acc[p][m] for m in range(1, M_MAX + 1)], axis=-1)  # (S, L, M)
        argmax_marker[p] = accM.argmax(axis=-1)
        for s in range(S):
            for l in range(L):
                accs = accM[s, l, :]
                if np.all(np.isnan(accs)): continue
                a_max = float(np.nanmax(accs))
                a_others = np.delete(accs, np.nanargmax(accs))
                selectivity[p][s, l] = a_max - float(np.nanmean(a_others))

    # ---- Q2: odd vs even mean accuracy per (probe, marker) ----
    odd_even_stats = {}
    for p in PROBES:
        odd_even_stats[p] = {}
        for m in range(1, M_MAX + 1):
            G = acc[p][m]  # (S, L)
            odd_mean = float(np.nanmean(G[odd_steps]))
            even_mean = float(np.nanmean(G[even_steps]))
            odd_max = float(np.nanmax(G[odd_steps]))
            even_max = float(np.nanmax(G[even_steps]))
            odd_argmax_flat = int(np.nanargmax(G[odd_steps]))
            o_s = odd_steps[odd_argmax_flat // L]; o_l = odd_argmax_flat % L
            even_argmax_flat = int(np.nanargmax(G[even_steps]))
            e_s = even_steps[even_argmax_flat // L]; e_l = even_argmax_flat % L
            odd_even_stats[p][m] = {
                "odd_mean": odd_mean, "even_mean": even_mean,
                "odd_max": odd_max, "even_max": even_max,
                "odd_max_cell": [o_s + 1, o_l],
                "even_max_cell": [e_s + 1, e_l],
                "delta_max_odd_minus_even": odd_max - even_max,
            }

    out = {
        "selectivity_pos": {p: selectivity[p].tolist() for p in PROBES},
        "argmax_marker": {p: argmax_marker[p].tolist() for p in PROBES},
        "odd_even": odd_even_stats,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    # ---- Slideshow ----
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: summary text
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.axis("off")
        body = "Multi-op probe — even/odd step analysis & per-cell marker selectivity\n\n"
        body += "Q1: per-cell selectivity = (best marker's acc) − mean(other markers' acc).\n"
        body += "    High = this cell specifically encodes one marker (not all).\n\n"
        body += "Q2: per-(probe, marker), mean accuracy on ODD steps (1,3,5) vs EVEN (2,4,6).\n\n"
        body += "Best cells (full unconstrained sweep, from original probe):\n"
        rb = D["best"]
        for p in PROBES:
            body += f"  {p}:\n"
            for m in range(1, M_MAX + 1):
                b = rb[p][str(m)]
                if b: body += f"    m={m}: step{b['step']} L{b['layer']:<2d}  acc={b['acc']:.3f}\n"
            body += "\n"
        body += "ODD-only vs EVEN-only best cell per marker:\n"
        for p in PROBES:
            body += f"  {p}:\n"
            for m in range(1, M_MAX + 1):
                st = odd_even_stats[p][m]
                body += (f"    m={m}: ODD step{st['odd_max_cell'][0]} L{st['odd_max_cell'][1]:<2d} acc={st['odd_max']:.3f}  "
                         f"||  EVEN step{st['even_max_cell'][0]} L{st['even_max_cell'][1]:<2d} acc={st['even_max']:.3f}  "
                         f"(Δ={st['delta_max_odd_minus_even']:+.3f})\n")
        ax.text(0.02, 0.98, body, va="top", ha="left", family="monospace", fontsize=9)
        ax.set_title("Setup & per-marker odd/even best cells", fontsize=12, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: bar chart — odd-mean vs even-mean per (probe, marker)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for col, p in enumerate(PROBES):
            ms = list(range(1, M_MAX + 1))
            odds  = [odd_even_stats[p][m]["odd_mean"]  for m in ms]
            evens = [odd_even_stats[p][m]["even_mean"] for m in ms]
            ax = axes[col]
            w = 0.4; xs = np.arange(len(ms))
            ax.bar(xs - w/2, odds, w, color="#4c72b0", edgecolor="black", label="odd steps (1,3,5)")
            ax.bar(xs + w/2, evens, w, color="#dd8452", edgecolor="black", label="even steps (2,4,6)")
            ax.axhline(CHANCE[p], color="black", ls=":", alpha=0.5, label=f"chance={CHANCE[p]:.2f}")
            ax.set_xticks(xs); ax.set_xticklabels([f"m={m}" for m in ms])
            ax.set_ylabel("mean acc across layers")
            ax.set_title(f"{p}  — odd vs even step mean acc", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
            for i, (o, e) in enumerate(zip(odds, evens)):
                ax.text(i - w/2, o + 0.005, f"{o:.2f}", ha="center", fontsize=7)
                ax.text(i + w/2, e + 0.005, f"{e:.2f}", ha="center", fontsize=7)
        fig.suptitle("ODD vs EVEN step mean accuracy per marker (averaged across all 12 layers)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: ODD-best vs EVEN-best max acc per marker (peak comparison)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for col, p in enumerate(PROBES):
            ms = list(range(1, M_MAX + 1))
            odd_max = [odd_even_stats[p][m]["odd_max"]  for m in ms]
            even_max = [odd_even_stats[p][m]["even_max"] for m in ms]
            ax = axes[col]
            w = 0.4; xs = np.arange(len(ms))
            ax.bar(xs - w/2, odd_max, w, color="#4c72b0", edgecolor="black", label="best odd cell")
            ax.bar(xs + w/2, even_max, w, color="#dd8452", edgecolor="black", label="best even cell")
            ax.axhline(CHANCE[p], color="black", ls=":", alpha=0.5)
            ax.set_xticks(xs); ax.set_xticklabels([f"m={m}" for m in ms])
            ax.set_ylabel("best cell acc (within parity)")
            ax.set_title(f"{p}  — best ODD vs best EVEN cell", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
            for i, m in enumerate(ms):
                st = odd_even_stats[p][m]
                ax.annotate(f"step{st['odd_max_cell'][0]} L{st['odd_max_cell'][1]}",
                            (i - w/2, odd_max[i] + 0.01), ha="center", fontsize=7, rotation=20)
                ax.annotate(f"step{st['even_max_cell'][0]} L{st['even_max_cell'][1]}",
                            (i + w/2, even_max[i] + 0.01), ha="center", fontsize=7, rotation=20)
        fig.suptitle("Best ODD-step cell vs best EVEN-step cell per marker  (peak acc)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: per-probe selectivity heatmap
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for col, p in enumerate(PROBES):
            ax = axes[col]
            G = selectivity[p]
            vmax = max(0.02, float(np.nanmax(G)))
            im = ax.imshow(G, aspect="auto", origin="lower", cmap="magma",
                           vmin=0, vmax=vmax)
            for s in range(S):
                for l in range(L):
                    am = argmax_marker[p][s, l]
                    txt = f"m{am+1}\n{G[s,l]*100:.0f}"
                    ax.text(l, s, txt, ha="center", va="center", fontsize=6,
                            color="white" if G[s, l] < vmax / 2 else "black")
            ax.set_xticks(range(L))
            ax.set_yticks(range(S)); ax.set_yticklabels([str(i+1) for i in range(S)])
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_title(f"{p}: argmax marker (top label) + selectivity Δacc (bottom number, pp)",
                         fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04, label="selectivity (Δacc)")
        fig.suptitle("Per-cell marker selectivity — for each (step, layer), which marker does it best predict, "
                     "and how specifically?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
