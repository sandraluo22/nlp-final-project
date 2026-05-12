"""Focused analysis of the step 2 → step 3 force-decode accuracy jump
(22.7% → 36.5%, the biggest single-step gain).

Combines four data sources:
  - flow_map_gsm8k.npz       (attention + block norms)
  - force_decode_per_step_gsm8k.json
  - logit_lens_gsm8k.json    (per-step modal token + confidence)
  - multi_op_probe_gsm8k.json (probe acc per (step, layer))

Renders ONE PDF:
  step2to3_focus_gsm8k.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
CP = REPO / "experiments" / "computation_probes"
OPER = PD / "operator-probe"
OUT_PDF = PD / "step2to3_focus_gsm8k.pdf"


def load_json(p):
    try: return json.load(open(p))
    except: return None


def main():
    flow = np.load(CP / "flow_map_gsm8k.npz")
    meta = load_json(CP / "flow_map_gsm8k_meta.json")
    ATTN = flow["mean_attn"]
    ATTN_NORM = flow["mean_attn_norm"]
    MLP_NORM = flow["mean_mlp_norm"]
    CLASS_NAMES = meta["class_names"]
    N_LAT = meta["N_latent_steps"]
    N_LAYERS = meta["N_layers"]

    fdec = load_json(CP / "force_decode_per_step_gsm8k.json")
    ll = load_json(CP / "logit_lens_gsm8k.json")
    mop = load_json(OPER / "multi_op_probe_gsm8k.json")

    with PdfPages(OUT_PDF) as pdf:
        # ============ Page 1: setup + accuracy summary ============
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")
        acc_per_step = [sum(c) / fdec["N"] for c in fdec["correct_per_step"]] if fdec else [None]*6
        body = (
            "Focused diagnosis: step 2 → step 3 (the biggest accuracy jump)\n\n"
            f"  Force-decode accuracy per latent step (full GSM8K test, N={fdec['N'] if fdec else '?'}):\n"
        )
        for s, a in enumerate(acc_per_step):
            highlight = "  ← +%.1fpp" % ((a - acc_per_step[s-1]) * 100) if s > 0 and a is not None else ""
            body += f"    step {s+1}: {a*100:5.1f}%{highlight}\n" if a is not None else ""
        body += (
            "\n  The 2→3 jump (+13.8pp) is the biggest single-step gain. It happens when:\n"
            "    1. step 3 first attends substantially to a PRIOR latent (L2, 13% of attention)\n"
            "    2. Q-attention drops to its lowest (77%) and stays there until step 4 rebounds\n"
            "    3. the LM-head prediction at L11 cycles back to '>>' (marker-close token)\n\n"
            "  Interpretation: step 2 commits a number; step 3 reads it back via attention and\n"
            "  closes the first marker. Before step 3 the model has no synthesized intermediate;\n"
            "  after step 3 it has marker 1's result in the cumulative residual.\n"
        )
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Step 2 → 3: what changes?", fontsize=13, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ============ Page 2: attention share at step 2 vs step 3 ============
        # per-layer attention shares to {Q, BOT, L1, L2}, averaged over heads
        attn_by_layer = ATTN[0, :, :, :, :].mean(axis=2)  # (step, layer, class)
        plot_classes = ["Q", "BOT", "L1", "L2"]
        plot_cls_idx = [CLASS_NAMES.index(c) for c in plot_classes]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(plot_classes)))
        for col, FS in enumerate([1, 2]):   # 0-indexed step 2 and 3
            ax = axes[col]
            stack = attn_by_layer[FS, :, :][:, plot_cls_idx]   # (layer, class)
            bot = np.zeros(N_LAYERS)
            for ci, cname in enumerate(plot_classes):
                vals = stack[:, ci]
                ax.bar(range(N_LAYERS), vals, bottom=bot, color=colors[ci],
                       edgecolor="white", linewidth=0.3, label=cname)
                bot += vals
            ax.set_xticks(range(N_LAYERS))
            ax.set_xlabel("layer")
            ax.set_ylabel("attention share")
            ax.set_title(f"step {FS+1}: per-layer attention shares",
                         fontsize=11, fontweight="bold")
            if col == 0:
                ax.legend(fontsize=9, ncol=2)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Q vs L1 vs L2 — per-layer breakdown at step 2 vs step 3",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ============ Page 3: block-output norms, step 2 vs step 3 ============
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for col, FS in enumerate([1, 2]):
            ax = axes[col]
            w = 0.4
            ax.bar(np.arange(N_LAYERS) - w/2, ATTN_NORM[0, FS], w,
                   color="#4c72b0", edgecolor="black", label="‖attn out‖")
            ax.bar(np.arange(N_LAYERS) + w/2, MLP_NORM[0, FS], w,
                   color="#dd8452", edgecolor="black", label="‖MLP out‖")
            ax.set_xticks(range(N_LAYERS))
            ax.set_xlabel("layer")
            if col == 0: ax.set_ylabel("mean output norm")
            ax.set_title(f"step {FS+1}: how much each block writes",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Block-output magnitudes per layer at step 2 vs step 3",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ============ Page 4: logit-lens at step 2 vs step 3 (per-layer, resid_post) ============
        if ll is not None:
            SUBL = ll["SUBLAYERS"]
            sl_i = SUBL.index("resid_post")
            modal = ll["modal_token"]
            conf = np.array(ll["mean_top1_conf"])
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            for col, FS in enumerate([1, 2]):
                ax = axes[col]
                vals = conf[FS, :, sl_i]
                ax.bar(range(N_LAYERS), vals, color="#4c72b0", edgecolor="black")
                for L in range(N_LAYERS):
                    tk = (modal[FS][L][sl_i] or "").replace("\n", "\\n")
                    tk_disp = tk[:6] if len(tk) > 6 else tk
                    ax.text(L, vals[L] + 0.005, f"{tk_disp!r}",
                            ha="center", fontsize=7, rotation=30)
                ax.set_xticks(range(N_LAYERS))
                ax.set_xlabel("layer")
                ax.set_ylabel("modal top-1 token confidence")
                ax.set_title(f"step {FS+1}: 'what would emit here?' per layer",
                             fontsize=11, fontweight="bold")
                ax.set_ylim(0, max(0.65, float(conf.max()) + 0.1))
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle("Logit-lens at resid_post: step 2 (number prediction) vs "
                         "step 3 ('>>' marker-close prediction)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ============ Page 5: probe accuracy at step 2 vs step 3 (per-marker) ============
        # multi_op_probe acc grids use 13 layers (CODI captures L0..L12 hidden states),
        # so plot range matches the grid's layer dim.
        if mop is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            probes = ["op", "a_ld", "c_ld"]
            chances = {"op": 0.25, "a_ld": 0.1, "c_ld": 0.1}
            for col, p in enumerate(probes):
                ax = axes[col]
                for FS in [1, 2]:
                    for m in range(1, 5):
                        G = np.array(mop["acc"][p][str(m)])
                        ax.plot(range(G.shape[1]), G[FS],
                                "-" if FS == 2 else "--",
                                lw=1.5 if FS == 2 else 1,
                                alpha=0.85 if FS == 2 else 0.5,
                                color=plt.cm.tab10(m / 5),
                                label=f"step{FS+1} m={m}")
                ax.axhline(chances[p], color="black", ls=":", alpha=0.5,
                           label=f"chance={chances[p]:.2f}")
                ax.set_xticks(range(0, G.shape[1], 2))
                ax.set_xlabel("layer")
                ax.set_ylabel("probe acc")
                ax.set_title(f"{p} — solid=step 3, dashed=step 2",
                             fontsize=10, fontweight="bold")
                ax.legend(fontsize=6, ncol=2)
                ax.grid(alpha=0.3)
            fig.suptitle("Probe accuracy at step 2 (dashed) vs step 3 (solid)\n"
                         "— what becomes decodable at step 3 that wasn't at step 2?",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
