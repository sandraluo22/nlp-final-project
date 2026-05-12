"""Logit-lens visualization for CODI's 6 latent steps on GSM8K.

Reads experiments/computation_probes/logit_lens_gsm8k.{json,npz} (produced by
logit_lens_gsm8k.py) and renders a dedicated PDF:

  - Page 1: setup
  - Pages 2-5: for each sublayer ∈ {resid_pre, attn_out, mlp_out, resid_post},
    a (step × layer) heatmap colored by mean top-1 confidence with the
    modal top-1 token written in each cell.
  - Page 6: per-step summary at the LAST layer (L11) across all 4 sublayers,
    showing how the "what would emit here" prediction evolves through the
    sublayers per latent step.
  - Page 7: top-5 token tables for the first 3 sample problems.

Output: gpt2-gsm8k/logit_lens_slideshow_gsm8k.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
IN_JSON = REPO / "experiments" / "computation_probes" / "logit_lens_gsm8k.json"
OUT_PDF = PD / "logit_lens_slideshow_gsm8k.pdf"


def main():
    d = json.load(open(IN_JSON))
    N_LAT = d["N_LAT"]; N_LAYERS = d["N_LAYERS"]
    SUBL = d["SUBLAYERS"]
    conf = np.array(d["mean_top1_conf"])      # (N_LAT, N_LAYERS, n_sublayers)
    modal = d["modal_token"]                  # nested list
    modal_freq = np.array(d["modal_freq"])    # (N_LAT, N_LAYERS, n_sublayers)
    samples = d.get("sample_top5", [])

    with PdfPages(OUT_PDF) as pdf:
        # ----- Page 1: setup -----
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = (
            "Logit-lens on CODI's 6 latent steps  (GSM8K test, N={N})\n\n"
            "Procedure: at each (latent step, layer, sublayer) cell at the LAST-TOKEN\n"
            "position, apply the model's final LayerNorm + LM head to the hidden\n"
            "state. Record the top-{K} tokens + softmax probabilities.\n\n"
            "Sublayers shown (in this order in the residual stream):\n"
            "  resid_pre  : residual stream coming INTO this block\n"
            "  attn_out   : the attention block's output (added to residual)\n"
            "  mlp_out    : the MLP block's output (added to residual)\n"
            "  resid_post : residual stream coming OUT of this block\n\n"
            "Heatmaps color cells by mean top-1 confidence. The text in each cell\n"
            "is the most-common (modal) top-1 token at that cell across {N} problems.\n\n"
            "Interpretation guide:\n"
            "  - High-confidence cells with the SAME modal token = consistent prediction.\n"
            "  - The progression from resid_pre → attn_out → mlp_out → resid_post within\n"
            "    a (step, layer) shows what each sublayer is contributing.\n"
            "  - Across steps at the LAST layer (L11) reveals how the model's\n"
            "    intended emission evolves through the latent loop.\n"
        ).format(N=d["N_examples"], K=d["TOP_K"])
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Logit-lens visualization", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ----- Pages 2-5: per-sublayer heatmaps -----
        for sl_i, sl in enumerate(SUBL):
            fig, ax = plt.subplots(figsize=(15, 6))
            G = conf[:, :, sl_i]  # (step, layer)
            vmax = max(0.3, float(G.max()))
            im = ax.imshow(G, aspect="auto", origin="lower", cmap="viridis",
                           vmin=0, vmax=vmax)
            ax.set_xticks(range(N_LAYERS))
            ax.set_yticks(range(N_LAT))
            ax.set_yticklabels([f"step {i+1}" for i in range(N_LAT)])
            ax.set_xlabel("layer")
            ax.set_title(f"{sl}: modal top-1 token (text) + mean confidence (color)\n"
                         f"N={d['N_examples']} GSM8K test problems",
                         fontsize=11, fontweight="bold")
            for s in range(N_LAT):
                for L in range(N_LAYERS):
                    tk = (modal[s][L][sl_i] or "").replace("\n", "\\n")
                    # Sanitize for display
                    tk_disp = tk[:6] if len(tk) > 6 else tk
                    c_val = G[s, L]
                    freq = modal_freq[s, L, sl_i]
                    label = f"{tk_disp!r}\n{c_val:.2f}"
                    color = "white" if c_val < vmax * 0.55 else "black"
                    ax.text(L, s, label, ha="center", va="center",
                            fontsize=6, color=color)
            fig.colorbar(im, ax=ax, fraction=0.04, label="mean top-1 confidence")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ----- Page 6: per-step trajectory at last layer (L11) across sublayers -----
        last_layer = N_LAYERS - 1
        fig, axes = plt.subplots(1, len(SUBL), figsize=(4 * len(SUBL), 5),
                                  sharey=True)
        if len(SUBL) == 1: axes = [axes]
        for sl_i, sl in enumerate(SUBL):
            ax = axes[sl_i]
            vals = conf[:, last_layer, sl_i]
            ax.bar(range(1, N_LAT + 1), vals,
                   color="#4c72b0", edgecolor="black")
            for s in range(N_LAT):
                tk = (modal[s][last_layer][sl_i] or "").replace("\n", "\\n")
                tk_disp = tk[:6] if len(tk) > 6 else tk
                ax.text(s + 1, vals[s] + 0.005, f"{tk_disp!r}",
                        ha="center", fontsize=8, rotation=20)
            ax.set_xticks(range(1, N_LAT + 1))
            ax.set_xlabel("latent step")
            if sl_i == 0:
                ax.set_ylabel("mean top-1 confidence")
            ax.set_title(f"{sl} @ L{last_layer}", fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, max(0.7, float(conf.max()) + 0.1))
        fig.suptitle(f"Per-step trajectory at last layer (L{last_layer}): "
                     "what would the LM head emit if we stopped here?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ----- Page 7: sample top-5 tokens for a few problems at resid_post L11 -----
        if samples:
            for s_i, samp in enumerate(samples[:3]):
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.axis("off")
                gold = samp.get("gold")
                txt = f"Problem {s_i + 1}  (gold = {gold})\n\n"
                txt += f"Q: {samp['q'][:200]}{'...' if len(samp['q']) > 200 else ''}\n\n"
                txt += "Top-5 tokens at each (step, sublayer) cell at last layer L11:\n\n"
                txt += f"  {'step':<5} "
                for sl in SUBL:
                    txt += f"{sl:<35s} "
                txt += "\n"
                for s in range(N_LAT):
                    row = f"  step{s+1:<2}  "
                    for sl in SUBL:
                        key = f"step{s+1}_L{N_LAYERS-1}_{sl}"
                        cell = samp.get("cells", {}).get(key)
                        if cell is None:
                            row += f"{'(no data)':<35s} "
                            continue
                        toks = cell["top_tokens"][:3]
                        probs = cell["top_probs"][:3]
                        s_str = ",".join(f"{repr(t)[:5]}({p:.2f})"
                                          for t, p in zip(toks, probs))
                        row += f"{s_str:<35s} "
                    txt += row + "\n"
                ax.text(0.01, 0.99, txt, va="top", ha="left",
                        family="monospace", fontsize=7)
                ax.set_title(f"Sample {s_i + 1}: top-3 tokens per "
                             f"(step, sublayer) at L{N_LAYERS-1}",
                             fontsize=11, fontweight="bold")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
