"""Slideshow: information-flow map of CODI-GPT-2 on SVAMP (N=1000).

Visualizes:
- Attention from the last token at each (phase, step, layer, head) back to
  position-class buckets (Q, BOT, L1..L6, EOT, D0..D9), averaged over examples.
- ||attention block output|| and ||MLP block output|| at the last token per
  (phase, step, layer), averaged over examples — i.e., how much each
  component writes to the residual.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
OUT = PD / "flow_map_slideshow.pdf"

d = np.load(PD / "flow_map.npz")
meta = json.load(open(PD / "flow_map_meta.json"))
ATTN = d["mean_attn"]            # (phase=2, step=10, layer=12, head=12, class=19)
ATTN_NORM = d["mean_attn_norm"]  # (phase, step, layer)
MLP_NORM = d["mean_mlp_norm"]
CLASS_NAMES = meta["class_names"]
N_LAT, N_DEC = 6, 10
N_LAYERS = ATTN.shape[2]
N_HEADS = ATTN.shape[3]
N_CLASS = ATTN.shape[4]


def text_slide(pdf, title, lines):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax = fig.add_axes([0.05, 0.04, 0.9, 0.86]); ax.axis("off")
    y = 0.97
    for ln in lines:
        if ln.startswith("# "):
            ax.text(0.0, y, ln[2:], fontsize=12, fontweight="bold",
                    transform=ax.transAxes); y -= 0.045
        elif ln.startswith("- "):
            ax.text(0.02, y, "•  " + ln[2:], fontsize=10,
                    transform=ax.transAxes); y -= 0.034
        elif ln == "":
            y -= 0.020
        else:
            ax.text(0.0, y, ln, fontsize=10, transform=ax.transAxes); y -= 0.034
    pdf.savefig(fig, dpi=140); plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        # ===== Slide 1 =====
        text_slide(pdf, "Information-flow map of CODI-GPT-2 on SVAMP (N=1000)",
            [
                "# What is captured",
                "- For each forward step (6 latent + first 10 decode positions),",
                "  at every (layer, head): attention from the last (newly-produced)",
                "  token to every prior key position, averaged across N=1000 SVAMP.",
                "  Each prior position is bucketed by class: Q (any question token),",
                "  BOT, L1..L6 (the 6 latent positions), EOT, D0..D9 (post-EOT decode).",
                "- At every (layer): ||attention-block output||_2 and ||MLP-block",
                "  output||_2 at the last token — what each layer writes to residual.",
                "",
                "# Why these views",
                "- Attention by class = WHERE information is read from at each step.",
                "- Attention vs MLP norms = WHERE the residual is most modified.",
                "- Per-head class affinity = which heads route which class.",
            ])

        # ===== Slide 2: attention by class, latent phase, heatmap per class =====
        # For each class, plot (step, layer) heatmap of mean attention (over heads).
        latent_classes = ["Q", "BOT", "L1", "L2", "L3", "L4", "L5", "L6"]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle("LATENT phase: mean attention from latent_k to each class "
                     "(averaged over heads, then over 1000 examples)",
                     fontsize=12, fontweight="bold")
        for i, cname in enumerate(latent_classes):
            ax = axes.ravel()[i]
            ci = CLASS_NAMES.index(cname)
            mat = ATTN[0, :N_LAT, :, :, ci].mean(axis=-1)  # (step=6, layer=12)
            im = ax.imshow(mat, aspect="auto", origin="lower",
                           cmap="viridis", vmin=0, vmax=mat.max() if mat.max() > 0 else 1)
            ax.set_title(f"to {cname}", fontsize=9)
            ax.set_xlabel("layer", fontsize=8); ax.set_ylabel("latent step", fontsize=8)
            ax.set_yticks(range(N_LAT)); ax.set_yticklabels([str(s + 1) for s in range(N_LAT)])
            ax.set_xticks(range(0, N_LAYERS, 2))
            for s in range(N_LAT):
                for l in range(N_LAYERS):
                    v = mat[s, l]
                    if v >= 0.05:
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=5, color="white" if v < 0.6 * mat.max() else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ===== Slide 3: attn vs MLP norm heatmaps, latent + decode =====
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        fig.suptitle("Residual contributions per layer (norms of attn / MLP outputs at last token)",
                     fontsize=12, fontweight="bold")
        for col, (phase_id, n_steps, ylabel) in enumerate([
            (0, N_LAT, "latent step"), (1, N_DEC, "decode step")
        ]):
            for row, (mat_name, mat) in enumerate([
                ("||attn output||", ATTN_NORM[phase_id, :n_steps]),
                ("||MLP output||",  MLP_NORM[phase_id, :n_steps]),
            ]):
                ax = axes[row, col]
                im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma")
                ax.set_title(f"{['LATENT', 'DECODE'][phase_id]}: {mat_name}",
                             fontsize=10)
                ax.set_xlabel("layer"); ax.set_ylabel(ylabel)
                ax.set_yticks(range(n_steps))
                ax.set_yticklabels([str(s + (0 if phase_id else 1)) for s in range(n_steps)])
                for s in range(n_steps):
                    for l in range(N_LAYERS):
                        v = mat[s, l]
                        ax.text(l, s, f"{v:.0f}", ha="center", va="center",
                                fontsize=5, color="white")
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ===== Slide 4: decode-phase attention by class =====
        decode_classes = ["Q", "BOT", "L1", "L6", "EOT", "D0", "D1", "D2"]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle("DECODE phase: mean attention from decode_d back to each class",
                     fontsize=12, fontweight="bold")
        for i, cname in enumerate(decode_classes):
            ax = axes.ravel()[i]
            ci = CLASS_NAMES.index(cname)
            mat = ATTN[1, :N_DEC, :, :, ci].mean(axis=-1)
            im = ax.imshow(mat, aspect="auto", origin="lower",
                           cmap="viridis", vmin=0,
                           vmax=mat.max() if mat.max() > 0 else 1)
            ax.set_title(f"to {cname}", fontsize=9)
            ax.set_xlabel("layer", fontsize=8); ax.set_ylabel("decode step", fontsize=8)
            ax.set_yticks(range(N_DEC))
            ax.set_xticks(range(0, N_LAYERS, 2))
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ===== Slide 5: per-head specialization, latent step 4, layer 10 =====
        # For each head, attention distribution across classes (averaged across examples).
        s_target, l_target = 3, 10  # latent step 4
        head_dist = ATTN[0, s_target, l_target, :, :9]  # (12, 9)
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(head_dist, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(9)); ax.set_xticklabels(CLASS_NAMES[:9])
        ax.set_yticks(range(N_HEADS)); ax.set_yticklabels([f"H{h}" for h in range(N_HEADS)])
        ax.set_title(f"Latent step {s_target+1}, layer {l_target}: attention distribution per head",
                     fontsize=11, fontweight="bold")
        for h in range(N_HEADS):
            for c in range(9):
                v = head_dist[h, c]
                if v >= 0.05:
                    ax.text(c, h, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ===== Slide 6: per-head specialization, decode pos 1, layer 8 =====
        s_target, l_target = 1, 8  # decode pos 1
        head_dist = ATTN[1, s_target, l_target, :, :9]
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(head_dist, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(9)); ax.set_xticklabels(CLASS_NAMES[:9])
        ax.set_yticks(range(N_HEADS)); ax.set_yticklabels([f"H{h}" for h in range(N_HEADS)])
        ax.set_title(f"Decode step {s_target}, layer {l_target}: attention distribution per head",
                     fontsize=11, fontweight="bold")
        for h in range(N_HEADS):
            for c in range(9):
                v = head_dist[h, c]
                if v >= 0.05:
                    ax.text(c, h, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ===== Slide 7: synthesis =====
        # Compute headline numbers
        attn_to_Q_lat = ATTN[0, :N_LAT, :, :, 0].mean(axis=(-1, -2))  # (step,) → mean over layers/heads
        attn_to_prevL_lat = []
        for s in range(N_LAT):
            # mean attention to L1..L(s) at layer 0, averaged over heads
            ci_first = 2; ci_last = 2 + s  # L1..L(s) is classes 2..2+s-1, but use s and below
            if s == 0:
                attn_to_prevL_lat.append(0.0)
            else:
                attn_to_prevL_lat.append(
                    float(ATTN[0, s, 0, :, 2:2 + s].sum(axis=-1).mean()))
        attn_to_Q_dec = ATTN[1, :N_DEC, :, :, 0].mean(axis=(-1, -2))
        attn_to_latents_dec = ATTN[1, :N_DEC, :, :, 2:8].sum(axis=-1).mean(axis=(-1, -2))

        text_slide(pdf, "Synthesis: where does CODI route information?",
            [
                "# LATENT loop",
                "- The latent token spends 50-97% of its attention on Q (question tokens)",
                "  at every (step, layer). Re-reading the question is the dominant op.",
                f"- Per-step mean attention to Q (over layers & heads): "
                f"{', '.join(f'{x*100:.0f}%' for x in attn_to_Q_lat)}.",
                "- Cross-latent attention (latent_k -> L1..L(k-1)) is mainly at LAYER 0:",
                f"  step-1 = 0%,  step-2 ≈ {attn_to_prevL_lat[1]*100:.0f}%,  "
                f"step-3 ≈ {attn_to_prevL_lat[2]*100:.0f}%,  step-4 ≈ {attn_to_prevL_lat[3]*100:.0f}%,",
                f"  step-5 ≈ {attn_to_prevL_lat[4]*100:.0f}%,  step-6 ≈ {attn_to_prevL_lat[5]*100:.0f}%  (at layer 0;",
                "  drops to 1-3% at layer 3+).",
                "- BOT and EOT get near-zero attention except at very early layers.",
                "",
                "# DECODE phase",
                f"- Attention to Q stays high (mean over heads & layers per decode step:",
                f"  {', '.join(f'{x*100:.0f}%' for x in attn_to_Q_dec)}).",
                f"- Attention to the 6 latent positions (summed): "
                f"{', '.join(f'{x*100:.0f}%' for x in attn_to_latents_dec)}. The latent loop's",
                "  output reaches decode mainly via residual continuity, not via re-attention.",
                "",
                "# Residual contributions",
                "- ||attn output|| and ||MLP output|| are smallest at middle layers (3-6)",
                "  and spike at layer 0 (embedding processing) and again at layers 10-11.",
                "- MLP writes ~more than attention at layer 0 and layer 10; attention dominates",
                "  layer 11.",
                "",
                "# Implications for the operator-steering null result",
                "- If operator info comes mainly from re-reading Q via attention at every step,",
                "  no single (step, layer) residual swap should be sufficient to flip it —",
                "  the model re-derives the operator at every subsequent step.",
                "- That matches what we observed: 4-method patching produces ≤7% flip at any cell.",
                "- A causal intervention needs to disrupt the attention pattern itself, or",
                "  intervene at the Q tokens. Bigger lever next: question-token patching.",
            ])

    print(f"saved {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
