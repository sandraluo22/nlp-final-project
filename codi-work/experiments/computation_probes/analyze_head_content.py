"""Decode what each (latent step, layer, head) writes to the residual.

Inputs:
  flow_head_content.npz   (S=6, L=12, H_heads=12, H=768) — mean per-head
                          residual contribution at the last token, averaged
                          across all 1000 SVAMP examples and per operator class.
  flow_map.npz            for attention-to-class baseline (to find which heads
                          are doing cross-latent reads).
  codi_gpt2_lm_head.npy   lm_head weights (vocab, H).
  codi_gpt2_ln_f.npz      ln_f gamma + beta.
  operator_centroids_layer10_step4.json  per-op residual centroids at one cell.

For each (step, layer, head), apply ln_f + lm_head to the head's mean output
and report top-K decoded tokens. Also compute cosine similarity to the four
operator centroids.

Outputs:
  head_content_topk.json  per-(step, layer, head) top-5 decoded tokens +
                          operator alignment + per-class top tokens
  head_content_slideshow.pdf
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoTokenizer

PD = Path(__file__).resolve().parent
FLOW = PD / "flow_head_content.npz"
FLOW_MAP = PD / "flow_map.npz"
LM_HEAD = PD / "codi_gpt2_lm_head.npy"
LN_F = PD / "codi_gpt2_ln_f.npz"
OUT_JSON = PD / "head_content_topk.json"
OUT_PDF = PD / "head_content_slideshow.pdf"
OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


def apply_ln_f(h, gamma, beta):
    mean = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    normed = (h - mean) / np.sqrt(var + 1e-5)
    return normed * gamma + beta


def main():
    print("loading", FLOW)
    fc = np.load(FLOW)
    head_resid = fc["mean_head_resid"]      # (S, L, H_heads, H_hidden)
    mlp_out = fc["mean_mlp_out"]            # (S, L, H_hidden)
    head_norm = fc["mean_head_norm"]        # (S, L, H_heads)
    head_attn = fc["mean_head_attn"]        # (S, L, H_heads, N_CLASS=8)
    op_head_resid = fc["op_mean_head_resid"] # (4, S, L, H_heads, H_hidden)
    op_mlp_out = fc["op_mean_mlp_out"]      # (4, S, L, H_hidden)

    S, L, H_heads, HID = head_resid.shape
    print(f"  S={S}, L={L}, H_heads={H_heads}, HID={HID}")

    print("loading lm_head + ln_f")
    W = np.load(LM_HEAD)
    ln = np.load(LN_F)
    gamma, beta = ln["weight"], ln["bias"]
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # Decode top-K tokens from a residual vector
    def top_tokens(h_vec, k=5):
        h_normed = apply_ln_f(h_vec[None, :], gamma, beta)[0]
        logits = h_normed @ W.T
        top_ids = np.argsort(-logits)[:k]
        return [(int(i), tok.decode([int(i)]), float(logits[i])) for i in top_ids]

    # Operator alignment: cos sim to op centroid (using flow's op_head_resid).
    # Each head's op-specific contribution: op_head_resid[op, s, l, h, :]
    # Average over ops to get "op-independent" baseline; deviation = op-specificity.

    print("computing top tokens per (step, layer, head) + per-op...")
    records = {}
    for s in range(S):
        for l in range(L):
            for h in range(H_heads):
                vec = head_resid[s, l, h, :]
                tops = top_tokens(vec, k=5)
                # per-op top tokens for this head
                op_tops = {}
                for o, op in enumerate(OPS):
                    op_vec = op_head_resid[o, s, l, h, :]
                    op_tops[op] = top_tokens(op_vec, k=3)
                # op alignment: cos sim between op_vec and overall vec
                norms = []
                for o in range(4):
                    op_vec = op_head_resid[o, s, l, h, :]
                    cs = float(np.dot(op_vec, vec)
                                / (np.linalg.norm(op_vec) * np.linalg.norm(vec) + 1e-12))
                    norms.append(cs)
                records[f"s{s+1}_l{l}_h{h}"] = {
                    "norm": float(head_norm[s, l, h]),
                    "attn_to_Q": float(head_attn[s, l, h, 0]),
                    "attn_to_BOT": float(head_attn[s, l, h, 1]),
                    "attn_to_priorL": float(head_attn[s, l, h, 2:8].sum()),
                    "top_decoded": [(t[1], t[2]) for t in tops],
                    "op_top_decoded": {k: [(t[1], t[2]) for t in v] for k, v in op_tops.items()},
                    "op_cos_to_overall": dict(zip(OPS, norms)),
                }

    # MLP per layer/step top tokens
    mlp_records = {}
    for s in range(S):
        for l in range(L):
            vec = mlp_out[s, l]
            tops = top_tokens(vec, k=5)
            op_tops = {}
            for o, op in enumerate(OPS):
                op_tops[op] = top_tokens(op_mlp_out[o, s, l], k=3)
            mlp_records[f"s{s+1}_l{l}"] = {
                "norm": float(np.linalg.norm(vec)),
                "top_decoded": [(t[1], t[2]) for t in tops],
                "op_top_decoded": {k: [(t[1], t[2]) for t in v] for k, v in op_tops.items()},
            }

    OUT_JSON.write_text(json.dumps({"heads": records, "mlp": mlp_records}, indent=2))
    print(f"saved {OUT_JSON}")

    # ===== Slideshow =====
    print("rendering slideshow...")
    with PdfPages(OUT_PDF) as pdf:
        # Slide 1: title + reading instructions
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle("What does each (step, layer, head) WRITE to the residual?",
                     fontsize=14, fontweight="bold")
        ax = fig.add_axes([0.05, 0.04, 0.9, 0.86]); ax.axis("off")
        ax.text(0.0, 0.95,
                "Each head's mean residual contribution (averaged across N=1000 SVAMP) "
                "is decoded via ln_f + lm_head to find its TOP 5 written tokens.\n\n"
                "Also: each head's attention-to-Q / BOT / prior latents fraction. "
                "Per-operator-type top tokens reveal whether a head writes operator-specific "
                "content.\n\n"
                "Heatmaps below show per-(step, layer, head) norms — bright cells = heads "
                "writing the most signal.",
                fontsize=10, transform=ax.transAxes, va="top", wrap=True)
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 2: head-output norm heatmap per (step, layer, head)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Per-head residual-contribution norm at the last token (mean over N=1000)",
                     fontsize=12, fontweight="bold")
        for s in range(S):
            ax = axes.ravel()[s]
            M = head_norm[s].T   # (n_heads, n_layers)
            im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower",
                           vmin=0, vmax=head_norm.max())
            ax.set_xticks(range(L)); ax.set_yticks(range(H_heads))
            ax.set_yticklabels([f"H{h}" for h in range(H_heads)])
            ax.set_xlabel("layer"); ax.set_title(f"step {s+1}")
            for h in range(H_heads):
                for l in range(L):
                    v = M[h, l]
                    if v >= head_norm.max() * 0.3:
                        ax.text(l, h, f"{v:.0f}", ha="center", va="center", fontsize=6,
                                color="white" if v < head_norm.max() * 0.6 else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 3: per (step, layer) MLP top tokens (text only)
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle("MLP output top decoded tokens per (step, layer)  —  what the MLP writes",
                     fontsize=12, fontweight="bold")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.92]); ax.axis("off")
        y = 0.97
        for s in range(S):
            ax.text(0.0, y, f"--- step {s+1} ---", fontsize=10, fontweight="bold",
                    transform=ax.transAxes)
            y -= 0.034
            for l in range(L):
                key = f"s{s+1}_l{l}"
                tops = mlp_records[key]["top_decoded"]
                line = f"  L{l}  ‖{mlp_records[key]['norm']:6.1f}  : " + \
                       " / ".join(f"{t[0]!r}" for t in tops[:4])
                if y < 0.04: break
                ax.text(0.0, y, line, fontsize=7, family="monospace", transform=ax.transAxes)
                y -= 0.022
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slides 4+: for each (step, layer) where a head has substantial attn-to-priorL,
        # show that head's profile.
        # Threshold: a head with attn_to_priorL > 5%.
        target_cells = []
        for s in range(1, S):  # step 2+
            for l in range(L):
                for h in range(H_heads):
                    if head_attn[s, l, h, 2:8].sum() >= 0.05:
                        target_cells.append((s, l, h))
        target_cells.sort(key=lambda x: -head_attn[x[0], x[1], x[2], 2:8].sum())
        print(f"  {len(target_cells)} (step, layer, head) cells with attn-to-priorL >= 5%")

        # Top 20 such heads
        n_show = min(40, len(target_cells))
        fig = plt.figure(figsize=(13.33, 9.5))
        fig.suptitle(f"Top {n_show} cross-latent-reading heads: attention pattern + top-3 tokens they write",
                     fontsize=12, fontweight="bold")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.92]); ax.axis("off")
        ax.text(0.0, 0.97,
                f"{'cell':<14} {'attn_Q':>7} {'BOT':>6} {'priorL':>7} {'norm':>6}  "
                f"{'top-3 tokens written'}",
                fontsize=8, fontweight="bold", family="monospace", transform=ax.transAxes)
        y = 0.94
        for s, l, h in target_cells[:n_show]:
            tops = records[f"s{s+1}_l{l}_h{h}"]["top_decoded"][:3]
            line = (f"  s{s+1}.L{l:2d}.H{h:<3d}  "
                    f"{head_attn[s,l,h,0]:>7.2f}  {head_attn[s,l,h,1]:>5.2f}  "
                    f"{head_attn[s,l,h,2:8].sum():>7.2f}  {head_norm[s,l,h]:>6.1f}  "
                    + " / ".join(f"{t[0]!r}" for t in tops))
            if y < 0.04: break
            ax.text(0.0, y, line, fontsize=7, family="monospace", transform=ax.transAxes)
            y -= 0.022
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide: among those, what fraction read from L_{k-1} specifically vs earlier L's
        fig, ax = plt.subplots(figsize=(11, 5))
        # For each step s>1, get total attn to L1, L2, ..., L(s) for all heads with priorL>=5%.
        breakdown = np.zeros((S - 1, 5))   # rows: step 2..6; cols: which prior latent (L1..L(s))
        for s in range(1, S):
            mask = head_attn[s, :, :, 2:8].sum(axis=-1) >= 0.05
            for l in range(L):
                for h in range(H_heads):
                    if not mask[l, h]: continue
                    for j in range(s):
                        breakdown[s - 1, j] += head_attn[s, l, h, 2 + j]
        # normalize per step
        for s in range(S - 1):
            denom = breakdown[s].sum()
            if denom > 0: breakdown[s] /= denom
        x = np.arange(S - 1)
        bottom = np.zeros(S - 1)
        colors = plt.cm.viridis(np.linspace(0, 1, 5))
        for j in range(5):
            ax.bar(x, breakdown[:, j], bottom=bottom, color=colors[j], label=f"L{j+1}")
            bottom += breakdown[:, j]
        ax.set_xticks(x); ax.set_xticklabels([f"step {s+2}" for s in range(S - 1)])
        ax.set_xlabel("reading step"); ax.set_ylabel("frac of cross-latent attn mass")
        ax.set_title("Among cross-latent-reading heads, which prior latent do they read?",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide: per-op deltas for the most-active head per (step, layer).
        # Pick the head with highest norm at each (s, l). Show its operator-specific top tokens.
        fig = plt.figure(figsize=(13.33, 9.5))
        fig.suptitle("Highest-norm head per (step, layer): per-operator top decoded tokens",
                     fontsize=12, fontweight="bold")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.92]); ax.axis("off")
        y = 0.97
        for s in range(S):
            for l in range(L):
                h_best = int(np.argmax(head_norm[s, l]))
                key = f"s{s+1}_l{l}_h{h_best}"
                rec = records[key]
                if y < 0.04: break
                line = (f"  s{s+1}.L{l:2d}.H{h_best:<3d} ‖={rec['norm']:>5.1f}  "
                        + " | ".join(f"{op}: {rec['op_top_decoded'][op][0][0]!r}"
                                    for op in OPS))
                ax.text(0.0, y, line, fontsize=6.5, family="monospace", transform=ax.transAxes)
                y -= 0.018
            if y < 0.04: break
        pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
