"""Slideshow: operator-steering interventions + attention.

Inputs:
  steering_operator_all.json
  attention_operator_latent.npz / .json
  attention_operator_decode.npz / .json
  gpt2_multipos_probes.json
  (existing) steering_operator_causality.json (for context)

Slides:
  1. Title + hypotheses
  2. Operator probe accuracy heatmaps (latent + decode regions)
  3. A_latent vs C_latent flip-rate heatmaps (4x4 src->tgt grid)
  4. A_decode vs C_decode flip-rate heatmaps
  5. All-four comparison: max flip across methods per src->tgt
  6. Attention from latent cell to question tokens (mean over heads)
  7. Attention from decode cell to question tokens (mean over heads)
  8. Top attended tokens at each cell (table)
  9. Synthesis
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
OUT = PD / "operator_steering_slideshow.pdf"

OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
OP_SHORT = {"Addition": "Add", "Subtraction": "Sub", "Multiplication": "Mul",
            "Common-Division": "Div"}

CUES_BY_OP = {
    "Addition": {"added", "add", "adds", "adding", "plus", "total", "altogether",
                 "combined", "together", "more", "sum"},
    "Subtraction": {"minus", "subtract", "subtracts", "subtracted", "subtracting",
                    "took", "take", "takes", "away", "left", "lost", "gave",
                    "remain", "remaining", "remove", "removed", "difference",
                    "fewer"},
    "Multiplication": {"times", "multiply", "multiplied", "multiplying", "each",
                       "every", "per", "double", "triple"},
    "Common-Division": {"divide", "divides", "divided", "split", "share",
                        "shared", "quotient"},
}
ALL_CUES = set().union(*CUES_BY_OP.values())


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


def flip_matrix(stats, key="frac_tgt"):
    """4x4 matrix; entry [src,tgt] is frac_tgt for src->tgt (NaN on diagonal)."""
    M = np.full((4, 4), np.nan)
    for i, src in enumerate(OPS):
        for j, tgt in enumerate(OPS):
            if src == tgt: continue
            k = f"{src}->{tgt}"
            if k in stats:
                M[i, j] = stats[k][key]
    return M


def flip_heatmap(ax, M, title, fmt="{:.1%}"):
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(0.10, np.nanmax(M)),
                   aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([OP_SHORT[o] for o in OPS])
    ax.set_yticks(range(4)); ax.set_yticklabels([OP_SHORT[o] for o in OPS])
    ax.set_xlabel("target op"); ax.set_ylabel("source op")
    ax.set_title(title, fontsize=10)
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            if np.isnan(v): ax.text(j, i, "—", ha="center", va="center",
                                     color="gray", fontsize=8)
            else: ax.text(j, i, fmt.format(v), ha="center", va="center",
                          color="white" if v < 0.05 else "black", fontsize=9)
    return im


def attention_diagnostic(meta, attn):
    """Compute (top-attended tokens, cue-attention by op) for one cell."""
    N, Hd, K = attn.shape
    decoded = meta["decoded_tokens"]
    types = np.array(meta["types"])

    # Top-1 attended tokens averaged across heads, counted by token type
    top1_counter = Counter()
    cue_attn_by_op = {op: [] for op in OPS}
    cue_attn_by_op_head = {op: np.zeros(Hd) for op in OPS}
    for i in range(N):
        toks = decoded[i]; k_used = len(toks)
        a_mean = attn[i, :, -k_used:].mean(axis=0)
        top_idx = int(np.argmax(a_mean))
        tok = toks[top_idx].strip().lower()
        if tok: top1_counter[tok] += 1
        # cumulative attention to operator-cue tokens (heads-wise)
        cue_mask = np.array([t.strip().lower().strip(".,?!") in ALL_CUES for t in toks])
        if cue_mask.any():
            op = types[i]
            if op in OPS:
                cue_attn_by_op[op].append(float(a_mean[cue_mask].sum()))
                for h in range(Hd):
                    cue_attn_by_op_head[op][h] += attn[i, h, -k_used:][cue_mask].sum()
    counts_by_op = {op: max(1, int((types == op).sum())) for op in OPS}
    for op in OPS:
        cue_attn_by_op_head[op] /= counts_by_op[op]
    return top1_counter, cue_attn_by_op, cue_attn_by_op_head, types


def main():
    d = json.load(open(PD / "steering_operator_all.json"))
    probes = json.load(open(PD / "gpt2_multipos_probes.json"))
    # latent-cell operator probe heatmap is not stored at same shape; we use
    # the operator_acc table from gpt2_multipos (decode positions)
    op_acc_decode = np.array(probes["operator_acc"])  # (P x L+1)

    # Existing earlier steering attempt (direction add) — for context
    try:
        old = json.load(open(PD / "steering_operator_causality.json"))
    except Exception:
        old = None

    meta_lat = json.load(open(PD / "attention_operator_latent_meta.json"))
    attn_lat = np.load(PD / "attention_operator_latent.npz")["attn"]
    meta_dec = json.load(open(PD / "attention_operator_decode_meta.json"))
    attn_dec = np.load(PD / "attention_operator_decode.npz")["attn"]

    M_A_lat = flip_matrix(d["A_centroid_patch"], "frac_tgt")
    M_C_lat = flip_matrix(d["C_cross_patch"], "frac_tgt")
    M_A_dec = flip_matrix(d.get("A_centroid_patch_decode", {}), "frac_tgt")
    M_C_dec = flip_matrix(d.get("C_cross_patch_decode", {}), "frac_tgt")
    M_max = np.fmax.reduce([M_A_lat, M_C_lat, M_A_dec, M_C_dec])

    top1_lat, cue_lat, cue_lat_head, _ = attention_diagnostic(meta_lat, attn_lat)
    top1_dec, cue_dec, cue_dec_head, _ = attention_diagnostic(meta_dec, attn_dec)

    with PdfPages(OUT) as pdf:
        # === Slide 1: title + hypotheses ===
        text_slide(pdf, "Operator steering on CODI-GPT-2: 4 methods × 2 cells",
            [
                "# Question",
                "- Can we causally flip the model's operator (Add↔Sub↔Mul↔Div) by",
                "  intervening on the residual stream at a single position?",
                "",
                "# Setup",
                f"- N = {d['N']}  SVAMP problems.  Baseline accuracy = {d['baseline_accuracy']*100:.1f}%.",
                "- Two candidate intervention cells (where operator info is most concentrated):",
                "  ▸ LATENT  cell  =  (latent step 4, layer 10).  Centroids file: operator_centroids_layer10_step4.json.",
                "  ▸ DECODE  cell  =  (decode pos 1, layer 8).  Operator probe acc 92.9% here.",
                "",
                "# Methods (full-residual swap at the last token of the target cell)",
                "- A_latent / A_decode = CENTROID patch: replace residual with the per-target-op mean activation.",
                "- C_latent / C_decode = CROSS patch: replace residual with a real activation from a target-op partner",
                "  (matched on magnitude bucket).",
                "",
                "# Flip-rate metric",
                "- For src-op problems with single-op Equation '(a op b)', count cases where the steered prediction",
                "  equals target_op(a, b) (= moved to target) vs source_op(a, b) (= preserved).",
            ])

        # === Slide 2: operator probe accuracy (decode positions only — we have that table) ===
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(op_acc_decode.T, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0.25, vmax=1.0)
        ax.set_title("Operator probe accuracy across decode positions × layer (gpt2_multipos_probes.json)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("decode position (0=EOT, 1=first emit, …)")
        ax.set_ylabel("layer")
        ax.set_xticks(range(op_acc_decode.shape[0]))
        ax.set_yticks(range(op_acc_decode.shape[1]))
        for p in range(op_acc_decode.shape[0]):
            for l in range(op_acc_decode.shape[1]):
                v = op_acc_decode[p, l]
                if v > 0.7:
                    ax.text(p, l, f"{v:.2f}", ha="center", va="center",
                            color="black" if v > 0.85 else "white", fontsize=6)
        # Mark the chosen decode cell
        ax.scatter([1], [8], s=200, facecolors="none", edgecolors="red", linewidths=2,
                   label="chosen DECODE cell (pos=1, layer=8)")
        ax.legend(loc="lower right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="probe acc")
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 3: A_latent / C_latent ===
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        flip_heatmap(axes[0], M_A_lat, "A_latent: centroid patch @ (step 4, layer 10)")
        flip_heatmap(axes[1], M_C_lat, "C_latent: cross patch  @ (step 4, layer 10)")
        fig.suptitle("Operator flip rates at the LATENT cell  (fraction of src problems that moved to target_op(a,b))",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 4: A_decode / C_decode ===
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        flip_heatmap(axes[0], M_A_dec, "A_decode: centroid patch @ (pos 1, layer 8)")
        flip_heatmap(axes[1], M_C_dec, "C_decode: cross patch  @ (pos 1, layer 8)")
        fig.suptitle("Operator flip rates at the DECODE cell  (probe acc here = 92.9%)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 5: max across methods ===
        fig, ax = plt.subplots(figsize=(7, 5))
        flip_heatmap(ax, M_max, "MAX flip rate across all 4 methods × cells")
        fig.suptitle("Best-case operator flip per (src → tgt)", fontsize=11, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 6: attention from latent cell ===
        for tag, attn, meta, cue_head, top1 in [
            ("LATENT (step 4, layer 10)", attn_lat, meta_lat, cue_lat_head, top1_lat),
            ("DECODE (pos 1, layer 8)",  attn_dec, meta_dec, cue_dec_head, top1_dec),
        ]:
            N, Hd, K = attn.shape
            fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.5))
            # Left: per-(op, head) cue attention bar
            ax = axes[0]
            width = 0.18
            xs = np.arange(Hd)
            for k, op in enumerate(OPS):
                ax.bar(xs + (k - 1.5) * width, cue_head[op], width,
                       label=OP_SHORT[op])
            ax.set_xticks(xs); ax.set_xticklabels([f"H{h}" for h in range(Hd)])
            ax.set_xlabel("attention head")
            ax.set_ylabel("mean cumulative attention to operator-cue tokens")
            ax.set_title(f"{tag}: cue-attention per head, by source op",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            # Right: top-1 attended tokens (mean over heads), top 15
            ax = axes[1]
            items = top1.most_common(15)
            toks = [t for t, _ in items][::-1]
            counts = [c for _, c in items][::-1]
            ax.barh(toks, counts, color="#3357aa")
            ax.set_xlabel(f"# examples (of {N}) where this is the top-attended token")
            ax.set_title(f"{tag}: most-attended tokens (mean over heads)",
                         fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 8: synthesis ===
        # Compute summary numbers
        def summarize(M):
            arr = M[~np.isnan(M)]
            return f"max={arr.max()*100:.1f}%  median={np.median(arr)*100:.2f}%  >5%: {int((arr>0.05).sum())}/{arr.size}"
        text_slide(pdf, "Synthesis: operator is encoded but not steerable here",
            [
                "# Headline numbers",
                f"- A_latent   ({d['target_step_1indexed']} / {d['target_layer']}): {summarize(M_A_lat)}",
                f"- C_latent   (same cell): {summarize(M_C_lat)}",
                f"- A_decode   (pos 1, layer 8): {summarize(M_A_dec)}",
                f"- C_decode   (same cell): {summarize(M_C_dec)}",
                f"- MAX across all 4: {summarize(M_max)}",
                "",
                "# Pattern",
                "- 11 of 12 src→tgt pairs flip ≤2% under every method at every cell.",
                "- Only persistent outlier: Sub → Common-Division (5.7-6.7% across A/C × latent/decode).",
                "- 'Preserved' count (src answer still emitted) ranges 20-61% — the model is robust to the patch,",
                "  it's not just outputting garbage.",
                "",
                "# Attention pattern",
                "- At the LATENT cell (step 4, layer 10), heads 1, 2, 6, 9 attend most to operator-cue words —",
                "  but the cumulative cue attention is only 2-5% of total mass.",
                "- At the DECODE cell (pos 1, layer 8), top-1 attention shifts toward question-structure words",
                "  ('many', 'how', 'much', 'the'). Cue attention is similarly diffuse.",
                "- The model is NOT reading operator from a small set of cue tokens at any single head/position.",
                "",
                "# Interpretation",
                "- 92.9% operator-probe accuracy at decode pos 1 / layer 8 confirms encoding.",
                "- Patching that same residual fully (centroid OR real partner activation) doesn't flip the op.",
                "- Operator is therefore encoded REDUNDANTLY: distributed across token positions AND",
                "  layers AND attention paths, in a way that a single-position residual swap can't dislodge.",
                "",
                "# What might work",
                "- DAS / orthogonal-subspace intervention (rotate to find a small subspace that IS causal).",
                "- Multi-position patching: swap residuals across all decode positions and all layers simultaneously.",
                "- Attention-pattern intervention at the question-word positions, not the latent/decode stream.",
                "- Input-space rewriting (replace 'added' with 'minus') — the lower bound on what 'changing the",
                "  operator' even means; useful as a sanity floor.",
            ])

    print(f"saved {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
