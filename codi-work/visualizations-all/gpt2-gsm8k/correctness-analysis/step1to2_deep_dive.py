"""Deep dive into the step 1->2 transition on CODI-GPT-2 / SVAMP.

Four CPU-only probes (the per-layer step-2 zero-ablation is a separate GPU
script: step2_layer_ablate.py).

1. Trajectory diff per layer:
   delta_l = mean(act[i, step=2, l] - act[i, step=1, l])  (over all examples)
   Apply ln_f + lm_head -> top tokens "added" from step 1 to step 2 at each layer.

2. w->r vs r->w cohort comparison:
   Use force_decode_per_step.json's correct_per_step.
   wr = examples that went wrong-at-step-1 -> right-at-step-2 (70 examples)
   rw = examples that went right-at-step-1 -> wrong-at-step-2 (34)
   For each layer, compare mean(delta) for wr cohort vs rw cohort. Show top
   tokens for each cohort's mean delta.

3. Attention-to-L1 correlation with w->r:
   Use flow_map.npz: per-example we have aggregated cell attention. Actually
   flow_map only stored MEANS, not per-example. So we can do this only at
   the aggregate level.  Instead we'll show: mean attention to L1 at step 2
   per layer (aggregate), and report the cohort breakdown using
   attention_operator_latent.npz if usable.

4. (Reported here from prior heatmap): at step 2 layer 0, ~10% of
   attention goes to L1; this is the dominant cross-step communication
   pathway from step 1.

Output: step1to2_deep_dive.{json,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
LM_HEAD = PD / "codi_gpt2_lm_head.npy"
LN_F = PD / "codi_gpt2_ln_f.npz"
FD = PD / "force_decode_per_step.json"
FLOW = PD / "flow_map.npz"
OUT_JSON = PD / "step1to2_deep_dive_gsm8k.json"
OUT_PDF = PD / "step1to2_deep_dive_gsm8k.pdf"


def apply_ln_f(h, gamma, beta):
    mean = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    normed = (h - mean) / np.sqrt(var + 1e-5)
    return normed * gamma + beta


def top_tokens(vec, W, gamma, beta, tok, k=8):
    """vec: (H,). Apply ln_f + lm_head and return top-k decoded tokens."""
    h = apply_ln_f(vec[None, :], gamma, beta)[0]
    logits = h @ W.T
    top_ids = np.argsort(-logits)[:k]
    return [(tok.decode([int(i)]), float(logits[i])) for i in top_ids]


def main():
    print("loading data", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    W = np.load(LM_HEAD)
    ln = np.load(LN_F); gamma, beta = ln["weight"], ln["bias"]
    fd = json.load(open(FD))
    correct = np.array(fd["correct_per_step"])    # (S=K_max, N)
    correct_at_1 = correct[0].astype(bool)
    correct_at_2 = correct[1].astype(bool)
    wr_mask = (~correct_at_1) & correct_at_2     # wrong at 1, right at 2
    rw_mask = correct_at_1 & (~correct_at_2)     # right at 1, wrong at 2
    same_mask = (correct_at_1 == correct_at_2)
    print(f"  N={N}  L={L}")
    print(f"  wr={int(wr_mask.sum())}  rw={int(rw_mask.sum())}  same={int(same_mask.sum())}")
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # --- Probe 1: per-layer trajectory diff (overall) ---
    delta_overall = a[:, 1, :, :] - a[:, 0, :, :]    # (N, L, H)
    delta_mean_overall = delta_overall.mean(axis=0)   # (L, H)
    delta_norm_overall = np.linalg.norm(delta_mean_overall, axis=-1)

    print("\nProbe 1: per-layer mean (step2 - step1), top tokens of delta:", flush=True)
    layer_top_tokens = {}
    for l in range(L):
        toks = top_tokens(delta_mean_overall[l], W, gamma, beta, tok, k=5)
        layer_top_tokens[l] = toks
        joined = " / ".join(f"{t[0]!r}" for t in toks)
        print(f"  L{l:2d}  ‖Δ‖={delta_norm_overall[l]:5.1f}  top: {joined}")

    # --- Probe 2: cohort comparison (wr vs rw) ---
    delta_mean_wr = a[wr_mask, 1, :, :].mean(axis=0) - a[wr_mask, 0, :, :].mean(axis=0)
    delta_mean_rw = a[rw_mask, 1, :, :].mean(axis=0) - a[rw_mask, 0, :, :].mean(axis=0)
    norm_wr = np.linalg.norm(delta_mean_wr, axis=-1)
    norm_rw = np.linalg.norm(delta_mean_rw, axis=-1)
    cohort_top_wr = {}
    cohort_top_rw = {}
    print("\nProbe 2: wr (70) vs rw (34) cohorts — top tokens of mean delta per layer:")
    for l in range(L):
        twr = top_tokens(delta_mean_wr[l], W, gamma, beta, tok, k=4)
        trw = top_tokens(delta_mean_rw[l], W, gamma, beta, tok, k=4)
        cohort_top_wr[l] = twr; cohort_top_rw[l] = trw
        print(f"  L{l:2d}  ‖Δwr‖={norm_wr[l]:5.1f}  wr top: " +
              " / ".join(f"{t[0]!r}" for t in twr))
        print(f"       ‖Δrw‖={norm_rw[l]:5.1f}  rw top: " +
              " / ".join(f"{t[0]!r}" for t in trw))

    # Difference of cohort deltas: what's in wr that's NOT in rw?
    print("\n  (Δwr − Δrw) top tokens per layer — what successful transitions specifically add:")
    diff_top = {}
    for l in range(L):
        diff = delta_mean_wr[l] - delta_mean_rw[l]
        td = top_tokens(diff, W, gamma, beta, tok, k=4)
        diff_top[l] = td
        print(f"  L{l:2d}  top: " + " / ".join(f"{t[0]!r}" for t in td))

    # --- Probe 4: attention from step-2 last token to L1 vs Q vs other ---
    # flow_map.npz aggregates over examples; we report the mean attention
    # distribution at step 2 per (layer, head, class).
    fm = np.load(FLOW)
    A = fm["mean_attn"]   # (phase=2, step=10, layer=12, head=12, class=19)
    # class names: ["Q", "BOT", "L1"...] index 0..6
    step = 1   # step 2 in 1-indexed -> 0-indexed = 1
    attn_to_Q = A[0, step, :, :, 0].mean(axis=-1)   # (L,) mean over heads
    attn_to_L1 = A[0, step, :, :, 2].mean(axis=-1)
    attn_to_BOT = A[0, step, :, :, 1].mean(axis=-1)

    print(f"\nProbe 4: step 2 attention from latent_2 to {{Q, BOT, L1}} per layer:")
    print(f"  layer:  " + "  ".join(f"L{l:2d}" for l in range(L - 1)))
    print(f"  to Q:   " + "  ".join(f"{v*100:4.1f}" for v in attn_to_Q))
    print(f"  to BOT: " + "  ".join(f"{v*100:4.1f}" for v in attn_to_BOT))
    print(f"  to L1:  " + "  ".join(f"{v*100:4.1f}" for v in attn_to_L1))

    # Save
    OUT_JSON.write_text(json.dumps({
        "N": int(N), "n_wr": int(wr_mask.sum()), "n_rw": int(rw_mask.sum()),
        "n_same": int(same_mask.sum()),
        "delta_norm_overall": delta_norm_overall.tolist(),
        "delta_norm_wr": norm_wr.tolist(),
        "delta_norm_rw": norm_rw.tolist(),
        "layer_top_tokens": {l: [(t[0], t[1]) for t in v] for l, v in layer_top_tokens.items()},
        "cohort_top_wr": {l: [(t[0], t[1]) for t in v] for l, v in cohort_top_wr.items()},
        "cohort_top_rw": {l: [(t[0], t[1]) for t in v] for l, v in cohort_top_rw.items()},
        "diff_wr_minus_rw_top": {l: [(t[0], t[1]) for t in v] for l, v in diff_top.items()},
        "attn_to_Q_step2": attn_to_Q.tolist(),
        "attn_to_BOT_step2": attn_to_BOT.tolist(),
        "attn_to_L1_step2": attn_to_L1.tolist(),
    }, indent=2))

    # --- Slideshow ---
    print(f"\nwriting {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        # Slide 1: delta norm per layer (overall + cohorts)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].plot(range(L), delta_norm_overall, "o-", lw=2, color="#1f77b4", label="all examples")
        axes[0].plot(range(L), norm_wr, "s-", color="#2ca02c", label=f"wrong→right (n={int(wr_mask.sum())})")
        axes[0].plot(range(L), norm_rw, "^-", color="#d62728", label=f"right→wrong (n={int(rw_mask.sum())})")
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("||Δ residual||  (step 2 − step 1)")
        axes[0].set_title("Magnitude of step-1→step-2 residual change per layer", fontsize=10)
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
        # Attention from step 2 per layer
        axes[1].plot(range(L - 1), attn_to_Q * 100, "o-", color="#1f77b4", label="to Q")
        axes[1].plot(range(L - 1), attn_to_L1 * 100, "s-", color="#2ca02c", label="to L1")
        axes[1].plot(range(L - 1), attn_to_BOT * 100, "^-", color="#ff7f0e", label="to BOT")
        axes[1].set_xlabel("layer"); axes[1].set_ylabel("mean attention (%)")
        axes[1].set_title("Step-2 latent token's attention by class per layer", fontsize=10)
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 100)
        fig.suptitle("Step 1 → 2 transition: where does the residual change & where does step 2 attend?",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 2: top tokens added per layer (overall)
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle("Per-layer top decoded tokens of mean (step 2 − step 1) residual",
                     fontsize=12, fontweight="bold")
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.90]); ax.axis("off")
        y = 0.95
        ax.text(0.0, y, f"{'layer':>5}  {'‖Δ‖':>6}  top 5 tokens 'added' by the transition",
                fontsize=10, fontweight="bold", family="monospace", transform=ax.transAxes)
        y -= 0.045
        for l in range(L):
            toks = " / ".join(f"{t[0]!r}" for t in layer_top_tokens[l][:5])
            ax.text(0.0, y, f"  L{l:>2}  {delta_norm_overall[l]:>5.1f}  {toks}",
                    fontsize=9, family="monospace", transform=ax.transAxes)
            y -= 0.060
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 3: cohort comparison (wr vs rw) — text columns
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle("wr (wrong→right, 70 ex) vs rw (right→wrong, 34 ex) — what each cohort 'adds' from step 1 to step 2",
                     fontsize=11, fontweight="bold")
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.90]); ax.axis("off")
        y = 0.94
        ax.text(0.00, y, f"{'L':>3} | {'wr top tokens (wrong→right)':<50} | {'rw top tokens (right→wrong)':<50}",
                fontsize=9, fontweight="bold", family="monospace", transform=ax.transAxes)
        y -= 0.042
        for l in range(L):
            wr_s = " / ".join(f"{t[0]!r}" for t in cohort_top_wr[l][:3])
            rw_s = " / ".join(f"{t[0]!r}" for t in cohort_top_rw[l][:3])
            ax.text(0.00, y, f"L{l:>2} | {wr_s[:50]:<50} | {rw_s[:50]:<50}",
                    fontsize=8.5, family="monospace", transform=ax.transAxes)
            y -= 0.055
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 4: diff (wr − rw) — tokens that successful transitions specifically add
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle("(Δ_wr − Δ_rw) — tokens that successful 1→2 transitions write but failed ones don't",
                     fontsize=11, fontweight="bold")
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.90]); ax.axis("off")
        y = 0.94
        ax.text(0.0, y, f"{'L':>3}  top 4 tokens in (Δ_wr − Δ_rw)",
                fontsize=10, fontweight="bold", family="monospace", transform=ax.transAxes)
        y -= 0.042
        for l in range(L):
            toks = " / ".join(f"{t[0]!r}" for t in diff_top[l][:4])
            ax.text(0.0, y, f"L{l:>2}  {toks}",
                    fontsize=9, family="monospace", transform=ax.transAxes)
            y -= 0.060
        pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_JSON}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
