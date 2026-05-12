"""GSM8K research poster — rebuilt with proper layout.

Strategy:
  - Title + Methods + Problem-statement text rendered as native PDF text
    (editable in Acrobat/Illustrator).
  - Figures EXTRACTED from existing source slideshows in master_evidence
    via pdf2image, then embedded as high-DPI PNGs. This way the figures
    are EXACTLY the ones from the agreed-upon master compendium; we do
    not redraw or re-style them.
  - The master_evidence PDF itself is untouched — we only READ from its
    constituent source PDFs.

Output: poster_gsm8k.pdf  (33×44 inch portrait, single page, editable text)
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42        # editable text
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

PD = Path(__file__).resolve().parent
OUT_PDF = PD / "poster_gsm8k.pdf"

MAROON = "#7a1f2b"
PALE = "#e8d8d8"


def page_image(rel_path: str, page_idx: int, dpi=180):
    """Extract a specific page of a source PDF as a PIL Image."""
    full = PD / rel_path
    images = convert_from_path(full, dpi=dpi, first_page=page_idx + 1,
                               last_page=page_idx + 1)
    return images[0]


def add_image(fig, x, y, w, h, img):
    """Add a PIL image to the figure at normalized (x, y, w, h)."""
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(img); ax.axis("off")
    return ax


def banner(fig, x, y, w, h, title, color=MAROON, text_color="white", fs=22):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=color,
                                     transform=ax.transAxes))
    ax.text(0.5, 0.5, title, ha="center", va="center",
            color=text_color, fontsize=fs, fontweight="bold",
            transform=ax.transAxes)


def section_box(fig, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def col_header(ax, y, text, fs=15):
    ax.add_patch(mpatches.Rectangle((0, y - 0.005), 1, 0.025, color=PALE,
                                     transform=ax.transAxes))
    ax.text(0.5, y + 0.0075, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#333",
            transform=ax.transAxes)


def text_block(ax, x, y, w, bullets, body_fs=10.5, label_fs=11.5, leading=0.012, dx=0.014):
    """bullets: list of (label, body_str). Returns y after."""
    cur_y = y
    for label, body in bullets:
        ax.text(x, cur_y, "■", ha="left", va="top",
                fontsize=label_fs, color=MAROON, fontweight="bold",
                transform=ax.transAxes)
        ax.text(x + dx, cur_y, label, ha="left", va="top",
                fontsize=label_fs, fontweight="bold", color="#000",
                transform=ax.transAxes)
        body_lines = body.split("\n")
        line_y = cur_y - 0.018
        for line in body_lines:
            ax.text(x + dx, line_y, line, ha="left", va="top",
                    fontsize=body_fs, color="#222",
                    transform=ax.transAxes)
            line_y -= 0.0145
        cur_y = line_y - leading
    return cur_y


def fig_caption(ax, y, caption, fs=9):
    ax.text(0.5, y, caption, ha="center", va="top",
            fontsize=fs, color="#444", style="italic",
            transform=ax.transAxes, wrap=True)


def main():
    print("extracting figures from source PDFs (this takes ~30s)...", flush=True)
    # Map of which page of which source we want
    figs = {}

    # RQ1 figures
    # Force-decode + Q1 chart from step_anatomy: page 2 (Q1 panel)
    figs["step_q1"]      = page_image("step_anatomy_gsm8k.pdf", 2)
    figs["step_q2_attn"] = page_image("step_anatomy_gsm8k.pdf", 3)
    figs["op_probe"]     = page_image("operator-probe/multi_op_probe_gsm8k.pdf", 1)
    figs["op_refined"]   = page_image("operator-probe/multi_op_probe_refined_gsm8k.pdf", 1)
    figs["shuffle"]      = page_image("operator-probe/multi_op_probe_shuffle_gsm8k.pdf", 4)
    figs["evenodd"]      = page_image("operator-probe/multi_op_probe_evenodd_gsm8k.pdf", 3)
    figs["logit_traj"]   = page_image("logit_lens_slideshow_gsm8k.pdf", 5)

    # RQ2 figures
    figs["transitions"]  = page_image("step_transitions_gsm8k.pdf", 1)
    figs["chaining"]     = page_image("mode1_chaining_gsm8k.pdf", 1)
    figs["late_recovery"]= page_image("late_step_behavior_gsm8k.pdf", 1)
    figs["sublayer_step2"]= page_image("sublayer_breakdown_gsm8k.pdf", 7)

    # Correctness figures
    figs["correctness"]  = page_image("operator-probe/correctness_probe_gsm8k_v2.pdf", 3)
    figs["steering_null"]= page_image("steering/steer_attn_op_token_gsm8k.pdf", 1)
    figs["benford"]      = page_image("final_wrong_patterns_gsm8k.pdf", 1)
    print("  extracted", len(figs), "figures.", flush=True)

    # ===== Build poster =====
    # Portrait 33×44 inches (standard academic)
    fig = plt.figure(figsize=(33, 44))

    # ----- HEADER (top 7%) -----
    banner(fig, 0, 0.965, 1, 0.035,
           "Latent Reasoning Interpretability in CODI on GSM8K",
           fs=34)
    sub = fig.add_axes([0, 0.935, 1, 0.030])
    sub.axis("off")
    sub.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=MAROON,
                                      transform=sub.transAxes))
    sub.text(0.5, 0.65, "Grad NLP Final Project",
             ha="center", va="center", color="white",
             fontsize=14, transform=sub.transAxes)
    sub.text(0.5, 0.30, "Karen Li, Sandra Luo, Hannah Tao",
             ha="center", va="center", color="white",
             fontsize=14, transform=sub.transAxes)

    # ----- Row 1: PROBLEM STATEMENT + METHODS -----
    banner(fig, 0.005, 0.905, 0.49, 0.025, "PROBLEM STATEMENT",
           color=MAROON, text_color="white", fs=16)
    banner(fig, 0.505, 0.905, 0.49, 0.025, "METHODS",
           color=MAROON, text_color="white", fs=16)

    # Problem statement panel
    ps = section_box(fig, 0.01, 0.69, 0.48, 0.20)
    y = 0.96
    y = text_block(ps, 0.01, y, 0.98, [
        ("Motivation.",
         "CoT prompting improves multi-step reasoning but at significant inference\n"
         "cost. CODI distills CoT into a fixed K=6 latent loop — no explicit\n"
         "token-level reasoning chain."),
        ("Setup.",
         "CODI-GPT-2 (small) trained on GSM8K. K=6 latent thoughts between\n"
         "<BOT> and <EOT>. The shared LM is both teacher (text CoT) and student\n"
         "(continuous latents). Each latent is the prior step's last-token hidden\n"
         "state fed back as an embedding (no decoding)."),
        ("Problem.",
         "Latent reasoning is unreadable. If models reason this way at scale, we\n"
         "lose the audit-trail that explicit CoT provides."),
        ("Research questions.",
         "RQ1: What is the role of the six latent thoughts?\n"
         "RQ2: How does the latent model compute the answer?\n"
         "RQ3: Can we causally control its output via single-cell steering?"),
    ], body_fs=11, label_fs=12.5)

    # Methods panel
    mt = section_box(fig, 0.51, 0.69, 0.48, 0.20)
    y = 0.96
    y = text_block(mt, 0.01, y, 0.98, [
        ("Setup.",
         "Model: public CODI-GPT-2 checkpoint, 12 blocks × 12 heads, H=768, K=6\n"
         "latent steps; greedy decoding; no fine-tuning."),
        ("Benchmarks.",
         "GSM8K test (N=1319). Auxiliary CF datasets we built to disentangle\n"
         "operators from operands: gsm8k_vary_operator (template-controlled),\n"
         "gsm8k_cf_op_strict, gsm8k_cf_natural."),
        ("Forced decoding.",
         "Cap the latent loop at step k ∈ {1..6}, decode, and measure accuracy.\n"
         "Captures how much each latent step actually contributes to the answer."),
        ("Multi-op probes.",
         "RidgeClassifier per (step, layer, marker position) predicting marker m's\n"
         "operator or last-digit. Class-balanced; shuffle-step null baseline."),
        ("Logit lens.",
         "Apply final LayerNorm + LM head at every (step, layer, sublayer) hidden\n"
         "state. Track which token is being predicted at every cell of the loop."),
        ("Activation patching + linear steering.",
         "Paired-CF interchange / zero / mean ablation at (step, layer, block).\n"
         "Linear steering: add α·v to residual or attention output at probe-winning\n"
         "cells, with v ∈ {LDA, vary-op template-difference, Adam-trained,\n"
         "LM-head row-difference}. Sweep α and measure operator-emit flip rate."),
        ("Correctness probe.",
         "Per-(step, layer) RidgeClassifier predicting whether the final force-\n"
         "decoded answer will be correct. Tests whether the model's residuals\n"
         "encode its own success / failure."),
    ], body_fs=10.5, label_fs=12)

    # ----- RESULTS banner -----
    banner(fig, 0, 0.660, 1, 0.030,
           "RESULTS  —  CODI-GPT-2 on GSM8K", fs=22)

    # ----- 3 result columns -----
    col_y, col_h = 0.035, 0.620
    cw = 0.32                     # column width
    cx = [0.005, 0.340, 0.675]    # x positions

    # ============== COLUMN 1: RQ1 ==============
    c1 = section_box(fig, cx[0], col_y, cw, col_h)
    col_header(c1, 0.972, "RQ1: Role of the six latent thoughts", fs=17)

    # Text bullets
    y = 0.95
    y = text_block(c1, 0.01, y, 0.98, [
        ("Even/odd storage vs computation.",
         "Logit-lens at L11 reveals a striking alternating rhythm. Odd steps\n"
         "(1, 3, 5) produce '>>' (marker close) as the modal top-1 token at the\n"
         "last layer; even steps (2, 4, 6) produce a digit (' 40', ' 9').\n"
         "→ The latent loop literally tries to emit <<a op b = c>> markers."),
    ], body_fs=10, label_fs=11.5)

    add_image(fig, cx[0] + 0.005 * cw, col_y + 0.84 * col_h,
              0.99 * cw, 0.08 * col_h, figs["logit_traj"])
    fig_caption(c1, 0.82,
                "Logit-lens at last layer L11 — modal tokens per latent step. "
                "Odd steps commit to '>>'; even steps to digits.")

    y = 0.79
    y = text_block(c1, 0.01, y, 0.98, [
        ("Operator info is highly decodable from CODI's residuals.",
         "4-class RidgeClassifier (`+, −, ×, ÷`) trained per (step, layer)\n"
         "cell to predict marker m's gold operator from the last-token\n"
         "residual. m=1..4 = position in the gold marker chain (<<a op b = c>>).\n"
         "Class-balanced; stratified 80/20 split per cell. Chance = 0.25.\n"
         "\n"
         "Best-cell acc on unrestricted GSM8K test (N=322-1264/marker):\n"
         "  m=1: step1/L11 = 0.573    m=3: step2/L7 = 0.485\n"
         "  m=2: step2/L0  = 0.493    m=4: step3/L8 = 0.569\n"
         "\n"
         "Filter to CODI-correct AND chain-length = m (N=46-246/marker):\n"
         "  m=1: 0.812   m=2: 0.800   m=3: 0.806   m=4: 0.900   (+10-22pp)\n"
         "→ The clean operator signal lives in genuine-compute cases.\n"
         "\n"
         "Shuffle-step null (per-example random permutation of 6 step indices,\n"
         "3 seeds): real beats shuffled by Δ +0.07-0.17pp — step INDEX itself\n"
         "carries marker-position information, not just step content."),
    ], body_fs=10, label_fs=11.5)
    add_image(fig, cx[0] + 0.01 * cw, col_y + 0.50 * col_h,
              0.98 * cw, 0.20 * col_h, figs["op_probe"])
    fig_caption(c1, 0.485,
                "Operator probe heatmaps per marker (4 panels). Bright cells = "
                "high probe accuracy. Best cells starred per marker.")

    y = 0.46
    y = text_block(c1, 0.01, y, 0.98, [
        ("Step ordering carries marker-position info.",
         "Shuffle-step null baseline: per-example random permutation of the 6\n"
         "step indices. Real beats shuffled by Δ +0.07-0.17pp on op and c_ld\n"
         "probes. Step index, not just step content, matters per marker."),
        ("Cross-marker selectivity.",
         "Best (step, layer) cells are weakly marker-specific (diagonal\n"
         "advantage +0.04 to +0.15pp). Cells encode operator structure\n"
         "generically with mild per-marker specialization."),
    ], body_fs=10, label_fs=11.5)
    add_image(fig, cx[0] + 0.01 * cw, col_y + 0.18 * col_h,
              0.98 * cw, 0.16 * col_h, figs["evenodd"])
    fig_caption(c1, 0.165,
                "Per-cell argmax-marker + selectivity (Δ acc). "
                "c_ld concentrates on odd steps (1, 3, 5) for all markers.")

    y = 0.13
    y = text_block(c1, 0.01, y, 0.98, [
        ("Attention writes the LM-head-aligned tokens; MLP role is opaque.",
         "Logit-lens per sublayer: attn-out at L9-L11 commits to ' *' (step 1\n"
         "L10, 0.58 conf), ' <<' (step 3 L10, 0.53), digits (even steps). MLP-\n"
         "out at the same cells projects to non-vocab-aligned directions —\n"
         "typical for transformer MLPs; doesn't mean they're 'noise', just that\n"
         "their contribution isn't readable via logit lens. Single-cell mean-\n"
         "ablations of either block change acc by <1.3pp (max), so the compute\n"
         "is distributed; neither block alone is causally critical at any cell."),
    ], body_fs=9.5, label_fs=11.5)

    # ============== COLUMN 2: RQ2 ==============
    c2 = section_box(fig, cx[1], col_y, cw, col_h)
    col_header(c2, 0.972, "RQ2: How does the model compute?", fs=17)

    y = 0.95
    y = text_block(c2, 0.01, y, 0.98, [
        ("Two calculation windows.",
         "Force-decode acc per step: 25%→23%→37%→37%→41%→42%. Biggest jump\n"
         "is step 2→3 (+13.8pp); a second jump at 4→5 (+4pp). Step 1→2 is\n"
         "actually a net REGRESSION (−30 net w↔r transitions)."),
        ("Commit-then-synthesize rhythm.",
         "Step 2 has the loop's largest MLP write (norm 564 vs 419-535\n"
         "elsewhere). Step 3 then attends back to L2 (13% — the loop's\n"
         "strongest step→prior-latent attention). Step 4→5 repeats it (L4 →\n"
         "step 5 = 12%)."),
    ], body_fs=10, label_fs=11.5)
    add_image(fig, cx[1] + 0.01 * cw, col_y + 0.66 * col_h,
              0.98 * cw, 0.16 * col_h, figs["transitions"])
    fig_caption(c2, 0.655,
                "Per-example w→r / r→w transitions per consecutive step. "
                "Step 1→2 is a NET REGRESSION (−30); step 2→3 rescues +182.")

    y = 0.62
    y = text_block(c2, 0.01, y, 0.98, [
        ("After step 3, almost no recovery.",
         "Of 838 problems wrong at step 3, only 117 (14%) recover by step 6.\n"
         "Wrong-final emits then stabilize: 48% have ONE unique value across\n"
         "steps 4-6; 93% have ≤ 2. Approximation mode kicks in."),
    ], body_fs=10, label_fs=11.5)
    add_image(fig, cx[1] + 0.01 * cw, col_y + 0.36 * col_h,
              0.98 * cw, 0.20 * col_h, figs["late_recovery"])
    fig_caption(c2, 0.345,
                "Recovery rate by 'first-wrong' step (left); late rescue dominated "
                "by longer chains (right, n_markers stacked).")

    y = 0.31
    y = text_block(c2, 0.01, y, 0.98, [
        ("Step chaining — sequential but compressed.",
         "Among Mode 1 (loop-rescued) ≥2-marker problems: avg first-emit\n"
         "step grows monotonically with marker position — m=1→1.64,\n"
         "m=2→2.74, m=3→3.10, m=4→3.67. Chain order is real (9/10\n"
         "monotone for 2-marker chains); gaps are compressed."),
        ("Mostly INTERNAL chaining.",
         "67% of Mode 1 problems emit only ONE marker's c-value (usually\n"
         "the final). The chain happens in residuals, not in emits."),
    ], body_fs=10, label_fs=11.5)
    add_image(fig, cx[1] + 0.01 * cw, col_y + 0.045 * col_h,
              0.98 * cw, 0.10 * col_h, figs["chaining"])
    fig_caption(c2, 0.025,
                "Mode 1 chaining: avg first-step per marker grows monotonically. "
                "Black stars = '2m−1' sequential prediction; data is compressed.")

    # ============== COLUMN 3: Correctness ==============
    c3 = section_box(fig, cx[2], col_y, cw, col_h)
    col_header(c3, 0.972, "Correctness — compute or approximate?", fs=17)

    y = 0.95
    y = text_block(c3, 0.01, y, 0.98, [
        ("Two operating regimes.",
         "CODI either (1) computes via the loop, or (2) approximates with a\n"
         "Benford-shaped guess. 551/1319 correct (41.8%) split as 280 by\n"
         "loop work (Mode 1) and 271 already at step 1 (Mode 2 — shortcut)."),
        ("Correctness probe — the model knows when it's wrong.",
         "Per-(step, layer) RidgeClassifier on the final-emit correctness\n"
         "label rises 66% → 78% (step 1 → step 6). Majority baseline = 58%\n"
         "→ +20pp at step 6; +8pp already at step 1 (pre-compute signal)."),
    ], body_fs=10, label_fs=11.5)
    add_image(fig, cx[2] + 0.01 * cw, col_y + 0.66 * col_h,
              0.98 * cw, 0.16 * col_h, figs["correctness"])
    fig_caption(c3, 0.655,
                "Correctness probe: best-over-layer acc per step. "
                "Probe 1 (predict final) rises 66% → 78%; probe 2 (this-step) similar.")

    y = 0.62
    y = text_block(c3, 0.01, y, 0.98, [
        ("Wrong emits match gold's distribution.",
         "Final wrong answers: 37% end in 0, 45% mult-of-5 (gold: 46%); first-\n"
         "digit follows Benford. Median |emit − gold| = 22 (0.5× relative).\n"
         "The model has learned the DISTRIBUTION of GSM8K answers without\n"
         "necessarily solving each problem."),
    ], body_fs=10, label_fs=11.5)
    add_image(fig, cx[2] + 0.01 * cw, col_y + 0.42 * col_h,
              0.98 * cw, 0.16 * col_h, figs["benford"])
    fig_caption(c3, 0.415,
                "Top wrong-final values. Most-common: 2, 4, 30, 5, 20, 60. "
                "Wrong answers are Benford-shaped natural-looking numbers.")

    y = 0.39
    y = text_block(c3, 0.01, y, 0.98, [
        ("Operation steering has NO effect (7 experiments).",
         "At probe-winning cells, add α·v to residual / attn-out. Vectors:\n"
         "LDA centroid, vary-op template-diff, Adam-optimized, LM-head row-\n"
         "diff. ~5000 steered runs at α magnitudes up to 100× direction norm.\n"
         "ZERO operator flips. Trained v makes loss drop 21.5→13.6 but\n"
         "argmax doesn't move. Probes find structure; steering doesn't."),
        ("Why? — 72-91% of attention goes back to Q.",
         "The latent loop is mostly question re-encoding. Modifications at\n"
         "any single cell get washed out by the next step's re-attention to\n"
         "the question tokens."),
    ], body_fs=10, label_fs=11.5)
    add_image(fig, cx[2] + 0.01 * cw, col_y + 0.04 * col_h,
              0.98 * cw, 0.16 * col_h, figs["steering_null"])
    fig_caption(c3, 0.025,
                "Last-attempt steering using the LM head's OWN row-difference at "
                "the cell where attn-out modal token is '*': 0 flips across α ∈ [−3, +12].")

    # ----- Footer -----
    banner(fig, 0, 0.0, 1, 0.030,
           "Take-home: probes find rich operator/marker/correctness signal in CODI's "
           "residuals (op 0.80–0.90, correctness 78%, +20pp baseline) — but linear "
           "single-cell steering can't move the emit. Compute rhythm = step 2 commit "
           "→ step 3 synthesize → step 4 commit → step 5 synthesize → approximation "
           "(Benford-shaped wrong guesses).",
           fs=11)

    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
