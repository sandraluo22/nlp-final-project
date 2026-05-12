"""Master evidence compendium: merge informative visualizations from all
existing PDFs into one master PDF, with claim-titled section separators.

Each claim gets a title-only page (rendered via matplotlib + PdfPages), then
the relevant PDF pages from existing slideshows are merged in via pypdf.

Output: gpt2-gsm8k/master_evidence_gsm8k.pdf
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

PD = Path(__file__).resolve().parent
OUT_PDF = PD / "master_evidence_gsm8k.pdf"


def make_title_pdf(title, subtitle="", footnote=""):
    """Create a single-page PDF as bytes containing a title slide."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.65, title, ha="center", va="center",
                fontsize=22, fontweight="bold", transform=ax.transAxes,
                wrap=True)
        if subtitle:
            ax.text(0.5, 0.45, subtitle, ha="center", va="center",
                    fontsize=14, transform=ax.transAxes,
                    family="monospace", color="#444")
        if footnote:
            ax.text(0.5, 0.10, footnote, ha="center", va="center",
                    fontsize=10, transform=ax.transAxes,
                    family="monospace", color="#666")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    buf.seek(0)
    return buf


# Sections: (title, subtitle, [list of (source_pdf, pages-or-None)])
# pages: None = include all; or list of 0-indexed page numbers
SECTIONS = [
    # ===== Title =====
    ("CODI-GPT-2 on GSM8K — Master Evidence Compendium",
     "Visualizations from every analysis we built.\nEach claim is preceded by a title slide.",
     "Section dividers separate descriptive findings, marker/operator structure,\n"
     "causal/steering experiments, and per-example evidence.",
     []),

    # ===== SECTION 0: bird's-eye view =====
    ("SECTION 0 — bird's-eye overview",
     "The combined step_anatomy slideshow already lays out Q1/Q2/Q3 across\n"
     "default, stop@3, and stop@5 endpoints.", "",
     [("step_anatomy_gsm8k.pdf", None)]),

    # ===== SECTION 1: descriptive (what the loop does) =====
    ("SECTION 1 — what the loop actually does", "Descriptive findings.", "",
     []),

    ("Claim 1.1: step importance is non-uniform; biggest gains at step 3 and step 5",
     "Force-decode accuracy per step, w→r vs r→w transitions, block-output norms.",
     "Sources: step_transitions, force_decode_per_step.",
     [("step_transitions_gsm8k.pdf", None)]),

    ("Claim 1.2: the 'commit-then-synthesize' rhythm — step 2 regresses, step 3 rescues",
     "Side-by-side focused diagnosis of step 2 vs step 3.", "",
     [("step2to3_focus_gsm8k.pdf", None)]),

    ("Claim 1.3: 72-91% of every step's attention goes back to the question",
     "Per-class attention shares averaged across heads and layers.", "",
     [("flow_map_gsm8k_slideshow.pdf", None)]),

    ("Claim 1.4: attention to NUMBER tokens specifically",
     "Per-step number-token attention; per-(layer, head) breakdown across\n"
     "all 6 latent steps + first 4 decode steps.", "",
     [("correctness-analysis/attn_to_numbers_gsm8k.pdf", None)]),

    ("Claim 1.5: attention carries structural work; MLPs are mostly noise at the token level",
     "Per-step attn_out vs mlp_out vs resid_post modal tokens.", "",
     [("sublayer_breakdown_gsm8k.pdf", None)]),

    ("Claim 1.6: logit-lens trajectory through every (step, layer, sublayer)",
     "Full per-sublayer heatmaps + per-step trajectory at L11 + sample top-5 tables.",
     "",
     [("logit_lens_slideshow_gsm8k.pdf", None)]),

    # ===== SECTION 2: marker/operator structure =====
    ("SECTION 2 — marker/operator structure", "What's actually being encoded.", "",
     []),

    ("Claim 2.1: operator info is highly decodable from CODI's residual stream",
     "Multi-op probe heatmaps + best-cell summary.",
     "RidgeClassifier per (step, layer, marker, probe_type).",
     [("operator-probe/multi_op_probe_gsm8k.pdf", None)]),

    ("Claim 2.2: refined probing (correct + length-matched) lifts op accuracy to 80-90%",
     "Probe accuracy under four data filters.",
     "Filters: original, length-matched, correct-only, both.",
     [("operator-probe/multi_op_probe_refined_gsm8k.pdf", None)]),

    ("Claim 2.3: step ORDERING carries marker-specific info (shuffle baseline)",
     "Real vs step-shuffled best-cell accuracy per marker per probe.",
     "If step order were arbitrary, shuffle would tie. Real beats by +0.07-0.17pp.",
     [("operator-probe/multi_op_probe_shuffle_gsm8k.pdf", None)]),

    ("Claim 2.4: c_ld emerges on odd steps, op on even steps (partial even-odd support)",
     "Per-marker odd-vs-even mean accuracy, best-cells, selectivity heatmaps.",
     "",
     [("operator-probe/multi_op_probe_evenodd_gsm8k.pdf", None)]),

    ("Claim 2.5: per-example chain trajectory + 'first correct step'",
     "When does each problem's probe first become correct per marker?",
     "",
     [("operator-probe/multi_op_probe_trajectory_gsm8k.pdf", None)]),

    ("Claim 2.6: cumulative max + persistence — does the signal fade?",
     "Best-anywhere-per-step + cumulative-max curves; persistence vs fade.", "",
     [("operator-probe/multi_op_probe_best_per_step_gsm8k.pdf", None)]),

    ("Claim 2.7: chain emergence at the EMIT level — markers appear gradually",
     "Per-marker fraction-present in emit, per step.",
     "Strict test: emit's final answer equals gold marker m's result.",
     [("chain_emergence_gsm8k.pdf", None)]),

    # ===== SECTION 3: causal experiments =====
    ("SECTION 3 — causal/steering experiments (all null)",
     "Seven independent steering attempts.", "",
     []),

    ("Claim 3.1: 3-mode {zero, mean, patch} ablation across CFs and blocks",
     "Per (CF, block) heatmaps of ablation damage and transfer rate.",
     "Color convention: red = cell matters, blue/white = doesn't matter.",
     [("gsm8k_3mode_slideshow.pdf", None)]),

    ("Claim 3.2: step-decomposed paired-CF patching is also flat across steps",
     "Per-step whole-step paired patching: 1-5% transfer regardless of step.", "",
     [("step_patch_decomposition_gsm8k.pdf", None)]),

    ("Claim 3.3: paired-CF patching detail",
     "Full patch_paired_cf heatmaps per (step, layer, block).", "",
     [("correctness-analysis/patch_paired_cf_gsm8k.pdf", None)]),

    ("Claim 3.4: recovery sweep — which (step, layer) cells restore behavior under patching?",
     "Recovery rate per (step, layer, block).", "",
     [("correctness-analysis/patch_recovery_sweep_gsm8k.pdf", None)]),

    ("Claim 3.5: LDA add↔mul direction at single cell — null on cf_op_strict",
     "Sweep α ∈ [-100, +100], 0 operator flips.", "",
     [("steering/steer_addmul_strict_gsm8k.pdf", None)]),

    ("Claim 3.6: per-marker LDA at refined cells — null on cf_natural",
     "4 markers × 9 alphas × 80 problems = 2880 runs, 0 target flips.", "",
     [("steering/steer_per_marker_natural_gsm8k.pdf", None)]),

    ("Claim 3.7: multi-cell joint intervention — null at 1, 2, 3, and 4 cells",
     "Distributed-representation hypothesis (4 cells joint) falsified.", "",
     [("steering/steer_multicell_varyop_gsm8k.pdf", None)]),

    ("Claim 3.8: gradient-trained v at the strongest probe cell — null argmax flips",
     "Loss drops 27→13 (gradient flows); but argmax doesn't budge across α=0..4.", "",
     [("steering/steer_trained_vector_gsm8k.pdf", None)]),

    ("Claim 3.9: LM-head direction at step 1 L10 attn_out — null",
     "Even using the LM head's own row-difference direction: 0 flips at α ∈ [-3, +12].", "",
     [("steering/steer_attn_op_token_gsm8k.pdf", None)]),

    ("Claim 3.10: the original lda_probe_and_steer 'success' (re-applying LDA to shifted features)",
     "By construction the LDA crosses its own boundary. Not a causal claim about the model.", "",
     [("operator-probe/lda_probe_and_steer_addmul_gsm8k.pdf", None)]),

    # ===== SECTION 4: per-example =====
    ("SECTION 4 — per-example walkthroughs", "Aggregates hide structure.", "",
     []),

    ("Claim 4.1: 5 concrete GSM8K problems — some correct, some wrong",
     "Per-step force-decode emit + L11 modal tokens.",
     "Reveals the regress-then-recover rhythm in specific cases.",
     [("per_example_walkthrough_gsm8k.pdf", None)]),

    ("Claim 4.2: 100 examples (50 correct + 50 wrong) — full sample with table of contents",
     "Browse representative right and wrong cases.", "",
     [("examples_100_slideshow_gsm8k.pdf", None)]),

    ("Claim 4.3: silver traces — model's reasoning trajectories on a held-out set",
     "", "",
     [("correctness-analysis/silver_traces_gsm8k.pdf", None)]),

    # ===== SECTION 4b: ERROR-PATTERN ANALYSES =====
    ("SECTION 4b — error patterns: when CODI is wrong, what does it emit?",
     "Statistical patterns in wrong emits (both rescued and never-fixed).", "",
     []),

    ("Claim 4b.1: Correct-trace patterns — when does each correct problem first become correct?",
     "First-correct-step distribution + chain-length breakdown + rescue-at-step-3 patterns.",
     "126 of 551 correct problems (23%) are rescued specifically at step 3 (wrong at 1+2).",
     [("correct_trace_patterns_gsm8k.pdf", None)]),

    ("Claim 4b.2: Pre-rescue wrong emit categories — same magnitude, near-gold, operand-pair, etc.",
     "What kinds of guesses precede the rescue?",
     "70% same magnitude as gold; 29% within 5 of gold; 15% are pair-op of question operands.",
     [("other_number_patterns_gsm8k.pdf", None)]),

    ("Claim 4b.3: Pre-rescue wrong VALUES — Benford-like first digits, biased to round last digits",
     "First-digit follows Benford's law; last digit 30% '0', 15% '5'.",
     "The model emits statistically natural numbers, not random.",
     [("wrong_value_patterns_gsm8k.pdf", None)]),

    ("Claim 4b.4: Final wrong answers — what does CODI emit when it never gets there?",
     "Even more round (37% end in 0); roundness matches gold's; median within 22 / 0.5× of gold.",
     "Wrong answers look like a Monte Carlo draw from 'GSM8K-shaped answers'.",
     [("final_wrong_patterns_gsm8k.pdf", None)]),

    ("Claim 4b.5: Late-step behavior — two calculation windows, then approximation mode",
     "Wrong at step 3 → 86% stay wrong. Step 5 catches LONGER chains (avg 3.27 markers).",
     "Wrong-final emits stabilize: 48% have only 1 unique value across steps 4-6. "
     "Benford profile is step-invariant — the wrong-guess generator doesn't update.",
     [("late_step_behavior_gsm8k.pdf", None)]),

    ("Claim 4b.6: Mode 1 chaining — sequential internal computation, mostly INTERNAL",
     "Among 236 Mode 1 ≥2-marker problems: 67% only have ONE marker visible in emits. "
     "But avg first-emit step grows monotonically: m=1→1.64, m=2→2.74, m=3→3.10, m=4→3.67.",
     "Where the chain externalizes, order is preserved (9/10 monotone for 2-mk). "
     "The chain is real but compressed; mostly happens in residuals, not emits.",
     [("mode1_chaining_gsm8k.pdf", None)]),

    ("Claim 4b.7: Mode-1-only probes — op accuracy jumps +10 to +22pp over unrestricted",
     "Filtering to Mode 1 (correct + loop-rescued) gives cleaner probe results: "
     "m=2 op acc 0.493 → 0.708 (+22pp), m=4: 0.569 → 0.750 (+18pp).",
     "Cross-marker selectivity also sharpens. Cleaner cells emerge under the filter.",
     [("operator-probe/mode1_probe_selectivity_gsm8k.pdf", None)]),

    # ===== SECTION 5: probe variants =====
    ("SECTION 5 — alternative probes (corroborating evidence)",
     "Different probing protocols all agree on the same picture.", "",
     []),

    ("Claim 5.1: operator probe (strict CF)",
     "All-same-operator chains; controlled operator labels.", "",
     [("operator-probe/operator_probe_strict_gsm8k.pdf", None)]),

    ("Claim 5.2: operator probe on natural CF",
     "Mixed-op chains.", "",
     [("operator-probe/operator_probe_natural_cf_gsm8k.pdf", None)]),

    ("Claim 5.3: operator probe — presence (operator presence in natural GSM8K)",
     "", "",
     [("operator-probe/operator_probe_presence_gsm8k.pdf", None)]),

    ("Claim 5.4: operator probe — transfer between CFs",
     "Train on one CF, test on another.", "",
     [("operator-probe/operator_probe_transfer_gsm8k.pdf", None)]),

    ("Claim 5.5: operator probe — vary-op CF (template-controlled)",
     "Only operator varies, operands held fixed.", "",
     [("operator-probe/operator_probe_vary_op_gsm8k.pdf", None)]),

    ("Claim 5.6: full operator probe slideshow",
     "Comprehensive operator-probe walkthrough.", "",
     [("operator-probe/operator_probe_full_slideshow.pdf", None)]),

    ("Claim 5.7: operator probe — CODI-attributed (which markers does CODI emit?)",
     "", "",
     [("operator-probe/operator_probe_codi_attributed.pdf", None)]),

    ("Claim 5.8: faithfulness probe",
     "Does CODI's emit match its internal reasoning?", "",
     [("correctness-analysis/faithfulness_probe_gsm8k.pdf", None)]),

    ("Claim 5.9: correctness probe (v1)",
     "Where in the residual is correctness most decodable? (single-page)",
     "",
     [("correctness-analysis/correctness_probe_gsm8k.pdf", None)]),

    ("Claim 5.9b: correctness probe v2 — predict final + self-step correctness",
     "Step-6 best: 78% acc predicting FINAL correctness (vs 58% baseline). "
     "Signal monotonically grows: step1 = 66%, step6 = 78%.",
     "From step 1 alone — before any latent compute — the residual predicts the model's "
     "eventual success/failure at +8pp over baseline. CODI 'knows' when it'll be wrong.",
     [("operator-probe/correctness_probe_gsm8k_v2.pdf", None)]),

    ("Claim 5.10: prompt KV probe",
     "Decoding from prompt-stage KV caches.", "",
     [("correctness-analysis/probe_prompt_kv_gsm8k.pdf", None)]),

    ("Claim 5.11: step-1→2 correctness distribution",
     "Probability mass on correctness emerging between steps.", "",
     [("correctness-analysis/step1to2_correctness_distribution_gsm8k.pdf", None)]),

    ("Claim 5.12: attention at emission step",
     "Where does attention go when the model emits the answer?", "",
     [("correctness-analysis/attn_at_emission_gsm8k.pdf", None)]),

    ("Claim 5.13: helix clock — numerical magnitude geometry in CODI",
     "", "",
     [("correctness-analysis/helix_clock_gsm8k.pdf", None)]),

    ("Claim 5.14: helix clock test — verification on held-out",
     "", "",
     [("correctness-analysis/helix_clock_test_gsm8k.pdf", None)]),

    ("Claim 5.15: latent clock slideshow",
     "Clock-like circular geometry of latent representations.", "",
     [("correctness-analysis/clock_slideshow_latent_gsm8k.pdf", None)]),

    # ===== SECTION 6: text-only synthesis =====
    ("SECTION 6 — text-only synthesis (numbers and claims)",
     "Compact text-summary slideshow (every claim with embedded counts).", "",
     [("evidence_slideshow_gsm8k.pdf", None)]),

    # ===== SECTION 7: all tables =====
    ("SECTION 7 — every table",
     "All 39 numerical tables consolidated for quick reference.", "",
     [("all_tables_summary_gsm8k.pdf", None)]),
]


def main():
    writer = PdfWriter()
    tmp_files = []   # keep title-pdf temp files alive

    for entry in SECTIONS:
        if len(entry) == 4:
            title, subtitle, footnote, sources = entry
        else:
            title, sources = entry[0], entry[-1]; subtitle = ""; footnote = ""

        # Add title page
        buf = make_title_pdf(title, subtitle, footnote)
        # Save to a temp file then read into pypdf
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(buf.getvalue()); tmp.flush(); tmp.close()
        tmp_files.append(tmp.name)
        for page in PdfReader(tmp.name).pages:
            writer.add_page(page)

        # Add source pages
        for src, pages in sources:
            src_path = PD / src
            if not src_path.exists():
                print(f"  skipping missing source: {src}"); continue
            try:
                reader = PdfReader(src_path)
            except Exception as e:
                print(f"  failed to read {src}: {e}"); continue
            page_idx = range(len(reader.pages)) if pages is None else pages
            for i in page_idx:
                if 0 <= i < len(reader.pages):
                    writer.add_page(reader.pages[i])
            print(f"  + {src}: {len(reader.pages)} pages")

    with open(OUT_PDF, "wb") as f:
        writer.write(f)
    # cleanup tmp files
    import os
    for fp in tmp_files:
        try: os.remove(fp)
        except: pass

    print(f"\nsaved {OUT_PDF}  ({OUT_PDF.stat().st_size / 1e6:.1f} MB, {len(writer.pages)} pages)")


if __name__ == "__main__":
    main()
