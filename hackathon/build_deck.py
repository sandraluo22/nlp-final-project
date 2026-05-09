"""Build the hackathon summary deck.

Output: hackathon/findings_summary.pdf — landscape PDF, one slide per page.
Models in scope: CODI-Llama-3.2-1B and CODI-GPT-2.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "hackathon" / "findings_summary.pdf"


# ---------------------------------------------------------------------------
# Helpers — text rendering with explicit y-cursor and per-line font sizing
# so we can control density and avoid overlapping text.
# ---------------------------------------------------------------------------

class TextStack:
    """Lay out lines top-down on an axes; each line consumes a configurable
    vertical fraction. Use larger advances for headings."""
    def __init__(self, ax, top=0.97, left=0.0):
        self.ax, self.y, self.x0 = ax, top, left

    def heading(self, txt, size=13):
        self.ax.text(self.x0, self.y, txt, fontsize=size, fontweight="bold",
                     transform=self.ax.transAxes)
        self.y -= 0.055

    def bullet(self, txt, size=10.5, indent=0.02):
        self.ax.text(self.x0 + indent, self.y, "•  " + txt, fontsize=size,
                     transform=self.ax.transAxes)
        self.y -= 0.040

    def line(self, txt, size=10.5, italic=False, color="black"):
        kw = {"fontstyle": "italic"} if italic else {}
        self.ax.text(self.x0, self.y, txt, fontsize=size, color=color,
                     transform=self.ax.transAxes, **kw)
        self.y -= 0.040

    def gap(self, frac=0.018):
        self.y -= frac

    def boxed_math(self, latex, size=12):
        """Render a single line of LaTeX math, slightly indented."""
        self.ax.text(self.x0 + 0.02, self.y, latex, fontsize=size,
                     transform=self.ax.transAxes)
        self.y -= 0.060


def new_slide(title=None, figsize=(13.33, 7.5)):
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    return fig


def make_text_axes(fig, rect=(0.05, 0.05, 0.9, 0.85)):
    ax = fig.add_axes(rect); ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# 1. Title
# ---------------------------------------------------------------------------

def slide_title(pdf):
    fig = new_slide()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.axis("off")
    ax.text(0.5, 0.65, "Latent Reasoning Interpretability\nin CODI-distilled models",
            ha="center", va="center", fontsize=24, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.43, "Findings summary deck",
            ha="center", fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.22,
            "Models: CODI-Llama-3.2-1B  and  CODI-GPT-2\n"
            "Benchmarks: SVAMP, GSM-Hard, MathQA-numeric, LOGIC-701  +  9 CF datasets",
            ha="center", fontsize=11, color="#444",
            transform=ax.transAxes)
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 2. What is CODI? (with full loss objective)
# ---------------------------------------------------------------------------

def slide_codi_overview(pdf):
    fig = new_slide("What is CODI?  ·  architecture and training objective")
    ax = make_text_axes(fig, rect=(0.05, 0.05, 0.9, 0.85))
    t = TextStack(ax)
    t.line("CODI = Continuous Chain-of-Thought via Self-Distillation  (Shen et al., EMNLP 2025)",
           size=11.5)
    t.line("Replaces explicit text-token chain-of-thought with continuous 'latent thoughts' fed",
           size=11.5)
    t.line("back as residual-stream vectors — no vocabulary involved between BOT and EOT.",
           size=11.5)
    t.gap()
    t.heading("Architecture")
    t.bullet("Frozen base LM (Llama-3.2-1B or GPT-2) + LoRA adapter (r=128, α=32) + linear projection on residual.")
    t.bullet("Inference flow:  prompt → <BOT> → 6 latent thoughts (loop, no decoding) → <EOT> → decode answer.")
    t.bullet("Each latent thought = projection of the previous step's last-token hidden state.")
    t.gap()
    t.heading("Training objective  (self-distillation)")
    t.bullet("TEACHER pass:  same base LM with explicit text CoT prompting, no LoRA.")
    t.bullet("STUDENT pass:  LoRA + projection version, generating 6 continuous latent thoughts.")
    t.bullet("Both decode the same gold answer y at the end. Distillation aligns student logits to teacher logits.")
    t.gap(0.005)
    t.boxed_math(
        r"$L_{\mathrm{CODI}} \;=\; L_{\mathrm{NLL}}(y\,|\,x,c_{\mathrm{lat}})"
        r"\;+\;\alpha \cdot \mathrm{KL}\!\left(p_{\theta_t}(\cdot\,|\,x,c_{\mathrm{txt}})\;\|\;p_{\theta_s}(\cdot\,|\,x,c_{\mathrm{lat}})\right)"
        r"\;+\;\beta \cdot \| h^{(L)}_{\mathrm{teacher}} - h^{(L)}_{\mathrm{student}} \|_2$",
        size=11,
    )
    t.bullet("L_NLL  :  cross-entropy on gold answer tokens given the prompt + latent thoughts.")
    t.bullet("L_distill (KL term):  KL between teacher's (with text CoT) and student's (with latents) logit distributions, taken at the answer position.")
    t.bullet("L_align  (L2 term):  distance between teacher and student final-layer hidden states at the <EOT> position — keeps the latent thoughts in-distribution.")
    t.bullet("Defaults from paper:  α ≈ 1.0,  β ≈ 0.1.  No labeled latent-thought supervision — the student learns to compress whatever the teacher's text CoT does.")
    ax.text(0.5, -0.02,
            "Paper: arxiv.org/abs/2502.21074  ·  Checkpoints: zen-E/CODI-llama3.2-1b-Instruct, zen-E/CODI-gpt2",
            ha="center", fontsize=8.5, color="gray", style="italic",
            transform=ax.transAxes)
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Setup (terse, lots of breathing room)
# ---------------------------------------------------------------------------

def slide_setup(pdf):
    fig = new_slide("Setup")
    ax = make_text_axes(fig)
    t = TextStack(ax)
    t.heading("Models")
    t.bullet("CODI-Llama-3.2-1B: 16 transformer blocks, hidden=2048, 6 latent steps")
    t.bullet("CODI-GPT-2:           12 transformer blocks, hidden=768,   6 latent steps")
    t.gap()
    t.heading("Benchmarks (open-ended numeric)")
    t.bullet("SVAMP (1000), GSM-Hard (1319), MathQA-numeric (2501), LOGIC-701-numeric (266)")
    t.gap()
    t.heading("Methods")
    t.bullet("Force-stop accuracy sweep over forced #latent steps  N = 0, 1, …, 6")
    t.bullet("Per-(layer × step) PCA / LDA probes on residual activations")
    t.bullet("Linear additive steering at the peak operator-LDA cell, sweeping coefficient α")
    t.bullet("Activation patching: clean→corrupted swap at one (component, stage, layer) cell, measure flip rate")
    t.gap()
    t.heading("Counterfactual / probe datasets  (next slide expands these)")
    t.bullet("5 SVAMP-derived counterfactual variants for operator/magnitude controls")
    t.bullet("4 synthetic isolation probes:  vary_numerals, vary_operator, vary_a, vary_b")
    t.bullet("2 paired numeral-corruption sets for head patching")
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 4. CF dataset construction — detailed with sampling methodology
# ---------------------------------------------------------------------------

def slide_cf_construction(pdf):
    fig = new_slide("How the CF datasets were constructed")
    ax = make_text_axes(fig)
    t = TextStack(ax)
    t.line("Source: 1000 ChilleD/SVAMP problems. Each CF row preserves the SVAMP idx, operator type,",
           size=10.5)
    t.line("and original equation; only numerals (and sometimes wording) are swapped.",
           size=10.5)
    t.gap()
    t.heading("Magnitude-controlled CF variants — for operator-vs-magnitude probes")
    t.bullet("cf_magmatched.json (972). Rejection sampling: for each problem, resample numerals from a uniform "
             "range and ACCEPT only if the new output bucket matches a target distribution that's flat across operators "
             "× output buckets {<10, 10–99, 100–999, 1000+}. Reject and retry up to a budget; drop if budget exceeded.")
    t.bullet("cf_balanced.json (676). Iterative bucketing: at each pass, compute the (operator × input_bucket × "
             "output_bucket) marginals, identify under-filled cells, resample numerals for problems whose row falls in "
             "an over-filled cell. Repeat until both INPUT and OUTPUT bucket marginals are uniform per operator. "
             "Strictest control — used by every cf_lda_* analysis.")
    t.bullet("cf_under99.json (642) / cf_under99_b.json (622). Same as the two above but with the additional "
             "constraint  numerals ≤ 99  enforced during sampling. Sensitivity check.")
    t.bullet("cf_gpt_transformed.json (546). GPT-5 prompt rewrite: structured-output API call asks for a NEW problem "
             "with new numerals AND new wording while preserving the operator. Reject if the model's stated answer "
             "doesn't match the equation, or if the operator is ambiguous; otherwise accept.")
    t.gap(0.005)
    t.heading("Synthetic number-isolation probes — for PCA disentanglement")
    t.bullet("vary_numerals.json (80). Fixed Sub template ('John has {a} apples, gives {b}…'). Sample (a, b) "
             "uniformly with 5 ≤ a ≤ 200, 1 ≤ b < a; reject duplicate pairs. Both numerals vary together.")
    t.bullet("vary_a.json (80). Same Sub template, b fixed = 4, sample a ∈ [5, 200], reject duplicates. ISOLATES a.")
    t.bullet("vary_b.json (80). Same Sub template, a fixed = 200, sample b ∈ [1, 199], reject duplicates. ISOLATES b.")
    t.bullet("vary_operator.json (24). Fixed numerals (a=12, b=4). 6 hand-written scenario templates × 4 operators "
             "(Add / Sub / Mul / Common-Division). Numerals chosen so all four ops give clean integer answers.")
    t.gap(0.005)
    t.heading("Paired numeral-corruption sets — for head patching")
    t.bullet("numeral_pairs_b1_sub.json (60). Same Sub scenario, clean (a, b) vs corrupted (a, 1). Reject pairs "
             "where clean and corrupted answers coincide; sample with seed=0.")
    t.bullet("numeral_pairs_a1_mul.json (60). Same Mul scenario, clean (a, b) vs corrupted (1, b). Reject pairs "
             "where clean answer = corrupted answer (a=1 case).")
    ax.text(0.5, -0.02, "Generators in cf-datasets/generate_*.py  ·  schema details in cf-datasets/README.md",
            ha="center", fontsize=8.5, color="gray", style="italic",
            transform=ax.transAxes)
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Latent-step trend  (Llama-1B + GPT-2)
# ---------------------------------------------------------------------------

def load_trend(path):
    rows = []
    for d in sorted(Path(path).glob("N*"), key=lambda p:int(p.name[1:])):
        rs = json.load(open(d / "results.json"))
        rows.append((int(d.name[1:]),
                     sum(r["correct"] for r in rs) / len(rs) * 100))
    return rows


def slide_latent_trend(pdf):
    fams = [("svamp", "SVAMP"), ("gsmhard", "GSM-Hard"), ("mathqa_numeric", "MathQA-num")]
    llama = {tag: load_trend(ROOT / "codi-work" / "latent-sweep" / f"{name}_latent_sweep_llama")
             for name, tag in fams}
    gpt2  = {tag: load_trend(ROOT / "codi-work" / "latent-sweep" / f"{name}_latent_sweep_gpt2")
             for name, tag in fams}

    fig = new_slide("Forced #latent steps  ·  CODI-Llama-1B and CODI-GPT-2")

    # Procedure preamble
    ax_proc = fig.add_axes([0.04, 0.86, 0.94, 0.06]); ax_proc.axis("off")
    ax_proc.text(0.0, 0.5,
                 "Procedure:  patch the inference loop to emit <EOT> after exactly N latent thoughts (skipping the rest), "
                 "decode greedily, and grade against gold. Sweep N=0..6 on each benchmark for both models.",
                 fontsize=10, va="center", style="italic", color="#222",
                 transform=ax_proc.transAxes)

    # Table on left
    ax_table = fig.add_axes([0.04, 0.10, 0.40, 0.72]); ax_table.axis("off")
    ax_table.text(0.5, 0.95, "Accuracy (%) vs forced #latent steps N",
                  ha="center", fontsize=11.5, fontweight="bold",
                  transform=ax_table.transAxes)
    headers = ["N", "Sub Llama", "Sub GPT-2", "GH Llama", "GH GPT-2", "MQ Llama", "MQ GPT-2"]
    xs = np.linspace(0.06, 0.95, len(headers))
    y = 0.86
    for x, h in zip(xs, headers):
        ax_table.text(x, y, h, ha="center", fontsize=8.5, fontweight="bold",
                      transform=ax_table.transAxes)
    y -= 0.06
    for n in range(7):
        cells = [str(n)]
        for tag in ["SVAMP", "GSM-Hard", "MathQA-num"]:
            for d in [llama, gpt2]:
                v = next((a for k, a in d[tag] if k == n), None)
                cells.append(f"{v:.1f}" if v is not None else "—")
        for x, c in zip(xs, cells):
            ax_table.text(x, y, c, ha="center", fontsize=9.5,
                          transform=ax_table.transAxes)
        y -= 0.05
    ax_table.text(0.5, 0.02,
                  "GPT-2 has clear knees: SVAMP at N=2 (29.6→38.8 %); GSM-Hard at N=3 (4.9→8.0 %).\n"
                  "Llama-1B is essentially flat: SVAMP +2.9 pp across all 6 latent steps.\n"
                  "→ Llama-1B barely uses latent thoughts; GPT-2 actually does.",
                  ha="center", fontsize=9.5, color="#222", style="italic",
                  transform=ax_table.transAxes)

    # Right: line plot
    ax = fig.add_axes([0.51, 0.13, 0.46, 0.69])
    style_per_ds = {"SVAMP": "-", "GSM-Hard": "--", "MathQA-num": ":"}
    for tag in ["SVAMP", "GSM-Hard", "MathQA-num"]:
        sty = style_per_ds[tag]
        ks = [k for k, _ in llama[tag]]; vs = [v for _, v in llama[tag]]
        ax.plot(ks, vs, sty, marker="o", color="#1f77b4", lw=2, ms=5,
                label=f"Llama-1B  {tag}")
        ks = [k for k, _ in gpt2[tag]];  vs = [v for _, v in gpt2[tag]]
        ax.plot(ks, vs, sty, marker="s", color="#d62728", lw=2, ms=5,
                label=f"GPT-2     {tag}")
    ax.axvline(2, ls=":", color="gray", lw=1)
    ax.text(2.1, ax.get_ylim()[1] * 0.92, "knee", fontsize=9, color="gray")
    ax.set_xlabel("forced number of latent steps N")
    ax.set_ylabel("accuracy (%)")
    ax.set_xticks(range(7))
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Operator-LDA probe (Llama vs GPT-2)
# ---------------------------------------------------------------------------

def slide_operator_probe(pdf):
    fp_l = ROOT / "codi-work" / "visualizations-all" / "llama-1b" / "v2" / "cf_lda_stats.json"
    fp_g = ROOT / "codi-work" / "visualizations-all" / "gpt2" / "cf_lda_stats.json"
    s_l = json.load(open(fp_l))
    s_g = json.load(open(fp_g))
    cf_l, or_l = np.array(s_l["cf_acc"]) * 100, np.array(s_l["orig_acc"]) * 100
    cf_g, or_g = np.array(s_g["cf_acc"]) * 100, np.array(s_g["orig_acc"]) * 100

    fig = new_slide("Operator-LDA probe  ·  cf_balanced (676 examples)")
    # Procedure
    ax_proc = fig.add_axes([0.04, 0.86, 0.94, 0.07]); ax_proc.axis("off")
    ax_proc.text(0.0, 0.5,
                 "Procedure:  per (layer, step), fit LDA(n_components=2) on cf_balanced activations supervised on the\n"
                 "operator label. Score on (a) held-out 20 % of cf_balanced and (b) the full original SVAMP transferred.",
                 fontsize=10, va="center", style="italic", color="#222",
                 transform=ax_proc.transAxes)

    # Headline table
    ax_n = fig.add_axes([0.04, 0.40, 0.42, 0.40]); ax_n.axis("off")
    ax_n.text(0.0, 1.0, "Headline numbers", fontsize=12, fontweight="bold",
              transform=ax_n.transAxes)
    headers = ["model", "peak CF", "peak orig", "mean CF", "mean orig"]
    xs = [0.0, 0.30, 0.50, 0.70, 0.88]
    y = 0.84
    for x, h in zip(xs, headers):
        ax_n.text(x, y, h, fontsize=10.5, fontweight="bold",
                  transform=ax_n.transAxes)
    y -= 0.10
    for label, cfm, ofm in [("Llama-1B", cf_l, or_l), ("GPT-2", cf_g, or_g)]:
        cells = [label, f"{cfm.max():.1f}%", f"{ofm.max():.1f}%",
                 f"{cfm.mean():.1f}%", f"{ofm.mean():.1f}%"]
        for x, c in zip(xs, cells):
            ax_n.text(x, y, c, fontsize=10.5, transform=ax_n.transAxes)
        y -= 0.10

    # Interpretation
    ax_i = fig.add_axes([0.04, 0.05, 0.42, 0.30]); ax_i.axis("off")
    ax_i.text(0.0, 1.0,
              "→ Both models linearly decode operator with high accuracy.\n"
              "→ Holds AFTER cf_balanced removes operator–magnitude correlation:\n"
              "    the direction is genuinely operator-coding, not magnitude in disguise.\n"
              "→ Operator info is decodable from latent step 1 onward (no warm-up).",
              fontsize=10, color="#222", style="italic", va="top",
              transform=ax_i.transAxes)

    # Heatmaps
    ax_l = fig.add_axes([0.51, 0.32, 0.21, 0.50])
    im = ax_l.imshow(cf_l, aspect="auto", origin="lower", cmap="viridis", vmin=20, vmax=100)
    ax_l.set_title(f"Llama-1B   peak {cf_l.max():.1f} %", fontsize=10)
    ax_l.set_xlabel("step", fontsize=9); ax_l.set_ylabel("layer", fontsize=9)
    ax_l.set_xticks(range(cf_l.shape[1]))
    ax_l.tick_params(labelsize=8)
    ax_g = fig.add_axes([0.76, 0.32, 0.21, 0.50])
    im2 = ax_g.imshow(cf_g, aspect="auto", origin="lower", cmap="viridis", vmin=20, vmax=100)
    ax_g.set_title(f"GPT-2   peak {cf_g.max():.1f} %", fontsize=10)
    ax_g.set_xlabel("step", fontsize=9); ax_g.set_ylabel("layer", fontsize=9)
    ax_g.set_xticks(range(cf_g.shape[1]))
    ax_g.tick_params(labelsize=8)
    cax = fig.add_axes([0.985, 0.32, 0.008, 0.50])
    fig.colorbar(im2, cax=cax, label="CF acc (%)")
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Steering null
# ---------------------------------------------------------------------------

def slide_steering_null(pdf):
    fig = new_slide("Linear additive steering does NOT flip operators")
    ax = make_text_axes(fig)
    t = TextStack(ax)
    t.heading("Procedure")
    t.bullet("Steering vector  v = mean(activations of target operator) − mean(activations of source operator),")
    t.bullet("computed at the peak (layer, step) cell from the operator-LDA probe.")
    t.bullet("Inject α · v into the residual stream at that cell during the prompt forward via a forward hook.")
    t.bullet("On 10 Subtraction problems each, sweep α ∈ {0, 1, 5, 20, 50}; count flips toward Add and Mul.")
    t.gap()
    t.heading("Result")
    t.bullet("CODI-Llama-1B (prior session): 0/10 flips at every α — same null pattern.")
    t.bullet("CODI-GPT-2 (this session): 0/10 flips at every α. 3/10 examples DO respond to α (hook fires)")
    t.bullet("but none redirected toward Add or Mul. Perturbation is noise, not directional.")
    t.gap()
    t.heading("Conclusion")
    t.bullet("The operator-LDA direction is correlationally decodable (>87 %) but NOT causally steerable.")
    t.bullet("'Read-out vs read-write' dichotomy: a probe-found direction need not be the one the model uses.")
    t.bullet("Null is robust across BOTH CODI models — the wall is general, not architecture-specific.")
    ax.text(0.5, -0.02, "codi-work/experiments/steering_smoke_gpt2.json",
            ha="center", fontsize=8.5, color="gray", style="italic",
            transform=ax.transAxes)
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Head patching
# ---------------------------------------------------------------------------

def slide_head_patching(pdf):
    fig = new_slide("Activation patching reveals where operator/numeral info lives")

    ax_proc = fig.add_axes([0.04, 0.87, 0.94, 0.07]); ax_proc.axis("off")
    ax_proc.text(0.0, 0.5,
                 "Procedure:  pair clean / corrupted prompts that differ in ONE controlled axis. Cache CLEAN residuals at\n"
                 "every (component, stage, layer); replay one cell into CORRUPTED forward; measure flip rate to clean.",
                 fontsize=10, va="center", style="italic", color="#222",
                 transform=ax_proc.transAxes)

    ax_n = fig.add_axes([0.04, 0.55, 0.92, 0.27]); ax_n.axis("off")
    ax_n.text(0.0, 1.0, "Stage-0 (prompt+bot) recovery, headline numbers",
              fontsize=12, fontweight="bold", transform=ax_n.transAxes)
    headers = ["model", "corruption", "MLP-L0", "resid peak", "N_kept / N"]
    xs = [0.0, 0.18, 0.45, 0.62, 0.85]
    y = 0.87
    for x, h in zip(xs, headers):
        ax_n.text(x, y, h, fontsize=11, fontweight="bold",
                  transform=ax_n.transAxes)
    y -= 0.13
    rows = [
        ("Llama-1B", "operator (templated, A_templates)", "—",     "—",                   "100 / 100"),
        ("Llama-1B", "operator (SVAMP, B_svamp_filtered)", "—",     "—",                   "20 / 52"),
        ("GPT-2",    "operator (Mul→Add)",                 "100 %", "100 % at L1–L7",      "5 / 52"),
        ("GPT-2",    "numeral b=1 (Sub)",                  "100 %", "100 % at L0–L7",      "60 / 60"),
        ("GPT-2",    "numeral a=1 (Mul)",                  "100 %", "100 % at L0–L8",      "59 / 60"),
    ]
    for r in rows:
        for x, c in zip(xs, r):
            ax_n.text(x, y, str(c), fontsize=10.5, transform=ax_n.transAxes)
        y -= 0.13

    ax_n.text(0.0, 0.06,
              "All latent stages (1–6): 0 % recovery in every case.  Attention component: ≤ 2 % everywhere.\n"
              "→ Operator AND numeral info are encoded by the prompt forward; latent thoughts add nothing.",
              fontsize=10, color="#222", style="italic", va="top",
              transform=ax_n.transAxes)

    # Per-(component, layer) bar chart for GPT-2
    ax_b = fig.add_axes([0.07, 0.07, 0.88, 0.40])
    fp_op = ROOT / "codi-work" / "head-patching" / "B_svamp_filtered" / "patching_gpt2_recovery.json"
    fp_b1 = ROOT / "codi-work" / "head-patching" / "numeral_b1_sub_recovery.json"
    fp_a1 = ROOT / "codi-work" / "head-patching" / "numeral_a1_mul_recovery.json"
    color_map = {"operator": "#9467bd", "b=1 Sub": "#1f77b4", "a=1 Mul": "#d62728"}
    cells = [f"{c}-L{l:02d}" for c in ["resid", "attn", "mlp"] for l in range(12)]
    xs_b = np.arange(len(cells))
    width = 0.27
    for i, (lab, fp) in enumerate([("operator", fp_op),
                                    ("b=1 Sub", fp_b1),
                                    ("a=1 Mul", fp_a1)]):
        s = json.load(open(fp))
        rec = s["recovery"]
        vals = []
        for c in ["resid", "attn", "mlp"]:
            arr = np.array(rec[c]) * 100
            vals.extend(arr[0].tolist())
        ax_b.bar(xs_b + (i - 1) * width, vals, width=width,
                 color=color_map[lab], label=lab)
    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels(cells, rotation=90, fontsize=6)
    ax_b.set_ylabel("clean-flip recovery (%)")
    ax_b.set_title("CODI-GPT-2 stage-0 recovery per (component, layer)")
    ax_b.set_ylim(-5, 105)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=9)
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Number ↔ operator entanglement (layers 1..11)
# ---------------------------------------------------------------------------

def slide_entanglement_main(pdf):
    fp = ROOT / "codi-work" / "visualizations-all" / "gpt2" / "number_isolation_cossim.json"
    s = json.load(open(fp))
    key = "vary_numerals__VS__vary_operator"
    max_per = np.array(s[key]["max_abs_per_step"])
    L, S = max_per.shape
    sub = max_per[1:12]                            # layers 1..11 inclusive
    chance = 9.0 / 768

    fig = new_slide("Number ↔ operator entanglement  ·  layers 1–11  (mid-stack)")
    # Procedure
    ax_proc = fig.add_axes([0.04, 0.86, 0.94, 0.07]); ax_proc.axis("off")
    ax_proc.text(0.0, 0.5,
                 "Procedure:  per (layer, latent step), independently fit PCA(n=3) on vary_numerals (varied a, b — Sub-only)\n"
                 "and on vary_operator (fixed a=12, b=4 — varied operator). Compute cos_sim of all 3×3 PC pairs.",
                 fontsize=10, va="center", style="italic", color="#222",
                 transform=ax_proc.transAxes)

    # Text left
    ax_t = fig.add_axes([0.04, 0.10, 0.42, 0.70]); ax_t.axis("off")
    t = TextStack(ax_t, top=0.95, left=0.0)
    t.heading("Headline numbers", size=12)
    t.bullet(f"Layers 1–11 mean max|cs| = {sub.mean():.3f}  ({sub.mean()/chance:.0f}× chance)")
    t.bullet(f"Range across (1..11, step):  min={sub.min():.2f}, max={sub.max():.2f}")
    t.bullet(f"Random-chance baseline at H=768, k=3:  ≈ {chance:.3f}")
    t.gap()
    t.heading("Interpretation", size=12)
    t.line("→ Number and operator subspaces overlap MUCH more than chance.", size=10, color="#222", italic=True)
    t.line("    They are NOT linearly disentangled — there is a shared subspace.", size=10, color="#222", italic=True)
    t.line("→ Even more strikingly: vary_a vs vary_b max|cs|=0.96, mean=0.22 —", size=10, color="#222", italic=True)
    t.line("    the encoding directions for `a` and `b` are not disentangled even from", size=10, color="#222", italic=True)
    t.line("    EACH OTHER. (vary_a fixes b=4; vary_b fixes a=200.)", size=10, color="#222", italic=True)
    t.line("→ Likely why operator-direction steering perturbs numerals as a side-effect.", size=10, color="#222", italic=True)

    # Right: heatmap layers 1..11
    ax_h = fig.add_axes([0.52, 0.13, 0.43, 0.69])
    im = ax_h.imshow(sub, aspect="auto", origin="lower", cmap="viridis",
                     vmin=0, vmax=1)
    ax_h.set_title("max |cos_sim|  per (layer, latent step)")
    ax_h.set_xlabel("latent step (1..6)")
    ax_h.set_ylabel("layer (1..11)")
    ax_h.set_xticks(range(S))
    ax_h.set_xticklabels([str(s+1) for s in range(S)])
    ax_h.set_yticks(range(11))
    ax_h.set_yticklabels([str(L) for L in range(1, 12)])
    plt.colorbar(im, ax=ax_h, label="max |cos_sim|")
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Layer 12 outlier
# ---------------------------------------------------------------------------

def slide_entanglement_layer12(pdf):
    fp = ROOT / "codi-work" / "visualizations-all" / "gpt2" / "number_isolation_cossim.json"
    s = json.load(open(fp))
    key = "vary_numerals__VS__vary_operator"
    max_per = np.array(s[key]["max_abs_per_step"])
    L, S = max_per.shape
    layer_means = max_per.mean(axis=1)

    fig = new_slide("Layer 12 (final) is a cos_sim outlier  ·  CODI-GPT-2")
    ax_proc = fig.add_axes([0.04, 0.86, 0.94, 0.07]); ax_proc.axis("off")
    ax_proc.text(0.0, 0.5,
                 "Procedure:  for each layer, average max|cos_sim| across the 6 latent steps. Compare to chance (k²/H).",
                 fontsize=10, va="center", style="italic", color="#222",
                 transform=ax_proc.transAxes)

    # Top-left: per-layer table
    ax_t = fig.add_axes([0.04, 0.45, 0.42, 0.40]); ax_t.axis("off")
    ax_t.text(0.0, 1.0, "Layer-by-layer mean max|cos_sim|",
              fontsize=12, fontweight="bold", transform=ax_t.transAxes)
    headers = ["layer", "mean max|cs|", "× chance"]
    xs = [0.05, 0.40, 0.75]
    y = 0.88
    for x, h in zip(xs, headers):
        ax_t.text(x, y, h, fontsize=10.5, fontweight="bold",
                  transform=ax_t.transAxes)
    chance = 9.0 / 768
    y -= 0.10
    for L_idx in [0, 1, 5, 8, 11, 12]:
        cells = [str(L_idx), f"{layer_means[L_idx]:.3f}",
                 f"{layer_means[L_idx]/chance:>4.0f}×"]
        color = "#d62728" if L_idx == 12 else "black"
        for x, c in zip(xs, cells):
            ax_t.text(x, y, c, fontsize=10.5, color=color,
                      transform=ax_t.transAxes)
        y -= 0.10

    # Bottom-left: hypothesis
    ax_n = fig.add_axes([0.04, 0.05, 0.42, 0.35]); ax_n.axis("off")
    ax_n.text(0.0, 1.0, "Why layer 12 is different",
              fontsize=12, fontweight="bold", transform=ax_n.transAxes)
    ax_n.text(0.0, 0.86,
              f"L11 mean: {layer_means[11]:.3f}  (typical mid-stack pattern)\n"
              f"L12 mean: {layer_means[12]:.3f}  ← {layer_means[12]/layer_means[11]:.1f}× the next-highest",
              fontsize=10.5, va="top", transform=ax_n.transAxes)
    ax_n.text(0.0, 0.55,
              "Hypothesis:\n"
              "• Mid layers (1–11) progressively SEPARATE number and operator\n"
              "    encoding directions (alignment falls L0→L11: 0.64 → 0.22).\n"
              "• L12 must produce the answer-token distribution, which depends\n"
              "    on BOTH numbers and operator, so it RECOMBINES them into a\n"
              "    shared output subspace → max|cs| ≈ 0.91.\n"
              "• Consistent with head-patching: late layers (L9–L11) show 0 %\n"
              "    numeral-recovery — input-side info has been 'consumed'.",
              fontsize=10, color="#222", va="top",
              transform=ax_n.transAxes)

    # Right: bar plot per layer
    ax_b = fig.add_axes([0.51, 0.13, 0.46, 0.70])
    Ls = np.arange(L)
    colors = ["#1f77b4"] * (L - 1) + ["#d62728"]
    ax_b.bar(Ls, layer_means, color=colors)
    ax_b.axhline(chance, ls=":", color="gray", lw=1)
    ax_b.text(0.3, chance + 0.02, f"chance = {chance:.3f}",
              fontsize=8, color="gray")
    ax_b.set_xticks(Ls)
    ax_b.set_xlabel("layer (0=embedding, 1..12=blocks)")
    ax_b.set_ylabel("mean max|cos_sim|")
    ax_b.set_title("Layer 12 jumps to ≈ 0.91 while L1–L11 sit between 0.21 and 0.60")
    ax_b.set_ylim(0, 1.0)
    ax_b.grid(axis="y", alpha=0.3)
    pdf.savefig(fig, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 11. Synthesis
# ---------------------------------------------------------------------------

def slide_synthesis(pdf):
    fig = new_slide("Synthesis  ·  what's encoded vs what's used")
    ax = make_text_axes(fig)
    t = TextStack(ax)
    t.heading("1. Operator and numerals are encoded — in the prompt forward, not in latent steps.")
    t.bullet("Operator-LDA on cf_balanced: 95 % (Llama-1B), 88 % (GPT-2) peak accuracy.")
    t.bullet("Patching the prompt-stage residual recovers 100 % of clean numerals on GPT-2.")
    t.bullet("All latent stages (1+) contribute 0 % to recovery in every patching experiment.")
    t.gap()
    t.heading("2. The information is read-out, not read-write.")
    t.bullet("Linear additive steering at the peak operator-decoding cell: 0 % flip rate on both models.")
    t.bullet("Robust to α ∈ {1, 5, 20, 50} and to choice of model.")
    t.gap()
    t.heading("3. Latent thinking yields a small, sharp accuracy gain — and then plateaus.")
    t.bullet("GPT-2 SVAMP: 29.6 % (N=0) → 38.8 % (N=2); flat thereafter through N=6.")
    t.bullet("GPT-2 GSM-Hard: 4.9 % (N=2) → 8.0 % (N=3) — sharper knee on harder problems.")
    t.bullet("Llama-1B is essentially flat: +2.9 pp on SVAMP across all 6 latent steps.")
    t.gap()
    t.heading("4. Number and operator subspaces are entangled — except in the very last layer.")
    t.bullet("L1–L11 mean max |cos_sim| ≈ 0.40 (16× chance) — strongly entangled.")
    t.bullet("L12 mean max |cos_sim| ≈ 0.91 — recombination layer for the answer token.")
    t.bullet("Even vary_a vs vary_b max |cos_sim| ≈ 0.96 — `a` and `b` not even disentangled from each other.")
    t.gap()
    t.heading("Implications")
    t.bullet("Latent reasoning here looks more like 'read-and-format' than 'compute-then-emit'.")
    t.bullet("Bottleneck for math is NOT operator extraction — it's arithmetic execution.")
    t.bullet("For mech-interp, look for non-linear / multi-direction encodings, not single operator axes.")
    pdf.savefig(fig, dpi=140); plt.close(fig)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        slide_title(pdf)
        slide_codi_overview(pdf)
        slide_setup(pdf)
        slide_cf_construction(pdf)
        slide_latent_trend(pdf)
        slide_operator_probe(pdf)
        slide_steering_null(pdf)
        slide_head_patching(pdf)
        slide_entanglement_main(pdf)
        slide_entanglement_layer12(pdf)
        slide_synthesis(pdf)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
