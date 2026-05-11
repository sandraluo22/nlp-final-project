"""Build a PDF slideshow + summary tables of the GPT-2 computation-probes
findings. Emphasises what's INTERPRETABLE and what's NOT."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "experiments" / "computation_probes"
OUT_PDF = PROBE_DIR / "computation_probes_slideshow.pdf"


def text_slide(pdf, title, lines, footer=None):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.85]); ax.axis("off")
    y = 0.97
    for ln in lines:
        if ln.startswith("# "):
            ax.text(0.0, y, ln[2:], fontsize=13, fontweight="bold",
                    transform=ax.transAxes); y -= 0.052
        elif ln.startswith("- "):
            ax.text(0.02, y, "•  " + ln[2:], fontsize=10.5, transform=ax.transAxes); y -= 0.040
        elif ln == "":
            y -= 0.020
        else:
            ax.text(0.0, y, ln, fontsize=10.5, transform=ax.transAxes); y -= 0.040
    if footer:
        ax.text(0.5, -0.02, footer, ha="center", fontsize=8.5, color="gray",
                style="italic", transform=ax.transAxes)
    pdf.savefig(fig, dpi=140); plt.close(fig)


def image_slide(pdf, title, image_path, caption=""):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
    if image_path.exists():
        img = plt.imread(image_path)
        ax.imshow(img); ax.set_aspect("auto")
    else:
        ax.text(0.5, 0.5, f"missing: {image_path.name}", ha="center", va="center",
                fontsize=12, color="red", transform=ax.transAxes)
    if caption:
        fig.text(0.5, 0.04, caption, ha="center", fontsize=9, style="italic")
    pdf.savefig(fig, dpi=140); plt.close(fig)


def table_slide(pdf, title, headers, rows, caption=""):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.05, 0.10, 0.9, 0.78]); ax.axis("off")
    n = len(headers)
    xs = np.linspace(0.02, 0.98, n + 1)
    centers = (xs[:-1] + xs[1:]) / 2
    y = 0.92
    for x, h in zip(centers, headers):
        ax.text(x, y, h, ha="center", fontsize=11, fontweight="bold",
                transform=ax.transAxes)
    ax.plot([0.02, 0.98], [y - 0.04, y - 0.04], color="gray", lw=1, transform=ax.transAxes)
    y -= 0.06
    for r in rows:
        for x, c in zip(centers, r):
            ax.text(x, y, str(c), ha="center", fontsize=10,
                    transform=ax.transAxes)
        y -= 0.05
    if caption:
        ax.text(0.5, -0.02, caption, ha="center", fontsize=9, color="gray",
                style="italic", transform=ax.transAxes)
    pdf.savefig(fig, dpi=140); plt.close(fig)


def main():
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    # Load summary jsons
    last_prompt = json.load(open(PROBE_DIR / "gpt2_logit_lens.json"))
    var_probes  = json.load(open(PROBE_DIR / "gpt2_var_probes.json"))
    decode      = json.load(open(PROBE_DIR / "gpt2_decode_probes.json"))
    narrowing   = json.load(open(PROBE_DIR / "gpt2_narrowing_summary.json"))

    with PdfPages(OUT_PDF) as pdf:
        # 1. Title
        text_slide(pdf, "CODI-GPT-2  ·  watching computation across latent depth",
            [
                "Goal: understand HOW operator and number representations interact during",
                "the 6 latent thought steps and the final decode.",
                "",
                "Setup:",
                "- Activations: 1000 SVAMP problems × (latent_steps=6, layers=13, hidden=768)",
                "- Probes fit on the residual stream at every (layer, latent_step).",
                "",
                "Three views:",
                "- (1) variable probes — what the residual ENCODES at each layer",
                "- (2) logit lens (CODI lm_head + ln_f) — what the model would PREDICT",
                "       *caveat*: at decode-position-1 the model emits 'The...', not the answer.",
                "- (3) candidate-rank trajectory across the 4 operator answers",
            ])

        # 2. Variable probes — last prompt token
        units = np.array(var_probes["units_digit_acc"])
        tens  = np.array(var_probes["tens_digit_acc"])
        oper  = np.array(var_probes["operator_acc"])

        text_slide(pdf, "(1a) Variable probes  ·  last prompt token",
            [
                "# Setup",
                "Per-(layer, latent_step) logistic regression on (residual, label).",
                "Three labels:",
                "- units digit of gold answer (10-class, chance 10%)",
                "- tens digit of gold answer (10-class, chance 10%)",
                "- operator type (4-class, chance 25%)",
                "",
                "# Peak accuracies",
                f"- operator: {oper.max()*100:.1f}%  (chance 25%)",
                f"- tens digit: {tens.max()*100:.1f}%",
                f"- units digit: {units.max()*100:.1f}%",
                "",
                "# Reading",
                "Operator is cleanly encoded across nearly all layers/steps.",
                "Tens digit is encoded ~4× chance, units digit barely above chance.",
                "Consistent with model getting the right OPERATOR but only roughly the right NUMBER.",
            ])
        image_slide(pdf, "(1a) Variable probes heatmap (last prompt token)",
                    PROBE_DIR / "gpt2_var_probes.png",
                    "operator probe is sharp and broad; digit probes are weak. "
                    "Probing the residual at the LAST prompt token, before EOT.")

        # 3. Variable probes — decode position
        ud = np.array(decode["units_digit_acc_per_layer"])
        td = np.array(decode["tens_digit_acc_per_layer"])
        od = np.array(decode["operator_acc_per_layer"])
        rows = [(f"L{l}", f"{ud[l]*100:.1f}%", f"{td[l]*100:.1f}%", f"{od[l]*100:.1f}%")
                for l in range(len(ud))]
        rows.append(("PEAK",
                     f"{ud.max()*100:.1f}% @L{int(np.argmax(ud))}",
                     f"{td.max()*100:.1f}% @L{int(np.argmax(td))}",
                     f"{od.max()*100:.1f}% @L{int(np.argmax(od))}"))
        table_slide(pdf, "(1b) Variable probes  ·  answer position (post-EOT, decode forward)",
                    ["layer", "units digit", "tens digit", "operator"], rows,
                    caption="Logistic regression on the residual at the FIRST decode forward "
                            "(immediately after EOT). Same pattern as last-prompt-token: "
                            "operator strong, digit probes weak.")

        # 4. Logit lens caveat slide
        ll_pct = np.array(decode["pct_top1_per_layer"])
        text_slide(pdf, "(2) Logit lens  ·  caveat  ·  WHY top-1 = 0% at decode-position-1",
            [
                "# What we did",
                "- Loaded CODI-GPT-2's actual lm_head (50260 × 768) and final layer norm (ln_f).",
                "- For each example at every layer of the first decode forward, computed",
                "  argmax(ln_f(residual) @ W^T) and compared to the gold answer's first token.",
                "",
                "# Result",
                "- 0% top-1 = gold answer at every layer.",
                "",
                "# Why",
                "- Inspecting the actual top-1 token at L12: '\"The\"' on 999/1000 examples.",
                "- The model's output template is 'The answer is: <number>'. So the FIRST",
                "  decode forward predicts 'The' with very high probability — NOT the number.",
                "- The numeric answer is only emitted several tokens later, mid-generation.",
                "",
                "# Implication",
                "- Logit-lens / candidate-rank measured at decode-position-1 is reading the",
                "  TEMPLATE position, not the answer position. The 'narrowing-down' interpretation",
                "  doesn't apply there.",
                "- A clean version requires capturing activations at the position where the model",
                "  is actually writing the answer NUMBER (mid-generation). Not done in this run.",
                "",
                "# What still IS valid here",
                "- The variable probes (slides 1a/1b) — they probe what the RESIDUAL ENCODES,",
                "  independent of which token comes out next.",
                "- The relative cos_sim trajectories from the broader number-isolation analysis.",
            ],
            footer="known issue, see svamp_decode_preds.json — first decoded token is overwhelmingly 'The'")

        # 5. Narrowing decode — included with caveat
        image_slide(pdf, "(3) Candidate-rank trajectory  ·  decode position-1  (caveat: template token)",
                    PROBE_DIR / "gpt2_narrowing_decode.png",
                    "Median rank of {sub, add, mul, div} candidate tokens at decode position-1. "
                    "DIV consistently lowest rank because small-valued tokens are inherently more "
                    "probable in the model's prior, NOT because the model is choosing division. "
                    "Reading is dominated by the template-token effect described in the previous slide.")

        # 6. Narrowing prompt — included with same caveat
        image_slide(pdf, "(3b) Candidate-rank trajectory  ·  last prompt token (per layer × step)",
                    PROBE_DIR / "gpt2_narrowing_prompt.png",
                    "Same caveat as (3): residual at last prompt token isn't the model's answer commitment, "
                    "it's the input to the latent loop. Useful for comparing relative trajectories of the 4 "
                    "operator candidates, NOT for absolute predictions.")

        # 7. Summary
        text_slide(pdf, "Summary",
            [
                "# What we can confidently say",
                "- Operator info is cleanly encoded throughout latent depth. Probe accuracy is",
                f"  {oper.max()*100:.1f}% (last prompt) and {od.max()*100:.1f}% (decode position).",
                "",
                "- Specific digit values are only weakly encoded. Tens digit ~40-45% (chance 10%),",
                "  units digit ~17-26%. The model commits firmly to OPERATOR but only roughly to VALUE.",
                "",
                "# What we cannot say",
                "- We cannot 'watch addition happen' from these probes — the logit-lens sees the",
                "  template token at decode-position-1, not the answer position.",
                "",
                "# What would answer the question",
                "- Capture activations at the position where the actual answer NUMBER is being",
                "  emitted (mid-decode, ~position 4-6 of 'The answer is: <num>').",
                "- Re-run logit lens at that position; track candidate ranks per layer there.",
                "- Per-position digit-classifier trained at each token slot might also work.",
            ])

    print(f"saved {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
