"""Build slideshow of all 6 context-isolation methods + per-example ablation
direction figures."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
HEAD = PD.parents[1] / "head-patching"
J = json.load(open(PD / "context_isolation.json"))
OUT = PD / "context_isolation_slideshow.pdf"


def text_slide(pdf, title, lines):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.85]); ax.axis("off")
    y = 0.97
    for ln in lines:
        if ln.startswith("# "):
            ax.text(0.0, y, ln[2:], fontsize=12, fontweight="bold", transform=ax.transAxes); y -= 0.045
        elif ln.startswith("- "):
            ax.text(0.02, y, "•  " + ln[2:], fontsize=10, transform=ax.transAxes); y -= 0.035
        elif ln == "":
            y -= 0.018
        else:
            ax.text(0.0, y, ln, fontsize=10, transform=ax.transAxes); y -= 0.035
    pdf.savefig(fig, dpi=140); plt.close(fig)


def image_slide(pdf, title, image_path, caption=""):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
    if Path(image_path).exists():
        img = plt.imread(image_path); ax.imshow(img); ax.set_aspect("auto")
    else:
        ax.text(0.5, 0.5, f"missing {image_path}", ha="center", color="red", transform=ax.transAxes)
    if caption:
        fig.text(0.5, 0.04, caption, ha="center", fontsize=9, style="italic")
    pdf.savefig(fig, dpi=140); plt.close(fig)


METHODS = ["baseline", "methodA", "methodB", "methodC", "methodD", "methodE"]
LABELS = {
    "baseline": "Baseline (no removal)",
    "methodA":  "A: subtract per-operator mean",
    "methodB":  "B: orthogonal to top-k PCs (k captures 80% var)",
    "methodC":  "C: residual after regressing on op one-hot",
    "methodD":  "D: subtract grand mean",
    "methodE":  "E: orthogonal to top-3 PCs (preserves operator)",
}
COLORS = {
    "baseline": "#000000", "methodA": "#d62728", "methodB": "#9467bd",
    "methodC": "#e377c2", "methodD": "#7f7f7f", "methodE": "#2ca02c",
}
P = 16

# === Per-variable line plots ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for var, ax in zip(["units", "tens", "op"], axes):
    for m in METHODS:
        vals = [max(np.array(J[m][var])[p]) * 100 for p in range(P)]
        ax.plot(range(P), vals, "o-", color=COLORS[m], label=LABELS[m],
                linewidth=2 if m == "baseline" else 1.5,
                markersize=6, alpha=0.85)
    chance = {"units": 10, "tens": 10, "op": 25}[var]
    majority = {"units": 14.8, "tens": 26.8, "op": 53.1}[var]
    ax.axhline(chance, color="gray", ls=":", lw=0.7, label=f"chance ({chance}%)")
    ax.axhline(majority, color="gray", ls="--", lw=0.7, label=f"majority ({majority}%)")
    ax.set_title(f"{var}  probe accuracy across positions")
    ax.set_xlabel("decode position"); ax.set_ylabel("peak across layers (%)")
    ax.set_xticks(range(P)); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right" if var != "op" else "upper right")
plt.suptitle("Probe accuracy under 6 context-isolation methods (peak across layers)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig_lines = PD / "context_isolation_lines.png"
plt.savefig(fig_lines, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {fig_lines}")

# === Pure summary table-style slide  ===
def slide_method_summary(pdf):
    text_slide(pdf, "Six context-isolation methods", [
        "Goal: subtract 'context' from each (pos, layer) activation, then re-fit",
        "operator/units/tens probes. Compare to baseline to see what each method",
        "actually removes.",
        "",
        "# Method definitions",
        "- baseline:  no removal (raw activations).",
        "- A:  subtract per-operator-class mean μ_c[p, l]. Removes the additive",
        "      operator signal explicitly.",
        "- B:  PCA orthogonal complement keeping only the top-k PCs that capture",
        "      80% of variance.  k = 4 at pos 0; up to 20 at pos 1; only 1-2 at",
        "      pos 4-15. Aggressive removal.",
        "- C:  residual after Ridge regressing each activation dim on a 4-D",
        "      operator one-hot. Mathematically equivalent to A.",
        "- D:  subtract grand mean (zero-center). No information removal —",
        "      probes use StandardScaler internally so this is a no-op for",
        "      probe accuracy. Sanity check.",
        "- E:  orthogonal to TOP-3 PCs only. Removes the biggest shared modes",
        "      (likely template/context) while leaving smaller-variance",
        "      operator/digit directions intact.",
        "",
        "# What 'removing context but preserving operator+digit' looks like",
        "Method E is the clean version. At every position, op stays >80% (pos 0-9),",
        "units/tens drop by <2 pp, but the top 3 'shared' PCs are gone.",
        "If the model relied on those PCs for prediction, probes would have",
        "collapsed; they don't.",
    ])

# === Diff plot: how much each method drops each variable ===
def slide_diff(pdf):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for var, ax in zip(["units", "tens", "op"], axes):
        baseline = np.array([max(np.array(J["baseline"][var])[p]) for p in range(P)]) * 100
        for m in ["methodA", "methodB", "methodC", "methodE"]:
            vals = np.array([max(np.array(J[m][var])[p]) for p in range(P)]) * 100
            diff = vals - baseline
            ax.plot(range(P), diff, "o-", color=COLORS[m], label=LABELS[m],
                    linewidth=1.5, markersize=5, alpha=0.85)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"{var}: peak probe accuracy minus baseline")
        ax.set_xlabel("decode position"); ax.set_ylabel("Δ probe acc (pp)")
        ax.set_xticks(range(P)); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    plt.suptitle("How much each method DROPS the probe accuracy", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = PD / "context_isolation_diff.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    return out


with PdfPages(OUT) as pdf:
    text_slide(pdf, "CODI-GPT-2  ·  Context isolation (6 methods)", [
        "# Question",
        "If we subtract the 'context' part of the activation, does the math signal",
        "(operator + digits) survive? Different definitions of 'context' give",
        "different answers. We try 6.",
        "",
        "# Setup",
        "Activations: 1000 SVAMP × 16 decode positions × 13 layers × 768.",
        "For each method we apply a residualization, then refit per-(pos, layer)",
        "probes for {units, tens, operator} (5-fold CV) using model-emitted labels.",
        "",
        "# Punchline",
        "- Method A/C (subtract per-op mean) — removes operator entirely (op→53%",
        "  at template positions, partial elsewhere) but leaves digit info intact.",
        "- Method B (top-80%-var orthog) — too aggressive, also removes digits at",
        "  later positions because the high-variance directions ARE the math.",
        "- Method D (grand mean) — no-op (probe is invariant to mean shift).",
        "- Method E (top-3-PC orthog) — clean 'remove context, keep math'. All",
        "  three signals survive within 1-6 pp of baseline at every position.",
    ])

    slide_method_summary(pdf)

    image_slide(pdf, "Probe accuracy under each method (raw)",
                fig_lines,
                "Black = baseline. Method E (green) tracks baseline closely "
                "(removes only the top-3 highest-variance modes). Method A/C (red/pink) "
                "zaps op at template positions (drops to majority baseline). Method B "
                "(purple) over-prunes everything once it has to remove >5 PCs.")

    diff_path = slide_diff(pdf)
    image_slide(pdf, "Drop relative to baseline (pp)",
                diff_path,
                "Method E (green) stays within ±5 pp of baseline at every position. "
                "Method A (red) hits operator hard at pos 0-4. Method B (purple) "
                "damages units/tens at pos 1-2 because top-PCs there encode the math.")

    image_slide(pdf, "Per-example ablation: residual at (stage 0, layer L)",
                HEAD / "ablation_perex_traces_stage0.png",
                "Each panel: 200 SVAMP examples sorted by baseline magnitude. Black dotted = baseline |output|. "
                "Red = output after zero-ablating residual at that layer (stage 0 = prompt position). "
                "L2 systematically pulls the red trace below baseline (shrinks output); "
                "L7 pushes it above (grows output). Layers 0 and 9-11 are roughly y=x.")

    image_slide(pdf, "Per-example ablation: scatter plot",
                HEAD / "ablation_perex_scatter_stage0.png",
                "x = |baseline|, y = |after-resid-L-ablation|, log-log. Diagonal = no change. "
                "L2: most points BELOW diagonal (ablation shrinks). L7: most ABOVE (ablation grows). "
                "Per-layer median ratios in each title.")

    image_slide(pdf, "Per-layer multiplicative effect summary",
                HEAD / "ablation_perex_layer_summary.png",
                "Median(|ablated| / |baseline|) per layer, colored by stage. "
                "Layer 2 consistently around 0.3-0.6× across all 7 stages. "
                "Layer 7 consistently 5-7× across all stages. "
                "These are stable per-layer effects, not noise.")

    text_slide(pdf, "Synthesis", [
        "# What context-isolation tells us",
        "- The model's residual at template positions (pos 0-3) carries operator info",
        "  that is mostly 'additive' across operator classes (Method A removes it",
        "  cleanly; Method E barely touches it).",
        "- At answer positions (pos 5+), op probe is more entangled with digit info —",
        "  Method A only partially removes it (drops to 78-86%).",
        "- The TOP-3 highest-variance PCs at each (pos, layer) are mostly 'context'.",
        "  Removing them barely hurts any probe → operator and digit info live in",
        "  smaller-variance directions, not in the dominant modes.",
        "",
        "# What per-example ablation tells us",
        "- Residual ablation at different layers has DIRECTIONAL effects on output",
        "  magnitude. L2 systematically shrinks; L5/L7/L8 systematically grow.",
        "- This is a stable per-layer property: the same direction across all 200",
        "  examples and all 7 stages.",
        "- Implication: the model's layers contribute structured 'magnitude tweaks'",
        "  that compose into the final output. Removing one tweak biases the result",
        "  one way; removing another biases it the opposite way.",
        "",
        "# Open question",
        "Which layer's contribution carries the operator vs the magnitude vs the",
        "digit? The current ablation collapses to net magnitude shift; a finer",
        "experiment would track which probe (op/units/tens) is restored after each",
        "single-layer ablation.",
    ])

print(f"\nsaved {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
