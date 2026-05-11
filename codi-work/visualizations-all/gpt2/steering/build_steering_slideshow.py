"""Build a slideshow summarizing all steering experiments on CODI-GPT-2:
- model-label probes (units, tens, operator) — alpha sweep
- cos-sim conditioning (LOW vs HIGH cos cell, operator add->sub)
- magnitude steering (toward small vs large bucket)

Emphasis: distinguish general-change from targeted hits."""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
J_ml = json.load(open(PD / "steering_modellabel_results.json"))
J_cm = json.load(open(PD / "steering_cossim_magnitude.json"))

OUT = PD / "steering_slideshow.pdf"


def text_slide(pdf, title, lines, footer=None):
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
            y -= 0.02
        else:
            ax.text(0.0, y, ln, fontsize=10, transform=ax.transAxes); y -= 0.035
    if footer:
        ax.text(0.5, -0.02, footer, ha="center", fontsize=8.5, color="gray",
                style="italic", transform=ax.transAxes)
    pdf.savefig(fig, dpi=140); plt.close(fig)


def image_slide(pdf, title, image_path, caption=""):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
    if image_path.exists():
        img = plt.imread(image_path); ax.imshow(img); ax.set_aspect("auto")
    else:
        ax.text(0.5, 0.5, f"missing {image_path.name}", ha="center", color="red", transform=ax.transAxes)
    if caption:
        fig.text(0.5, 0.04, caption, ha="center", fontsize=9, style="italic")
    pdf.savefig(fig, dpi=140); plt.close(fig)


# === FIG 1: operator + digit steering vs alpha (general change) ===
ALPHAS_ML = [1.0, 2.0, 4.0, 8.0]
def get_ml(name, key):
    return [J_ml[f"{name}|alpha={a}"][key] for a in ALPHAS_ML]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(ALPHAS_ML, get_ml("operator_add->sub", "n_changed_full"), "o-", label="operator add→sub")
ax.plot(ALPHAS_ML, get_ml("operator_sub->add", "n_changed_full"), "s-", label="operator sub→add")
ax.plot(ALPHAS_ML, get_ml("operator_add->mul", "n_changed_full"), "^-", label="operator add→mul")
ax.plot(ALPHAS_ML, get_ml("tens_to_5", "n_changed_full"), "o--", label="tens → 5")
ax.plot(ALPHAS_ML, get_ml("tens_to_8", "n_changed_full"), "s--", label="tens → 8")
ax.plot(ALPHAS_ML, get_ml("units_to_3", "n_changed_full"), "o:", label="units → 3")
ax.plot(ALPHAS_ML, get_ml("units_to_7", "n_changed_full"), "s:", label="units → 7")
ax.plot(ALPHAS_ML, get_ml("units_to_9", "n_changed_full"), "^:", label="units → 9")
ax.set_xlabel("alpha (steering scale)")
ax.set_ylabel("n_changed_full / 100")
ax.set_title("Total outputs changed vs alpha\n(any output difference from baseline)")
ax.legend(fontsize=8, loc="upper left", ncol=2); ax.grid(alpha=0.3)

# Targeted hit-rate curves with baseline reference
base_units = J_ml["baseline_units_dist"]; base_tens = J_ml["baseline_tens_dist"]
ax = axes[1]
def get_target(name, key, baseline_dist, target):
    base = baseline_dist.get(str(target), 0)
    return [J_ml[f"{name}|alpha={a}"][key] - base for a in ALPHAS_ML]
ax.plot(ALPHAS_ML, get_target("tens_to_5", "n_tens_to_target", base_tens, 5), "s-", label="tens→5  (baseline=9)")
ax.plot(ALPHAS_ML, get_target("tens_to_8", "n_tens_to_target", base_tens, 8), "^-", label="tens→8  (baseline=3)")
ax.plot(ALPHAS_ML, get_target("units_to_3", "n_units_to_target", base_units, 3), "o-", label="units→3  (baseline=6)")
ax.plot(ALPHAS_ML, get_target("units_to_7", "n_units_to_target", base_units, 7), "s-", label="units→7  (baseline=12)")
ax.plot(ALPHAS_ML, get_target("units_to_9", "n_units_to_target", base_units, 9), "^-", label="units→9  (baseline=4)")
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("alpha")
ax.set_ylabel("# at target − baseline (out of 100)")
ax.set_title("Targeted hit rate above baseline")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
fig1 = PD / "steering_alpha_curves.png"
plt.savefig(fig1, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {fig1}")

# === FIG 2: cos-sim conditioning ===
ALPHAS_CM = [4.0, 8.0]
def get_cm_changed(tag): return [J_cm[f"exp1_{tag}|alpha={a}"]["n_changed"] for a in ALPHAS_CM]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ALPHAS_CM, get_cm_changed("LOW"), "o-", color="#2ca02c", linewidth=2.5,
        markersize=10, label=f"LOW cos-sim cell (pos {J_cm['exp1_low_cell'][0]}, L{J_cm['exp1_low_cell'][1]}, cos={J_cm['exp1_low_cell'][2]:.3f})")
ax.plot(ALPHAS_CM, get_cm_changed("HIGH"), "s-", color="#d62728", linewidth=2.5,
        markersize=10, label=f"HIGH cos-sim cell (pos {J_cm['exp1_high_cell'][0]}, L{J_cm['exp1_high_cell'][1]}, cos={J_cm['exp1_high_cell'][2]:.3f})")
ax.set_xlabel("alpha")
ax.set_ylabel("n_changed_full / 200")
ax.set_title("Operator (add→sub) steering: LOW-cos vs HIGH-cos cell")
ax.set_ylim(-5, 70); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
fig2 = PD / "steering_cossim_curve.png"
plt.savefig(fig2, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {fig2}")

# === FIG 3: magnitude steering ===
ALPHAS_MAG = [4.0, 8.0, 16.0]
median_small = [J_cm[f"exp2_to_bucket0|alpha={a}"]["median_abs"] for a in ALPHAS_MAG]
median_large = [J_cm[f"exp2_to_bucket4|alpha={a}"]["median_abs"] for a in ALPHAS_MAG]
base_med = J_cm["baseline_median_abs"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(base_med, color="black", ls="--", lw=1.5, label=f"baseline median (191919)")
ax.plot(ALPHAS_MAG, median_small, "o-", color="#1f77b4", linewidth=2.5, markersize=10,
        label="→ bucket 0 (small, 1-9)")
ax.plot(ALPHAS_MAG, median_large, "s-", color="#ff7f0e", linewidth=2.5, markersize=10,
        label="→ bucket 4 (large, 10000+)")
ax.set_xlabel("alpha")
ax.set_ylabel("median |int| of model output")
ax.set_yscale("log")
ax.set_title("Magnitude steering: median |output| vs alpha\n(probe at pos 1, L11; acc 74.6%, majority baseline 66.3%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
plt.tight_layout()
fig3 = PD / "steering_magnitude_curve.png"
plt.savefig(fig3, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {fig3}")

# === FIG 4: cos_sim heatmap to show why LOW/HIGH cells were chosen ===
cossim_grid = np.array(J_cm["cossim_grid"])
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cossim_grid.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xlabel("decode position")
ax.set_ylabel("layer")
ax.set_title("|cos_sim(operator_dir, units_principal_dir)| per (pos × layer)")
ax.set_xticks(range(cossim_grid.shape[0]))
ax.set_yticks(range(cossim_grid.shape[1]))
plt.colorbar(im, ax=ax, fraction=0.04)
# annotate the chosen cells
lp, ll = J_cm["exp1_low_cell"][:2]
hp, hl = J_cm["exp1_high_cell"][:2]
ax.scatter([lp], [ll], marker="o", s=180, edgecolors="white", facecolors="none", linewidth=2.5)
ax.text(lp, ll - 0.6, f"LOW", color="white", ha="center", fontsize=9, fontweight="bold")
ax.scatter([hp], [hl], marker="s", s=180, edgecolors="white", facecolors="none", linewidth=2.5)
ax.text(hp, hl + 0.7, f"HIGH", color="white", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
fig4 = PD / "steering_cossim_grid.png"
plt.savefig(fig4, dpi=140, bbox_inches="tight"); plt.close()
print(f"saved {fig4}")

# === Build PDF ===
with PdfPages(OUT) as pdf:
    text_slide(pdf, "CODI-GPT-2  ·  Steering experiments summary",
        [
            "# Question",
            "Are operator/digit/magnitude probe directions causal?",
            "Concretely: at the (pos, layer) cell where each is most decodable,",
            "fit a multinomial probe, take its row vector for class c as a steering",
            "direction, scale by alpha, add to the residual at the same cell during",
            "inference. Measure the effect on the model's emitted answer.",
            "",
            "# Steering vector convention",
            "v = sigma * (W[c_target] - W[c_source])  in raw residual space, where",
            "sigma is the StandardScaler std and W is the LR coefficient matrix.",
            "",
            "# Crucial caveat about \"changed\"",
            "Most plots show TOTAL OUTPUTS CHANGED — i.e., the steered output",
            "differs from baseline at all. This is NOT the same as 'output now",
            "matches the target class'. We also include targeted hit-rate plots",
            "to disentangle: did the model move toward the requested digit/op?",
        ])

    text_slide(pdf, "What 'changed' means here",
        [
            "# changed_full",
            "Number of examples where the steered output's parsed integer differs",
            "from the baseline output's parsed integer. Counts perturbation, not",
            "intent.",
            "",
            "# targeted hit-rate (where applicable)",
            "Number whose specific digit (units or tens) equals the requested target,",
            "minus the baseline count of that same target. Lift > 0 means causal.",
            "",
            "# Headline finding when both are looked at",
            "- operator add->sub  changed_full=40/100 at alpha=8  but no operator-match",
            "  is recoverable (model's outputs are nonsense to begin with).",
            "- tens/units sweeps: targeted hit lift +0 to +4 across all alphas.",
            "  That is roughly noise — the steering moves outputs but not toward",
            "  the requested digit.",
            "- magnitude: median |int| moves toward the target bucket (-26% / +170%).",
            "  Direction-correct effect on output magnitude, even if the bucket",
            "  classification rate barely shifts.",
        ])

    image_slide(pdf, "Operator / tens / units steering vs alpha",
                fig1,
                "Left: total changed (general perturbation). Operator sweeps reach 30-40 at alpha=8. "
                "Tens/units stay at <10 across the alpha range — barely steerable. "
                "Right: targeted hit rate above baseline. Even strong alphas give +0 to +4 — essentially noise. "
                "Operator probes are 91% accurate but the lift on output is via 'general perturbation', not 'flip the operation'.")

    image_slide(pdf, "Cos-sim conditioning: where steering works",
                fig4,
                "Per-(pos, layer) heatmap of |cos_sim(operator-direction, units-principal-direction)|. "
                "Layer 0 (the embedding) is universally tangled (cos~0.9). Cells at layers 6-12 "
                "and positions 1-7 are nearly orthogonal (cos<0.05). LOW cell selected at pos 1 layer 7; "
                "HIGH cell at pos 2 layer 0.")

    image_slide(pdf, "Cos-sim conditioning result: LOW vs HIGH cell",
                fig2,
                "At the LOW-cos cell (orthogonal directions), operator steering changes 41-55 of 200 outputs. "
                "At the HIGH-cos cell (tangled directions), the SAME steering vector produces ZERO change. "
                "The probe direction is causal only where it's geometrically separable from other features.")

    image_slide(pdf, "Magnitude steering: median |output| vs alpha",
                fig3,
                "Steering toward 'small' (bucket 0) reduces median |int| from 192k to 141k (~26%). "
                "Steering toward 'large' (bucket 4) grows it from 192k to 515k (+170%). "
                "Direction-correct effect at all alphas tested. Asymmetric because the model's prior is "
                "already toward huge numbers.")

    text_slide(pdf, "What this tells us about computation",
        [
            "# Operator",
            "- Probe accuracy 91% — strong linear signal everywhere.",
            "- Causally steerable ONLY at (pos, layer) where its direction is",
            "  orthogonal to digit directions. Where they're tangled, the same",
            "  vector has zero behavioral effect.",
            "- Steering perturbs outputs but doesn't recover correct subtraction",
            "  results — because the model isn't doing arithmetic to begin with.",
            "",
            "# Specific digits (units, tens)",
            "- Pre-emission probe accuracy ~44%; targeted hit-rate lift = 0 to +4.",
            "- NOT a clean steerable feature in the residual before emission.",
            "- Probe accuracy at later positions (>=7) is 80%+, but that's after",
            "  the digit has already been emitted into the KV cache — not useful",
            "  for influencing the answer.",
            "",
            "# Magnitude (5 buckets by log10)",
            "- Probe accuracy 74.6% at pos 1 L11 (majority baseline 66.3%).",
            "- Steering produces direction-correct shifts in median |output|:",
            "  -26% toward small, +170% toward large. Real effect at scale level,",
            "  weak effect at bucket-classification level.",
            "",
            "# Synthesis",
            "Operator and magnitude exist as steerable linear features (within the",
            "right cells). Specific digit values do NOT. The model's 'computation'",
            "is best described as: identify operator + commit to a magnitude",
            "envelope, then fill in digits one-at-a-time at emission. The digit",
            "stream is generated, not pre-computed.",
        ])

print(f"\nsaved {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
