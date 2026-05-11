"""Slideshow synthesizing the 'where does latent-loop computation live?' picture.

Combines:
- Force-decode per-step transitions (force_decode_per_step.json): net flips
  vs total churn per transition.
- Mean-CF patch recovery (patch_recovery_sweep.json): 4-tier per-step,
  per-layer-residual, per-layer-attn, per-layer-MLP recovery rates.
- Zero-ablation sanity check (patch_sanity_check.json): per-step recovery
  under zero (extreme) intervention.
- std/‖mean‖ heatmap (patch_sanity_check.json): how tightly clustered each
  cell's activations are.

Outputs: latent_loop_compute_slideshow.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
OUT = PD / "latent_loop_compute_slideshow.pdf"


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
    fd = json.load(open(PD / "force_decode_per_step.json"))
    rec = json.load(open(PD / "patch_recovery_sweep.json"))
    san = json.load(open(PD / "patch_sanity_check.json"))

    with PdfPages(OUT) as pdf:
        # === Slide 1: setup ===
        text_slide(pdf, "Where is latent-loop computation done? (CODI-GPT-2)",
            [
                "# Approach",
                "Three orthogonal probes of the 6-step latent loop:",
                "- Force-decode per step: at each k, cut latent loop short and decode.",
                "  Accuracy + how many examples flipped right↔wrong vs prior step.",
                "- Mean-CF patch recovery: replace one cell's activation with the mean",
                "  over a CF set (same template, varied numbers). Recovery rate = fraction",
                "  of predictions unchanged. Low recovery = cell mattered.",
                "- Zero-ablation: replace a whole step's last-token outputs with ZEROS.",
                "  Sanity check that the hook works. Big accuracy drop = real disruption.",
                "",
                "# Plus diagnostics",
                "- std/‖mean‖ per cell: how tight is the CF-set activation cluster?",
                "  Small ratio = mean-patching is a near-no-op (the mean IS each example).",
            ])

        # === Slide 2: Force-decode net + total flips ===
        trs = fd["transitions"][:5]
        labels = [f"{t['from_step']}→{t['to_step']}" for t in trs]
        wr = np.array([t["wrong_to_right"] for t in trs])
        rw = np.array([t["right_to_wrong"] for t in trs])
        net = wr - rw
        total = wr + rw
        accs = np.array(fd["accuracy_per_step"][:6]) * 100

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Force-decode per latent step (GPT-2 on SVAMP, N=1000)",
                     fontsize=12, fontweight="bold")
        xs = np.arange(len(labels))
        w = 0.4
        axes[0].bar(xs - w/2, wr, w, color="#2ca02c", label="wrong→right")
        axes[0].bar(xs + w/2, -rw, w, color="#d62728", label="right→wrong (neg)")
        axes[0].axhline(0, color="black", lw=0.5)
        axes[0].set_xticks(xs); axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("# examples (of 1000)")
        axes[0].set_title("Net flips per transition (BIG NET at 1→2)", fontsize=10)
        for i, (a, b) in enumerate(zip(wr, rw)):
            axes[0].text(xs[i], max(a, 0) + 3, f"net={a-b:+d}", ha="center", fontsize=8)
            axes[0].text(xs[i], -b - 4, f"{b}", ha="center", fontsize=7, color="darkred")
            axes[0].text(xs[i], a + 0.5, f"{a}", ha="center", fontsize=7, color="darkgreen")
        axes[0].legend(fontsize=8, loc="upper right"); axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(xs, total, color="#1f77b4", edgecolor="black", linewidth=0.5)
        axes[1].set_xticks(xs); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("# examples that FLIPPED (right↔wrong total)")
        axes[1].set_title("Total CHURN per transition (1→2 ≈ 2→3, then decay)", fontsize=10)
        for i, v in enumerate(total):
            axes[1].text(xs[i], v + 1, f"{v}", ha="center", fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 2b: per-K accuracy ===
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(range(1, 7), accs, "o-", lw=2, markersize=9, color="#1f77b4")
        for k, v in enumerate(accs, start=1):
            ax.annotate(f"{v:.1f}%", (k, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=9)
        ax.set_xlabel("latent step K"); ax.set_ylabel("force-decode accuracy (%)")
        ax.set_title("Force-decode accuracy per K (gain ≈ +3.6 pp from step 1 to peak)",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, 7))
        ax.set_ylim(35, 42); ax.grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 3: Recovery sweep — bar grid (Sweep A) for both CF sets ===
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle("Sweep A — MEAN-CF patch: ablate whole STEP (all 12 layers × attn+mlp)\n"
                     "recovery rate = fraction unchanged from baseline",
                     fontsize=11, fontweight="bold")
        for ax, (cf_name, color) in zip(axes, [("vary_a_2digit", "#2ca02c"),
                                                ("vary_numerals", "#1f77b4")]):
            r = rec["cf_sets"][cf_name]; sw = r["sweeps"]
            vals = [sw[f"step{s+1}"]["recovery_rate"] * 100 for s in range(6)]
            ax.bar(np.arange(6), vals, color=color, edgecolor="black", linewidth=0.5)
            ax.set_xticks(np.arange(6)); ax.set_xticklabels([f"s{s+1}" for s in range(6)])
            ax.set_ylim(0, 105); ax.axhline(100, color="gray", lw=0.5, ls=":")
            ax.set_title(f"{cf_name}  (baseline acc {r['baseline_accuracy']*100:.1f}%)",
                         fontsize=10)
            ax.set_ylabel("recovery (%)")
            for s, v in enumerate(vals):
                ax.text(s, v + 1, f"{v:.0f}", ha="center", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 4: Recovery sweep — per-layer (B/C/D) for vary_numerals only ===
        cf_name = "vary_numerals"
        r = rec["cf_sets"][cf_name]; sw = r["sweeps"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f"Sweep B/C/D — MEAN-CF patch by LAYER ({cf_name}, baseline acc {r['baseline_accuracy']*100:.1f}%)",
                     fontsize=11, fontweight="bold")
        for ax, (key, color, title) in zip(axes, [
            ("resid", "#9467bd", "B: residual stream per layer"),
            ("attn",  "#ff7f0e", "C: attn per layer"),
            ("mlp",   "#2ca02c", "D: MLP per layer"),
        ]):
            vals = [sw[f"{key}_L{l}"]["recovery_rate"] * 100 for l in range(12)]
            ax.bar(np.arange(12), vals, color=color, edgecolor="black", linewidth=0.5)
            ax.set_xticks(np.arange(12)); ax.set_xticklabels([f"L{l}" for l in range(12)])
            ax.set_ylim(0, 105); ax.axhline(100, color="gray", lw=0.5, ls=":")
            ax.set_title(title, fontsize=10); ax.set_ylabel("recovery (%)")
            for l, v in enumerate(vals):
                if v < 100:
                    ax.text(l, v + 1, f"{v:.0f}", ha="center", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 5: ZERO ablation per step (sanity) ===
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle("ZERO ablation per STEP (sanity): replace step k's outputs with ZEROS\n"
                     "If hook works, dropping a useful step should crash accuracy.",
                     fontsize=11, fontweight="bold")
        for ax, cf_name in zip(axes, ["vary_a_2digit", "vary_numerals"]):
            r = san[cf_name]; z = r["zero_ablation_per_step"]
            recovery = [z[f"step{s+1}"]["recovery_rate"] * 100 for s in range(6)]
            acc_after = [z[f"step{s+1}"]["accuracy"] * 100 for s in range(6)]
            base = r["baseline_accuracy"] * 100
            xs = np.arange(6); w = 0.4
            ax.bar(xs - w/2, recovery, w, color="#1f77b4", label="recovery%", edgecolor="black", linewidth=0.5)
            ax.bar(xs + w/2, acc_after, w, color="#d62728", label="acc% after zero", edgecolor="black", linewidth=0.5)
            ax.axhline(base, color="black", lw=0.7, ls=":", label=f"baseline acc {base:.0f}%")
            ax.set_xticks(xs); ax.set_xticklabels([f"s{s+1}" for s in range(6)])
            ax.set_ylim(0, 110); ax.set_ylabel("%")
            ax.set_title(f"{cf_name}", fontsize=10)
            for x, (rv, av) in enumerate(zip(recovery, acc_after)):
                ax.text(x - w/2, rv + 1, f"{rv:.0f}", ha="center", fontsize=7)
                ax.text(x + w/2, av + 1, f"{av:.0f}", ha="center", fontsize=7)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 6: std/‖mean‖ heatmap per (step, layer) for both blocks, both CF sets ===
        for cf_name in ["vary_numerals", "vary_a_2digit"]:
            r = san[cf_name]
            attn = np.array(r["std_over_mean_attn"])
            mlp = np.array(r["std_over_mean_mlp"])
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"std/‖mean‖ across {r['N']} CF examples — {cf_name}",
                         fontsize=12, fontweight="bold")
            for ax, mat, title in [
                (axes[0], attn, "attn-block at last token"),
                (axes[1], mlp, "MLP-block at last token"),
            ]:
                im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                               vmin=0, vmax=max(1.2, float(mat.max())))
                ax.set_xticks(range(mat.shape[1]))
                ax.set_xticklabels([f"L{l}" for l in range(mat.shape[1])], fontsize=8)
                ax.set_yticks(range(mat.shape[0]))
                ax.set_yticklabels([f"step {s+1}" for s in range(mat.shape[0])], fontsize=8)
                ax.set_title(title, fontsize=10)
                for s in range(mat.shape[0]):
                    for l in range(mat.shape[1]):
                        v = mat[s, l]
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=6.5, color="white" if v < 0.5 else "black")
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Synthesis slide ===
        text_slide(pdf, "Synthesis — what is each latent step actually doing?",
            [
                "# Step 1: REAL number-dependent computation lives here",
                "- Force-decode acc at step 1: 36.0%. At step 2: 39.6% (+3.6 pp).",
                "- Step 1 zero-ablated on vary_numerals: 36/80 changed (recovery 55%); acc 43.8% (baseline 72.5%).",
                "- std/‖mean‖ at step 1 L0-L3 MLP ≈ 0.85-1.08 — large per-example variance.",
                "  This is where number-specific signal actually differs between examples.",
                "",
                "# Step 1 → 2: the only NET-useful transition",
                "- 70 wrong→right vs 34 right→wrong → +36 net gain (3.6 pp).",
                "- Both this and the 90-flip churn at step 2→3 are 'big', but only 1→2",
                "  has positive net. The model EXTRACTS something useful here.",
                "",
                "# Step 2 → 3: high churn, zero net (90 flips, net = 0)",
                "- 45 wrong→right, 45 right→wrong. Cancels out.",
                "- Not 'just formatting' — there's genuine per-example movement.",
                "- But movement is symmetric so it doesn't accumulate utility.",
                "  Possibly noisy refinement that helps some examples, hurts equally many.",
                "",
                "# Steps 3-6: closer to formatting",
                "- Total churn drops to 16-27 flips per transition.",
                "- std/‖mean‖ at step 3-6 mostly 0.04-0.34 — tight cluster, template-shared.",
                "- Zero-ablating steps 3-6 changes 2-9/80 examples on vary_numerals.",
                "- Late MLPs (L8-L11) at even steps write 'answer is :' tokens",
                "  (from head_content analysis) — literal answer-template formatting.",
                "",
                "# Why mean-CF patching looked weak before",
                "- At steps 2-6, the CF set's residuals are nearly identical (small variance).",
                "- Mean ≈ each example → patching with mean ≈ no perturbation.",
                "- Zero-ablation (extreme) reveals the actual fragility: step 1 IS necessary.",
            ])

    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
