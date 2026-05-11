"""Master deck on 'where does CODI-GPT-2 do its latent-loop computation?'

Synthesizes 6 prior probes:
1. Force-decode per K (latent_use_slideshow / force_decode_per_step.json).
2. Mean-CF patch recovery (patch_recovery_sweep.json).
3. Zero-ablation per step + std/‖mean‖ (patch_sanity_check.json).
4. Step-2 per-layer zero ablation (step2_layer_ablate.json).
5. Step 1→2 trajectory diff + cohort analysis (step1to2_deep_dive.json).
6. Step 1→2 feature-direction projections (step1to2_feature_projection.json).

Audience: someone who already has the flow_map + head_content background.
Use after `flow_map_slideshow.pdf` and `head_content_slideshow.pdf`.

Output: latent_compute_master_slideshow.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
OUT = PD / "latent_compute_master_slideshow.pdf"


def text_slide(pdf, title, lines, fontsize_body=10):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax = fig.add_axes([0.05, 0.04, 0.9, 0.86]); ax.axis("off")
    y = 0.97
    for ln in lines:
        if ln.startswith("# "):
            ax.text(0.0, y, ln[2:], fontsize=12, fontweight="bold",
                    transform=ax.transAxes); y -= 0.045
        elif ln.startswith("- "):
            ax.text(0.02, y, "•  " + ln[2:], fontsize=fontsize_body,
                    transform=ax.transAxes); y -= 0.034
        elif ln == "":
            y -= 0.020
        else:
            ax.text(0.0, y, ln, fontsize=fontsize_body, transform=ax.transAxes); y -= 0.034
    pdf.savefig(fig, dpi=140); plt.close(fig)


def main():
    fd = json.load(open(PD / "force_decode_per_step.json"))
    rec = json.load(open(PD / "patch_recovery_sweep.json"))
    san = json.load(open(PD / "patch_sanity_check.json"))
    s2lab = json.load(open(PD / "step2_layer_ablate.json"))
    dd = json.load(open(PD / "step1to2_deep_dive.json"))
    fp = json.load(open(PD / "step1to2_feature_projection.json"))

    with PdfPages(OUT) as pdf:
        # ===== Title =====
        text_slide(pdf,
            "CODI-GPT-2 latent loop: where is the computation done?",
            [
                "# Six probes, one synthesis",
                "1. Force-decode per latent step — where is the answer first decodable?",
                "2. Mean-CF activation patching — does patching individual cells change output?",
                "3. Zero-ablation per step + std/‖mean‖ — sanity check + variance audit.",
                "4. Step-2 per-layer zero ablation — which layer of step 2 does the 1→2 work?",
                "5. Step 1→2 trajectory diff + cohort comparison (wr vs rw)",
                "   — what's added in the residual between steps 1 and 2?",
                "6. Step 1→2 delta projected onto feature directions",
                "   — is the addition encoding operator/magnitude/correctness/answer?",
                "",
                "# Where to find the data",
                "- All intermediate JSONs in this directory.",
                "- Activations on HuggingFace: sandrajyluo/codi-gpt2-svamp-activations",
                "",
                "# Companion decks",
                "- flow_map_slideshow.pdf — per-(step, layer, head) attention pattern.",
                "- head_content_slideshow.pdf — per-head residual contribution decode.",
                "- latent_use_slideshow.pdf — original force-decode + LM-head probes.",
            ])

        # ===== Probe 1: force-decode =====
        trs = fd["transitions"][:5]
        labels = [f"{t['from_step']}→{t['to_step']}" for t in trs]
        wr = np.array([t["wrong_to_right"] for t in trs])
        rw = np.array([t["right_to_wrong"] for t in trs])
        total = wr + rw
        accs = np.array(fd["accuracy_per_step"][:6]) * 100
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle("Probe 1 — Force-decode per latent step (GPT-2 on SVAMP, N=1000)",
                     fontsize=12, fontweight="bold")
        axes[0].plot(range(1, 7), accs, "o-", lw=2, markersize=9, color="#1f77b4")
        for k, v in enumerate(accs, start=1):
            axes[0].annotate(f"{v:.1f}%", (k, v), xytext=(0, 7), textcoords="offset points",
                             ha="center", fontsize=9)
        axes[0].set_xlabel("latent step K"); axes[0].set_ylabel("force-decode accuracy (%)")
        axes[0].set_title("Accuracy per K (peak at K=4, all small variation)")
        axes[0].set_xticks(range(1, 7)); axes[0].set_ylim(35, 42); axes[0].grid(alpha=0.3)
        xs = np.arange(5); w = 0.4
        axes[1].bar(xs - w/2, wr, w, color="#2ca02c", label="w→r")
        axes[1].bar(xs + w/2, -rw, w, color="#d62728", label="r→w (−)")
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_xticks(xs); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("# examples"); axes[1].set_title("Net flips per transition")
        for i, (a, b) in enumerate(zip(wr, rw)):
            axes[1].text(xs[i], max(a, 0) + 3, f"net={a-b:+d}", ha="center", fontsize=8)
        axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
        axes[2].bar(xs, total, color="#1f77b4", edgecolor="black", lw=0.5)
        axes[2].set_xticks(xs); axes[2].set_xticklabels(labels)
        axes[2].set_ylabel("total flips (right↔wrong)")
        axes[2].set_title("Total CHURN per transition")
        for i, v in enumerate(total):
            axes[2].text(xs[i], v + 2, f"{v}", ha="center", fontsize=8)
        axes[2].grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)
        text_slide(pdf, "Probe 1 — interpretation",
            [
                "# Per-step accuracy",
                "- step 1: 36.0%, step 2: 39.6%, peaks at step 4: 39.7%, drifts back to 39.2%.",
                "- Almost ALL the gain is achieved at step 1→2 (+3.6 pp). Later steps drift ±1 pp.",
                "",
                "# Net flips",
                "- 1→2: +36 (the only meaningfully positive transition).",
                "- 2→3: 0 (45 w→r and 45 r→w — perfectly canceling).",
                "- 3→4 to 5→6: |net| ≤ 3.",
                "",
                "# Total churn",
                "- 1→2: 104 examples flipped (in either direction).",
                "- 2→3: 90 examples flipped — almost as much, but cancels out.",
                "- 3→4, 4→5, 5→6: 27 / 25 / 16 flips — much smaller.",
                "",
                "# Takeaway",
                "- Step 1→2 is the ONLY NET-useful transition.",
                "- Step 2→3 still has a lot of activity, just symmetric — likely noisy refinement.",
                "- Steps 3-6 are quiet (small total flips, near-zero net).",
            ])

        # ===== Probe 2 & 3: recovery + sanity =====
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle("Probe 2 vs Probe 3 — mean-CF patching (weak) vs zero ablation (strong)",
                     fontsize=12, fontweight="bold")
        for ax, key in zip(axes, ["vary_a_2digit", "vary_numerals"]):
            sw = rec["cf_sets"][key]["sweeps"]
            base = rec["cf_sets"][key]["baseline_accuracy"] * 100
            rec_step = [sw[f"step{s+1}"]["recovery_rate"] * 100 for s in range(6)]
            zero = san[key]["zero_ablation_per_step"]
            zero_step = [zero[f"step{s+1}"]["recovery_rate"] * 100 for s in range(6)]
            zero_acc = [zero[f"step{s+1}"]["accuracy"] * 100 for s in range(6)]
            xs = np.arange(6); w = 0.35
            ax.bar(xs - w, rec_step, w, color="#9467bd", label="mean-CF patch", edgecolor="black", lw=0.4)
            ax.bar(xs, zero_step, w, color="#ff7f0e", label="ZERO ablate", edgecolor="black", lw=0.4)
            ax.bar(xs + w, zero_acc, w, color="#d62728", label="acc after ZERO", edgecolor="black", lw=0.4)
            ax.axhline(base, color="black", lw=0.5, ls=":", label=f"baseline acc {base:.0f}%")
            ax.axhline(100, color="gray", lw=0.5, ls=":")
            ax.set_xticks(xs); ax.set_xticklabels([f"s{s+1}" for s in range(6)])
            ax.set_ylim(0, 110); ax.set_ylabel("%")
            ax.set_title(f"{key}  (baseline acc {base:.0f}%)", fontsize=10)
            ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ===== std/‖mean‖ heatmap =====
        for cf_name in ["vary_numerals"]:
            r = san[cf_name]
            attn = np.array(r["std_over_mean_attn"])
            mlp = np.array(r["std_over_mean_mlp"])
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Probe 3b — std/‖mean‖ across {r['N']} CF examples — {cf_name}\n"
                         "Small ratio → activations are tightly clustered → mean-patching is ~no-op",
                         fontsize=11, fontweight="bold")
            for ax, mat, title in [
                (axes[0], attn, "attn-block at last token"),
                (axes[1], mlp, "MLP-block at last token"),
            ]:
                im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                               vmin=0, vmax=max(1.2, float(mat.max())))
                ax.set_xticks(range(mat.shape[1]))
                ax.set_xticklabels([f"L{l}" for l in range(mat.shape[1])], fontsize=8)
                ax.set_yticks(range(mat.shape[0]))
                ax.set_yticklabels([f"s{s+1}" for s in range(mat.shape[0])], fontsize=8)
                ax.set_title(title, fontsize=10)
                for s in range(mat.shape[0]):
                    for l in range(mat.shape[1]):
                        v = mat[s, l]
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=6.5, color="white" if v < 0.5 else "black")
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # ===== Probe 4: step-2 per-layer =====
        L = 12
        delta_acc = [s2lab["per_layer"][str(l)]["delta_acc_vs_baseline_s2"] * 100 for l in range(L)]
        delta_net = [s2lab["per_layer"][str(l)]["delta_net_vs_baseline"] for l in range(L)]
        changed = [s2lab["per_layer"][str(l)]["n_changed_from_baseline_s2"] for l in range(L)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle("Probe 4 — zero-ablate one layer of step 2 (which layer carries the 1→2 work?)",
                     fontsize=12, fontweight="bold")
        axes[0].bar(range(L), delta_acc,
                    color=['#d62728' if v < 0 else '#2ca02c' for v in delta_acc],
                    edgecolor='black', lw=0.5)
        axes[0].axhline(0, color="black", lw=0.5)
        axes[0].set_xticks(range(L)); axes[0].set_xlabel("zero-ablated layer of step 2")
        axes[0].set_ylabel("Δ step-2 acc (pp)")
        axes[0].set_title("Step-2 accuracy change"); axes[0].grid(axis="y", alpha=0.3)
        axes[1].bar(range(L), delta_net,
                    color=['#d62728' if v < 0 else '#2ca02c' for v in delta_net],
                    edgecolor='black', lw=0.5)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_xticks(range(L)); axes[1].set_xlabel("zero-ablated layer of step 2")
        axes[1].set_ylabel("Δ 1→2 net flips")
        axes[1].set_title("Change in 1→2 net flip count"); axes[1].grid(axis="y", alpha=0.3)
        axes[2].bar(range(L), changed, color="#1f77b4", edgecolor="black", lw=0.5)
        axes[2].set_xticks(range(L)); axes[2].set_xlabel("zero-ablated layer of step 2")
        axes[2].set_ylabel("# examples with changed step-2 pred (of 1000)")
        axes[2].set_title("Per-example churn"); axes[2].grid(axis="y", alpha=0.3)
        for x, v in zip(range(L), delta_acc):
            axes[0].text(x, v + (0.05 if v >= 0 else -0.1), f"{v:+.1f}", ha="center", fontsize=7)
        for x, v in zip(range(L), delta_net):
            axes[1].text(x, v + (0.2 if v >= 0 else -0.4), f"{v:+d}", ha="center", fontsize=7)
        for x, v in zip(range(L), changed):
            axes[2].text(x, v + 2, f"{v}", ha="center", fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        text_slide(pdf, "Probe 4 — interpretation",
            [
                "# What the per-layer zero ablation reveals",
                "- Δ step-2 acc is ±0.5 pp at every layer (baseline 39.5%).",
                "- Δ net 1→2 flips is ±5 (baseline +41). No single layer absorbs the +41 gain.",
                "- BUT per-layer ablation changes 51-174 individual predictions.",
                "  Most active: L0 (174 changed), L1-L6 (75-96), L7-L10 (51-75), L11 (0).",
                "",
                "# Read",
                "- The +36 net 1→2 gain is REDUNDANT across all 11 effective layers of step 2.",
                "- L11's output isn't even used by decode (KV cache locked before its outputs).",
                "- L0 of step 2 sees the most individual movement — that's the layer that",
                "  reads L1 (step-1's just-computed latent) in the residual. But even that",
                "  isn't 'where the computation lives' in a localized sense.",
                "",
                "# Conclusion",
                "- Distributed computation. No single layer of step 2 is necessary.",
            ])

        # ===== Probe 5: trajectory diff =====
        L = 13
        norm_overall = dd["delta_norm_overall"]
        norm_wr = dd["delta_norm_wr"]
        norm_rw = dd["delta_norm_rw"]
        attn_Q = dd["attn_to_Q_step2"]
        attn_L1 = dd["attn_to_L1_step2"]
        attn_BOT = dd["attn_to_BOT_step2"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle("Probe 5 — Step 1→2 trajectory: residual change magnitude + step-2 attention",
                     fontsize=12, fontweight="bold")
        axes[0].plot(range(L), norm_overall, "o-", lw=2, color="#1f77b4", label="all")
        axes[0].plot(range(L), norm_wr, "s-", color="#2ca02c", label=f"wr (n={dd['n_wr']})")
        axes[0].plot(range(L), norm_rw, "^-", color="#d62728", label=f"rw (n={dd['n_rw']})")
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("‖mean Δ residual‖ (step 2 − step 1)")
        axes[0].set_title("Magnitude of 1→2 residual change per layer")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
        axes[1].plot(range(len(attn_Q)), [v * 100 for v in attn_Q], "o-", color="#1f77b4", label="to Q")
        axes[1].plot(range(len(attn_L1)), [v * 100 for v in attn_L1], "s-", color="#2ca02c", label="to L1")
        axes[1].plot(range(len(attn_BOT)), [v * 100 for v in attn_BOT], "^-", color="#ff7f0e", label="to BOT")
        axes[1].set_xlabel("layer"); axes[1].set_ylabel("mean attention (%)")
        axes[1].set_title("Step-2 latent attention by class per layer")
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 100)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # ===== Probe 6: feature projection =====
        fkeys = fp["features"]
        for cohort, title in [("overall", "all 1000"),
                              ("wr", "wr cohort (n=70)"),
                              ("rw", "rw cohort (n=34)")]:
            mat = np.array([fp["cos_per_layer"][k][cohort] for k in fkeys])
            fig, ax = plt.subplots(figsize=(13, 5))
            vmax = max(0.1, np.abs(mat).max())
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels([f"L{l}" for l in range(mat.shape[1])], fontsize=8)
            ax.set_yticks(range(len(fkeys))); ax.set_yticklabels(fkeys, fontsize=9)
            ax.set_title(f"Probe 6 — cos(mean delta, feature direction) per layer — {title}",
                         fontsize=11, fontweight="bold")
            for i in range(len(fkeys)):
                for l in range(mat.shape[1]):
                    v = mat[i, l]
                    if abs(v) >= 0.03:
                        ax.text(l, i, f"{v:+.2f}", ha="center", va="center",
                                fontsize=6.5,
                                color="white" if abs(v) > 0.6 * vmax else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # wr − rw differential
        diff = np.array([np.array(fp["cos_per_layer"][k]["wr"]) - np.array(fp["cos_per_layer"][k]["rw"])
                         for k in fkeys])
        fig, ax = plt.subplots(figsize=(13, 5))
        vmax = max(0.05, np.abs(diff).max())
        im = ax.imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(diff.shape[1])); ax.set_xticklabels([f"L{l}" for l in range(diff.shape[1])], fontsize=8)
        ax.set_yticks(range(len(fkeys))); ax.set_yticklabels(fkeys, fontsize=9)
        ax.set_title("Δ cos(wr) − Δ cos(rw) — features successful 1→2 transitions add MORE OF",
                     fontsize=11, fontweight="bold")
        for i in range(len(fkeys)):
            for l in range(diff.shape[1]):
                v = diff[i, l]
                if abs(v) >= 0.02:
                    ax.text(l, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(v) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        text_slide(pdf, "Probe 6 — interpretation",
            [
                "# Method",
                "- Fit linear probe directions at step 1, each layer, for: operator (4-class),",
                "  magnitude (regression on log10 answer), correctness (step-1 success),",
                "  faithfulness (judged labels), answer-token (per-example lm_head row).",
                "- Project the (step 2 − step 1) mean delta onto each direction.",
                "- cos = how much of the delta lies along that feature axis.",
                "",
                "# Results",
                "- All projections are small: |cos| ≤ 0.07 at every (feature, layer).",
                "- Answer-token cos peaks at L12 (+0.08); negative/near-zero elsewhere.",
                "- wr − rw differential: the correctness direction shows the cleanest signal,",
                "  with wr ~+0.02-0.03 vs rw ~−0.02 at mid layers (L4-L7).",
                "",
                "# Conclusion",
                "- The 1→2 delta lives in a HIGH-DIMENSIONAL subspace orthogonal to all single",
                "  feature directions tested.",
                "- It's not 'add the operator' or 'add the magnitude' — it's a distributed update.",
                "- The lm_head decoding of mid-layer deltas (Probe 5 results: 'Berm', 'pmwiki') was",
                "  noise — these tokens aren't actually in the residual; they just have unembedding",
                "  rows with non-trivial cosine to whatever direction the delta points at L4-L8.",
                "- Only L10-L12 lm_head decoding is meaningful (number tokens like 'one', '37', '67').",
            ])

        # ===== Synthesis =====
        text_slide(pdf, "Synthesis — what is each latent step doing?",
            [
                "# Step 1: NUMBER-DEPENDENT COMPUTATION",
                "- Force-decode acc 36.0%. std/‖mean‖ at L0-L3 MLP ≈ 0.85-1.08 (large per-example variance).",
                "- Zero-ablate step 1 on vary_numerals: 36/80 changed, acc 72.5%→43.8%.",
                "- Real number-specific signal lives here.",
                "",
                "# Step 1→2: ONLY NET-USEFUL TRANSITION (+36)",
                "- Step 2 attends 64% to Q + 10% to L1 at L0; 87-96% to Q at later layers.",
                "- ‖Δ‖ grows from L0 (16) to L11 (140); drops to 33 at L12.",
                "- Late layers (L10-L11) write CANDIDATE NUMBER tokens ('one', 'two', '37', '67').",
                "- BUT each individual layer is unnecessary — the +36 gain survives any single",
                "  layer's ablation. Redundant, distributed computation.",
                "- The delta is orthogonal to all standard feature probes — it's a distributed",
                "  update, not a simple 'add operator' or 'add magnitude'.",
                "",
                "# Step 2→3: HIGH CHURN, ZERO NET (90 flips, net 0)",
                "- Per-example movement is large, but symmetric (45 each way).",
                "- Std/‖mean‖ drops to 0.05-0.3 — latent residual is now near a template centroid.",
                "- Likely noisy refinement around the now-committed answer.",
                "",
                "# Steps 3-6: FORMATTING / RE-COMMITTING",
                "- 16-27 total flips, |net| ≤ 3.",
                "- Late MLPs at even steps write 'answer is :' tokens (from head_content analysis).",
                "- Mean-patching does nothing; zero-ablation changes 2-9/80 examples.",
                "",
                "# Big picture",
                "- The latent loop is closer to 'encode question → propose answer' than to",
                "  'iterative reasoning.'",
                "- Most useful work happens at step 1→2.",
                "- Subsequent steps are mostly stylistic re-commitment of the same answer.",
            ])

    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
