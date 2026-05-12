"""Anatomy of CODI's 6 latent steps on GSM8K — combines all existing data.

Answers three questions using only already-computed JSON/NPZ files (no GPU):

  Q1. Are some steps more important than others?
      → Aggregate ZERO-ablation damage per step (mean over layers, mean over CF
        sets). Force-decode accuracy per step (does the answer settle later?).
        Per-step ||attn|| and ||MLP|| block-output norms.

  Q2. What is each step reading from?
      → flow_map attention to position classes (Q, BOT, L1..L6, EOT, D0..D9)
        averaged over heads and layers, per latent step.

  Q3. Is there chaining structure between markers?
      → Per-step attention to the IMMEDIATELY PRIOR latent (L_{k-1}) vs older
        latents (L_{1..k-2}). High step-k→L_{k-1} attention is the chaining
        signature.
      → Cross-marker probe transfer: train op probe at cell_m on marker m's
        label, test it on marker m's label at the SAME cell. Then test on
        marker m' ≠ m's label at the same cell. If the cell is marker-specific
        (the chaining-of-markers picture), cross-marker accuracy should drop
        sharply.

Inputs (all already on disk):
  experiments/computation_probes/flow_map_gsm8k.npz + meta
  experiments/computation_probes/ablation_mlp_attn_gsm8k.json
  experiments/computation_probes/force_decode_per_step_gsm8k.json
  visualizations-all/gpt2-gsm8k/correctness-analysis/zero_mean_per_cell_gsm8k.json
  visualizations-all/gpt2-gsm8k/operator-probe/multi_op_probe_gsm8k.json

Output: visualizations-all/gpt2-gsm8k/step_anatomy_gsm8k.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
CP = REPO / "experiments" / "computation_probes"
CA = PD / "correctness-analysis"
OPER = PD / "operator-probe"

OUT_PDF = PD / "step_anatomy_gsm8k.pdf"


def load(p):
    try:
        if str(p).endswith(".npz"):
            return np.load(p)
        return json.load(open(p))
    except FileNotFoundError:
        return None


def main():
    flow = load(CP / "flow_map_gsm8k.npz")
    flow_meta = load(CP / "flow_map_gsm8k_meta.json")
    abl_attn = load(CP / "ablation_mlp_attn_gsm8k.json")
    fdec = load(CP / "force_decode_per_step_gsm8k.json")
    zm = load(CA / "zero_mean_per_cell_gsm8k.json")
    mop = load(OPER / "multi_op_probe_gsm8k.json")

    if flow is None or flow_meta is None:
        print("flow_map missing"); return

    # ----- Common dims -----
    ATTN = flow["mean_attn"]         # (phase=2, decode_steps=10, layers, heads, classes)
    ATTN_NORM = flow["mean_attn_norm"]  # (phase, step, layer)
    MLP_NORM = flow["mean_mlp_norm"]
    CLASS_NAMES = flow_meta["class_names"]
    N_LAT = flow_meta["N_latent_steps"]
    N_DEC = flow_meta["N_decode_steps"]
    N_LAYERS = flow_meta["N_layers"]
    N_HEADS = flow_meta["N_heads"]

    print(f"N_LAT={N_LAT} N_LAYERS={N_LAYERS} N_HEADS={N_HEADS} "
          f"classes={CLASS_NAMES}")

    with PdfPages(OUT_PDF) as pdf:
        # ============================================================
        # Page 1: Setup
        # ============================================================
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = ("Anatomy of CODI's 6 latent steps — what they read, do, and chain to.\n\n"
                "  Q1: Step importance (ablation damage, force-decode acc, block norms)\n"
                "  Q2: What is each step reading from?\n"
                "  Q3: Chaining: does step k attend to L_{k-1}, AND is each cell\n"
                "      marker-specific (per probe transfer matrix)?\n\n"
                "All from existing JSONs and the GSM8K flow_map. No new compute.\n")
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=11)
        ax.set_title("Step anatomy — overview", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ============================================================
        # Q1/Q2/Q3 are emitted THREE times, parameterized by stop_step ∈
        # {6 (default), 3, 5}. For each stop_step, the analyses are
        # restricted to latent steps 1..stop_step. Force-decode accuracy
        # is shown specifically as "if we stopped here".
        # ============================================================
        latent_classes_all = ["Q", "BOT", "L1", "L2", "L3", "L4", "L5", "L6"]
        attn_avg = ATTN[0, :N_LAT, :, :, :].mean(axis=(1, 2))  # (step, classes)
        colors = plt.cm.tab10(np.linspace(0, 1, len(latent_classes_all)))

        # Cross-marker probe transfer (stop_step-independent; computed once)
        xfer = None
        if mop is not None and "best" in mop and "acc" in mop:
            best = mop["best"]
            M_MAX = 4
            xfer = np.full((M_MAX, M_MAX), np.nan)
            for m1 in range(1, M_MAX + 1):
                b1 = best["op"].get(str(m1))
                if b1 is None: continue
                s_1 = b1["step"] - 1; l_1 = b1["layer"]
                for m2 in range(1, M_MAX + 1):
                    G2 = np.array(mop["acc"]["op"][str(m2)])
                    xfer[m1 - 1, m2 - 1] = G2[s_1, l_1]

        def render_q1(stop_step):
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            steps_keep = np.arange(1, stop_step + 1)
            # (a) Force-decode accuracy at each step ≤ stop_step
            if fdec is not None and "correct_per_step" in fdec:
                N = fdec["N"]
                acc_per_step = [sum(c) / N for c in fdec["correct_per_step"][:stop_step]]
                axes[0].bar(steps_keep, [a * 100 for a in acc_per_step],
                            color="#4c72b0", edgecolor="black")
                for s, a in zip(steps_keep, acc_per_step):
                    axes[0].text(s, a * 100 + 0.5, f"{a*100:.1f}", ha="center", fontsize=8)
                axes[0].axvline(stop_step, color="#d62728", ls="--", alpha=0.6,
                                label=f"stop@{stop_step}: acc={acc_per_step[-1]*100:.1f}%")
                axes[0].set_xlabel("latent step (≤ stop_step)")
                axes[0].set_ylabel("force-decode acc (%)")
                axes[0].set_title(f"Force-decode acc per step (stop@{stop_step})",
                                  fontsize=10, fontweight="bold")
                axes[0].set_xticks(steps_keep)
                axes[0].legend(fontsize=8)
                axes[0].grid(axis="y", alpha=0.3)
            # (b) ZERO-ablation damage per step ≤ stop_step (data is from
            #     full-6-step runs; we slice it)
            if zm is not None:
                per_step = np.zeros(N_LAT); cnt = np.zeros(N_LAT)
                for cf, sub in zm.items():
                    for key, c in sub["conditions"].items():
                        if c.get("mode") != "zero": continue
                        s = c["step"] - 1
                        per_step[s] += -c["delta_acc"]
                        cnt[s] += 1
                mean_dmg = per_step / np.where(cnt == 0, 1, cnt)
                axes[1].bar(steps_keep, mean_dmg[:stop_step] * 100,
                            color="#d62728", edgecolor="black")
                for s, d in zip(steps_keep, mean_dmg[:stop_step]):
                    axes[1].text(s, d * 100 + 0.3, f"{d*100:+.1f}",
                                  ha="center", fontsize=8)
                axes[1].set_xlabel("latent step"); axes[1].set_ylabel("mean damage (pp)")
                axes[1].set_title("ZERO-ablation damage per step\n"
                                  "(measured under full-6 decoding; sliced to steps 1..stop)",
                                  fontsize=9, fontweight="bold")
                axes[1].set_xticks(steps_keep); axes[1].grid(axis="y", alpha=0.3)
            # (c) Block-output norms per step ≤ stop_step
            attn_norm_per_step = ATTN_NORM[0, :stop_step].mean(axis=-1)
            mlp_norm_per_step = MLP_NORM[0, :stop_step].mean(axis=-1)
            w = 0.4
            axes[2].bar(steps_keep - w/2, attn_norm_per_step, w, color="#4c72b0",
                        edgecolor="black", label="‖attn out‖")
            axes[2].bar(steps_keep + w/2, mlp_norm_per_step, w, color="#dd8452",
                        edgecolor="black", label="‖MLP out‖")
            axes[2].set_xlabel("latent step"); axes[2].set_ylabel("mean block-output norm")
            axes[2].set_title("How much each step writes to residual",
                              fontsize=10, fontweight="bold")
            axes[2].set_xticks(steps_keep); axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)
            fig.suptitle(f"Q1: Step importance — STOP@{stop_step}  (sub-Q1 for this endpoint)",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        def render_q2(stop_step):
            # Stacked attention shares for steps 1..stop_step, classes Q, BOT,
            # and L1..L_{stop_step}. Higher-index latents are not yet visible
            # at any step ≤ stop_step.
            cls_subset = ["Q", "BOT"] + [f"L{j}" for j in range(1, stop_step + 1)]
            cls_idx = [CLASS_NAMES.index(c) for c in cls_subset]
            attn_subset = attn_avg[:stop_step, cls_idx]  # (stop_step, len(cls_subset))
            fig, ax = plt.subplots(figsize=(13, 5.5))
            steps_keep = np.arange(1, stop_step + 1)
            bot = np.zeros(stop_step)
            for ci, cname in enumerate(cls_subset):
                vals = attn_subset[:, ci]
                ax.bar(steps_keep, vals, bottom=bot,
                       color=colors[ci % len(colors)], label=cname,
                       edgecolor="white", linewidth=0.3)
                bot += vals
            ax.set_xlabel("latent step"); ax.set_ylabel("mean attention (avg over heads, layers)")
            ax.set_xticks(steps_keep)
            ax.legend(ncol=8, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.10))
            ax.set_title(f"Q2: What does each step read from?  STOP@{stop_step}",
                         fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        def render_q3(stop_step):
            # Chaining: step k → L_{k-1} vs older priors, for k=2..stop_step.
            # Cross-marker probe-transfer matrix (stop_step-independent) included.
            chain_imm = np.zeros(stop_step)
            chain_old = np.zeros(stop_step)
            for k in range(stop_step):   # 0-indexed step
                if k == 0: continue
                imm_ci = CLASS_NAMES.index(f"L{k}")
                imm = attn_avg[k, imm_ci]
                older = 0
                for j in range(1, k):
                    ci = CLASS_NAMES.index(f"L{j}")
                    older += attn_avg[k, ci]
                chain_imm[k] = imm
                chain_old[k] = max(0, older - imm)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            steps_keep = np.arange(1, stop_step + 1)
            w = 0.4
            axes[0].bar(steps_keep - w/2, chain_imm, w, color="#2ca02c",
                        edgecolor="black", label="L_{k-1} (immediate prior)")
            axes[0].bar(steps_keep + w/2, chain_old, w, color="#cccccc",
                        edgecolor="black", label="L_{1..k-2} (older priors)")
            axes[0].set_xlabel("latent step k"); axes[0].set_ylabel("mean attention")
            axes[0].set_xticks(steps_keep)
            axes[0].set_title("Chaining attention per step (stop window)",
                              fontsize=10, fontweight="bold")
            axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
            if xfer is not None:
                im = axes[1].imshow(xfer, aspect="auto", cmap="viridis",
                                    origin="upper", vmin=0.2, vmax=max(0.6, np.nanmax(xfer)))
                axes[1].set_xticks(range(xfer.shape[1]))
                axes[1].set_xticklabels([f"predict m={m+1}" for m in range(xfer.shape[1])])
                axes[1].set_yticks(range(xfer.shape[0]))
                axes[1].set_yticklabels([f"cell from m={m+1}" for m in range(xfer.shape[0])])
                for i in range(xfer.shape[0]):
                    for j in range(xfer.shape[1]):
                        axes[1].text(j, i, f"{xfer[i, j]:.2f}", ha="center", va="center",
                                     fontsize=10,
                                     color="white" if xfer[i, j] < 0.42 else "black",
                                     fontweight="bold" if i == j else "normal")
                axes[1].set_title("Cross-marker probe-transfer (stop-step independent)",
                                  fontsize=10, fontweight="bold")
                fig.colorbar(im, ax=axes[1], fraction=0.045, label="test acc")
            fig.suptitle(f"Q3: Chaining structure — STOP@{stop_step}",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Emit Q1, Q2, Q3 three times — default 6, then stop@3, then stop@5.
        for stop_step in [6, 3, 5]:
            # Section header
            fig, ax = plt.subplots(figsize=(11, 2))
            ax.axis("off")
            label = "DEFAULT (full 6 latent steps)" if stop_step == 6 else f"STOP@{stop_step}"
            ax.text(0.5, 0.5, f"——— {label} ———\nQ1, Q2, Q3 below restricted to steps 1..{stop_step}",
                    ha="center", va="center", fontsize=14, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            render_q1(stop_step)
            render_q2(stop_step)
            render_q3(stop_step)

        # ============================================================
        # Page 5: Decode-phase reading patterns
        # ============================================================
        # During decode (the first 10 tokens after EOT), what's attention focus?
        # ATTN[1, dec_step, layer, head, class]
        decode_classes = ["Q", "BOT", "L1", "L2", "L3", "L4", "L5", "L6", "EOT"]
        dec_cls_idx = [CLASS_NAMES.index(c) for c in decode_classes]
        attn_dec = ATTN[1, :N_DEC, :, :, :].mean(axis=(1, 2))   # (dec_step, classes)
        attn_dec_classes = attn_dec[:, dec_cls_idx]

        fig, ax = plt.subplots(figsize=(13, 5))
        bot = np.zeros(N_DEC)
        for ci, cname in enumerate(decode_classes):
            vals = attn_dec_classes[:, ci]
            ax.bar(np.arange(1, N_DEC + 1), vals, bottom=bot,
                   color=colors[ci % len(colors)], label=cname,
                   edgecolor="white", linewidth=0.3)
            bot += vals
        ax.set_xlabel("decode step (token after EOT)"); ax.set_ylabel("mean attention")
        ax.set_xticks(range(1, N_DEC + 1))
        ax.legend(ncol=9, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.10))
        ax.set_title("Decode-phase attention: which latents/positions feed the emit?\n"
                     "(Especially decode step 1 — the first emitted answer token.)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ================================================================
        # NEW: Step 3 and Step 5 detailed views — these are the "reading"
        # steps where Q-attention drops and prior-latent attention rises.
        # For each of those steps: ablation damage per layer (Q1), attention
        # breakdown per layer (Q2), and per-layer attention to prior latents
        # (Q3) — all three Q's at the marker-emergence steps.
        # ================================================================
        for FOCUS_STEP_1IDX in [3, 5]:
            FS = FOCUS_STEP_1IDX - 1   # 0-indexed
            # Q1: per-LAYER ablation damage AT this step (zero-ablation Δacc)
            damage_by_layer = np.full(N_LAYERS, np.nan)
            if zm is not None:
                tot = np.zeros(N_LAYERS); cnt_l = np.zeros(N_LAYERS)
                for cf, sub in zm.items():
                    for key, c in sub["conditions"].items():
                        if c.get("mode") != "zero": continue
                        if c["step"] != FOCUS_STEP_1IDX: continue
                        tot[c["layer"]] += -c["delta_acc"]
                        cnt_l[c["layer"]] += 1
                damage_by_layer = tot / np.where(cnt_l == 0, 1, cnt_l)

            # Q2: per-(layer, class) attention shares AT this step
            # ATTN shape: (phase, step, layer, head, class). Avg over heads.
            attn_step_layer_class = ATTN[0, FS, :, :, :].mean(axis=1)  # (layer, class)

            # Q3: per-layer attention from this step to L_{k-1} vs older priors
            chain_immediate_pl = np.zeros(N_LAYERS)
            chain_earlier_pl = np.zeros(N_LAYERS)
            if FS > 0:
                imm_ci = CLASS_NAMES.index(f"L{FS}")  # 1-indexed prior
                for L in range(N_LAYERS):
                    chain_immediate_pl[L] = attn_step_layer_class[L, imm_ci]
                    for j in range(1, FS):
                        ci = CLASS_NAMES.index(f"L{j}")
                        chain_earlier_pl[L] += attn_step_layer_class[L, ci]
                chain_earlier_pl = np.maximum(0, chain_earlier_pl - chain_immediate_pl)

            # Page: combined per-layer panels for this step
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            # (a) damage per layer at this step
            axes[0].bar(np.arange(N_LAYERS), damage_by_layer * 100,
                        color="#d62728", edgecolor="black")
            for L, d in enumerate(damage_by_layer):
                if not np.isnan(d):
                    axes[0].text(L, d * 100 + 0.2, f"{d*100:+.1f}",
                                  ha="center", fontsize=7)
            axes[0].set_xlabel("layer"); axes[0].set_ylabel("damage (pp)")
            axes[0].set_xticks(range(N_LAYERS))
            axes[0].set_title(f"Q1: per-layer zero-ablation damage @ step {FOCUS_STEP_1IDX}",
                              fontsize=10, fontweight="bold")
            axes[0].grid(axis="y", alpha=0.3)

            # (b) per-layer attention shares (stacked) — what does each layer read?
            classes_to_plot = ["Q", "BOT"] + [f"L{j}" for j in range(1, FS + 1)]
            cls_idx_p = [CLASS_NAMES.index(c) for c in classes_to_plot]
            stack = attn_step_layer_class[:, cls_idx_p]   # (layer, len(classes_to_plot))
            bot_acc = np.zeros(N_LAYERS)
            for ci, cname in enumerate(classes_to_plot):
                axes[1].bar(np.arange(N_LAYERS), stack[:, ci], bottom=bot_acc,
                             color=colors[ci % len(colors)], label=cname,
                             edgecolor="white", linewidth=0.3)
                bot_acc += stack[:, ci]
            axes[1].set_xlabel("layer"); axes[1].set_ylabel("attention share")
            axes[1].set_xticks(range(N_LAYERS))
            axes[1].set_title(f"Q2: per-layer attention @ step {FOCUS_STEP_1IDX}",
                              fontsize=10, fontweight="bold")
            axes[1].legend(fontsize=7, ncol=2)
            axes[1].grid(axis="y", alpha=0.3)

            # (c) per-layer chaining attention: L_{k-1} vs older priors
            w = 0.4
            axes[2].bar(np.arange(N_LAYERS) - w/2, chain_immediate_pl, w,
                         color="#2ca02c", edgecolor="black",
                         label=f"L_{FS} (immediate prior)" if FS > 0 else "(no prior)")
            axes[2].bar(np.arange(N_LAYERS) + w/2, chain_earlier_pl, w,
                         color="#cccccc", edgecolor="black",
                         label="older priors")
            axes[2].set_xlabel("layer"); axes[2].set_ylabel("attention")
            axes[2].set_xticks(range(N_LAYERS))
            axes[2].set_title(f"Q3: per-layer chaining attention @ step {FOCUS_STEP_1IDX}",
                              fontsize=10, fontweight="bold")
            axes[2].legend(fontsize=7); axes[2].grid(axis="y", alpha=0.3)

            fdec_acc_at_step = (sum(fdec["correct_per_step"][FS]) / fdec["N"]
                                if fdec is not None else None)
            title = (f"Detailed view of LATENT step {FOCUS_STEP_1IDX} "
                     f"— the 'reading step' that re-attends to prior latents")
            if fdec_acc_at_step is not None:
                title += f"\n(force-decode acc at step {FOCUS_STEP_1IDX} = {fdec_acc_at_step*100:.1f}%)"
            fig.suptitle(title, fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ================================================================
        # Force-decode-at-step comparison: what does step 3 emit vs step 5
        # vs step 6 (the natural end)?
        # ================================================================
        if fdec is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            # left: cumulative force-decode accuracy
            ax = axes[0]
            N_PROBS = fdec["N"]
            per_step_acc = [sum(c) / N_PROBS for c in fdec["correct_per_step"]]
            steps = np.arange(1, len(per_step_acc) + 1)
            ax.bar(steps, [a * 100 for a in per_step_acc], color="#4c72b0", edgecolor="black")
            for s, a in zip(steps, per_step_acc):
                ax.text(s, a * 100 + 0.3, f"{a*100:.1f}", ha="center", fontsize=8)
            for s_mark in [3, 5]:
                ax.axvline(s_mark, color="#d62728", ls="--", alpha=0.5,
                           label=f"step {s_mark} (reading step)" if s_mark == 3 else None)
            ax.set_xticks(steps); ax.set_xlabel("force-decoded after step")
            ax.set_ylabel("emit accuracy (%)")
            ax.set_title("If we force-decode after step k, what's the accuracy?\n"
                         "(red dashed = steps 3 & 5, the 'reading' steps)",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
            # right: change in emit between consecutive steps
            ax = axes[1]
            deltas = [per_step_acc[i] - per_step_acc[i - 1] for i in range(1, len(per_step_acc))]
            ax.bar(steps[1:], [d * 100 for d in deltas],
                   color=["#2ca02c" if d > 0 else "#d62728" for d in deltas],
                   edgecolor="black")
            for s, d in zip(steps[1:], deltas):
                ax.text(s, d * 100 + (0.2 if d >= 0 else -0.5), f"{d*100:+.1f}",
                        ha="center", fontsize=8)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(steps[1:]); ax.set_xlabel("step k (vs step k-1)")
            ax.set_ylabel("Δ acc (pp)")
            ax.set_title("Per-step accuracy gain — where does answer get refined?",
                         fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            fig.suptitle("Force-decode comparison across the 6 latent steps",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ================================================================
        # NEW: Logit-lens summary (if available)
        # ================================================================
        ll_path = CP / "logit_lens_gsm8k.json"
        if ll_path.exists():
            ll = json.load(open(ll_path))
            mean_conf = np.array(ll["mean_top1_conf"])  # (N_LAT, N_LAYERS, n_sublayers)
            modal = ll["modal_token"]
            SUBL = ll["SUBLAYERS"]
            fig, ax = plt.subplots(figsize=(12, 6.5))
            ax.axis("off")
            txt = (f"Logit-lens on CODI's 6 latent steps  (N={ll['N_examples']})\n"
                   f"For each (step, layer, sublayer): apply ln_f + lm_head to that\n"
                   f"hidden state at the last token position; record top-1 token + prob.\n\n"
                   f"  Modal top-1 token | mean confidence, per (step, layer, sublayer)\n\n")
            for s in range(ll["N_LAT"]):
                txt += f"  STEP {s+1}\n"
                for L in range(ll["N_LAYERS"]):
                    row = []
                    for sl_i, sl in enumerate(SUBL):
                        tk = modal[s][L][sl_i] or ""
                        tk_d = tk.replace("\n", "\\n")[:8]
                        row.append(f"{sl[:4]}:{tk_d}({mean_conf[s, L, sl_i]:.2f})")
                    txt += f"    L{L:2d}  " + "  ".join(row) + "\n"
                txt += "\n"
            ax.text(0.01, 0.99, txt, va="top", ha="left", family="monospace", fontsize=6)
            ax.set_title("Logit-lens: most-common top token + mean confidence per cell",
                         fontsize=11, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            # Heatmap of mean confidence per (step, layer) for each sublayer
            fig, axes = plt.subplots(1, len(SUBL), figsize=(5 * len(SUBL), 5))
            if len(SUBL) == 1: axes = [axes]
            for sl_i, sl in enumerate(SUBL):
                im = axes[sl_i].imshow(mean_conf[:, :, sl_i], aspect="auto",
                                       origin="lower", cmap="viridis",
                                       vmin=0, vmax=max(0.3, float(mean_conf.max())))
                axes[sl_i].set_xticks(range(ll["N_LAYERS"]))
                axes[sl_i].set_yticks(range(ll["N_LAT"]))
                axes[sl_i].set_yticklabels([str(i+1) for i in range(ll["N_LAT"])])
                axes[sl_i].set_xlabel("layer"); axes[sl_i].set_ylabel("latent step")
                axes[sl_i].set_title(f"{sl}: top-1 confidence", fontsize=10, fontweight="bold")
                for s in range(ll["N_LAT"]):
                    for L in range(ll["N_LAYERS"]):
                        v = mean_conf[s, L, sl_i]
                        if v > 0.05:
                            axes[sl_i].text(L, s, f"{v:.2f}", ha="center", va="center",
                                            fontsize=6,
                                            color="white" if v < 0.3 else "black")
                fig.colorbar(im, ax=axes[sl_i], fraction=0.04, label="confidence")
            fig.suptitle("Logit-lens top-1 confidence per (step, layer, sublayer)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
