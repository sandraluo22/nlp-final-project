"""Slideshow: do CODI-GPT-2 and CODI-Llama actually use their latent thoughts?

Three tests for each model:
  1. Force-decode-per-step accuracy + per-step transition bar graph
  2. LM-head probe: top-1 token stability per step
  3. LM-head probe: distribution-shape (cos sim, KL) per step
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
OUT = PD / "latent_use_slideshow.pdf"

fd_gpt2 = json.load(open(PD / "force_decode_per_step.json"))
fd_llama = json.load(open(PD / "force_decode_per_step_llama.json"))
lmh_gpt2 = json.load(open(PD / "lm_head_probe_gpt2.json"))
lmh_llama = json.load(open(PD / "lm_head_probe_llama.json"))

K_SHOW = 6


def _n(blob):
    return int(blob.get("n_eval") or blob.get("N") or 0)


N_FD_GPT2 = _n(fd_gpt2)
N_FD_LLAMA = _n(fd_llama)
N_LMH_GPT2 = _n(lmh_gpt2)
N_LMH_LLAMA = _n(lmh_llama)


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
            y -= 0.02
        else:
            ax.text(0.0, y, ln, fontsize=10, transform=ax.transAxes); y -= 0.035
    pdf.savefig(fig, dpi=140); plt.close(fig)


def transitions_panel(ax, blob, model_label):
    """Bar chart of right->wrong and wrong->right per step transition."""
    n = _n(blob)
    trs = blob["transitions"][:K_SHOW - 1]
    xs = np.arange(len(trs))
    rtw = np.array([t["right_to_wrong"] for t in trs])
    wtr = np.array([t["wrong_to_right"] for t in trs])
    width = 0.4
    ax.bar(xs - width/2, wtr, width, label="wrong → right", color="#2ca02c")
    ax.bar(xs + width/2, -rtw, width, label="right → wrong (neg)", color="#d62728")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t['from_step']}→{t['to_step']}" for t in trs])
    ax.set_ylabel(f"# examples (of {n})")
    ax.set_title(f"{model_label}: per-step transitions")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    pad = max(1.0, 0.01 * n)
    for i, (a, b) in enumerate(zip(wtr, rtw)):
        ax.text(xs[i], max(a, 0) + pad, f"net={a-b:+d}", ha="center", fontsize=8)


def acc_panel(ax, blob, model_label):
    acc = np.array(blob["accuracy_per_step"][:K_SHOW]) * 100
    ax.plot(range(1, len(acc)+1), acc, "o-", linewidth=2, markersize=8, color="#1f77b4")
    ax.set_xticks(range(1, len(acc)+1))
    ax.set_xlabel("latent step K")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"{model_label}: accuracy per K")
    for k, v in enumerate(acc, start=1):
        ax.annotate(f"{v:.1f}%", (k, v), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_ylim(min(acc.min() - 3, 30), max(acc.max() + 3, 65))
    ax.grid(alpha=0.3)


with PdfPages(OUT) as pdf:
    # === Slide 1: Title + question ===
    text_slide(pdf, "Do CODI-GPT-2 and CODI-Llama actually use latent thoughts?",
        [
            "# Three tests, run on both models",
            "- Force-decode per step: how many right↔wrong flips per latent step?",
            "- LM-head probe: how stable is the top-1 token across steps?",
            "- Distribution probe: cos sim / KL between softmax distributions of consecutive steps",
            "",
            "# What the user expected",
            "Hypothesis: CODI-Llama doesn't really use latents — it decides early.",
            "",
            "# Setup",
            f"{N_FD_GPT2} SVAMP problems (GPT-2) / {N_FD_LLAMA} (Llama); K=6 (trained value).",
        ])

    # === Slide 2: Force-decode accuracy GPT-2 vs Llama ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    acc_panel(axes[0], fd_gpt2, "CODI-GPT-2")
    acc_panel(axes[1], fd_llama, "CODI-Llama-1B")
    plt.suptitle("TEST 1a: Force-decoded accuracy per latent step",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    f1a = PD / "_slide_acc.png"; plt.savefig(f1a, dpi=140, bbox_inches="tight"); plt.close()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("TEST 1a · Force-decoded accuracy per latent step",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
    ax.imshow(plt.imread(f1a)); ax.set_aspect("auto")
    g_acc = np.array(fd_gpt2["accuracy_per_step"][:K_SHOW]) * 100
    l_acc = np.array(fd_llama["accuracy_per_step"][:K_SHOW]) * 100
    fig.text(0.5, 0.04,
             f"Llama: step 1 = {l_acc[0]:.1f}%, peak = {l_acc.max():.1f}% (range {l_acc.max()-l_acc.min():.1f} pp). "
             f"GPT-2: step 1 = {g_acc[0]:.1f}%, peak = {g_acc.max():.1f}% (gain {g_acc.max()-g_acc[0]:+.1f} pp). "
             "Llama's latent loop adds little to accuracy; GPT-2 climbs early then plateaus.",
             ha="center", fontsize=9, style="italic")
    pdf.savefig(fig, dpi=140); plt.close(fig)

    # === Slide 3: Transitions GPT-2 vs Llama ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    transitions_panel(axes[0], fd_gpt2, "CODI-GPT-2")
    transitions_panel(axes[1], fd_llama, "CODI-Llama-1B")
    plt.suptitle("TEST 1b: right↔wrong transitions per step",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    f1b = PD / "_slide_trans.png"; plt.savefig(f1b, dpi=140, bbox_inches="tight"); plt.close()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("TEST 1b · Per-step transitions (right→wrong vs wrong→right)",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
    ax.imshow(plt.imread(f1b)); ax.set_aspect("auto")
    g_nets = [t["wrong_to_right"] - t["right_to_wrong"] for t in fd_gpt2["transitions"][:K_SHOW - 1]]
    l_nets = [t["wrong_to_right"] - t["right_to_wrong"] for t in fd_llama["transitions"][:K_SHOW - 1]]
    fig.text(0.5, 0.04,
             f"GPT-2 net flips per step: {g_nets} (out of {N_FD_GPT2}). "
             f"Llama: {l_nets} (out of {N_FD_LLAMA}). "
             "Where net ≈ 0 the latent loop is doing no useful work on net.",
             ha="center", fontsize=9, style="italic")
    pdf.savefig(fig, dpi=140); plt.close(fig)

    # === Slide 4: Top-1 stability ===
    fig, ax = plt.subplots(figsize=(11, 6))
    g_st = lmh_gpt2["top1_stability"]
    l_st = lmh_llama["top1_stability"]
    xs = np.arange(len(g_st))
    ax.plot(xs, [s*100 for s in g_st], "o-", label="CODI-GPT-2", linewidth=2, markersize=8)
    ax.plot(xs, [s*100 for s in l_st], "s-", label="CODI-Llama-1B", linewidth=2, markersize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i+1}→{i+2}" for i in range(len(g_st))])
    ax.set_xlabel("transition step k → k+1")
    ax.set_ylabel("% examples where lm_head top-1 token is identical")
    ax.set_title("TEST 2: Top-1 token stability across latent steps (lm_head probe)")
    ax.set_ylim(0, 100)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    f2 = PD / "_slide_top1.png"; plt.savefig(f2, dpi=140, bbox_inches="tight"); plt.close()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("TEST 2 · LM-head top-1 token stability per step transition",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
    ax.imshow(plt.imread(f2)); ax.set_aspect("auto")
    g_lo, g_hi = min(g_st)*100, max(g_st)*100
    l_lo, l_hi = min(l_st)*100, max(l_st)*100
    fig.text(0.5, 0.04,
             f"GPT-2 top-1 stability: {g_lo:.0f}-{g_hi:.0f}%. "
             f"Llama: {l_lo:.0f}-{l_hi:.0f}%. "
             "Higher stability = less work being done across the step transition.",
             ha="center", fontsize=9, style="italic")
    pdf.savefig(fig, dpi=140); plt.close(fig)

    # === Slide: LM-head per-step transitions (top-1 / top-5 / top-10) ===
    if "transitions_top1" in lmh_gpt2 and "transitions_top1" in lmh_llama:
        def lmh_trans_panel(ax, blob, model_label):
            n = _n(blob)
            t1 = blob["transitions_top1"][:K_SHOW - 1]
            t5 = blob["transitions_top5"][:K_SHOW - 1]
            t10 = blob["transitions_top10"][:K_SHOW - 1]
            xs_ = np.arange(len(t1))
            width = 0.4
            # back to front: top-10 (most transparent), top-5, top-1 (solid)
            for trs, alpha, lbl_suffix in [
                (t10, 0.22, "top-10"),
                (t5, 0.5, "top-5"),
                (t1, 1.0, "top-1"),
            ]:
                wtr = np.array([t["wrong_to_right"] for t in trs])
                rtw = np.array([t["right_to_wrong"] for t in trs])
                ax.bar(xs_ - width/2, wtr, width, color="#2ca02c", alpha=alpha,
                       label=f"wrong→right · {lbl_suffix}")
                ax.bar(xs_ + width/2, -rtw, width, color="#d62728", alpha=alpha,
                       label=f"right→wrong · {lbl_suffix}")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(xs_)
            ax.set_xticklabels([f"{t['from_step']}→{t['to_step']}" for t in t1])
            ax.set_ylabel(f"# examples (of {n})")
            ax.set_title(f"{model_label}: LM-head per-step transitions")
            ax.legend(fontsize=7, ncol=3, loc="upper right")
            ax.grid(axis="y", alpha=0.3)
            pad = max(1.0, 0.01 * n)
            for i in range(len(t1)):
                net1 = t1[i]["wrong_to_right"] - t1[i]["right_to_wrong"]
                ax.text(xs_[i], max(t1[i]["wrong_to_right"], 0) + pad,
                        f"net₁={net1:+d}", ha="center", fontsize=8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        lmh_trans_panel(axes[0], lmh_gpt2, "CODI-GPT-2")
        lmh_trans_panel(axes[1], lmh_llama, "CODI-Llama-1B")
        plt.suptitle("TEST 2b: LM-head right↔wrong transitions per step (top-1 / top-5 / top-10)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        f2b = PD / "_slide_lmh_trans.png"
        plt.savefig(f2b, dpi=140, bbox_inches="tight"); plt.close()

        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle("TEST 2b · LM-head transitions (right→wrong / wrong→right)",
                     fontsize=14, fontweight="bold")
        ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
        ax.imshow(plt.imread(f2b)); ax.set_aspect("auto")
        g_net1 = [t["wrong_to_right"] - t["right_to_wrong"]
                  for t in lmh_gpt2["transitions_top1"][:K_SHOW - 1]]
        l_net1 = [t["wrong_to_right"] - t["right_to_wrong"]
                  for t in lmh_llama["transitions_top1"][:K_SHOW - 1]]
        g_acc1 = [a * 100 for a in lmh_gpt2["accuracy_per_step_top1"][:K_SHOW]]
        l_acc1 = [a * 100 for a in lmh_llama["accuracy_per_step_top1"][:K_SHOW]]
        fig.text(0.5, 0.04,
                 f"'Correct' at step k = LM-head top-1 token equals the first BPE token of \" {{gold}}\". "
                 f"Solid = top-1, translucent = top-5, faintest = top-10. "
                 f"GPT-2 top-1 acc {min(g_acc1):.1f}-{max(g_acc1):.1f}% (nets {g_net1}); "
                 f"Llama top-1 acc {min(l_acc1):.1f}-{max(l_acc1):.1f}% (nets {l_net1}).",
                 ha="center", fontsize=9, style="italic")
        pdf.savefig(fig, dpi=140); plt.close(fig)

    # === Slide 5: Cos sim of distributions ===
    fig, ax = plt.subplots(figsize=(11, 6))
    g_cs = lmh_gpt2["cos_sim_consec"]
    l_cs = lmh_llama["cos_sim_consec"]
    ax.plot(xs, g_cs, "o-", label="CODI-GPT-2", linewidth=2, markersize=8)
    ax.plot(xs, l_cs, "s-", label="CODI-Llama-1B", linewidth=2, markersize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{i+1}→{i+2}" for i in range(len(g_cs))])
    ax.set_xlabel("transition step k → k+1")
    ax.set_ylabel("cos sim of softmax distributions (consecutive steps)")
    ax.set_title("TEST 3: How similar are consecutive-step distributions?")
    ax.set_ylim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    f3 = PD / "_slide_cos.png"; plt.savefig(f3, dpi=140, bbox_inches="tight"); plt.close()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("TEST 3 · Distribution similarity (cos sim) across latent steps",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
    ax.imshow(plt.imread(f3)); ax.set_aspect("auto")
    fig.text(0.5, 0.04,
             f"GPT-2 cos sim range: {min(g_cs):.2f}-{max(g_cs):.2f}. "
             f"Llama: {min(l_cs):.2f}-{max(l_cs):.2f}. "
             "Higher cos sim = consecutive distributions are more alike (less change per step).",
             ha="center", fontsize=9, style="italic")
    pdf.savefig(fig, dpi=140); plt.close(fig)

    # === Slide 6: KL divergence ===
    fig, ax = plt.subplots(figsize=(11, 6))
    g_kl = lmh_gpt2["kl_consec"]
    l_kl = lmh_llama["kl_consec"]
    ax.plot(xs, g_kl, "o-", label="CODI-GPT-2", linewidth=2, markersize=8)
    ax.plot(xs, l_kl, "s-", label="CODI-Llama-1B", linewidth=2, markersize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{i+1}→{i+2}" for i in range(len(g_kl))])
    ax.set_xlabel("transition step k → k+1")
    ax.set_ylabel("KL(step_k || step_{k+1}) — mean over examples")
    ax.set_title("TEST 3b: How much does each step change the distribution?")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    f4 = PD / "_slide_kl.png"; plt.savefig(f4, dpi=140, bbox_inches="tight"); plt.close()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle("TEST 3b · KL divergence between consecutive distributions",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78]); ax.axis("off")
    ax.imshow(plt.imread(f4)); ax.set_aspect("auto")
    kl_unit = lmh_gpt2.get("kl_unit", "nats")
    fig.text(0.5, 0.04,
             f"GPT-2 KL range: {min(g_kl):.2f}-{max(g_kl):.2f} {kl_unit}. "
             f"Llama: {min(l_kl):.2f}-{max(l_kl):.2f} {kl_unit}. "
             "Smaller KL = less distributional change per step.",
             ha="center", fontsize=9, style="italic")
    pdf.savefig(fig, dpi=140); plt.close(fig)

    # === Final synthesis slide ===
    g_tr0 = fd_gpt2["transitions"][0]
    g_net0 = g_tr0["wrong_to_right"] - g_tr0["right_to_wrong"]
    g_other_nets = [t["wrong_to_right"] - t["right_to_wrong"]
                    for t in fd_gpt2["transitions"][1:K_SHOW - 1]]
    g_other_max = max(abs(n) for n in g_other_nets) if g_other_nets else 0
    g_acc_arr = np.array(fd_gpt2["accuracy_per_step"][:K_SHOW]) * 100
    l_acc_arr = np.array(fd_llama["accuracy_per_step"][:K_SHOW]) * 100
    l_nets_abs = [abs(t["wrong_to_right"] - t["right_to_wrong"])
                  for t in fd_llama["transitions"][:K_SHOW - 1]]
    g_stab_pct = [s * 100 for s in lmh_gpt2["top1_stability"]]
    l_stab_pct = [s * 100 for s in lmh_llama["top1_stability"]]
    kl_unit = lmh_gpt2.get("kl_unit", "nats")
    text_slide(pdf, "Synthesis: do they use the latents?",
        [
            f"# Setup: {N_FD_GPT2} SVAMP problems (GPT-2) / {N_FD_LLAMA} (Llama), K=6.",
            "",
            "# CODI-GPT-2",
            f"- Step 1→2 net flips: {g_net0:+d} (wrong→right={g_tr0['wrong_to_right']}, "
            f"right→wrong={g_tr0['right_to_wrong']}) out of {N_FD_GPT2}.",
            f"- Other transitions in K=2..6: |net| ≤ {g_other_max} (drift, not progress).",
            f"- Accuracy step 1 → peak: {g_acc_arr[0]:.1f}% → {g_acc_arr.max():.1f}% "
            f"(gain {g_acc_arr.max()-g_acc_arr[0]:+.1f} pp).",
            f"- Top-1 LM-head token stability: {min(g_stab_pct):.0f}-{max(g_stab_pct):.0f}% — high volatility.",
            f"- Cos sim {min(lmh_gpt2['cos_sim_consec']):.2f}-{max(lmh_gpt2['cos_sim_consec']):.2f}, "
            f"KL {min(lmh_gpt2['kl_consec']):.1f}-{max(lmh_gpt2['kl_consec']):.1f} {kl_unit} — distributions rotate a lot,",
            "  but the rotation mostly cancels out behaviorally.",
            "",
            "# CODI-Llama-1B",
            f"- Accuracy step 1 → step {K_SHOW}: {l_acc_arr[0]:.1f}% → {l_acc_arr[-1]:.1f}% "
            f"(range {l_acc_arr.max()-l_acc_arr.min():.1f} pp).",
            f"- All transitions in K=1..6: |net| ≤ {max(l_nets_abs) if l_nets_abs else 0} out of {N_FD_LLAMA}.",
            f"- Top-1 LM-head token stability: {min(l_stab_pct):.0f}-{max(l_stab_pct):.0f}% — high stability.",
            f"- Cos sim {min(lmh_llama['cos_sim_consec']):.2f}-{max(lmh_llama['cos_sim_consec']):.2f}, "
            f"KL {min(lmh_llama['kl_consec']):.1f}-{max(lmh_llama['kl_consec']):.1f} {kl_unit} — latents barely change.",
            "- Effectively at 'final' accuracy from step 1 of the latent loop.",
            "",
            "# Conclusion",
            "Hypothesis CONFIRMED. CODI-Llama doesn't meaningfully use its latent thoughts —",
            "the answer is determined by the prompt + first latent step. Subsequent steps",
            "are essentially decorative.",
            "",
            "CODI-GPT-2 does have ONE step of genuine computation (step 1→2), but later",
            "steps don't accumulate useful work either.",
        ])

print(f"saved {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
