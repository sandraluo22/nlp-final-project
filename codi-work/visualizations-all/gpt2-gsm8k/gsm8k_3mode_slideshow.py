"""Per-(cf, block, step, layer) 3-mode comparison slideshow for GSM8K.

Reads:
  - correctness-analysis/zero_mean_per_cell_gsm8k.json   (zero/all + mean/resid)
  - ../../experiments/computation_probes/patch_cf_mean_gsm8k.json  (mean/attn+mlp)
  - correctness-analysis/patch_paired_cf_gsm8k.json     (patch/all, paired CF)

For each (CF set, block ∈ {attn, mlp, resid}) plots side-by-side heatmaps over
(latent step, layer) for the three ablation modes:
  - ZERO   : set the block output at the last token to 0
  - MEAN   : set it to the mean across baseline runs of that CF set
  - PATCH  : interchange-patch from a paired CF (followed-source rate)

The heatmap value is "Δ accuracy" relative to that CF set's baseline for
zero/mean (lower = more damaging), and "transfer rate" (P(followed source))
for patch (higher = more carrying).

Also: an overall summary page per CF set comparing zero/mean against patch.

Output: gsm8k_3mode_slideshow.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
CA = PD / "correctness-analysis"
CP = REPO / "experiments" / "computation_probes"

OUT_PDF = PD / "gsm8k_3mode_slideshow.pdf"

BLOCKS = ["attn", "mlp", "resid"]


def load(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


def heatmap(ax, M, title, vmin, vmax, cmap, n_for_pct=None, label_pct=True):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    n_steps, n_layers = M.shape
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_steps)); ax.set_yticklabels([str(s + 1) for s in range(n_steps)])
    ax.set_xlabel("layer"); ax.set_ylabel("latent step")
    ax.set_title(title, fontsize=9, fontweight="bold")
    fmt = "{:+.0f}" if vmin < 0 and label_pct else "{:.0f}"
    for s in range(n_steps):
        for l in range(n_layers):
            v = M[s, l]
            if np.isnan(v): continue
            if label_pct:
                txt = fmt.format(v * 100)
            else:
                txt = f"{v:.2f}"
            color = "white" if abs(v - vmin) < (vmax - vmin) * 0.4 else "black"
            ax.text(l, s, txt, ha="center", va="center", fontsize=6, color=color)
    return im


def cell_matrix_from_zero_mean(conditions, block, mode, n_steps=6, n_layers=12, key="delta_acc"):
    """conditions key format: f'{block}_{mode}_step{k}_L{l}' → value with key."""
    M = np.full((n_steps, n_layers), np.nan)
    for k, c in conditions.items():
        if c.get("block") == block and c.get("mode") == mode:
            s = c["step"] - 1; l = c["layer"]
            M[s, l] = c[key]
    return M


def cell_matrix_from_patch_mean(conditions, block, n_steps=6, n_layers=12, n_examples=1):
    """patch_cf_mean uses 'stepK_L{l}_{block}' keys with 'acc' and 'baseline'.
    Returns delta_acc = acc - baseline."""
    M = np.full((n_steps, n_layers), np.nan)
    for s in range(1, n_steps + 1):
        for l in range(n_layers):
            key = f"step{s}_L{l}_{block}"
            c = conditions.get(key)
            if c is None: continue
            if "delta_acc" in c:
                M[s - 1, l] = c["delta_acc"]
            elif "acc" in c and "baseline" in c:
                M[s - 1, l] = c["acc"] - c["baseline"]
    return M


def cell_matrix_from_patch_paired(conditions, block, metric_key, n_steps=6, n_layers=12, N=1):
    """patch_paired_cf uses 'stepK_L{l}_{block}' keys with counts.
    metric_key in {n_followed_source, n_followed_target, n_other}; we divide by N."""
    M = np.full((n_steps, n_layers), np.nan)
    for s in range(1, n_steps + 1):
        for l in range(n_layers):
            key = f"step{s}_L{l}_{block}"
            c = conditions.get(key)
            if c is None: continue
            M[s - 1, l] = c[metric_key] / N
    return M


def main():
    zm = load(CA / "zero_mean_per_cell_gsm8k.json") or {}
    cm = load(CP / "patch_cf_mean_gsm8k.json") or {"cf_sets": {}}
    pp = load(CA / "patch_paired_cf_gsm8k.json") or {"cf_sets": {}}

    cf_set_names = sorted(set(list(zm.keys()) + list(cm.get("cf_sets", {}).keys())
                              + list(pp.get("cf_sets", {}).keys())))
    # cf_balanced has near-zero model baseline (~0-1%); interventions are
    # uninformative on it. Drop from slideshow.
    cf_set_names = [n for n in cf_set_names if n != "gsm8k_cf_balanced"]

    if not cf_set_names:
        print("no inputs found; aborting")
        return

    print(f"CF sets present: {cf_set_names}")
    print(f"  zero+mean has: {sorted(zm.keys())}")
    print(f"  cf_mean has:   {sorted(cm.get('cf_sets', {}).keys())}")
    print(f"  paired_cf has: {sorted(pp.get('cf_sets', {}).keys())}")

    with PdfPages(OUT_PDF) as pdf:
        # ---- Title page ----
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        body = ("GSM8K — 3-mode per-(block, step, layer) ablation comparison\n\n"
                "For each CF set × block (attn / mlp / resid):\n"
                "  ZERO  : replace block output at last token with 0\n"
                "  MEAN  : replace with mean over CF baseline runs\n"
                "  PATCH : interchange-patch from a paired CF (followed-source rate)\n\n"
                "Color convention everywhere: RED = cell matters; WHITE = no effect;\n"
                "                            BLUE = ablation actually helped (rare/noise).\n"
                "Heatmaps:\n"
                "  zero/mean → 'damage' = -Δacc (pp). Red = ablation broke the model.\n"
                "  patch     → transfer rate P(followed source). Red = cell carries the\n"
                "              source's answer over to the target.\n\n"
                f"CF sets present:\n")
        for n in cf_set_names:
            zm_ok = "✓" if n in zm else "−"
            cm_ok = "✓" if n in cm.get("cf_sets", {}) else "−"
            pp_ok = "✓" if n in pp.get("cf_sets", {}) else "−"
            body += f"  {n:<28s}  zero/mean {zm_ok}   mean(attn,mlp) {cm_ok}   patch {pp_ok}\n"
        ax.text(0.04, 0.96, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("3-mode ablation slideshow", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ---- Per CF, per block: 3 heatmaps in a row ----
        for cf in cf_set_names:
            zm_cf = zm.get(cf)
            cm_cf = cm.get("cf_sets", {}).get(cf)
            pp_cf = pp.get("cf_sets", {}).get(cf)
            N_zm = zm_cf["N"] if zm_cf else None
            N_cm = cm_cf["N"] if cm_cf else None
            N_pp = pp_cf["N"] if pp_cf else None
            base_zm = zm_cf["baseline_accuracy"] if zm_cf else None
            base_cm = cm_cf.get("baseline_accuracy") if cm_cf else None
            base_pp = pp_cf.get("baseline_accuracy") if pp_cf else None

            for block in BLOCKS:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

                # Color convention: RED = "cell matters", BLUE/white = "doesn't matter".
                # ZERO/MEAN show damage as a magnitude (we negate Δacc so a 50pp drop
                # plots as +50 → red). The label still says "Δacc (pp)" but with the
                # sign flipped to "damage" so red consistently means "matters".

                # Column 1: ZERO — plot damage = -Δacc (so red = breaks model)
                if zm_cf:
                    M_zero_dacc = cell_matrix_from_zero_mean(zm_cf["conditions"], block, "zero")
                else:
                    M_zero_dacc = np.full((6, 12), np.nan)
                M_zero = -M_zero_dacc   # damage magnitude
                vmax_d = max(0.05, float(np.nanmax(np.abs(M_zero))) + 0.01) if not np.all(np.isnan(M_zero)) else 0.5
                im0 = heatmap(axes[0], M_zero,
                              f"ZERO damage (=-Δacc) — {block}\n(base {('?' if base_zm is None else f'{base_zm*100:.1f}%')})",
                              -vmax_d, vmax_d, "coolwarm")
                fig.colorbar(im0, ax=axes[0], fraction=0.05, label="damage (pp) — red = breaks model")

                # Column 2: MEAN — same convention
                if block == "resid":
                    if zm_cf:
                        M_mean_dacc = cell_matrix_from_zero_mean(zm_cf["conditions"], "resid", "mean")
                    else:
                        M_mean_dacc = np.full((6, 12), np.nan)
                    base_for_mean = base_zm
                else:
                    if cm_cf:
                        M_mean_dacc = cell_matrix_from_patch_mean(cm_cf["conditions"], block)
                    else:
                        M_mean_dacc = np.full((6, 12), np.nan)
                    base_for_mean = base_cm
                M_mean = -M_mean_dacc
                vmax_dm = max(0.05, float(np.nanmax(np.abs(M_mean))) + 0.01) if not np.all(np.isnan(M_mean)) else 0.5
                im1 = heatmap(axes[1], M_mean,
                              f"MEAN damage (=-Δacc) — {block}\n(base {('?' if base_for_mean is None else f'{base_for_mean*100:.1f}%')})",
                              -vmax_dm, vmax_dm, "coolwarm")
                fig.colorbar(im1, ax=axes[1], fraction=0.05, label="damage (pp) — red = breaks model")

                # Column 3: PATCH (transfer rate) — high transfer = matters = red
                if pp_cf:
                    M_patch = cell_matrix_from_patch_paired(pp_cf["conditions"], block,
                                                            "n_followed_source", N=N_pp)
                else:
                    M_patch = np.full((6, 12), np.nan)
                vmax_patch = max(0.4, float(np.nanmax(M_patch)) + 0.05) if not np.all(np.isnan(M_patch)) else 0.4
                im2 = heatmap(axes[2], M_patch,
                              f"PATCH transfer — {block}\n(P(followed source))",
                              -vmax_patch, vmax_patch, "coolwarm")
                fig.colorbar(im2, ax=axes[2], fraction=0.05, label="P(source) — red = carries answer")

                fig.suptitle(f"{cf} / {block}   "
                             f"N(zero/mean(resid))={N_zm} N(mean(attn,mlp))={N_cm} N(patch)={N_pp}",
                             fontsize=11, fontweight="bold")
                fig.tight_layout(rect=(0, 0, 1, 0.93))
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            # Per-CF summary: best cells under each mode
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            ax = axes[0]
            # for each block, find argmin Δacc under zero
            rows = []
            for block in BLOCKS:
                if zm_cf:
                    M = cell_matrix_from_zero_mean(zm_cf["conditions"], block, "zero")
                    if not np.all(np.isnan(M)):
                        i = int(np.nanargmin(M.flatten())); s = i // 12; l = i % 12
                        rows.append((block, "zero", s+1, l, float(M[s, l]) * 100))
                if block == "resid" and zm_cf:
                    M = cell_matrix_from_zero_mean(zm_cf["conditions"], "resid", "mean")
                    if not np.all(np.isnan(M)):
                        i = int(np.nanargmin(M.flatten())); s = i // 12; l = i % 12
                        rows.append((block, "mean", s+1, l, float(M[s, l]) * 100))
                if block != "resid" and cm_cf:
                    M = cell_matrix_from_patch_mean(cm_cf["conditions"], block)
                    if not np.all(np.isnan(M)):
                        i = int(np.nanargmin(M.flatten())); s = i // 12; l = i % 12
                        rows.append((block, "mean", s+1, l, float(M[s, l]) * 100))
                if pp_cf:
                    M = cell_matrix_from_patch_paired(pp_cf["conditions"], block,
                                                       "n_followed_source", N=N_pp)
                    if not np.all(np.isnan(M)):
                        i = int(np.nanargmax(M.flatten())); s = i // 12; l = i % 12
                        rows.append((block, "patch", s+1, l, float(M[s, l]) * 100))
            ax.axis("off")
            txt = "Most-damaging / most-transferring cell per (block, mode):\n\n"
            txt += f"{'block':<7s} {'mode':<6s} {'cell':<14s} {'value':<12s}\n"
            for r in rows:
                block, mode, st, ly, val = r
                if mode == "patch":
                    txt += f"  {block:<6s} {mode:<6s} step{st} L{ly:<2d}    {val:+.1f}% transfer\n"
                else:
                    txt += f"  {block:<6s} {mode:<6s} step{st} L{ly:<2d}    {val:+.1f}pp Δacc\n"
            ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10)
            ax.set_title(f"{cf} — best cells", fontsize=11, fontweight="bold")
            axes[1].axis("off")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
