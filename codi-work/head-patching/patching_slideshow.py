"""Combined patching-recovery slideshow: one slide per experiment, plotting
recovery rate vs layer index for each (component) at stage 0 (prompt+bot).

Loads every *recovery*.json under codi-work/head-patching/ and renders a
side-by-side panel grid in a single PDF. Recovery is the fraction of patched
runs (corrupted prompt + clean activation injected at one cell) that flip the
predicted answer to the clean answer.

Each slide:
  Left  panel : per-(component, layer) bar at stage 0 of the experiment
  Right panel : full (stage × layer) heatmap so we can see latent-stage
                contribution if any
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


HEAD = Path(__file__).resolve().parent
OUT_PDF = HEAD / "patching_slideshow.pdf"

# (label, json path, model_tag, corruption_kind)
EXPERIMENTS = [
    ("Llama-1B  ·  A_templates  ·  Mul→Add (operator)",
     HEAD / "A_templates" / "results.json", "llama-1b", "operator-templated"),
    ("Llama-1B  ·  B_svamp_filtered  ·  Mul→Add (operator)",
     HEAD / "B_svamp_filtered" / "results.json", "llama-1b", "operator-svamp"),
    ("CODI-GPT2  ·  B_svamp_filtered  ·  Mul→Add (operator)",
     HEAD / "B_svamp_filtered" / "patching_gpt2_recovery.json", "gpt2", "operator-svamp"),
    ("CODI-GPT2  ·  numeral_pairs_b1_sub  ·  b=1 corruption (Sub)",
     HEAD / "numeral_b1_sub_recovery.json", "gpt2", "numeral-b1-sub"),
    ("CODI-GPT2  ·  numeral_pairs_a1_mul  ·  a=1 corruption (Mul)",
     HEAD / "numeral_a1_mul_recovery.json", "gpt2", "numeral-a1-mul"),
]

COMP_COLORS = {"resid": "#1f77b4", "attn": "#2ca02c", "mlp": "#d62728"}
COMP_MARKERS = {"resid": "o", "attn": "^", "mlp": "s"}


def render_experiment_slide(pdf, label, recovery, n_kept, n_pairs):
    """`recovery` is either a {comp: 2D list} dict or a 2D list (resid-only,
    older schema). Render to one slide either way."""
    if isinstance(recovery, list):
        # Older schema: just residual.
        arr = {"resid": np.array(recovery) * 100}
        components = ["resid"]
    else:
        components = ["resid", "attn", "mlp"]
        arr = {c: np.array(recovery[c]) * 100 for c in components if c in recovery}
    n_stages, n_layers = arr["resid"].shape

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle(label, fontsize=12)

    # --- Left: stage-0 bars over a single axis of (component, layer) cells ---
    # ordered: all resid layers, then all attn, then all mlp.
    ax = axes[0]
    cells, vals, colors = [], [], []
    for c in ["resid", "attn", "mlp"]:
        if c not in arr: continue
        for l in range(n_layers):
            cells.append(f"{c}-L{l:02d}")
            vals.append(arr[c][0, l])
            colors.append(COMP_COLORS[c])
    xs = np.arange(len(cells))
    ax.bar(xs, vals, color=colors, width=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(cells, rotation=90, fontsize=7)
    ax.set_ylabel("clean-flip recovery (%)")
    ax.set_title(
        f"Stage 0 (prompt+bot) recovery per (component, layer) cell  "
        f"(N_kept={n_kept}/{n_pairs})"
    )
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", alpha=0.3)
    # custom legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=COMP_COLORS[c], label=c)
                      for c in ["resid", "attn", "mlp"] if c in arr]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    # --- Right: full (stage × layer) heatmap, max across available components ---
    ax = axes[1]
    grid = np.maximum.reduce([arr[c] for c in arr])
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=100)
    ax.set_xlabel("layer index")
    ax.set_ylabel("stage")
    ax.set_yticks(range(n_stages))
    yt = ["prompt+bot"] + [f"latent {s}" for s in range(1, n_stages)]
    ax.set_yticklabels(yt, fontsize=9)
    ax.set_xticks(range(n_layers))
    ax.set_title("Max recovery across {resid, attn, mlp} per (stage, layer)")
    cb = plt.colorbar(im, ax=ax, label="% recovery")
    fig.tight_layout()
    pdf.savefig(fig, dpi=130)
    plt.close(fig)


def render_summary_slide(pdf, summary_rows):
    """summary_rows: list of (label, max_recovery_resid_stage0, peak_layer, n_kept)."""
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(summary_rows) + 1.2))
    ax.axis("off")
    title = "Summary: peak resid stage-0 recovery per experiment"
    ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=13,
            fontweight="bold", transform=ax.transAxes)
    headers = ["Experiment", "Resid peak (%)", "@ Layer", "MLP-L0 (%)", "N_kept / N_pairs"]
    col_x = [0.02, 0.62, 0.74, 0.83, 0.92]
    y = 0.90
    for label, hx in zip(headers, col_x):
        ax.text(hx, y, label, ha="left" if hx < 0.5 else "right",
                fontsize=10, fontweight="bold", transform=ax.transAxes)
    y -= 0.06
    for row in summary_rows:
        for val, hx in zip(row, col_x):
            ax.text(hx, y, str(val), ha="left" if hx < 0.5 else "right",
                    fontsize=10, transform=ax.transAxes)
        y -= 0.06
    pdf.savefig(fig, dpi=130)
    plt.close(fig)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print(f"writing {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        for label, fp, model, kind in EXPERIMENTS:
            if not fp.exists():
                print(f"  SKIP (missing): {fp}")
                continue
            s = json.load(open(fp))
            print(f"\n=== {label} ===")
            print(f"  N_pairs={s['n_pairs']}  N_kept={s['n_kept_pairs']}  "
                  f"baseline_clean={s['baseline_clean_correct']}  "
                  f"baseline_corr={s['baseline_corr_correct']}")
            render_experiment_slide(
                pdf, label, s["recovery"],
                n_kept=s["n_kept_pairs"], n_pairs=s["n_pairs"],
            )
            rec = s["recovery"]
            if isinstance(rec, list):
                resid0 = np.array(rec)[0] * 100
                mlp0 = np.array([0.0])
            else:
                resid0 = np.array(rec["resid"])[0] * 100
                mlp0 = np.array(rec.get("mlp", [[0.0]]))[0] * 100
            summary_rows.append((
                label[:60],
                f"{resid0.max():.0f}",
                f"L{int(np.argmax(resid0)):02d}",
                f"{mlp0[0]:.0f}" if mlp0.size > 0 else "—",
                f"{s['n_kept_pairs']}/{s['n_pairs']}",
            ))
        render_summary_slide(pdf, summary_rows)
    print(f"\ndone -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
