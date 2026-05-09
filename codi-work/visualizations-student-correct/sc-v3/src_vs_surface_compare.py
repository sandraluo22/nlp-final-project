"""Centerpiece sc-v3 analysis: does activation track surface text or source structure?

For each (layer, latent_step):
  1. Fit LDA on original SVAMP student activations supervised on the 4-operator
     label (Addition / Subtraction / Multiplication / Common-Division).
  2. Predict the operator class of each cf_gpt_transformed row using that LDA
     classifier.
  3. Tally the prediction by cf_gpt's surface label (Add/Sub) and source
     label (Mul/Div):

       For cf_gpt rows that are surface=Add, was-Mul:
         - prediction == "Addition"      → activation tracks SURFACE text
         - prediction == "Multiplication" → activation retains SOURCE structure
         - other                          → mixed / other

This directly tests whether the operator representation is anchored to the
visible operator words or to the underlying scenario structure.

Outputs:
  - PDF slideshow: per-layer line plots of "tracks-surface" vs "tracks-source"
    rate, plus a confusion table at the peak layer.
  - JSON stats with per-(layer, step) prediction counts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


REPO = Path(__file__).resolve().parent.parent.parent
ORIG_ACTS = REPO / "inference" / "runs" / "svamp_student" / "activations.pt"
ORIG_RESULTS = REPO / "inference" / "runs" / "svamp_student" / "results.json"
CF_ACTS = REPO / "inference" / "runs" / "cf_gpt_student" / "activations.pt"
CF_RESULTS = REPO / "inference" / "runs" / "cf_gpt_student" / "results.json"
CF_DATA = REPO.parent / "cf-datasets" / "cf_gpt_transformed.json"
OUT_PDF = REPO / "visualizations-student-correct" / "sc-v3" / "src_vs_surface_compare.pdf"
OUT_STATS = REPO / "visualizations-student-correct" / "sc-v3" / "src_vs_surface_stats.json"

OPS_FULL = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
SHORT = {"Addition": "Add", "Subtraction": "Sub",
         "Multiplication": "Mul", "Common-Division": "Div"}
PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}


def load_orig():
    print(f"loading orig SVAMP from {ORIG_ACTS}", flush=True)
    a = torch.load(ORIG_ACTS, map_location="cpu", weights_only=True).float().numpy()
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array(
        [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    )
    sr = json.load(open(ORIG_RESULTS))
    correct_mask = np.array([r["correct"] for r in sr], dtype=bool)
    a = a[correct_mask]
    types = types[correct_mask]
    print(f"  filtered to student-correct: {a.shape[0]} / {len(sr)}")
    return a, types


def load_cf():
    """CF dataset = svamp Mul+Div (originals) + gpt Add+Sub (transformed),
    each filtered to student-correct on its respective student run."""
    from _combined_loader import load_combined_cf
    a, m = load_combined_cf()
    print(f"  combined CF: N={a.shape[0]}", flush=True)
    return a, m["type"], m["src_type"], m["origin"]


def main():
    orig_acts, orig_types = load_orig()
    cf_acts, cf_surface, cf_source, cf_origin = load_cf()
    L = orig_acts.shape[2]
    S = orig_acts.shape[1]
    print(f"\norig: {orig_acts.shape}  cf: {cf_acts.shape}")
    print(f"  cf surface counts: {dict(Counter(cf_surface))}")
    print(f"  cf origin counts:  {dict(Counter(cf_origin))}")
    # Restrict the surface-vs-source analysis to the transformed rows where
    # surface != source (the orig Mul/Div rows have surface == source by
    # construction and would inflate both rates).
    transformed_mask = cf_origin == "gpt_transformed"
    print(f"  rows where surface != source (transformed): {int(transformed_mask.sum())}")

    # Per (surface × source) cell, tally how many predictions go to each of the
    # 4 original-operator classes. That gives us 4 cells × 4 prediction
    # buckets = 16 numbers per (layer, step).
    pred_counts = {}  # (layer, step) -> per-cell counts (transformed rows only)
    orig_acc_per_layer = np.zeros((L, S))  # how well LDA classifies orig Mul/Div
    for layer in range(L):
        for step in range(S):
            X_orig = orig_acts[:, step, layer, :]
            X_cf = cf_acts[:, step, layer, :]
            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(X_orig, orig_types)
            preds = lda.predict(X_cf)
            # Restrict surface-vs-source analysis to transformed rows.
            cnt = Counter()
            for surf, srcv, predv, is_t in zip(cf_surface, cf_source, preds, transformed_mask):
                if is_t:
                    cnt[(surf, srcv, predv)] += 1
            pred_counts[(layer, step)] = dict(cnt)
            # Sanity: classification accuracy on the orig Mul/Div CF rows.
            orig_mask = ~transformed_mask
            orig_acc_per_layer[layer, step] = (
                preds[orig_mask] == cf_surface[orig_mask]
            ).mean()
        print(f"  layer {layer:>2d}/{L-1} done", flush=True)

    # Aggregate per layer (mean across latent steps) into "tracks surface" /
    # "tracks source" rates per cell.
    # Now the CF has all 4 surface ops; src is either same as surface (for
    # the unchanged Mul/Div rows) or Mul/Div (for transformed Add/Sub rows).
    # Cells where src == surface are trivial — those rows IS the original
    # SVAMP, so prediction == surface == src. The interesting cells are the
    # transformed Add/Sub.
    cells = []
    for surf in ("Addition", "Subtraction", "Multiplication", "Common-Division"):
        for src in ("Multiplication", "Common-Division", "Addition", "Subtraction"):
            cells.append((surf, src))

    def rate(layer, want_surface_match: bool):
        """Across all 6 latent steps, fraction of cf rows whose prediction
        matches the surface op (if want_surface_match=True) or the source op."""
        num = 0
        den = 0
        for step in range(S):
            for surf, src, pred in pred_counts[(layer, step)]:
                c = pred_counts[(layer, step)][(surf, src, pred)]
                if want_surface_match:
                    num += c if pred == surf else 0
                else:
                    num += c if pred == src else 0
                den += c
        return num / max(den, 1)

    surface_rate = np.array([rate(L_, True) for L_ in range(L)])
    source_rate = np.array([rate(L_, False) for L_ in range(L)])
    other_rate = 1.0 - surface_rate - source_rate

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUT_PDF) as pdf:
        # --- Slide 1: Surface vs Source line plot ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(L), surface_rate * 100, marker="o", label="prediction == surface op (tracks visible Add/Sub)", color="#1f77b4")
        ax.plot(range(L), source_rate * 100, marker="s", label="prediction == source op (tracks original Mul/Div)", color="#d62728")
        ax.plot(range(L), other_rate * 100, marker="^", label="prediction == other op", color="#999999")
        ax.set_xticks(range(L))
        ax.set_xlabel("layer", fontsize=11)
        ax.set_ylabel("fraction of cf_gpt rows (%)", fontsize=11)
        ax.set_title("cf_gpt rows projected onto original-SVAMP-trained 4-op LDA: where does the prediction land?",
                     fontsize=12)
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
        fig.tight_layout()
        pdf.savefig(fig, dpi=140)
        plt.close(fig)

        # --- Slide 2: per-cell breakdown stacked bar at peak layer ---
        peak_layer = int(np.argmax(surface_rate + source_rate))  # most decisive layer
        fig, ax = plt.subplots(figsize=(12, 6))
        cell_labels = []
        cell_data = {op: [] for op in OPS_FULL}
        for surf, src in cells:
            label = f"{SHORT[surf]} (was {SHORT[src]})"
            cell_labels.append(label)
            n_cell = sum(c for (s, sr, p), c in pred_counts[(peak_layer, 0)].items()
                         if s == surf and sr == src)
            # Aggregate across steps for this cell
            n_total = 0
            tally = Counter()
            for step in range(S):
                for (s, sr, p), c in pred_counts[(peak_layer, step)].items():
                    if s == surf and sr == src:
                        tally[p] += c
                        n_total += c
            for op in OPS_FULL:
                cell_data[op].append(tally.get(op, 0) / max(n_total, 1) * 100)

        bottom = np.zeros(len(cell_labels))
        x = np.arange(len(cell_labels))
        for op in OPS_FULL:
            heights = np.array(cell_data[op])
            ax.bar(x, heights, bottom=bottom, label=f"predicted: {op}",
                   color=PROBLEM_TYPE_COLORS[op], width=0.7)
            bottom += heights
        ax.set_xticks(x)
        ax.set_xticklabels(cell_labels, fontsize=11)
        ax.set_ylabel("% of cf_gpt rows in this cell", fontsize=11)
        ax.set_title(f"Layer {peak_layer}: cf_gpt rows by (surface, source) cell — predicted operator distribution",
                     fontsize=12)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140)
        plt.close(fig)

    # Save numeric stats
    stats = {
        "surface_rate_per_layer": surface_rate.tolist(),
        "source_rate_per_layer": source_rate.tolist(),
        "other_rate_per_layer": other_rate.tolist(),
        "peak_layer_for_decision": int(peak_layer),
        "cell_counts": {
            f"layer{L_}_step{s+1}": {
                f"{surf}|{src}|pred={p}": c
                for (surf, src, p), c in pred_counts[(L_, s)].items()
            }
            for L_ in range(L) for s in range(S)
        },
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))

    print("\n=== Surface vs Source rate by layer ===")
    print(f"{'layer':>5s}  {'surface%':>9s}  {'source%':>9s}  {'other%':>9s}")
    for L_ in range(L):
        print(f"  {L_:>3d}  {surface_rate[L_]*100:>8.1f}%  "
              f"{source_rate[L_]*100:>8.1f}%  {other_rate[L_]*100:>8.1f}%")
    print(f"\nsaved -> {OUT_PDF}")
    print(f"saved -> {OUT_STATS}")


if __name__ == "__main__":
    main()
