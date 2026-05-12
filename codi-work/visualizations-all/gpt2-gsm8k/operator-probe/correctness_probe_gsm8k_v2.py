"""Correctness probe on CODI-GPT-2's GSM8K activations.

For each (latent_step, layer) cell, train a binary classifier predicting:
  - 'will the model's FINAL (step-6) emit equal the gold answer?'
  - 'will the model's force-decode at THIS step equal the gold answer?'

The first variant asks 'can we see correctness coming from this cell'.
The second asks 'is this cell's state itself correct at this step'.

We use:
  - gsm8k_latent_acts.pt        (1319 × 6 × 13 × 768 — already local)
  - force_decode_per_step_gsm8k.json (per-example per-step force-decoded emit)

Trained with RidgeClassifier (class-balanced) + StandardScaler, 80/20
stratified split per cell.

Outputs:
  correctness_probe_gsm8k_v2.{json,pdf}
"""
from __future__ import annotations

import json, re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PD = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
FD_JSON = REPO / "experiments" / "computation_probes" / "force_decode_per_step_gsm8k.json"

OUT_JSON = PD / "correctness_probe_gsm8k_v2.json"
OUT_PDF = PD / "correctness_probe_gsm8k_v2.pdf"


def emit_final(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def fit_probe(X, y):
    if len(set(y)) < 2 or len(y) < 30:
        return None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = float((pred == yte).mean())
    f1 = float(f1_score(yte, pred, labels=[False, True], average="macro",
                        zero_division=0))
    # Class-balanced baseline: predicting majority class
    maj = max(Counter(yte).values()) / len(yte)
    return {"acc": acc, "f1": f1, "n_test": len(yte), "maj_baseline": float(maj),
            "y_dist": dict(Counter(yte))}


def main():
    print("loading acts + force_decode", flush=True)
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = acts.shape
    print(f"  acts shape: {acts.shape}")
    fd = json.load(open(FD_JSON))
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]

    # Build per-example labels
    rows = {r["idx"]: r for r in fd["rows"]}
    final_correct = np.full(N, False)
    per_step_correct = np.full((N, S), False)
    for i in range(N):
        r = rows.get(i)
        if r is None: continue
        ex = ds[i]
        gm = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if gm is None: continue
        gold = float(gm.group(1))
        for k in range(S):
            v = emit_final(r["step_emits"][k])
            if v is not None and abs(v - gold) < 1e-3:
                per_step_correct[i, k] = True
        final_correct[i] = per_step_correct[i, -1]
    print(f"  final-correct: {final_correct.sum()}/{N} ({final_correct.mean()*100:.1f}%)")
    for k in range(S):
        print(f"  step{k+1}-correct: {per_step_correct[:, k].sum()}/{N} "
              f"({per_step_correct[:, k].mean()*100:.1f}%)")

    # === Probe 1: per-(step, layer), predict FINAL correctness ===
    print("\n[probe 1] predicting FINAL correctness from per-(step, layer) acts", flush=True)
    grid_final = np.full((S, L), np.nan)
    f1_final = np.full((S, L), np.nan)
    for k in range(S):
        for l in range(L):
            X = acts[:, k, l, :]
            r = fit_probe(X, final_correct)
            if r is not None:
                grid_final[k, l] = r["acc"]
                f1_final[k, l] = r["f1"]
        print(f"  step{k+1} done", flush=True)

    # === Probe 2: per-(step, layer), predict THAT-STEP correctness ===
    print("\n[probe 2] predicting THIS-STEP force-decode correctness", flush=True)
    grid_self = np.full((S, L), np.nan)
    f1_self = np.full((S, L), np.nan)
    for k in range(S):
        y = per_step_correct[:, k]
        for l in range(L):
            X = acts[:, k, l, :]
            r = fit_probe(X, y)
            if r is not None:
                grid_self[k, l] = r["acc"]
                f1_self[k, l] = r["f1"]
        print(f"  step{k+1} done", flush=True)

    # Class baseline (majority-class)
    maj_baseline_final = max(final_correct.mean(), 1 - final_correct.mean()) * 100
    maj_baseline_per_step = [max(per_step_correct[:, k].mean(),
                                  1 - per_step_correct[:, k].mean()) * 100
                              for k in range(S)]

    # Best cells
    def best_cell(G):
        if np.all(np.isnan(G)): return None
        i = int(np.nanargmax(G.flatten())); s, l = i // L, i % L
        return {"step": s+1, "layer": l, "acc": float(G[s, l])}

    out = {
        "N": int(N), "S": int(S), "L": int(L),
        "final_correct_pct": float(final_correct.mean() * 100),
        "per_step_correct_pct": [float(per_step_correct[:, k].mean() * 100)
                                  for k in range(S)],
        "maj_baseline_final_pct": float(maj_baseline_final),
        "probe1_predict_final": grid_final.tolist(),
        "probe1_f1": f1_final.tolist(),
        "probe1_best": best_cell(grid_final),
        "probe2_predict_self_step": grid_self.tolist(),
        "probe2_f1": f1_self.tolist(),
        "probe2_best": best_cell(grid_self),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}")
    print(f"  Probe 1 best: {out['probe1_best']}")
    print(f"  Probe 2 best: {out['probe2_best']}")
    print(f"  Majority-baseline (final): {maj_baseline_final:.1f}%")

    # Plot
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: setup
        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        ax.axis("off")
        body = (
            f"Correctness probe on CODI-GPT-2 GSM8K activations\n\n"
            f"Source: gsm8k_latent_acts.pt (N={N}, {S} steps × {L} layers × 768)\n"
            f"Labels from: force_decode_per_step_gsm8k.json\n\n"
            f"  final-correct fraction: {final_correct.mean()*100:.1f}%\n"
            f"  majority-class baseline (final): {maj_baseline_final:.1f}%\n\n"
            f"  per-step correct fraction:\n"
        )
        for k in range(S):
            body += (f"    step {k+1}: {per_step_correct[:, k].mean()*100:5.1f}%  "
                     f"(maj baseline {maj_baseline_per_step[k]:.1f}%)\n")
        body += (f"\nProbe 1: predict the FINAL (step-6) emit's correctness from acts at (step, layer).\n"
                 f"  Best cell: step{out['probe1_best']['step']} L{out['probe1_best']['layer']}  "
                 f"acc = {out['probe1_best']['acc']*100:.1f}%  (vs maj baseline {maj_baseline_final:.1f}%)\n"
                 f"\nProbe 2: predict 'force-decode at THIS step is correct' from acts at (step, layer).\n"
                 f"  Best cell: step{out['probe2_best']['step']} L{out['probe2_best']['layer']}  "
                 f"acc = {out['probe2_best']['acc']*100:.1f}%\n")
        ax.text(0.04, 0.97, body, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_title("Correctness probe — setup", fontsize=13, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: heatmap of probe 1 acc
        fig, ax = plt.subplots(figsize=(14, 5))
        vmin = max(0.4, maj_baseline_final / 100 - 0.05)
        vmax = max(0.65, float(np.nanmax(grid_final)) + 0.02)
        im = ax.imshow(grid_final, aspect="auto", origin="lower",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        for s in range(S):
            for l in range(L):
                ax.text(l, s, f"{grid_final[s, l]*100:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if grid_final[s, l] < (vmin + vmax)/2 else "black")
        ax.set_xticks(range(L))
        ax.set_yticks(range(S)); ax.set_yticklabels([str(i+1) for i in range(S)])
        ax.set_xlabel("layer"); ax.set_ylabel("latent step")
        ax.set_title(f"Probe 1: predict FINAL step-6 correctness from each (step, layer) cell\n"
                     f"(majority baseline = {maj_baseline_final:.1f}%; chance = 50%)",
                     fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.04, label="test acc")
        b = out["probe1_best"]
        if b:
            ax.scatter([b["layer"]], [b["step"]-1], marker="*",
                       s=200, c="white", edgecolors="black", linewidths=1.5,
                       label=f"best: step{b['step']} L{b['layer']} acc={b['acc']*100:.1f}%")
            ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: heatmap of probe 2 acc
        fig, ax = plt.subplots(figsize=(14, 5))
        vmin = 0.55
        vmax = max(0.80, float(np.nanmax(grid_self)) + 0.02)
        im = ax.imshow(grid_self, aspect="auto", origin="lower",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        for s in range(S):
            for l in range(L):
                ax.text(l, s, f"{grid_self[s, l]*100:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if grid_self[s, l] < (vmin + vmax)/2 else "black")
        ax.set_xticks(range(L))
        ax.set_yticks(range(S)); ax.set_yticklabels([str(i+1) for i in range(S)])
        ax.set_xlabel("layer"); ax.set_ylabel("latent step")
        ax.set_title(f"Probe 2: predict THIS-step force-decode correctness from each (step, layer) cell\n"
                     f"(label changes per step — predicting whether emit at step k matches gold)",
                     fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.04, label="test acc")
        b = out["probe2_best"]
        if b:
            ax.scatter([b["layer"]], [b["step"]-1], marker="*",
                       s=200, c="white", edgecolors="black", linewidths=1.5)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: per-step best across layers (max-over-layer)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        best_per_step_final = np.nanmax(grid_final, axis=1)
        ax.bar(range(1, S+1), best_per_step_final * 100,
               color="#4c72b0", edgecolor="black")
        ax.axhline(maj_baseline_final, color="black", ls="--",
                   label=f"maj baseline {maj_baseline_final:.1f}%")
        for s, v in enumerate(best_per_step_final):
            ax.text(s+1, v*100 + 0.3, f"{v*100:.1f}",
                    ha="center", fontsize=9)
        ax.set_xticks(range(1, S+1))
        ax.set_xlabel("latent step")
        ax.set_ylabel("best-over-layer acc (%)")
        ax.set_title("Probe 1 (predict FINAL correctness):\nbest-anywhere per step",
                     fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

        ax = axes[1]
        best_per_step_self = np.nanmax(grid_self, axis=1)
        ax.bar(range(1, S+1), best_per_step_self * 100,
               color="#dd8452", edgecolor="black")
        for s, v in enumerate(best_per_step_self):
            ax.text(s+1, v*100 + 0.3, f"{v*100:.1f}",
                    ha="center", fontsize=9)
            ax.text(s+1, maj_baseline_per_step[s] - 2,
                    f"maj={maj_baseline_per_step[s]:.0f}",
                    ha="center", fontsize=8, color="gray")
        ax.set_xticks(range(1, S+1))
        ax.set_xlabel("latent step")
        ax.set_ylabel("best-over-layer acc (%)")
        ax.set_title("Probe 2 (predict THIS-step correctness):\nbest-anywhere per step",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Where can correctness be read off?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
