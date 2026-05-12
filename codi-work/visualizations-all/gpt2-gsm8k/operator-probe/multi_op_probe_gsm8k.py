"""Multi-operation probe on GSM8K: per-marker operator + operand + result.

A GSM8K answer trace is a chain `<<a1 op1 b1 = c1>><<a2 op2 b2 = c2>>...`
of arithmetic markers.  CODI runs a 6-step latent loop.  If each latent
step corresponds to a step in the gold chain, then probing at step k
should best predict the k-th gold marker's operation.

For each MARKER POSITION m in 1..4 we run THREE probes:
  - op_m         (4-class: +, −, ×, /)
  - a_m last digit  (10-class)
  - c_m last digit  (10-class)

Each probe is a RidgeClassifier(class-balanced) with StandardScaler,
single 80/20 stratified split per (step, layer, marker).  Per cell
output: accuracy + macro-F1.

Output: heatmaps for each (probe-type × marker-position) cell, plus a
summary showing the best (step, layer) per (marker, probe-type) to test
whether the optimal step shifts as we move down the chain.

Outputs: multi_op_probe_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
PD = Path(__file__).resolve().parent
OUT_PDF = PD / "multi_op_probe_gsm8k.pdf"
OUT_JSON = PD / "multi_op_probe_gsm8k.json"

OPS = ["+", "-", "*", "/"]
M_MAX = 4         # probe markers 1..4 (m=5+ has <130 examples)
N_DIGITS = 10


def parse_markers(answer_text: str):
    """Return list of (a, op, b, c) tuples from <<...>> markers."""
    ans = answer_text.replace(",", "")
    ms = re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", ans)
    return [(float(a), op, float(b), float(c)) for a, op, b, c in ms]


def last_digit(x: float) -> int:
    return int(abs(int(round(x))) % 10)


def fit_probe(X, y, n_classes_hint=None):
    """Single 80/20 stratified split; return acc, macro-F1, n."""
    if len(set(y)) < 2:
        return None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(Counter(y).values()) >= 2 else None
    )
    scaler = StandardScaler().fit(Xtr)
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))
    acc = float((pred == yte).mean())
    classes = sorted(set(y))
    f1 = float(f1_score(yte, pred, labels=classes, average="macro", zero_division=0))
    return {"acc": acc, "f1": f1, "n_train": len(ytr), "n_test": len(yte)}


def main():
    print("loading GSM8K", flush=True)
    ds = load_dataset("gsm8k", "main")["test"]
    markers_per_problem = [parse_markers(ex["answer"]) for ex in ds]
    N_total = len(markers_per_problem)
    print(f"N={N_total}; markers/problem distribution: "
          f"{dict(Counter(len(m) for m in markers_per_problem))}")

    print("loading activations", flush=True)
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = acts.shape
    assert N == N_total
    print(f"  shape={acts.shape}")

    # Build per-marker label arrays + masks
    # For each marker position m: which problems have >=m markers
    marker_meta = {}
    for m in range(1, M_MAX + 1):
        mask = np.array([len(ms) >= m for ms in markers_per_problem])
        ops_m = np.array([
            (markers_per_problem[i][m-1][1] if mask[i] else "")
            for i in range(N)
        ])
        a_m_ld = np.array([
            (last_digit(markers_per_problem[i][m-1][0]) if mask[i] else -1)
            for i in range(N)
        ])
        c_m_ld = np.array([
            (last_digit(markers_per_problem[i][m-1][3]) if mask[i] else -1)
            for i in range(N)
        ])
        marker_meta[m] = {
            "mask": mask, "n": int(mask.sum()),
            "ops": ops_m, "a_ld": a_m_ld, "c_ld": c_m_ld,
            "ops_dist": dict(Counter(ops_m[mask].tolist())),
        }
        print(f"  m={m}: N={mask.sum()}  op dist: {marker_meta[m]['ops_dist']}")

    # ---- Sweep: per (probe_type, marker, step, layer) ----
    probes = ["op", "a_ld", "c_ld"]
    chance = {"op": 0.25, "a_ld": 0.10, "c_ld": 0.10}
    chance_label = {"op": "4-cls chance=0.25", "a_ld": "10-cls chance=0.10",
                    "c_ld": "10-cls chance=0.10"}

    acc_grids = {p: {m: np.full((S, L), np.nan) for m in range(1, M_MAX+1)}
                 for p in probes}
    f1_grids  = {p: {m: np.full((S, L), np.nan) for m in range(1, M_MAX+1)}
                 for p in probes}

    t0 = time.time()
    total_cells = len(probes) * M_MAX * S * L
    done = 0
    for probe in probes:
        for m in range(1, M_MAX + 1):
            mask = marker_meta[m]["mask"]
            if probe == "op":
                y = marker_meta[m]["ops"][mask]
            elif probe == "a_ld":
                y = marker_meta[m]["a_ld"][mask].astype(int)
            else:
                y = marker_meta[m]["c_ld"][mask].astype(int)
            idx = np.where(mask)[0]
            for step in range(S):
                for layer in range(L):
                    X = acts[idx, step, layer, :]
                    r = fit_probe(X, y)
                    if r is not None:
                        acc_grids[probe][m][step, layer] = r["acc"]
                        f1_grids[probe][m][step, layer] = r["f1"]
                    done += 1
            print(f"  {probe} m={m} done  ({done}/{total_cells}, "
                  f"{time.time()-t0:.0f}s)", flush=True)

    # ---- Best (step, layer) per (probe, marker) ----
    best = {}
    for probe in probes:
        best[probe] = {}
        for m in range(1, M_MAX+1):
            G = acc_grids[probe][m]
            flat = G.flatten()
            if np.all(np.isnan(flat)):
                best[probe][m] = None; continue
            i = int(np.nanargmax(flat))
            s, l = i // L, i % L
            best[probe][m] = {
                "step": s+1, "layer": l,
                "acc": float(G[s, l]), "f1": float(f1_grids[probe][m][s, l]),
            }

    print("\nBest cell per (probe, marker):")
    print(f"  {'probe':<6s} {'mkr':<3s}  best step  best layer   acc     F1")
    for probe in probes:
        for m in range(1, M_MAX+1):
            b = best[probe][m]
            if b is None: continue
            print(f"  {probe:<6s}  m{m}    step{b['step']:<2d}     L{b['layer']:<2d}    "
                  f"{b['acc']:.3f}  {b['f1']:.3f}")

    out = {
        "N_total": N_total,
        "M_MAX": M_MAX,
        "marker_n": {m: marker_meta[m]["n"] for m in range(1, M_MAX+1)},
        "marker_ops_dist": {m: marker_meta[m]["ops_dist"] for m in range(1, M_MAX+1)},
        "acc": {p: {m: acc_grids[p][m].tolist() for m in range(1, M_MAX+1)} for p in probes},
        "f1":  {p: {m:  f1_grids[p][m].tolist() for m in range(1, M_MAX+1)} for p in probes},
        "best": best,
        "chance": chance,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    # ---- Slideshow ----
    print("rendering slideshow", flush=True)
    probe_title = {"op": "operator (4-cls)",
                   "a_ld": "a-operand last digit (10-cls)",
                   "c_ld": "result last digit (10-cls)"}

    with PdfPages(OUT_PDF) as pdf:
        # PAGE 1: setup
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axis("off")
        txt = "Multi-operation probe on GSM8K\n\n"
        txt += f"  N_total = {N_total}; markers per problem distribution:\n"
        cnt = dict(Counter(len(m) for m in markers_per_problem))
        for k in sorted(cnt.keys()):
            txt += f"    chain_len={k}: {cnt[k]}\n"
        txt += f"\n  Probing markers m=1..{M_MAX} (m=5+ has <130 examples).\n"
        txt += "\n  N usable per marker position:\n"
        for m in range(1, M_MAX+1):
            txt += f"    m={m}: N={marker_meta[m]['n']}  ops: {marker_meta[m]['ops_dist']}\n"
        txt += "\n  Three probes per marker:\n"
        txt += "    op       — 4-class operator (+, −, ×, /)\n"
        txt += "    a_ld     — last digit of left operand (10-class)\n"
        txt += "    c_ld     — last digit of result (10-class)\n"
        txt += "  RidgeClassifier(α=1.0, class-balanced) on standardized acts,\n"
        txt += "  single 80/20 stratified split per (step, layer) cell.\n"
        ax.text(0.04, 0.96, txt, va="top", ha="left", family="monospace", fontsize=11)
        ax.set_title("Setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGES 2-4: one page per probe-type, 2x2 marker-position panel of heatmaps
        for probe in probes:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for m, ax in zip(range(1, M_MAX+1), axes.flat):
                G = acc_grids[probe][m]
                # color range: chance to max
                c = chance[probe]
                vmin = max(0.0, c - 0.05)
                vmax = max(0.3, float(np.nanmax(G)) + 0.02)
                im = ax.imshow(G, origin="lower", aspect="auto",
                               vmin=vmin, vmax=vmax, cmap="viridis")
                ax.set_xlabel("layer"); ax.set_ylabel("step")
                ax.set_xticks(range(L)); ax.set_yticks(range(S))
                ax.set_yticklabels([str(i+1) for i in range(S)])
                b = best[probe][m]
                if b is not None:
                    ax.scatter([b["layer"]], [b["step"]-1], marker="*",
                               s=150, c="white", edgecolors="black",
                               linewidths=1, zorder=5)
                    ax.set_title(f"m={m}  (N={marker_meta[m]['n']})  "
                                 f"best acc={b['acc']:.3f} @ step{b['step']}, L{b['layer']}")
                else:
                    ax.set_title(f"m={m}: no data")
                plt.colorbar(im, ax=ax, fraction=0.045)
            fig.suptitle(f"Per-marker {probe_title[probe]} probe   "
                         f"({chance_label[probe]})",
                         fontsize=13, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 5: best-cell-per-marker summary (does optimal step shift across markers?)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        ms = list(range(1, M_MAX+1))
        for col, probe in enumerate(probes):
            best_steps = [best[probe][m]["step"] if best[probe][m] else np.nan for m in ms]
            best_layers = [best[probe][m]["layer"] if best[probe][m] else np.nan for m in ms]
            best_accs   = [best[probe][m]["acc"] if best[probe][m] else np.nan for m in ms]
            ax = axes[col]
            ax2 = ax.twinx()
            ax.plot(ms, best_steps, "-o", color="C0", label="best step")
            ax.plot(ms, best_layers, "-s", color="C1", label="best layer")
            ax2.plot(ms, best_accs, "-^", color="C2", label="best acc")
            ax.set_xlabel("marker position m"); ax.set_xticks(ms)
            ax.set_ylabel("best (step, layer)")
            ax2.set_ylabel("best acc"); ax2.axhline(chance[probe], ls="--", c="gray", alpha=0.5)
            ax.set_title(f"{probe_title[probe]}")
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="lower right", fontsize=8)
            ax.set_ylim(-0.5, max(S, L) + 0.5)
            ax2.set_ylim(0, max(0.7, max([x for x in best_accs if not np.isnan(x)], default=0.3) + 0.05))
        fig.suptitle("Does the optimal latent step shift across marker positions?  "
                     "(if CODI computes chain sequentially, later markers should peak at later steps)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
