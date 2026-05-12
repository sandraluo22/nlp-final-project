"""Re-run the multi-op probe + cross-marker selectivity analysis but on
MODE 1 problems only (correct + loop-rescued; no step-1 'shortcut' wins,
no wrong-final guesses).

Mode 1 = problems CODI got correct at step 6 AND was wrong at step 1
(i.e., the loop actually rescued them). N = 280.

The hypothesis: if the loop is doing real computation on these problems,
the per-cell selectivity should be CLEANER on Mode 1 than on the full
heterogeneous set.

For each (probe_type ∈ {op, a_ld, c_ld}, marker m ∈ {1..4}, step k, layer l):
  X = gsm8k_latent_acts[Mode1_indices_with_≥m_markers, k, l, :]
  y = marker m's label (op or last-digit)
Fit RidgeClassifier with stratified 80/20 split.

Compute the same selectivity heatmap as multi_op_probe_evenodd_gsm8k.pdf:
  per (step, layer): argmax marker + selectivity (best acc − mean of others)

Output: mode1_probe_selectivity_gsm8k.{json,pdf}
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

OUT_JSON = PD / "mode1_probe_selectivity_gsm8k.json"
OUT_PDF = PD / "mode1_probe_selectivity_gsm8k.pdf"

PROBES = ["op", "a_ld", "c_ld"]
M_MAX = 4
CHANCE = {"op": 0.25, "a_ld": 0.10, "c_ld": 0.10}


def parse_markers(s):
    s = s.replace(",", "")
    return re.findall(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)>>", s)


def emit_final(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def last_digit(x):
    return int(abs(int(round(float(x)))) % 10)


def fit_probe(X, y):
    if len(set(y)) < 2 or len(y) < 30:
        return None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(Counter(y).values()) >= 2 else None
    )
    sc = StandardScaler().fit(Xtr)
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = float((pred == yte).mean())
    classes = sorted(set(y))
    f1 = float(f1_score(yte, pred, labels=classes, average="macro", zero_division=0))
    return {"acc": acc, "f1": f1, "n_test": len(yte)}


def main():
    print("loading data", flush=True)
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = acts.shape
    print(f"  acts: {acts.shape}")
    fd = json.load(open(FD_JSON))
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]

    # Identify Mode 1 indices
    rows = {r["idx"]: r for r in fd["rows"]}
    mode1_mask = np.zeros(N, dtype=bool)
    markers_per_problem = [[] for _ in range(N)]
    for i in range(N):
        ex = ds[i]
        gm = re.search(r"####\s*(-?\d+\.?\d*)", ex["answer"].replace(",", ""))
        if gm is None: continue
        gold = float(gm.group(1))
        markers_per_problem[i] = parse_markers(ex["answer"])
        r = rows.get(i)
        if r is None: continue
        emit_vals = [emit_final(e) for e in r["step_emits"]]
        sc = [v is not None and abs(v - gold) < 1e-3 for v in emit_vals]
        # Mode 1 = correct at step 6 AND wrong at step 1
        if sc[-1] and not sc[0]:
            mode1_mask[i] = True
    print(f"  Mode 1 problems: {mode1_mask.sum()}")

    # Per-marker labels (only for Mode 1 problems)
    marker_meta = {}
    for m in range(1, M_MAX + 1):
        mask_geq = np.array([len(ms) >= m for ms in markers_per_problem])
        combined = mask_geq & mode1_mask
        ops_m = np.array([(markers_per_problem[i][m-1][1] if combined[i] else "")
                          for i in range(N)])
        a_m_ld = np.array([(last_digit(markers_per_problem[i][m-1][0]) if combined[i] else -1)
                           for i in range(N)])
        c_m_ld = np.array([(last_digit(markers_per_problem[i][m-1][3]) if combined[i] else -1)
                           for i in range(N)])
        marker_meta[m] = {"mask": combined, "ops": ops_m, "a_ld": a_m_ld, "c_ld": c_m_ld,
                          "n": int(combined.sum())}
        print(f"  m={m}: Mode 1 subset N={combined.sum()}")

    # Sweep probes
    print("\nrunning probe sweep (Mode 1 only)...", flush=True)
    acc_grids = {p: {m: np.full((S, L), np.nan) for m in range(1, M_MAX + 1)}
                 for p in PROBES}
    f1_grids = {p: {m: np.full((S, L), np.nan) for m in range(1, M_MAX + 1)}
                for p in PROBES}
    for p in PROBES:
        for m in range(1, M_MAX + 1):
            mask = marker_meta[m]["mask"]
            idx = np.where(mask)[0]
            if len(idx) < 30:
                print(f"  {p} m={m}: too few ({len(idx)}); skip"); continue
            if p == "op":
                y = marker_meta[m]["ops"][mask]
            elif p == "a_ld":
                y = marker_meta[m]["a_ld"][mask].astype(int)
            else:
                y = marker_meta[m]["c_ld"][mask].astype(int)
            for k in range(S):
                for l in range(L):
                    X = acts[idx, k, l, :]
                    r = fit_probe(X, y)
                    if r is not None:
                        acc_grids[p][m][k, l] = r["acc"]
                        f1_grids[p][m][k, l] = r["f1"]
            print(f"  {p} m={m} done", flush=True)

    # Best cell + selectivity (cross-marker)
    best = {p: {} for p in PROBES}
    selectivity = {p: np.full((S, L), np.nan) for p in PROBES}
    argmax_marker = {p: np.full((S, L), -1, dtype=int) for p in PROBES}
    for p in PROBES:
        for m in range(1, M_MAX + 1):
            G = acc_grids[p][m]
            if np.all(np.isnan(G)):
                best[p][m] = None; continue
            i = int(np.nanargmax(G.flatten()))
            s, l = i // L, i % L
            best[p][m] = {"step": s + 1, "layer": l,
                           "acc": float(G[s, l]), "f1": float(f1_grids[p][m][s, l])}
        # selectivity per cell
        accM = np.stack([acc_grids[p][m] for m in range(1, M_MAX + 1)], axis=-1)  # (S, L, M)
        argmax_marker[p] = accM.argmax(axis=-1)
        for s in range(S):
            for l in range(L):
                accs = accM[s, l, :]
                if np.all(np.isnan(accs)): continue
                a_max = float(np.nanmax(accs))
                a_others = np.delete(accs, np.nanargmax(accs))
                selectivity[p][s, l] = a_max - float(np.nanmean(a_others))

    # Print summary
    print("\nBest cells (Mode 1 only):")
    for p in PROBES:
        print(f"  {p}:")
        for m in range(1, M_MAX + 1):
            b = best[p].get(m)
            if b: print(f"    m={m}: step{b['step']} L{b['layer']}  acc={b['acc']:.3f}  f1={b['f1']:.3f}")

    out = {
        "n_mode1": int(mode1_mask.sum()),
        "marker_n_mode1": {m: marker_meta[m]["n"] for m in range(1, M_MAX + 1)},
        "best": {p: {m: best[p][m] for m in range(1, M_MAX + 1) if best[p][m]} for p in PROBES},
        "acc": {p: {m: acc_grids[p][m].tolist() for m in range(1, M_MAX + 1)} for p in PROBES},
        "f1":  {p: {m: f1_grids[p][m].tolist()  for m in range(1, M_MAX + 1)} for p in PROBES},
        "selectivity": {p: selectivity[p].tolist() for p in PROBES},
        "argmax_marker": {p: argmax_marker[p].tolist() for p in PROBES},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # ===== Plots =====
    with PdfPages(OUT_PDF) as pdf:
        # Setup page
        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        ax.axis("off")
        body = (
            f"Mode 1 probe + selectivity analysis\n\n"
            f"Mode 1 problems (correct + loop-rescued, NO shortcut wins): N={mode1_mask.sum()}\n\n"
            f"Per-marker Mode 1 sample sizes:\n"
        )
        for m in range(1, M_MAX + 1):
            body += f"  m={m}: N={marker_meta[m]['n']}\n"
        body += ("\nBest cells under Mode-1 filter (op probe):\n")
        for m in range(1, M_MAX + 1):
            b = best["op"].get(m)
            if b: body += f"  m={m}: step{b['step']} L{b['layer']}  acc={b['acc']:.3f}\n"
        body += ("\nCompare against the unrestricted probe best cells:\n"
                 "  m=1: step1 L11 acc=0.573  (unrestricted)\n"
                 "  m=2: step2 L0  acc=0.493\n"
                 "  m=3: step2 L7  acc=0.485\n"
                 "  m=4: step3 L8  acc=0.569\n"
                 "If the loop's representations are CLEANER for Mode 1 problems,\n"
                 "the new best-cell accs should be HIGHER and/or selectivities SHARPER.\n")
        ax.text(0.04, 0.97, body, va="top", ha="left",
                family="monospace", fontsize=10)
        ax.set_title("Mode 1 probe — setup", fontsize=13, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Per-probe per-marker heatmaps (op, a_ld, c_ld)
        for p in PROBES:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for m, ax in zip(range(1, M_MAX + 1), axes.flat):
                G = acc_grids[p][m]
                if np.all(np.isnan(G)):
                    ax.text(0.5, 0.5, f"m={m}: no data",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.axis("off"); continue
                c = CHANCE[p]
                vmin = max(0, c - 0.05)
                vmax = max(0.4, float(np.nanmax(G)) + 0.02)
                im = ax.imshow(G, aspect="auto", origin="lower",
                               cmap="viridis", vmin=vmin, vmax=vmax)
                b = best[p].get(m)
                if b:
                    ax.scatter([b["layer"]], [b["step"]-1], marker="*",
                               s=200, c="white", edgecolors="black", linewidths=1.5)
                ax.set_xticks(range(L)); ax.set_yticks(range(S))
                ax.set_yticklabels([str(i+1) for i in range(S)])
                ax.set_xlabel("layer"); ax.set_ylabel("step")
                title = f"m={m}  (Mode 1 N={marker_meta[m]['n']})"
                if b: title += f"  best={b['acc']:.3f} @ step{b['step']} L{b['layer']}"
                ax.set_title(title, fontsize=9, fontweight="bold")
                plt.colorbar(im, ax=ax, fraction=0.045)
            fig.suptitle(f"{p} probe — per-(step, layer) heatmap (Mode 1 only)  "
                         f"chance={CHANCE[p]:.2f}",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Selectivity heatmaps (the analog of slide 88 from master)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for col, p in enumerate(PROBES):
            ax = axes[col]
            G = selectivity[p]
            vmax_p = max(0.02, float(np.nanmax(G))) if not np.all(np.isnan(G)) else 0.1
            im = ax.imshow(G, aspect="auto", origin="lower",
                           cmap="magma", vmin=0, vmax=vmax_p)
            for s in range(S):
                for l in range(L):
                    if np.isnan(G[s, l]): continue
                    am = argmax_marker[p][s, l]
                    txt = f"m{am+1}\n{G[s,l]*100:.0f}"
                    ax.text(l, s, txt, ha="center", va="center", fontsize=6,
                            color="white" if G[s, l] < vmax_p / 2 else "black")
            ax.set_xticks(range(L))
            ax.set_yticks(range(S))
            ax.set_yticklabels([str(i+1) for i in range(S)])
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_title(f"{p}: argmax marker (top) + selectivity Δacc (bottom, pp)",
                         fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04, label="selectivity (Δacc)")
        fig.suptitle("MODE 1 selectivity per cell — which marker does this cell predict best, and how specifically?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Best-cell-per-marker comparison: Mode 1 vs unrestricted
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ms = list(range(1, M_MAX + 1))
        # Unrestricted op probe best cells (hardcoded from earlier runs)
        unrest_op = {1: 0.573, 2: 0.493, 3: 0.485, 4: 0.569}
        mode1_op = [best["op"][m]["acc"] if best["op"].get(m) else np.nan
                     for m in ms]
        unrest_vals = [unrest_op[m] for m in ms]
        w = 0.35
        ax.bar(np.array(ms) - w/2, unrest_vals, w, color="#7f7f7f",
               edgecolor="black", label="unrestricted (all problems with ≥m markers)")
        ax.bar(np.array(ms) + w/2, mode1_op, w, color="#2ca02c",
               edgecolor="black", label="Mode 1 (correct + loop-rescued only)")
        for m in ms:
            ax.text(m - w/2, unrest_vals[m-1] + 0.01, f"{unrest_vals[m-1]:.2f}",
                    ha="center", fontsize=8)
            if not np.isnan(mode1_op[m-1]):
                ax.text(m + w/2, mode1_op[m-1] + 0.01, f"{mode1_op[m-1]:.2f}",
                        ha="center", fontsize=8)
        ax.axhline(CHANCE["op"], color="black", ls=":", alpha=0.5,
                    label=f"chance = {CHANCE['op']:.2f}")
        ax.set_xticks(ms); ax.set_xlabel("marker position")
        ax.set_ylabel("best-cell op-probe accuracy")
        ax.set_title("op probe: best-cell accuracy — Mode 1 vs unrestricted",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
