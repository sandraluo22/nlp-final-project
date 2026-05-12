"""Operator probe trained on CODI-attributed labels (gold for correct preds +
inferred silver-1 op for wrong-but-1-step-arithmetic preds).

Per-row label assignment:
  - category == 'gold'    : label = gsm8k Type (first-marker op of gold trace)
  - category == 'silver1' : label = op parsed from silver_expression
  - otherwise             : SKIP (silver2/silver3/nonsense are not single-op)

Two test setups:

  A) Held-out random 20% split on the COMBINED (gold + silver-1) pool.
     Reports per-(layer, step) accuracy / per-class P/R/F1.

  B) Cross-attribution generalization:
        - Train on gold-only,    test on silver-1-only
        - Train on silver1-only, test on gold-only
     Tests whether 'operator-direction' is the same across the two sources.

Compares with a 'gold-only' baseline that ignores silver-1 entirely.

Output:
  operator_probe_codi_attributed.json   (per-config metrics over L × S)
  operator_probe_codi_attributed.pdf    (4-page slideshow)
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
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[3]
ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
SILVER = REPO / "visualizations-all" / "gpt2-gsm8k" / "correctness-analysis" / "silver_traces_gsm8k.json"
PD = Path(__file__).resolve().parent
OUT_PDF = PD / "operator_probe_codi_attributed.pdf"
OUT_JSON = PD / "operator_probe_codi_attributed.json"

OP_SYM_TO_NAME = {
    "+": "Addition", "-": "Subtraction",
    "*": "Multiplication", "/": "Common-Division",
}
OP_NAME_TO_SYM = {v: k for k, v in OP_SYM_TO_NAME.items()}
OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


class _GSMShim:
    def __init__(self, ds):
        self.ds = ds
        first_ops = []
        golds = []
        for ex in ds:
            ans = ex["answer"].replace(",", "")
            m_first = re.search(r"<<.*?([+\-*/]).*?=", ans)
            if m_first:
                first_ops.append(OP_SYM_TO_NAME[m_first.group(1)])
            else:
                first_ops.append("Addition")  # fallback if no marker
            mg = re.search(r"####\s*(-?\d+\.?\d*)", ans)
            golds.append(float(mg.group(1)) if mg else 0.0)
        self.Type = first_ops
        self.Answer = golds

    def __getitem__(self, k):
        if k == "Type": return self.Type
        if k == "Answer": return self.Answer
        return self.ds[k]


def op_from_silver_expr(expr: str) -> str | None:
    """Extract single-op symbol from a silver-1 expression like '7+5=12'."""
    if not expr:
        return None
    # Drop the '=...' part and the answer
    lhs = expr.split("=")[0]
    # Find an op symbol between two number literals (so '-1+2' picks '+')
    m = re.search(r"(?<=[\d.])\s*([+\-*/])\s*(?=[\d.])", lhs)
    return m.group(1) if m else None


def build_dataset():
    silver = json.load(open(SILVER))
    ds = load_dataset("gsm8k", "main")
    shim = _GSMShim(ds["test"])
    types = shim["Type"]

    rows = []
    for r in silver["rows"]:
        i = r["idx"]
        cat = r["category"]
        if cat == "gold":
            rows.append({"idx": i, "label": types[i], "source": "gold"})
        elif cat == "silver1":
            sym = op_from_silver_expr(r.get("silver_expression"))
            if sym is None:
                continue
            rows.append({
                "idx": i, "label": OP_SYM_TO_NAME[sym], "source": "silver1"
            })
    return rows


def fit_eval_split(X, y, sources, n_splits=5, test_frac=0.2, seed=42):
    """Stratified-by-label random splits.  Returns mean per-class P/R/F1 and acc."""
    rng = np.random.RandomState(seed)
    accs = []
    f1s = []
    per_class = {op: {"p": [], "r": [], "f": []} for op in OPS}
    src_acc = {"gold": [], "silver1": []}
    for s in range(n_splits):
        Xtr, Xte, ytr, yte, src_tr, src_te = train_test_split(
            X, y, sources, test_size=test_frac, random_state=seed + s, stratify=y
        )
        scaler = StandardScaler().fit(Xtr)
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf.fit(scaler.transform(Xtr), ytr)
        pred = clf.predict(scaler.transform(Xte))
        accs.append(float((pred == yte).mean()))
        f1s.append(float(f1_score(yte, pred, labels=OPS, average="macro", zero_division=0)))
        p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=OPS, zero_division=0)
        for j, op in enumerate(OPS):
            per_class[op]["p"].append(p[j]); per_class[op]["r"].append(r[j]); per_class[op]["f"].append(f[j])
        for src in ("gold", "silver1"):
            mask = np.array([x == src for x in src_te])
            if mask.sum() > 0:
                src_acc[src].append(float((pred[mask] == yte[mask]).mean()))
    return {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "per_class": {op: {k: float(np.mean(v)) for k, v in d.items()} for op, d in per_class.items()},
        "src_acc": {k: (float(np.mean(v)) if v else None) for k, v in src_acc.items()},
        "n_test_per_split": int(len(yte)),
    }


def fit_cross(X, y, sources, train_src, test_src):
    """Train on rows where source==train_src, test on rows where source==test_src."""
    mask_tr = np.array([s == train_src for s in sources])
    mask_te = np.array([s == test_src for s in sources])
    if mask_tr.sum() < 12 or mask_te.sum() < 12:
        return None
    scaler = StandardScaler().fit(X[mask_tr])
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(scaler.transform(X[mask_tr]), y[mask_tr])
    pred = clf.predict(scaler.transform(X[mask_te]))
    acc = float((pred == y[mask_te]).mean())
    p, r, f, _ = precision_recall_fscore_support(y[mask_te], pred, labels=OPS, zero_division=0)
    return {
        "acc": acc, "n_train": int(mask_tr.sum()), "n_test": int(mask_te.sum()),
        "per_class": {OPS[j]: {"p": float(p[j]), "r": float(r[j]), "f": float(f[j])} for j in range(4)},
    }


def main():
    rows = build_dataset()
    cnt = Counter((r["source"], r["label"]) for r in rows)
    print(f"loaded {len(rows)} rows")
    print(f"  by source: {Counter(r['source'] for r in rows)}")
    print(f"  by label:  {Counter(r['label'] for r in rows)}")
    for op in OPS:
        g = cnt.get(("gold", op), 0); s = cnt.get(("silver1", op), 0)
        print(f"    {op:18s}  gold={g:4d}  silver1={s:3d}")

    print("loading activations", flush=True)
    acts = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    print(f"  shape={acts.shape}")  # (N, S, L, H)
    N, S, L, H = acts.shape

    idx = np.array([r["idx"] for r in rows])
    y = np.array([r["label"] for r in rows])
    sources = np.array([r["source"] for r in rows])

    # Filter to indices that exist
    valid = idx < N
    idx, y, sources = idx[valid], y[valid], sources[valid]
    print(f"  N_valid={len(idx)}")

    # Subset accordingly
    acts_sub = acts[idx]  # (n, S, L, H)

    # ---- Grid sweep: per (layer, step), 4 configs ----
    print("running probe grid sweep", flush=True)
    grid = {}  # (config) -> (S x L) metric maps

    configs = {
        "combined_split":     ("all",     "all"),       # train on gold+silver, 5-fold 80/20
        "gold_only_split":    ("gold",    "gold"),      # train on gold only, 5-fold split
        "silver1_only_split": ("silver1", "silver1"),
        "train_gold_test_silver":   ("gold",    "silver1"),
        "train_silver_test_gold":   ("silver1", "gold"),
    }

    metrics = {cfg: np.full((S, L), np.nan) for cfg in configs}
    f1s = {cfg: np.full((S, L), np.nan) for cfg in configs}
    detail = {}  # winning cell will store full per-class breakdown

    t0 = time.time()
    for step in range(S):
        for layer in range(L):
            X = acts_sub[:, step, layer, :]  # (n, H)

            for cfg, (tr_src, te_src) in configs.items():
                if tr_src == "all":
                    res = fit_eval_split(X, y, sources)
                    metrics[cfg][step, layer] = res["acc_mean"]
                    f1s[cfg][step, layer] = res["f1_mean"]
                    detail.setdefault(cfg, {})[(step, layer)] = res
                elif tr_src == te_src:
                    # within-source 5-fold split
                    mask = sources == tr_src
                    if mask.sum() < 24: continue
                    res = fit_eval_split(X[mask], y[mask], sources[mask])
                    metrics[cfg][step, layer] = res["acc_mean"]
                    f1s[cfg][step, layer] = res["f1_mean"]
                    detail.setdefault(cfg, {})[(step, layer)] = res
                else:
                    res = fit_cross(X, y, sources, tr_src, te_src)
                    if res is None: continue
                    metrics[cfg][step, layer] = res["acc"]
                    f1s[cfg][step, layer] = float(np.mean([res["per_class"][o]["f"] for o in OPS]))
                    detail.setdefault(cfg, {})[(step, layer)] = res
        print(f"  step {step+1}/{S} done ({time.time()-t0:.0f}s)", flush=True)

    # Pick winners (best layer per step) for each config (by acc)
    summary = {}
    for cfg in configs:
        flat = metrics[cfg].flatten()
        if np.all(np.isnan(flat)):
            summary[cfg] = None; continue
        best_idx = int(np.nanargmax(flat))
        s, l = best_idx // L, best_idx % L
        summary[cfg] = {
            "best_acc": float(metrics[cfg][s, l]),
            "best_f1": float(f1s[cfg][s, l]),
            "best_step": s + 1, "best_layer": l,
            "best_detail": detail.get(cfg, {}).get((s, l)),
        }

    out = {
        "n_rows": int(len(idx)),
        "by_source": {k: int(v) for k, v in Counter(sources.tolist()).items()},
        "by_label":  {k: int(v) for k, v in Counter(y.tolist()).items()},
        "by_source_label": {f"{s}__{o}": int(v) for (s, o), v in cnt.items()},
        "configs": list(configs.keys()),
        "summary": summary,
        "metrics_acc": {cfg: metrics[cfg].tolist() for cfg in configs},
        "metrics_f1":  {cfg: f1s[cfg].tolist() for cfg in configs},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    # ---- Build slideshow ----
    print("rendering slideshow", flush=True)
    with PdfPages(OUT_PDF) as pdf:
        # PAGE 1: data summary
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        txt = "CODI-attributed operator probe — GSM8K\n\n"
        txt += f"Total rows: {len(idx)}\n"
        txt += f"  gold (correct preds, label = gsm8k Type):     {int((sources=='gold').sum())}\n"
        txt += f"  silver1 (1-step wrong, label = parsed op):     {int((sources=='silver1').sum())}\n\n"
        txt += "Per (source × label) counts:\n"
        txt += f"  {'Operator':<20s}  gold  silver1\n"
        for op in OPS:
            g = cnt.get(("gold", op), 0); s = cnt.get(("silver1", op), 0)
            txt += f"  {op:<20s}  {g:4d}  {s:5d}\n"
        txt += "\nProbe: RidgeClassifier (α=1.0, class-balanced) on standardized acts.\n"
        txt += "5-fold stratified 80/20 splits for within-set configs.\n"
        txt += "Cross-attribution configs use full train→full test single fit.\n"
        ax.text(0.05, 0.95, txt, va="top", ha="left", family="monospace", fontsize=11)
        ax.set_title("Setup", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 2: summary bar chart of best (acc, f1) per config
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cfg_names = list(configs.keys())
        cfg_labels = ["combined\n(gold+silver1)", "gold only", "silver1 only",
                      "train gold→\ntest silver1", "train silver1→\ntest gold"]
        accs = [summary[c]["best_acc"] if summary[c] else 0 for c in cfg_names]
        f1v  = [summary[c]["best_f1"]  if summary[c] else 0 for c in cfg_names]
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]
        axes[0].bar(range(len(cfg_names)), accs, color=colors)
        axes[0].set_xticks(range(len(cfg_names))); axes[0].set_xticklabels(cfg_labels, rotation=20, ha="right")
        axes[0].set_ylabel("best accuracy"); axes[0].set_ylim(0, 1.0)
        axes[0].axhline(0.25, ls="--", c="k", alpha=0.4, label="chance (4-class)")
        for i, v in enumerate(accs): axes[0].text(i, v+0.01, f"{v:.2f}", ha="center", fontsize=9)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title("Best accuracy (over L × S)")

        axes[1].bar(range(len(cfg_names)), f1v, color=colors)
        axes[1].set_xticks(range(len(cfg_names))); axes[1].set_xticklabels(cfg_labels, rotation=20, ha="right")
        axes[1].set_ylabel("best macro-F1"); axes[1].set_ylim(0, 1.0)
        axes[1].axhline(0.25, ls="--", c="k", alpha=0.4, label="chance")
        for i, v in enumerate(f1v): axes[1].text(i, v+0.01, f"{v:.2f}", ha="center", fontsize=9)
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].set_title("Best macro-F1 (over L × S)")
        fig.suptitle("Operator probe — silver+gold attribution vs gold-only vs cross",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 3: heatmaps of acc per config (S × L)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for k, cfg in enumerate(cfg_names):
            ax = axes.flat[k]
            im = ax.imshow(metrics[cfg], origin="lower", aspect="auto",
                           vmin=0.2, vmax=1.0, cmap="viridis")
            ax.set_title(f"{cfg_labels[k]}")
            ax.set_xlabel("layer"); ax.set_ylabel("step")
            ax.set_xticks(range(L)); ax.set_yticks(range(S))
            ax.set_yticklabels([str(i+1) for i in range(S)])
            plt.colorbar(im, ax=ax, fraction=0.04)
        axes.flat[-1].axis("off")
        fig.suptitle("Per-(layer, step) accuracy", fontweight="bold", fontsize=13)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 4: per-class P/R/F1 at the winning cell of combined_split
        win = summary["combined_split"]
        if win and win["best_detail"]:
            d = win["best_detail"]
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            for col, metric in enumerate(["p", "r", "f"]):
                vals = [d["per_class"][op][metric] for op in OPS]
                axes[col].bar(OPS, vals, color=[
                    "#ff7f0e", "#1f77b4", "#d62728", "#2ca02c"])
                axes[col].set_ylim(0, 1.0)
                axes[col].set_title({"p": "precision", "r": "recall", "f": "F1"}[metric])
                axes[col].set_xticklabels(OPS, rotation=20, ha="right")
                for i, v in enumerate(vals):
                    axes[col].text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
            fig.suptitle(f"combined_split winner: step {win['best_step']}, "
                         f"layer {win['best_layer']}  (acc={win['best_acc']:.3f}, "
                         f"F1={win['best_f1']:.3f})", fontweight="bold", fontsize=12)
            # also report src-conditional accuracy
            src_a = d.get("src_acc", {})
            txt = (f"\naccuracy on gold-test rows:    {src_a.get('gold', 'n/a'):.3f}" if src_a.get('gold') is not None else "")
            txt += (f"\naccuracy on silver1-test rows: {src_a.get('silver1', 'n/a'):.3f}" if src_a.get('silver1') is not None else "")
            fig.text(0.5, -0.02, txt, ha="center", family="monospace", fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
