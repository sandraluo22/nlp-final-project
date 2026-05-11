"""Unified slideshow combining all ':' position analyses for CODI-GPT-2.

Reads:
  svamp_colon_acts.pt + meta
  {ds}_colon_acts.pt + meta for each CF dataset
  correctness_probe_colon.json
  operator_probe_colon.json
  pca_bifurcation_colon.json

Renders:
  1. Title + position semantics
  2. SVAMP: correctness probe vs latent-loop probe (side by side)
  3. SVAMP: operator probe at ':' (compared to latent-step operator probe)
  4. SVAMP: PCA bifurcation at ':' — heatmap of |r| per layer per feature
  5. Per-CF dataset: operator probe and ':' acc per layer
  6. CF LDA at ':': mean ':' acts per operator (cosine similarity grid)
  7. Synthesis
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PD = Path(__file__).resolve().parent
OUT = PD / "colon_slideshow.pdf"

CF_DATASETS = ["cf_balanced", "cf_magmatched", "cf_under99", "cf_under99_b",
               "vary_a", "vary_a_2digit", "vary_b", "vary_b_2digit",
               "vary_both_2digit", "vary_numerals", "vary_operator",
               "numeral_pairs_a1_mul", "numeral_pairs_b1_sub"]
OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


def text_slide(pdf, title, lines):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax = fig.add_axes([0.05, 0.04, 0.9, 0.86]); ax.axis("off")
    y = 0.97
    for ln in lines:
        if ln.startswith("# "):
            ax.text(0.0, y, ln[2:], fontsize=12, fontweight="bold",
                    transform=ax.transAxes); y -= 0.045
        elif ln.startswith("- "):
            ax.text(0.02, y, "•  " + ln[2:], fontsize=10,
                    transform=ax.transAxes); y -= 0.034
        elif ln == "":
            y -= 0.020
        else:
            ax.text(0.0, y, ln, fontsize=10, transform=ax.transAxes); y -= 0.034
    pdf.savefig(fig, dpi=140); plt.close(fig)


def load_acts_meta(prefix):
    acts_p = PD / f"{prefix}_colon_acts.pt"
    meta_p = PD / f"{prefix}_colon_acts_meta.json"
    if not acts_p.exists() or not meta_p.exists():
        return None, None
    a = torch.load(acts_p, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(meta_p))
    return a, meta


def operator_acc_per_layer(a, types):
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y = np.array([op_to_idx.get(t, -1) for t in types])
    valid = y >= 0
    if valid.sum() < 50: return None
    a = a[valid]; y = y[valid]
    if len(np.unique(y)) < 2:
        return {"single_class": OPS[int(y[0])], "n": int(valid.sum())}
    accs = np.zeros(a.shape[1])
    from sklearn.model_selection import train_test_split
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2,
                                      random_state=0, stratify=y)
    for l in range(a.shape[1]):
        X = a[:, l, :]
        sc = StandardScaler().fit(X[idx_tr])
        clf = LogisticRegression(max_iter=2000, C=0.1,
                                 solver="lbfgs").fit(sc.transform(X[idx_tr]), y[idx_tr])
        accs[l] = (clf.predict(sc.transform(X[idx_te])) == y[idx_te]).mean()
    baseline = float(np.max([np.mean(y == c) for c in range(4)]))
    return {"accs": accs.tolist(), "baseline": baseline, "n": int(valid.sum())}


def main():
    a_sv, meta_sv = load_acts_meta("svamp")
    if a_sv is None:
        raise SystemExit("svamp_colon_acts.pt missing — run capture_colon_acts.py first")
    L_plus1 = a_sv.shape[1]

    cp = json.load(open(PD / "correctness_probe_colon.json")) if (PD / "correctness_probe_colon.json").exists() else None
    op = json.load(open(PD / "operator_probe_colon.json")) if (PD / "operator_probe_colon.json").exists() else None
    pca = json.load(open(PD / "pca_bifurcation_colon.json")) if (PD / "pca_bifurcation_colon.json").exists() else None

    # Reference: latent-loop probe results for comparison
    cp_lat = None
    for cand in (PD.parent.parent / "visualizations-all" / "gpt2" / "correctness_probe.json",):
        if cand.exists(): cp_lat = json.load(open(cand)); break

    with PdfPages(OUT) as pdf:
        # === Slide 1: setup ===
        text_slide(pdf, "CODI-GPT-2 at the `:` residual — what is encoded right before the answer?",
            [
                "# What is the `:` position?",
                "Decode template: 'The answer is: <number>'. After the latent loop, the model decodes:",
                "  step 0: input EOT → output ' The'   ─ residual = 'at EOT'",
                "  step 1: input ' The' → output ' answer'   ─ residual = 'at The'",
                "  step 2: input ' answer' → output ' is'   ─ residual = 'at answer'",
                "  step 3: input ' is' → output ':'   ─ residual = 'at is'",
                "  step 4: input ':' → output ' <digit>'   ─ residual = ':' ← THIS ONE (captured)",
                "",
                "# Why this cell",
                "- Canonical 'right before the answer' mechanistic cell: model is fully committed,",
                "  the next emitted token IS the answer digit. lm_head reads logits from this position.",
                "- Captured for ALL 1000 SVAMP problems (and all CF variants below). All examples",
                "  consistently emit ':' at decode step 3, so the residual lives at decode index 4.",
                "",
                "# Shape",
                f"`svamp_colon_acts.pt`  →  ({a_sv.shape[0]}, {a_sv.shape[1]}, {a_sv.shape[2]})  bf16",
                "  axis 1 = layer (0 = input embedding, 1..12 = transformer block outputs).",
                "",
                "# Companion latent-loop versions",
                "- correctness_probe.json (latent), pca_bifurcation_diagnose.json (latent),",
                "  steering_operator_all.json (latent + 'The' at decode pos 1) are still on disk.",
            ])

        # === Slide 2: correctness probe — colon vs latent ===
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        if cp is not None:
            xs = np.arange(len(cp["test_auc"]))
            axes[0].plot(xs, cp["test_auc"], "s-", color="#1f77b4", label="`:` cell (this work)")
            if cp_lat is not None:
                # latent has shape (S=6, L=13) test_auc; take max across steps per layer for fair plot
                lat = np.array(cp_lat["test_auc"])  # (S, L)
                axes[0].plot(xs, lat.max(axis=0), "o--", color="#ff7f0e",
                             label="latent loop (best step per layer)")
            axes[0].axhline(0.5, color="gray", ls=":", label="chance")
            axes[0].set_xlabel("layer"); axes[0].set_ylabel("test AUC")
            axes[0].set_title(f"correctness probe AUC: ':' vs latent (SVAMP)",
                              fontsize=10)
            axes[0].set_ylim(0.4, 1.0); axes[0].legend(); axes[0].grid(alpha=0.3)
        if op is not None:
            xs = np.arange(len(op["test_acc"]))
            axes[1].plot(xs, np.array(op["test_acc"]) * 100, "s-", color="#2ca02c",
                         label="`:` cell")
            axes[1].axhline(op["baseline"] * 100, color="gray", ls=":",
                            label=f"baseline {op['baseline']*100:.0f}%")
            axes[1].set_xlabel("layer"); axes[1].set_ylabel("test accuracy (%)")
            axes[1].set_title(f"4-class operator probe @ ':' (SVAMP)", fontsize=10)
            axes[1].set_ylim(0, 100); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.suptitle("Probes at the ':' residual on SVAMP",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 3: PCA bifurcation at colon ===
        if pca is not None:
            corr = np.array(pca["abs_corr_per_layer_pc_feature"])  # (L, 3, F)
            names = pca["features"]
            mat = corr[:, 0, :]  # PC1 correlations per layer per feature
            fig, ax = plt.subplots(figsize=(13, 5))
            im = ax.imshow(mat.T, aspect="auto", origin="lower",
                           cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(mat.shape[0]))
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("layer")
            ax.set_title(f"|r(PC1, feature)| at ':' residual per layer "
                         f"(best: layer {pca['best_layer']}, feature={pca['best_feature']}, "
                         f"|r|={pca['best_corr']:.3f})", fontsize=11, fontweight="bold")
            for l in range(mat.shape[0]):
                for f in range(len(names)):
                    v = mat[l, f]
                    if v >= 0.4:
                        ax.text(l, f, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.6 else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 4+: per-CF dataset operator-probe accuracy ===
        cf_results = {}
        for ds in CF_DATASETS:
            a, m = load_acts_meta(ds)
            if a is None:
                cf_results[ds] = None; continue
            r = operator_acc_per_layer(a, m["types"])
            if r is not None:
                cf_results[ds] = r

        cf_avail = [ds for ds in CF_DATASETS
                    if cf_results.get(ds) is not None
                    and "accs" in cf_results[ds]]
        cf_single = [ds for ds in CF_DATASETS
                     if cf_results.get(ds) is not None
                     and "single_class" in cf_results[ds]]
        if cf_avail:
            n_plots = len(cf_avail)
            ncols = 4
            nrows = int(np.ceil(n_plots / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
            for i, ds in enumerate(cf_avail):
                ax = axes.ravel()[i] if n_plots > 1 else axes
                r = cf_results[ds]
                xs = np.arange(len(r["accs"]))
                ax.plot(xs, np.array(r["accs"]) * 100, "s-", color="#1f77b4")
                ax.axhline(r["baseline"] * 100, color="gray", ls=":")
                ax.set_title(f"{ds}  (n={r['n']}; base={r['baseline']*100:.0f}%)",
                             fontsize=9)
                ax.set_xlabel("layer", fontsize=8)
                ax.set_ylabel("op acc", fontsize=8)
                ax.set_ylim(0, 100); ax.grid(alpha=0.3)
            for j in range(n_plots, nrows * ncols):
                axes.ravel()[j].axis("off")
            fig.suptitle("Operator probe accuracy per layer at ':' residual — per CF dataset",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Synthesis ===
        lines = ["# Headline numbers"]
        if cp is not None:
            lines.append(f"- correctness probe @ ':' : best layer {cp['best_layer']}, "
                         f"test AUC {cp['best_test_auc']:.3f}, acc {cp['best_test_acc']*100:.1f}% "
                         f"(baseline {cp['baseline']*100:.1f}%).")
        if op is not None:
            lines.append(f"- operator probe @ ':' : best layer {op['best_layer']}, "
                         f"test acc {op['best_test_acc']*100:.1f}% "
                         f"(baseline {op['baseline']*100:.1f}%).")
        if pca is not None:
            lines.append(f"- PCA bifurcation @ ':' : PC1 most correlated with "
                         f"`{pca['best_feature']}` at layer {pca['best_layer']} "
                         f"(|r|={pca['best_corr']:.3f}).")
        lines.append("")
        lines.append("# Counterfactual datasets")
        if cf_avail:
            lines.append(f"- ':' residuals captured for {len(cf_avail)} CF datasets.")
            for ds in cf_avail[:6]:
                r = cf_results[ds]
                best_layer = int(np.argmax(r["accs"]))
                lines.append(f"  - {ds}: best op-probe acc {r['accs'][best_layer]*100:.1f}% at layer {best_layer} (n={r['n']}, baseline {r['baseline']*100:.0f}%)")
        text_slide(pdf, "Synthesis — ':' residual on CODI-GPT-2", lines)

    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
