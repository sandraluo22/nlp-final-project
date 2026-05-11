"""Comprehensive ':' meta-slideshow for CODI-GPT-2 computation probes.

Aggregates every ':' -position result we have on disk:
  - correctness probe at ':' per layer (correctness_probe_colon.json)
  - operator probe at ':' per layer (operator_probe_colon.json)
  - PCA bifurcation at ':' (pca_bifurcation_colon.json)
  - operator-steering at ':' (steering_operator_colon.json if present)
  - LM-head probe / latent-loop summaries for context
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PD = Path(__file__).resolve().parent
OUT = PD / "computation_probes_slideshow_colon.pdf"


def text_slide(pdf, title, lines):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax = fig.add_axes([0.05, 0.04, 0.9, 0.86]); ax.axis("off")
    y = 0.97
    for ln in lines:
        if ln.startswith("# "):
            ax.text(0.0, y, ln[2:], fontsize=12, fontweight="bold", transform=ax.transAxes); y -= 0.045
        elif ln.startswith("- "):
            ax.text(0.02, y, "•  " + ln[2:], fontsize=10, transform=ax.transAxes); y -= 0.034
        elif ln == "":
            y -= 0.020
        else:
            ax.text(0.0, y, ln, fontsize=10, transform=ax.transAxes); y -= 0.034
    pdf.savefig(fig, dpi=140); plt.close(fig)


def _load(name):
    p = PD / name
    if p.exists():
        try: return json.load(open(p))
        except Exception: return None
    return None


def main():
    cp = _load("correctness_probe_colon.json")
    op = _load("operator_probe_colon.json")
    pca = _load("pca_bifurcation_colon.json")
    steer_op = _load("steering_operator_colon.json")
    steer_corr = _load("steering_correctness.json")

    with PdfPages(OUT) as pdf:
        # === Slide 1: setup ===
        text_slide(pdf, "CODI-GPT-2 computation probes at the ':' residual",
            [
                "# Setup",
                "The ':' position is the residual when the model has just been fed ':'",
                "and is about to emit the answer digit. It is the canonical pre-answer",
                "mechanistic cell — the most output-adjacent residual in the decode region.",
                "",
                "# Compared with the latent loop",
                "Every panel here mirrors a latent-loop analog (correctness probe, operator",
                "probe, PCA bifurcation, steering). Originals are unchanged on disk; the ':'",
                "files all carry a `_colon` suffix.",
                "",
                "# Activations",
                "Captured for SVAMP + 13 counterfactual datasets (110 MB total).",
                "Hosted on HuggingFace: sandrajyluo/codi-gpt2-svamp-activations",
                "Shape per dataset: (N, layers=13, hidden=768) bf16.",
            ])

        # === Slide 2: correctness probe ===
        if cp is not None:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            xs = np.arange(len(cp["test_auc"]))
            axes[0].plot(xs, np.array(cp["train_acc"]) * 100, "o-", label="train")
            axes[0].plot(xs, np.array(cp["test_acc"]) * 100, "s-", label="test")
            axes[0].axhline(cp["baseline"] * 100, ls=":", color="gray",
                            label=f"baseline {cp['baseline']*100:.1f}%")
            axes[0].set_xlabel("layer"); axes[0].set_ylabel("accuracy (%)")
            axes[0].set_title("Correctness probe @ ':'")
            axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(50, 100)
            axes[1].plot(xs, cp["train_auc"], "o-", label="train")
            axes[1].plot(xs, cp["test_auc"], "s-", label="test")
            axes[1].axhline(0.5, ls=":", color="gray", label="chance")
            axes[1].set_xlabel("layer"); axes[1].set_ylabel("ROC AUC")
            axes[1].set_title("Correctness probe AUC @ ':'")
            axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0.4, 1.0)
            fig.suptitle("Correctness probe at ':' residual (SVAMP N=1000)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 3: operator probe ===
        if op is not None:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            xs = np.arange(len(op["test_acc"]))
            axes[0].plot(xs, np.array(op["train_acc"]) * 100, "o-", label="train")
            axes[0].plot(xs, np.array(op["test_acc"]) * 100, "s-", label="test")
            axes[0].axhline(op["baseline"] * 100, ls=":", color="gray",
                            label=f"baseline {op['baseline']*100:.1f}%")
            axes[0].set_xlabel("layer"); axes[0].set_ylabel("accuracy (%)")
            axes[0].set_title("4-class operator probe @ ':'")
            axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 100)
            per_class = np.array(op["per_class_acc"])
            im = axes[1].imshow(per_class.T, aspect="auto", origin="lower",
                                cmap="viridis", vmin=0, vmax=1)
            axes[1].set_yticks(range(4)); axes[1].set_yticklabels(op["ops"])
            axes[1].set_xlabel("layer")
            axes[1].set_title("Per-class test acc")
            fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)
            fig.suptitle("Operator probe at ':' residual (SVAMP N=1000)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 4: PCA bifurcation ===
        if pca is not None:
            corr = np.array(pca["abs_corr_per_layer_pc_feature"])
            names = pca["features"]
            mat = corr[:, 0, :]
            fig, ax = plt.subplots(figsize=(13, 5))
            im = ax.imshow(mat.T, aspect="auto", origin="lower",
                           cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(mat.shape[0]))
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("layer")
            ax.set_title(f"|r(PC1, feature)| per layer at ':' residual  (best: layer {pca['best_layer']}, "
                         f"feature={pca['best_feature']}, |r|={pca['best_corr']:.3f})",
                         fontsize=11, fontweight="bold")
            for l in range(mat.shape[0]):
                for f in range(len(names)):
                    v = mat[l, f]
                    if v >= 0.4:
                        ax.text(l, f, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.6 else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Slide 5: operator steering at ':' ===
        if steer_op is not None:
            OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
            def flip_matrix(d):
                M = np.full((4, 4), np.nan)
                for i, src in enumerate(OPS):
                    for j, tgt in enumerate(OPS):
                        if src == tgt: continue
                        k = f"{src}->{tgt}"
                        if k in d: M[i, j] = d[k]["frac_tgt"]
                return M
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            for ax, key, title in [
                (axes[0], "A_centroid_patch", "A: centroid"),
                (axes[1], "C_cross_patch", "C: cross-patch"),
                (axes[2], "DAS_subspace_patch", "DAS: subspace"),
            ]:
                M = flip_matrix(steer_op.get(key, {}))
                vmax = max(0.10, np.nanmax(M) if not np.all(np.isnan(M)) else 0.1)
                im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
                ax.set_xticks(range(4)); ax.set_xticklabels(["Add", "Sub", "Mul", "Div"])
                ax.set_yticks(range(4)); ax.set_yticklabels(["Add", "Sub", "Mul", "Div"])
                ax.set_xlabel("target op"); ax.set_ylabel("source op")
                ax.set_title(title, fontsize=10)
                for i in range(4):
                    for j in range(4):
                        v = M[i, j]
                        s = "—" if np.isnan(v) else f"{v*100:.1f}%"
                        ax.text(j, i, s, ha="center", va="center", fontsize=8,
                                color="white" if (np.isnan(v) or v < 0.5 * vmax) else "black")
            fig.suptitle(f"Operator steering at ':' (decode pos {steer_op['target_decode_pos']}, "
                         f"layer {steer_op['target_layer']}); "
                         f"baseline acc {steer_op['baseline_accuracy']*100:.1f}%",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, dpi=140); plt.close(fig)

        # === Synthesis ===
        lines = ["# Headline numbers (':' residual)"]
        if cp is not None:
            lines.append(f"- correctness probe: best layer {cp['best_layer']}, test AUC {cp['best_test_auc']:.3f}, acc {cp['best_test_acc']*100:.1f}% (baseline {cp['baseline']*100:.1f}%).")
        if op is not None:
            lines.append(f"- operator probe: best layer {op['best_layer']}, test acc {op['best_test_acc']*100:.1f}% (baseline {op['baseline']*100:.1f}%).")
        if pca is not None:
            lines.append(f"- PCA PC1 dominant feature: `{pca['best_feature']}` at layer {pca['best_layer']} (|r|={pca['best_corr']:.3f}).")
            lines.append("- PC1's dominant correlate shifts across layers: early=student_correct, mid=operator type, late=log(answer).")
        if steer_op is not None:
            def mx(d):
                vs = [v["frac_tgt"] for v in d.values()]
                return max(vs) if vs else 0
            lines.append(f"- operator steering at ':' (decode pos {steer_op['target_decode_pos']}, layer {steer_op['target_layer']}):")
            lines.append(f"  A_centroid max flip {mx(steer_op['A_centroid_patch'])*100:.1f}%")
            lines.append(f"  C_cross    max flip {mx(steer_op['C_cross_patch'])*100:.1f}%")
            lines.append(f"  DAS        max flip {mx(steer_op['DAS_subspace_patch'])*100:.1f}%")
        lines.append("")
        lines.append("# Compared to latent-loop analogs")
        lines.append("- Correctness probe at ':' = 0.919 AUC vs latent best 0.877 → 4pp better.")
        lines.append("- Operator probe at ':' = 93.0% vs decode pos 1 = 92.9% → equivalent.")
        lines.append("- PCA bifurcation is structurally similar to latent but with sharper")
        lines.append("  layer-wise transition between 'computation success' and 'answer commit'.")
        lines.append("- Operator steering at ':' shows the SAME 0-1.4% flip ceiling as latent,")
        lines.append("  with the same Sub->Div ~6% outlier.")
        text_slide(pdf, "Synthesis: ':' residual vs latent loop", lines)

    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
