"""PCA bifurcation diagnostic at the ':' residual on SVAMP (CODI-GPT-2).

Per layer, fits PCA-3 and reports |Pearson r| of each PC with: log(answer+1),
parity, digit_count, answer_value, problem-type indicators, prediction
correctness. Identifies which feature drives the bifurcation at each layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.decomposition import PCA

PD = Path(__file__).resolve().parent
ACTS = PD / "svamp_colon_acts.pt"
META = PD / "svamp_colon_acts_meta.json"
OUT_PDF = PD / "pca_bifurcation_colon.pdf"
OUT_JSON = PD / "pca_bifurcation_colon.json"


def safe_corr(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(META))
    N, L, H = a.shape
    print(f"colon acts shape={a.shape}")

    # Features
    types = np.array(meta["types"])
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    pred = np.array([np.nan if v is None else float(v) for v in meta["pred_int_extracted"]])
    student_correct = ((~np.isnan(pred)) & (~np.isnan(gold))
                       & (np.abs(pred - gold) < 1e-3)).astype(float)
    log_answer = np.log10(np.where(np.isnan(gold), 1.0, np.maximum(gold, 1)) + 1)
    parity = np.where(np.isnan(gold), 0, gold.astype(int) % 2).astype(float)
    digits = np.where(np.isnan(gold), 1, np.floor(np.log10(np.maximum(np.abs(gold), 1))).astype(int) + 1).astype(float)
    ans_abs = np.abs(np.where(np.isnan(gold), 0, gold))

    feats = [
        ("log10(ans+1)", log_answer),
        ("parity", parity),
        ("digit_count", digits),
        ("ans_abs", ans_abs),
        ("student_correct", student_correct),
    ]
    for op in ["Addition", "Subtraction", "Multiplication", "Common-Division"]:
        feats.append((f"type={op}", (types == op).astype(float)))
    names = [n for n, _ in feats]
    F = np.stack([v for _, v in feats], axis=1)

    pca_xyz = np.zeros((L, N, 3), dtype=np.float32)
    var_ratio = np.zeros((L, 3), dtype=np.float32)
    print("fitting PCA-3 per layer...")
    for l in range(L):
        X = a[:, l, :]
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
        pca_xyz[l] = pca.fit_transform(X)
        var_ratio[l] = pca.explained_variance_ratio_

    corr = np.zeros((L, 3, len(names)))
    for l in range(L):
        for pc in range(3):
            for i, name in enumerate(names):
                corr[l, pc, i] = abs(safe_corr(pca_xyz[l, :, pc], F[:, i]))

    pc1_iqr = (np.percentile(pca_xyz[:, :, 0], 75, axis=-1)
               - np.percentile(pca_xyz[:, :, 0], 25, axis=-1))
    l_best = int(corr[:, 0, :].max(axis=-1).argmax())
    feat_best = names[corr[l_best, 0, :].argmax()]
    print(f"strongest PC1-feature correlation: layer={l_best}, feature={feat_best}, "
          f"|r|={corr[l_best, 0, :].max():.3f}")

    print("\nLayer | top PC1 feature | |r|")
    for l in range(L):
        i = corr[l, 0, :].argmax()
        print(f"  {l:>2} | {names[i]:<22} {corr[l, 0, i]:.3f}  (IQR={pc1_iqr[l]:.1f})")

    OUT_JSON.write_text(json.dumps({
        "shape": [N, L, H], "features": names,
        "abs_corr_per_layer_pc_feature": corr.tolist(),
        "pc_var_ratio": var_ratio.tolist(),
        "pc1_iqr": pc1_iqr.tolist(),
        "best_layer": l_best, "best_feature": feat_best,
        "best_corr": float(corr[l_best, 0, :].max()),
    }, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    # 1: heatmap of |r(PC1, feat)| per (layer, feat)
    ax = axes[0, 0]
    mat = corr[:, 0, :]
    im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(L)); ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("layer")
    ax.set_title("|r(PC1, feature)| at ':' residual")
    for l in range(L):
        for f in range(len(names)):
            v = mat[l, f]
            if v >= 0.4:
                ax.text(l, f, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # 2: PC1 IQR per layer
    ax = axes[0, 1]
    ax.bar(range(L), pc1_iqr, color="#3357aa")
    ax.set_xlabel("layer"); ax.set_ylabel("PC1 IQR")
    ax.set_title("PC1 IQR at ':' residual (bifurcation proxy)")

    # 3-4: scatter at l_best, colored by student_correct and by log(answer)
    xy = pca_xyz[l_best]
    for j, (title, vals, cmap) in enumerate([
        ("student correct", student_correct, "viridis"),
        ("log10(answer+1)", log_answer, "viridis"),
    ]):
        ax = axes[1, j]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap, s=6, alpha=0.6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"PCA at layer {l_best} (':' residual), colored by {title}")
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(f"PCA bifurcation diagnostic at ':' residual (SVAMP N={N})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PDF, dpi=140)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
