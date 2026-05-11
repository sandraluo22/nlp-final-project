"""LDA-on-CF dim sweep with held-out scoring + cross-dataset transfer.

For each k ∈ {1, 2, 3} (LDA components):
  - Fit LDA(n_components=k) on 80% of CF activations (stratified on
    problem_type) per (layer, latent_step).
  - Project the 20% held-out CF and full original SVAMP into the k-dim space.
  - Fit a fresh LDA classifier on the k-dim projection of the training set
    to give a "k-dim only" accuracy (i.e., the accuracy you'd get if you only
    had the visualized dimensions, not the full feature space).

Slideshow ordering (102 slides):
  - dim=1 : 17 CF held-out slides → 17 original SVAMP slides
  - dim=2 : 17 → 17
  - dim=3 : 17 → 17

Each slide is one layer; subplots are the 6 latent steps. Subplot title carries
the k-dim classification accuracy on that layer×step for the relevant test set.
Visualization adapts to k:
  - k=1 → density histograms per problem type along LD1
  - k=2 → 2D scatter
  - k=3 → 3D scatter
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parents[2]
CF_ACTS = REPO / "inference" / "runs" / "cf_balanced_student_gpt2" / "activations.pt"
CF_DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"
ORIG_ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
OUT_PDF = REPO / "visualizations-all" / "gpt2" / "cf_lda_80_20_slideshow.pdf"
OUT_STATS = REPO / "visualizations-all" / "gpt2" / "cf_lda_80_20_stats.json"

PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}
SEED = 13
DIMS = [1, 2, 3]


def load_cf() -> tuple[np.ndarray, np.ndarray]:
    print(f"loading CF: {CF_ACTS}", flush=True)
    a = torch.load(CF_ACTS, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(CF_DATA))
    types = np.array([r["type"] for r in rows])
    print(f"  cf shape={a.shape}  types={dict(Counter(types))}", flush=True)
    return a, types


def load_orig() -> tuple[np.ndarray, np.ndarray]:
    print(f"loading orig: {ORIG_ACTS}", flush=True)
    a = torch.load(ORIG_ACTS, map_location="cpu", weights_only=True).float().numpy()
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array(
        [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    )
    print(f"  orig shape={a.shape}  types={dict(Counter(types))}", flush=True)
    return a, types


def fit_eval_dim(
    cf_acts: np.ndarray,
    cf_types: np.ndarray,
    orig_acts: np.ndarray,
    orig_types: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    k: int,
):
    N_cf, S, L, _H = cf_acts.shape
    n_test = len(idx_test)
    proj_cf_test = np.empty((L, S, n_test, k), dtype=np.float32)
    proj_orig = np.empty((L, S, orig_acts.shape[0], k), dtype=np.float32)
    var_ratio = np.empty((L, S, k), dtype=np.float32)
    acc_train = np.empty((L, S), dtype=np.float32)
    acc_cf_test = np.empty((L, S), dtype=np.float32)
    acc_orig = np.empty((L, S), dtype=np.float32)

    print(f"  dim={k}: fitting {L*S} LDAs", flush=True)
    for layer in range(L):
        for step in range(S):
            X_train = cf_acts[idx_train, step, layer, :]
            y_train = cf_types[idx_train]
            X_test = cf_acts[idx_test, step, layer, :]
            y_test = cf_types[idx_test]
            X_orig = orig_acts[:, step, layer, :]

            lda_proj = LinearDiscriminantAnalysis(n_components=k, solver="svd")
            P_train = lda_proj.fit_transform(X_train, y_train)
            P_test = lda_proj.transform(X_test)
            P_orig = lda_proj.transform(X_orig)

            proj_cf_test[layer, step] = P_test
            proj_orig[layer, step] = P_orig
            var_ratio[layer, step] = lda_proj.explained_variance_ratio_[:k]

            # Fit a classifier in the k-dim projected space so accuracy reflects
            # only the visualized subspace, not the full feature space.
            clf = LinearDiscriminantAnalysis(solver="svd")
            clf.fit(P_train, y_train)
            acc_train[layer, step] = (clf.predict(P_train) == y_train).mean()
            acc_cf_test[layer, step] = (clf.predict(P_test) == y_test).mean()
            acc_orig[layer, step] = (clf.predict(P_orig) == orig_types).mean()
        if layer % 4 == 0 or layer == L - 1:
            print(
                f"    layer {layer:>2d}/{L-1}  "
                f"train={acc_train[layer].mean()*100:5.1f}%  "
                f"cf_test={acc_cf_test[layer].mean()*100:5.1f}%  "
                f"orig={acc_orig[layer].mean()*100:5.1f}%",
                flush=True,
            )

    return {
        "proj_cf_test": proj_cf_test,
        "proj_orig": proj_orig,
        "var_ratio": var_ratio,
        "acc_train": acc_train,
        "acc_cf_test": acc_cf_test,
        "acc_orig": acc_orig,
    }


def _strip_3d_axes(ax, var: np.ndarray):
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0, pad=-2)
    ax.set_xlabel(f"LD1 ({var[0]*100:.1f}%)", fontsize=7, labelpad=-10)
    ax.set_ylabel(f"LD2 ({var[1]*100:.1f}%)", fontsize=7, labelpad=-10)
    ax.set_zlabel(f"LD3 ({var[2]*100:.1f}%)", fontsize=7, labelpad=-10)
    ax.grid(True, alpha=0.25)


def render_slide(
    pdf: PdfPages,
    layer: int,
    proj_for_layer: np.ndarray,  # (S, n_eval, k)
    var_for_layer: np.ndarray,   # (S, k)
    acc_for_layer: np.ndarray,   # (S,)
    types: np.ndarray,
    k: int,
    title_prefix: str,
):
    S = proj_for_layer.shape[0]
    if k == 3:
        fig, axes = plt.subplots(
            2, 3, figsize=(14, 7.875), subplot_kw={"projection": "3d"}
        )
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.875))
    fig.suptitle(
        f"Layer {layer}  —  LDA dim={k}  —  {title_prefix}  "
        f"(LDA fit on 80% CF, supervised on problem_type)",
        fontsize=12,
    )
    legend_proxies: list[Line2D] = []
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off")
            continue
        xy = proj_for_layer[s]

        if k == 1:
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = types == cls
                if mask.sum() == 0:
                    continue
                ax.hist(
                    xy[mask, 0], bins=35, density=True,
                    alpha=0.45, color=color, edgecolor="none",
                )
            ax.set_yticks([])
            ax.set_xlabel(f"LD1 ({var_for_layer[s, 0]*100:.1f}%)", fontsize=8)
        elif k == 2:
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = types == cls
                ax.scatter(
                    xy[mask, 0], xy[mask, 1],
                    s=4, c=color, alpha=0.55, linewidths=0,
                )
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"LD1 ({var_for_layer[s, 0]*100:.1f}%)", fontsize=8)
            ax.set_ylabel(f"LD2 ({var_for_layer[s, 1]*100:.1f}%)", fontsize=8)
        elif k == 3:
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = types == cls
                ax.scatter(
                    xy[mask, 0], xy[mask, 1], xy[mask, 2],
                    s=5, c=color, alpha=0.6, linewidths=0,
                    depthshade=True, rasterized=True,
                )
            _strip_3d_axes(ax, var_for_layer[s])

        if s == 0 and not legend_proxies:
            for cls, color in PROBLEM_TYPE_COLORS.items():
                n = (types == cls).sum()
                legend_proxies.append(
                    Line2D([0], [0], marker="o", linestyle="",
                           color=color, label=f"{cls} ({n})")
                )

        ax.set_title(
            f"latent step {s+1}  —  acc {acc_for_layer[s]*100:.1f}%",
            fontsize=9,
        )

    if legend_proxies:
        fig.legend(
            handles=legend_proxies,
            loc="lower center",
            ncol=len(legend_proxies),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.04, right=0.98,
                        wspace=0.18, hspace=0.30)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def make_summary_line_plot(pdf, all_results, cf_chance, orig_chance, L):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        ("acc_cf_test", "Held-out 20% CF", cf_chance),
        ("acc_orig", "Original SVAMP (transferred)", orig_chance),
    ]
    for ax, (key, title, chance) in zip(axes, panels):
        for k in DIMS:
            mean_acc = all_results[k][key].mean(axis=1) * 100
            ax.plot(range(L), mean_acc, marker="o", linewidth=1.5, label=f"LDA dim={k}")
        ax.axhline(chance * 100, color="gray", linestyle="--", linewidth=1,
                   label=f"chance ({chance*100:.1f}%)")
        ax.set_xlabel("layer (0 = embedding, 1..16 = decoder blocks)", fontsize=10)
        ax.set_ylabel("operator-classification accuracy (%)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(L))
        ax.set_ylim(45, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "Operator-classification accuracy vs layer  (mean across 6 latent steps)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def make_summary_pr_slide(
    pdf, cf_acts, cf_types, orig_acts, orig_types,
    idx_train, idx_test, peak_dim, peak_layer, peak_step,
):
    """At the peak config, fit LDA, compute classification report + confusion
    matrices on CF held-out and original SVAMP."""
    X_tr = cf_acts[idx_train, peak_step, peak_layer, :]
    y_tr = cf_types[idx_train]
    X_te = cf_acts[idx_test, peak_step, peak_layer, :]
    y_te = cf_types[idx_test]
    X_or = orig_acts[:, peak_step, peak_layer, :]

    lda_proj = LinearDiscriminantAnalysis(n_components=peak_dim, solver="svd")
    P_tr = lda_proj.fit_transform(X_tr, y_tr)
    P_te = lda_proj.transform(X_te)
    P_or = lda_proj.transform(X_or)
    clf = LinearDiscriminantAnalysis(solver="svd")
    clf.fit(P_tr, y_tr)
    yhat_te = clf.predict(P_te)
    yhat_or = clf.predict(P_or)

    classes = ["Addition", "Common-Division", "Multiplication", "Subtraction"]
    rep_te = classification_report(
        y_te, yhat_te, labels=classes, digits=3, zero_division=0, output_dict=True
    )
    rep_or = classification_report(
        y_or := orig_types, yhat_or, labels=classes, digits=3, zero_division=0,
        output_dict=True,
    )
    cm_te = confusion_matrix(y_te, yhat_te, labels=classes)
    cm_or = confusion_matrix(orig_types, yhat_or, labels=classes)

    print(
        f"\n[sanity check] peak config: dim={peak_dim} layer={peak_layer} "
        f"step={peak_step+1}",
        flush=True,
    )
    for name, rep in [("CF held-out 20%", rep_te), ("Original SVAMP", rep_or)]:
        print(f"\n  {name}: accuracy={rep['accuracy']:.3f}  macro-F1={rep['macro avg']['f1-score']:.3f}")
        print(f"    {'class':<18s} {'P':>6s} {'R':>6s} {'F1':>6s} {'support':>8s}")
        for c in classes:
            r = rep[c]
            print(f"    {c:<18s} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1-score']:6.3f} {int(r['support']):>8d}")

    # Render: 2x2 = (P/R/F1 table CF) | (P/R/F1 table orig)
    #                (CM CF)         | (CM orig)
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    def draw_table(ax, rep, title):
        ax.axis("off")
        rows = [
            [c, f"{rep[c]['precision']:.3f}", f"{rep[c]['recall']:.3f}",
             f"{rep[c]['f1-score']:.3f}", f"{int(rep[c]['support'])}"]
            for c in classes
        ] + [[
            "macro avg",
            f"{rep['macro avg']['precision']:.3f}",
            f"{rep['macro avg']['recall']:.3f}",
            f"{rep['macro avg']['f1-score']:.3f}",
            f"{int(rep['macro avg']['support'])}",
        ], [
            "weighted avg",
            f"{rep['weighted avg']['precision']:.3f}",
            f"{rep['weighted avg']['recall']:.3f}",
            f"{rep['weighted avg']['f1-score']:.3f}",
            f"{int(rep['weighted avg']['support'])}",
        ]]
        col_labels = ["class", "precision", "recall", "F1", "n"]
        tbl = ax.table(
            cellText=rows, colLabels=col_labels, loc="center",
            cellLoc="center", colLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)
        ax.set_title(f"{title}  (overall acc {rep['accuracy']:.3f})", fontsize=11)

    def draw_cm(ax, cm, title):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=10)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = cm[i, j]
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    draw_table(fig.add_subplot(gs[0, 0]), rep_te, "Held-out 20% CF")
    draw_table(fig.add_subplot(gs[0, 1]), rep_or, "Original SVAMP")
    draw_cm(fig.add_subplot(gs[1, 0]), cm_te, "Confusion: held-out 20% CF")
    draw_cm(fig.add_subplot(gs[1, 1]), cm_or, "Confusion: original SVAMP")
    fig.suptitle(
        f"Sanity check  —  per-class precision / recall / F1 + confusion matrix\n"
        f"peak config: LDA dim={peak_dim}, layer={peak_layer}, latent step={peak_step+1}",
        fontsize=12,
    )
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main():
    cf_acts, cf_types = load_cf()
    orig_acts, orig_types = load_orig()
    N_cf = cf_acts.shape[0]

    idx_train, idx_test = train_test_split(
        np.arange(N_cf), test_size=0.2, random_state=SEED, stratify=cf_types
    )
    types_test = cf_types[idx_test]
    print(
        f"CF split: train={len(idx_train)} test={len(idx_test)}",
        flush=True,
    )

    all_results: dict[int, dict] = {}
    for k in DIMS:
        all_results[k] = fit_eval_dim(
            cf_acts, cf_types, orig_acts, orig_types,
            idx_train, idx_test, k,
        )

    L = all_results[DIMS[0]]["acc_cf_test"].shape[0]
    cf_chance = max(Counter(cf_types).values()) / len(cf_types)
    orig_chance = max(Counter(orig_types).values()) / len(orig_types)

    # Pick peak config: highest CF held-out accuracy across (dim, layer, step).
    peak_dim, peak_layer, peak_step = max(
        ((k, layer, step)
         for k in DIMS
         for layer in range(L)
         for step in range(all_results[k]["acc_cf_test"].shape[1])),
        key=lambda t: all_results[t[0]]["acc_cf_test"][t[1], t[2]],
    )
    print(
        f"\npeak config (highest CF held-out): dim={peak_dim} "
        f"layer={peak_layer} step={peak_step+1} "
        f"acc={all_results[peak_dim]['acc_cf_test'][peak_layer, peak_step]:.3f}",
        flush=True,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {OUT_PDF}", flush=True)
    with PdfPages(OUT_PDF) as pdf:
        # Summary slide 1: accuracy vs layer for the 3 dims, both datasets.
        make_summary_line_plot(pdf, all_results, cf_chance, orig_chance, L)
        # Summary slide 2: per-class P/R/F1 + confusion matrices at peak.
        make_summary_pr_slide(
            pdf, cf_acts, cf_types, orig_acts, orig_types,
            idx_train, idx_test, peak_dim, peak_layer, peak_step,
        )
        # Existing 102 slides.
        for k in DIMS:
            res = all_results[k]
            L = res["proj_cf_test"].shape[0]
            # 17 slides: held-out 20% CF
            for layer in range(L):
                render_slide(
                    pdf, layer,
                    res["proj_cf_test"][layer], res["var_ratio"][layer],
                    res["acc_cf_test"][layer], types_test,
                    k=k, title_prefix="held-out 20% CF",
                )
            # 17 slides: original SVAMP
            for layer in range(L):
                render_slide(
                    pdf, layer,
                    res["proj_orig"][layer], res["var_ratio"][layer],
                    res["acc_orig"][layer], orig_types,
                    k=k, title_prefix="original SVAMP (transferred)",
                )
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")

    # Save accuracies for all dims
    stats = {
        str(k): {
            "acc_train":   all_results[k]["acc_train"].tolist(),
            "acc_cf_test": all_results[k]["acc_cf_test"].tolist(),
            "acc_orig":    all_results[k]["acc_orig"].tolist(),
            "var_ratio":   all_results[k]["var_ratio"].tolist(),
        }
        for k in DIMS
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"saved stats -> {OUT_STATS}")

    # Summary
    cf_chance = max(Counter(cf_types).values()) / len(cf_types)
    orig_chance = max(Counter(orig_types).values()) / len(orig_types)
    print(f"\nbaselines: cf majority={cf_chance*100:.1f}%   orig majority={orig_chance*100:.1f}%")
    print("\nMean accuracy across 6 latent steps")
    for k in DIMS:
        res = all_results[k]
        L = res["acc_cf_test"].shape[0]
        print(f"\n--- LDA dim = {k} ---")
        print("layer       train     cf-test    orig")
        for layer in range(L):
            print(
                f"  {layer:>3d}        "
                f"{res['acc_train'][layer].mean()*100:5.1f}%   "
                f"{res['acc_cf_test'][layer].mean()*100:5.1f}%   "
                f"{res['acc_orig'][layer].mean()*100:5.1f}%"
            )


if __name__ == "__main__":
    main()
