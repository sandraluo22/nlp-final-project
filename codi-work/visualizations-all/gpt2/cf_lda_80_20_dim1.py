"""Per-LD-axis 1D analyses with multiple class-grouping evaluations.

For each (layer, latent_step) we fit LDA(n_components=3) supervised on the
4-class problem-type label, then *project* onto each LD axis individually
(LD1 only, LD2 only, LD3 only) and treat that single axis as a 1D feature.

For every (LD axis × grouping scheme), we score four classifications:
  - indiv : 4-class Addition / Subtraction / Multiplication / Common-Division
  - AS|M|D: 3-class {Add+Sub} / Multiplication / Common-Division
  - AS|MD : 2-class {Add+Sub} / {Mul+Div}
  - AM|SD : 2-class {Add+Mul} / {Sub+Div}

Slideshow ordering (102 slides + 2 summary):
  - Summary slide 1: accuracy vs layer per (grouping, dataset), 3 lines (LD1/2/3)
  - Summary slide 2: per-class P/R/F1 + confusion matrix at peak
  - LD1 held-out CF (17), LD1 orig SVAMP (17),
    LD2 held-out CF (17), LD2 orig SVAMP (17),
    LD3 held-out CF (17), LD3 orig SVAMP (17)

Each subplot is one latent step, with a 4-class-colored histogram along that LD
axis. Subplot title carries all four grouping accuracies for that step.
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parents[2]
CF_ACTS = REPO / "inference" / "runs" / "cf_balanced_student_gpt2" / "activations.pt"
CF_DATA = REPO.parent / "cf-datasets" / "cf_balanced.json"
ORIG_ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
OUT_PDF = REPO / "visualizations-all" / "gpt2" / "cf_lda_80_20_dim1.pdf"
OUT_STATS = REPO / "visualizations-all" / "gpt2" / "cf_lda_80_20_dim1_stats.json"

PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}
SEED = 13
LD_AXES = [0, 1, 2]  # 0-indexed: LD1, LD2, LD3
LD_NAMES = {0: "LD1", 1: "LD2", 2: "LD3"}

# Grouping schemes: {scheme_name: {original_class: grouped_class}}
GROUPINGS: dict[str, dict[str, str]] = {
    "indiv": {
        "Subtraction": "Subtraction",
        "Addition": "Addition",
        "Multiplication": "Multiplication",
        "Common-Division": "Common-Division",
    },
    "AS|M|D": {
        "Subtraction": "AddSub",
        "Addition": "AddSub",
        "Multiplication": "Multiplication",
        "Common-Division": "Common-Division",
    },
    "AS|MD": {
        "Subtraction": "AddSub",
        "Addition": "AddSub",
        "Multiplication": "MulDiv",
        "Common-Division": "MulDiv",
    },
    "AM|SD": {
        "Subtraction": "SubDiv",
        "Addition": "AddMul",
        "Multiplication": "AddMul",
        "Common-Division": "SubDiv",
    },
}
GROUPING_ORDER = ["indiv", "AS|M|D", "AS|MD", "AM|SD"]


def map_labels(types: np.ndarray, mapping: dict[str, str]) -> np.ndarray:
    return np.array([mapping[t] for t in types])


def majority_chance(labels: np.ndarray) -> float:
    c = Counter(labels)
    return max(c.values()) / len(labels)


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


def fit_eval(
    cf_acts: np.ndarray,
    cf_types: np.ndarray,
    orig_acts: np.ndarray,
    orig_types: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
):
    """Fit LDA(n_components=3) per (layer, step), then evaluate each (LD axis,
    grouping) combination. Returns nested dict
    results[ld_axis][grouping_name][which] where which ∈ {acc_train, acc_cf_test,
    acc_orig, var_ratio, projection}."""
    N_cf, S, L, _H = cf_acts.shape
    n_test = len(idx_test)

    proj_cf_test = np.empty((L, S, n_test, 3), dtype=np.float32)
    proj_orig = np.empty((L, S, orig_acts.shape[0], 3), dtype=np.float32)
    var_ratio = np.empty((L, S, 3), dtype=np.float32)

    # Per (LD axis, grouping) accuracies, shape (L, S)
    accs: dict[int, dict[str, dict[str, np.ndarray]]] = {
        ld: {
            g: {
                "acc_train": np.empty((L, S), dtype=np.float32),
                "acc_cf_test": np.empty((L, S), dtype=np.float32),
                "acc_orig": np.empty((L, S), dtype=np.float32),
            }
            for g in GROUPING_ORDER
        }
        for ld in LD_AXES
    }

    print(f"fitting {L*S} LDA(3) and scoring 3 axes × 4 groupings", flush=True)
    for layer in range(L):
        for step in range(S):
            X_train = cf_acts[idx_train, step, layer, :]
            y_train = cf_types[idx_train]
            X_test = cf_acts[idx_test, step, layer, :]
            y_test = cf_types[idx_test]
            X_orig = orig_acts[:, step, layer, :]

            lda_proj = LinearDiscriminantAnalysis(n_components=3, solver="svd")
            P_train = lda_proj.fit_transform(X_train, y_train)
            P_test = lda_proj.transform(X_test)
            P_orig = lda_proj.transform(X_orig)

            proj_cf_test[layer, step] = P_test
            proj_orig[layer, step] = P_orig
            var_ratio[layer, step] = lda_proj.explained_variance_ratio_[:3]

            for ld in LD_AXES:
                p_tr = P_train[:, ld : ld + 1]
                p_te = P_test[:, ld : ld + 1]
                p_or = P_orig[:, ld : ld + 1]
                for g in GROUPING_ORDER:
                    mapping = GROUPINGS[g]
                    y_tr_g = map_labels(y_train, mapping)
                    y_te_g = map_labels(y_test, mapping)
                    y_or_g = map_labels(orig_types, mapping)
                    clf = LinearDiscriminantAnalysis(solver="svd")
                    clf.fit(p_tr, y_tr_g)
                    accs[ld][g]["acc_train"][layer, step] = (clf.predict(p_tr) == y_tr_g).mean()
                    accs[ld][g]["acc_cf_test"][layer, step] = (clf.predict(p_te) == y_te_g).mean()
                    accs[ld][g]["acc_orig"][layer, step] = (clf.predict(p_or) == y_or_g).mean()

        if layer % 4 == 0 or layer == L - 1:
            # Summarize the indiv (4-class) accuracy per LD as a sanity print.
            ld_summary = "  ".join(
                f"LD{ld+1}={accs[ld]['indiv']['acc_cf_test'][layer].mean()*100:5.1f}%"
                for ld in LD_AXES
            )
            print(f"  layer {layer:>2d}/{L-1}  (cf_test, indiv 4-class) {ld_summary}",
                  flush=True)

    return {
        "accs": accs,
        "proj_cf_test": proj_cf_test,
        "proj_orig": proj_orig,
        "var_ratio": var_ratio,
    }


# ---------------------------------------------------------------------------
# Summary slides
# ---------------------------------------------------------------------------

def chance_for_grouping(grouping: str, types: np.ndarray) -> float:
    return majority_chance(map_labels(types, GROUPINGS[grouping]))


def make_summary_line_plots(
    pdf, accs, cf_types, orig_types, types_test, L,
):
    """One slide per grouping: 1x2 panels (CF held-out, original SVAMP),
    each with 3 lines (one per LD axis) plus a chance baseline."""
    for g in GROUPING_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        panels = [
            ("acc_cf_test", "Held-out 20% CF", types_test),
            ("acc_orig",    "Original SVAMP (transferred)", orig_types),
        ]
        for ax, (key, title, ref_types) in zip(axes, panels):
            chance = chance_for_grouping(g, ref_types)
            for ld in LD_AXES:
                mean_acc = accs[ld][g][key].mean(axis=1) * 100
                ax.plot(range(L), mean_acc, marker="o", linewidth=1.5,
                        label=LD_NAMES[ld])
            ax.axhline(chance * 100, color="gray", linestyle="--", linewidth=1,
                       label=f"chance ({chance*100:.1f}%)")
            ax.set_xticks(range(L))
            ax.set_xlabel("layer (0 = embedding, 1..16 = decoder blocks)", fontsize=10)
            ax.set_ylabel("accuracy (%)", fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.set_ylim(40, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc="lower right")
        fig.suptitle(
            f"Single-LD-axis accuracy vs layer  —  grouping = {g}\n"
            "(LDA basis fit on 4-class operator; eval labels remapped per grouping)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140)
        plt.close(fig)


def make_pr_slide(
    pdf, cf_acts, cf_types, orig_acts, orig_types, idx_train, idx_test,
    accs, types_test, L,
):
    """Find peak (LD axis, layer, step) on indiv 4-class CF held-out, then run
    the full P/R + confusion matrix sanity check there."""
    best = None
    best_acc = -1.0
    for ld in LD_AXES:
        m = accs[ld]["indiv"]["acc_cf_test"]
        idx = np.unravel_index(np.argmax(m), m.shape)
        if m[idx] > best_acc:
            best_acc = float(m[idx])
            best = (ld, int(idx[0]), int(idx[1]))
    peak_ld, peak_layer, peak_step = best
    print(
        f"\n[sanity] peak (indiv) on CF held-out: "
        f"{LD_NAMES[peak_ld]} layer={peak_layer} step={peak_step+1} "
        f"acc={best_acc:.3f}",
        flush=True,
    )

    X_tr = cf_acts[idx_train, peak_step, peak_layer, :]
    y_tr = cf_types[idx_train]
    X_te = cf_acts[idx_test,  peak_step, peak_layer, :]
    y_te = cf_types[idx_test]
    X_or = orig_acts[:,        peak_step, peak_layer, :]

    lda_proj = LinearDiscriminantAnalysis(n_components=3, solver="svd")
    P_tr = lda_proj.fit_transform(X_tr, y_tr)[:, peak_ld : peak_ld + 1]
    P_te = lda_proj.transform(X_te)[:, peak_ld : peak_ld + 1]
    P_or = lda_proj.transform(X_or)[:, peak_ld : peak_ld + 1]
    clf = LinearDiscriminantAnalysis(solver="svd")
    clf.fit(P_tr, y_tr)
    yhat_te = clf.predict(P_te)
    yhat_or = clf.predict(P_or)

    classes = ["Addition", "Common-Division", "Multiplication", "Subtraction"]
    rep_te = classification_report(
        y_te, yhat_te, labels=classes, digits=3, zero_division=0, output_dict=True
    )
    rep_or = classification_report(
        orig_types, yhat_or, labels=classes, digits=3, zero_division=0,
        output_dict=True,
    )
    cm_te = confusion_matrix(y_te, yhat_te, labels=classes)
    cm_or = confusion_matrix(orig_types, yhat_or, labels=classes)

    for name, rep in [("CF held-out 20%", rep_te), ("Original SVAMP", rep_or)]:
        print(f"\n  {name}: accuracy={rep['accuracy']:.3f}  "
              f"macro-F1={rep['macro avg']['f1-score']:.3f}")
        print(f"    {'class':<18s} {'P':>6s} {'R':>6s} {'F1':>6s} {'support':>8s}")
        for c in classes:
            r = rep[c]
            print(f"    {c:<18s} {r['precision']:6.3f} {r['recall']:6.3f} "
                  f"{r['f1-score']:6.3f} {int(r['support']):>8d}")

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.25)

    def draw_table(ax, rep, title):
        ax.axis("off")
        rows = [
            [c, f"{rep[c]['precision']:.3f}", f"{rep[c]['recall']:.3f}",
             f"{rep[c]['f1-score']:.3f}", f"{int(rep[c]['support'])}"]
            for c in classes
        ] + [
            ["macro avg",
             f"{rep['macro avg']['precision']:.3f}",
             f"{rep['macro avg']['recall']:.3f}",
             f"{rep['macro avg']['f1-score']:.3f}",
             f"{int(rep['macro avg']['support'])}"],
            ["weighted avg",
             f"{rep['weighted avg']['precision']:.3f}",
             f"{rep['weighted avg']['recall']:.3f}",
             f"{rep['weighted avg']['f1-score']:.3f}",
             f"{int(rep['weighted avg']['support'])}"],
        ]
        col_labels = ["class", "precision", "recall", "F1", "n"]
        tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                       cellLoc="center", colLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)
        ax.set_title(f"{title}  (overall acc {rep['accuracy']:.3f})",
                     fontsize=11)

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
                        color="white" if v > cm.max() / 2 else "black",
                        fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    draw_table(fig.add_subplot(gs[0, 0]), rep_te, "Held-out 20% CF (4-class)")
    draw_table(fig.add_subplot(gs[0, 1]), rep_or, "Original SVAMP (4-class)")
    draw_cm(fig.add_subplot(gs[1, 0]), cm_te, "Confusion: held-out 20% CF")
    draw_cm(fig.add_subplot(gs[1, 1]), cm_or, "Confusion: original SVAMP")
    fig.suptitle(
        f"Sanity check  —  per-class P / R / F1 + confusion matrix\n"
        f"peak config: {LD_NAMES[peak_ld]} only, layer={peak_layer}, "
        f"latent step={peak_step+1}  (4-class indiv)",
        fontsize=12,
    )
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-(LD, dataset, layer) detail slides
# ---------------------------------------------------------------------------

def render_layer_slide(
    pdf,
    layer: int,
    proj_for_layer: np.ndarray,  # (S, n, 3)
    var_for_layer: np.ndarray,   # (S, 3)
    accs_for_layer: dict[str, np.ndarray],  # acc per grouping at this LD, shape (S,)
    types: np.ndarray,
    ld: int,
    title_prefix: str,
):
    S = proj_for_layer.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.875))
    fig.suptitle(
        f"Layer {layer}  —  {LD_NAMES[ld]} only  —  {title_prefix}",
        fontsize=12,
    )
    legend_proxies: list[Line2D] = []
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off")
            continue
        xy = proj_for_layer[s, :, ld]
        for cls, color in PROBLEM_TYPE_COLORS.items():
            mask = types == cls
            if mask.sum() == 0:
                continue
            ax.hist(xy[mask], bins=35, density=True, alpha=0.45, color=color,
                    edgecolor="none")
        ax.set_yticks([])
        ax.set_xlabel(
            f"{LD_NAMES[ld]} ({var_for_layer[s, ld]*100:.1f}%)", fontsize=8
        )
        title_lines = (
            f"step {s+1}  |  "
            f"indiv {accs_for_layer['indiv'][s]*100:.1f}%  |  "
            f"AS|M|D {accs_for_layer['AS|M|D'][s]*100:.1f}%\n"
            f"AS|MD {accs_for_layer['AS|MD'][s]*100:.1f}%  |  "
            f"AM|SD {accs_for_layer['AM|SD'][s]*100:.1f}%"
        )
        ax.set_title(title_lines, fontsize=8)

        if s == 0 and not legend_proxies:
            for cls, color in PROBLEM_TYPE_COLORS.items():
                n = (types == cls).sum()
                legend_proxies.append(
                    Line2D([0], [0], marker="s", linestyle="", color=color,
                           label=f"{cls} ({n})")
                )
    if legend_proxies:
        fig.legend(handles=legend_proxies, loc="lower center",
                   ncol=len(legend_proxies), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.04, right=0.98,
                        wspace=0.18, hspace=0.55)
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
    print(f"CF split: train={len(idx_train)} test={len(idx_test)}", flush=True)

    res = fit_eval(cf_acts, cf_types, orig_acts, orig_types, idx_train, idx_test)
    L = res["proj_cf_test"].shape[0]

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {OUT_PDF}", flush=True)
    with PdfPages(OUT_PDF) as pdf:
        make_summary_line_plots(
            pdf, res["accs"], cf_types, orig_types, types_test, L,
        )
        make_pr_slide(
            pdf, cf_acts, cf_types, orig_acts, orig_types,
            idx_train, idx_test, res["accs"], types_test, L,
        )

        # Detail slides: outer loop = LD axis, inner = (CF held-out, then orig).
        for ld in LD_AXES:
            # held-out 20% CF
            for layer in range(L):
                accs_layer = {
                    g: res["accs"][ld][g]["acc_cf_test"][layer]
                    for g in GROUPING_ORDER
                }
                render_layer_slide(
                    pdf, layer, res["proj_cf_test"][layer], res["var_ratio"][layer],
                    accs_layer, types_test, ld,
                    title_prefix="held-out 20% CF",
                )
            # original SVAMP
            for layer in range(L):
                accs_layer = {
                    g: res["accs"][ld][g]["acc_orig"][layer]
                    for g in GROUPING_ORDER
                }
                render_layer_slide(
                    pdf, layer, res["proj_orig"][layer], res["var_ratio"][layer],
                    accs_layer, orig_types, ld,
                    title_prefix="original SVAMP (transferred)",
                )
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")

    # Save numeric stats
    stats: dict = {
        "by_ld": {
            f"LD{ld+1}": {
                g: {
                    "acc_train":   res["accs"][ld][g]["acc_train"].tolist(),
                    "acc_cf_test": res["accs"][ld][g]["acc_cf_test"].tolist(),
                    "acc_orig":    res["accs"][ld][g]["acc_orig"].tolist(),
                }
                for g in GROUPING_ORDER
            }
            for ld in LD_AXES
        },
        "var_ratio": res["var_ratio"].tolist(),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"saved stats -> {OUT_STATS}")

    # Print mean-across-steps table per LD per grouping
    print("\nMean accuracy across 6 latent steps  (cf-test / orig)")
    for g in GROUPING_ORDER:
        chance_cf = chance_for_grouping(g, types_test)
        chance_or = chance_for_grouping(g, orig_types)
        print(f"\n--- grouping = {g}  (chance: cf={chance_cf*100:.1f}%  "
              f"orig={chance_or*100:.1f}%) ---")
        print("layer  " + "  ".join(f"{LD_NAMES[ld]:>14s}" for ld in LD_AXES))
        for layer in range(L):
            row_parts = []
            for ld in LD_AXES:
                cf_acc = res["accs"][ld][g]["acc_cf_test"][layer].mean() * 100
                or_acc = res["accs"][ld][g]["acc_orig"][layer].mean() * 100
                row_parts.append(f"{cf_acc:5.1f}/{or_acc:5.1f}%")
            print(f" {layer:>3d}    " + "  ".join(f"{p:>14s}" for p in row_parts))


if __name__ == "__main__":
    main()
