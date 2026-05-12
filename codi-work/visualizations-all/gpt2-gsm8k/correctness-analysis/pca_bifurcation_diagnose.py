"""Diagnose what drives the bifurcation in the per-(layer, step) PCA of
CODI-GPT-2 student activations on SVAMP.

For each (step, layer):
  - Fit PCA-3 on the 1000 student latent activations.
  - For each PC, compute |Pearson r| with several candidate explanatory
    variables: log10(answer+1), answer parity, digit count, problem-type
    one-hots, student correctness, faithful label, |answer|.
  - Pick the (step, layer) with the largest PC1 spread (a proxy for
    bifurcation strength) and render a 6-panel side-by-side comparison:
    plain, log(answer) continuous, parity, digit count, problem type,
    student correctness.

Outputs:
  - pca_bifurcation_diagnose.json  per-(step,layer) correlation table
  - pca_bifurcation_diagnose.pdf   correlation heatmaps + best-slide panel
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset

import re as _gsm_re
def _gsm_first_op(_text):
    _SYM = {"+":"Addition","-":"Subtraction","*":"Multiplication","/":"Common-Division"}
    for _expr, _ in _gsm_re.findall(r"<<(.+?)=(-?\d+\.?\d*)>>", _text):
        _s = _expr.strip(); _toks = _gsm_re.findall(r"[+\-*/]", _s)
        if _s.startswith("-") and _toks and _toks[0]=="-": _toks=_toks[1:]
        if _toks: return _SYM.get(_toks[0],"unknown")
    return "unknown"
def _gsm_gold(_text):
    _m = _gsm_re.search(r"####\s*(-?\d+\.?\d*)", _text.replace(",",""))
    return float(_m.group(1)) if _m else 0.0
class _GSMShim:
    def __init__(self, ds): self.ds = ds
    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            row = self.ds[key]
            if isinstance(key, int):
                ans_text = row.get("answer", "")
                row = dict(row)
                row["Type"] = _gsm_first_op(ans_text)
                row["Answer"] = _gsm_gold(ans_text)
            return row
        if key == "Type":  return [_gsm_first_op(a) for a in self.ds["answer"]]
        if key == "Answer": return [_gsm_gold(a) for a in self.ds["answer"]]
        return self.ds[key]
    def __iter__(self):
        for i in range(len(self.ds)):
            row = self.ds[i]
            ans_text = row.get("answer", "")
            d = dict(row)
            d["Type"] = _gsm_first_op(ans_text)
            d["Answer"] = _gsm_gold(ans_text)
            yield d
    def __len__(self): return len(self.ds)

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[3]
ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
SILVER = REPO / "visualizations-all" / "gpt2-gsm8k" / "correctness-analysis" / "silver_traces_gsm8k.json"
JUDGED = REPO.parent / "cf-datasets" / "gsm8k_judged.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2-gsm8k" / "pca_bifurcation_diagnose.pdf"
OUT_JSON = REPO / "visualizations-all" / "gpt2-gsm8k" / "pca_bifurcation_diagnose.json"


def load_metadata():
    ds = load_dataset("gsm8k", "main")
    full = _GSMShim(ds["test"])
    types = np.array([t.replace("Common-Divison", "Common-Division") for t in full["Type"]])
    answers = np.array([float(str(a).replace(",", "")) for a in full["Answer"]], dtype=np.float64)
    student_correct = np.zeros(len(types), dtype=bool)
    if SILVER.exists():
        sv = json.load(open(SILVER))
        for r in sv["rows"]:
            i = r["idx"]
            if 0 <= i < len(types):
                student_correct[i] = (r["category"] == "gold")
    judged = json.load(open(JUDGED)) if JUDGED.exists() else []
    label_by_idx = {j["idx"]: j["label"] for j in judged}
    faithful = np.array([label_by_idx.get(i, "teacher_incorrect") for i in range(len(types))])
    return {
        "type": types,
        "answer": answers,
        "log_answer": np.log10(np.maximum(answers, 1) + 1),
        "parity": (answers.astype(int) % 2).astype(float),
        "digits": np.floor(np.log10(np.maximum(np.abs(answers), 1))).astype(int) + 1,
        "student_correct": student_correct.astype(float),
        "faithful_unf": (faithful == "unfaithful").astype(float),
        "faithful_f": (faithful == "faithful").astype(float),
    }


def feature_matrix(meta):
    """Return (feature_names, F: (N, n_feat)) for correlation analysis."""
    types = sorted(set(meta["type"].tolist()))
    cols = [
        ("log10(ans+1)", meta["log_answer"]),
        ("parity",       meta["parity"]),
        ("digit_count",  meta["digits"].astype(float)),
        ("ans_value",    meta["answer"]),
        ("ans_abs",      np.abs(meta["answer"])),
        ("student_correct", meta["student_correct"]),
        ("faithful_unf", meta["faithful_unf"]),
        ("faithful_f",   meta["faithful_f"]),
    ]
    for t in types:
        cols.append((f"type={t}", (meta["type"] == t).astype(float)))
    names = [c[0] for c in cols]
    F = np.stack([c[1] for c in cols], axis=1)
    return names, F


def safe_corr(x, y):
    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    print("loading metadata", flush=True)
    meta = load_metadata()
    names, F = feature_matrix(meta)
    print(f"  N={len(meta['type'])}  features={names}")

    print(f"loading activations from {ACTS}", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    print(f"  shape={a.shape}")

    print(f"fitting {S*L} PCAs (3 components each)", flush=True)
    pca_xyz = np.zeros((S, L, N, 3), dtype=np.float32)
    var_ratio = np.zeros((S, L, 3), dtype=np.float32)
    for s in range(S):
        for l in range(L):
            X = a[:, s, l, :]
            pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
            pca_xyz[s, l] = pca.fit_transform(X)
            var_ratio[s, l] = pca.explained_variance_ratio_
        print(f"  step {s+1}/{S} fitted")

    # correlation table: |corr| of each PC with each feature, per (s, l)
    corr = np.zeros((S, L, 3, len(names)), dtype=np.float32)
    for s in range(S):
        for l in range(L):
            for pc in range(3):
                v = pca_xyz[s, l, :, pc]
                for i in range(len(names)):
                    corr[s, l, pc, i] = abs(safe_corr(v, F[:, i]))

    # Top feature per (s, l, PC)
    top_idx = corr.argmax(axis=-1)  # (S, L, 3)
    top_corr = corr.max(axis=-1)    # (S, L, 3)

    # PC1 spread (range / IQR) as bifurcation proxy
    pc1 = pca_xyz[..., 0]  # (S, L, N)
    pc1_iqr = np.percentile(pc1, 75, axis=-1) - np.percentile(pc1, 25, axis=-1)
    pc1_std = pc1.std(axis=-1)

    # Pick the (s, l) with the strongest correlation between PC1 and any feature
    flat_pc1_corr = corr[:, :, 0, :].max(axis=-1)  # (S, L)
    s_best, l_best = np.unravel_index(flat_pc1_corr.argmax(), flat_pc1_corr.shape)
    feat_best = names[corr[s_best, l_best, 0, :].argmax()]
    print(f"\nstrongest PC1-feature correlation: step={s_best+1}, layer={l_best}, "
          f"feature={feat_best}, |r|={flat_pc1_corr[s_best, l_best]:.3f}")

    # Console summary
    print("\n=== Per-(step, layer) PC1: top feature, |r|, IQR(PC1) ===")
    print(f"  {'step':>4} {'layer':>5}  {'top feat':<22} {'|r|':>5}  {'IQR':>7}")
    for s in range(S):
        for l in range(L):
            f = names[top_idx[s, l, 0]]
            r = top_corr[s, l, 0]
            iqr = pc1_iqr[s, l]
            print(f"  {s+1:>4} {l:>5}  {f:<22} {r:5.3f}  {iqr:7.2f}")

    # Save JSON
    out_dict = {
        "shape": [N, S, L, H],
        "features": names,
        "pc_var_ratio": var_ratio.tolist(),
        "pc1_iqr": pc1_iqr.tolist(),
        "pc1_std": pc1_std.tolist(),
        "abs_corr_per_step_layer_pc_feature": corr.tolist(),
        "top_feature_per_step_layer_pc": [
            [[names[int(top_idx[s, l, pc])] for pc in range(3)] for l in range(L)] for s in range(S)
        ],
        "best_step_1indexed": int(s_best + 1),
        "best_layer": int(l_best),
        "best_feature": feat_best,
        "best_corr": float(flat_pc1_corr[s_best, l_best]),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_dict, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # --- PDF ---
    print(f"\nwriting {OUT_PDF}", flush=True)
    with PdfPages(OUT_PDF) as pdf:
        # Slide 1: heatmap of |r(PC1, feature)| per (s, l) for each feature
        feat_to_show = ["log10(ans+1)", "parity", "digit_count", "ans_abs",
                        "student_correct", "faithful_unf",
                        "type=Addition", "type=Subtraction",
                        "type=Multiplication", "type=Common-Division"]
        feat_to_show = [f for f in feat_to_show if f in names]
        n_feat = len(feat_to_show)
        ncols = 4
        nrows = int(np.ceil(n_feat / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13.33, 2.0 * nrows + 1))
        fig.suptitle("|Pearson r| between PC1 and candidate features, per (step, layer)",
                     fontsize=13, fontweight="bold")
        for i, fname in enumerate(feat_to_show):
            ax = axes.ravel()[i]
            fi = names.index(fname)
            mat = corr[:, :, 0, fi]  # (S, L)
            im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
            ax.set_title(fname, fontsize=9)
            ax.set_xlabel("layer", fontsize=8)
            ax.set_ylabel("latent step", fontsize=8)
            ax.set_yticks(range(S)); ax.set_yticklabels([str(s + 1) for s in range(S)], fontsize=7)
            ax.set_xticks(range(0, L, 2)); ax.set_xticklabels([str(l) for l in range(0, L, 2)], fontsize=7)
            for s_ in range(S):
                for l_ in range(L):
                    val = mat[s_, l_]
                    if val > 0.4:
                        ax.text(l_, s_, f"{val:.2f}", ha="center", va="center",
                                fontsize=5.5, color="white" if val < 0.6 else "black")
        for j in range(n_feat, nrows * ncols):
            axes.ravel()[j].axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 2: PC1 spread (IQR) heatmap to show where bifurcation grows
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(pc1_iqr, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("IQR of PC1 across examples, per (step, layer)  — bifurcation proxy",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("layer"); ax.set_ylabel("latent step")
        ax.set_yticks(range(S)); ax.set_yticklabels([str(s + 1) for s in range(S)])
        for s_ in range(S):
            for l_ in range(L):
                ax.text(l_, s_, f"{pc1_iqr[s_, l_]:.1f}", ha="center", va="center",
                        fontsize=6, color="white")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 3: side-by-side 3D scatter at best (s_best, l_best), six colorings
        xy = pca_xyz[s_best, l_best]
        v = var_ratio[s_best, l_best]
        fig = plt.figure(figsize=(14, 9))
        fig.suptitle(f"PCA of GPT-2 student acts at step {s_best+1}, layer {l_best}  "
                     f"(PC1={v[0]*100:.1f}%, PC2={v[1]*100:.1f}%, PC3={v[2]*100:.1f}%)\n"
                     f"strongest PC1 driver: {feat_best} (|r|={flat_pc1_corr[s_best, l_best]:.3f})",
                     fontsize=12, fontweight="bold")
        # Pull faithful label vector so we can color by it.
        from datasets import load_dataset, concatenate_datasets  # noqa: F811
        judged = json.load(open(JUDGED)) if JUDGED.exists() else []
        label_by_idx = {j["idx"]: j["label"] for j in judged}
        n_meta = len(meta["type"])
        faithful_labels = np.array(
            [label_by_idx.get(i, "teacher_incorrect") for i in range(n_meta)]
        )
        coloring_specs = [
            ("plain", None, None),
            ("log10(answer+1)", "viridis", meta["log_answer"]),
            ("parity (0/1)", "coolwarm", meta["parity"]),
            ("problem type", None, meta["type"]),
            ("student correct", None, meta["student_correct"].astype(int)),
            ("faithful (3 classes)", None, faithful_labels),
        ]
        for i, (title, cmap, vals) in enumerate(coloring_specs):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            if title == "plain":
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=5, c="#3357aa",
                           alpha=0.55, linewidths=0, rasterized=True)
            elif title == "problem type":
                colors = {"Addition": "#ff7f0e", "Subtraction": "#1f77b4",
                          "Multiplication": "#d62728", "Common-Division": "#2ca02c"}
                for cls, col in colors.items():
                    m = (meta["type"] == cls)
                    ax.scatter(xy[m, 0], xy[m, 1], xy[m, 2], s=5, c=col,
                               alpha=0.6, linewidths=0, rasterized=True, label=cls)
                ax.legend(fontsize=6, loc="upper right")
            elif title == "student correct":
                for val, col, lbl in [(0, "#cccccc", "wrong"), (1, "#2ca02c", "right")]:
                    m = vals == val
                    ax.scatter(xy[m, 0], xy[m, 1], xy[m, 2], s=5, c=col,
                               alpha=0.6, linewidths=0, rasterized=True, label=lbl)
                ax.legend(fontsize=6, loc="upper right")
            elif title == "faithful (3 classes)":
                colors_f = {"teacher_incorrect": "#cccccc", "faithful": "#2ca02c", "unfaithful": "#d62728"}
                # background first, unfaithful on top
                for cls in ["teacher_incorrect", "faithful", "unfaithful"]:
                    m = vals == cls
                    s_pt = 14 if cls == "unfaithful" else 5
                    alph = 0.95 if cls == "unfaithful" else 0.45
                    n = int(m.sum())
                    ax.scatter(xy[m, 0], xy[m, 1], xy[m, 2], s=s_pt, c=colors_f[cls],
                               alpha=alph, linewidths=0, rasterized=True,
                               label=f"{cls} ({n})")
                ax.legend(fontsize=5, loc="upper right")
            else:
                sc = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=5, c=vals,
                                cmap=cmap, alpha=0.6, linewidths=0, rasterized=True)
                cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
                cb.ax.tick_params(labelsize=6)
            ax.set_title(title, fontsize=10)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
            ax.set_zlabel("PC3", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 4: same six colorings but at last step / last layer (representative of final bifurcation)
        s_last, l_last = S - 1, L - 1
        xy = pca_xyz[s_last, l_last]
        v = var_ratio[s_last, l_last]
        fig = plt.figure(figsize=(14, 9))
        fig.suptitle(f"PCA at step {s_last+1}, layer {l_last} (final)  "
                     f"(PC1={v[0]*100:.1f}%, PC2={v[1]*100:.1f}%, PC3={v[2]*100:.1f}%)",
                     fontsize=12, fontweight="bold")
        for i, (title, cmap, vals) in enumerate(coloring_specs):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            if title == "plain":
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=5, c="#3357aa",
                           alpha=0.55, linewidths=0, rasterized=True)
            elif title == "problem type":
                colors = {"Addition": "#ff7f0e", "Subtraction": "#1f77b4",
                          "Multiplication": "#d62728", "Common-Division": "#2ca02c"}
                for cls, col in colors.items():
                    m = (meta["type"] == cls)
                    ax.scatter(xy[m, 0], xy[m, 1], xy[m, 2], s=5, c=col,
                               alpha=0.6, linewidths=0, rasterized=True, label=cls)
                ax.legend(fontsize=6, loc="upper right")
            elif title == "student correct":
                for val, col, lbl in [(0, "#cccccc", "wrong"), (1, "#2ca02c", "right")]:
                    m = vals == val
                    ax.scatter(xy[m, 0], xy[m, 1], xy[m, 2], s=5, c=col,
                               alpha=0.6, linewidths=0, rasterized=True, label=lbl)
                ax.legend(fontsize=6, loc="upper right")
            elif title == "faithful (3 classes)":
                colors_f = {"teacher_incorrect": "#cccccc", "faithful": "#2ca02c", "unfaithful": "#d62728"}
                # background first, unfaithful on top
                for cls in ["teacher_incorrect", "faithful", "unfaithful"]:
                    m = vals == cls
                    s_pt = 14 if cls == "unfaithful" else 5
                    alph = 0.95 if cls == "unfaithful" else 0.45
                    n = int(m.sum())
                    ax.scatter(xy[m, 0], xy[m, 1], xy[m, 2], s=s_pt, c=colors_f[cls],
                               alpha=alph, linewidths=0, rasterized=True,
                               label=f"{cls} ({n})")
                ax.legend(fontsize=5, loc="upper right")
            else:
                sc = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=5, c=vals,
                                cmap=cmap, alpha=0.6, linewidths=0, rasterized=True)
                cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
                cb.ax.tick_params(labelsize=6)
            ax.set_title(title, fontsize=10)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
