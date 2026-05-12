"""Per-(layer, latent-step) 3D PCA of student activations on SVAMP, colored
four ways and laid out as a slideshow.

Activations: (N=1000, latent_steps=6, layers+1=17, hidden=2048)

Layout (one slide per page, 6 subplots per slide):
  - one slide per layer, plain                  (17 slides)
  - one slide per layer, faithful/unfaithful    (17 slides)
  - one slide per layer, problem type           (17 slides)
  - one slide per layer, magnitude bucket       (17 slides)
Total: 68 slides in a single PDF.

Magnitude coloring is bucketed (<10, 10-99, 100-999, 1000+) because raw SVAMP
answers span 1 to ~2.2e7 and a continuous colormap is saturated by the outliers.
"""

from __future__ import annotations

import json
from collections import Counter
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
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.decomposition import PCA


REPO = Path(__file__).resolve().parents[2]
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
SILVER_PATH = REPO / "visualizations-all" / "gpt2-gsm8k" / "correctness-analysis" / "silver_traces_gsm8k.json"
JUDGED_PATH = REPO.parent / "cf-datasets" / "gsm8k_judged.json"
OUT_PDF = REPO / "visualizations-all" / "gpt2-gsm8k" / "pca_slideshow.pdf"

PROBLEM_TYPE_COLORS = {
    "Subtraction": "#1f77b4",
    "Addition": "#ff7f0e",
    "Common-Division": "#2ca02c",
    "Multiplication": "#d62728",
}
FAITHFUL_COLORS = {
    "faithful": "#2ca02c",
    "unfaithful": "#d62728",
    "teacher_incorrect": "#cccccc",
}
MAG_BUCKETS = ["<10", "10-99", "100-999", "1000+"]
MAG_COLORS = {
    "<10": "#440154",
    "10-99": "#3b528b",
    "100-999": "#21918c",
    "1000+": "#fde725",
}


def bucket_magnitude(ans: np.ndarray) -> np.ndarray:
    out = np.empty(len(ans), dtype=object)
    out[ans < 10] = "<10"
    out[(ans >= 10) & (ans < 100)] = "10-99"
    out[(ans >= 100) & (ans < 1000)] = "100-999"
    out[ans >= 1000] = "1000+"
    return out


def load_metadata() -> dict:
    print("loading metadata", flush=True)
    ds = load_dataset("gsm8k", "main")
    full = _GSMShim(ds["test"])
    types = [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    answers = np.array(
        [float(str(a).replace(",", "")) for a in full["Answer"]], dtype=np.float64
    )

    judged = json.load(open(JUDGED_PATH))
    label_by_idx = {j["idx"]: j["label"] for j in judged}
    faithful_label = [
        label_by_idx.get(i, "teacher_incorrect") for i in range(len(types))
    ]

    # CODI-on-GSM8K correctness from silver_traces_gsm8k.json (per-idx)
    correct = np.zeros(len(types), dtype=bool)
    try:
        silver = json.load(open(SILVER_PATH))
        for r in silver["rows"]:
            i = r["idx"]
            if 0 <= i < len(types):
                correct[i] = (r["category"] == "gold")
    except Exception as e:
        print(f"  WARN: could not load silver_traces_gsm8k.json: {e}")

    return {
        "n": len(types),
        "problem_type": np.array(types),
        "answer": answers,
        "log_answer": np.log10(np.maximum(answers, 1) + 1),
        "faithful": np.array(faithful_label),
        "magnitude": bucket_magnitude(answers),
        "correct": correct,
    }


def load_activations() -> np.ndarray:
    print(f"loading activations from {ACTS_PATH}", flush=True)
    t = torch.load(ACTS_PATH, map_location="cpu", weights_only=True)
    a = t.float().numpy()  # (N, latent_steps, layers+1, hidden)
    print(f"  shape={a.shape}  bytes={a.nbytes/1e9:.2f} GB", flush=True)
    return a


def fit_all_pca_3d(acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return projected coords (L, S, N, 3) and var-ratio (L, S, 3)."""
    N, S, L, _H = acts.shape
    out = np.empty((L, S, N, 3), dtype=np.float32)
    var_ratio = np.empty((L, S, 3), dtype=np.float32)
    print(f"fitting {L*S} PCAs (3D, per layer × per latent step)", flush=True)
    for layer in range(L):
        for step in range(S):
            X = acts[:, step, layer, :]
            pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
            out[layer, step] = pca.fit_transform(X)
            var_ratio[layer, step] = pca.explained_variance_ratio_
        print(f"  layer {layer}/{L-1} done", flush=True)
    return out, var_ratio


def _strip_3d_axes(ax, var_ratio: np.ndarray | None = None):
    """Hide tick labels but keep axis labels (with variance % if provided)."""
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0, pad=-2)
    if var_ratio is not None:
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=7, labelpad=-10)
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=7, labelpad=-10)
        ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)", fontsize=7, labelpad=-10)
    else:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_visible(False)
    ax.grid(True, alpha=0.25)


def render_slide(
    pdf: PdfPages,
    layer: int,
    pca_for_layer: np.ndarray,  # (S, N, 3)
    var_for_layer: np.ndarray,  # (S, 3)
    coloring: str,
    meta: dict,
):
    S = pca_for_layer.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.875), subplot_kw={"projection": "3d"})
    fig.suptitle(
        f"Layer {layer}  —  3D PCA of student latent activations  —  coloring: {coloring}",
        fontsize=12,
    )
    legend_proxies: list[Line2D] = []
    for s, ax in enumerate(axes.ravel()):
        if s >= S:
            ax.axis("off")
            continue
        xy = pca_for_layer[s]

        if coloring == "plain":
            ax.scatter(
                xy[:, 0], xy[:, 1], xy[:, 2],
                s=4, c="#3357aa", alpha=0.5, linewidths=0, depthshade=True,
                rasterized=True,
            )

        elif coloring == "faithful":
            order = ["teacher_incorrect", "faithful", "unfaithful"]
            for cls in order:
                mask = meta["faithful"] == cls
                ax.scatter(
                    xy[mask, 0], xy[mask, 1], xy[mask, 2],
                    s=10 if cls == "unfaithful" else 4,
                    c=FAITHFUL_COLORS[cls],
                    alpha=0.9 if cls == "unfaithful" else 0.4,
                    linewidths=0, depthshade=True, rasterized=True,
                )
            if s == 0:
                for cls in order:
                    n = (meta["faithful"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=FAITHFUL_COLORS[cls], label=f"{cls} ({n})")
                    )

        elif coloring == "problem_type":
            for cls, color in PROBLEM_TYPE_COLORS.items():
                mask = meta["problem_type"] == cls
                ax.scatter(
                    xy[mask, 0], xy[mask, 1], xy[mask, 2],
                    s=4, c=color, alpha=0.55, linewidths=0,
                    depthshade=True, rasterized=True,
                )
            if s == 0:
                for cls, color in PROBLEM_TYPE_COLORS.items():
                    n = (meta["problem_type"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=color, label=f"{cls} ({n})")
                    )

        elif coloring == "magnitude":
            for cls in MAG_BUCKETS:
                mask = meta["magnitude"] == cls
                ax.scatter(
                    xy[mask, 0], xy[mask, 1], xy[mask, 2],
                    s=4, c=MAG_COLORS[cls], alpha=0.6, linewidths=0,
                    depthshade=True, rasterized=True,
                )
            if s == 0:
                for cls in MAG_BUCKETS:
                    n = (meta["magnitude"] == cls).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=MAG_COLORS[cls], label=f"{cls} ({n})")
                    )

        elif coloring == "log_answer":
            log_ans = meta["log_answer"]
            sc = ax.scatter(
                xy[:, 0], xy[:, 1], xy[:, 2],
                s=4, c=log_ans, cmap="viridis",
                vmin=float(log_ans.min()), vmax=float(log_ans.max()),
                alpha=0.6, linewidths=0, depthshade=True, rasterized=True,
            )

        elif coloring == "correct":
            for val, color, label in [(False, "#d62728", "wrong"),
                                      (True, "#2ca02c", "right")]:
                mask = meta["correct"] == val
                ax.scatter(
                    xy[mask, 0], xy[mask, 1], xy[mask, 2],
                    s=4, c=color, alpha=0.55, linewidths=0,
                    depthshade=True, rasterized=True,
                )
            if s == 0:
                for val, color, label in [(False, "#d62728", "wrong"),
                                          (True, "#2ca02c", "right")]:
                    n = (meta["correct"] == val).sum()
                    legend_proxies.append(
                        Line2D([0], [0], marker="o", linestyle="",
                               color=color, label=f"{label} ({n})")
                    )
        else:
            raise ValueError(coloring)

        ax.set_title(f"latent step {s+1}", fontsize=10)
        _strip_3d_axes(ax, var_ratio=var_for_layer[s])

    if legend_proxies:
        fig.legend(
            handles=legend_proxies,
            loc="lower center",
            ncol=len(legend_proxies),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    if coloring == "log_answer":
        log_ans = meta["log_answer"]
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=float(log_ans.min()), vmax=float(log_ans.max())),
        )
        sm.set_array([])
        cax = fig.add_axes([0.25, 0.04, 0.5, 0.018])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("log10(answer + 1)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.02, right=0.98, wspace=0.08, hspace=0.12)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main():
    meta = load_metadata()
    print(
        "labels: faithful=", dict(Counter(meta["faithful"])),
        " | type=", dict(Counter(meta["problem_type"])),
        " | mag=", dict(Counter(meta["magnitude"])),
        flush=True,
    )

    acts = load_activations()
    pca_all, var_all = fit_all_pca_3d(acts)
    L = pca_all.shape[0]

    print(f"writing {OUT_PDF}", flush=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for coloring in ["plain", "faithful", "problem_type", "magnitude", "log_answer", "correct"]:
            print(f"  coloring: {coloring}", flush=True)
            for layer in range(L):
                render_slide(pdf, layer, pca_all[layer], var_all[layer], coloring, meta)
    print(f"done -> {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
