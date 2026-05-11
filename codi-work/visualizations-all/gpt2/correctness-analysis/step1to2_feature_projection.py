"""Project the (step 2 − step 1) residual delta onto FEATURE directions
(not lm_head). Catches what the delta encodes outside of token-space.

Probe directions fit per (step, layer):
  - operator (4-class logreg)        → 4 direction vectors per layer
  - magnitude (regression on log10(answer+1))
  - correctness (binary logreg on student_correct at step 1)
  - faithfulness (binary logreg on judged labels)
  - answer-token direction (lm_head row for each example's gold first token)

For each layer L, compute:
  - ‖mean_delta_L‖
  - cos(mean_delta_L, each probe direction)
  - fraction of delta's energy along each direction
  - cohort-specific projections (wr vs rw)

Output: step1to2_feature_projection.{json,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
from datasets import concatenate_datasets, load_dataset

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
ACTS = REPO / "inference" / "runs" / "svamp_student_gpt2" / "activations.pt"
LM_HEAD = PD / "codi_gpt2_lm_head.npy"
FD = PD / "force_decode_per_step.json"
STUDENT = REPO / "inference" / "runs" / "svamp_student_gpt2" / "results.json"
JUDGED = REPO.parent / "cf-datasets" / "svamp_judged.json"
OUT_JSON = PD / "step1to2_feature_projection.json"
OUT_PDF = PD / "step1to2_feature_projection.pdf"

OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def main():
    print("loading activations", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = a.shape
    W = np.load(LM_HEAD)
    fd = json.load(open(FD))
    correct = np.array(fd["correct_per_step"])
    correct_s1 = correct[0].astype(bool)
    correct_s2 = correct[1].astype(bool)
    wr_mask = (~correct_s1) & correct_s2
    rw_mask = correct_s1 & (~correct_s2)
    print(f"  N={N} L={L}; wr={int(wr_mask.sum())} rw={int(rw_mask.sum())}")

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array([t.replace("Common-Divison", "Common-Division") for t in full["Type"]])
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    log_ans = np.log10(np.maximum(golds, 1) + 1)
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y_op = np.array([op_to_idx.get(t, -1) for t in types])
    valid_op = y_op >= 0

    # Student correctness at step 1 (binary for probe target)
    y_correct_s1 = correct_s1.astype(int)

    # Faithfulness label
    judged = json.load(open(JUDGED))
    label_by_idx = {j["idx"]: j["label"] for j in judged}
    fai_labels = np.array([label_by_idx.get(i, "teacher_incorrect") for i in range(N)])
    fai_mask = (fai_labels == "faithful") | (fai_labels == "unfaithful")
    y_fai = (fai_labels == "faithful").astype(int)

    # Per-example answer direction
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    def first_tok(g):
        gs = str(int(g)) if float(g).is_integer() else str(g)
        ids = tok.encode(" " + gs, add_special_tokens=False)
        return ids[0] if ids else -1
    targets = np.array([first_tok(g) for g in golds])
    W_target = W[targets]   # (N, H)

    # Compute delta per example, per layer
    delta = a[:, 1, :, :] - a[:, 0, :, :]   # (N, L, H)
    delta_norm_overall = np.linalg.norm(delta.mean(axis=0), axis=-1)

    # Per-layer probe directions (fit on step-1 activations, project delta)
    # Operator: 4-class logreg coef -> 4 direction vectors of size H
    print("fitting per-layer probes at step 1 activations...")
    op_axes = np.zeros((L, 4, H))    # 4-class coefs
    mag_axes = np.zeros((L, H))
    cor_axes = np.zeros((L, H))
    fai_axes = np.zeros((L, H))
    for l in range(L):
        X = a[:, 0, l, :]   # step 1 residual at layer l
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        clf_op = LogisticRegression(max_iter=4000, C=1.0,
                                    solver="lbfgs").fit(Xs[valid_op], y_op[valid_op])
        # coef in scaled space; convert back via sigma
        op_axes[l] = clf_op.coef_ / sc.scale_
        ridge_mag = Ridge(alpha=1.0).fit(Xs, log_ans)
        mag_axes[l] = ridge_mag.coef_ / sc.scale_
        clf_cor = LogisticRegression(max_iter=4000, C=0.1,
                                     solver="lbfgs",
                                     class_weight="balanced").fit(Xs, y_correct_s1)
        cor_axes[l] = clf_cor.coef_[0] / sc.scale_
        if fai_mask.sum() >= 20:
            clf_fai = LogisticRegression(max_iter=4000, C=0.1,
                                          solver="lbfgs",
                                          class_weight="balanced").fit(Xs[fai_mask], y_fai[fai_mask])
            fai_axes[l] = clf_fai.coef_[0] / sc.scale_

    # Per-layer projections of mean delta onto each axis (cos sim and abs fraction)
    print("projecting delta onto axes...")
    mean_delta = delta.mean(axis=0)   # (L, H)
    mean_delta_wr = delta[wr_mask].mean(axis=0) if wr_mask.any() else np.zeros((L, H))
    mean_delta_rw = delta[rw_mask].mean(axis=0) if rw_mask.any() else np.zeros((L, H))

    def cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb + 1e-12))

    proj = {"layers": list(range(L)),
            "norm_overall": delta_norm_overall.tolist(),
            "norm_wr": np.linalg.norm(mean_delta_wr, axis=-1).tolist(),
            "norm_rw": np.linalg.norm(mean_delta_rw, axis=-1).tolist()}

    feature_keys = {
        "operator_Add":  ("op", 0),
        "operator_Sub":  ("op", 1),
        "operator_Mul":  ("op", 2),
        "operator_Div":  ("op", 3),
        "operator_Add_vs_Sub": ("op_diff", (0, 1)),
        "magnitude":     ("mag", None),
        "correctness":   ("cor", None),
        "faithfulness":  ("fai", None),
    }

    cos_results = {k: {"overall": [], "wr": [], "rw": []} for k in feature_keys}
    for l in range(L):
        for fk, (kind, p) in feature_keys.items():
            if kind == "op": ax = op_axes[l, p]
            elif kind == "op_diff": ax = op_axes[l, p[0]] - op_axes[l, p[1]]
            elif kind == "mag": ax = mag_axes[l]
            elif kind == "cor": ax = cor_axes[l]
            elif kind == "fai": ax = fai_axes[l]
            ax = unit(ax)
            cos_results[fk]["overall"].append(cos(mean_delta[l], ax))
            cos_results[fk]["wr"].append(cos(mean_delta_wr[l], ax))
            cos_results[fk]["rw"].append(cos(mean_delta_rw[l], ax))

    # Answer direction (per example -> per-example cos, then average)
    ans_cos = {"overall": [], "wr": [], "rw": []}
    for l in range(L):
        per_ex_cos = []
        per_ex_cos_wr = []
        per_ex_cos_rw = []
        for i in range(N):
            v = delta[i, l]
            w = W_target[i]
            n = np.linalg.norm(v) * np.linalg.norm(w)
            if n < 1e-12: continue
            c = float(np.dot(v, w) / n)
            per_ex_cos.append(c)
            if wr_mask[i]: per_ex_cos_wr.append(c)
            if rw_mask[i]: per_ex_cos_rw.append(c)
        ans_cos["overall"].append(float(np.mean(per_ex_cos)))
        ans_cos["wr"].append(float(np.mean(per_ex_cos_wr)) if per_ex_cos_wr else 0.0)
        ans_cos["rw"].append(float(np.mean(per_ex_cos_rw)) if per_ex_cos_rw else 0.0)
    cos_results["answer_token"] = ans_cos

    out = {**proj, "cos_per_layer": cos_results, "features": list(cos_results.keys())}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_JSON}")

    # === Slideshow ===
    fkeys = ["operator_Add", "operator_Sub", "operator_Mul", "operator_Div",
             "operator_Add_vs_Sub", "magnitude", "correctness", "faithfulness",
             "answer_token"]
    with PdfPages(OUT_PDF) as pdf:
        # Slide 1: title
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle("Projecting the (step 2 − step 1) residual delta onto FEATURE directions",
                     fontsize=14, fontweight="bold")
        ax = fig.add_axes([0.05, 0.04, 0.9, 0.85]); ax.axis("off")
        ax.text(0.0, 0.95,
                "lm_head decoding of the delta only catches token-aligned content. To probe "
                "non-token computation (operator, magnitude, correctness, faithfulness), fit "
                "a probe on step-1 activations and project the delta onto the probe direction.\n\n"
                "Each row below shows cos(mean_delta_at_layer_L, probe_direction_at_layer_L) for "
                "three cohorts:\n"
                "  • all 1000 examples\n"
                "  • wr (n=70, wrong-at-1 → right-at-2 — successful 1→2 transitions)\n"
                "  • rw (n=34, right-at-1 → wrong-at-2 — backfires)\n\n"
                "If a feature direction has high cos with the delta, the 1→2 transition is "
                "writing along that feature axis.",
                fontsize=10, transform=ax.transAxes, va="top", wrap=True)
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 2: feature × layer heatmap for the overall mean delta
        mat = np.array([cos_results[k]["overall"] for k in fkeys])  # (n_feat, L)
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-max(0.1, np.abs(mat).max()),
                       vmax=max(0.1, np.abs(mat).max()))
        ax.set_xticks(range(L)); ax.set_xticklabels([f"L{l}" for l in range(L)], fontsize=8)
        ax.set_yticks(range(len(fkeys))); ax.set_yticklabels(fkeys, fontsize=9)
        ax.set_title("cos(mean delta, feature direction)  per (feature × layer) — all 1000",
                     fontsize=11, fontweight="bold")
        for i, f in enumerate(fkeys):
            for l in range(L):
                v = mat[i, l]
                if abs(v) >= 0.1:
                    ax.text(l, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(v) > 0.6 * np.abs(mat).max() else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 3: same for wr cohort
        mat = np.array([cos_results[k]["wr"] for k in fkeys])
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-max(0.1, np.abs(mat).max()),
                       vmax=max(0.1, np.abs(mat).max()))
        ax.set_xticks(range(L)); ax.set_xticklabels([f"L{l}" for l in range(L)], fontsize=8)
        ax.set_yticks(range(len(fkeys))); ax.set_yticklabels(fkeys, fontsize=9)
        ax.set_title("cos(mean delta, feature direction) — wr cohort (n=70)",
                     fontsize=11, fontweight="bold")
        for i, f in enumerate(fkeys):
            for l in range(L):
                v = mat[i, l]
                if abs(v) >= 0.1:
                    ax.text(l, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if abs(v) > 0.6 * np.abs(mat).max() else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 4: rw cohort
        mat = np.array([cos_results[k]["rw"] for k in fkeys])
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-max(0.1, np.abs(mat).max()),
                       vmax=max(0.1, np.abs(mat).max()))
        ax.set_xticks(range(L)); ax.set_xticklabels([f"L{l}" for l in range(L)], fontsize=8)
        ax.set_yticks(range(len(fkeys))); ax.set_yticklabels(fkeys, fontsize=9)
        ax.set_title("cos(mean delta, feature direction) — rw cohort (n=34)",
                     fontsize=11, fontweight="bold")
        for i, f in enumerate(fkeys):
            for l in range(L):
                v = mat[i, l]
                if abs(v) >= 0.1:
                    ax.text(l, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if abs(v) > 0.6 * np.abs(mat).max() else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 5: wr − rw differential per feature
        diff = np.array([np.array(cos_results[k]["wr"]) - np.array(cos_results[k]["rw"])
                         for k in fkeys])
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-max(0.1, np.abs(diff).max()),
                       vmax=max(0.1, np.abs(diff).max()))
        ax.set_xticks(range(L)); ax.set_xticklabels([f"L{l}" for l in range(L)], fontsize=8)
        ax.set_yticks(range(len(fkeys))); ax.set_yticklabels(fkeys, fontsize=9)
        ax.set_title("Δ cos(wr) − Δ cos(rw) — what successful 1→2 transitions add MORE OF",
                     fontsize=11, fontweight="bold")
        for i, f in enumerate(fkeys):
            for l in range(L):
                v = diff[i, l]
                if abs(v) >= 0.05:
                    ax.text(l, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if abs(v) > 0.6 * np.abs(diff).max() else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
