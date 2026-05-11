"""GSM8K version of correctness probe on the latent residuals.

For each (step, layer), fit a binary logreg probe to predict
is_correct_at_step from the residual.  Same as the SVAMP correctness_probe
but uses gsm8k_latent_acts.pt + the gsm8k_colon_acts_meta.json predictions
as labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
LAT_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
META_PATH = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "correctness_probe_gsm8k.json"
OUT_PDF = PD / "correctness_probe_gsm8k.pdf"


def main():
    acts = torch.load(LAT_PATH, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = acts.shape
    meta = json.load(open(META_PATH))
    pred = np.array([np.nan if v is None else float(v) for v in meta["pred_int_extracted"]])
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    keep = ~(np.isnan(pred) | np.isnan(gold))
    acts = acts[keep]
    correct = (np.abs(pred[keep] - gold[keep]) < 1e-3).astype(int)
    N_kept = acts.shape[0]
    print(f"GSM8K: kept N={N_kept} of {N}, baseline acc = {correct.mean()*100:.1f}%")

    # 70/30 train/test stratified split
    idx_tr, idx_te = train_test_split(np.arange(N_kept), test_size=0.30, random_state=0,
                                       stratify=correct)
    train_acc = np.zeros((S, L))
    test_acc = np.zeros((S, L))
    test_auc = np.zeros((S, L))
    for s in range(S):
        for l in range(L):
            X = acts[:, s, l, :]
            sc = StandardScaler().fit(X[idx_tr])
            Xtr = sc.transform(X[idx_tr]); Xte = sc.transform(X[idx_te])
            clf = LogisticRegression(max_iter=4000, C=0.1, solver="lbfgs",
                                      class_weight="balanced").fit(Xtr, correct[idx_tr])
            train_acc[s, l] = clf.score(Xtr, correct[idx_tr])
            test_acc[s, l]  = clf.score(Xte, correct[idx_te])
            test_auc[s, l]  = roc_auc_score(correct[idx_te],
                                            clf.predict_proba(Xte)[:, 1])
        print(f"  step {s+1}: best L test_auc = {test_auc[s].max():.3f}")

    OUT_JSON.write_text(json.dumps({
        "shape": list(acts.shape), "N": int(N_kept),
        "baseline_acc": float(correct.mean()),
        "train_acc": train_acc.tolist(),
        "test_acc": test_acc.tolist(),
        "test_auc": test_auc.tolist(),
    }, indent=2))

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, M, title, vmin, vmax in [
            (axes[0], train_acc, "GSM8K — train acc",  0.5, 1.0),
            (axes[1], test_acc,  "GSM8K — test acc",   0.5, 1.0),
            (axes[2], test_auc,  "GSM8K — test AUC",   0.5, 1.0),
        ]:
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                            vmin=vmin, vmax=vmax)
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_xticks(range(L)); ax.set_yticks(range(S))
            ax.set_yticklabels([str(s+1) for s in range(S)])
            ax.set_title(title, fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04)
            for s in range(S):
                for l in range(L):
                    v = M[s, l]
                    if v >= 0.7 or v <= 0.55:
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.75 else "black")
        fig.suptitle(f"GSM8K correctness probe on CODI latent residuals "
                     f"(N={N_kept}, baseline acc={correct.mean()*100:.1f}%)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_JSON} and {OUT_PDF}")


if __name__ == "__main__":
    main()
