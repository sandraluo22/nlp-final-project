"""Faithfulness probe on GSM8K.

Fast version of faithfulness_probe_v2 for GSM8K: per-(step, layer) probe
of CODI's latent residual for whether CODI's internal CoT is faithful to
a correct solution.  Uses gsm8k_judged.json labels (433 faithful +
118 unfaithful = 551 labeled examples; 768 teacher_incorrect excluded).

Reports AUC, accuracy, per-class recall on 5-fold CV against:
  - Real latent residuals.
  - Random Gaussian features (baseline).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
LAT = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
JUDGED = REPO.parent / "cf-datasets" / "gsm8k_judged.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "faithfulness_probe_gsm8k.json"
OUT_PDF = PD / "faithfulness_probe_gsm8k.pdf"
SEED = 0


def cv_metrics(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    aucs, accs, r0s, r1s = [], [], [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        score = clf.decision_function(sc.transform(X[te]))
        yhat = (score >= 0).astype(int)
        try: aucs.append(roc_auc_score(y[te], score))
        except ValueError: aucs.append(0.5)
        accs.append(accuracy_score(y[te], yhat))
        r0s.append(recall_score(y[te], yhat, pos_label=0, zero_division=0))
        r1s.append(recall_score(y[te], yhat, pos_label=1, zero_division=0))
    return {"auc": float(np.mean(aucs)), "acc": float(np.mean(accs)),
            "recall_unfaithful": float(np.mean(r0s)),
            "recall_faithful": float(np.mean(r1s))}


def main():
    acts = torch.load(LAT, map_location="cpu", weights_only=True).float().numpy()
    N, S, L, H = acts.shape
    print(f"acts={acts.shape}")
    judged = json.load(open(JUDGED))
    labels = ["teacher_incorrect"] * N
    for j in judged:
        if 0 <= j["idx"] < N: labels[j["idx"]] = j["label"]
    keep = [i for i in range(N) if labels[i] in {"faithful", "unfaithful"}]
    y = np.array([1 if labels[i] == "faithful" else 0 for i in keep])
    acts_k = acts[keep]
    print(f"kept N={len(keep)}: faithful={(y==1).sum()} unfaithful={(y==0).sum()}")
    baseline = max(y.mean(), 1 - y.mean())
    print(f"majority baseline acc = {baseline:.3f}")

    rng = np.random.default_rng(SEED)
    rand = rng.standard_normal((len(keep), H)).astype(np.float32)

    real_auc = np.zeros((S, L)); real_acc = np.zeros((S, L))
    real_r0 = np.zeros((S, L));  real_r1 = np.zeros((S, L))
    for s in range(S):
        for l in range(L):
            r = cv_metrics(acts_k[:, s, l, :], y)
            real_auc[s, l] = r["auc"]; real_acc[s, l] = r["acc"]
            real_r0[s, l] = r["recall_unfaithful"]; real_r1[s, l] = r["recall_faithful"]
        print(f"  step {s+1}: best L_auc={int(real_auc[s].argmax())} "
              f"AUC={real_auc[s].max():.3f}  acc={real_acc[s].max():.3f}")
    rand_metrics = cv_metrics(rand, y)
    print(f"\nrandom Gaussian baseline: AUC={rand_metrics['auc']:.3f}  "
          f"acc={rand_metrics['acc']:.3f}  "
          f"recall_unfaithful={rand_metrics['recall_unfaithful']:.3f}  "
          f"recall_faithful={rand_metrics['recall_faithful']:.3f}")

    s_b, l_b = np.unravel_index(real_auc.argmax(), real_auc.shape)
    best = cv_metrics(acts_k[:, s_b, l_b, :], y)
    print(f"\nbest cell: step={s_b+1}, L={l_b}")
    print(f"  AUC={best['auc']:.3f}  acc={best['acc']:.3f}  "
          f"recall(unfaith)={best['recall_unfaithful']:.3f}  "
          f"recall(faith)={best['recall_faithful']:.3f}")

    OUT_JSON.write_text(json.dumps({
        "n_faithful": int((y == 1).sum()), "n_unfaithful": int((y == 0).sum()),
        "baseline_acc": float(baseline),
        "real_auc": real_auc.tolist(), "real_acc": real_acc.tolist(),
        "real_recall_unfaithful": real_r0.tolist(), "real_recall_faithful": real_r1.tolist(),
        "rand_auc": rand_metrics["auc"], "rand_acc": rand_metrics["acc"],
        "rand_recall_unfaithful": rand_metrics["recall_unfaithful"],
        "rand_recall_faithful": rand_metrics["recall_faithful"],
        "best_step_1indexed": int(s_b + 1), "best_layer": int(l_b),
    }, indent=2))
    print(f"saved {OUT_JSON}")

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, M, title, vmin, vmax in [
            (axes[0, 0], real_auc, f"AUC (random={rand_metrics['auc']:.2f})", 0.4, 1.0),
            (axes[0, 1], real_acc, f"acc (majority={baseline:.2f})", 0.4, 1.0),
            (axes[1, 0], real_r0, "recall(unfaithful)", 0.0, 1.0),
            (axes[1, 1], real_r1, "recall(faithful)", 0.0, 1.0),
        ]:
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_xticks(range(L)); ax.set_yticks(range(S))
            ax.set_yticklabels([str(s+1) for s in range(S)])
            ax.set_title(title, fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04)
            for s in range(S):
                for l in range(L):
                    v = M[s, l]
                    if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.25:
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < (vmin + vmax) / 2 else "black")
        fig.suptitle(f"GSM8K faithfulness probe — N={len(keep)} "
                     f"({(y==1).sum()} faithful / {(y==0).sum()} unfaithful)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
