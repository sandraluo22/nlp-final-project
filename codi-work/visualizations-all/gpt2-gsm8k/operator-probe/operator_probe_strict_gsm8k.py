"""4-class operator probe on the GSM8K-style operator-PURE CF dataset.

Uses gsm8k_cf_op_strict_latent_acts.pt (N=324, balanced across 4 operators
× 3 magnitude buckets) where EVERY step of every problem uses one operator.

Outputs per-(step, layer) 4-class accuracy. Also computes balanced-acc
within each magnitude bucket separately, to confirm magnitude isn't the
driver.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
LAT_PATH = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_op_strict_latent_acts.pt"
COL_PATH = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_op_strict_colon_acts.pt"
META_PATH = REPO / "visualizations-all" / "gpt2-gsm8k" / "counterfactuals" / "gsm8k_cf_op_strict_colon_acts_meta.json"
CF_PATH = REPO.parent / "cf-datasets" / "gsm8k_cf_op_strict.json"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "operator_probe_strict_gsm8k.json"
OUT_PDF = PD / "operator_probe_strict_gsm8k.pdf"

SEED = 0
OP_NAMES = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


def cv_acc(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def main():
    lat = torch.load(LAT_PATH, map_location="cpu", weights_only=True).float().numpy()
    col = torch.load(COL_PATH, map_location="cpu", weights_only=True).float().numpy()
    rows = json.load(open(CF_PATH))
    N = lat.shape[0]
    assert N == len(rows)
    op_to_int = {n: i for i, n in enumerate(OP_NAMES)}
    y = np.array([op_to_int[r["type"]] for r in rows])
    mag = np.array([r["magnitude_bucket"] for r in rows])
    print(f"strict CF: N={N}, label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    S, L, H = lat.shape[1], lat.shape[2], lat.shape[3]
    # Latent (step, layer) 4-class probe
    lat_acc = np.zeros((S, L))
    for s in range(S):
        for l in range(L):
            lat_acc[s, l] = cv_acc(lat[:, s, l, :], y)
        print(f"  step {s+1}: best L={int(lat_acc[s].argmax())} acc={lat_acc[s].max():.3f}")

    # Colon-position per-layer probe
    col_acc = np.zeros(L)
    for l in range(L):
        col_acc[l] = cv_acc(col[:, l, :], y)
    print(f"  colon best L={int(col_acc.argmax())} acc={col_acc.max():.3f}")

    # Within-magnitude probes to confirm not magnitude-driven
    mag_acc = {}
    for m in np.unique(mag):
        idx = np.where(mag == m)[0]
        if len(idx) < 20: continue
        # use best latent cell (overall) for within-mag accuracy
        s_b, l_b = np.unravel_index(lat_acc.argmax(), lat_acc.shape)
        mag_acc[m] = cv_acc(lat[idx, s_b, l_b, :], y[idx], n_folds=3)
        print(f"  within-magnitude {m}: N={len(idx)}, acc={mag_acc[m]:.3f}  "
              f"at best (step={s_b+1}, L={l_b})")

    OUT_JSON.write_text(json.dumps({
        "N": int(N), "S": int(S), "L": int(L),
        "latent_acc": lat_acc.tolist(),
        "colon_acc_per_layer": col_acc.tolist(),
        "within_magnitude_acc": mag_acc,
    }, indent=2))
    print(f"saved {OUT_JSON}")

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        im = ax.imshow(lat_acc, aspect="auto", origin="lower", cmap="viridis",
                        vmin=0.25, vmax=1.0)
        ax.set_xlabel("layer"); ax.set_ylabel("latent step")
        ax.set_xticks(range(L)); ax.set_yticks(range(S))
        ax.set_yticklabels([str(s + 1) for s in range(S)])
        ax.set_title(f"GSM8K-strict CF latent 4-class operator probe (N={N})",
                     fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.04)
        for s in range(S):
            for l in range(L):
                v = lat_acc[s, l]
                if v >= 0.4 or v <= 0.3:
                    ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.6 else "black")
        ax = axes[1]
        ax.bar(range(L), col_acc, color="#1f77b4")
        ax.axhline(0.25, color="gray", ls=":", label="chance (1/4)")
        ax.set_xlabel("layer"); ax.set_ylabel("4-class accuracy")
        ax.set_xticks(range(L))
        ax.set_title(f"colon residual probe — best L={int(col_acc.argmax())} "
                     f"acc={col_acc.max():.2f}", fontsize=10, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1)
        fig.suptitle("Operator probe on GSM8K-style operator-pure CF "
                     "(strict labels, magnitude-balanced)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
