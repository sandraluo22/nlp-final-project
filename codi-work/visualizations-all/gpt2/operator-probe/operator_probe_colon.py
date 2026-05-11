"""4-class operator probe at the ':' residual on SVAMP (CODI-GPT-2)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PD = Path(__file__).resolve().parent
ACTS = PD / "svamp_colon_acts.pt"
META = PD / "svamp_colon_acts_meta.json"
OUT_PDF = PD / "operator_probe_colon.pdf"
OUT_JSON = PD / "operator_probe_colon.json"

OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]


def main():
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    meta = json.load(open(META))
    N, L, H = a.shape
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y = np.array([op_to_idx.get(t, -1) for t in meta["types"]])
    valid = y >= 0
    a, y = a[valid], y[valid]
    N2 = len(y)
    print(f"colon acts shape={a.shape}  N={N2} (valid ops)")
    print(f"  class counts: {dict(zip(*np.unique([OPS[i] for i in y], return_counts=True)))}")
    baseline = float(np.max([np.mean(y == c) for c in range(4)]))

    idx_tr, idx_te = train_test_split(np.arange(N2), test_size=0.2, random_state=0, stratify=y)
    y_tr, y_te = y[idx_tr], y[idx_te]

    train_acc = np.zeros(L); test_acc = np.zeros(L)
    per_class_acc = np.zeros((L, 4))
    for l in range(L):
        X = a[:, l, :]
        sc = StandardScaler().fit(X[idx_tr])
        Xtr, Xte = sc.transform(X[idx_tr]), sc.transform(X[idx_te])
        clf = LogisticRegression(max_iter=2000, C=0.1,
                                 solver="lbfgs").fit(Xtr, y_tr)
        train_acc[l] = accuracy_score(y_tr, clf.predict(Xtr))
        pred_te = clf.predict(Xte)
        test_acc[l] = accuracy_score(y_te, pred_te)
        for c in range(4):
            mask = (y_te == c)
            if mask.any():
                per_class_acc[l, c] = (pred_te[mask] == c).mean()
    l_best = int(test_acc.argmax())
    print(f"\nLayer | test_acc")
    for l in range(L):
        print(f"  {l:>2} | {test_acc[l]*100:6.2f}%")
    print(f"\nbest layer={l_best} test_acc={test_acc[l_best]*100:.1f}% (baseline {baseline*100:.1f}%)")

    OUT_JSON.write_text(json.dumps({
        "shape": [N2, L, H], "ops": OPS, "baseline": baseline,
        "train_acc": train_acc.tolist(), "test_acc": test_acc.tolist(),
        "per_class_acc": per_class_acc.tolist(),
        "best_layer": l_best, "best_test_acc": float(test_acc[l_best]),
    }, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(L)
    axes[0].plot(xs, train_acc * 100, "o-", label="train")
    axes[0].plot(xs, test_acc * 100, "s-", label="test")
    axes[0].axhline(baseline * 100, color="gray", ls=":", label=f"baseline {baseline*100:.1f}%")
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("4-class operator probe @ ':' residual")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 100)

    im = axes[1].imshow(per_class_acc.T, aspect="auto", origin="lower",
                        cmap="viridis", vmin=0, vmax=1)
    axes[1].set_xlabel("layer"); axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(OPS)
    axes[1].set_title("Per-class test accuracy")
    fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)

    fig.suptitle(f"CODI-GPT-2 operator probe at ':' residual (SVAMP N={N2})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=140)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
