"""GSM8K version of helix Clock test on latent residuals (no operand probes;
just gold-answer encoding at each (step, layer, period)).

Tests whether the latent loop encodes the gold answer on Fourier helices.
Since GSM8K is multi-step (no clean a/b operand structure), we only probe
for the gold answer's helical encoding here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[3]
LAT_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
META_PATH = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts_meta.json"
COL_PATH = REPO / "experiments" / "computation_probes" / "gsm8k_colon_acts.pt"
PD = Path(__file__).resolve().parent
OUT_JSON = PD / "helix_clock_gsm8k.json"
OUT_PDF = PD / "helix_clock_gsm8k.pdf"

PERIODS = [2, 5, 10, 50, 100, 1000, 10000]


def fourier_target(n, T):
    return np.stack([np.cos(2 * np.pi * n / T), np.sin(2 * np.pi * n / T)], axis=-1)


def cv_r2(X, Y, alpha=1.0, n_folds=5, seed=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y, dtype=np.float64)
    for tr, te in kf.split(X):
        clf = Ridge(alpha=alpha).fit(X[tr], Y[tr])
        preds[te] = clf.predict(X[te])
    ss_res = float(np.sum((Y - preds) ** 2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main():
    meta = json.load(open(META_PATH))
    gold = np.array([np.nan if v is None else float(v) for v in meta["gold"]])
    keep = ~np.isnan(gold)
    # Latents
    lat = torch.load(LAT_PATH, map_location="cpu", weights_only=True).float().numpy()
    lat = lat[keep]
    g = gold[keep]
    # Colon
    col = torch.load(COL_PATH, map_location="cpu", weights_only=True).float().numpy()
    col = col[:lat.shape[0]] if col.shape[0] == lat.shape[0] else col[keep[:col.shape[0]]]
    # Truncate or filter to match N
    N = min(lat.shape[0], col.shape[0])
    lat = lat[:N]; col = col[:N]; g = g[:N]
    # Trim outliers in gold (GSM8K has answers up to 2.8M; restrict to small range
    # so periods are meaningful)
    mask = (g >= 0) & (g <= 10000)
    lat = lat[mask]; col = col[mask]; g = g[mask]
    print(f"After filtering to gold ∈ [0, 10000]: N={lat.shape[0]}, "
          f"gold range [{g.min():.0f},{g.max():.0f}]")

    S, L = lat.shape[1], lat.shape[2]
    results = {"periods": PERIODS, "N": int(lat.shape[0]),
               "latent": {"R2_gold_per_step_layer_period": None},
               "colon": {"R2_gold_per_layer_period": None}}

    # Latent sweep
    print("Sweeping LATENT (step × layer × period)...")
    R2_lat = np.zeros((S, L, len(PERIODS)))
    for s in range(S):
        for l in range(L):
            X = lat[:, s, l, :]
            for ti, T in enumerate(PERIODS):
                R2_lat[s, l, ti] = cv_r2(X, fourier_target(g, T))
        print(f"  step {s+1}: best R² @T=100 = {R2_lat[s, :, PERIODS.index(100)].max():.3f}, "
              f"@T=1000 = {R2_lat[s, :, PERIODS.index(1000)].max():.3f}")
    results["latent"]["R2_gold_per_step_layer_period"] = R2_lat.tolist()

    # Colon sweep
    print("Sweeping COLON (layer × period)...")
    Lc = col.shape[1]
    R2_col = np.zeros((Lc, len(PERIODS)))
    for l in range(Lc):
        X = col[:, l, :]
        for ti, T in enumerate(PERIODS):
            R2_col[l, ti] = cv_r2(X, fourier_target(g, T))
    results["colon"]["R2_gold_per_layer_period"] = R2_col.tolist()
    print("  best per period:")
    for ti, T in enumerate(PERIODS):
        bl = int(np.argmax(R2_col[:, ti]))
        print(f"    T={T:5d}: best L{bl} R²={R2_col[bl, ti]:+.3f}")

    OUT_JSON.write_text(json.dumps(results, indent=2))

    with PdfPages(OUT_PDF) as pdf:
        # Latent: one heatmap per period.
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for ti, T in enumerate(PERIODS):
            ax = axes[ti // 4, ti % 4]
            im = ax.imshow(R2_lat[:, :, ti], aspect="auto", origin="lower",
                           cmap="viridis", vmin=-0.5, vmax=1.0)
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_yticks(range(S)); ax.set_yticklabels([str(s+1) for s in range(S)])
            ax.set_title(f"T={T}, max R²={R2_lat[:, :, ti].max():.2f}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.04)
        if len(PERIODS) < 8: axes[1, 3].axis("off")
        fig.suptitle(f"GSM8K — helix R²(gold) on LATENT residual (N={lat.shape[0]})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Colon: line per period.
        fig, ax = plt.subplots(figsize=(11, 5))
        for ti, T in enumerate(PERIODS):
            ax.plot(range(Lc), R2_col[:, ti], "o-", lw=2, label=f"T={T}")
        ax.axhline(0, color="black", lw=0.3)
        ax.set_xlabel("layer"); ax.set_ylabel("R²(gold)"); ax.set_xticks(range(Lc))
        ax.set_title(f"GSM8K — helix R²(gold) at COLON residual (N={col.shape[0]})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_JSON} and {OUT_PDF}")


if __name__ == "__main__":
    main()
