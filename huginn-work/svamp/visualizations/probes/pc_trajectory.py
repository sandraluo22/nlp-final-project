"""Track how the operator-PC and number-PC evolve across K=1..32, and what
the difference (operator − number) direction looks like across recurrence steps.

For each model and one chosen core block:
  - At every K, fit PCA on vary_operator and on vary_a_2digit independently,
    take their top PC v_op[K] and v_num[K].
  - Compute:
      cos(v_op[K], v_num[K])         — alignment per K
      cos(v_op[K], v_op[K-1])        — operator direction rotation across K
      cos(v_num[K], v_num[K-1])      — number direction rotation across K
      diff[K] = v_op[K] − v_num[K]   (then normalize)
      cos(diff[K], diff[K-1])        — does the *difference* rotate coherently?
  - Project all PCs into the joint 2D PCA of the stacked trajectory and
    plot the trajectories.

Output:
  huginn-work/visualizations/probes/pc_trajectory_huginn.png
  huginn-work/visualizations/probes/pc_trajectory_gpt2.png
  huginn-work/visualizations/probes/pc_trajectory.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


HUGINN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = HUGINN_ROOT.parent
OUT_DIR = HUGINN_ROOT / "visualizations" / "probes"

EXPERIMENTS = [
    ("Huginn", 1,        # core block to analyze (Huginn LDA peak was at block 1, K=16)
        HUGINN_ROOT / "latent-sweep" / "huginn_vary_a_2digit" / "K32" / "activations.pt",
        HUGINN_ROOT / "latent-sweep" / "huginn_vary_operator" / "K32" / "activations.pt",
        "K"),
    ("CODI-GPT-2", 10,   # CODI-GPT-2 LDA peak was around layer 10
        PROJECT_ROOT / "codi-work" / "inference" / "runs" / "gpt2_vary_a_2digit" / "activations.pt",
        PROJECT_ROOT / "codi-work" / "inference" / "runs" / "gpt2_vary_operator" / "activations.pt",
        "latent step"),
]


def load(p): return torch.load(p, map_location="cpu", weights_only=True).float().numpy()


def top_pc(X):
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    pca.fit(X); return pca.components_[0]   # (H,)


def cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for label, layer, num_path, op_path, step_label in EXPERIMENTS:
        if not (num_path.exists() and op_path.exists()):
            print(f"[{label}] skipping (missing acts)"); continue
        print(f"\n========== {label} (block/layer {layer}) ==========")
        a_num = load(num_path)   # (N, S, L, H)
        a_op  = load(op_path)
        N_num, S, _, H = a_num.shape
        # Top PC per K, ABSOLUTE-VALUE-aligned to their K=1 reference (PCA sign is arbitrary)
        v_num = np.zeros((S, H), dtype=np.float32)
        v_op  = np.zeros((S, H), dtype=np.float32)
        for K in range(S):
            v_num[K] = top_pc(a_num[:, K, layer, :])
            v_op[K]  = top_pc(a_op[:,  K, layer, :])
            # sign-align successive K so trajectory doesn't flip arbitrarily
            if K > 0:
                if np.dot(v_num[K], v_num[K-1]) < 0: v_num[K] = -v_num[K]
                if np.dot(v_op[K],  v_op[K-1])  < 0: v_op[K]  = -v_op[K]

        # Pairwise alignments
        align_op_vs_num = np.array([cos(v_op[K], v_num[K]) for K in range(S)])
        rot_op = np.array([cos(v_op[K], v_op[K-1]) if K > 0 else 1.0 for K in range(S)])
        rot_num = np.array([cos(v_num[K], v_num[K-1]) if K > 0 else 1.0 for K in range(S)])

        diff = v_op - v_num
        diff_norm = np.linalg.norm(diff, axis=1)
        diff_unit = diff / (diff_norm[:, None] + 1e-12)
        rot_diff = np.array([cos(diff_unit[K], diff_unit[K-1]) if K > 0 else 1.0 for K in range(S)])
        # Also: does the diff direction equal a constant direction (use diff[0] as anchor)
        align_diff_to_K1 = np.array([cos(diff_unit[K], diff_unit[0]) for K in range(S)])

        print(f"  cos(v_op, v_num)     mean={align_op_vs_num.mean():.3f}  range [{align_op_vs_num.min():.3f}, {align_op_vs_num.max():.3f}]")
        print(f"  rotation v_op K→K+1: mean={rot_op[1:].mean():.3f}  min={rot_op[1:].min():.3f}")
        print(f"  rotation v_num K→K+1:mean={rot_num[1:].mean():.3f}  min={rot_num[1:].min():.3f}")
        print(f"  rotation diff K→K+1: mean={rot_diff[1:].mean():.3f}  min={rot_diff[1:].min():.3f}")
        print(f"  cos(diff[K], diff[1]): mean={align_diff_to_K1.mean():.3f}  range [{align_diff_to_K1.min():.3f}, {align_diff_to_K1.max():.3f}]")
        print(f"  ‖diff[K]‖₂: mean={diff_norm.mean():.3f}  min={diff_norm.min():.3f} max={diff_norm.max():.3f}")

        summary[label] = {
            "block": layer,
            "align_op_vs_num": align_op_vs_num.tolist(),
            "rot_op": rot_op.tolist(),
            "rot_num": rot_num.tolist(),
            "rot_diff": rot_diff.tolist(),
            "align_diff_to_K1": align_diff_to_K1.tolist(),
            "diff_norm": diff_norm.tolist(),
        }

        # Project all 3 trajectories (v_op, v_num, diff_unit) into a shared 2D space
        # via joint PCA of the stack (3*S, H).
        joint = np.concatenate([v_op, v_num, diff_unit], axis=0)
        joint_pca = PCA(n_components=2).fit(joint)
        proj = joint_pca.transform(joint).reshape(3, S, 2)

        Ks = np.arange(1, S + 1)
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

        # Panel 1: cosines per K
        ax = axes[0]
        ax.plot(Ks, align_op_vs_num, "o-", label="cos(v_op, v_num) per K", color="#d62728")
        ax.plot(Ks, rot_op,  "s-", label="cos(v_op, v_op_prev)",   color="#2ca02c", alpha=0.8)
        ax.plot(Ks, rot_num, "s-", label="cos(v_num, v_num_prev)", color="#1f77b4", alpha=0.8)
        ax.plot(Ks, rot_diff,"^--", label="cos(diff, diff_prev)",   color="#9467bd")
        ax.axhline(0.0, color="gray", lw=1, ls=":")
        ax.set_xlabel(step_label); ax.set_ylabel("cosine")
        ax.set_title(f"{label}  block {layer}: per-K alignments")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3); ax.set_ylim(-1.05, 1.05)

        # Panel 2: ‖diff[K]‖ across K
        ax = axes[1]
        ax.plot(Ks, diff_norm, "o-", color="#9467bd", lw=2)
        ax.set_xlabel(step_label); ax.set_ylabel("‖v_op[K] − v_num[K]‖₂")
        ax.set_title("Magnitude of difference vector")
        ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

        # Panel 3: 2D trajectory in joint PCA
        ax = axes[2]
        for arr, color, lab in [
            (proj[0], "#d62728", "v_op"),
            (proj[1], "#1f77b4", "v_num"),
            (proj[2], "#9467bd", "v_op − v_num"),
        ]:
            ax.plot(arr[:, 0], arr[:, 1], "-", color=color, alpha=0.5)
            ax.scatter(arr[:, 0], arr[:, 1], c=Ks, cmap="viridis",
                       edgecolors=color, s=40, linewidths=1.2, label=lab)
            ax.text(arr[0, 0], arr[0, 1], "K=1", fontsize=8, color=color)
            ax.text(arr[-1, 0], arr[-1, 1], f"K={S}", fontsize=8, color=color)
        ax.set_xlabel("joint-PCA dim 1"); ax.set_ylabel("joint-PCA dim 2")
        ax.set_title("Trajectory (color = K)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        fig.suptitle(f"{label}  ·  trajectory of operator-PC1 and number-PC1 across {step_label}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        out = OUT_DIR / f"pc_trajectory_{label.lower().replace('-', '').replace(' ', '_')}.png"
        fig.savefig(out, dpi=140)
        print(f"  saved {out}")

    (OUT_DIR / "pc_trajectory.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR/'pc_trajectory.json'}")


if __name__ == "__main__":
    main()
