"""Project the latent-loop residual at each (step, layer) onto the model's
'answer direction' per example.

Answer direction = the lm_head row for the first BPE token of " {gold}" (with
leading space, matching CODI's post-EOT emission template).

Output:
  - answer_direction_projection.npz : per (step, layer) mean of:
      * raw inner product  <h, w_ans>
      * cosine similarity   cos(h, w_ans)
      * fraction of examples where w_ans is in the top-K of lm_head @ ln_f(h)
  - answer_direction_projection.pdf : heatmaps + per-step layerwise curves
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
PD = REPO / "experiments" / "computation_probes"
ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
LM_HEAD = PD / "codi_gpt2_lm_head.npy"
LN_F = PD / "codi_gpt2_ln_f.npz"
OUT_NPZ = PD / "answer_direction_projection_gsm8k.npz"
OUT_PDF = PD / "answer_direction_projection_gsm8k.pdf"
TOPK_LIST = [1, 5, 10, 50]


def main():
    print("loading activations + lm_head + ln_f")
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    # shape (N=1000, S=6, L=13, H=768)
    N, S, L, H = a.shape
    print(f"  acts shape={a.shape}")
    W = np.load(LM_HEAD)  # (vocab, H) — lm_head weights
    ln_f = np.load(LN_F)
    gamma = ln_f["weight"]; beta = ln_f["bias"]
    print(f"  lm_head shape={W.shape}; ln_f: gamma {gamma.shape}, beta {beta.shape}")

    # Targets: first BPE token of " {gold}" per example (leading-space form)
    ds = load_dataset("gsm8k", "main")
    full = concatenate_datasets([ds["train"], ds["test"]])
    golds = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    def first_tok(g):
        gs = str(int(g)) if float(g).is_integer() else str(g)
        ids = tok.encode(" " + gs, add_special_tokens=False)
        return ids[0] if ids else -1

    targets = np.array([first_tok(g) for g in golds])
    print(f"  N={N}; sample targets: {[(golds[i], targets[i], tok.decode([targets[i]])) for i in range(5)]}")
    valid = targets >= 0
    print(f"  valid targets: {int(valid.sum())}/{N}")

    # Per-example answer direction in raw-residual space.
    # GPT-2's lm_head is tied to the input embedding; the relevant scoring is
    # logit(token) = ln_f(h) @ W[token]. So the 'answer direction' in raw
    # residual space is W[target] but the residual needs ln_f first. Two ways:
    #   - project ln_f(h) onto W[target]  (what the model literally does)
    #   - project h onto something pre-ln_f
    # Using the post-ln_f version because it's what determines emission.
    W_target = W[targets]  # (N, H)

    # Apply ln_f to acts: gamma * (h - mean) / std + beta, per row.
    # Layer-norm normalizes across the H dimension.
    def apply_ln_f(h):  # h: (..., H)
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        normed = (h - mean) / np.sqrt(var + 1e-5)
        return normed * gamma + beta

    print("computing projections (this iterates over 6 steps × 13 layers)...")
    proj_mean = np.zeros((S, L))   # mean <h_normed, w_ans> / ||h_normed||
    cos_mean = np.zeros((S, L))    # mean cos(h_normed, w_ans)
    proj_z = np.zeros((S, L))      # mean z-score of target logit vs all vocab
    topk_in = {k: np.zeros((S, L)) for k in TOPK_LIST}

    for s in range(S):
        for l in range(L):
            h = a[:, s, l, :]                         # (N, H)
            h_n = apply_ln_f(h)                       # (N, H)
            # cos sim per example
            num = (h_n * W_target).sum(axis=-1)       # (N,)
            den = np.linalg.norm(h_n, axis=-1) * np.linalg.norm(W_target, axis=-1) + 1e-12
            cos_mean[s, l] = float(num[valid].mean() / 1.0)  # mean of <,> (not normalized)
            proj_mean[s, l] = float((num / den)[valid].mean())  # mean cos
            # z-score: how far above the average logit is the answer token's logit?
            logits = h_n @ W.T                        # (N, vocab)
            mu = logits.mean(axis=-1)
            sd = logits.std(axis=-1) + 1e-12
            target_logit = logits[np.arange(N), targets]
            z = (target_logit - mu) / sd
            proj_z[s, l] = float(z[valid].mean())
            # top-K hit rate
            for k in TOPK_LIST:
                topk_ids = np.argpartition(-logits, kth=k, axis=-1)[:, :k]
                hit = (topk_ids == targets[:, None]).any(axis=-1)
                topk_in[k][s, l] = float(hit[valid].mean())
        print(f"  step {s+1}/{S} done")

    np.savez(OUT_NPZ,
             proj_mean=proj_mean.astype(np.float32),
             cos_mean=cos_mean.astype(np.float32),
             proj_z=proj_z.astype(np.float32),
             **{f"topk{k}": v.astype(np.float32) for k, v in topk_in.items()})
    print(f"saved {OUT_NPZ}")

    # ===== Slideshow =====
    with PdfPages(OUT_PDF) as pdf:
        # Slide 1: cos sim heatmap
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, M, title in [(axes[0], proj_mean, "Mean cos(h_normed, w_answer) per (step, layer)"),
                              (axes[1], proj_z, "Mean z-score of answer-token logit (within vocab)")]:
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_yticks(range(S)); ax.set_yticklabels([str(s+1) for s in range(S)])
            ax.set_title(title, fontsize=10)
            for s in range(S):
                for l in range(L):
                    v = M[s, l]
                    if abs(v) >= (0.4 * (M.max() - M.min())):
                        ax.text(l, s, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < (M.max()+M.min())/2 else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.suptitle("Answer-direction alignment in the latent loop (SVAMP, N=1000)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 2: top-K hit rates
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        for ax, k in zip(axes.ravel(), TOPK_LIST):
            M = topk_in[k]
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
            ax.set_xlabel("layer"); ax.set_ylabel("latent step")
            ax.set_yticks(range(S)); ax.set_yticklabels([str(s+1) for s in range(S)])
            ax.set_title(f"Frac examples where answer token in top-{k} of lm_head", fontsize=10)
            for s in range(S):
                for l in range(L):
                    v = M[s, l]
                    if v >= 0.05:
                        ax.text(l, s, f"{v*100:.0f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.5 else "black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.suptitle("Top-K hit rate of the answer token at each (step, layer) — SVAMP",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 3: line plots per step
        fig, ax = plt.subplots(figsize=(11, 5))
        for s in range(S):
            ax.plot(range(L), proj_mean[s], "o-", label=f"step {s+1}")
        ax.set_xlabel("layer"); ax.set_ylabel("mean cos(h, w_answer)")
        ax.set_title("Answer-direction cosine similarity per layer, per latent step",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

        # Slide 4: top-1 hit rate per step
        fig, ax = plt.subplots(figsize=(11, 5))
        for s in range(S):
            ax.plot(range(L), topk_in[1][s] * 100, "o-", label=f"step {s+1}")
        ax.set_xlabel("layer"); ax.set_ylabel("frac examples where answer is top-1 (%)")
        ax.set_title("Top-1 answer-token rate per layer, per latent step",
                     fontsize=11, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, max(20, topk_in[1].max()*100*1.2))
        fig.tight_layout(); pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
