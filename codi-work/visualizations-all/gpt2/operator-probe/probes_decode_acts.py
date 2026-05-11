"""Probe (4)-style analyses on the answer-position activations: at the first
decode forward, the model is committing to the answer token. We probe that
position's residual at every layer for:

  - units digit of gold answer (10-class)
  - tens digit of gold answer (10-class)
  - operator type (4-class)
  - PCA of activations colored by answer magnitude

Plus the analog of (1) and (3) at the answer position: rank of the gold
token + projection onto the gold token's unembedding row.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parents[2]
DECODE = REPO / "experiments" / "computation_probes" / "svamp_decode_acts.pt"
OUT = REPO / "experiments" / "computation_probes"


def main():
    print(f"loading {DECODE}", flush=True)
    a = torch.load(DECODE, map_location="cpu", weights_only=True).float().numpy()
    print(f"  shape={a.shape}", flush=True)              # (1000, 13, 768)
    N, L, H = a.shape

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array([t.replace("Common-Divison", "Common-Division") for t in full["Type"]])[:N]
    answers = np.array([float(str(x).replace(",", "")) for x in full["Answer"]])[:N]

    # Probes per layer (no step axis since this is the single decode forward)
    units = (answers.astype(int) % 10).astype(int)
    tens  = ((answers.astype(int) // 10) % 10).astype(int)
    op_idx = np.array(
        [{n: i for i, n in enumerate(np.unique(types))}[t] for t in types]
    )

    def probe(labels, label_name):
        idx = np.arange(N)
        tr, te = train_test_split(idx, test_size=0.2, random_state=0,
                                  stratify=labels if len(np.unique(labels)) > 1 else None)
        accs = np.empty(L, dtype=np.float32)
        for layer in range(L):
            X = a[:, layer, :]
            clf = LogisticRegression(max_iter=300, n_jobs=1)
            clf.fit(X[tr], labels[tr])
            accs[layer] = clf.score(X[te], labels[te])
        peak = int(accs.argmax())
        print(f"  {label_name:<14} peak={accs.max()*100:>5.1f}% at L{peak}", flush=True)
        return accs

    print("\n=== probes on answer-position activations ===")
    units_acc = probe(units, "units_digit")
    tens_acc  = probe(tens,  "tens_digit")
    op_acc    = probe(op_idx, "operator")

    # Logit-lens / projection at decode-position
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    W = model.get_output_embeddings().weight.detach().cpu().numpy()
    gold_ids = []
    for ans in answers:
        toks = tok(f" {int(ans) if ans == int(ans) else ans}", add_special_tokens=False)["input_ids"]
        gold_ids.append(toks[0] if toks else tok.eos_token_id)
    gold_ids = np.array(gold_ids)
    gold_W = W[gold_ids]                                      # (N, H)

    proj_per_layer = np.empty((L,), dtype=np.float32)
    rank_per_layer = np.empty((L,), dtype=np.float32)
    pct_top1 = np.empty((L,), dtype=np.float32)
    for layer in range(L):
        R = a[:, layer, :]
        proj = (R * gold_W).sum(axis=1)
        proj_per_layer[layer] = proj.mean()
        logits = R @ W.T
        gold_logit = logits[np.arange(N), gold_ids]
        rank = (logits > gold_logit[:, None]).sum(axis=1)
        rank_per_layer[layer] = np.median(rank)
        pct_top1[layer] = (rank == 0).mean() * 100

    # Plot summary
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    Ls = np.arange(L)
    ax = axes[0]
    ax.plot(Ls, units_acc * 100, "o-", label="units digit (chance 10%)")
    ax.plot(Ls, tens_acc * 100,  "s-", label="tens digit (chance 10%)")
    ax.plot(Ls, op_acc * 100,    "^-", label="operator (chance 25%)")
    ax.axhline(10, ls=":", color="gray", lw=1)
    ax.axhline(25, ls=":", color="gray", lw=1)
    ax.set_xlabel("layer (0=embedding, 1..12=blocks)")
    ax.set_ylabel("probe accuracy (%)")
    ax.set_title("(4) Variable probes at ANSWER position\n(after EOT, model is producing the answer)")
    ax.set_xticks(Ls); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    ax = axes[1]
    ax.plot(Ls, proj_per_layer, "o-", color="#9467bd", lw=2)
    ax.set_xlabel("layer"); ax.set_ylabel("residual · W[gold_token] (mean across N=1000)")
    ax.set_title("(3) Projection onto gold-answer-token at answer position")
    ax.set_xticks(Ls); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(Ls, pct_top1, "o-", color="#d62728", lw=2)
    ax.set_xlabel("layer"); ax.set_ylabel("% top-1 (gold token is argmax)")
    ax.set_title("(1) Logit-lens top-1 % at answer position")
    ax.set_xticks(Ls); ax.grid(alpha=0.3); ax.set_ylim(0, 100)

    fig.suptitle("CODI-GPT-2  ·  decode-position activations  ·  watching the answer commit",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_p = OUT / "gpt2_decode_probes.png"
    fig.savefig(out_p, dpi=140); plt.close(fig)
    print(f"saved {out_p}")

    summary = {
        "units_digit_acc_per_layer": units_acc.tolist(),
        "tens_digit_acc_per_layer":  tens_acc.tolist(),
        "operator_acc_per_layer":    op_acc.tolist(),
        "proj_mean_per_layer":  proj_per_layer.tolist(),
        "median_rank_per_layer": rank_per_layer.tolist(),
        "pct_top1_per_layer":   pct_top1.tolist(),
    }
    (OUT / "gpt2_decode_probes.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT/'gpt2_decode_probes.json'}")


if __name__ == "__main__":
    main()
