"""Watch operator and number representations interact during CODI-GPT-2's
latent computation, on saved SVAMP activations.

Runs three analyses on the already-saved (N=1000, S=6, L=13, H=768) tensor:

  (1) Logit lens at the last-prompt-token, per (layer, step)
       project residual onto the model's unembedding,
       record the rank of the gold-answer-string's first token.

  (3) Direct projection onto the gold-answer-token's unembedding row
       — "how much does the model 'point at' the right answer at this depth?"
       Trajectory across layers.

  (4) Computational-variable probes on the activations:
       - units digit of the gold answer  (10 classes)
       - tens digit of the gold answer    (≤10 classes)
       - operator type                    (4 classes)
       For each (layer, step), fit a linear probe; chart accuracy.

Outputs:
  codi-work/experiments/computation_probes/gpt2_logit_lens.json
  codi-work/experiments/computation_probes/gpt2_answer_proj.png
  codi-work/experiments/computation_probes/gpt2_var_probes.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parents[3]                     # codi-work/
PROJECT = REPO.parent
ACTS = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
OUT_DIR = REPO / "experiments" / "computation_probes"


def load_acts():
    print(f"loading {ACTS}", flush=True)
    a = torch.load(ACTS, map_location="cpu", weights_only=True).float().numpy()
    print(f"  shape={a.shape}  bytes={a.nbytes/1e9:.2f} GB", flush=True)
    return a


def load_svamp_labels(n):
    ds = load_dataset("gsm8k", "main")
    full = _GSMShim(ds["test"])
    types = np.array([t.replace("Common-Divison", "Common-Division") for t in full["Type"]])[:n]
    answers = np.array([float(str(a).replace(",", "")) for a in full["Answer"]])[:n]
    return types, answers


def load_unembedding():
    print("loading GPT-2 unembedding", flush=True)
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    W = model.get_output_embeddings().weight.detach().cpu().numpy()  # (vocab, H)
    print(f"  W shape: {W.shape}", flush=True)
    return W, tok


def gold_token_ids(answers, tok):
    """For each gold answer, return the token id of its FIRST token (incl. the
    leading space variant — most common GPT-2 tokenisation for numbers in text)."""
    ids = []
    for a in answers:
        # Use leading-space variant ' 8' since SVAMP gold prints as plain int.
        s = f" {int(a) if a == int(a) else a}"
        toks = tok(s, add_special_tokens=False)["input_ids"]
        ids.append(toks[0] if toks else tok.eos_token_id)
    return np.array(ids, dtype=np.int64)


def logit_lens_and_projection(acts, W, tok, gold_ids, answers):
    """For each (layer, step), compute:
      proj[ex, layer, step] = residual[ex, step, layer] · W[gold_ids[ex]]
      rank[ex, layer, step] = rank of gold token among all vocab logits"""
    N, S, L, H = acts.shape
    print(f"  logit lens over (L={L}, S={S}) = {L*S} cells", flush=True)
    proj = np.empty((N, L, S), dtype=np.float32)
    rank = np.empty((N, L, S), dtype=np.float32)
    # Gather W rows for each example
    gold_W = W[gold_ids]                                           # (N, H)
    for layer in range(L):
        for step in range(S):
            R = acts[:, step, layer, :]                             # (N, H)
            proj[:, layer, step] = (R * gold_W).sum(axis=1)
            # rank computation: project onto whole vocab
            logits = R @ W.T                                        # (N, V)
            # rank of gold_id: number of vocab tokens with > logit than gold's
            gold_logit = logits[np.arange(N), gold_ids]
            rank[:, layer, step] = (logits > gold_logit[:, None]).sum(axis=1)
        if layer % 4 == 0 or layer == L - 1:
            print(f"    layer {layer}/{L-1} done", flush=True)
    return proj, rank


def fit_probes(acts, labels, n_splits=1, seed=0):
    """Per-(layer, step) linear probe accuracy for the given labels."""
    N, S, L, H = acts.shape
    grid = np.empty((L, S), dtype=np.float32)
    idx = np.arange(N)
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                              stratify=labels if len(np.unique(labels)) > 1 else None)
    for layer in range(L):
        for step in range(S):
            X = acts[:, step, layer, :]
            clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=1)
            clf.fit(X[tr], labels[tr])
            grid[layer, step] = clf.score(X[te], labels[te])
    return grid


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    acts = load_acts()
    N = acts.shape[0]
    types, answers = load_svamp_labels(N)
    W, tok = load_unembedding()
    gold_ids = gold_token_ids(answers, tok)

    # --- (1) + (3) Logit lens + projection on gold answer ---
    print("\n=== (1)+(3) Logit lens + projection ===")
    proj, rank = logit_lens_and_projection(acts, W, tok, gold_ids, answers)
    L = proj.shape[1]; S = proj.shape[2]

    # Aggregate: mean projection and median rank across examples, per (layer, step)
    proj_mean = proj.mean(axis=0)         # (L, S)
    rank_med  = np.median(rank, axis=0)
    rank_mean = rank.mean(axis=0)

    # Plot: 2-panel — projection trajectory + rank trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    for s in range(S):
        ax.plot(range(L), proj_mean[:, s], "o-", label=f"latent step {s+1}", lw=1.6, ms=4)
    ax.set_xlabel("layer (0=embedding, 1..12=blocks)")
    ax.set_ylabel("residual · W[gold_token]  (mean across N=1000)")
    ax.set_title("(3) Direct projection onto gold-answer-token row of unembedding")
    ax.set_xticks(range(L)); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for s in range(S):
        ax.plot(range(L), rank_med[:, s], "o-", label=f"latent step {s+1}", lw=1.6, ms=4)
    ax.set_xlabel("layer")
    ax.set_ylabel("median rank of gold token  (lower is better, 0 = top-1)")
    ax.set_title("(1) Logit-lens rank of gold answer token")
    ax.set_xticks(range(L)); ax.set_yscale("symlog", linthresh=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"CODI-GPT-2 SVAMP  ·  N={N}  ·  watching the answer token emerge across depth",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_p = OUT_DIR / "gpt2_answer_proj.png"
    fig.savefig(out_p, dpi=140); plt.close(fig)
    print(f"saved {out_p}")

    # Save numbers
    summary = {
        "N": int(N), "shape": list(acts.shape),
        "proj_mean_layer_step": proj_mean.tolist(),
        "rank_median_layer_step": rank_med.tolist(),
        "rank_mean_layer_step":   rank_mean.tolist(),
        "frac_top1_layer_step":   (rank == 0).mean(axis=0).tolist(),
        "frac_top10_layer_step":  (rank < 10).mean(axis=0).tolist(),
        "frac_top100_layer_step": (rank < 100).mean(axis=0).tolist(),
    }
    (OUT_DIR / "gpt2_logit_lens.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT_DIR/'gpt2_logit_lens.json'}")
    # Quick text summary
    print(f"\n  rank-0 (top-1) % over (layer, step), N={N}:")
    pct_top1 = (rank == 0).mean(axis=0) * 100
    print("    layer\\step  " + "  ".join(f"s{s+1:>2d}" for s in range(S)))
    for layer in range(L):
        print(f"    L{layer:>2d}        " + "  ".join(f"{pct_top1[layer, s]:>4.1f}" for s in range(S)))

    # --- (4) Computational-variable probes ---
    print("\n=== (4) Computational-variable probes ===")
    # units digit of gold answer
    units = (answers.astype(int) % 10).astype(int)
    tens  = ((answers.astype(int) // 10) % 10).astype(int)
    # operator label
    op_map = {n: i for i, n in enumerate(np.unique(types))}
    op_idx = np.array([op_map[t] for t in types])

    print("  fitting units-digit probes...", flush=True)
    grid_units = fit_probes(acts, units)
    print("  fitting tens-digit probes...", flush=True)
    grid_tens = fit_probes(acts, tens)
    print("  fitting operator probes...", flush=True)
    grid_op = fit_probes(acts, op_idx)

    # Plot: 3-panel heatmap
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, grid, title, vmin, vmax in [
        (axes[0], grid_units, "units digit of answer  (10-class, chance ≈ 10%)", 0, 100),
        (axes[1], grid_tens,  "tens digit of answer (10-class, chance ≈ 10%)",  0, 100),
        (axes[2], grid_op,    "operator type (4-class, chance ≈ 25%)",          20, 100),
    ]:
        im = ax.imshow(grid * 100, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                       origin="lower")
        ax.set_xlabel("latent step (1..6)")
        ax.set_ylabel("layer (0..12)")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([str(s+1) for s in range(grid.shape[1])])
        ax.set_yticks(range(grid.shape[0]))
        ax.set_title(title, fontsize=10)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                ax.text(c, r, f"{grid[r, c]*100:.0f}", ha="center", va="center", fontsize=6,
                        color="black" if grid[r, c] > 0.5 else "white")
        plt.colorbar(im, ax=ax, label="probe acc (%)")
    fig.suptitle("CODI-GPT-2  ·  per-(layer, step) probe accuracy on computational variables",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_p = OUT_DIR / "gpt2_var_probes.png"
    fig.savefig(out_p, dpi=140); plt.close(fig)
    print(f"saved {out_p}")

    summary_p = {
        "units_digit_acc": grid_units.tolist(),
        "tens_digit_acc":  grid_tens.tolist(),
        "operator_acc":    grid_op.tolist(),
        "operator_classes": list(op_map.keys()),
    }
    (OUT_DIR / "gpt2_var_probes.json").write_text(json.dumps(summary_p, indent=2))
    print(f"saved {OUT_DIR/'gpt2_var_probes.json'}")

    # Print headline numbers
    print(f"\n  peak units-digit probe acc: {grid_units.max()*100:.1f} % "
          f"at (L={int(np.unravel_index(grid_units.argmax(), grid_units.shape)[0])}, "
          f"S={int(np.unravel_index(grid_units.argmax(), grid_units.shape)[1])+1})")
    print(f"  peak tens-digit probe acc:  {grid_tens.max()*100:.1f} %")
    print(f"  peak operator probe acc:    {grid_op.max()*100:.1f} %")


if __name__ == "__main__":
    main()
