"""Per-(step, layer, head) attention from the last token TO number tokens in
the question, on GSM8K test.

flow_map_gsm8k already buckets attention by position class (Q, BOT, L1..L6,
EOT, D0..D9) but lumps ALL question tokens into the single 'Q' class. This
script breaks 'Q' apart by which question tokens are NUMBERS vs other text.

If CODI's latent loop is mostly question re-encoding (per the flow_map
finding that 72-91% of step attention goes to Q), the natural follow-up is:
which question tokens specifically? If most of Q attention is to number
tokens, the model is essentially recomputing arithmetic each step.

Captures:
  attn_num_frac[phase, step, layer, head] = mean over examples of
    sum(attn[last, p] for p in number_positions) /
    sum(attn[last, p] for p in any_Q_position)

Outputs:
  attn_to_numbers_gsm8k.npz   (attn_num_frac, attn_num_count, etc.)
  attn_to_numbers_gsm8k.{json,pdf}
"""
from __future__ import annotations

import json, os, re, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "codi"))

N_DECODE = 10


def main():
    BS = 16

    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=["c_attn", "c_proj", "c_fc"],
                          init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_atn", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=768,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=6,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False); model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16); model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id
    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    N_LAT = 6
    N_LAYERS = len(transformer.h)
    N_HEADS = model.codi.config.n_head

    ds = load_dataset("gsm8k", "main")["test"]
    questions = [ex["question"].strip().replace("  ", " ") for ex in ds]
    N = len(questions)
    print(f"  N={N}  layers={N_LAYERS} heads={N_HEADS}")

    # Aggregators: per (phase, step, layer, head) — fraction of Q-attention going to NUMBER tokens
    sum_num = np.zeros((2, N_DECODE, N_LAYERS, N_HEADS), dtype=np.float64)
    sum_q = np.zeros((2, N_DECODE, N_LAYERS, N_HEADS), dtype=np.float64)
    cnt = np.zeros((2, N_DECODE), dtype=np.int64)

    # Patterns for number tokens
    NUM_RE = re.compile(r"^-?\d+\.?\d*$")

    @torch.no_grad()
    def run_batch(qs):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        Lq_pad = batch["input_ids"].shape[1]
        ids = batch["input_ids"]
        q_lens = batch["attention_mask"].sum(dim=-1).cpu().numpy()
        # per-example NUMBER positions in the Q range
        num_masks = np.zeros((B, Lq_pad), dtype=bool)
        q_masks = np.zeros((B, Lq_pad), dtype=bool)
        for b in range(B):
            pad_amt = Lq_pad - int(q_lens[b])
            for p in range(pad_amt, Lq_pad):
                tk_id = int(ids[b, p].item())
                tk = tok.decode([tk_id]).strip()
                q_masks[b, p] = True
                if NUM_RE.match(tk):
                    num_masks[b, p] = True

        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)

        def aggregate_attn(phase_id, step, attns):
            # attns: tuple of per-layer (B, H, q_len, k_len). last-token row at index -1.
            for L_i, w in enumerate(attns):
                last_row = w[:, :, -1, :Lq_pad].float().cpu().numpy()  # (B, H, Lq_pad)
                for b in range(B):
                    qm = q_masks[b]; nm = num_masks[b]
                    if not qm.any(): continue
                    qsum = last_row[b][:, qm].sum(axis=-1)  # (H,)
                    nsum = last_row[b][:, nm].sum(axis=-1) if nm.any() else 0
                    sum_q[phase_id, step, L_i] += qsum
                    sum_num[phase_id, step, L_i] += nsum
            cnt[phase_id, step] += B

        for s in range(N_LAT):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             output_attentions=True, past_key_values=past)
            past = out.past_key_values
            aggregate_attn(0, s, out.attentions)
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)

        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        done = [False] * B
        for d in range(N_DECODE):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_attentions=True,
                              past_key_values=past)
            past = sout.past_key_values
            aggregate_attn(1, d, sout.attentions)
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if not done[b] and int(next_ids[b].item()) == eos_id:
                    done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)

    t0 = time.time()
    for s in range(0, N, BS):
        run_batch(questions[s:s+BS])
        if (s + BS) % 64 == 0 or s + BS >= N:
            print(f"  {min(s+BS, N)}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    mean_num = sum_num / np.where(cnt[:, :, None, None] == 0, 1, cnt[:, :, None, None])
    mean_q = sum_q / np.where(cnt[:, :, None, None] == 0, 1, cnt[:, :, None, None])
    num_frac = np.where(mean_q > 1e-9, mean_num / np.where(mean_q == 0, 1, mean_q), 0)
    # also a phase/step-only summary (avg over layers, heads)
    OUT_NPZ = PD / "attn_to_numbers_gsm8k.npz"
    np.savez(OUT_NPZ,
             mean_num=mean_num.astype(np.float32),
             mean_q=mean_q.astype(np.float32),
             num_frac=num_frac.astype(np.float32),
             cnt=cnt)
    OUT_JSON = PD / "attn_to_numbers_gsm8k.json"
    summary = {
        "N": int(N), "N_layers": int(N_LAYERS), "N_heads": int(N_HEADS),
        "N_latent_steps": int(N_LAT), "N_decode_steps": int(N_DECODE),
        "per_step_num_frac_avg_lh": {
            "latent": [float(num_frac[0, s].mean()) for s in range(N_LAT)],
            "decode": [float(num_frac[1, d].mean()) for d in range(N_DECODE)],
        },
        "per_step_q_attn_total_avg_lh": {
            "latent": [float(mean_q[0, s].mean()) for s in range(N_LAT)],
            "decode": [float(mean_q[1, d].mean()) for d in range(N_DECODE)],
        },
    }
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(f"saved {OUT_NPZ} and {OUT_JSON}")

    # --- Plot ---
    OUT_PDF = PD / "attn_to_numbers_gsm8k.pdf"
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: per-step number-attention fraction (avg over layers + heads)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        steps = np.arange(1, N_LAT + 1)
        ax.bar(steps, [num_frac[0, s].mean() * 100 for s in range(N_LAT)],
               color="#4c72b0", edgecolor="black")
        ax.set_xlabel("latent step"); ax.set_ylabel("% of Q attention to NUMBER tokens")
        ax.set_xticks(steps)
        ax.set_title("Fraction of Q-directed attention to number tokens — LATENT phase",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax = axes[1]
        decs = np.arange(1, N_DECODE + 1)
        ax.bar(decs, [num_frac[1, d].mean() * 100 for d in range(N_DECODE)],
               color="#dd8452", edgecolor="black")
        ax.set_xlabel("decode step"); ax.set_ylabel("% of Q attention to NUMBER tokens")
        ax.set_xticks(decs)
        ax.set_title("Fraction — DECODE phase",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("How much of each step's question-attention is to NUMBER tokens?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Per-(layer, head) heatmap for ALL 6 latent steps + first 4 decode steps
        plot_targets = [(0, s, f"latent step {s+1}") for s in range(N_LAT)] + \
                       [(1, d, f"decode step {d+1}") for d in range(4)]
        for phase, step, label in plot_targets:
            fig, ax = plt.subplots(figsize=(13, 6))
            G = num_frac[phase, step] * 100  # (layers, heads)
            im = ax.imshow(G, aspect="auto", cmap="viridis", vmin=0, vmax=max(50, float(G.max())))
            ax.set_xlabel("head"); ax.set_ylabel("layer")
            ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_LAYERS))
            for L in range(N_LAYERS):
                for H in range(N_HEADS):
                    if G[L, H] > 5:
                        ax.text(H, L, f"{G[L,H]:.0f}", ha="center", va="center",
                                fontsize=6, color="white" if G[L,H] < 30 else "black")
            ax.set_title(f"{label}: % of Q attention to NUMBER tokens, per (layer, head)",
                         fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.045, label="% to numbers")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
