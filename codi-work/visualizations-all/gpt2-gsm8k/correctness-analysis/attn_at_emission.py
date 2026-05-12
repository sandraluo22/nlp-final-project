"""Attention at emission time: where does the answer-decoding position read from?

For each SVAMP / CF example we run CODI-GPT-2 normally and capture attention
weights from the FIRST answer-emission token's position back to all prior
positions. We aggregate that attention mass by token category:

   prompt_template  — non-numeric prompt tokens (template words)
   prompt_operand_a — token positions corresponding to the first operand
                      number's digits
   prompt_operand_b — token positions for the second operand's digits
   prompt_operator  — operator words ("more", "less", "times", etc.)
   bot              — the BOT special token
   latent_step_1..6 — the 6 latent-loop positions
   eot              — the EOT special token

For each (layer, head), we get the fraction of attention mass each category
received, averaged over examples. This tells us WHERE the model looks to
find the answer.

We also rank heads by:
   - mean attention to operand tokens (the candidate "operand readers")
   - mean attention to latent tokens   (the candidate "latent readers")
   - attention sparsity / entropy

Output: attn_at_emission.{json, pdf}.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

CF_SETS = ["gsm8k_vary_operator"]
OUT_JSON = PD / "attn_at_emission_gsm8k.json"
OUT_PDF = PD / "attn_at_emission_gsm8k.pdf"


def codi_extract(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def load_cf(name):
    rows = json.load(open(CF_DIR / f"{name}.json"))
    qs = [r["question_concat"].strip().replace("  ", " ") for r in rows]
    golds = [float(r["answer"]) for r in rows]
    return qs, golds, rows


def find_operand_positions(input_ids, tok, a_value, b_value):
    """Locate the token positions in input_ids that correspond to digits of
    operands a and b. Returns (positions_a, positions_b) as Python lists.

    Strategy: tokenize " {a}" and " {b}" each on their own and search for
    the resulting token-id subsequence inside input_ids. If multiple matches
    (e.g. operand digit also appears elsewhere) take the first.
    """
    text_ids = input_ids.tolist()
    def find_sub(needle):
        out = []
        for s in range(len(text_ids) - len(needle) + 1):
            if text_ids[s:s+len(needle)] == needle:
                out.append(list(range(s, s+len(needle))))
        return out
    candidates_a = []
    candidates_b = []
    for prefix in (" ", ""):
        ids_a = tok.encode(f"{prefix}{int(a_value)}", add_special_tokens=False)
        ids_b = tok.encode(f"{prefix}{int(b_value)}", add_special_tokens=False)
        matches_a = find_sub(ids_a)
        matches_b = find_sub(ids_b)
        if matches_a: candidates_a.extend(matches_a)
        if matches_b: candidates_b.extend(matches_b)
    # Pick first match for each — prefer one that doesn't overlap the other.
    pos_a = candidates_a[0] if candidates_a else []
    pos_b = []
    for cb in candidates_b:
        if not set(cb) & set(pos_a):
            pos_b = cb; break
    if not pos_b and candidates_b:
        pos_b = candidates_b[0]
    return pos_a, pos_b


def main():
    BS = 8           # smaller batch — attention output is large
    N_LIMIT = 60     # cap per CF set to keep run fast
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
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
    targs = TrainingArguments(output_dir="/tmp/_ae", bf16=True,
                              use_lora=True, use_prj=True, prj_dim=768,
                              prj_no_ln=False, prj_dropout=0.0,
                              num_latent=6, inf_latent_iterations=6,
                              remove_eos=True, greedy=True,
                              model_max_length=512, seed=11)
    model = CODI(margs, targs, lora_cfg)
    sd_safe = Path(ckpt) / "model.safetensors"
    sd_bin = Path(ckpt) / "pytorch_model.bin"
    sd = load_file(str(sd_safe)) if sd_safe.exists() else torch.load(str(sd_bin), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.codi.tie_weights()
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", model_max_length=512,
                                                     padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id or tok.convert_tokens_to_ids("[PAD]")
    model = model.to("cuda").to(torch.bfloat16); model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    eos_id = tok.eos_token_id

    N_LAYERS = model.codi.config.n_layer
    N_HEADS = model.codi.config.n_head
    print(f"  layers={N_LAYERS}  heads_per_layer={N_HEADS}")

    @torch.no_grad()
    def run_one_with_attn(q, a_val, b_val):
        """Run one example and capture attention from the FIRST answer-emission
        token to all prior positions. Returns:
          attn_mass_per_layer_head_by_category: (L, H, n_categories) array
          tokens_decoded: list of emitted strings
          info: dict with positions of operands/template/latent.
        """
        batch = tok([q], return_tensors="pt", padding="longest").to("cuda")
        prompt_ids = batch["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])
        bot = torch.full((1, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        # Track positions:
        #   0..prompt_len-1     prompt tokens
        #   prompt_len          BOT
        #   prompt_len+1..+6    latent steps 1..6
        #   prompt_len+7        EOT
        #   prompt_len+8...     emission tokens
        latent_positions = []
        for step in range(6):
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            o = model.codi(inputs_embeds=latent, attention_mask=attn,
                           use_cache=True, output_hidden_states=True,
                           past_key_values=past)
            past = o.past_key_values
            latent = o.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
            latent_positions.append(prompt_len + 1 + step)
        # Emission: first emit returns attention weights from the
        # emission position over the entire prefix (prompt + BOT + latents + EOT).
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
        eot_position = prompt_len + 1 + 6
        s = model.codi(inputs_embeds=output, attention_mask=attn,
                       use_cache=True, output_hidden_states=False,
                       output_attentions=True,
                       past_key_values=past)
        past = s.past_key_values
        # attentions: tuple of (1, n_heads, q_len=1, k_len=eot_position+1) per layer
        attns = s.attentions
        first_emit_attn = np.zeros((N_LAYERS, N_HEADS, eot_position + 1), dtype=np.float32)
        for li, a in enumerate(attns):
            first_emit_attn[li] = a[0, :, 0, :].float().cpu().numpy()

        # Decode a few more emission tokens to record what the model said
        nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
        emitted = [int(nid.item())]
        for _ in range(48):
            if emitted[-1] == eos_id: break
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(nid).unsqueeze(1)
            s = model.codi(inputs_embeds=output, attention_mask=attn,
                           use_cache=True, past_key_values=past)
            past = s.past_key_values
            nid = torch.argmax(s.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
            emitted.append(int(nid.item()))
        decoded = tok.decode(emitted, skip_special_tokens=True)

        # Locate operand positions inside prompt_ids
        pos_a, pos_b = find_operand_positions(prompt_ids, tok, a_val, b_val)

        categories = {
            "prompt_template": [],
            "prompt_operand_a": pos_a,
            "prompt_operand_b": pos_b,
            "bot": [prompt_len],
            "latent": latent_positions,
            "eot": [eot_position],
        }
        used = set(pos_a) | set(pos_b) | set(categories["bot"]) \
               | set(categories["latent"]) | set(categories["eot"])
        for p in range(prompt_len):
            if p not in used: categories["prompt_template"].append(p)

        # Aggregate
        agg = {}
        for cat, ps in categories.items():
            if ps:
                agg[cat] = first_emit_attn[:, :, ps].sum(axis=-1)
            else:
                agg[cat] = np.zeros((N_LAYERS, N_HEADS), dtype=np.float32)

        return agg, decoded, {
            "prompt_len": prompt_len, "pos_a": pos_a, "pos_b": pos_b,
            "eot_position": eot_position, "latent_positions": latent_positions,
        }

    results = {}
    for cf_name in CF_SETS:
        print(f"\n=== CF set: {cf_name} ===", flush=True)
        qs, golds, rows = load_cf(cf_name)
        N = min(len(qs), N_LIMIT)
        cats = ["prompt_template", "prompt_operand_a", "prompt_operand_b",
                "bot", "latent", "eot"]
        sums = {c: np.zeros((N_LAYERS, N_HEADS), dtype=np.float64) for c in cats}
        counts = 0
        n_correct = 0
        per_ex = []
        t0 = time.time()
        for i in range(N):
            a_val = rows[i].get("a"); b_val = rows[i].get("b")
            if a_val is None or b_val is None: continue
            agg, decoded, info = run_one_with_attn(qs[i], a_val, b_val)
            pred = codi_extract(decoded)
            correct = pred is not None and abs(pred - golds[i]) < 1e-3
            n_correct += int(correct)
            for c in cats:
                sums[c] += agg[c]
            counts += 1
            per_ex.append({"idx": i, "a": a_val, "b": b_val, "gold": golds[i],
                           "pred": pred, "correct": bool(correct),
                           "decoded": decoded,
                           "len_pos_a": len(info["pos_a"]),
                           "len_pos_b": len(info["pos_b"]),
                           })
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{N}  acc={n_correct/(i+1):.2f}  ({time.time()-t0:.0f}s)",
                      flush=True)
        means = {c: (sums[c] / max(counts, 1)).tolist() for c in cats}
        results[cf_name] = {
            "N": counts, "n_correct": n_correct, "accuracy": n_correct / max(counts, 1),
            "category_mean_attn_per_layer_head": means,
            "categories": cats, "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS,
            "per_example": per_ex,
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_JSON}")

    # PDF
    with PdfPages(OUT_PDF) as pdf:
        for cf_name, r in results.items():
            cats = r["categories"]
            means = {c: np.array(r["category_mean_attn_per_layer_head"][c]) for c in cats}
            # Page 1: per-category layer-aggregate attention.
            fig, ax = plt.subplots(figsize=(12, 5))
            xs = np.arange(r["N_LAYERS"])
            colors = {"prompt_template": "#7f7f7f", "prompt_operand_a": "#d62728",
                      "prompt_operand_b": "#9467bd", "bot": "#1f77b4",
                      "latent": "#2ca02c", "eot": "#ff7f0e"}
            for c in cats:
                ax.plot(xs, means[c].sum(axis=1), "o-", lw=2, color=colors[c],
                        label=c)
            ax.set_xlabel("layer"); ax.set_ylabel("mean attention mass from emission position")
            ax.set_title(f"{cf_name} — total attention mass per category (sum over heads)",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
            ax.set_xticks(xs)
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)

            # Page 2: per (layer, head) heatmap of attention to operand_a
            for tgt in ["prompt_operand_a", "prompt_operand_b", "latent",
                        "prompt_template"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                M = means[tgt]   # (L, H)
                im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                               vmin=0.0, vmax=max(0.3, M.max() * 1.05))
                ax.set_xlabel("head"); ax.set_ylabel("layer")
                ax.set_title(f"{cf_name} — mean attention to {tgt}  "
                             f"(emission position, N={r['N']})",
                             fontsize=10, fontweight="bold")
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                             label="mean attention mass")
                for l in range(M.shape[0]):
                    for h in range(M.shape[1]):
                        v = M[l, h]
                        if v >= 0.15:
                            ax.text(h, l, f"{v:.2f}", ha="center", va="center",
                                    fontsize=6, color="white" if v < 0.4 else "black")
                fig.tight_layout()
                pdf.savefig(fig, dpi=140); plt.close(fig)

            # Page N: top attention heads ranked
            fig, ax = plt.subplots(figsize=(13, 6))
            heads_op = means["prompt_operand_a"] + means["prompt_operand_b"]
            heads_lt = means["latent"]
            top_op = sorted([(l, h, heads_op[l, h]) for l in range(r["N_LAYERS"])
                             for h in range(r["N_HEADS"])], key=lambda x: -x[2])[:15]
            top_lt = sorted([(l, h, heads_lt[l, h]) for l in range(r["N_LAYERS"])
                             for h in range(r["N_HEADS"])], key=lambda x: -x[2])[:15]
            xs = np.arange(15); w = 0.4
            ax.bar(xs - w/2, [v for _, _, v in top_op], w, color="#d62728",
                   label="attn to operands (top 15)")
            ax.bar(xs + w/2, [v for _, _, v in top_lt], w, color="#2ca02c",
                   label="attn to latents (top 15)")
            labels_op = [f"L{l}.h{h}" for l, h, _ in top_op]
            labels_lt = [f"L{l}.h{h}" for l, h, _ in top_lt]
            ax.set_xticks(xs)
            ax.set_xticklabels([f"op:{l_op} / lat:{l_lt}"
                                for l_op, l_lt in zip(labels_op, labels_lt)],
                               rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("mean attention mass")
            ax.set_title(f"{cf_name} — top heads by attention mass",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig, dpi=140); plt.close(fig)

    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
