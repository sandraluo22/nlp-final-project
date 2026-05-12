"""GSM8K-native per-head attention ablation, scored with a GSM8K-fit op probe.

This is the GSM8K-native counterpart to per_head_ablation_svamp2gsm8k_probe.py.
Both use a 12-layer × 12-head ablation grid, but here BOTH the probe and the
ablation are on GSM8K.

For each of 144 (layer, head) cells:
  1. Zero that head's contribution at the prompt-end position.
  2. Run CODI on a held-out subset of GSM8K test; capture residual at the
     GSM8K best-op cell — (latent_step=0, layer=11) from multi_op_probe.
  3. Apply a probe TRAINED ON BASELINE GSM8K activations at the same cell
     (first-marker operator label) to predict operator.
  4. Report probe accuracy + task accuracy under the ablation.

Output: per_head_ablation_gsm8k_native.json
"""

from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
ACTS_PATH = REPO / "visualizations-all" / "gpt2" / "counterfactuals" / "gsm8k_latent_acts.pt"
sys.path.insert(0, str(REPO / "codi"))

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
# From multi_op_probe_gsm8k.json: op m=1 best cell is step=1 (1-indexed), L=11
PROBE_POS = 0          # 0-indexed latent step (= step 1 in the 1-indexed sense)
PROBE_LAYER = 11
N_EVAL = 200

OP_CHAR_TO_IDX = {"+": 0, "-": 1, "*": 2, "/": 3}


def codi_extract(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def first_marker_op_and_gold(answer: str):
    a = answer.replace(",", "")
    mm = re.search(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=", a)
    mg = re.search(r"####\s*(-?\d+\.?\d*)", a)
    if mm is None or mg is None:
        return None, None
    return OP_CHAR_TO_IDX.get(mm.group(2), -1), float(mg.group(1))


def main():
    # === 1. Build GSM8K op labels + golds aligned with gsm8k_latent_acts.pt ===
    print("loading GSM8K + activations...", flush=True)
    ds = load_dataset("gsm8k", "main")["test"]
    ops, golds, questions = [], [], []
    for ex in ds:
        op, gold = first_marker_op_and_gold(ex["answer"])
        if op is None:
            ops.append(-1); golds.append(np.nan); questions.append("")
        else:
            ops.append(op); golds.append(gold)
            questions.append(ex["question"].strip().replace("  ", " "))
    ops = np.array(ops)
    golds = np.array(golds)
    acts_all = torch.load(ACTS_PATH, map_location="cpu", weights_only=True).float().numpy()
    # acts shape: (N_total, S, L+1, H)
    N_total = acts_all.shape[0]
    assert N_total == len(ops), f"{N_total} vs {len(ops)}"

    # === 2. Train op probe at (PROBE_POS, PROBE_LAYER) on GSM8K ===
    print(f"training op probe at (step={PROBE_POS+1}, L={PROBE_LAYER})...", flush=True)
    valid = ops >= 0
    rng = np.random.default_rng(0)
    perm = rng.permutation(N_total)
    train_idx = perm[:int(N_total * 0.8)]
    held_idx  = perm[int(N_total * 0.8):]
    train_idx = train_idx[valid[train_idx]]
    held_idx  = held_idx[valid[held_idx]]
    X_tr = acts_all[train_idx, PROBE_POS, PROBE_LAYER, :]
    y_tr = ops[train_idx]
    X_te = acts_all[held_idx,  PROBE_POS, PROBE_LAYER, :]
    y_te = ops[held_idx]
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs").fit(
        sc.transform(X_tr), y_tr)
    tr_acc = clf.score(sc.transform(X_tr), y_tr)
    te_acc = clf.score(sc.transform(X_te), y_te)
    print(f"  GSM8K-fit op probe: train={tr_acc*100:.1f}%  held-out={te_acc*100:.1f}%")

    # === 3. Pick N_EVAL ablation examples from the HELD-OUT set ===
    eval_perm = rng.permutation(len(held_idx))[:N_EVAL]
    eval_idx_global = held_idx[eval_perm]
    eval_qs = [questions[int(i)] for i in eval_idx_global]
    eval_ops = ops[eval_idx_global]
    eval_golds = golds[eval_idx_global]
    print(f"  eval set: {len(eval_qs)} held-out GSM8K problems")
    print(f"  op-class distribution: ",
          {k: int((eval_ops == v).sum()) for k, v in OP_CHAR_TO_IDX.items()})

    # === 4. Load CODI ===
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_phgsm", bf16=True,
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
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    embed_fn = model.get_embd(model.codi, model.model_name)
    transformer = (model.codi.transformer if hasattr(model.codi, "transformer")
                   else model.codi.base_model.model.transformer)
    eos_id = tok.eos_token_id

    HOOK = {"phase": "off", "layer": -1, "head": -1, "active": False}

    def make_pre_hook(layer_idx):
        def fn(module, inputs):
            if not HOOK["active"]: return None
            if HOOK["phase"] != "prompt_end": return None
            if HOOK["layer"] != layer_idx: return None
            x = inputs[0].clone()
            h = HOOK["head"]
            x[:, -1, h*HEAD_DIM:(h+1)*HEAD_DIM] = 0
            return (x,) + inputs[1:]
        return fn

    handles = []
    for L, blk in enumerate(transformer.h):
        attn_mod = getattr(blk, "self_attn", None) or getattr(blk, "attn", None)
        c_proj = attn_mod.c_proj
        handles.append(c_proj.register_forward_pre_hook(make_pre_hook(L)))

    @torch.no_grad()
    def run_batch_capture(qs, *, layer, head):
        """Run inference with attn head h ablated at prompt-end. Capture
        residual at (PROBE_POS, PROBE_LAYER) of the latent loop, and decode."""
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"phase": "prompt_end", "layer": layer, "head": head, "active": layer >= 0})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        HOOK["active"] = False

        captured = None
        for s in range(targs.inf_latent_iterations):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            if s == PROBE_POS:
                captured = out.hidden_states[PROBE_LAYER][:, -1, :].to(torch.float32).cpu().numpy()
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)

        # Greedy decode for task-accuracy
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(48):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B):
                if not done[b]:
                    tokens[b].append(int(next_ids[b].item()))
                    if int(next_ids[b].item()) == eos_id:
                        done[b] = True
            if all(done): break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        return captured, [tok.decode(t, skip_special_tokens=True) for t in tokens]

    BS = 16
    def run_full(layer, head):
        all_caps, all_strs = [], []
        for s in range(0, N_EVAL, BS):
            cap, strs = run_batch_capture(eval_qs[s:s+BS], layer=layer, head=head)
            all_caps.append(cap); all_strs += strs
        return np.concatenate(all_caps, axis=0), all_strs

    # Baseline
    print("\n=== Baseline (no ablation) ===", flush=True)
    base_cap, base_strs = run_full(-1, -1)
    base_pred = clf.predict(sc.transform(base_cap))
    mask = eval_ops >= 0
    base_probe_acc = float(np.mean(base_pred[mask] == eval_ops[mask]))
    base_correct = sum(1 for s, g in zip(base_strs, eval_golds)
                        if codi_extract(s) is not None and abs(codi_extract(s) - g) < 1e-3)
    print(f"  baseline GSM8K-fit op probe on GSM8K: {base_probe_acc*100:.1f}%  "
          f"task accuracy: {base_correct}/{N_EVAL}")

    # Per-head grid
    grid = np.zeros((N_LAYERS, N_HEADS), dtype=float)
    grid_correct = np.zeros((N_LAYERS, N_HEADS), dtype=int)
    t0 = time.time()
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            cap, strs = run_full(L, H)
            pred = clf.predict(sc.transform(cap))
            probe_acc = float(np.mean(pred[mask] == eval_ops[mask]))
            n_correct = sum(1 for s, g in zip(strs, eval_golds)
                             if codi_extract(s) is not None and abs(codi_extract(s) - g) < 1e-3)
            grid[L, H] = probe_acc; grid_correct[L, H] = n_correct
            print(f"  L{L:2d} H{H:2d}: probe={probe_acc*100:5.1f}% "
                  f"(Δ {(probe_acc - base_probe_acc)*100:+.1f}pp)  "
                  f"correct={n_correct}/{N_EVAL}  ({time.time()-t0:.0f}s)", flush=True)

    out = Path(__file__).resolve().parent / "per_head_ablation_gsm8k_native.json"
    out.write_text(json.dumps({
        "probe_pos": PROBE_POS, "probe_layer": PROBE_LAYER, "n_eval": N_EVAL,
        "probe_train_acc": float(tr_acc), "probe_heldout_acc": float(te_acc),
        "base_probe_acc": float(base_probe_acc),
        "base_n_correct": int(base_correct),
        "grid_probe_acc": grid.tolist(),
        "grid_correct":   grid_correct.tolist(),
    }, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
