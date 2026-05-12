"""Per-head attention ablation + probe accuracy test.

For each of 144 (layer, head) cells:
  1. Zero that head's contribution at the prompt-end position.
  2. Re-run the model on 200 SVAMP examples; capture residual at the cell
     where the operator probe is strongest (pos 1, L8 from earlier).
  3. Apply a probe TRAINED ON BASELINE ACTIVATIONS to predict operator.
  4. Report accuracy under the ablation.

A head whose ablation drops probe accuracy is encoding operator info in its
contribution to the residual.

Also records per-op model correctness so we can disentangle "the probe drops
because the residual changed" vs "the probe drops because the model fails."
"""

from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
PROBE_POS, PROBE_LAYER = 1, 8   # cell where op probe got 91% in earlier run


def codi_extract(s: str):
    s = s.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    # === 1. Train baseline probe on SVAMP activations (transfer source) ===
    print("loading SVAMP baseline activations + training op probe...", flush=True)
    base_acts = torch.load(PD / "svamp_fixed_acts.pt", map_location="cpu").to(torch.float32).numpy()
    svamp = load_dataset("ChilleD/SVAMP")
    svamp_full = svamp["train"].select(range(len(svamp["train"])))  # just to materialize
    # Concatenate SVAMP train + test (probe was originally fit on combined SVAMP)
    from datasets import concatenate_datasets as _concat
    svamp_full = _concat([svamp["train"], svamp["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in svamp_full])

    # SVAMP train/eval split for the probe itself
    np.random.seed(0)
    perm = np.random.permutation(len(operators))
    train_idx = perm[:800]
    test_idx  = perm[800:]
    X_train = base_acts[train_idx, PROBE_POS, PROBE_LAYER, :]
    y_train = operators[train_idx]
    X_test  = base_acts[test_idx,  PROBE_POS, PROBE_LAYER, :]
    y_test  = operators[test_idx]
    mask_tr = y_train >= 0; mask_te = y_test >= 0
    sc = StandardScaler().fit(X_train[mask_tr])
    clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs").fit(
        sc.transform(X_train[mask_tr]), y_train[mask_tr])
    train_acc = clf.score(sc.transform(X_train[mask_tr]), y_train[mask_tr])
    test_acc  = clf.score(sc.transform(X_test[mask_te]),  y_test[mask_te])
    print(f"  SVAMP-fit op probe at (pos={PROBE_POS}, L={PROBE_LAYER}): "
          f"train_acc={train_acc*100:.1f}%  SVAMP-held-out_acc={test_acc*100:.1f}%")

    # === 2. Load CODI ===
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
    targs = TrainingArguments(output_dir="/tmp/_phab", bf16=True,
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
            x = inputs[0]
            x = x.clone()
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
        """Run inference with attention head h ablated at prompt-end. Capture
        residual at (pos=PROBE_POS, layer=PROBE_LAYER) of the FIRST decode
        positions. Return (captured_acts, predicted_strings)."""
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
        for _ in range(targs.inf_latent_iterations):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        # decode and capture residual at PROBE_POS
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        captured = None
        for p in range(64):  # decode at most 64 tokens for accuracy + capture
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True,
                              output_hidden_states=(p == PROBE_POS),
                              past_key_values=past)
            past = sout.past_key_values
            if p == PROBE_POS:
                captured = sout.hidden_states[PROBE_LAYER][:, -1, :].to(torch.float32).cpu().numpy()
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

    # === GSM8K eval set: 200 problems, label = first-marker operator ===
    op_char_to_idx = {"+": 0, "-": 1, "*": 2, "/": 3}
    gsm = load_dataset("gsm8k", "main")["test"]
    gsm_qs, gsm_ops, gsm_golds = [], [], []
    for ex in gsm:
        ans = ex["answer"].replace(",", "")
        mm = re.search(r"<<(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=", ans)
        mg = re.search(r"####\s*(-?\d+\.?\d*)", ans)
        if mm is None or mg is None:
            continue
        gsm_qs.append(ex["question"].strip().replace("  ", " "))
        gsm_ops.append(op_char_to_idx.get(mm.group(2), -1))
        gsm_golds.append(float(mg.group(1)))
    np.random.seed(0)
    eval_idx = np.random.choice(len(gsm_qs), size=200, replace=False)
    eval_qs = [gsm_qs[int(i)] for i in eval_idx]
    eval_ops = np.array([gsm_ops[int(i)] for i in eval_idx])
    eval_golds = np.array([gsm_golds[int(i)] for i in eval_idx])

    BS = 16
    def run_full(layer, head):
        all_caps = []
        all_strs = []
        for s in range(0, 200, BS):
            cap, strs = run_batch_capture(eval_qs[s:s+BS], layer=layer, head=head)
            all_caps.append(cap); all_strs += strs
        return np.concatenate(all_caps, axis=0), all_strs

    # Baseline (no ablation)
    print("\n=== Baseline (no ablation) ===", flush=True)
    base_cap, base_strs = run_full(-1, -1)
    base_pred = clf.predict(sc.transform(base_cap))
    mask = eval_ops >= 0
    base_probe_acc = float(np.mean(base_pred[mask] == eval_ops[mask]))
    base_correct = sum(1 for s, g in zip(base_strs, eval_golds)
                        if codi_extract(s) is not None and abs(codi_extract(s) - g) < 1e-3)
    print(f"  baseline SVAMP-fit op probe on GSM8K: {base_probe_acc*100:.1f}%  "
          f"task accuracy: {base_correct}/200")

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
            print(f"  L{L:2d} H{H:2d}: probe_acc={probe_acc*100:5.1f}% "
                  f"(Δ {(probe_acc - base_probe_acc)*100:+.1f}pp)  "
                  f"correct={n_correct}/200  ({time.time()-t0:.0f}s)", flush=True)

    out = Path(__file__).resolve().parent / "per_head_ablation_svamp2gsm8k_probe.json"
    out.write_text(json.dumps({
        "base_probe_acc": float(base_probe_acc),
        "base_n_correct": int(base_correct),
        "grid_probe_acc": grid.tolist(),
        "grid_correct":   grid_correct.tolist(),
    }, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
