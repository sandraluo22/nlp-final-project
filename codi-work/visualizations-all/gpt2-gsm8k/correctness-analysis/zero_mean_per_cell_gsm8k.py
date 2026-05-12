"""Zero + mean-resid per-cell ablation on GSM8K CF datasets.

Fills the gap left by patch_cf_mean_gsm8k (mean for attn+mlp only) and
patch_paired_cf_gsm8k (paired-CF patching for all three blocks). After this
runs we have, per (step, layer) cell:

  ┌─────────┬───────┬───────┬───────┐
  │ mode    │ attn  │ mlp   │ resid │
  ├─────────┼───────┼───────┼───────┤
  │ zero    │  new  │  new  │  new  │   ← this script
  │ mean    │ have  │ have  │  new  │   ← this script (resid only)
  │ patch   │ have  │ have  │ have  │   ← patch_paired_cf
  └─────────┴───────┴───────┴───────┘

For each CF dataset, runs the GSM8K-CF prompts (target side) and:
  1. Pre-pass collects mean(resid|step,layer) across all examples.
  2. Baseline pass: no ablation; record per-example baseline answer.
  3. Sweep cells: for each (block ∈ {attn,mlp,resid}, step, layer, mode ∈
     {zero} ∪ ({mean} if block=="resid")), ablate that single cell at the
     last-token position during that step, force-decode the answer, and
     record acc/delta/n_changed.

Output: zero_mean_per_cell_gsm8k.json
"""
from __future__ import annotations

import json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
PD = Path(__file__).resolve().parent
CF_DIR = REPO.parent / "cf-datasets"
sys.path.insert(0, str(REPO / "codi"))

CF_NAMES = [
    "gsm8k_vary_operator",
    "gsm8k_cf_op_strict",
    "gsm8k_cf_balanced",
    "gsm8k_cf_natural",
]


def codi_extract(s: str):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def main():
    BS = 16
    OUT_JSON = PD / "zero_mean_per_cell_gsm8k.json"

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
    targs = TrainingArguments(output_dir="/tmp/_zm", bf16=True,
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
    N_LAYERS = model.codi.config.n_layer
    HID = model.codi.config.n_embd
    N_LAT = 6

    # ---- Hooks ----
    # CAP drives mode per (block, step, layer). When CAP["mode"] is "zero",
    # the hook replaces output at the last position with zeros. When
    # "mean", it replaces with the precomputed mean for that (step, layer).
    # CAP["block"] selects which hook acts; "attn", "mlp", or "resid".
    CAP = {
        "active": False, "step": -1, "block": None,
        "ab_step": -1, "ab_layer": -1, "mode": "off",
        "mean_attn": None, "mean_mlp": None, "mean_resid": None,
        "capture_for_mean": False,
        "cap_attn_sum": None, "cap_mlp_sum": None, "cap_resid_sum": None,
        "cap_count": 0,
    }

    def make_attn_hook(idx):
        def fn(_m, _i, output):
            if CAP["capture_for_mean"] and CAP["step"] >= 0:
                CAP["cap_attn_sum"][CAP["step"], idx] += output[0][:, -1, :].sum(dim=0).float().detach().cpu().numpy()
            if (CAP["mode"] in ("zero", "mean") and CAP["block"] == "attn"
                    and CAP["step"] == CAP["ab_step"] and idx == CAP["ab_layer"]):
                a = output[0].clone()
                if CAP["mode"] == "zero":
                    a[:, -1, :] = 0
                else:
                    a[:, -1, :] = torch.from_numpy(CAP["mean_attn"][CAP["step"], idx]).to(a.device, dtype=a.dtype)
                return (a,) + output[1:]
            return output
        return fn

    def make_mlp_hook(idx):
        def fn(_m, _i, output):
            if CAP["capture_for_mean"] and CAP["step"] >= 0:
                CAP["cap_mlp_sum"][CAP["step"], idx] += output[:, -1, :].sum(dim=0).float().detach().cpu().numpy()
            if (CAP["mode"] in ("zero", "mean") and CAP["block"] == "mlp"
                    and CAP["step"] == CAP["ab_step"] and idx == CAP["ab_layer"]):
                o = output.clone()
                if CAP["mode"] == "zero":
                    o[:, -1, :] = 0
                else:
                    o[:, -1, :] = torch.from_numpy(CAP["mean_mlp"][CAP["step"], idx]).to(o.device, dtype=o.dtype)
                return o
            return output
        return fn

    def make_resid_hook(idx):
        """Hook the block's output (residual stream after attn+mlp+ln).
        We intercept transformer.h[idx]'s forward output."""
        def fn(_m, _i, output):
            # output is a tuple (hidden_state, presents, attentions) for GPT2Block
            h = output[0]
            if CAP["capture_for_mean"] and CAP["step"] >= 0:
                CAP["cap_resid_sum"][CAP["step"], idx] += h[:, -1, :].sum(dim=0).float().detach().cpu().numpy()
            if (CAP["mode"] in ("zero", "mean") and CAP["block"] == "resid"
                    and CAP["step"] == CAP["ab_step"] and idx == CAP["ab_layer"]):
                h = h.clone()
                if CAP["mode"] == "zero":
                    h[:, -1, :] = 0
                else:
                    h[:, -1, :] = torch.from_numpy(CAP["mean_resid"][CAP["step"], idx]).to(h.device, dtype=h.dtype)
                return (h,) + output[1:]
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))
        handles.append(blk.register_forward_hook(make_resid_hook(i)))

    @torch.no_grad()
    def run_batch(qs, *, ab_step=-1, ab_layer=-1, block=None, mode="off",
                  capture_mean=False):
        B = len(qs)
        CAP["ab_step"] = ab_step; CAP["ab_layer"] = ab_layer
        CAP["block"] = block; CAP["mode"] = mode
        CAP["capture_for_mean"] = capture_mean
        CAP["step"] = -1
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for s in range(N_LAT):
            CAP["step"] = s
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        CAP["step"] = -1
        CAP["mode"] = "off"; CAP["capture_for_mean"] = False
        # Decode answer
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
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    results = {}
    for cf_name in CF_NAMES:
        cf_path = CF_DIR / f"{cf_name}.json"
        if not cf_path.exists():
            print(f"  SKIP {cf_name}: {cf_path} not found")
            continue
        rows = json.load(open(cf_path))
        # Field name heuristics across the four CF schemas:
        #   vary_operator / cf_op_strict / cf_natural: question_concat + answer
        #   cf_balanced: cf_question_concat + cf_gold
        questions, golds = [], []
        for r in rows:
            q = r.get("cf_question_concat") or r.get("question_concat") or r.get("question") or r.get("q_A")
            g = r.get("cf_gold") if r.get("cf_gold") is not None else r.get("answer")
            if q is None or g is None: continue
            questions.append(str(q).strip().replace("  ", " "))
            try: golds.append(float(g))
            except Exception:
                m = re.search(r"-?\d+\.?\d*", str(g))
                golds.append(float(m.group()) if m else float("nan"))
        # Cap for tractability — sweep is 290 cells × N examples.
        N_CAP = 100
        if len(questions) > N_CAP:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(questions), size=N_CAP, replace=False)
            questions = [questions[int(i)] for i in sel]
            golds = [golds[int(i)] for i in sel]
        N = len(questions)
        if N == 0:
            print(f"  SKIP {cf_name}: no parsable rows")
            continue
        print(f"\n=== {cf_name}: N={N} ===", flush=True)
        golds = np.array(golds)

        # 1) Pre-pass: collect mean(attn, mlp, resid) per (step, layer)
        CAP["cap_attn_sum"]  = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
        CAP["cap_mlp_sum"]   = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
        CAP["cap_resid_sum"] = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
        t0 = time.time()
        for s in range(0, N, BS):
            run_batch(questions[s:s+BS], capture_mean=True)
        CAP["mean_attn"]  = (CAP["cap_attn_sum"]  / N).astype(np.float32)
        CAP["mean_mlp"]   = (CAP["cap_mlp_sum"]   / N).astype(np.float32)
        CAP["mean_resid"] = (CAP["cap_resid_sum"] / N).astype(np.float32)
        print(f"  collected means in {time.time()-t0:.0f}s", flush=True)

        # 2) Baseline
        base_strs = []
        for s in range(0, N, BS):
            base_strs += run_batch(questions[s:s+BS])
        base_ints = np.array([codi_extract(s) if codi_extract(s) is not None else np.nan for s in base_strs])
        base_correct = np.abs(base_ints - golds) < 1e-3
        base_acc = float(np.nanmean(base_correct))
        print(f"  baseline acc {base_acc*100:.1f}%", flush=True)

        # 3) Sweep
        conditions = {}
        t1 = time.time()
        # Build sweep list: zero(attn/mlp/resid) for each cell + mean(resid) for each cell
        sweep = []
        for s in range(N_LAT):
            for L in range(N_LAYERS):
                sweep.append(("attn",  "zero", s, L))
                sweep.append(("mlp",   "zero", s, L))
                sweep.append(("resid", "zero", s, L))
                sweep.append(("resid", "mean", s, L))
        for ci, (block, mode, s, L) in enumerate(sweep):
            strs = []
            for b in range(0, N, BS):
                strs += run_batch(questions[b:b+BS], ab_step=s, ab_layer=L,
                                  block=block, mode=mode)
            ints = np.array([codi_extract(x) if codi_extract(x) is not None else np.nan for x in strs])
            correct = np.abs(ints - golds) < 1e-3
            acc = float(np.nanmean(correct))
            n_changed = int(np.sum((ints != base_ints) & ~(np.isnan(ints) & np.isnan(base_ints))))
            wr = int(((~base_correct) & correct).sum())
            rw = int((base_correct & ~correct).sum())
            key = f"{block}_{mode}_step{s+1}_L{L}"
            conditions[key] = {
                "block": block, "mode": mode, "step": s+1, "layer": L,
                "acc": acc, "delta_acc": acc - base_acc,
                "n_changed": n_changed, "wrong_to_right": wr, "right_to_wrong": rw,
            }
            if (ci + 1) % 12 == 0 or ci + 1 == len(sweep):
                print(f"    {cf_name} sweep {ci+1}/{len(sweep)}  ({time.time()-t1:.0f}s)", flush=True)

        results[cf_name] = {
            "N": N, "N_LAT": N_LAT, "N_LAYERS": N_LAYERS,
            "baseline_accuracy": base_acc,
            "conditions": conditions,
        }
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(f"  saved partial results to {OUT_JSON}")

    print(f"\nsaved {OUT_JSON}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
