"""Find the (latent_step, layer) cell where the OPERATOR direction and a
NUMERAL/value direction are most orthogonal, then steer there. Hypothesis:
operator steering is least confounded by numeral content at the orthogonal
cell.

Operator dir = mean(add_correct) - mean(sub_correct) on cf_balanced
Numeral dir  = mean(parity=odd) - mean(parity=even) on svamp correct subset

Activations: cf_balanced and svamp_student inference runs (latent loop only,
shape (N, 6, 13, 768))."""

from __future__ import annotations
import json, os, re, sys, time, pickle
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))


def codi_extract(s: str):
    s = s.replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None


def parse_two_operands(equation):
    nums = re.findall(r"-?\d+\.?\d*", equation)
    if len(nums) < 2: return None
    try: return float(nums[0]), float(nums[1])
    except: return None


def main():
    print("loading activations + labels...")
    cf_acts = torch.load(REPO / "inference/runs/cf_balanced_student_gpt2/activations.pt",
                          map_location="cpu").to(torch.float32).numpy()  # (676, 6, 13, 768)
    sv_acts = torch.load(REPO / "inference/runs/gsm8k_latent_acts.pt",
                          map_location="cpu").to(torch.float32).numpy()
    cf_results = json.load(open(REPO / "inference/runs/cf_balanced_student_gpt2/results.json"))
    sv_results = json.load(open(REPO / "inference/runs/results.json"))
    cf_correct = np.array([r.get("correct", False) for r in cf_results])
    sv_correct = np.array([r.get("correct", False) for r in sv_results])

    cf_data = json.load(open(REPO.parent / "cf-datasets" / "cf_balanced.json"))
    cf_op = np.array([{"Addition": 0, "Subtraction": 1, "Multiplication": 2,
                        "Common-Division": 3}.get(d["type"], -1) for d in cf_data])[:cf_acts.shape[0]]

    ds = load_dataset("gsm8k")
    full = concatenate_datasets([ds["train"], ds["test"]])
    sv_gold = np.array([float(str(ex["Answer"]).replace(",", "")) for ex in full])[:sv_acts.shape[0]]
    sv_parity = (np.array([int(round(g)) for g in sv_gold]) % 2).astype(int)

    N_cf, S, Lp1, H = cf_acts.shape

    # === Compute operator and numeral direction per (latent_step, layer) ===
    print("\n=== Computing operator + numeral directions per cell ===")
    op_dir = np.zeros((S, Lp1, H), dtype=np.float32)
    num_dir = np.zeros((S, Lp1, H), dtype=np.float32)
    cf_keep = cf_correct & (cf_op >= 0)
    add_mask = cf_keep & (cf_op == 0)
    sub_mask = cf_keep & (cf_op == 1)
    print(f"  CF correct: add={add_mask.sum()}  sub={sub_mask.sum()}")
    sv_keep = sv_correct
    odd_mask = sv_keep & (sv_parity == 1)
    even_mask = sv_keep & (sv_parity == 0)
    print(f"  SVAMP correct: odd={odd_mask.sum()}  even={even_mask.sum()}")

    cossim = np.zeros((S, Lp1))
    for s in range(S):
        for l in range(Lp1):
            op_v = cf_acts[add_mask, s, l].mean(0) - cf_acts[sub_mask, s, l].mean(0)
            num_v = sv_acts[odd_mask, s, l].mean(0) - sv_acts[even_mask, s, l].mean(0)
            op_dir[s, l] = op_v; num_dir[s, l] = num_v
            cossim[s, l] = float(np.dot(op_v, num_v) /
                                  (np.linalg.norm(op_v) * np.linalg.norm(num_v) + 1e-9))
    print("\n  |cos_sim(op_dir, num_dir)| heatmap (rows = latent step, cols = layer):")
    print(np.array2string(np.abs(cossim), formatter={"float_kind": lambda x: f"{x:5.2f}"}, max_line_width=200))

    # Pick lowest |cos_sim| cell
    abs_cs = np.abs(cossim)
    bs, bl = np.unravel_index(int(np.argmin(abs_cs)), abs_cs.shape)
    print(f"\n  most-orthogonal cell: step {bs} L{bl}  cos_sim={cossim[bs, bl]:+.3f}")
    # Highest abs cos sim for contrast
    hs, hl = np.unravel_index(int(np.argmax(abs_cs)), abs_cs.shape)
    print(f"  most-aligned cell:    step {hs} L{hl}  cos_sim={cossim[hs, hl]:+.3f}")

    # ======================================================================
    # Steering at orthogonal cell on real SVAMP add and sub problems
    # ======================================================================
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"\nloading CODI-GPT-2 from {ckpt}", flush=True)
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
    targs = TrainingArguments(output_dir="/tmp/_orth", bf16=True,
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

    HOOK = {"active": False, "vec": None, "alpha": 0.0,
            "tgt_step": -1, "tgt_layer": -1, "cur_step": -99}

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if block_idx != HOOK["tgt_layer"] - 1: return output
            if HOOK["cur_step"] != HOOK["tgt_step"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn
    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, vec, alpha, layer, latent_step):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"vec": vec, "alpha": alpha, "tgt_step": latent_step,
                     "tgt_layer": layer, "cur_step": -99, "active": True})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for s in range(targs.inf_latent_iterations):
            HOOK["cur_step"] = s
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        HOOK["active"] = False
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(256):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
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

    add_idx = [i for i, ex in enumerate(full) if ex["Type"] == "Addition"]
    sub_idx = [i for i, ex in enumerate(full) if ex["Type"] == "Subtraction"]
    np.random.seed(0)
    add_eval = np.random.choice(add_idx, size=100, replace=False).tolist()
    sub_eval = np.random.choice(sub_idx, size=100, replace=False).tolist()

    def candidates(idx_list):
        cs = []
        for i in idx_list:
            ab = parse_two_operands(full[int(i)]["Equation"])
            if ab is None: cs.append(None); continue
            a, b = ab
            cs.append({"add": round(a+b), "sub": round(a-b),
                        "mul": round(a*b), "div": round(a/b) if b else None})
        return cs
    add_cs = candidates(add_eval); sub_cs = candidates(sub_eval)

    def gen(idx_list, vec, alpha, layer, latent_step):
        qs = [full[int(i)]["question_concat"].strip().replace("  ", " ") for i in idx_list]
        out = []
        for s in range(0, len(qs), 16):
            out += run_batch(qs[s:s+16], vec=vec, alpha=alpha, layer=layer, latent_step=latent_step)
        return out

    def score(strs, cs):
        o = {"add": 0, "sub": 0, "mul": 0, "div": 0, "n_valid": 0}
        for s, c in zip(strs, cs):
            v = codi_extract(s)
            if v is None: continue
            o["n_valid"] += 1
            if c is None: continue
            if v == c["add"]: o["add"] += 1
            if v == c["sub"]: o["sub"] += 1
            if c["mul"] is not None and v == c["mul"]: o["mul"] += 1
            if c["div"] is not None and v == c["div"]: o["div"] += 1
        return o

    H = cf_acts.shape[3]
    print(f"\n=== Baseline (no steering) ===")
    base_a = gen(add_eval, torch.zeros(H), 0.0, 0, 0)
    base_s = gen(sub_eval, torch.zeros(H), 0.0, 0, 0)
    print(f"  ADD baseline: {score(base_a, add_cs)}")
    print(f"  SUB baseline: {score(base_s, sub_cs)}")

    summary = {"orthogonal_cell": [int(bs), int(bl)],
                "orthogonal_cossim": float(cossim[bs, bl]),
                "aligned_cell": [int(hs), int(hl)],
                "aligned_cossim": float(cossim[hs, hl]),
                "add_baseline": score(base_a, add_cs),
                "sub_baseline": score(base_s, sub_cs)}

    for cell_name, (cs_step, cs_layer) in [("ORTHOGONAL", (bs, bl)),
                                              ("ALIGNED", (hs, hl))]:
        v_op = op_dir[cs_step, cs_layer]
        v_op = v_op / max(np.linalg.norm(v_op), 1e-9)
        v_op_t = torch.tensor(v_op, dtype=torch.float32)
        for alpha in [50.0, 100.0, 200.0, 400.0]:
            print(f"\n=== Cell {cell_name} (step {cs_step}, L{cs_layer})  alpha={alpha} ===")
            sa = gen(add_eval, v_op_t, alpha, cs_layer, cs_step)
            ss = gen(sub_eval, v_op_t, -alpha, cs_layer, cs_step)
            sc_a = score(sa, add_cs); sc_s = score(ss, sub_cs)
            print(f"  ADD steered (add->sub): {sc_a}")
            print(f"  SUB steered (sub->add): {sc_s}")
            summary[f"{cell_name}|alpha={alpha}|add"] = sc_a
            summary[f"{cell_name}|alpha={alpha}|sub"] = sc_s

    out = PD / "steer_orthogonal_cell.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
