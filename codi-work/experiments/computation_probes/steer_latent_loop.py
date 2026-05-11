"""LATENT LOOP STEERING: intervene at each of CODI's 6 internal thinking
steps with the operator direction derived from svamp_like correct subset.

State machine for the hook:
  current_phase = 'prompt' / 'latent_step_X' / 'decode'
  We only inject at the chosen (latent_step, layer) cell.

For each of several (latent_step, layer) cells, sweep alpha and measure:
  on real SVAMP add problems steered toward sub: matches a-b instead of a+b?
  on real SVAMP sub problems steered toward add: matches a+b instead of a-b?
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
    print("loading svamp_like latent activations...")
    blob = torch.load(PD / "svamp_like_latent_acts.pt", map_location="cpu")
    latent_acts = blob["latent"].to(torch.float32).numpy()       # (N, 6, L+1, H)
    pe_acts = blob["prompt_end"].to(torch.float32).numpy()       # (N, L+1, H)
    meta = json.load(open(PD / "svamp_like_latent_meta.json"))
    correct = np.array(meta["correct"], dtype=bool)
    op_idx = np.array([m["op_idx"] for m in meta["meta"]])
    N, S, Lp1, H = latent_acts.shape
    print(f"  latent: {latent_acts.shape}  prompt_end: {pe_acts.shape}")
    print(f"  correct: {correct.sum()}/{N}")

    # CORRECT-subset per-op means at each (latent_step, layer)
    means_lat = np.zeros((4, S, Lp1, H), dtype=np.float32)
    means_pe  = np.zeros((4, Lp1, H), dtype=np.float32)
    for c in range(4):
        m = (op_idx == c) & correct
        if m.sum() == 0: continue
        means_lat[c] = latent_acts[m].mean(axis=0)
        means_pe[c]  = pe_acts[m].mean(axis=0)

    # Diff vectors (add->sub at each latent step+layer, and at prompt_end)
    add_to_sub_lat = means_lat[1] - means_lat[0]    # (S, L+1, H)
    add_to_sub_pe  = means_pe[1]  - means_pe[0]     # (L+1, H)

    # SNR per cell to pick best ones
    snr_lat = np.zeros((S, Lp1))
    grand_mean_lat = latent_acts.mean(axis=0)        # (S, L+1, H)
    var_per_cell = ((latent_acts - grand_mean_lat[None]) ** 2).mean(axis=(0, 3))
    diff_sq = (add_to_sub_lat ** 2).sum(axis=2)
    snr_lat = diff_sq / np.sqrt(var_per_cell + 1e-9)
    print("\n  SNR(add->sub) per (latent_step × layer):")
    print(np.array2string(snr_lat, formatter={"float_kind": lambda x: f"{x:7.0f}"}, max_line_width=200))

    best = np.unravel_index(int(np.argmax(snr_lat)), snr_lat.shape)
    print(f"  best (latent_step, layer): step {best[0]} L{best[1]}  SNR={snr_lat[best]:.0f}")

    # Cells to test
    CELLS = {
        "BEST":     ("latent", best[0], best[1]),
        "STEP0_L8": ("latent", 0, 8),
        "STEP3_L6": ("latent", 3, 6),
        "STEP5_L11":("latent", 5, 11),
        "PROMPT_L8":("prompt_end", -1, 8),
    }

    # ---- Load CODI ----
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
    targs = TrainingArguments(output_dir="/tmp/_lat", bf16=True,
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

    HOOK = {"phase": "off", "step": -99, "layer": -1,
            "vec": None, "alpha": 0.0, "active": False}
    # phase: 'prompt_end' (fires during prompt forward at last position),
    #        'latent' (fires during latent step matching HOOK['step']),
    # step: which latent step to intervene at (only for 'latent')
    # layer: 1-indexed layer (we hook on block_idx = layer - 1)

    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"]: return output
            if block_idx != HOOK["layer"] - 1: return output
            phase = HOOK["phase"]
            cur_phase = HOOK.get("cur_phase", "off")
            if phase == "latent":
                if cur_phase != "latent": return output
                if HOOK["step"] != HOOK.get("cur_step", -99): return output
            elif phase == "prompt_end":
                if cur_phase != "prompt_end": return output
            else:
                return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, vec, alpha, layer, phase, latent_step=-99, max_new=256):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)

        HOOK.update({"vec": vec, "alpha": alpha, "layer": layer,
                     "phase": phase, "step": latent_step,
                     "cur_phase": "prompt_end", "cur_step": -99,
                     "active": True})
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)

        # latent loop
        for s in range(targs.inf_latent_iterations):
            HOOK["cur_phase"] = "latent"
            HOOK["cur_step"] = s
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)

        # decode (no intervention)
        HOOK["cur_phase"] = "off"
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_new):
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
        HOOK["active"] = False
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    # SVAMP eval set
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    add_idx, sub_idx = [], []
    for i, ex in enumerate(full):
        t = ex["Type"].lower()
        if t == "addition":      add_idx.append(i)
        elif t == "subtraction": sub_idx.append(i)
    np.random.seed(0)
    add_eval = np.random.choice(add_idx, size=min(100, len(add_idx)), replace=False).tolist()
    sub_eval = np.random.choice(sub_idx, size=min(100, len(sub_idx)), replace=False).tolist()

    def candidates(idx_list):
        cs = []
        for i in idx_list:
            ex = full[int(i)]
            ab = parse_two_operands(ex["Equation"])
            if ab is None: cs.append(None); continue
            a, b = ab
            cs.append({"add": round(a+b), "sub": round(a-b),
                       "mul": round(a*b), "div": round(a/b) if b != 0 else None})
        return cs
    add_cs = candidates(add_eval); sub_cs = candidates(sub_eval)

    def gen_eval(idx_list, vec, alpha, layer, phase, latent_step=-99):
        qs = [full[int(i)]["question_concat"].strip().replace("  ", " ") for i in idx_list]
        out = []
        for s in range(0, len(qs), 16):
            out += run_batch(qs[s:s+16], vec=vec, alpha=alpha, layer=layer,
                              phase=phase, latent_step=latent_step, max_new=256)
        return out

    def score(strs, idx_list, cs):
        o = {"add_match": 0, "sub_match": 0, "mul_match": 0, "div_match": 0,
             "n_valid": 0}
        for s, c in zip(strs, cs):
            v = codi_extract(s)
            if v is None: continue
            o["n_valid"] += 1
            if c is None: continue
            if v == c["add"]: o["add_match"] += 1
            if v == c["sub"]: o["sub_match"] += 1
            if c["mul"] is not None and v == c["mul"]: o["mul_match"] += 1
            if c["div"] is not None and v == c["div"]: o["div_match"] += 1
        return o

    # Baseline
    print("\n=== Baseline (no steering) ===")
    base_add = gen_eval(add_eval, torch.zeros(H), 0.0, 0, "off")
    base_sub = gen_eval(sub_eval, torch.zeros(H), 0.0, 0, "off")
    print(f"  ADD baseline: {score(base_add, add_eval, add_cs)}")
    print(f"  SUB baseline: {score(base_sub, sub_eval, sub_cs)}")

    summary = {"add_baseline": score(base_add, add_eval, add_cs),
               "sub_baseline": score(base_sub, sub_eval, sub_cs)}

    for cell_name, (kind, step, layer) in CELLS.items():
        if kind == "latent":
            v = add_to_sub_lat[step, layer]
            phase = "latent"
        else:
            v = add_to_sub_pe[layer]
            phase = "prompt_end"
        v_unit = v / max(np.linalg.norm(v), 1e-9)
        v_t = torch.tensor(v_unit, dtype=torch.float32)
        for alpha in [50.0, 100.0, 200.0, 400.0]:
            print(f"\n=== Cell {cell_name} ({kind} step={step} L{layer})  alpha={alpha} ===")
            sa = gen_eval(add_eval, v_t, alpha, layer, phase, step)
            ss = gen_eval(sub_eval, v_t, -alpha, layer, phase, step)
            sc_a = score(sa, add_eval, add_cs)
            sc_s = score(ss, sub_eval, sub_cs)
            print(f"  ADD (add->sub steered): {sc_a}")
            print(f"  SUB (sub->add steered): {sc_s}")
            summary[f"{cell_name}|alpha={alpha}|add"] = sc_a
            summary[f"{cell_name}|alpha={alpha}|sub"] = sc_s

    out = PD / "steer_latent_loop_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
