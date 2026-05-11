"""Test multiple answer-extraction strategies on CODI-GPT-2's SVAMP outputs
to find out if the model is getting answers and we just don't see them."""

from __future__ import annotations
import os, re, sys, time
from pathlib import Path

import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "codi"))


def parsers(s):
    """Return dict of parsed answers under different strategies."""
    out = {}
    # 1. first integer in string
    m = re.search(r"-?\d+", s)
    out["first_int"] = int(m.group(0)) if m else None
    # 2. last integer in string
    ms = re.findall(r"-?\d+", s)
    out["last_int"] = int(ms[-1]) if ms else None
    # 3. first integer after "answer is:"
    m = re.search(r"answer is\s*:\s*(-?\d+)", s)
    out["after_answer_is"] = int(m.group(1)) if m else None
    # 4. integer after the LAST "=" sign
    if "=" in s:
        tail = s.rsplit("=", 1)[1]
        m = re.search(r"-?\d+", tail)
        out["after_last_equals"] = int(m.group(0)) if m else None
    else: out["after_last_equals"] = None
    # 5. eval the expression (if any "+","-","*","/")
    expr_m = re.search(r"(?:answer is\s*:\s*)?([\d\.\+\-\*/\(\) ]+)", s)
    if expr_m:
        e = expr_m.group(1).strip().rstrip("=")
        try:
            v = eval(e, {"__builtins__": {}}, {})
            if isinstance(v, (int, float)) and abs(v) < 1e9:
                out["eval_expr"] = round(v)
            else: out["eval_expr"] = None
        except Exception: out["eval_expr"] = None
    else: out["eval_expr"] = None
    # 6. integer with float-like value matching gold (parse all floats too)
    out["floats"] = [float(x) for x in re.findall(r"-?\d+\.?\d*", s) if x]
    return out


def main():
    ckpt = os.path.expanduser("~/codi_ckpt/CODI-gpt2")
    print(f"loading CODI-GPT-2 from {ckpt}", flush=True)
    _orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "use_fast": True})
    )
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                          r=128, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, init_lora_weights=True)
    margs = ModelArguments(model_name_or_path="gpt2", full_precision=True,
                           train=False, lora_init=True, ckpt_dir=ckpt)
    targs = TrainingArguments(output_dir="/tmp/_par", bf16=True,
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

    @torch.no_grad()
    def run_batch(qs, max_new=64):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        for _ in range(targs.inf_latent_iterations):
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             past_key_values=past)
            past = out.past_key_values
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device="cuda"))
        output = eot_emb.unsqueeze(0).expand(B, -1, -1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
        tokens = [[] for _ in range(B)]
        eos_id = tok.eos_token_id
        for _ in range(max_new):
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B): tokens[b].append(int(next_ids[b].item()))
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    answers = [int(round(float(str(ex["Answer"]).replace(",", "")))) for ex in full]
    N_eval = 200
    import numpy as np
    eval_idx = np.random.RandomState(0).choice(len(questions), size=N_eval, replace=False)
    eval_qs = [questions[i] for i in eval_idx]
    eval_a  = [answers[i] for i in eval_idx]

    BS = 16
    preds = []
    t0 = time.time()
    for s in range(0, N_eval, BS):
        preds += run_batch(eval_qs[s:s+BS], max_new=64)
        if (s + BS) % 64 == 0 or s + BS >= N_eval:
            print(f"  {min(s+BS, N_eval)}/{N_eval}  ({time.time()-t0:.0f}s)", flush=True)

    # Score under each parser
    PARSERS = ["first_int", "last_int", "after_answer_is", "after_last_equals", "eval_expr", "any_match"]
    correct = {p: 0 for p in PARSERS}
    near_correct_any = 0
    for q, gold, pred in zip(eval_qs, eval_a, preds):
        ps = parsers(pred)
        if ps["first_int"] == gold: correct["first_int"] += 1
        if ps["last_int"] == gold: correct["last_int"] += 1
        if ps["after_answer_is"] == gold: correct["after_answer_is"] += 1
        if ps["after_last_equals"] == gold: correct["after_last_equals"] += 1
        if ps["eval_expr"] == gold: correct["eval_expr"] += 1
        # any_match: any of the floats matches gold (within 1e-3)
        if any(abs(f - gold) < 1e-3 for f in ps["floats"]):
            correct["any_match"] += 1

    print(f"\n=== Accuracy on {N_eval} SVAMP examples under different parsers ===")
    for p in PARSERS:
        print(f"  {p:>20s}: {correct[p]}/{N_eval}  ({100*correct[p]/N_eval:.1f}%)")

    # Show 12 example outputs with gold
    print(f"\n=== 12 example (gold, prediction) ===")
    for i in range(12):
        ps = parsers(preds[i])
        match_set = [k for k, v in {"first_int": ps["first_int"],
                                       "last_int": ps["last_int"],
                                       "eval_expr": ps["eval_expr"]}.items() if v == eval_a[i]]
        any_in_floats = any(abs(f - eval_a[i]) < 1e-3 for f in ps["floats"])
        print(f"  gold={eval_a[i]:>5d}  pred={preds[i][:80]!r}  match_via={match_set} float_present={any_in_floats}")


if __name__ == "__main__":
    main()
