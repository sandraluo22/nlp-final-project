"""Logit lens on saved CODI activations.

For every (question, [step,] layer) snapshot in an activations.pt file,
project through the base model's LM head and record the top-k tokens.
Works on both teacher activations (N, 17, 2048) and student activations
(N, 6, 17, 2048).

Outputs (under --out_dir):
  logit_lens.pt           tensors of top-k ids and logits
  logit_lens_sample.json  human-readable view of the first --num_samples questions

Run on all four (model, dataset) combos:
  python logit_lens.py --activations runs/svamp_teacher/activations.pt \\
      --results runs/svamp_teacher/results.json --out_dir runs/svamp_teacher
  python logit_lens.py --activations runs/svamp_student/activations.pt \\
      --results runs/svamp_student/results.json --out_dir runs/svamp_student
  python logit_lens.py --activations runs/logic701_teacher/activations.pt \\
      --results runs/logic701_teacher/results.json --out_dir runs/logic701_teacher
  python logit_lens.py --activations runs/logic701_student/activations.pt \\
      --results runs/logic701_student/results.json --out_dir runs/logic701_student

Notes:
  - The base model's lm_head is used for both teacher and student. CODI's LoRA
    target_modules cover q_proj/k_proj/v_proj/o_proj/up_proj/down_proj/gate_proj
    only, so lm_head is unchanged across the two models. The same head decodes
    both. Verify by re-reading codi/src/model.py if the checkpoint ever changes.
  - Llama-3.2-1B has tie_word_embeddings=True; lm_head.weight and
    embed_tokens.weight share storage. Either works.
  - Activations are bf16 on disk; we cast to float32 before the matmul to keep
    softmax/topk numerically clean.
  - Logit-lens output is noisy at early layers (the head was only trained
    against the final layer). Expect garbage at layers 1..8 and sensible
    numerics at layers 12..16. The sanity print at the end shows the final
    layer for question 0 so you can eyeball it.
"""

import argparse
import json
from pathlib import Path

import torch


def load_lm_head(base_model: str, trust_remote_code: bool = False):
    """Return (lm_head_weight, tokenizer). CPU only; peak ~5GB during load,
    ~1GB after we drop the rest of the model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[logit_lens] loading {base_model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=trust_remote_code,
    )
    lm_head = model.lm_head.weight.detach().clone().float()  # (V, H)
    del model
    print(f"[logit_lens]   lm_head shape = {tuple(lm_head.shape)}", flush=True)
    return lm_head, tokenizer


def apply_logit_lens(acts: torch.Tensor, lm_head: torch.Tensor, topk: int):
    """
    acts:    (..., H) any float dtype
    lm_head: (V, H)   float32
    Returns top_ids (..., topk) long, top_logits (..., topk) float32.
    """
    H = acts.shape[-1]
    assert lm_head.shape[1] == H, (
        f"hidden dim mismatch: acts H={H} vs head H={lm_head.shape[1]}"
    )
    flat = acts.reshape(-1, H).float()                       # (M, H)
    logits = flat @ lm_head.T                                # (M, V)
    top_logits, top_ids = torch.topk(logits, topk, dim=-1)   # (M, topk)
    out_shape = acts.shape[:-1] + (topk,)
    return top_ids.reshape(out_shape), top_logits.reshape(out_shape)


def chunked_logit_lens(acts, lm_head, topk, batch_size):
    """Chunk along axis 0 to keep peak memory bounded.

    With Llama-3.2 vocab ~128k and batch_size=32 on student activations, the
    intermediate logits tensor is roughly 32 * 6 * 17 * 128k * 4 bytes = ~1.7 GB.
    """
    N = acts.shape[0]
    id_chunks, logit_chunks = [], []
    for i in range(0, N, batch_size):
        chunk = acts[i : i + batch_size]
        ids, logs = apply_logit_lens(chunk, lm_head, topk)
        id_chunks.append(ids)
        logit_chunks.append(logs)
        print(f"[logit_lens] decoded {min(i + batch_size, N)}/{N}", flush=True)
    return torch.cat(id_chunks, dim=0), torch.cat(logit_chunks, dim=0)


def decode_topk_for_question(top_ids_q, tokenizer):
    """
    top_ids_q: (S, L, topk) for student or (L, topk) for teacher.
    Returns dict mapping "step{s}_layer{l}" or "layer{l}" -> list of strings.
    """
    out = {}
    if top_ids_q.dim() == 3:                                 # student
        S, L, _ = top_ids_q.shape
        for s in range(S):
            for l in range(L):
                ids = top_ids_q[s, l].tolist()
                out[f"step{s}_layer{l}"] = [tokenizer.decode([i]) for i in ids]
    elif top_ids_q.dim() == 2:                               # teacher
        L, _ = top_ids_q.shape
        for l in range(L):
            ids = top_ids_q[l].tolist()
            out[f"layer{l}"] = [tokenizer.decode([i]) for i in ids]
    else:
        raise ValueError(f"unexpected top_ids shape per question: {top_ids_q.shape}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True,
                   help="Path to an activations.pt produced by run_eval_with_hooks.py")
    p.add_argument("--out_dir", required=True,
                   help="Where to write logit_lens.pt and logit_lens_sample.json")
    p.add_argument("--results", default=None,
                   help="Optional results.json from the same run; used to enrich the sample JSON")
    p.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="required for models with custom code (e.g. Huginn-0125)")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Questions per matmul; reduce if you run out of RAM")
    p.add_argument("--num_samples", type=int, default=5,
                   help="How many questions to dump into logit_lens_sample.json")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load activations
    print(f"[logit_lens] loading {args.activations}", flush=True)
    acts = torch.load(args.activations, map_location="cpu")
    print(f"[logit_lens]   shape = {tuple(acts.shape)}, dtype = {acts.dtype}", flush=True)
    if acts.dim() not in (3, 4):
        raise ValueError(f"expected 3D (teacher) or 4D (student) activations, got {acts.shape}")

    # 2. Load LM head + tokenizer
    lm_head, tokenizer = load_lm_head(args.base_model, trust_remote_code=args.trust_remote_code)

    # 3. Project + take top-k
    print(f"[logit_lens] computing top-{args.topk} for {acts.shape[0]} questions", flush=True)
    top_ids, top_logits = chunked_logit_lens(acts, lm_head, args.topk, args.batch_size)
    print(f"[logit_lens]   top_ids shape = {tuple(top_ids.shape)}", flush=True)

    # 4. Sanity print: question 0, final layer.
    is_student = top_ids.dim() == 4
    print("\n[logit_lens] sanity (question 0, final layer top-k):", flush=True)
    if is_student:
        for s in range(top_ids.shape[1]):
            ids = top_ids[0, s, -1].tolist()
            print(f"  step {s}: {[tokenizer.decode([i]) for i in ids]}", flush=True)
    else:
        ids = top_ids[0, -1].tolist()
        print(f"  {[tokenizer.decode([i]) for i in ids]}", flush=True)

    # 5. Save tensors.
    save_path = out_dir / "logit_lens.pt"
    torch.save(
        {
            "top_ids": top_ids,                  # long, exact token IDs
            "top_logits": top_logits,            # float32, for confidence/probs
            "topk": args.topk,
            "base_model": args.base_model,
            "activations_path": str(args.activations),
            "activations_shape": list(acts.shape),
        },
        save_path,
    )
    print(f"\n[logit_lens] saved tensors -> {save_path}", flush=True)

    # 6. Human-readable sample.
    results = None
    if args.results is not None:
        with open(args.results) as f:
            results = json.load(f)

    n_show = min(args.num_samples, top_ids.shape[0])
    sample = []
    for q in range(n_show):
        entry = {"idx": q}
        if results is not None:
            r = results[q]
            entry.update({
                "question": r.get("question"),
                "gold": r.get("gold"),
                "pred": r.get("pred"),
                "correct": r.get("correct"),
                "response": r.get("response"),
            })
        entry["topk"] = decode_topk_for_question(top_ids[q], tokenizer)
        sample.append(entry)

    sample_path = out_dir / "logit_lens_sample.json"
    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"[logit_lens] wrote sample for {n_show} questions -> {sample_path}", flush=True)


if __name__ == "__main__":
    main()
