"""Per-head + per-MLP residual contribution capture for the latent loop.

For each example, at each (latent step, layer, head):
  - residual contribution of head h's attention output at the last token,
    decomposed as: c_proj[:, h*hd:(h+1)*hd] @ attn_head_h_output_last_token.
For each (step, layer):
  - MLP block output at the last token.

Aggregates online to keep memory small:
  - mean per-head residual contribution: shape (S, L, n_heads, H)
  - mean MLP output:                     shape (S, L, H)
  - mean per-head attn-output norm:      shape (S, L, n_heads)
  - mean per-head attn weight from last token to L1..L6, Q, BOT: shape (S, L, n_heads, n_classes)
  - per-operator-class breakdowns of the per-head residual contribution:
    shape (4, S, L, n_heads, H)

Saves as flow_head_content.npz.
"""
from __future__ import annotations

import json
import os
import sys
import time
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

OUT_NPZ = PD / "flow_head_content.npz"
OPS = ["Addition", "Subtraction", "Multiplication", "Common-Division"]
N_CLASS = 8  # Q, BOT, L1..L6


def main():
    BS = 16

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
    targs = TrainingArguments(output_dir="/tmp/_fc", bf16=True,
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
    cfg = model.codi.config
    N_LAYERS = cfg.n_layer
    N_HEADS = cfg.n_head
    HID = cfg.n_embd
    HEAD_DIM = HID // N_HEADS
    N_LAT = 6
    print(f"  GPT-2 layers={N_LAYERS}, heads={N_HEADS}, hidden={HID}, head_dim={HEAD_DIM}")

    # Extract c_proj weights per layer to decompose per-head residual contributions.
    # In GPT-2 HF, attention's c_proj has weight shape (HID, HID) for nn.Linear,
    # mapping the concatenated attn output (B, T, HID) → (B, T, HID). But GPT-2
    # uses Conv1D where the weight is (HID, HID) and applied as x @ W (so weight
    # rows correspond to input dims, cols to output dims). Actually for Conv1D,
    # weight shape is (nf, nx) where nx is input dim. Let's check at runtime.
    c_proj_weights = []
    c_proj_biases = []
    # When LoRA wraps the layer, we need the merged effective weight. Use the
    # base layer if possible; otherwise the LoRA-wrapped one.
    for i, blk in enumerate(transformer.h):
        attn = blk.attn
        # blk.attn.c_proj may be a Conv1D with weight (nx=HID, nf=HID) — applied as x @ weight.
        # In transformers.pytorch_utils.Conv1D, the weight is (nx, nf), bias (nf,).
        # For LoRA-wrapped layers we may have base_layer.
        try:
            base = attn.c_proj.base_layer
        except AttributeError:
            base = attn.c_proj
        w = base.weight  # for Conv1D: (HID, HID); for Linear: (HID, HID)
        b = base.bias
        c_proj_weights.append(w.detach().to("cuda"))
        c_proj_biases.append(b.detach().to("cuda"))

    # Online aggregators.
    mean_head_resid = np.zeros((N_LAT, N_LAYERS, N_HEADS, HID), dtype=np.float64)
    mean_mlp_out = np.zeros((N_LAT, N_LAYERS, HID), dtype=np.float64)
    mean_head_norm = np.zeros((N_LAT, N_LAYERS, N_HEADS), dtype=np.float64)
    mean_head_attn = np.zeros((N_LAT, N_LAYERS, N_HEADS, N_CLASS), dtype=np.float64)
    cnt = 0
    # Per-op breakdowns (mean residual contribution by operator type)
    op_cnt = np.zeros(4, dtype=np.int64)
    op_mean_head_resid = np.zeros((4, N_LAT, N_LAYERS, N_HEADS, HID), dtype=np.float64)
    op_mean_mlp_out = np.zeros((4, N_LAT, N_LAYERS, HID), dtype=np.float64)

    # Hook state.
    CAP = {"active": False, "step": -1}
    # Per-layer-per-step accumulators (reset each step):
    # attn_per_head_last[layer]: (B, n_heads, head_dim) for last token
    # mlp_out_last[layer]: (B, HID)
    # attn_weight_last[layer]: (B, n_heads, k_len) attention weights from last token to all keys
    attn_per_head_buf = [None] * N_LAYERS
    mlp_out_buf = [None] * N_LAYERS
    attn_weight_buf = [None] * N_LAYERS

    def make_attn_hook(idx):
        def fn(_module, _inputs, output):
            if not CAP["active"]: return output
            # output of GPT2Attention: (attn_output, present, attn_weights)
            # attn_output shape: (B, T, HID) (already after c_proj)
            # We need PRE-c_proj per-head outputs. Let's intercept differently —
            # but the attention block forward returns post-c_proj output. To get
            # per-head we register an inner hook on the c_proj module.
            # For attn weights, output[2] if output_attentions=True: (B, n_heads, q_len, k_len)
            if len(output) > 2 and output[2] is not None:
                w = output[2]
                attn_weight_buf[idx] = w[:, :, -1, :].float().detach()
            return output
        return fn

    def make_c_proj_pre_hook(idx):
        # Pre-hook on c_proj captures the concatenated per-head output (input to c_proj).
        def fn(_module, inputs):
            if not CAP["active"]: return None
            x = inputs[0]  # (B, T, HID)
            # take last position only
            last = x[:, -1, :]  # (B, HID)
            # reshape into (B, n_heads, head_dim)
            attn_per_head_buf[idx] = last.view(-1, N_HEADS, HEAD_DIM).float().detach()
            return None
        return fn

    def make_mlp_hook(idx):
        def fn(_module, _inputs, output):
            if not CAP["active"]: return output
            mlp_out_buf[idx] = output[:, -1, :].float().detach()
            return output
        return fn

    handles = []
    for i, blk in enumerate(transformer.h):
        handles.append(blk.attn.register_forward_hook(make_attn_hook(i)))
        # GPT-2's c_proj is at blk.attn.c_proj; LoRA wrapper might wrap.
        try:
            c_proj = blk.attn.c_proj
            handles.append(c_proj.register_forward_pre_hook(make_c_proj_pre_hook(i)))
        except Exception as e:
            print(f"WARN: couldn't hook c_proj on layer {i}: {e}")
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))

    # Load SVAMP
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array([t.replace("Common-Divison", "Common-Division") for t in full["Type"]])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]
    N = len(questions)
    op_to_idx = {op: i for i, op in enumerate(OPS)}
    y_op = np.array([op_to_idx.get(t, -1) for t in types])

    @torch.no_grad()
    def run_batch(start):
        qs = questions[start:start + BS]
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        Lq_pad = batch["input_ids"].shape[1]
        q_lens = batch["attention_mask"].sum(dim=-1).cpu().numpy()
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        CAP["active"] = False
        out = model.codi(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if targs.use_prj: latent = model.prj(latent)
        CAP["active"] = True
        for step in range(N_LAT):
            CAP["step"] = step
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            out = model.codi(inputs_embeds=latent, attention_mask=attn,
                             use_cache=True, output_hidden_states=True,
                             output_attentions=True,
                             past_key_values=past)
            past = out.past_key_values
            aggregate_step(step, B, q_lens, Lq_pad, start)
            latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if targs.use_prj: latent = model.prj(latent)
        CAP["active"] = False

    def aggregate_step(step, B, q_lens, Lq_pad, start):
        # For each layer, decompose per-head residual contribution.
        # attn_per_head_buf[layer]: (B, n_heads, head_dim) in fp32, on GPU.
        # c_proj weight: (HID_input, HID_output) where HID_input = HID; applied as
        # x @ weight + bias. Per-head residual contribution h:
        #     ph_input = attn_per_head_buf[layer][:, h, :]                 # (B, head_dim)
        #     # Reconstruct full HID input with only head h's slice:
        #     ph_full = zeros(B, HID); ph_full[:, h*hd:(h+1)*hd] = ph_input
        #     contrib_h = ph_full @ c_proj_weight    # (B, HID)
        # Equivalently:
        #     contrib_h = ph_input @ c_proj_weight[h*hd:(h+1)*hd, :]
        # The full per-layer attn output = sum_h contrib_h + bias.
        ops_b = y_op[start:start + B]
        for l in range(N_LAYERS):
            ph = attn_per_head_buf[l]  # (B, n_heads, head_dim)
            if ph is None: continue
            w = c_proj_weights[l]  # (HID, HID)
            # Compute per-head contribution: (B, n_heads, HID)
            # ph[:, h, :] @ w[h*hd:(h+1)*hd, :]
            # Stack as bmm: ph (B, n_heads, head_dim) @ w_blocks (n_heads, head_dim, HID)
            w_blocks = w.view(N_HEADS, HEAD_DIM, HID).float()  # (n_heads, head_dim, HID) fp32
            # einsum: ph[b,h,d] * w_blocks[h,d,o] -> contrib[b,h,o]
            contrib = torch.einsum("bhd,hdo->bho", ph.float(), w_blocks)  # (B, n_heads, HID)
            # contribute to aggregators
            mean_head_resid[step, l] += contrib.sum(dim=0).cpu().numpy()
            # per-head norm
            norms = contrib.norm(dim=-1)  # (B, n_heads)
            mean_head_norm[step, l] += norms.sum(dim=0).cpu().numpy()
            # MLP
            if mlp_out_buf[l] is not None:
                mean_mlp_out[step, l] += mlp_out_buf[l].sum(dim=0).cpu().numpy()
            # Attention to position classes — w_buf[l]: (B, n_heads, k_len)
            w_attn = attn_weight_buf[l]
            if w_attn is not None:
                k_len = w_attn.shape[-1]
                # Q region per example, BOT at Lq_pad, L1..L_step at Lq_pad+1..Lq_pad+step
                for b in range(B):
                    pad_amt = Lq_pad - int(q_lens[b])
                    if Lq_pad > pad_amt and Lq_pad <= k_len:
                        q_mass = w_attn[b, :, pad_amt:Lq_pad].sum(dim=-1)  # (n_heads,)
                        mean_head_attn[step, l, :, 0] += q_mass.cpu().numpy()
                if Lq_pad < k_len:
                    mean_head_attn[step, l, :, 1] += w_attn[:, :, Lq_pad].sum(dim=0).cpu().numpy()
                for li in range(min(step + 1, 6)):
                    p = Lq_pad + 1 + li
                    if p < k_len:
                        mean_head_attn[step, l, :, 2 + li] += w_attn[:, :, p].sum(dim=0).cpu().numpy()

            # Per-op
            for op in range(4):
                m = ops_b == op
                if not m.any(): continue
                op_mean_head_resid[op, step, l] += contrib[m].sum(dim=0).cpu().numpy()
                if mlp_out_buf[l] is not None:
                    op_mean_mlp_out[op, step, l] += mlp_out_buf[l][m].sum(dim=0).cpu().numpy()

        # update op counts at step 0 only (per-batch)
        if step == 0:
            for op in range(4):
                op_cnt[op] += int((ops_b == op).sum())

    t0 = time.time()
    for s in range(0, N, BS):
        run_batch(s)
        done = s + BS
        if done % 64 == 0 or done >= N:
            print(f"  {min(done, N)}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    # Normalize means
    mean_head_resid = mean_head_resid / N
    mean_mlp_out = mean_mlp_out / N
    mean_head_norm = mean_head_norm / N
    mean_head_attn = mean_head_attn / N
    for op in range(4):
        n_op = max(op_cnt[op], 1)
        op_mean_head_resid[op] /= n_op
        op_mean_mlp_out[op] /= n_op

    np.savez(OUT_NPZ,
             mean_head_resid=mean_head_resid.astype(np.float32),
             mean_mlp_out=mean_mlp_out.astype(np.float32),
             mean_head_norm=mean_head_norm.astype(np.float32),
             mean_head_attn=mean_head_attn.astype(np.float32),
             op_mean_head_resid=op_mean_head_resid.astype(np.float32),
             op_mean_mlp_out=op_mean_mlp_out.astype(np.float32),
             op_cnt=op_cnt)
    print(f"saved {OUT_NPZ}  shapes: head_resid {mean_head_resid.shape}, mlp {mean_mlp_out.shape}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
