"""Two combined experiments on top of the multi-position decode capture:

EXP 1 - Cos-sim conditioned operator steering
  For each (pos, layer), compute cos_sim between the (sub - add) operator
  direction and the principal digit-probe direction (mean of units coef rows).
  Pick the LOW-COS-SIM cell (clean separation) and HIGH-COS-SIM cell (tangled)
  in the pre-emission region pos in {1..4}. Run add->sub steering at each,
  compare how many outputs change.

EXP 2 - Magnitude probe + steering
  Bucket each example's model-emitted integer by magnitude:
    0: 1-9, 1: 10-99, 2: 100-999, 3: 1000-9999, 4: 10000+
  Fit per-(pos, layer) magnitude-bucket probe. Pick best cell with pos<=4.
  Steer toward bucket 0 (small) and bucket 4 (large). Measure median |int|
  shift in the steered runs.
"""

from __future__ import annotations
import json, os, re, sys
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "experiments" / "computation_probes"
sys.path.insert(0, str(REPO / "codi"))


def parse_int(s):
    m = re.search(r"answer is\s*:\s*(-?\d+)", s)
    if not m: return None
    try: return int(m.group(1))
    except: return None


def fit_probe(X, y, cv_splits=0):
    """Fit on all data; return (sc, clf, [optional CV acc])."""
    mask = y >= 0
    X = X[mask]; y = y[mask]
    if len(np.unique(y)) < 2: return None, None, 0.0
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs").fit(Xs, y)
    if cv_splits >= 2:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
        accs = []
        for tr, te in skf.split(X, y):
            sc2 = StandardScaler().fit(X[tr])
            clf2 = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs").fit(
                sc2.transform(X[tr]), y[tr])
            accs.append(clf2.score(sc2.transform(X[te]), y[te]))
        return sc, clf, float(np.mean(accs))
    return sc, clf, 0.0


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("loading multi-pos decode activations + preds...", flush=True)
    acts = torch.load(PD / "svamp_multipos_decode_acts.pt", map_location="cpu").to(torch.float32).numpy()
    N, P, Lp1, H = acts.shape
    print(f"  acts shape {acts.shape}")
    preds = json.load(open(PD / "svamp_multipos_decode_preds.json"))

    # parse model answers
    model_ans_py = [parse_int(s) for s in preds["preds"]]
    valid_mask = np.array([v is not None for v in model_ans_py])
    abs_ans = np.array([abs(v) if v is not None else -1 for v in model_ans_py])
    # magnitude buckets
    def bucket(v):
        if v is None: return -1
        a = abs(v)
        if a < 10: return 0
        if a < 100: return 1
        if a < 1000: return 2
        if a < 10000: return 3
        return 4
    mag_label = np.array([bucket(v) for v in model_ans_py], dtype=np.int64)
    model_units = np.array([(abs(v) % 10) if v is not None else -1
                            for v in model_ans_py], dtype=np.int64)

    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    op_map = {"addition": 0, "subtraction": 1, "multiplication": 2,
              "common-division": 3, "common-divison": 3}
    operators = np.array([op_map.get(ex["Type"].lower(), -1) for ex in full])
    questions = [ex["question_concat"].strip().replace("  ", " ") for ex in full]

    # =========== EXP 1 PREP =================
    # For every (pos, layer): fit operator probe + units probe, compute
    # cos_sim between (W_sub - W_add) and the principal units direction.
    print("\n[EXP1] computing per-(pos,layer) cos_sim(op_dir, units_dir)...", flush=True)
    cossim = np.zeros((P, Lp1))
    op_acc_grid = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            sc_o, clf_o, _ = fit_probe(X, operators)
            sc_u, clf_u, _ = fit_probe(X, model_units)
            if clf_o is None or clf_u is None: continue
            # operator direction in raw space: sigma_o * (W[1] - W[0])  (sub - add)
            op_dir = sc_o.scale_ * (clf_o.coef_[1] - clf_o.coef_[0])
            # units principal direction: top-1 SVD direction of W_u
            U, S, Vt = np.linalg.svd(clf_u.coef_, full_matrices=False)
            digits_dir = sc_u.scale_ * Vt[0]
            cossim[p, l] = abs(cos(op_dir, digits_dir))
            # also record operator probe acc on training data (rough)
            op_acc_grid[p, l] = clf_o.score(sc_o.transform(X), operators[operators >= 0])
        print(f"  pos {p:2d}: cos_sim min={cossim[p].min():.3f} max={cossim[p].max():.3f}", flush=True)

    # Pick LOW and HIGH cos_sim cells in pre-emission (pos 1..4)
    pre_mask = np.zeros((P, Lp1), dtype=bool); pre_mask[1:5, :] = True
    cossim_pre = np.where(pre_mask, cossim, np.inf)
    low_pos, low_lay = np.unravel_index(int(np.argmin(cossim_pre)), cossim.shape)
    cossim_pre_h = np.where(pre_mask, cossim, -np.inf)
    high_pos, high_lay = np.unravel_index(int(np.argmax(cossim_pre_h)), cossim.shape)
    print(f"\n  LOW  cos cell (pos<=4): pos {low_pos} layer {low_lay}  cos={cossim[low_pos, low_lay]:.3f}")
    print(f"  HIGH cos cell (pos<=4): pos {high_pos} layer {high_lay} cos={cossim[high_pos, high_lay]:.3f}")

    # Fit operator probes at those two cells for steering
    sc_op_low,  clf_op_low,  _ = fit_probe(acts[:, low_pos, low_lay, :], operators)
    sc_op_high, clf_op_high, _ = fit_probe(acts[:, high_pos, high_lay, :], operators)

    def op_steer_vec(sc, clf, target, source):
        return torch.tensor(sc.scale_ * (clf.coef_[target] - clf.coef_[source]), dtype=torch.float32)

    # =========== EXP 2 PREP =================
    print("\n[EXP2] fitting magnitude-bucket probes...", flush=True)
    print(f"  bucket distribution: {np.bincount(mag_label[mag_label>=0])}")
    mag_acc = np.zeros((P, Lp1))
    for p in range(P):
        for l in range(Lp1):
            X = acts[:, p, l, :]
            _, _, cvacc = fit_probe(X, mag_label, cv_splits=5)
            mag_acc[p, l] = cvacc
        print(f"  pos {p:2d}: mag-bucket peak={mag_acc[p].max()*100:.1f}%", flush=True)
    # best pre-emission cell (pos<=4)
    M = np.where(pre_mask, mag_acc, 0.0)
    mag_pos, mag_lay = np.unravel_index(int(np.argmax(M)), M.shape)
    print(f"  magnitude cell (pos<=4): pos {mag_pos} layer {mag_lay} acc={mag_acc[mag_pos, mag_lay]*100:.1f}%")
    sc_mag, clf_mag, _ = fit_probe(acts[:, mag_pos, mag_lay, :], mag_label)

    def mag_steer_vec(target, source=None):
        if source is None:
            return torch.tensor(sc_mag.scale_ * clf_mag.coef_[target], dtype=torch.float32)
        return torch.tensor(sc_mag.scale_ * (clf_mag.coef_[target] - clf_mag.coef_[source]), dtype=torch.float32)

    # =========== Load model + hook =================
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
    targs = TrainingArguments(output_dir="/tmp/_csm", bf16=True,
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
    HOOK = {"step": -1, "active": False, "vec": None, "p_target": None,
            "layer": None, "alpha": 0.0}
    def make_hook(block_idx):
        def fn(module, inputs, output):
            if not HOOK["active"] or HOOK["layer"] is None: return output
            if block_idx != HOOK["layer"] - 1: return output
            if HOOK["step"] != HOOK["p_target"]: return output
            h = output[0] if isinstance(output, tuple) else output
            v = HOOK["vec"].to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + HOOK["alpha"] * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return fn
    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(transformer.h)]

    @torch.no_grad()
    def run_batch(qs, *, vec, alpha, layer, p_target):
        B = len(qs)
        batch = tok(qs, return_tensors="pt", padding="longest").to("cuda")
        bot = torch.full((B, 1), model.bot_id, dtype=torch.long, device="cuda")
        input_ids = torch.cat([batch["input_ids"], bot], dim=1)
        attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1)
        HOOK.update({"vec": vec, "p_target": p_target, "layer": layer,
                     "alpha": alpha, "active": True, "step": -1})
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
        for step in range(12):
            HOOK["step"] = step
            sout = model.codi(inputs_embeds=output, attention_mask=attn,
                              use_cache=True, output_hidden_states=False,
                              past_key_values=past)
            past = sout.past_key_values
            logits = sout.logits[:, -1, :model.codi.config.vocab_size - 1]
            next_ids = torch.argmax(logits, dim=-1)
            for b in range(B): tokens[b].append(int(next_ids[b].item()))
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], dim=1)
            output = embed_fn(next_ids).unsqueeze(1)
        HOOK.update({"active": False, "step": -1})
        return [tok.decode(t, skip_special_tokens=True) for t in tokens]

    N_eval = 200
    eval_idx = np.random.RandomState(0).choice(N, size=N_eval, replace=False)
    eval_qs = [questions[i] for i in eval_idx]
    BS = 16

    def run_full(vec, alpha, layer, p_target):
        out = []
        for s in range(0, N_eval, BS):
            out += run_batch(eval_qs[s:s+BS], vec=vec, alpha=alpha,
                              layer=layer, p_target=p_target)
        return out

    # baseline
    print("\n=== Baseline ===", flush=True)
    base_strs = run_full(torch.zeros(H), 0.0, 1, 0)
    base_ints = [parse_int(s) for s in base_strs]
    base_abs = [abs(v) for v in base_ints if v is not None]
    print(f"  baseline median |int|: {int(np.median(base_abs))}")
    print(f"  baseline distribution of magnitude buckets:")
    for b in range(5):
        n = sum(1 for v in base_ints if v is not None and bucket(v) == b)
        rng = ["1-9", "10-99", "100-999", "1000-9999", "10000+"][b]
        print(f"    {rng}: {n}/{N_eval}")

    summary = {"exp1_low_cell": [int(low_pos), int(low_lay), float(cossim[low_pos, low_lay])],
               "exp1_high_cell": [int(high_pos), int(high_lay), float(cossim[high_pos, high_lay])],
               "exp2_mag_cell": [int(mag_pos), int(mag_lay), float(mag_acc[mag_pos, mag_lay])],
               "baseline_median_abs": int(np.median(base_abs)),
               "baseline_bucket_counts": [sum(1 for v in base_ints if v is not None and bucket(v) == b) for b in range(5)],
               "cossim_grid": cossim.tolist(),
               "mag_acc_grid": mag_acc.tolist(),
               }

    # === EXP 1 === add->sub at low vs high cos cell ===
    for tag, p_target, layer, sc, clf in [
        ("LOW", low_pos, low_lay, sc_op_low, clf_op_low),
        ("HIGH", high_pos, high_lay, sc_op_high, clf_op_high),
    ]:
        v = op_steer_vec(sc, clf, 1, 0)
        for a in [4.0, 8.0]:
            print(f"\n=== EXP1 {tag} cos | add->sub at pos {p_target} L {layer}  alpha={a} ===")
            strs = run_full(v, a, layer, p_target)
            ints = [parse_int(s) for s in strs]
            n_changed = sum(1 for i, b in zip(ints, base_ints) if i != b)
            abs_vals = [abs(v) for v in ints if v is not None]
            med = int(np.median(abs_vals)) if abs_vals else 0
            buckets = [sum(1 for v in ints if v is not None and bucket(v) == b) for b in range(5)]
            print(f"  changed_full={n_changed}/{N_eval}  median |int|={med}  buckets={buckets}")
            summary[f"exp1_{tag}|alpha={a}"] = {"n_changed": n_changed, "median_abs": med,
                                                 "buckets": buckets, "preds": strs[:50]}

    # === EXP 2 === magnitude steering ===
    # bucket 0 = small, bucket 4 = large.
    for tgt in [0, 4]:
        v = mag_steer_vec(tgt)
        for a in [4.0, 8.0, 16.0]:
            print(f"\n=== EXP2 magnitude->bucket{tgt}  pos {mag_pos} L {mag_lay}  alpha={a} ===")
            strs = run_full(v, a, mag_lay, mag_pos)
            ints = [parse_int(s) for s in strs]
            abs_vals = [abs(v) for v in ints if v is not None]
            med = int(np.median(abs_vals)) if abs_vals else 0
            buckets = [sum(1 for v in ints if v is not None and bucket(v) == b) for b in range(5)]
            n_changed = sum(1 for i, b in zip(ints, base_ints) if i != b)
            print(f"  changed_full={n_changed}/{N_eval}  median |int|={med}  buckets={buckets}")
            summary[f"exp2_to_bucket{tgt}|alpha={a}"] = {"n_changed": n_changed,
                                                         "median_abs": med,
                                                         "buckets": buckets,
                                                         "preds": strs[:50]}

    out = PD / "steering_cossim_magnitude.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")
    for h in handles: h.remove()


if __name__ == "__main__":
    main()
