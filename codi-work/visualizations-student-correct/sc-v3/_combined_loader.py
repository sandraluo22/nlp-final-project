"""Helper: load the combined sc-v3 CF dataset.

The CF dataset = (original SVAMP Mul + Div) + (GPT-transformed Add + Sub).
Both pieces are filtered to student-correct on their respective student runs.

Returns:
  acts: (N, 6, 17, 2048) np.float32   — concatenated activations
  meta: dict with keys:
    - 'type'     : surface operator label per row
                   (Multiplication / Common-Division / Addition / Subtraction)
    - 'src_type' : underlying source operator
                   (= 'type' for unchanged Mul/Div, original Mul/Div for transformed Add/Sub)
    - 'origin'   : 'svamp_original'  or  'gpt_transformed'
"""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset


REPO = Path(__file__).resolve().parent.parent.parent


def load_combined_cf():
    # --- Original SVAMP Mul + Div ---
    svamp_acts = torch.load(
        REPO / "inference" / "runs" / "svamp_student" / "activations.pt",
        map_location="cpu", weights_only=True,
    ).float().numpy()
    svamp_results = json.load(
        open(REPO / "inference" / "runs" / "svamp_student" / "results.json")
    )
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    types = np.array(
        [t.replace("Common-Divison", "Common-Division") for t in full["Type"]]
    )
    correct = np.array([r["correct"] for r in svamp_results], dtype=bool)
    is_md = np.isin(types, ["Multiplication", "Common-Division"])
    keep = correct & is_md
    md_acts = svamp_acts[keep]
    md_types = types[keep]
    print(f"  svamp Mul+Div, student-correct: {md_acts.shape[0]}", flush=True)

    # --- GPT-transformed Add + Sub ---
    gpt_acts = torch.load(
        REPO / "inference" / "runs" / "cf_gpt_student" / "activations.pt",
        map_location="cpu", weights_only=True,
    ).float().numpy()
    gpt_results = json.load(
        open(REPO / "inference" / "runs" / "cf_gpt_student" / "results.json")
    )
    gpt_meta = json.load(open(REPO.parent / "cf-datasets" / "cf_gpt_transformed.json"))
    correct_g = np.array([r["correct"] for r in gpt_results], dtype=bool)
    gpt_acts_kept = gpt_acts[correct_g]
    gpt_types = np.array([r["type"] for r in gpt_meta])[correct_g]
    gpt_src = np.array([r["src_type"] for r in gpt_meta])[correct_g]
    print(f"  gpt Add+Sub,  student-correct: {gpt_acts_kept.shape[0]}", flush=True)

    # --- Concatenate ---
    acts = np.concatenate([md_acts, gpt_acts_kept], axis=0)
    type_arr = np.concatenate([md_types, gpt_types])
    src_arr = np.concatenate([md_types, gpt_src])  # for original Mul/Div, src == type
    origin = np.array(
        ["svamp_original"] * len(md_types) + ["gpt_transformed"] * len(gpt_types)
    )
    print(f"  combined: N={acts.shape[0]}, "
          f"type counts={dict(zip(*np.unique(type_arr, return_counts=True)))}",
          flush=True)
    return acts, {
        "type": type_arr,
        "src_type": src_arr,
        "origin": origin,
    }
