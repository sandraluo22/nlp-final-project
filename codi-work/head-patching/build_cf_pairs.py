"""Convert cf-datasets/{cf_balanced.json, vary_numerals.json} into the
'pairs.json' format that patching_gpt2.py expects:
  [{src_idx, clean: {text, answer}, corrupted: {text, answer}}, ...]

For cf_balanced: pair (orig SVAMP problem) with (CF problem of same src_idx).
For vary_numerals: pair the original SVAMP problem (looked up by src_idx) with the
varied version.
"""

from __future__ import annotations
import json, os
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

CF_DIR = Path("/Users/sandraluo/nlp-final-project/cf-datasets")
HEAD_DIR = Path("/Users/sandraluo/nlp-final-project/codi-work/head-patching")


def build_svamp_lookup():
    ds = load_dataset("ChilleD/SVAMP")
    full = concatenate_datasets([ds["train"], ds["test"]])
    out = {}
    for i, ex in enumerate(full):
        out[i] = {
            "text": ex["question_concat"].strip().replace("  ", " "),
            "answer": int(round(float(str(ex["Answer"]).replace(",", ""))))
        }
    return out


def main():
    svamp = build_svamp_lookup()
    print(f"loaded {len(svamp)} SVAMP examples")

    # === cf_balanced -> pairs ===
    cf = json.load(open(CF_DIR / "cf_balanced.json"))
    pairs_cf = []
    skipped = 0
    for row in cf:
        src = row.get("src_idx")
        if src is None or src not in svamp:
            skipped += 1; continue
        cf_text = row.get("cf_question_concat") or row.get("cf_body","") + " " + row.get("cf_question","")
        cf_text = cf_text.strip()
        if not cf_text:
            skipped += 1; continue
        try: cf_ans = int(round(float(row["cf_answer"])))
        except: skipped += 1; continue
        pairs_cf.append({
            "src_idx": int(src),
            "clean": svamp[src],
            "corrupted": {"text": cf_text, "answer": cf_ans},
        })
    print(f"cf_balanced: built {len(pairs_cf)} pairs, skipped {skipped}")
    out_path = HEAD_DIR / "cf_balanced_pairs.json"
    json.dump(pairs_cf, open(out_path, "w"), indent=2)
    print(f"  saved {out_path}")

    # === vary_numerals -> pairs ===
    vn = json.load(open(CF_DIR / "vary_numerals.json"))
    pairs_vn = []
    skipped = 0
    print(f"vary_numerals first row keys: {list(vn[0].keys())}")
    for row in vn:
        src = row.get("idx") or row.get("src_idx")
        if src is None or src not in svamp:
            skipped += 1; continue
        vn_text = row.get("question_concat", "").strip()
        if not vn_text: skipped += 1; continue
        try: vn_ans = int(round(float(row["answer"])))
        except: skipped += 1; continue
        if vn_text == svamp[src]["text"]:
            skipped += 1; continue   # no change
        pairs_vn.append({
            "src_idx": int(src),
            "clean": svamp[src],
            "corrupted": {"text": vn_text, "answer": vn_ans},
        })
    print(f"vary_numerals: built {len(pairs_vn)} pairs, skipped {skipped}")
    out_path = HEAD_DIR / "vary_numerals_pairs.json"
    json.dump(pairs_vn, open(out_path, "w"), indent=2)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
