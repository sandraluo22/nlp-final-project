"""Upload CODI-GPT-2 latent-loop activations to the existing HF dataset.

Adds {dataset}_latent_acts.pt files (shape (N, 6 latent_steps, 13 layers, 768))
alongside the existing {dataset}_colon_acts.pt files in
sandrajyluo/codi-gpt2-svamp-activations.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, upload_file

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "inference" / "runs"
PD = Path(__file__).resolve().parent

LATENT_SOURCES = {
    # name in HF -> local path
    "svamp": RUNS / "svamp_student_gpt2" / "activations.pt",
    "cf_balanced": RUNS / "cf_balanced_student_gpt2" / "activations.pt",
    "vary_a": RUNS / "gpt2_vary_a" / "activations.pt",
    "vary_a_2digit": RUNS / "gpt2_vary_a_2digit" / "activations.pt",
    "vary_b": RUNS / "gpt2_vary_b" / "activations.pt",
    "vary_b_2digit": RUNS / "gpt2_vary_b_2digit" / "activations.pt",
    "vary_both_2digit": RUNS / "gpt2_vary_both_2digit" / "activations.pt",
    "vary_numerals": RUNS / "gpt2_vary_numerals" / "activations.pt",
    "vary_operator": RUNS / "gpt2_vary_operator" / "activations.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="sandrajyluo/codi-gpt2-svamp-activations")
    args = ap.parse_args()
    api = HfApi()
    tmp_dir = PD / "_upload_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for name, src in LATENT_SOURCES.items():
        if not src.exists():
            print(f"  SKIP {name}: {src} missing")
            continue
        target_filename = f"{name}_latent_acts.pt"
        local_tmp = tmp_dir / target_filename
        if not local_tmp.exists() or local_tmp.stat().st_size != src.stat().st_size:
            shutil.copy(src, local_tmp)
        size_mb = local_tmp.stat().st_size / 1e6
        print(f"  uploading {target_filename}  ({size_mb:.1f} MB)...", flush=True)
        upload_file(path_or_fileobj=str(local_tmp), path_in_repo=target_filename,
                    repo_id=args.repo, repo_type="dataset")
        print("     done")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\nhttps://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
