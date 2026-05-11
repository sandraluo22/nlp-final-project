# counterfactuals/ — CF dataset activations and runs

## Captured GPT-2 activations
For each CF dataset (cf_balanced, cf_magmatched, cf_under99, cf_under99_b, vary_a, vary_a_2digit,
vary_b, vary_b_2digit, vary_both_2digit, vary_numerals, vary_operator, numeral_pairs_a1_mul,
numeral_pairs_b1_sub), we have:
- `{dataset}_colon_acts.pt` — shape (N, 13, 768) bf16, residual at the ':' token (decode position 4).
- `{dataset}_colon_acts_meta.json` — captured predictions, golds, types, emit positions.
- `{dataset}_latent_acts.pt` — shape (N, 6, 13, 768) bf16, residuals at each latent step.

## Scripts
- `capture_colon_acts.py` — runs CODI-GPT-2 on a CF dataset and saves the ':' residual.
- `capture_latent_acts.py` — same but saves the 6-step latent residuals.
- `upload_colon_to_hf.py`, `upload_latent_to_hf.py` — upload to HuggingFace dataset.

## HuggingFace mirror
https://huggingface.co/datasets/sandrajyluo/codi-gpt2-svamp-activations

## Base SVAMP activations (NOT moved; still in original locations)
- Latent: `codi-work/inference/runs/svamp_student_gpt2/activations.pt`
- Colon:  `codi-work/experiments/computation_probes/svamp_colon_acts.pt`
