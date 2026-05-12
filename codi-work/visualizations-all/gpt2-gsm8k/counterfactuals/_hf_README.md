# CODI-GPT-2 activations at the `:` token (SVAMP + counterfactuals)

This dataset contains the residual stream of **CODI-GPT-2** captured **at
the `:` token position during answer emission**, for SVAMP and a family of
counterfactual SVAMP variants.

## What is the `:` position?

CODI-GPT-2 emits answers in the template `The answer is: <number>`. After the
latent reasoning loop (6 latent steps) ends with the EOT marker, the model
autoregressively decodes:

| decode step | input (= previously emitted) | residual = "at this token" | next emit |
|---:|---|---|---|
| 0 | `<EOT>` | EOT | ` The` |
| 1 | ` The` | The | ` answer` |
| 2 | ` answer` | answer | ` is` |
| 3 | ` is` | is | `:` |
| 4 | `:` | **`:` residual ← captured here** | ` <digit>` |

The `:` residual is the canonical *right-before-the-answer* mechanistic
intervention site: the model has just been fed `:` and the next token it will
emit is the answer digit(s).

## Files

For each dataset, two files:
- `{dataset}_colon_acts.pt` — torch tensor, shape `(N, 13, 768)`, bf16.
  Axis 1 = layer (0 = input embedding, 1..12 = transformer block outputs).
- `{dataset}_colon_acts_meta.json` — preds, golds, types, validity mask.

Datasets included:
- `svamp` — full SVAMP (train+test), 1000 problems.
- `cf_balanced`, `cf_magmatched`, `cf_under99`, `cf_under99_b` — bucketed counterfactuals.
- `vary_a`, `vary_a_2digit`, `vary_b`, `vary_b_2digit`, `vary_both_2digit`,
  `vary_numerals`, `vary_operator` — single-variable counterfactual sweeps.
- `numeral_pairs_a1_mul`, `numeral_pairs_b1_sub` — clean/corrupted pairs.

## Model

CODI checkpoint: `~/codi_ckpt/CODI-gpt2` (Shen et al. 2025), GPT-2-small base,
LoRA r=128 fine-tuned, prj_dim=768, K=6 latent iterations.

## How to load

```python
import torch, json
acts = torch.load("gsm8k_colon_acts.pt", map_location="cpu")
meta = json.load(open("gsm8k_colon_acts_meta.json"))
print(acts.shape)  # (1000, 13, 768)
print(meta["types"][:3], meta["gold"][:3])
```

## Companion repo

Scripts and figures: <https://github.com/sandraluo22/nlp-final-project>
