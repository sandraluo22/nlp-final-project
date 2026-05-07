# Paired teacher/student activations on SVAMP and LOGIC-701

Activations and per-question results from running `inference/run_eval_with_hooks.py`
on the [`nlp-final-project`](https://github.com/sandraluo22/nlp-final-project) repo.

- **Teacher**: `unsloth/Llama-3.2-1B-Instruct` (mirror of `meta-llama/Llama-3.2-1B-Instruct`),
  explicit chain-of-thought via chat-template prompting, greedy decoding.
- **Student**: `zen-E/CODI-llama3.2-1b-Instruct` (LoRA + projection on the same base),
  6 latent thought iterations, greedy decoding.
- **Hook position (teacher)**: residual stream at the last prompt token, every layer.
- **Hook position (student)**: residual stream at every latent step, every layer.

## Files

| Path | Shape | Dtype | Notes |
|---|---|---|---|
| `svamp_teacher/activations.pt`    | `(1000, 17, 2048)`    | bf16 | (N, num_layers + 1 emb, hidden) |
| `svamp_student/activations.pt`    | `(1000, 6, 17, 2048)` | bf16 | (N, latent_steps, num_layers + 1 emb, hidden) |
| `logic701_teacher/activations.pt` | `(701, 17, 2048)`     | bf16 | |
| `logic701_student/activations.pt` | `(701, 6, 17, 2048)`  | bf16 | |
| `*/results.json`                  | —                     | text | per-question {idx, question, gold, pred, correct, response} |

`num_layers + 1 = 17` includes the embedding output (index 0) plus 16 decoder block outputs (1..16).

## Confusion (teacher × student)

**SVAMP (N=1000)** — teacher 0.622, student 0.608, agreement 0.708
|                       | Student correct | Student incorrect |
|---|---:|---:|
| Teacher correct       | 469 | 153 |
| Teacher incorrect     | 139 | 239 |

**LOGIC-701 (N=701)** — teacher 0.233, student 0.031, agreement 0.745
|                       | Student correct | Student incorrect |
|---|---:|---:|
| Teacher correct       |   3 | 160 |
| Teacher incorrect     |  19 | 519 |

## Reproduction

```bash
git clone git@github.com:sandraluo22/nlp-final-project.git
cd nlp-final-project/inference
# H100 / single GPU; takes ~10 min total
bash run_sweep.sh
```

LOGIC-701 student accuracy is artificially low because ~96% of free-text responses
contain no parseable option number. A log-likelihood scorer over the five options
would give a more honest comparison.
