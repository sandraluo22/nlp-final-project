# Paired teacher/student activations on LOGIC-701 (3 latent steps)

Activations and per-question results from running
`inference/run_eval_with_hooks_3_latent_steps.py` on the
[`nlp-final-project`](https://github.com/sandraluo22/nlp-final-project) repo.

- **Teacher**: `unsloth/Llama-3.2-1B-Instruct` (mirror of `meta-llama/Llama-3.2-1B-Instruct`),
  explicit chain-of-thought via chat-template prompting, greedy decoding.
- **Student**: `zen-E/CODI-llama3.2-1b-Instruct` (LoRA + projection on the same base),
  **3 latent thought iterations**, greedy decoding.
- **Hook position (teacher)**: residual stream at the last prompt token, every layer.
- **Hook position (student)**: residual stream at every latent step, every layer.

## Files

| Path                                    | Shape                | Dtype | Notes                                                       |
| --------------------------------------- | -------------------- | ----- | ----------------------------------------------------------- |
| `logic701_teacher_3step/activations.pt` | `(701, 17, 2048)`    | bf16  | (N, num_layers + 1 emb, hidden)                             |
| `logic701_student_3step/activations.pt` | `(701, 3, 17, 2048)` | bf16  | (N, latent_steps, num_layers + 1 emb, hidden)               |
| `*/results.json`                        | --                   | text  | per-question {idx, question, gold, pred, correct, response} |

`num_layers + 1 = 17` includes the embedding output (index 0) plus 16 decoder block outputs (1..16).

## Confusion (teacher x student)

**LOGIC-701, 3 latent steps (N=701)** -- teacher 0.254, student 0.039, agreement 0.719

|                   | Student correct | Student incorrect |
| ----------------- | --------------: | ----------------: |
| Teacher correct   |               4 |               174 |
| Teacher incorrect |              23 |               500 |

## Reproduction

```bash
git clone git@github.com:sandraluo22/nlp-final-project.git
cd nlp-final-project/inference
# H100 / single GPU; takes ~10 min total
python -u run_eval_with_hooks_3_latent_steps.py \
    --mode teacher --dataset logic701 \
    --batch_size 32 --max_new_tokens 384 \
    --out_dir runs/logic701_teacher_3step
python -u run_eval_with_hooks_3_latent_steps.py \
    --mode student --dataset logic701 \
    --batch_size 32 --max_new_tokens 384 \
    --out_dir runs/logic701_student_3step
```

LOGIC-701 student accuracy is artificially low because most free-text responses
contain no parseable option number (the CODI student was trained on GSM8k-Aug
arithmetic, so it tends to emit numeric answers rather than option indices 1..5).
A log-likelihood scorer over the five options would give a more honest comparison.

Compared to the 6-latent-step run in the parent README (student 0.031, agreement 0.745),
3 steps gives slightly higher raw student accuracy (0.039) but lower teacher-student
agreement (0.719), with most of the joint mass on `(teacher incorrect, student incorrect)`.
