# When Do CODI Students Diverge From Their Teachers?

**Authors:** Karen Li, Sandra Luo, Hannah Tao
**Course:** NLP Final Project, Spring 2026
**Repository:** <https://github.com/sandraluo22/nlp-final-project>

---

> **About this document.** This is the working scaffold for the project report.
> Each section lists (i) what to say, (ii) the figures / numbers we already have
> in the repo to back it up, and (iii) `TODO` markers for prose we still need to
> write. Treat the `BACKING` blocks as the inventory of evidence and the bullet
> lists as the argument outline. Trim aggressively before submission.

---

## 0. Abstract *(≤ 250 words)*

**Pitch.** CODI compresses an explicit chain of thought into a handful of
continuous "latent thought" tokens and reports matching teacher accuracy on
GSM8k. Aggregate accuracy hides the more interesting question: on which
problems does the latent student diverge from its explicit-CoT teacher, and
what mechanism explains the gap?

**Contributions** *(write one sentence each):*
1. A paired teacher / student evaluation harness covering math (SVAMP, GSM-Hard,
   MathQA-numeric) and logic (LOGIC-701) at two scales (GPT-2 small, LLaMA-3.2-1B).
2. A latent-step sweep ( `N=0..6` ) showing that *most* of the student's
   accuracy is already present at `N=0`; extra latent iterations buy only
   marginal gains and at GPT-2 scale can hurt.
3. Probe evidence that the residual stream at the answer-emission position
   linearly encodes (a) the operator, (b) the output magnitude bucket and
   (c) eventual correctness — with the correctness signal localising to
   late-step / mid-late-layer cells.
4. A causal demonstration via steering and per-head ablation that those
   directions are not just decodable but mechanistically used: pushing along
   the correctness direction drops accuracy from 0.39 → 0.10, and an
   operator-direction push converts addition predictions to subtraction.
5. A cross-paradigm baseline using Huginn-3.5B (recurrent-depth) showing that
   the same SVAMP problems remain hard at much larger scale under a different
   latent-reasoning recipe.

`TODO`: tighten + add a one-line takeaway ("compression preserves the operator
and the magnitude direction; failures are mostly small numeric slips, not
structural misunderstandings of the problem").

---

## 1. Introduction

### 1.1 Motivation

- Chain-of-thought (CoT) is expensive: tokens are emitted serially and each one
  costs a full forward pass [Wei et al., 2022].
- A growing line of work compresses CoT into the model's *hidden state* —
  implicit CoT [Deng et al., 2023/2024], Coconut [Hao et al., 2024], CODI
  [Shen et al., 2025], CoT² [Gozeten et al., 2026].
- CODI's self-distillation recipe matches GSM8k-Aug accuracy at GPT-2 scale
  with **2.7–5.9× inference speedup**. But aggregate accuracy collapses two
  different questions into one — does the student *reason*, or does it
  *memorise the answers the teacher would have written*?
- The right object of study is the **off-diagonal of the teacher×student
  confusion table** — (T✓ S✗) and (T✗ S✓) cells. Those cells are where the
  compression visibly succeeds or fails.

### 1.2 Research questions

We characterise teacher / student divergence along three axes:

- **Where:** Which datasets and problem types produce systematic divergence?
- **What:** What geometric, sparsity, and mechanistic features distinguish
  divergent latent trajectories?
- **Why:** Does the latent bottleneck force *useful abstraction* (student
  better than teacher under surface paraphrase), or destroy detail the teacher
  relies on (student worse under structural / numeric variation)?

### 1.3 Contributions (in detail)

1. **Workstream 1 — Divergence map.** Paired teacher / student inference on
   five benchmarks at two scales; per-question outcomes and per-layer / per-step
   residual stream saved for downstream analysis. (`inference/`, `codi-work/inference/`)
2. **Workstream 2 — Counterfactuals.** Five SVAMP counterfactual datasets
   (`cf_balanced`, `cf_magmatched`, `cf_under99`, `cf_under99_b`,
   `cf_gpt_transformed`) plus two number-isolation probe sets
   (`vary_numerals`, `vary_operator`) that fix one factor and vary the other.
   (`cf-datasets/`)
3. **Workstream 3 — Trajectory characterisation.** PCA / LDA visualisations,
   linear probes for operator / magnitude / correctness, logit-lens hit rates,
   answer-direction projections, attention head content decomposition.
   (`codi-work/experiments/computation_probes/`,
   `codi-work/visualizations-all/`)
4. **Workstream 4 — Mechanism.** Activation patching and steering on operator
   directions (Sub→Add, Mul→Com); correctness steering; bare-math magnitude
   steering. (`codi-work/experiments/`, `codi-work/head-patching/`)
5. **Cross-paradigm baseline.** Huginn-3.5B recurrent-depth model at
   `K=1..32` on the same SVAMP problems plus the counterfactual variants.
   (`huginn-work/`)

`TODO`: write the contributions list as flowing prose, not a bullet dump.

---

## 2. Background and Related Work

### 2.1 Chain-of-thought and its compression

- **Explicit CoT.** Wei et al. (2022) — prompt the model with reasoning steps,
  accuracy up across math and logic benchmarks.
- **Implicit / curriculum-distilled CoT.** Deng et al. (2023, 2024) gradually
  remove intermediate tokens during training so the model learns to skip them
  at inference time.
- **Continuous CoT — Coconut.** Hao et al. (2024) feed the previous hidden
  state back in as a "continuous thought token", with a multi-stage curriculum.
- **CODI.** Shen et al. (2025) replace curriculum learning with **single-stage
  self-distillation**: the same backbone is run twice — once with explicit CoT
  (the teacher), once with `K` "thought" iterations through a small
  LoRA + projection adapter (the student) — and the student's pre-answer
  hidden state is trained to match the teacher's. The result matches
  GSM8k-Aug accuracy at GPT-2 small at 2.7–5.9× inference speedup.
- **CoT² and related.** Gozeten et al. (2026) form continuous tokens as
  convex combinations over the vocabulary embedding, allowing the model to
  pursue multiple reasoning paths in superposition.

### 2.2 Interpretability of latent reasoning

- Suleymanzade et al. (2025) — continuous CoT degrades on multi-path
  problems and at depth.
- Orgad et al. (2025) — *pre-answer* hidden states encode the eventual
  reasoning chain in explicit-CoT models; this motivates probing the residual
  stream at the answer-emission token.
- The original CODI paper probes individual latent thought tokens and finds
  partial decodability of intermediate computations. We extend this to the
  position **right before the answer is emitted** — the `:` token in
  "The answer is: <number>" — which we call the **colon residual** and find to
  be the highest-signal probe position.

### 2.3 Mechanistic analysis as a target

- Kantamneni & Tegmark (2025) show LLMs represent numbers as points on a
  *helix* and execute addition trigonometrically. We adopt this style of
  analysis — fit a structured probe per (layer, step) and ask whether the
  same mechanism survives compression into 6 latent tokens.

### 2.4 Alternative latent-reasoning paradigms

- **Huginn-3.5B (recurrent-depth).** Geiping et al. (2025) train a 3.5B
  model whose middle block is *iterated* `K` times at inference, exposing a
  knob analogous to CODI's latent step count. We use it as a control:
  do other latent-reasoning recipes hit the same failures on the same
  problems, or does CODI fail in idiosyncratic ways?

`TODO`: confirm Huginn citation and arXiv ID; we have the model loaded but
the reference is currently `huginn-work/inference/run_huginn_with_hooks.py`
header only.

---

## 3. Models, Datasets, and Setup

### 3.1 Models

| Role     | Checkpoint                                | Notes                                                                 |
| -------- | ----------------------------------------- | --------------------------------------------------------------------- |
| Teacher  | `unsloth/Llama-3.2-1B-Instruct`           | Base model + explicit CoT via chat-template prompting, greedy decode. |
| Student  | `zen-E/CODI-llama3.2-1b-Instruct`         | LoRA r=128 + projection on the same base; 6 latent iterations default. |
| Smaller  | `zen-E/CODI-gpt2`                         | 124M base; LoRA r=128, prj_dim=768, 6 latent iterations.              |
| Control  | `tomg-group-umd/huginn-0125` (3.5B)       | Recurrent-depth; `K` recurrences through the 4-block core.            |

> **CODI hook positions.** The student's residual stream is captured at
> (a) the last token of every latent step, every layer (used for trajectory
> analysis); and (b) the `:` token of `"The answer is: "` during answer
> emission (used for operator / magnitude / correctness probes). The teacher's
> residual stream is captured at the last prompt token, every layer.

### 3.2 Datasets

| Dataset             | Size  | Task                          | Notes                                                                 |
| ------------------- | ----- | ----------------------------- | --------------------------------------------------------------------- |
| SVAMP               | 1000  | Open-ended numeric            | In-distribution for CODI's training data (GSM8k-Aug).                 |
| GSM-Hard            | 1319  | Open-ended numeric            | OOD; larger numerals.                                                 |
| MathQA-numeric      | 2501  | Open-ended numeric            | OOD; multi-domain (geometry, physics, finance).                       |
| LOGIC-701 (MC)      | 701   | 5-way multiple choice         | Multi-step deductive reasoning.                                       |
| LOGIC-701 (numeric) | 266   | Open-ended numeric subset     | The numerically-answerable subset of LOGIC-701.                       |

**Counterfactual SVAMP** (all five derived from the original 1000 SVAMP
problems by re-numeralising and optionally rewriting; see `cf-datasets/README.md`):

| File                     | N    | Control                                                                                       |
| ------------------------ | ---: | --------------------------------------------------------------------------------------------- |
| `cf_magmatched.json`     |  972 | Output magnitude bucket matched across operators.                                             |
| `cf_balanced.json`       |  676 | Iteratively balanced: both input *and* output magnitude buckets controlled per operator.       |
| `cf_under99.json`        |  642 | Same as `cf_magmatched` but restricted to numerals ≤ 99.                                      |
| `cf_under99_b.json`      |  622 | Same as `cf_balanced` but restricted to numerals ≤ 99.                                        |
| `cf_gpt_transformed.json`|  546 | Re-numeralised *and* re-worded by GPT-5; preserves operator.                                  |

**Number-isolation probes** (used for PCA-based subspace alignment):

| File                     | N  | Held fixed                       | Varies                          |
| ------------------------ | -: | -------------------------------- | ------------------------------- |
| `vary_numerals.json`     | 80 | Scenario template, operator      | `(a, b)` with `5 ≤ b < a ≤ 200` |
| `vary_operator.json`     | 24 | `(a, b) = (12, 4)` across all rows | Operator (4 ops × 6 scenarios)  |

**Faithfulness labels.** `svamp_judged.json` — GPT-5 judges the Llama-3.2-1B
teacher's CoT trace on each SVAMP problem the teacher gets correct, labelling
it `faithful` (n=581) or `unfaithful` (n=41). Used as the supervision signal
for the faithfulness probe (Section 5.3).

### 3.3 Compute and reproducibility

- Single H100 (80GB). One full four-way `{teacher, student} × {SVAMP, LOGIC-701}`
  sweep takes ~10 minutes wall-clock.
- All activations saved in `bf16`. SVAMP student activations are
  `(1000, 6, 17, 2048)` ≈ 800 MB per dataset; LOGIC-701 is the same shape with
  N=701.
- Activations are gitignored; mirrored on HuggingFace as
  `sandrajyluo/nlp-final-project-activations` and (for the GPT-2 colon
  residuals) a dataset documented in `codi-work/experiments/computation_probes/_hf_README.md`.

`TODO`: confirm HF dataset names + add public URLs in the report draft.

---

## 4. Workstream 1 — The Divergence Map

### 4.1 Headline confusion tables

**SVAMP (N=1000)** — teacher 0.622, student 0.608, agreement 0.708

|                   | Student correct | Student incorrect |
| ----------------- | --------------: | ----------------: |
| Teacher correct   |             469 |               153 |
| Teacher incorrect |             139 |               239 |

**LOGIC-701 (N=701)** — teacher 0.233, student 0.031, agreement 0.745

|                   | Student correct | Student incorrect |
| ----------------- | --------------: | ----------------: |
| Teacher correct   |               3 |               160 |
| Teacher incorrect |              19 |               519 |

**Headline reading:**
- On SVAMP the student's *aggregate* accuracy is within 1.4 points of the
  teacher, but the off-diagonal — 153 (T✓ S✗) + 139 (T✗ S✓) = 292 problems,
  29% of the set — is the actual research object.
- On LOGIC-701 the student's parseable-answer rate is the bottleneck:
  ~96% of free-text responses contain no parseable option index. A
  log-likelihood scorer over the five options is the right comparison; the
  current numbers should be read as a *lower bound* on student capability.

`BACKING`: `inference/runs/{svamp,logic701}_{teacher,student}/results.json`,
`inference/confusion.py`, README §"Headline results so far".

### 4.2 Latent-step sweep

Accuracy as a function of forced number of latent iterations `N ∈ {0, …, 6}`:

| Dataset                     | GPT-2 CODI (best)   | LLaMA-1B CODI (best) |
| --------------------------- | ------------------- | -------------------- |
| SVAMP (N=1000)              | 0.388 @ N=2–3       | 0.612 @ N=5          |
| GSM-Hard (N=1319)           | 0.089 @ N=6         | 0.128 @ N=6          |
| MathQA-numeric (N=2501)     | 0.023 @ N=5         | 0.106 @ N=6          |
| LOGIC-701 MC (N=701)        | 0.113 @ N=0         | 0.050 @ N=0          |
| LOGIC-701 numeric (N=266)   | 0.086 @ N=3         | 0.128 @ N=0–1        |

**Reading:**

- **Most of the answer is already at `N=0`.** On SVAMP, LLaMA-1B CODI goes
  from 0.579 at `N=0` to 0.612 at `N=5` — the six latent iterations buy
  ~3.3 points, not 60.
- **GPT-2 plateaus or regresses past `N=3`.** Sweep is monotone for
  LLaMA-1B but non-monotone at GPT-2 scale, consistent with the smaller
  model having insufficient capacity to "use" extra latent steps coherently.
- **LOGIC-701 MC peaks at `N=0`** because the latent iterations make the
  student even less likely to emit a parseable option index — additional
  latents bias the output distribution toward numerals.

`BACKING`: `codi-work/latent-sweep/summary.txt`,
`codi-work/inference/analysis/latent_sweep/*.png`, `analysis/sweep_by_dataset.pdf`.

`TODO`: pick the canonical figure for the report (`sweep_by_dataset.pdf` is the
4-panel combined figure; the per-dataset `sweep_*.png` are clearer for slides).

### 4.3 Cross-paradigm sanity check (Huginn-3.5B)

We run Huginn-3.5B on the same SVAMP problems at `K ∈ {1, 2, 4, 8, 16, 32}`:

| K   | Correct | Accuracy |
| --: | ------: | -------: |
|   1 |       4 |    0.004 |
|   2 |      11 |    0.011 |
|   4 |      26 |    0.026 |
|   8 |      26 |    0.026 |
|  16 |     131 |    0.131 |
|  32 |     142 |    0.142 |

**Reading.** A 3.5B parameter recurrent-depth model with 32 iterations
hits 0.142 on the same problems where CODI-LLaMA-1B (~1B parameters,
6 iterations) hits 0.612. This is not a fair head-to-head — Huginn has not
been distilled on this task — but it puts an upper bound on how much of CODI's
score is *the distillation* vs *the latent compute*: it is mostly the
distillation.

`BACKING`: `huginn-work/svamp/latent-sweep/huginn_svamp/summary.txt`.

`TODO`: ablate — does Huginn improve on the counterfactual SVAMP variants
faster than vanilla? See `huginn-work/svamp/latent-sweep/huginn_cf_balanced/`.

### 4.4 SVAMP T✓ S✗ error taxonomy

A 30-question sample of the 153 SVAMP cells where the teacher is right and
the student is wrong (`codi-work/inference/analysis/svamp/...` /
`analysis/svamp_tcsi_error_analysis.md`):

| Bucket                  | % of sample |
| ----------------------- | ----------: |
| other                   |       43.3% |
| order_of_magnitude      |       26.7% |
| small_arithmetic_slip   |       23.3% |
| copy_input_number       |        3.3% |
| off_by_one              |        3.3% |

**Reading.** ~50% of the visible failures are *small numeric errors* —
arithmetic slips or order-of-magnitude scaling — not structural
misunderstandings of the problem. This already pre-figures the mechanistic
result (Section 6): the compression preserves the *operator* but loses
*precision in the magnitude*.

`BACKING`: `analysis/error_analysis_tcsi.py`, `analysis/svamp_tcsi_error_analysis.md`.

---

## 5. Workstream 2/3 — Probes on the Latent Trajectory

We probe the residual stream at two positions:

1. **The colon residual** — the residual stream at the `:` token of
   `"The answer is: "`, captured during answer emission. The next token the
   model will emit is the first digit of the answer. This is the
   highest-signal probe position we find.
2. **The latent residual** — the residual stream at the last token of each
   of the 6 latent thought iterations, every layer. Used for *trajectory*-level
   analysis (PCA, LDA over time).

For GPT-2 CODI the residual has 13 layer outputs (0 = embedding, 1–12 = block
outputs) and hidden dim 768. For LLaMA-1B CODI it has 17 layer outputs and
hidden dim 2048.

### 5.1 Operator probing

Train a 4-way linear classifier (Addition / Subtraction / Multiplication /
Common-Division) on the colon residual at each (layer, step):

- **GPT-2 CODI**, train on SVAMP colon acts, 80/20 split:
  - Train acc peaks at 1.00 (multiple cells); test acc peaks ~0.85.
  - Test acc rises from 0.51 (chance ≈ 0.53) at layer 0 to 0.85 by layer 4.
- **Generalisation to CF.** Trained on CF (`cf_balanced`), tested on SVAMP:
  best cell = (layer 8, step 3), CF train 1.00, CF test 0.853, SVAMP transfer **0.854**.

**Reading.** The operator is linearly decodable from a single residual vector
at the answer-emission position, *and the same direction transfers across
distribution shifts* (CF→SVAMP, surface re-worded). This is consistent with
the operator being part of the **compressed message** the student sends from
the latent loop into the answer head, not a per-problem confound.

`BACKING`: `codi-work/experiments/computation_probes/operator_probe_colon.{json,pdf}`,
`canonical_probe_grid.json`, `correct_only_probes.json`.

### 5.2 Number / magnitude probing — number isolation

PCA on the `vary_numerals` activations (operator fixed, `(a, b)` varied)
isolates *number*-encoding directions; PCA on `vary_operator` (numerals fixed
at `(12, 4)`, operator varied) isolates *operator/scenario*-encoding
directions. Comparing principal subspaces:

`‖U_A^T U_B‖²_F / k`  — measures how much the number subspace and the
operator subspace overlap.

**Reading.** *Operator and number directions live in approximately orthogonal
subspaces* of the residual stream (cosine alignment ≪ 1). This is the
geometric fact that allows the operator probe to transfer across CF, and the
geometric explanation for why operator-direction steering (Section 6.1) does
not destroy number information.

`BACKING`: `huginn-work/svamp/visualizations/number_isolation_pca.{py,json,png}`,
`codi-work/visualizations-all/gpt2/number_isolation_combined.{py,pdf}`,
`number_isolation_cossim.json`.

`TODO`: this currently has Huginn numbers; rerun on CODI-GPT-2 colon residuals
(scaffolding is in
`codi-work/visualizations-all/gpt2/number_isolation_combined_colon.py`).

### 5.3 Correctness / faithfulness probing

**Faithfulness probe v2** — predict whether the teacher's CoT is faithful
(581 faithful, 41 unfaithful), 5-fold CV, C-sweep, permutation null,
random-feature baseline. Best cell: **step 6, layer 10**, AUC 0.774.
Permutation null AUC ≈ 0.49; random-feature baseline AUC ≈ 0.42.

**Correctness probe at the colon position** — predict whether the student
got the answer right from the colon residual; clean signal in the upper
layers of the LLaMA-1B residual.

**Reading.** The model's own residual stream *knows* whether it's about to
be wrong, several layers before the answer token is emitted. This is
consistent with the Orgad et al. (2025) pre-answer-state result generalised
to latent CoT.

`BACKING`: `codi-work/experiments/computation_probes/faithfulness_probe_v2.{json,pdf}`,
`correctness_probe_colon.{json,pdf}`, `correct_only_probes.json`.

### 5.4 Logit lens

Project each saved (layer × step) activation through the base model's LM head
and report hit-rate at top-K (K∈{5, 10, 50}) against the gold answer token.

**Reading** *(write this once we finish the rendering pass):*
- The student's residual stream is "answer-shaped" several layers before the
  last layer, especially in late latent steps.
- Hit-rate rises monotonically with both layer index and step index; the
  steepest gradient is at the last 2–3 layers / last 2 steps.

`BACKING`: `inference/logit_lens.py`, `aggregate_logit_lens.py`,
`inference/analysis/{svamp,logic701}/hit_rate_topk*.png`.

### 5.5 Trajectory geometry — PCA / LDA slideshows

Two slideshow families:
1. **`pca_slideshow.pdf` / `lda_slideshow.pdf`** — 2D / 3D embedding of latent
   residuals at each (layer, step), coloured by operator or by output magnitude.
2. **`cf_lda_*` slideshows** — same, but trained on CF and visualised over
   the original SVAMP problems; tests whether the operator manifold is the
   same across surface re-writing.

**Reading.** Operator clusters separate cleanly by mid-network, are
consistent across scale (GPT-2 vs LLaMA-1B), and survive surface rewriting
(`cf_gpt_transformed`). Magnitude is partially separable but never cleanly so.

`BACKING`: `codi-work/visualizations-all/gpt2/{pca,lda,cf_lda_*}_slideshow*.pdf`,
`codi-work/visualizations-all/llama-1b/v1|v2/*.pdf`,
`codi-work/visualizations-student-correct/sc-v3/src_vs_surface_compare.{py,pdf}`.

### 5.6 Per-(layer, step) correctness probes — heatmaps

Train a logreg / LDA probe for *correctness* (was the final answer right?) at
every (step, layer) cell, with shuffled-label and random-feature controls:

- Best cell on SVAMP — TODO fill in from
  `inference/analysis/svamp/probe_accuracy/{teacher,student}_acc.png`.
- Shuffled-label control sits at ~chance; random-feature control sits at
  ~chance, confirming the signal is in the activations, not the labels.

`BACKING`: `inference/probe_accuracy.py`,
`inference/analysis/svamp/probe_accuracy/*.png`.

---

## 6. Workstream 4 — Causal Tests (Steering and Ablation)

Probes show what is *decodable*; steering and ablation show what is *used*.

### 6.1 Operator steering (Sub → Add, Mul → Com)

- Take LDA direction trained on CF `cf_balanced` to discriminate Subtraction
  vs Addition.
- Add `α · v_op` at peak (layer, step) = (10, 3) on subtraction problems and
  ask whether the answer flips to addition-like.

| Method                  | Cell                | tgt rate | src rate | Notes                                  |
| ----------------------- | ------------------- | -------: | -------: | -------------------------------------- |
| Baseline                | —                   |     0.01 |     0.29 | Sub problems decoded normally          |
| Single layer 1 / step 1 | (1, 1)              |     0.01 |     0.29 | No effect (early layer)                |
| `TODO`                  | (10, 3)             |  `TODO`  |  `TODO`  | Peak cell from probe                   |

**Reading** *(write up once table is filled):* operator steering at the peak
probe cell *causes* the student to emit answers consistent with the target
operator. The same direction generalises from CF training to SVAMP test
(per the probe transfer results).

`BACKING`: `codi-work/experiments/steering.py`,
`steering_results.json`, `steering_tt_*.json`,
`computation_probes/steering_operator_all.{json}`, `operator_steering_slideshow.pdf`.

### 6.2 Correctness steering

Push along the LDA "correct vs incorrect" direction at (step=2, layer=6).
Baseline 0.391. At α = −16: accuracy → 0.096 (898 changed, 24 wrong→right,
**319 right→wrong**, Δ = −0.295). At α = −8: 0.154 (Δ = −0.237). At α = −4:
intermediate.

**Reading.** A *single, linear direction* in the residual stream causally
controls whether the next answer will be correct, *independent of the
content of the problem*. This is the strongest evidence that the
correctness probe is reading a real mechanism, not a confound.

`BACKING`: `codi-work/experiments/computation_probes/steering_correctness.json`,
`steer_correctness.py`.

### 6.3 Bare-math magnitude steering

Disentangle "the model is computing the answer" from "the model is copying a
training-distribution answer". Take pure arithmetic prompts (`bare_math_*`),
identify a direction associated with output magnitude, push along it and
read off the median output magnitude vs baseline.

- Baseline median `|ans|` ≈ 191919 (corruption-style baseline; mostly
  meaningless answers).
- `HIGH | α = 800`: 183 / 200 changed, median `|ans|` shrinks to 660.
- `LOW | α = 800`: 187 / 200 changed, median `|ans|` shrinks to 9.

**Reading.** The student's *output magnitude* is also controllable by a
single direction. Combined with the operator result, this gives a 2-factor
decomposition of "what the colon residual is saying" — `(operator, magnitude)`
— each in approximately orthogonal subspaces.

`BACKING`: `codi-work/experiments/computation_probes/bare_math_steering.{py,json}`,
`bare_math_meta.json`.

### 6.4 Per-head ablation

For each attention head at each layer of CODI-GPT-2, zero out its output and
re-run on a fixed SVAMP slice. Report Δaccuracy.

- A small number of heads carry most of the answer-relevant signal.
- Ablating those heads on canonical (in-distribution) problems destroys
  accuracy; ablating them on the bare-math prompts has smaller effect — these
  heads are doing *problem-text → operator* routing.

`BACKING`: `codi-work/experiments/computation_probes/per_head_ablation_*.{py,json}`,
`codi-work/head-patching/ablate_codi_gpt2.{py}`, `ablation_codi_gpt2.{json}`,
`ablation_grid.png`, `ablation_perex_layer_summary.png`, `head_content_slideshow.pdf`.

### 6.5 Activation patching across CF pairs

For each (clean, corrupted) pair from `numeral_pairs_*`, run the model on
the corrupted input but patch in the clean activation at one
(layer, head/MLP) at a time; measure recovery toward the clean answer.

**Reading.** Recovery is concentrated at the *operator* sites identified
by the per-head ablation; numeral sites are spread more broadly. This is
consistent with the operator being a low-dimensional, late-emerging signal
and the numerals being represented throughout the network.

`BACKING`: `codi-work/head-patching/patching_gpt2.{py}`,
`numeral_a1_mul_recovery.json`, `numeral_b1_sub_recovery.json`,
`numeral_patching_summary.png`, `patching_slideshow.pdf`.

---

## 7. Workstream 3 (continued) — Cross-Model Comparison

### 7.1 Teacher / student representational distance

`TODO`: this analysis was added recently (see `audit_distance_analysis.md`;
implementation in `inference/analysis/svamp/distance_heatmap/`). Required
plot: per-(layer, step) cosine similarity / CKA between teacher final-layer
residual and student (layer, step) residual.

**Expected reading** *(write up after plots land):* student converges to
teacher *only* in late steps × upper layers; the off-diagonal cells of the
T×S confusion table show systematically lower convergence.

`BACKING`: `inference/analysis/svamp/distance_heatmap/`, `audit_distance_analysis.md`.

### 7.2 Same-question, same-prompt geometry

For the four cells (T✓ S✓), (T✓ S✗), (T✗ S✓), (T✗ S✗): compute the mean
trajectory in PCA(K=2 or 3) space of the student's latent residuals. Cluster
separation between (T✓ S✗) and (T✓ S✓) localises where the divergence
happens.

`BACKING`: `codi-work/visualizations-student-correct/sc-v1|v2-B|v2-C|sc-v3/`,
`cos_sim_b_vs_c.{py,pdf}`.

---

## 8. Discussion

### 8.1 What is the compression preserving?

- **Operator.** Linearly decodable at colon residual; transfers across
  surface rewriting; steerable.
- **Output magnitude.** Linearly decodable as a single direction;
  controllable via steering.
- **Correctness.** Linearly decodable several layers before the answer
  token; causally controllable.

### 8.2 What is the compression losing?

- **Numeric precision.** ~50% of T✓ S✗ failures are arithmetic slips or
  order-of-magnitude errors, not structural mistakes.
- **Free-text format compliance.** The student is over-trained on the
  GSM8k-Aug "The answer is: <number>" template and emits numerals even
  when the task asks for an option index (LOGIC-701 MC).
- **Generalisation OOD.** GSM-Hard accuracy is ~5× lower than SVAMP at the
  same scale.

### 8.3 Compression as abstraction?

- The CF surface-rewrite set (`cf_gpt_transformed`) is the right place to
  ask this — same operator, same numerals, fresh wording. `TODO`: run the
  current operator probe trained on SVAMP, evaluate on
  `cf_gpt_transformed`, and compare student vs teacher
  invariance.

### 8.4 Scale (GPT-2 vs LLaMA-1B)

- LLaMA-1B benefits monotonically from more latent steps on SVAMP;
  GPT-2 plateaus by `N=2`. Capacity matters for *using* latent compute.
- Operator probe transfers at both scales, with similar peak accuracy —
  the *representations* are scale-invariant in structure even if the
  *competence* is not.

### 8.5 Cross-paradigm

Huginn-3.5B at `K=32` reaches 0.142 on SVAMP — lower than a 1B distilled
CODI student at `K=6`. Distillation > pure recurrent depth at this task.
This bounds how much of CODI's score is "more compute" vs "the
right inductive bias from the teacher".

---

## 9. Limitations and Threats to Validity

- **LOGIC-701 parsing.** Student accuracy is currently underestimated; the
  log-likelihood scorer over the five options is the right fix and is
  not yet implemented.
- **Single-seed numbers.** All headline numbers are single-seed. Greedy
  decoding removes one source of variance but doesn't justify the absence
  of CI bars. `TODO`: bootstrap CIs on the confusion-cell counts.
- **Faithfulness labels.** GPT-5 judging is one labeller; we have 41
  unfaithful examples, which is a thin tail.
- **Probes are correlational by default.** The steering and ablation
  experiments are what convert probe results into causal claims; outside
  Section 6 we should hedge accordingly.
- **Helix probe not yet fit.** The mechanistic comparison to Kantamneni
  & Tegmark (2025) is in the proposal but is not yet implemented on
  this codebase.
- **Cross-model alignment.** Comparing teacher and student in the *same*
  residual space relies on the LoRA-adapted student sharing the base
  model's basis; this is by construction but worth stating.

---

## 10. Conclusion

`TODO`: rewrite once the report draft is complete. One-paragraph version:

> CODI matches teacher accuracy on the easy half of SVAMP, but diverges on
> a structured tail of problems — and the divergence is mostly numeric, not
> structural. The compressed latent stream linearly encodes the operator,
> the output magnitude, and the model's own correctness, in approximately
> orthogonal directions; steering and per-head ablation confirm the model
> *uses* these directions. The compression preserves the message; the
> arithmetic is what fails.

---

## 11. Author Contributions

- **Karen Li:** `TODO` (suggested: head patching / per-head ablation,
  Huginn cross-paradigm baseline, T×S distance heatmap).
- **Sandra Luo:** `TODO` (suggested: paired inference harness,
  latent-step sweep, GPT-2 colon-residual probes, computation-probes
  slideshows, faithfulness probe).
- **Hannah Tao:** `TODO` (suggested: probe-accuracy heatmaps, aggregate
  logit lens, Huginn inference + visualisations).

`TODO`: confirm with team; current attribution is inferred from commit log.

---

## 12. References

`TODO`: convert to proper BibTeX in final draft. Working list:

- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in
  Large Language Models.* NeurIPS. <https://arxiv.org/abs/2201.11903>
- Deng, Y. et al. (2023, 2024). *Implicit chain-of-thought via curriculum
  internalization.* <https://arxiv.org/html/2405.14838v1>
- Hao, S. et al. (2024). *Training Large Language Models to Reason in a
  Continuous Latent Space (Coconut).* <https://arxiv.org/abs/2412.06769>
- Shen, Z. et al. (2025). *CODI: Compressing Chain-of-Thought into
  Continuous Space via Self-Distillation.* EMNLP.
  <https://arxiv.org/abs/2502.21074>
- Gozeten, A. et al. (2026). *CoT² — multi-path continuous reasoning.*
  <https://arxiv.org/abs/2505.23648>
- Suleymanzade, A. et al. (2025). *On the limits of continuous chain-of-thought.*
  <https://openreview.net/forum?id=UQFTJPqJAc>
- Orgad, H. et al. (2025). *Pre-answer hidden states encode reasoning.*
  <https://openreview.net/forum?id=KRnsX5Em3W>
- Arcuschin, J. et al. (2025). *Faithful CoT under perturbation.*
  <https://arxiv.org/pdf/2503.08679>
- Kantamneni, S. & Tegmark, M. (2025). *Language Models Use Trigonometry
  to Do Addition.* <https://arxiv.org/pdf/2502.00873>
- Geiping, J. et al. (2025). *Huginn — recurrent-depth scaling at 3.5B.*
  `TODO`: confirm citation.

---

## Appendix A — Repository Map (what lives where)

```
nlp-final-project/
├── README.md                          Project entry point
├── V3_NLP_final_proj_proposal.md      Original proposal
├── REPORT.md                          (this file)
├── audit_distance_analysis.md         Audit of cross-model distance work
│
├── cf-datasets/                       Counterfactual + probe datasets
│   ├── README.md                      Generators + schemas
│   ├── cf_balanced.json               Canonical CF (676)
│   ├── cf_magmatched.json             Output-magnitude-matched CF (972)
│   ├── cf_under99{,_b}.json           ≤99-restricted CF variants
│   ├── cf_gpt_transformed.json        GPT-5 surface-rewritten CF (546)
│   ├── vary_numerals.json             Number-isolation probe (80)
│   ├── vary_operator.json             Operator-isolation probe (24)
│   ├── svamp_judged.json              GPT-5 faithfulness labels
│   └── generate_*.py                  Deterministic generators
│
├── codi/                              Vendored CODI reference impl
│
├── codi-work/                         Our CODI experiments
│   ├── inference/                     Paired teacher/student harness
│   │   ├── run_eval_with_hooks.py
│   │   ├── confusion.py
│   │   ├── logit_lens.py, aggregate_logit_lens.py
│   │   ├── probe_accuracy.py
│   │   ├── distance_heatmap.py
│   │   ├── question_features.py
│   │   ├── latent_filter_step_overview.py
│   │   ├── latent_sweep_step_overview.py
│   │   └── analysis/                  Outputs (heatmaps, plots, JSONs)
│   │
│   ├── latent-sweep/                  N=0..6 sweep, per dataset/scale
│   │   └── summary.txt                Headline table
│   │
│   ├── experiments/                   Causal + probe scripts
│   │   ├── compute_gpt2_centroids.py
│   │   ├── parallelogram.py
│   │   ├── reanalyze_4ops.py
│   │   ├── steering.py
│   │   ├── steering_train_test.py
│   │   ├── steering_smoke_gpt2.py
│   │   ├── runs_*.json                Probe results
│   │   ├── steering_*.json            Steering results
│   │   └── computation_probes/        The bulk of mechanistic analysis
│   │       ├── operator_probe_colon.{py,json,pdf}
│   │       ├── correctness_probe_colon.{py,json,pdf}
│   │       ├── faithfulness_probe{,_v2}.{py,json,pdf}
│   │       ├── canonical_probe.{py,pkl}
│   │       ├── correct_only_probes.{py,json}
│   │       ├── steer_correctness.py
│   │       ├── steer_operator_*.py
│   │       ├── steering_correctness.json
│   │       ├── bare_math_*.py
│   │       ├── force_decode_per_step{,_llama}.{py,json}
│   │       ├── flow_map.{py,npz}
│   │       ├── per_head_ablation*.{py,json}
│   │       ├── answer_direction_projection.{py,pdf,npz}
│   │       ├── pca_bifurcation_colon.{py,pdf,json}
│   │       ├── computation_probes_slideshow{,_colon}.pdf
│   │       ├── context_isolation_slideshow{,_colon}.pdf
│   │       ├── steering_slideshow{,_colon}.pdf
│   │       ├── operator_steering_slideshow.pdf
│   │       ├── head_content_slideshow.pdf
│   │       ├── latent_use_slideshow.pdf
│   │       └── _hf_README.md          HF dataset card for the colon-acts
│   │
│   ├── head-patching/                 Per-head ablation + patching
│   │   ├── ablate_codi_gpt2.py
│   │   ├── patching_gpt2.py, patching.py
│   │   ├── build_*_figure.py
│   │   ├── ablation_codi_gpt2.{json,.perex.npz}
│   │   ├── ablation_grid.png
│   │   ├── ablation_perex_*.png
│   │   ├── numeral_patching_summary.png
│   │   ├── patching_slideshow.pdf
│   │   ├── A_templates/, B_svamp_filtered/
│   │   └── build_cf_pairs.py
│   │
│   ├── visualizations-all/            PCA / LDA over full SVAMP
│   │   ├── gpt2/                      Pre- and colon-position decks
│   │   └── llama-1b/{v1,v2}/          Two iterations at LLaMA scale
│   │
│   ├── visualizations-student-correct/ PCA / LDA over student-correct subset
│   │   ├── sc-v1, sc-v2-B, sc-v2-C, sc-v3
│   │   └── cos_sim_b_vs_c{,_lda}.{py,pdf,json}
│   │
│   └── codi/                          Old vendored CODI copy
│
├── huginn-work/                       Recurrent-depth control
│   ├── inference/
│   │   ├── run_huginn_with_hooks.py
│   │   ├── diagnose_huginn_acts.py
│   │   ├── run_huginn_sweep.sh
│   │   └── analysis/                  svamp_huginn{,_smoke}/
│   ├── svamp/
│   │   ├── huginn/                    Patching + probing scripts
│   │   │   ├── patching_K_scan.py
│   │   │   ├── patching_per_head{,_all_K}.py
│   │   │   ├── probe_depth.py
│   │   │   ├── run_eval.py
│   │   │   └── steering_smoke.py
│   │   ├── latent-sweep/              K=1..32 per dataset
│   │   └── visualizations/            PCA / LDA / cos-sim decks + probes/
│   │
├── inference/                         (Newer harness, separate from codi-work/)
│   ├── analysis/svamp/distance_heatmap/   New T↔S distance work
│   ├── runs/{svamp,logic701}_{teacher,student}/
│   └── uv.lock
│
└── logic561.parquet                   Cleaned LOGIC-701 subset
```

---

## Appendix B — Reproduction Commands

### B.1 Paired inference (LLaMA-1B)

```bash
cd inference   # or codi-work/inference
bash run_sweep.sh                              # SVAMP+LOGIC-701, both modes
```

### B.2 Latent-step sweep

```bash
for N in 0 1 2 3 4 5 6; do
    python run_eval_with_hooks.py \
        --mode student --dataset svamp \
        --num_latent_steps $N \
        --out_dir ../latent-sweep/svamp_latent_sweep_llama/N$N
done
```

### B.3 Probes at the colon position

```bash
cd codi-work/experiments/computation_probes
python capture_colon_acts.py --dataset svamp           # writes svamp_colon_acts.pt
python operator_probe_colon.py                         # 4-way operator probe
python correctness_probe_colon.py                      # correctness probe
python faithfulness_probe_v2.py                        # faithfulness probe
```

### B.4 Steering

```bash
cd codi-work/experiments
python steering.py --src Subtraction --tgt Addition \
    --peak_layer 10 --peak_step 3 \
    --out steering_results.json

python computation_probes/steer_correctness.py \
    --target_layer 6 --target_step_1indexed 2 \
    --out computation_probes/steering_correctness.json
```

### B.5 Per-head ablation

```bash
cd codi-work/head-patching
python ablate_codi_gpt2.py --dataset svamp
python patching_gpt2.py --pairs numeral_pairs_a1_mul.json
python build_ablation_figure.py
```

### B.6 Huginn baseline

```bash
cd huginn-work/inference
bash run_huginn_sweep.sh                # SVAMP across K=1..32
```

---

## Appendix C — Figure Inventory

The figures we've already rendered, by which section will use them:

- **§4.1 Confusion tables.** Render from README markdown or
  `inference/runs/*/results.json`.
- **§4.2 Latent-step sweep.** `analysis/sweep_by_dataset.pdf`,
  `codi-work/inference/analysis/latent_sweep/sweep_*.png`,
  `analysis/headline_figure.png`.
- **§4.3 Huginn sweep.** `huginn-work/svamp/latent-sweep/huginn_svamp/summary.txt`
  — needs a plot script.
- **§5.1 Operator probe.** `operator_probe_colon.pdf`.
- **§5.2 Number isolation.** `number_isolation_combined.pdf`,
  `vary_numerals_pca_slideshow.pdf`, `vary_operator_pca_slideshow.pdf`.
- **§5.3 Faithfulness.** `faithfulness_probe_v2.pdf`,
  `correctness_probe_colon.pdf`, `pca_bifurcation_colon.pdf`.
- **§5.4 Logit lens.** `inference/analysis/svamp/hit_rate_topk{5,10,50}.png`,
  `inference/analysis/logic701/hit_rate_topk*.png`.
- **§5.5 PCA / LDA.** `pca_slideshow.pdf`, `lda_slideshow.pdf`,
  `cf_lda_slideshow.pdf`, `cf_lda_compare.pdf`,
  `src_vs_surface_compare.pdf`.
- **§5.6 Per-cell correctness.** `inference/analysis/svamp/probe_accuracy/*.png`.
- **§6.1 Operator steering.** `operator_steering_slideshow.pdf`,
  `steering_slideshow_colon.pdf`.
- **§6.2 Correctness steering.** `steering_alpha_curves.png` (`TODO`: confirm),
  `steering_cossim_curve.png`.
- **§6.3 Bare-math steering.** `steering_magnitude_curve.png`.
- **§6.4 Per-head ablation.** `ablation_grid.png`,
  `ablation_perex_layer_summary.png`, `head_content_slideshow.pdf`.
- **§6.5 Activation patching.** `numeral_patching_summary.png`,
  `patching_slideshow.pdf`.
- **§7.1 Distance heatmap.** `inference/analysis/svamp/distance_heatmap/*.png`.
- **§7.2 Same-question geometry.** `cos_sim_b_vs_c.pdf`,
  `cos_sim_b_vs_c_lda.pdf`, `sc-v3/src_vs_surface_compare.pdf`.

---

## Appendix D — Open TODOs Before Submission

1. **Numbers needing CIs.** Confusion-table cell counts; latent-step
   sweep best-`N` accuracies. Bootstrap is enough.
2. **LOGIC-701 LL scorer.** Re-score the student via log-likelihood over the
   five MC options; report next to the parse-based number.
3. **Helix probe.** Implement on student arithmetic activations, fit on
   convergent vs divergent SVAMP cells.
4. **Distance heatmap interpretation.** Plots exist; write the reading.
5. **Author contributions.** Sync with Sandra and Hannah.
6. **Bibliography.** Convert to BibTeX, lock down Huginn / Geiping citation.
7. **Trim.** This scaffold is intentionally long; the final report should be
   tight enough that every figure earns its slot.
