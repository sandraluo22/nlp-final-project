# NLP Final Project Proposal: When Do CODI Students Diverge from Their Teachers?

Karen Li, Sandra Luo, Hannah Tao

## Motivation

CODI (Continuous Chain-of-Thought via Self-Distillation) trains a student model to compress explicit chain-of-thought reasoning into a continuous latent space, achieving 2.7–5.9x inference speedups while matching teacher performance on GSM8k [4]. But matching aggregate accuracy hides a more interesting question: **on which problems does the student diverge from its teacher, and what is special about those cases?**

Divergence is informative in both directions. Cases where the teacher succeeds but the student fails reveal what reasoning structures resist compression — perhaps deeper chains, or problems whose solution depends on details squeezed out of the latent bottleneck. Cases where the student succeeds but the teacher fails may indicate that compression forces useful abstraction over irrelevant surface form.

We characterize divergence along three axes:

- **Where:** What domains and problem types produce systematic divergence?
- **What:** What geometric, sparsity, or mechanistic features distinguish divergent latent trajectories?
- **Why:** Does compression help or hurt the model's ability to abstract across surface variation?

## Related Work

**Latent reasoning.** Deng et al. (2023, 2024) distill CoT into models that skip intermediate tokens via curriculum-based internalization [2]. Coconut autoregressively propagates hidden states as continuous thought tokens via a multi-stage curriculum [3]. CODI replaces curriculum learning with single-stage self-distillation, matching explicit CoT-SFT on GSM8k at GPT-2 scale [4]. CoT² forms continuous tokens as convex combinations of vocabulary embeddings to track multiple reasoning paths [5].

**Interpretability of latent reasoning.** Recent work shows continuous CoT degrades under depth and multi-path diversity [6]. The original CODI paper probes individual latent tokens and their attention patterns [4], building on Orgad et al. (2025), who show pre-answer hidden states encode essential reasoning information [7]. These analyses focus on what latent tokens encode *in aggregate*; we focus on the cases where the compression visibly fails.

**Mechanistic analysis of arithmetic.** Kantamneni & Tegmark (2025) show LLMs represent numbers on a helix and execute addition via trigonometric operations [9]. We adapt this style of analysis to ask whether latent reasoning solves arithmetic via similar mechanisms in convergent cases, and whether that structure breaks down in divergent ones.

## Proposed Research

**Workstream 1: Mapping divergence.** We run paired teacher/student inference across:

- **In-distribution math:** GSM8k (calibration).
- **OOD math:** GSM1k, SVAMP, MATH (varied difficulty and surface form).
- **Logic:** LOGIC-701 (multi-step deductive reasoning).
- **Commonsense:** CommonsenseQA (non-numeric reasoning).

For each example we record (teacher correct, student correct) and isolate the off-diagonal cells.

**Workstream 2: Counterfactual perturbations.** To localize *what kind* of variation causes divergence, we construct paired problems that independently vary one factor at a time:

- Surface context (apples → tickets, equation fixed)
- Numbers (template fixed, values changed)
- Logic structure (wording fixed, operation changed)
- Distractors (irrelevant sentence or number added)
- Presentation order (facts shuffled)

If the student diverges from the teacher on distractor variants but not surface variants, that localizes a specific compression failure mode.

**Workstream 3: Characterizing divergent trajectories.** For divergent vs convergent cases we compare:

- **Geometry:** Do divergent latent trajectories cluster separately? Are they further from teacher hidden states under cosine similarity and activation patching?
- **Sparsity:** Do failures correlate with over-dense or under-sparse latent features (e.g., activation L0, or SAE features where available)?
- **Mechanism:** For arithmetic, do divergent cases show degraded helix structure in number representations relative to convergent cases [9]?

**Workstream 4 (stretch): Compression and abstraction.** Does the bottleneck help or hurt abstraction? We compare CODI at GPT-2 scale against a larger backbone (LLaMA-3 8B if a CODI variant is available or trainable in scope) on the counterfactual paired sets. If compression forces useful abstraction, students should be *more* invariant than teachers to surface and distractor changes. If compression destroys detail, students should be *less* invariant to numeric and structural changes.

## Evaluation

**Divergence map.** Per-dataset confusion of teacher vs student outcomes; rates of (T✓S✗) and (T✗S✓) with bootstrap CIs against majority-class and self-agreement baselines.

**Counterfactual sensitivity.** For each perturbation axis, report change in student accuracy and change in teacher–student agreement. A perturbation matters if it shifts agreement significantly more than a no-op rephrasing baseline.

**Trajectory characterization.** AUC of a binary classifier predicting divergence from latent-trajectory features (geometric, sparsity, mechanistic) at trajectory prefixes of 25/50/75/100%. Shuffled-trajectory and random-feature controls confirm signal.

**Mechanistic.** For arithmetic problems, fit the helix probe of [9] on student latent states and report fit quality on convergent vs divergent subsets.

**Robustness.** Linear vs nonlinear probes; train on GSM8k, evaluate on GSM1k/SVAMP to test generalization; report all metrics with seed variance.

## Implementation

- **Data:** GSM8k, GSM1k, SVAMP, MATH, LOGIC-701, CommonsenseQA. Counterfactual paired sets generated semi-automatically (template + LLM rewriting, with manual checks).
- **Models:** Released CODI checkpoints (https://github.com/zhenyi4/codi). Primary: GPT-2 scale. Stretch: LLaMA-3 8B.
- **Tooling:** PyTorch, transformers, scikit-learn for probes, matplotlib.

## References

[1] Wei et al. (2023) — https://arxiv.org/abs/2201.11903
[2] Deng et al. (2023, 2024) — https://arxiv.org/html/2405.14838v1
[3] Hao et al. (2024) — https://arxiv.org/abs/2412.06769
[4] Shen et al. (2025) — https://aclanthology.org/2025.emnlp-main.36.pdf
[5] Gozeten et al. (2026) — https://arxiv.org/abs/2505.23648
[6] Suleymanzade et al. (2025) — https://openreview.net/forum?id=UQFTJPqJAc
[7] Orgad et al. (2025) — https://openreview.net/forum?id=KRnsX5Em3W
[8] Arcuschin et al. (2025) — https://arxiv.org/pdf/2503.08679
[9] Kantamneni & Tegmark (2025), "Language Models Use Trigonometry to Do Addition" — https://arxiv.org/pdf/2502.00873
