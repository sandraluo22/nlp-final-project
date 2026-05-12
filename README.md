# Latent Reasoning Interpretability in CODI-Distilled Models

## Repository layout

| Path | Purpose |
|------|---------|
| [`codi-work/`](codi-work/) | Main experiments: `inference/` (hooks, logit lens, probes), `visualizations-*`, `experiments/`, `latent-sweep/`, counterfactual tooling. |
| [`huginn-work/`](huginn-work/) | Huginn sweep scripts, SVAMP analyses that mirror CODI-style figures, and related visualizations. |
| [`inference/`](inference/) | Root-level Huginn evaluation helpers used with [`run_remote_huginn.sh`](run_remote_huginn.sh) (rsync + tmux remote GPU runs). |
| [`codi/`](codi/) | Upstream CODI training code and configs (from the original CODI paper release). |
| [`cf-datasets/`](cf-datasets/) | Counterfactual and judged SVAMP variants for robustness / faithfulness. |
| [`latent-sweep/`](latent-sweep/) | Latent-step sweep artifacts (also mirrored under `codi-work/latent-sweep/`). |
| [`REPORT.md`](REPORT.md) | Full project report scaffold: claims, figure inventory, and writing `TODO`s. |

Large tensors (`*.pt`, checkpoints) are **gitignored** and documented under `codi-work/inference/runs/README.md`.

---

## Setup

**Python 3.10+** recommended. Dependencies vary by subproject; typical stacks include `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `safetensors`, `scikit-learn`, `matplotlib`, `huggingface_hub`.

```bash
git clone https://github.com/sandraluo22/nlp-final-project.git
cd nlp-final-project
```

For gated models (e.g. Meta Llama mirrors), authenticate once:

```bash
hf auth login
```

---

## Data & activations

Per-question **`results.json`** files are committed under `codi-work/inference/runs/*/`.

**Activation tensors** (`activations.pt`) are large (~tens–hundreds of MB per split) and live in some companion Hugging Face datasets:

**[sandrajyluo/nlp-final-project-activations](https://huggingface.co/datasets/sandrajyluo/nlp-final-project-activations)**

**[sandrajyluo/codi-gpt2-svamp-activations](https://huggingface.co/datasets/sandrajyluo/codi-gpt2-svamp-activations)**

**[karenli1/nlp-final-project-activations-3step](https://huggingface.co/datasets/karenli1/nlp-final-project-activations-3step)**

Place downloaded files next to the corresponding `results.json` (see `codi-work/inference/runs/README.md` for shapes and naming).

---

## Running inference (CODI teacher / student)

From `codi-work/inference/` (GPU required):

```bash
python run_eval_with_hooks.py --mode teacher --dataset svamp --out_dir runs/svamp_teacher
python run_eval_with_hooks.py --mode student --dataset svamp --out_dir runs/svamp_student
```

Orchestration for multi-dataset sweeps may use `run_sweep.sh` and related scripts in the same directory.

---

## Huginn on a remote GPU

Huginn needs CUDA, `trust_remote_code`, and a recent PyTorch stack (see project notes in `huginn-work/inference/` and root `inference/`).

From the **repository root**:

```bash
chmod +x run_remote_huginn.sh
./run_remote_huginn.sh push          # rsync to your GPU host
./run_remote_huginn.sh smoke 32     # optional 50-example smoke test
./run_remote_huginn.sh run 32       # full SVAMP pipeline in tmux
./run_remote_huginn.sh pull          # pull analyses (default: skip giant *.pt)
PULL_ACTS=1 ./run_remote_huginn.sh pull   # include activations.pt if needed locally
```

Override `REMOTE`, `REMOTE_DIR`, and `REMOTE_PY` as documented in the script header.

---

## Analysis utilities

| Script / area | Role |
|---------------|------|
| `logit_lens.py` | Project saved residuals through the model LM head (unsupervised “what token?”). |
| `aggregate_logit_lens.py` | Gold-token rank heatmaps by teacher/student outcome cell. |
| `probe_accuracy.py` | Cross-validated linear probes for correctness vs. hidden states. |
| `filter_codi_latent_steps.py` | Filter examples by logit-lens rank transitions across latent steps. |
| `distance_heatmap.py` | Cosine / distance summaries for activation geometry. |

Paths above exist under **`codi-work/inference/`**; Huginn-specific entry points may also sit under **`huginn-work/inference/`** or root **`inference/`** depending on your checkout layout.

---

## Documentation

- **Evidence and narrative outline:** [`REPORT.md`](REPORT.md)  
- **Activation bundle schema & confusion tables:** [`codi-work/inference/runs/README.md`](codi-work/inference/runs/README.md)

---

## Citation

If you use this codebase or the released activation bundle, please cite the original **CODI** paper and credit this project as appropriate for your venue:

> Shen et al., *Chain of Implicit Thought: Implicit Reasoning in Language Models Through Continuous Latent Thought*, 2025.  
> [arXiv:2502.10970](https://arxiv.org/abs/2502.10970)

Huginn-0125: see the model card at [tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125).
