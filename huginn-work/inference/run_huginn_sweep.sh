#!/usr/bin/env bash
# Run Huginn-0125 with hooks across SVAMP and LOGIC-701, then run logit_lens.py
# and aggregate_logit_lens.py on the saved activations. Output appended to
# ./huginn_sweep.log.
#
# REQUIRES: NVIDIA GPU (H100 recommended), trust_remote_code-friendly env.
# Will NOT work on a Mac without CUDA.
set -e
cd "$(dirname "$0")"
CODI_INF="../../codi-work/inference"
PY=${PY:-~/svamp-env/bin/python}
LOG=huginn_sweep.log
NUM_STEPS=${NUM_STEPS:-32}

run() {
    local tag="$1"; shift
    echo "=== $tag ===" | tee -a "$LOG"
    $PY -u "$@" 2>&1 | tee -a "$LOG" | grep -vE "(FutureWarning|warnings\\.warn|huggingface_hub)"
}

# ---- 1. Collect activations on SVAMP ----
run "huginn SVAMP eval (num_steps=$NUM_STEPS)" \
    run_huginn_with_hooks.py \
    --dataset svamp \
    --num_steps "$NUM_STEPS" \
    --out_dir runs/svamp_huginn

# ---- 2. Logit-lens decode through Huginn's own LM head ----
run "huginn SVAMP logit lens" \
    "$CODI_INF/logit_lens.py" \
    --activations runs/svamp_huginn/activations.pt \
    --results    runs/svamp_huginn/results.json \
    --base_model tomg-group-umd/huginn-0125 \
    --trust_remote_code \
    --out_dir    runs/svamp_huginn

# ---- 3. Aggregate gold-rank heatmaps vs the existing teacher run ----
run "huginn SVAMP aggregate logit lens" \
    "$CODI_INF/aggregate_logit_lens.py" \
    --student_acts    runs/svamp_huginn/activations.pt \
    --student_results runs/svamp_huginn/results.json \
    --teacher_results "$CODI_INF/runs/svamp_teacher/results.json" \
    --base_model      tomg-group-umd/huginn-0125 \
    --trust_remote_code \
    --out_dir         analysis/svamp_huginn \
    --dataset_name    "SVAMP (Huginn-0125, num_steps=$NUM_STEPS)"

# ---- 4. (Optional) repeat for LOGIC-701; uncomment when ready ----
# run "huginn LOGIC-701 eval" \
#     run_huginn_with_hooks.py \
#     --dataset logic701 --num_steps "$NUM_STEPS" --max_new_tokens 384 \
#     --out_dir runs/logic701_huginn
# run "huginn LOGIC-701 logit lens" \
#     "$CODI_INF/logit_lens.py" --activations runs/logic701_huginn/activations.pt \
#     --results runs/logic701_huginn/results.json \
#     --base_model tomg-group-umd/huginn-0125 --trust_remote_code \
#     --out_dir runs/logic701_huginn
# run "huginn LOGIC-701 aggregate logit lens" \
#     "$CODI_INF/aggregate_logit_lens.py" \
#     --student_acts    runs/logic701_huginn/activations.pt \
#     --student_results runs/logic701_huginn/results.json \
#     --teacher_results "$CODI_INF/runs/logic701_teacher/results.json" \
#     --base_model      tomg-group-umd/huginn-0125 \
#     --trust_remote_code \
#     --out_dir         analysis/logic701_huginn \
#     --dataset_name    "LOGIC-701 (Huginn-0125)"

echo "=== ALL DONE ===" | tee -a "$LOG"
