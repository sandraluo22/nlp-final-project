#!/usr/bin/env bash
# Run the four-way sweep: {teacher,student} x {svamp,logic701}.
# Output is appended to ./sweep.log.
set -e
cd "$(dirname "$0")"
PY=~/svamp-env/bin/python
LOG=sweep.log
BS=${BS:-32}

run() {
    local tag="$1"; shift
    echo "=== $tag ===" | tee -a "$LOG"
    $PY -u run_eval_with_hooks.py "$@" 2>&1 | tee -a "$LOG" | grep -vE "(top_p|FutureWarning|warnings\\.warn|multivariate)"
}

run "teacher SVAMP"      --mode teacher --dataset svamp    --batch_size "$BS" --out_dir runs/svamp_teacher
run "student SVAMP"      --mode student --dataset svamp    --batch_size "$BS" --out_dir runs/svamp_student
run "teacher LOGIC-701"  --mode teacher --dataset logic701 --batch_size "$BS" --max_new_tokens 384 --out_dir runs/logic701_teacher
run "student LOGIC-701"  --mode student --dataset logic701 --batch_size "$BS" --max_new_tokens 384 --out_dir runs/logic701_student
echo "=== ALL DONE ===" | tee -a "$LOG"
