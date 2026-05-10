#!/usr/bin/env bash
# Orchestrate the Huginn sweep on a remote GPU box: push code, run inside
# tmux on the remote, wait for completion, pull results back. SSH disconnects
# on this side cannot kill the run because the work happens inside tmux on the
# remote; the local "wait" loop is just a poller, safe to Ctrl-C and resume.
#
# Usage:
#   ./run_remote_huginn.sh push           # rsync code  -> remote
#   ./run_remote_huginn.sh smoke [N]      # 50-question SVAMP dry run, NUM_STEPS=N (default 32)
#   ./run_remote_huginn.sh run [N]        # full sweep with NUM_STEPS=N (default 32)
#   ./run_remote_huginn.sh status         # tail the remote log
#   ./run_remote_huginn.sh wait           # block until remote tmux session ends
#   ./run_remote_huginn.sh pull           # rsync results -> local
#   ./run_remote_huginn.sh attach         # tmux attach to the live session
#   ./run_remote_huginn.sh kill           # kill the remote tmux session
#   ./run_remote_huginn.sh all [N]        # push + run + wait + pull
#
# Environment overrides (defaults shown):
#   REMOTE=ubuntu@89.169.113.79
#   REMOTE_DIR=nlp-final-project        # path relative to remote $HOME
#   REMOTE_PY=~/svamp-env/bin/python
#   SESSION=huginn
#   PULL_ACTS=0                         # set to 1 to also rsync the giant *.pt files
set -euo pipefail

REMOTE=${REMOTE:-ubuntu@89.169.113.79}
REMOTE_DIR=${REMOTE_DIR:-nlp-final-project}
REMOTE_PY=${REMOTE_PY:-\~/svamp-env/bin/python}
SESSION=${SESSION:-huginn}
LOCAL_DIR=$(cd "$(dirname "$0")" && pwd)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

usage() { sed -n '2,/^set -/p' "$0" | sed 's/^# \?//' | head -n 25; }

remote_log() { echo "$REMOTE_DIR/inference/huginn_sweep.log"; }

ssh_q() { ssh -o LogLevel=ERROR "$REMOTE" "$@"; }

# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------

cmd_push() {
    echo "[push] rsync $LOCAL_DIR/ -> $REMOTE:$REMOTE_DIR/"
    # Skip git + caches + giant binary tensors that get regenerated on the GPU
    # side anyway. --delete keeps the remote tree in sync (removes files
    # deleted locally) but only inside the rsync'd subset.
    #
    # IMPORTANT: protect anything inside inference/runs/ and inference/analysis/
    # on the receiving side so that remote-generated artifacts (results.json,
    # meta.json, activations.pt, *.png, etc.) are NOT deleted just because they
    # don't exist locally. Local files in those trees still get pushed normally.
    rsync -avz --delete \
        --filter='P inference/runs/***' \
        --filter='P inference/analysis/***' \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='inference/runs/*/activations.pt' \
        --exclude='inference/runs/*/logit_lens.pt' \
        --exclude='inference/analysis/*/ranks.pt' \
        --exclude='inference/analysis/*/*_predictions.pt' \
        --exclude='inference/sweep.log' \
        --exclude='inference/huginn_sweep.log' \
        --exclude='latent-sweep' \
        "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"
}

cmd_run() {
    local num_steps=${1:-32}
    echo "[run] launching sweep on $REMOTE (NUM_STEPS=$num_steps, tmux=$SESSION)"

    # Kill stale session, if any, so we always start clean.
    ssh_q "tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION || true"

    # Truncate the previous log so cmd_status / cmd_wait don't show stale tail.
    ssh_q ": > $(remote_log) || true"

    # The inner command runs inside the remote shell. We wrap it in a script
    # that always writes a sentinel last line so cmd_wait can detect "done"
    # even if the sweep itself crashed (the session ends regardless, but the
    # sentinel makes the log easier to read).
    local inner="cd $REMOTE_DIR/inference && PY=$REMOTE_PY NUM_STEPS=$num_steps bash run_huginn_sweep.sh; echo \"=== EXIT \$? ===\""
    ssh_q "tmux new-session -d -s $SESSION '$inner'"

    echo "[run] launched. Track with:"
    echo "    $0 status        # tail log"
    echo "    $0 wait          # block until done"
    echo "    $0 attach        # interactive tmux"
}

cmd_status() {
    echo "[status] tailing $(remote_log)  (Ctrl-C to stop)"
    ssh -t "$REMOTE" "tail -f $(remote_log)"
}

cmd_attach() {
    echo "[attach] attaching to tmux session '$SESSION'  (Ctrl-b d to detach)"
    ssh -t "$REMOTE" "tmux attach -t $SESSION"
}

cmd_kill() {
    echo "[kill] killing tmux session '$SESSION' on $REMOTE"
    ssh_q "tmux kill-session -t $SESSION 2>/dev/null || true"
}

cmd_wait() {
    echo "[wait] polling until tmux session '$SESSION' ends..."
    local last_seen=""
    while ssh_q "tmux has-session -t $SESSION 2>/dev/null"; do
        local line
        line=$(ssh_q "tail -n1 $(remote_log) 2>/dev/null" || true)
        if [[ "$line" != "$last_seen" ]]; then
            # Use `date` rather than printf %(...)T so we work on macOS bash 3.2.
            local ts
            ts=$(date +%H:%M:%S)
            # Truncate to 180 chars without printf %.180s (which can choke on % in $line).
            local trimmed=${line:0:180}
            echo "[wait $ts] $trimmed"
            last_seen=$line
        fi
        sleep 30
    done
    echo "[wait] session ended. last log lines:"
    ssh_q "tail -n 5 $(remote_log)" || true
}

cmd_pull() {
    echo "[pull] rsync $REMOTE:$REMOTE_DIR/inference/{runs,analysis,*.log} -> $LOCAL_DIR/inference/"
    mkdir -p \
        "$LOCAL_DIR/inference/runs/svamp_huginn" \
        "$LOCAL_DIR/inference/analysis/svamp_huginn"

    # By default skip the giant *.pt activations; pass PULL_ACTS=1 to include
    # them (they're ~1-3 GB for SVAMP and gitignored anyway).
    # NOTE: rsync source root is the run/analysis dir itself, so files like
    # activations.pt are at the top level. `*/foo.pt` requires a directory
    # prefix and would NOT match top-level files; we use bare basenames here.
    local excludes=()
    if [[ "${PULL_ACTS:-0}" != "1" ]]; then
        excludes+=( --exclude='activations.pt'
                    --exclude='logit_lens.pt'
                    --exclude='ranks.pt'
                    --exclude='*_predictions.pt' )
    fi

    # ${arr[@]+"${arr[@]}"} is the bash idiom for "expand if set, else expand
    # to nothing" — needed because set -u trips on "${arr[@]}" when arr=()
    # (empty array case happens when PULL_ACTS=1).
    rsync -avz ${excludes[@]+"${excludes[@]}"} \
        "$REMOTE:$REMOTE_DIR/inference/runs/svamp_huginn/" \
        "$LOCAL_DIR/inference/runs/svamp_huginn/"

    rsync -avz \
        "$REMOTE:$REMOTE_DIR/inference/analysis/svamp_huginn/" \
        "$LOCAL_DIR/inference/analysis/svamp_huginn/"

    rsync -avz \
        "$REMOTE:$REMOTE_DIR/inference/huginn_sweep.log" \
        "$LOCAL_DIR/inference/huginn_sweep.log" || true

    echo "[pull] done. local files:"
    ls -lh "$LOCAL_DIR/inference/runs/svamp_huginn/" 2>/dev/null || true
    ls -lh "$LOCAL_DIR/inference/analysis/svamp_huginn/" 2>/dev/null || true
}

cmd_smoke() {
    local num_steps=${1:-32}
    echo "[smoke] 50-question SVAMP eval on $REMOTE (NUM_STEPS=$num_steps, tmux=$SESSION-smoke)"
    ssh_q "tmux has-session -t ${SESSION}-smoke 2>/dev/null && tmux kill-session -t ${SESSION}-smoke || true"
    # IMPORTANT: $log is given relative to HOME (so `: > $log` works pre-cd) but
    # tee runs *after* `cd $REMOTE_DIR/inference`, so we tee to the basename
    # (`huginn_smoke.log`) which resolves to the same file on the remote.
    local log="$REMOTE_DIR/inference/huginn_smoke.log"
    ssh_q ": > $log || true"
    local inner="cd $REMOTE_DIR/inference && $REMOTE_PY -u run_huginn_with_hooks.py --dataset svamp --num_examples 50 --num_steps $num_steps --out_dir runs/svamp_huginn_smoke 2>&1 | tee huginn_smoke.log; echo \"=== EXIT \$? ===\""
    ssh_q "tmux new-session -d -s ${SESSION}-smoke '$inner'"
    echo "[smoke] launched. Watch with:  ssh $REMOTE 'tail -f $log'"
    echo "[smoke] or attach: SESSION=${SESSION}-smoke $0 attach"
}

cmd_all() {
    cmd_push
    cmd_run "${1:-32}"
    cmd_wait
    cmd_pull
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

case "${1:-}" in
    push)    shift; cmd_push   "$@" ;;
    smoke)   shift; cmd_smoke  "$@" ;;
    run)     shift; cmd_run    "$@" ;;
    status)  shift; cmd_status "$@" ;;
    wait)    shift; cmd_wait   "$@" ;;
    attach)  shift; cmd_attach "$@" ;;
    kill)    shift; cmd_kill   "$@" ;;
    pull)    shift; cmd_pull   "$@" ;;
    all)     shift; cmd_all    "$@" ;;
    -h|--help|help|"") usage; exit 0 ;;
    *) echo "unknown subcommand: $1" >&2; usage; exit 1 ;;
esac
