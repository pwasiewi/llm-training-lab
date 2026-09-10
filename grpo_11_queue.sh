#!/usr/bin/env bash
# Sequential experiment runner for the grpo_11 family: train -> merge -> eval -> paired compare.
# One GPU, one run at a time, no babysitting.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$HERE/grpo_11_qwen3_17b_math.py"
EVAL="$HERE/grpo_11_qwen3_17b_math_test.py"
BASE_EVAL="$HERE/eval_math500_base.jsonl"
STEPS="${STEPS:-200}"
MAXTOK="${MAXTOK:-2048}"

die() { echo "ERROR: $*" >&2; exit 1; }

help() {
    cat <<'EOF'
grpo_11_queue.sh — run a queue of grpo_11 experiments back to back.

  grpo_11_queue.sh TAG=ENV[,ENV...] [TAG=ENV...]

Each argument is one experiment: a tag, '=', then a comma-separated list of the
GRPO_* environment settings that make this experiment differ from e0. Use '-' for
"no overrides". Per experiment the script runs:

  1. training      GRPO_TAG=<tag> GRPO_MAX_STEPS=$STEPS  -> grpo11_<tag>_run.log
  2. merge         done by the training script itself (offline, CPU, bug (d) safe)
  3. eval          greedy MATH-500                       -> eval_math500_<tag>.jsonl
  4. compare       exact McNemar vs eval_math500_base.jsonl

Environment:
  STEPS   optimizer steps per experiment (default 200)
  MAXTOK  eval max completion tokens (default 2048)

Examples:
  grpo_11_queue.sh e1=GRPO_STEP=1,GRPO_W_STEP=0.5
  STEPS=400 grpo_11_queue.sh e2=GRPO_LR=1e-5 e3=GRPO_G=8

Refuses to start while another process holds the GPU (a leftover VLLM::EngineCore
from a previous eval will otherwise OOM the training run).
EOF
}

[[ $# -eq 0 || ${1:-} == -h || ${1:-} == --help ]] && { help; exit 0; }
[[ -f $TRAIN ]] || die "missing $TRAIN"
[[ -f $EVAL ]] || die "missing $EVAL"
[[ -f $BASE_EVAL ]] || die "missing baseline $BASE_EVAL — run the base eval first"

gpu_busy() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
    [[ $used -gt 3000 ]]
}

wait_for_gpu() {
    local waited=0
    while gpu_busy; do
        [[ $waited -ge 600 ]] && die "GPU still busy after 10 min: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"
        echo "[queue] GPU busy, waiting... ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
        sleep 30
        waited=$((waited + 30))
    done
}

cd "$HERE"
for spec in "$@"; do
    tag="${spec%%=*}"
    envs="${spec#*=}"
    [[ -n $tag && $tag != "$spec" ]] || die "bad spec '$spec' (want TAG=ENV=VAL,...)"

    declare -a envargs=()
    if [[ $envs != "-" ]]; then
        IFS=',' read -r -a parts <<< "$envs"
        for kv in "${parts[@]}"; do
            [[ $kv == *=* ]] || die "bad env '$kv' in '$spec'"
            envargs+=("$kv")
        done
    fi

    echo "=== [queue] experiment $tag: ${envargs[*]:-no overrides}  (steps=$STEPS) ==="
    wait_for_gpu

    echo "[queue] $tag: training -> grpo11_${tag}_run.log"
    if ! env "${envargs[@]}" GRPO_TAG="$tag" GRPO_MAX_STEPS="$STEPS" GRPO_PRINT_EVERY=25 \
            python3 "$TRAIN" > "grpo11_${tag}_run.log" 2>&1; then
        echo "[queue] $tag: TRAINING FAILED (see grpo11_${tag}_run.log), skipping to next"
        continue
    fi

    merged="outputs/lora-grpo-qwen3-17b-math-${tag}"
    [[ -d $merged ]] || { echo "[queue] $tag: no merged model at $merged, skipping eval"; continue; }

    wait_for_gpu
    echo "[queue] $tag: eval -> eval_math500_${tag}.jsonl"
    if ! python3 "$EVAL" "$merged" --max-tokens "$MAXTOK" --out "eval_math500_${tag}.jsonl" \
            > "grpo11_${tag}_eval.log" 2>&1; then
        echo "[queue] $tag: EVAL FAILED (see grpo11_${tag}_eval.log)"
        continue
    fi

    echo "[queue] $tag: paired comparison vs base"
    python3 "$EVAL" --compare "$BASE_EVAL" "eval_math500_${tag}.jsonl" | tee "grpo11_${tag}_compare.txt"
done
echo "=== [queue] done ==="
