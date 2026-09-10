#!/usr/bin/env bash
# Phase 2 of the grpo_11 experiments, run in the FOREGROUND so the output is visible.
#   1. difficulty pre-score of the training prompts (skipped if the file exists)
#   2. one long run with partial credit + the difficulty filter + a dev-eval learning curve
#   3. final greedy MATH-500 eval and the paired comparison against the base model
#
# Everything is also tee'd to a log, so it can be followed from another terminal with:
#   tail -f ~/Claude/llm/grpo11_g1_run.log | grep -E 'DEVEVAL|TRIPWIRE'
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TAG="${TAG:-g1}"
STEPS="${STEPS:-800}"
PASS="${PASS:-math_l35_pass.jsonl}"
PRESCORE_LIMIT="${PRESCORE_LIMIT:-3000}"
# 0.85 OOMs during engine warm-up while a KDE desktop holds ~2 GB of the 16.
PRESCORE_GMU="${PRESCORE_GMU:-0.75}"

die() { echo "ERROR: $*" >&2; exit 1; }

help() {
    cat <<'EOF'
grpo_11_phase2.sh — pre-score, one long GRPO run, final eval. Foreground, visible output.

  ./grpo_11_phase2.sh                 # tag g1, 800 steps
  STEPS=400 TAG=g2 ./grpo_11_phase2.sh

Environment: TAG, STEPS, PASS (pass-filter file), PRESCORE_LIMIT.
Takes roughly 1 hour of pre-scoring plus ~40 s per training step plus ~12 min of eval.
EOF
}

[[ ${1:-} == -h || ${1:-} == --help ]] && { help; exit 0; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ $used -gt 3000 ]] && die "GPU busy (${used} MiB). A leftover VLLM::EngineCore will OOM the run: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"

echo "=== [phase2] $(date +%H:%M) step 1/3: difficulty pre-score ==="
if [[ -f $PASS ]]; then
    echo "[phase2] $PASS exists, skipping pre-score ($(wc -l < "$PASS") prompts scored)"
else
    python3 grpo_11_prescore.py --limit "$PRESCORE_LIMIT" --n 4 --gmu "$PRESCORE_GMU" --out "$PASS" 2>&1 | tee "grpo11_prescore.log"
fi

echo
echo "=== [phase2] $(date +%H:%M) step 2/3: long run '$TAG' ($STEPS steps) ==="
echo "[phase2] partial credit + pass filter + dev-eval every 50 steps"
GRPO_TAG="$TAG" \
GRPO_LR=1e-5 \
GRPO_STEP=1 GRPO_W_STEP=0.5 \
GRPO_PASS_FILTER="$PASS" \
GRPO_DEVEVAL=50 GRPO_DEVN=200 \
GRPO_MAX_STEPS="$STEPS" \
GRPO_PRINT_EVERY=50 \
    python3 grpo_11_qwen3_17b_math.py 2>&1 | tee "grpo11_${TAG}_run.log"

merged="outputs/lora-grpo-qwen3-17b-math-${TAG}"
[[ -d $merged ]] || die "no merged model at $merged"

echo
echo "=== [phase2] $(date +%H:%M) step 3/3: final eval + paired comparison ==="
python3 grpo_11_qwen3_17b_math_test.py "$merged" --max-tokens 2048 \
    --out "eval_math500_${TAG}.jsonl" 2>&1 | tee "grpo11_${TAG}_eval.log"
python3 grpo_11_qwen3_17b_math_test.py --compare eval_math500_base.jsonl "eval_math500_${TAG}.jsonl" \
    | tee "grpo11_${TAG}_compare.txt"

echo
echo "=== [phase2] $(date +%H:%M) done. Learning curve: ==="
grep -h DEVEVAL "grpo11_${TAG}_run.log" | tail -1
echo "Trend:  python3 grpo_11_analyze.py grpo11_${TAG}_run.log"
