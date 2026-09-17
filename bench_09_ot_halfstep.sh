#!/bin/bash
# bench_09_ot_halfstep.sh — expert placement finer than --n-cpu-moe, and a
# check of llama-server's --fit auto-placement against the hand-tuned floors.
#
# Why (BENCH.md → Literature, ATSInfer arXiv:2607.10183): the paper's whole
# result is that per-TENSOR placement beats per-layer placement (up to 3.29×
# decode vs llama.cpp on a 6 GB card, GPT-OSS-20B / Qwen3-30B-A3B among the
# models). llama.cpp cannot re-plan at runtime like ATSInfer, but its
# --override-tensor already allows tensor-level placement. --n-cpu-moe N moves
# ALL THREE expert tensors (ffn_up_exps, ffn_gate_exps, ffn_down_exps — equal
# sized, ~268M params each in Qwen3.6-35B-A3B) of layers 0..N-1 to the CPU, so
# the VRAM floor is probed in whole-layer steps of ~3 tensors. This script adds
# the steps in between: base N, plus for layer N only some of the three.
#   HALF=down      layer N: ffn_down_exps → CPU        (1/3 step)
#   HALF=up,gate   layer N: ffn_up_exps + ffn_gate_exps → CPU  (2/3 step)
# The gain is a lower floor for the same context (or a context bump at the same
# tg) whenever the hand-found floor sits just above a whole-layer boundary.
# Shared-expert, attention, norm and ssm tensors are never touched: the paper
# and llama.cpp agree they belong on the GPU.
#
# Mode -f starts llama-server with --fit on (unset args adjusted to fit device
# memory) and prints what it chose, to compare with the models.conf floors.
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL="${MODEL:-$MODEL_ROOT/qwen36-35b-a3b/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf}"
NCMOE="${NCMOE:-14}"                 # whole-layer base (the current floor from models.conf)
HALF_LIST="${HALF_LIST:-none,down,up+gate}"   # extra placement for layer NCMOE
STEPS="${STEPS:-0,-1}"               # base offsets to try: 0 = NCMOE, -1 = NCMOE-1 (+half)
THREADS="${THREADS:-16}"
CTX="${CTX:-131072}"                 # server mode only
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
FA_FLAG="${FA_FLAG:-on}"
BENCH_BIN="${BENCH_BIN:-llama-bench}"
SERVER_BIN="${SERVER_BIN:-llama-server}"
PORT="${PORT:-8095}"   # 8090 = the LAN ntfy server, which also answers /health
FIT_TARGET="${FIT_TARGET:-512}"       # MiB margin --fit leaves free on the GPU
DRY_RUN="${DRY_RUN:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }

help() {
	cat <<EOF
Usage: ${0##*/} [-b|-f] [-h]
  -b   llama-bench sweep: for each STEP (NCMOE+offset) and each HALF, run
       pp512/tg128 with --n-cpu-moe and an --override-tensor for one more layer
  -f   llama-server --fit probe at CTX: prints the placement --fit chose
  none: -b
Environment (current values):
  MODEL       GGUF                        ($MODEL)
  NCMOE       whole-layer base            ($NCMOE)
  STEPS       offsets from NCMOE          ($STEPS)
  HALF_LIST   extra tensors of layer base ($HALF_LIST: none|down|up+gate|up|gate)
  CTX         server context for -f       ($CTX)
  FIT_TARGET  --fit-target MiB            ($FIT_TARGET)
  THREADS / CACHE_TYPE / FA_FLAG / BENCH_BIN / SERVER_BIN / PORT
  DRY_RUN=1   print commands only
Read the result as: the lowest (base, half) that does not OOM is the new
floor; compare tg against the whole-layer rows to see what the half step costs.
EOF
	exit 0
}

port_free() {
	# ntfy (:8090) and other daemons answer /health with 200 — a bench that
	# "starts" on a taken port measures the wrong process. Refuse up front.
	if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then
		die "port $PORT is already in use (ss -ltnp | grep :$PORT); set PORT=..."
	fi
}

vram_check() {
	local used
	used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1) || return 0
	[[ -n "$used" && "$used" -gt 3072 ]] &&
		echo "WARNING: ${used} MiB VRAM already in use — OOM floor and numbers will be off" >&2
	return 0
}

# --override-tensor regex for "layer L, tensors HALF → CPU"
ot_for() {
	local layer=$1 half=$2
	case "$half" in
		none)    echo "" ;;
		down)    echo "blk\\.${layer}\\.ffn_down_exps=CPU" ;;
		up+gate) echo "blk\\.${layer}\\.ffn_(up|gate)_exps=CPU" ;;
		up)      echo "blk\\.${layer}\\.ffn_up_exps=CPU" ;;
		gate)    echo "blk\\.${layer}\\.ffn_gate_exps=CPU" ;;
		*) die "unknown HALF value: $half" ;;
	esac
}

bench_sweep() {
	command -v "$BENCH_BIN" >/dev/null || die "binary not found: $BENCH_BIN"
	local off base half ot fails=0 total=0
	for off in ${STEPS//,/ }; do
		base=$((NCMOE + off))
		for half in ${HALF_LIST//,/ }; do
			ot=$(ot_for "$base" "$half")
			((total++)) || true
			echo "=== --n-cpu-moe $base  layer $base extra→CPU: $half ${ot:+(-ot $ot)} ==="
			local cmd=("$BENCH_BIN" -m "$MODEL" -ngl 99 -ncmoe "$base" -fa "$FA_FLAG" -t "$THREADS"
				-ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE")
			[[ -n "$ot" ]] && cmd+=(-ot "$ot")
			if [[ "$DRY_RUN" == 1 ]]; then printf '  %q ' "${cmd[@]}"; echo; continue; fi
			# one OOM must not end the sweep: the rows below the floor are the point
			if ! "${cmd[@]}" 2>/dev/null | grep -E '^\|' | column -t -s'|' -o'|'; then
				echo "  FAILED (OOM?) — continuing" >&2; ((fails++)) || true
			fi
		done
	done
	[[ "$DRY_RUN" == 1 || $fails -lt $total ]] || die "every configuration failed — check nvidia-smi"
}

fit_probe() {
	command -v "$SERVER_BIN" >/dev/null || die "binary not found: $SERVER_BIN"
	local logfile spid i
	local cmd=("$SERVER_BIN" -m "$MODEL" -c "$CTX" -fa "$FA_FLAG" -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE"
		-t "$THREADS" --port "$PORT" -np 1 --fit on --fit-target "$FIT_TARGET" --verbose)
	echo "=== --fit on, ctx $CTX, target ${FIT_TARGET} MiB (no -ngl / --n-cpu-moe given: --fit decides) ==="
	if [[ "$DRY_RUN" == 1 ]]; then printf '  %q ' "${cmd[@]}"; echo; return; fi
	logfile=$(mktemp /tmp/bench09-fit.XXXXXX.log)
	"${cmd[@]}" >"$logfile" 2>&1 &
	spid=$!
	trap "kill $spid 2>/dev/null || true" EXIT
	for i in $(seq 1 60); do
		curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
		kill -0 "$spid" 2>/dev/null || { grep -iE 'fit|cpu_moe|n_gpu_layers|override|error' "$logfile" | tail -20; die "server died — log: $logfile"; }
		sleep 5
	done
	echo "--- what --fit chose (from the server log):"
	grep -iE 'fit|n_cpu_moe|n_gpu_layers|override|CPU buffer|CUDA0 model buffer|KV self size' "$logfile" | head -30
	echo "--- VRAM now:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
	kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true
	trap - EXIT
	echo "(full log kept: $logfile)"
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && help
[[ -f "$MODEL" ]] || die "no such GGUF: $MODEL"
[[ "$DRY_RUN" == 1 ]] || { port_free; vram_check; }
case "${1:-}" in
	-b|"") bench_sweep ;;
	-f)    fit_probe ;;
	*)     die "unknown option: $1 (use -h)" ;;
esac
