#!/bin/bash
# bench_04_qwythos.sh — benchmark Qwythos-9B-v2 (dense hybrid Gated-DeltaNet, arch qwen35)
# Part of the bench_NN_* series: 01=llama.cpp MoE sweep, 02=ik_llama.cpp, 03=ollama.
# Qwythos is dense (no MoE), so instead of an --n-cpu-moe sweep this measures:
#   1. pp/tg vs context depth (linear attention should keep pp near-flat)
#   2. a real llama-server request at long context
#   3. MTP speculative decoding on/off (native MTP head, --spec-type draft-mtp)
# Description: BENCH.md
set -euo pipefail

HF_HOME="${HF_HOME:-/mnt/db1/huggingface}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL="${MODEL:-$MODEL_ROOT/qwythos-9b-v2/Qwythos-9B-v2-Q8_0.gguf}"
MTP_MODEL="${MTP_MODEL:-$MODEL_ROOT/qwythos-9b-v2/Qwythos-9B-v2-MTP-Q8_0.gguf}"
export HF_HOME
DEPTH_LIST="${DEPTH_LIST:-0,16384,65536,131072}"
THREADS="${THREADS:-16}"
CTX="${CTX:-131072}"
PORT="${PORT:-8090}"
NPREDICT="${NPREDICT:-256}"
PROMPT_REPEATS="${PROMPT_REPEATS:-60}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
BENCH_BIN="${BENCH_BIN:-llama-bench}"
SERVER_BIN="${SERVER_BIN:-llama-server}"
FA_FLAG="${FA_FLAG:-on}"
REPS="${REPS:-2}"

die() { echo "ERROR: $*" >&2; exit 1; }

pomoc() {
	cat <<EOF
Usage: ${0##*/} [-b|-s|-m] [-h]

Benchmark Qwythos-9B-v2 (9B dense hybrid, fully on GPU, no MoE offload):
depth sweep with llama-bench (pp512/tg128 at several context depths),
a real llama-server request, and an MTP speculative-decoding comparison.

Options:
  -b    depth sweep only (llama-bench -d $DEPTH_LIST)
  -s    server test only (real prompt, ctx $CTX)
  -m    MTP comparison only (baseline vs --spec-type draft-mtp)
  -h    this help
  no option: all three tests

Environment variables (current values):
  MODEL          main GGUF                 ($MODEL)
  MTP_MODEL      GGUF with MTP head        ($MTP_MODEL)
  MODEL_ROOT     GGUF model root           ($MODEL_ROOT)
  HF_HOME        Hugging Face cache        ($HF_HOME)
  DEPTH_LIST     llama-bench -d sweep      ($DEPTH_LIST)
  REPS           llama-bench repetitions   ($REPS)
  THREADS        CPU threads               ($THREADS)
  CTX            server context            ($CTX)
  PORT           server port               ($PORT)
  NPREDICT       output tokens             ($NPREDICT)
  PROMPT_REPEATS prompt sentence repeats   ($PROMPT_REPEATS)
  CACHE_TYPE     KV cache type             ($CACHE_TYPE)
  BENCH_BIN      bench binary              ($BENCH_BIN)
  SERVER_BIN     server binary             ($SERVER_BIN)
EOF
	exit 0
}

build_prompt() {
	local base="Analyze the following bash script and point out the bugs. " p="" i
	for ((i = 0; i < PROMPT_REPEATS; i++)); do p+="$base"; done
	p+="Write a bash function that safely mounts a LUKS device with UUID validation and error handling. Explain every step."
	printf '%s' "$p"
}

bench_depth() {
	command -v "$BENCH_BIN" >/dev/null || die "binary not found: $BENCH_BIN"
	echo "=== $BENCH_BIN: pp512/tg128 at depths $DEPTH_LIST (r=$REPS) ==="
	"$BENCH_BIN" -m "$MODEL" -ngl 99 -fa "$FA_FLAG" -t "$THREADS" \
		-p 512 -n 128 -d "$DEPTH_LIST" -r "$REPS" \
		2>/dev/null | grep -E '^\|' || die "llama-bench failed (OOM at high depth?)"
}

# run_server_query MODEL_PATH LABEL [extra server args...]
# Starts the server, sends the benchmark prompt, prints pp/tg, stops the server.
run_server_query() {
	command -v "$SERVER_BIN" >/dev/null || die "binary not found: $SERVER_BIN"
	local model="$1" label="$2"; shift 2
	local logfile spid i prompt resp
	logfile=$(mktemp /tmp/bench-qwythos.XXXXXX.log)
	echo "=== $SERVER_BIN [$label]: ctx $CTX, KV $CACHE_TYPE, args: ${*:-none} (log: $logfile) ==="
	"$SERVER_BIN" -m "$model" -ngl 99 -c "$CTX" \
		-fa "$FA_FLAG" -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE" -t "$THREADS" \
		--port "$PORT" "$@" >"$logfile" 2>&1 &
	spid=$!
	# value expanded now — local spid does not exist in the EXIT trap context
	trap "kill $spid 2>/dev/null || true" EXIT

	for i in $(seq 1 60); do
		curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
		kill -0 "$spid" 2>/dev/null || die "server died during startup — check $logfile"
		sleep 5
	done
	curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || die "server not up after 300 s — $logfile"

	prompt=$(build_prompt)
	# without the || die, a server crash mid-request (e.g. VRAM OOM on first
	# decode) kills the script silently via set -e on the assignment
	resp=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
		-d "{\"prompt\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$prompt"),\"n_predict\":$NPREDICT,\"cache_prompt\":false}") \
		|| die "server died during the request (VRAM OOM on decode?) — check $logfile"
	python3 - "$resp" "$label" <<-'PY'
		import json, sys
		d = json.loads(sys.argv[1])
		t = d.get("timings") or {}
		if not t:
		    sys.exit("no timings field in server response: " + json.dumps(d)[:300])
		label = sys.argv[2]
		print(f"[{label}] prompt_n={t.get('prompt_n')}  pp={t.get('prompt_per_second', 0):.1f} tok/s")
		print(f"[{label}] gen_n={t.get('predicted_n')}  tg={t.get('predicted_per_second', 0):.1f} tok/s")
	PY

	kill "$spid" 2>/dev/null || true
	wait "$spid" 2>/dev/null || true
	trap - EXIT
}

server_test() {
	run_server_query "$MODEL" "baseline"
}

mtp_test() {
	[[ -f "$MTP_MODEL" ]] || die "MTP GGUF not found: $MTP_MODEL"
	echo "=== MTP speculative decoding: baseline vs draft-mtp (same prompt) ==="
	run_server_query "$MODEL" "no-mtp"
	run_server_query "$MTP_MODEL" "mtp" --spec-type draft-mtp
	echo "Compare the two tg values above; MTP pays off when acceptance rate is high (see server log)."
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && pomoc
[[ -f "$MODEL" ]] || die "GGUF model not found: $MODEL"

case "${1:-}" in
	-b) bench_depth ;;
	-s) server_test ;;
	-m) mtp_test ;;
	"") bench_depth; server_test; mtp_test ;;
	*)  die "unknown option: $1 (use -h)" ;;
esac
