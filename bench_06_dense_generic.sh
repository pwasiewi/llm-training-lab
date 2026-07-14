#!/bin/bash
# bench_06_dense_generic.sh — generic bench for dense (non-MoE) GGUF models
# Part of the bench_NN_* series: 01=llama.cpp MoE sweep, 02=ik_llama.cpp, 03=ollama,
# 04=Qwythos-9B-v2 specifically. This is bench_04 generalized to any dense model
# (no hardcoded defaults) for the 27B/12B/9B candidate round on 2026-07-14.
# Same three measurements: pp/tg vs context depth, a real server request,
# and an MTP speculative-decoding comparison (only if MTP_MODEL is set).
# Description: BENCH.md
set -euo pipefail

HF_HOME="${HF_HOME:-/mnt/db1/huggingface}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL="${MODEL:-}"
MTP_MODEL="${MTP_MODEL:-}"
export HF_HOME
DEPTH_LIST="${DEPTH_LIST:-0,16384,65536,131072}"
THREADS="${THREADS:-16}"
CTX="${CTX:-65536}"
PORT="${PORT:-8090}"
NPREDICT="${NPREDICT:-256}"
PROMPT_REPEATS="${PROMPT_REPEATS:-60}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
BENCH_BIN="${BENCH_BIN:-llama-bench}"
SERVER_BIN="${SERVER_BIN:-llama-server}"
FA_FLAG="${FA_FLAG:-on}"
REPS="${REPS:-2}"
NGL="${NGL:-99}"

die() { echo "ERROR: $*" >&2; exit 1; }

pomoc() {
	cat <<EOF
Usage: MODEL=/path/to.gguf ${0##*/} [-b|-s|-m] [-h]

Generic dense-model bench (no --n-cpu-moe sweep — model fits GPU or it doesn't):
per-depth llama-bench sweep (pp512/tg128, each depth run separately so one
OOM doesn't kill the whole sweep), a real llama-server request, and an
optional MTP speculative-decoding comparison (needs MTP_MODEL set).

Options:
  -b    depth sweep only (llama-bench -d $DEPTH_LIST, one run per value)
  -s    server test only (real prompt, ctx $CTX)
  -m    MTP comparison only (baseline vs --spec-type draft-mtp; requires MTP_MODEL)
  -h    this help
  no option: all tests (skips -m if MTP_MODEL unset)

Environment variables (current values):
  MODEL          main GGUF, required        ($MODEL)
  MTP_MODEL      GGUF with MTP head, optional ($MTP_MODEL)
  MODEL_ROOT     GGUF model root           ($MODEL_ROOT)
  HF_HOME        Hugging Face cache        ($HF_HOME)
  NGL            -ngl (layers on GPU)      ($NGL)
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

# per-depth loop: one OOM must not lose the rows for the depths that work
bench_depth() {
	command -v "$BENCH_BIN" >/dev/null || die "binary not found: $BENCH_BIN"
	echo "=== $BENCH_BIN: pp512/tg128 at depths $DEPTH_LIST (ngl=$NGL, r=$REPS) ==="
	local d out
	IFS=',' read -ra depths <<<"$DEPTH_LIST"
	for d in "${depths[@]}"; do
		out=$("$BENCH_BIN" -m "$MODEL" -ngl "$NGL" -fa "$FA_FLAG" -t "$THREADS" \
			-p 512 -n 128 -d "$d" -r "$REPS" 2>/dev/null | grep -E '^\|') \
			&& printf '%s\n' "$out" \
			|| echo "  depth=$d: FAILED (OOM?) — skipped"
	done
}

# run_server_query MODEL_PATH LABEL [extra server args...]
run_server_query() {
	command -v "$SERVER_BIN" >/dev/null || die "binary not found: $SERVER_BIN"
	local model="$1" label="$2"; shift 2
	local logfile spid i prompt resp
	logfile=$(mktemp /tmp/bench-dense.XXXXXX.log)
	echo "=== $SERVER_BIN [$label]: ctx $CTX, ngl $NGL, KV $CACHE_TYPE, args: ${*:-none} (log: $logfile) ==="
	"$SERVER_BIN" -m "$model" -ngl "$NGL" -c "$CTX" \
		-fa "$FA_FLAG" -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE" -t "$THREADS" \
		--port "$PORT" "$@" >"$logfile" 2>&1 &
	spid=$!
	trap "kill $spid 2>/dev/null || true" EXIT

	for i in $(seq 1 60); do
		curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
		kill -0 "$spid" 2>/dev/null || { echo "  [$label] server died during startup — check $logfile"; trap - EXIT; return 1; }
		sleep 5
	done
	curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "  [$label] server not up after 300s — $logfile"; kill "$spid" 2>/dev/null || true; trap - EXIT; return 1; }

	prompt=$(build_prompt)
	resp=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
		-d "{\"prompt\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$prompt"),\"n_predict\":$NPREDICT,\"cache_prompt\":false}") \
		|| { echo "  [$label] server died during request (VRAM OOM on decode?) — $logfile"; kill "$spid" 2>/dev/null || true; trap - EXIT; return 1; }
	python3 - "$resp" "$label" <<-'PY' || echo "  [$label] bad response, see $logfile"
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
	[[ -n "$MTP_MODEL" ]] || { echo "MTP_MODEL not set — skipping MTP comparison"; return 0; }
	[[ -f "$MTP_MODEL" ]] || die "MTP GGUF not found: $MTP_MODEL"
	echo "=== MTP speculative decoding: baseline vs draft-mtp (same prompt) ==="
	run_server_query "$MODEL" "no-mtp"
	run_server_query "$MTP_MODEL" "mtp" --spec-type draft-mtp
	echo "Compare the two tg values above; MTP pays off when acceptance rate is high (see server log)."
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && pomoc
[[ -n "$MODEL" ]] || die "MODEL not set (export MODEL=/path/to.gguf)"
[[ -f "$MODEL" ]] || die "GGUF model not found: $MODEL"

case "${1:-}" in
	-b) bench_depth ;;
	-s) server_test ;;
	-m) mtp_test ;;
	"") bench_depth; server_test; mtp_test ;;
	*)  die "unknown option: $1 (use -h)" ;;
esac
