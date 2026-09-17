#!/bin/bash
# bench_08_spec_offload.sh — speculative decoding × MoE offload: sweep the
# draft length on a llama-server and read tg + acceptance from its timings.
#
# Why (BENCH.md → Literature, SpecMoEOff arXiv:2508.21706): with experts on
# the CPU the GPU idles while expert weights stream over PCIe, so speculative
# decoding pays off MORE on an offloaded MoE than on a dense model that fits.
# The paper's throughput peaks at 5–6 draft tokens and falls beyond that;
# llama-server's default is --spec-draft-n-max 3. This script measures where
# OUR peak is, per model, instead of trusting either number.
#
# Runs on any GGUF that llama-server can draft for: MTP builds
# (`--spec-type draft-mtp`, e.g. Qwythos-9B-v2-MTP now, Ornith-MTP /
# qwen36-MTP once downloaded), a separate draft model (`SPEC_TYPE=draft-simple
# DRAFT_MODEL=...`), or the model-free n-gram drafters (`SPEC_TYPE=ngram-mod`)
# which need nothing extra and are the cheapest thing to try first.
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL="${MODEL:-$MODEL_ROOT/qwythos-9b-v2/Qwythos-9B-v2-MTP-Q8_0.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-}"            # only for SPEC_TYPE=draft-simple/eagle3
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"       # none | draft-mtp | draft-simple | ngram-mod | ...
DRAFT_LIST="${DRAFT_LIST:-0,3,4,5,6,8}"   # 0 = --spec-type none (baseline)
NCMOE="${NCMOE:-0}"                       # experts on CPU for the first N layers
CTX="${CTX:-32768}"
THREADS="${THREADS:-16}"
PORT="${PORT:-8095}"   # 8090 = the LAN ntfy server, which also answers /health
NPREDICT="${NPREDICT:-384}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
FA_FLAG="${FA_FLAG:-on}"
SERVER_BIN="${SERVER_BIN:-llama-server}"
PROMPT_FILE="${PROMPT_FILE:-}"            # a real prompt from your workload beats the built-in one
REPEATS="${REPEATS:-2}"                   # measurements per draft length (page cache, noise)
DRY_RUN="${DRY_RUN:-0}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

die() { echo "ERROR: $*" >&2; exit 1; }

help() {
	cat <<EOF
Usage: ${0##*/} [-h]
Starts llama-server once per draft length (DRAFT_LIST), sends the same prompt
REPEATS times and prints tg tok/s plus draft acceptance for each.
Environment (current values):
  MODEL          GGUF                     ($MODEL)
  SPEC_TYPE      drafter                  ($SPEC_TYPE)
  DRAFT_MODEL    draft GGUF (draft-simple) ($DRAFT_MODEL)
  DRAFT_LIST     --spec-draft-n-max sweep ($DRAFT_LIST; 0 = no speculation;
                 for ngram-* types the value is a label only, use 0,1)
  NCMOE          --n-cpu-moe              ($NCMOE)
  CTX / THREADS / PORT / NPREDICT / CACHE_TYPE / FA_FLAG
  PROMPT_FILE    prompt text file         (${PROMPT_FILE:-built-in coding prompt})
  REPEATS        runs per setting         ($REPEATS)
  DRY_RUN=1      print the server commands only
Acceptance is read from the server's timings (draft_n / draft_n_accepted);
a build without those fields prints tg only.
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
		echo "WARNING: ${used} MiB VRAM already in use (aillama/aildr in the background?) — results will be skewed" >&2
	return 0
}

build_prompt() {
	if [[ -n "$PROMPT_FILE" ]]; then cat "$PROMPT_FILE"; return; fi
	cat <<'EOF'
Write a Python module that parses a systemd journal export (JSON lines) and
aggregates messages per unit and priority into a table, with a CLI that accepts
--since/--until ISO timestamps, a --top N option, and unit tests using pytest
with fixtures for at least three edge cases. Explain the design briefly first.
EOF
}

# Fills the global SARGS array. An array, not a string: the first version
# built a string and ran it through `eval ... &`, which backgrounds a subshell
# — `$!` was the subshell, `kill $!` killed the subshell and the llama-server
# it had spawned lived on, holding 10 GB of VRAM after the sweep had ended.
server_args() {
	local k=$1
	SARGS=(-m "$MODEL" -ngl 99 -c "$CTX" -fa "$FA_FLAG" -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE"
		-t "$THREADS" --port "$PORT" -np 1)
	[[ "$NCMOE" -gt 0 ]] && SARGS+=(--n-cpu-moe "$NCMOE")
	if [[ "$k" == "0" ]]; then
		SARGS+=(--spec-type none)
	elif [[ "$SPEC_TYPE" == ngram-* ]]; then
		# the n-gram drafters have their own knobs (--spec-ngram-mod-n-max,
		# --spec-ngram-simple-size-m, ...); --spec-draft-n-max is not one of
		# them, so k is only a row label here — DRAFT_LIST=0,1 is enough
		SARGS+=(--spec-type "$SPEC_TYPE")
	else
		SARGS+=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$k")
		# a separate draft GGUF: draft-simple/eagle3 drafters, or the ggml-org
		# style MTP head shipped as its own file (nemotron: mtp-*.gguf, 19 tensors)
		[[ -n "$DRAFT_MODEL" ]] && SARGS+=(-md "$DRAFT_MODEL" -ngld 99)
	fi
	# shellcheck disable=SC2206
	[[ -n "$EXTRA_SERVER_ARGS" ]] && SARGS+=($EXTRA_SERVER_ARGS)
	return 0
}

wait_health() {
	local spid=$1 logfile=$2 i
	for i in $(seq 1 60); do
		curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
		kill -0 "$spid" 2>/dev/null || die "server died during start — see $logfile"
		sleep 5
	done
	die "server not up after 300 s — $logfile"
}

measure() {
	local prompt=$1 resp
	resp=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
		-d "{\"prompt\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$prompt"),\"n_predict\":$NPREDICT,\"cache_prompt\":false,\"temperature\":0}") \
		|| die "server died during the request (OOM at first decode?)"
	python3 - "$resp" <<-'PY'
		import json, sys
		d = json.loads(sys.argv[1]); t = d.get("timings") or {}
		if not t: sys.exit("no timings in the answer: " + json.dumps(d)[:300])
		tg = t.get("predicted_per_second", 0); pp = t.get("prompt_per_second", 0)
		dn = t.get("draft_n"); da = t.get("draft_n_accepted")
		acc = f"  accept={da}/{dn} ({100*da/dn:.0f}%)" if dn else ""
		print(f"pp={pp:.0f} tok/s  tg={tg:.1f} tok/s  gen_n={t.get('predicted_n')}{acc}")
	PY
}

run_one() {
	local k=$1 logfile spid prompt r
	server_args "$k"
	echo "=== draft-n-max=$k  (ncmoe=$NCMOE, spec=$([[ $k == 0 ]] && echo none || echo "$SPEC_TYPE")) ==="
	if [[ "$DRY_RUN" == 1 ]]; then printf '  %q ' "$SERVER_BIN" "${SARGS[@]}"; echo; return; fi
	# A server left over from a previous setting would answer /health for this
	# one and every row would measure the same process (that is exactly how
	# the first version of this script produced five identical rows).
	port_free
	logfile=$(mktemp /tmp/bench08.XXXXXX.log)
	"$SERVER_BIN" "${SARGS[@]}" >"$logfile" 2>&1 &
	spid=$!
	trap "kill $spid 2>/dev/null || true" EXIT
	wait_health "$spid" "$logfile"
	prompt=$(build_prompt)
	for r in $(seq 1 "$REPEATS"); do
		printf '  run %d: ' "$r"; measure "$prompt"
	done
	kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true
	trap - EXIT
	pgrep -f "port $PORT" >/dev/null && echo "WARNING: a llama-server on :$PORT survived the kill — check pgrep -a llama-server" >&2
	if [[ "$k" != 0 ]]; then
		echo "  server log, speculative lines:"; grep -iE 'spec|draft' "$logfile" | grep -viE 'warning: option' | head -4 | sed 's/^/    /'
	fi
	rm -f "$logfile"
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && help
[[ -f "$MODEL" ]] || die "no such GGUF: $MODEL"
command -v "$SERVER_BIN" >/dev/null || die "binary not found: $SERVER_BIN"
[[ "$DRY_RUN" == 1 ]] || { port_free; vram_check; }
for k in ${DRAFT_LIST//,/ }; do run_one "$k"; done
