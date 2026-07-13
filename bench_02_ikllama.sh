#!/bin/bash
# bench_02_ikllama.sh — benchmark ik_llama.cpp (fork ikawrakow) na modelu GGUF (MoE offload)
# Część serii bench_NN_*: 01=llama.cpp, 02=ik_llama.cpp, 03=ollama. Opis: BENCH.md
set -euo pipefail

HF_HOME="${HF_HOME:-/mnt/db1/huggingface}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL="${MODEL:-$MODEL_ROOT/glm-4.7-flash/GLM-4.7-Flash-UD-Q4_K_XL.gguf}"
export HF_HOME
# Descending: probe the floor from a known-good high value downward — every
# run measures something; an ascending list burns runs on OOM crashes first.
NCMOE_LIST="${NCMOE_LIST:-14,10,8}"
SERVER_NCMOE="${SERVER_NCMOE:-18}"
THREADS="${THREADS:-16}"
CTX="${CTX:-32768}"
PORT="${PORT:-8091}"
NPREDICT="${NPREDICT:-256}"
PROMPT_REPEATS="${PROMPT_REPEATS:-60}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
BENCH_BIN="${BENCH_BIN:-ik-llama-bench}"
SERVER_BIN="${SERVER_BIN:-ik-llama-server}"
FA_FLAG="${FA_FLAG:-1}"

die() { echo "BŁĄD: $*" >&2; exit 1; }

pomoc() {
	cat <<EOF
Użycie: ${0##*/} [-b|-s] [-h]

Benchmark ik_llama.cpp: sweep ik-llama-bench (pp512/tg128) po wartościach
--n-cpu-moe oraz realne zapytanie do ik-llama-server z pomiarem timings.

Opcje:
  -b    tylko sweep ik-llama-bench
  -s    tylko test serwera (realny prompt ~PROMPT_REPEATS*20 tokenów)
  -h    ta pomoc
  bez opcji: oba testy

Zmienne środowiskowe (aktualne wartości):
  MODEL          ścieżka GGUF            ($MODEL)
  MODEL_ROOT     katalog modeli GGUF      ($MODEL_ROOT)
  HF_HOME        cache Hugging Face        ($HF_HOME)
  NCMOE_LIST     sweep --n-cpu-moe       ($NCMOE_LIST)
  SERVER_NCMOE   --n-cpu-moe serwera     ($SERVER_NCMOE)
  THREADS        wątki CPU               ($THREADS)
  CTX            kontekst serwera        ($CTX)
  PORT           port serwera            ($PORT)
  NPREDICT       tokeny wyjściowe        ($NPREDICT)
  PROMPT_REPEATS powtórzenia zdania      ($PROMPT_REPEATS)
  CACHE_TYPE     typ KV cache            ($CACHE_TYPE)
  BENCH_BIN      binarka bench           ($BENCH_BIN)
  SERVER_BIN     binarka server          ($SERVER_BIN)
EOF
	exit 0
}

build_prompt() {
	local base="Przeanalizuj poniższy skrypt bash i wskaż błędy. " p="" i
	for ((i = 0; i < PROMPT_REPEATS; i++)); do p+="$base"; done
	p+="Napisz funkcję w bash, która bezpiecznie montuje urządzenie LUKS z walidacją UUID i obsługą błędów. Wyjaśnij każdy krok."
	printf '%s' "$p"
}

bench_sweep() {
	command -v "$BENCH_BIN" >/dev/null || die "brak binarki: $BENCH_BIN"
	local n fails=0 total=0
	echo "=== $BENCH_BIN: sweep --n-cpu-moe ($NCMOE_LIST) ==="
	# OOM at one value (below the floor) must not kill the sweep — the
	# remaining workable values would go untested. Log per value, die only
	# when the whole sweep failed.
	for n in ${NCMOE_LIST//,/ }; do
		((total++)) || true
		if ! "$BENCH_BIN" -m "$MODEL" -ngl 99 --n-cpu-moe "$n" -fa "$FA_FLAG" -t "$THREADS" \
			2>/dev/null | grep -E '^\|'; then
			echo "UWAGA: bench --n-cpu-moe $n nie powiódł się (OOM?) — kontynuuję sweep" >&2
			((fails++)) || true
		fi
	done
	[[ $fails -lt $total ]] || die "wszystkie wartości sweep zawiodły (OOM? sprawdź VRAM: nvidia-smi)"
}

server_test() {
	command -v "$SERVER_BIN" >/dev/null || die "brak binarki: $SERVER_BIN"
	local logfile spid i prompt resp
	logfile=$(mktemp /tmp/bench-server.XXXXXX.log)
	echo "=== $SERVER_BIN: --n-cpu-moe $SERVER_NCMOE, ctx $CTX, KV $CACHE_TYPE (log: $logfile) ==="
	# shellcheck disable=SC2086
	"$SERVER_BIN" -m "$MODEL" -ngl 99 --n-cpu-moe "$SERVER_NCMOE" -c "$CTX" \
		-fa "$FA_FLAG" -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE" -t "$THREADS" \
		--port "$PORT" ${EXTRA_SERVER_ARGS:-} >"$logfile" 2>&1 &
	spid=$!
	# wartość rozwinięta od razu — local spid nie istnieje w kontekście trap EXIT
	trap "kill $spid 2>/dev/null || true" EXIT

	for i in $(seq 1 60); do
		curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
		kill -0 "$spid" 2>/dev/null || die "serwer padł podczas startu — sprawdź $logfile"
		sleep 5
	done
	curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || die "serwer nie wstał w 300 s — $logfile"

	prompt=$(build_prompt)
	# without the || die, a server crash mid-request (e.g. VRAM OOM on first
	# decode) kills the script silently via set -e on the assignment
	resp=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
		-d "{\"prompt\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$prompt"),\"n_predict\":$NPREDICT,\"cache_prompt\":false}") \
		|| die "serwer padł podczas zapytania (OOM przy dekodzie?) — sprawdź $logfile"
	python3 - "$resp" <<-'PY'
		import json, sys
		d = json.loads(sys.argv[1])
		t = d.get("timings") or {}
		if not t:
		    sys.exit("brak pola timings w odpowiedzi serwera: " + json.dumps(d)[:300])
		print(f"prompt_n={t.get('prompt_n')}  pp={t.get('prompt_per_second', 0):.1f} tok/s")
		print(f"gen_n={t.get('predicted_n')}  tg={t.get('predicted_per_second', 0):.1f} tok/s")
	PY

	kill "$spid" 2>/dev/null || true
	wait "$spid" 2>/dev/null || true
	trap - EXIT
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && pomoc
[[ -f "$MODEL" ]] || die "brak modelu GGUF: $MODEL"

case "${1:-}" in
	-b) bench_sweep ;;
	-s) server_test ;;
	"") bench_sweep; server_test ;;
	*)  die "nieznana opcja: $1 (użyj -h)" ;;
esac
