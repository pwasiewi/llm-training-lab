#!/bin/bash
# bench_03_ollama.sh — benchmark ollamy (dodatek porównawczy do bench_01/bench_02)
# Część serii bench_NN_*: 01=llama.cpp, 02=ik_llama.cpp, 03=ollama. Opis: BENCH.md
set -euo pipefail

MODEL="${MODEL:-$HOME/models/glm-4.7-flash/GLM-4.7-Flash-UD-Q4_K_XL.gguf}"
OLLAMA_MODEL="${OLLAMA_MODEL:-glm47-flash-q4}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_MODELS="${OLLAMA_MODELS:-/mnt/db1/ollama/models}"
NPREDICT="${NPREDICT:-256}"
PROMPT_REPEATS="${PROMPT_REPEATS:-60}"
export OLLAMA_MODELS

die() { echo "BŁĄD: $*" >&2; exit 1; }

pomoc() {
	cat <<EOF
Użycie: ${0##*/} [-k] [-h]

Benchmark ollamy na tym samym GGUF co bench_01/bench_02 (import przez
Modelfile, jeśli modelu nie ma). Mierzy dwa zapytania /api/generate:
zimne (z load) i rozgrzane. Ollama sama dobiera split CPU/GPU po całych
warstwach — brak odpowiednika --n-cpu-moe; to właśnie mierzy ten test.

Opcje:
  -k    zostaw daemon ollamy uruchomiony po teście (domyślnie: zatrzymywany,
        jeśli to skrypt go wystartował)
  -h    ta pomoc

Zmienne środowiskowe (aktualne wartości):
  MODEL          ścieżka GGUF do importu   ($MODEL)
  OLLAMA_MODEL   nazwa modelu w ollamie    ($OLLAMA_MODEL)
  OLLAMA_URL     adres API                 ($OLLAMA_URL)
  OLLAMA_MODELS  katalog modeli ollamy      ($OLLAMA_MODELS)
  NPREDICT       tokeny wyjściowe          ($NPREDICT)
  PROMPT_REPEATS powtórzenia zdania        ($PROMPT_REPEATS)

Uwaga: import tworzy kopię bloba (~rozmiar GGUF) w OLLAMA_MODELS.
Usunięcie po testach: ollama rm $OLLAMA_MODEL
EOF
	exit 0
}

build_prompt() {
	local base="Przeanalizuj poniższy skrypt bash i wskaż błędy. " p="" i
	for ((i = 0; i < PROMPT_REPEATS; i++)); do p+="$base"; done
	p+="Napisz funkcję w bash, która bezpiecznie montuje urządzenie LUKS z walidacją UUID i obsługą błędów. Wyjaśnij każdy krok."
	printf '%s' "$p"
}

run_query() {
	local label="$1" suffix="${2:-}" resp
	resp=$(curl -s "$OLLAMA_URL/api/generate" \
		-d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$(build_prompt)$suffix"),\"stream\":false,\"options\":{\"num_predict\":$NPREDICT}}")
	python3 -c '
import json, sys
d = json.loads(sys.argv[1])
if "error" in d:
    sys.exit("ollama: " + d["error"])
pe, pd = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 1)
e, ed = d.get("eval_count", 0), d.get("eval_duration", 1)
load = d.get("load_duration", 0) / 1e9
print(f"[{sys.argv[2]}] prompt_n={pe}  pp={pe/(pd/1e9):.1f} tok/s   "
      f"gen_n={e}  tg={e/(ed/1e9):.1f} tok/s   (load {load:.1f} s)")
' "$resp" "$label"
}

KEEP_DAEMON=0
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && pomoc
[[ "${1:-}" == "-k" ]] && KEEP_DAEMON=1

command -v ollama >/dev/null || die "brak binarki ollama"

started_daemon=0
daemon_pid=""
if ! curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
	echo "* daemon nie działa — startuję ollama serve w tle"
	nohup ollama serve >/tmp/bench03-ollama.log 2>&1 &
	daemon_pid=$!
	started_daemon=1
	for i in $(seq 1 12); do
		curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1 && break
		sleep 5
	done
	curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1 \
		|| die "daemon ollamy nie wstał — /tmp/bench03-ollama.log"
fi

if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qE "^$OLLAMA_MODEL(:latest)?\$"; then
	[[ -f "$MODEL" ]] || die "brak GGUF do importu: $MODEL"
	echo "* importuję $MODEL jako $OLLAMA_MODEL (kopia bloba ~rozmiar GGUF)"
	mf=$(mktemp /tmp/bench03-modelfile.XXXXXX)
	printf 'FROM %s\n' "$MODEL" >"$mf"
	ollama create "$OLLAMA_MODEL" -f "$mf" || { rm -f "$mf"; die "ollama create nie powiódł się"; }
	rm -f "$mf"
fi

echo "=== ollama: $OLLAMA_MODEL, num_predict=$NPREDICT ==="
run_query "zimny  "
run_query "rozgrzany" " Podaj też wariant z gocryptfs."   # inny sufiks = bez cache promptu
echo "--- split CPU/GPU wg ollamy: ---"
ollama ps

if (( started_daemon )) && (( ! KEEP_DAEMON )); then
	echo "* zatrzymuję daemon ollamy (uruchomiony przez skrypt; -k aby zostawić)"
	kill "$daemon_pid" 2>/dev/null || true
fi
