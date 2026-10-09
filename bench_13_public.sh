#!/bin/bash
# bench_13_public.sh — fleet-internal runners for two PUBLIC benchmarks:
#   swe   — SWE-rebench (Nebius) fresh Python tasks, agent = mini-swe-agent
#           (bash tool calls, litellm openai/<profile>), verdict = the official
#           SWE-rebench harness fork (FAIL_TO_PASS + PASS_TO_PASS in the task's
#           prebuilt swerebench/* image)
#   aider — Aider Polyglot hard subset (6 languages), aider's own
#           benchmark/benchmark.py inside the aider-benchmark image, edit
#           format "whole", 2 tries -> pass_1 / pass_2
# Purpose: rank models INSIDE our fleet (night3-plan row 24, research/22 + 24),
# not to reproduce the public leaderboards. Task lists + the reason each was
# picked: bench13/swe_screen.tsv, bench13/swe_core.tsv (made by
# bench13/select_core.py from a screen run), bench13/aider_hard.tsv.
#
# Output follows bench_05: one "=== [model/task] VERDICT (Ns) [score] — note"
# line per task-run, a per-model summary table, then a PAIRED per-task table
# across models (same tasks, same order) with discordant tasks marked.
#   SWE score   = FAIL_TO_PASS tests passing (k/n); note flags broken P2P tests.
#   aider score = pass_1 | pass_2 | 0/2 tries.
# Harness/infrastructure failure (no report, harness exception, env never
# started, verdict parser crash): verdict ERROR ("HARNESS ERROR ...; task not
# scored"), counted as lost, never as FAIL — the bench_12 VERIFY-ERROR rule.
# Server death: verdict DIED ("SERVER DIED (<sig>); task not scored"), counted
# as lost, never as FAIL. Two detectors: (1) a non-PASS task whose logs carry
# a transport error (SERVER_DEATH_RE) while /health no longer answers;
# (2) aider only — a /health watchdog during the exercise (aider itself
# retries a dead endpoint for 24 h): HEALTH_FAILS failed polls in a row kill
# the container. After every task the endpoint must answer /health, otherwise
# the rest of that model's tasks are SKIPPED (lost) until the next switch.
# Description: BENCH.md, night/research/24-official-adapter.md
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B13="$HERE/bench13"
PUB="${PUB:-/mnt/cc/strata/bench-public}"

MODELS="${MODELS:-ornith15-128k-mc,tiel-128k-mc}"
SUITE="${SUITE:-both}"              # swe | aider | both
SET="${SET:-screen}"                # screen | core
RUNS="${RUNS:-}"                    # default: screen 1, core 2 (swe); aider always 1 unless set
WORKROOT="${WORKROOT:-$PUB/runs}"
TAG="${TAG:-$(date +%Y%m%d-%H%M)-$SET}"
SWE_TIMEOUT="${SWE_TIMEOUT:-2700}"          # agent wall clock per SWE task (s)
SWE_EVAL_TIMEOUT="${SWE_EVAL_TIMEOUT:-1200}" # harness run per SWE task (s)
AIDER_TIMEOUT="${AIDER_TIMEOUT:-1500}"       # both tries of one exercise (s)
AIDER_EDIT_FORMAT="${AIDER_EDIT_FORMAT:-whole}"
AIDER_IMAGE="${AIDER_IMAGE:-aider-benchmark}"
HEALTH_POLL="${HEALTH_POLL:-15}"            # aider watchdog: /health poll period (s)
HEALTH_FAILS="${HEALTH_FAILS:-3}"           # consecutive failed polls = server dead
AILLAMA_BIN="${AILLAMA_BIN:-aillama}"
BASE_URL="${BASE_URL:-}"            # empty = take it from '$AILLAMA_BIN env'
NO_SWITCH="${NO_SWITCH:-0}"         # 1 = never 'aillama switch' (mock dry runs)
APPEND="${APPEND:-0}"               # 1 = add rows to an existing WORKROOT/TAG/results.tsv
AILLAMA_DIR="${AILLAMA_DIR:-$HOME/.aillama}"   # current-profile + llama-server.pid (same-server check)
SWE_TASKS="${SWE_TASKS:-}"          # optional comma list overriding the set file
AIDER_TASKS="${AIDER_TASKS:-}"      # optional comma list (lang/exercise)
VENV="${VENV:-$PUB/venv}"
SWE_PARQUET="${SWE_PARQUET:-$PUB/swe/2026_03.parquet}"
AIDER_SRC="${AIDER_SRC:-$PUB/aider/aider}"
POLYGLOT="${POLYGLOT:-$PUB/aider/polyglot-benchmark}"
export MSWEA_COST_TRACKING=ignore_errors
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="${MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT:-6}"

die() { echo "ERROR: $*" >&2; exit 1; }

# Transport-level failures only (litellm/httpx/openai/aider wording). A 500
# from a live server (bad template, context overflow) is NOT a death. Aider
# reports a dropped socket as "InternalServerError: OpenAIException -
# Connection error." (rich wraps it over two lines: logs are unwrapped first).
SERVER_DEATH_RE='APIConnectionError|Connection refused|ConnectError|Server disconnected|RemoteProtocolError|ECONNREFUSED|ECONNRESET|Connection reset by peer|Remote end closed connection|OpenAIException - Connection error'

# server_died FILE... — print the first server-death signature in the files
server_died() {
	local sig f
	for f in "$@"; do
		[[ -f "$f" ]] || continue
		sig=$(tr -s '\n ' '  ' <"$f" | grep -aoE "$SERVER_DEATH_RE" | head -n1) || true
		[[ -n "$sig" ]] && { echo "$sig"; return 0; }
	done
	return 1
}

help() {
	cat <<EOF
Usage: ${0##*/} [-h]

Fleet-internal SWE-rebench + Aider Polyglot runs against the local
OpenAI-compatible endpoint. For each aillama profile in MODELS: switch the
server to it, run the selected tasks, verify with the official harness.

Environment variables (current values):
  MODELS            aillama profiles (comma list)          ($MODELS)
  SUITE             swe | aider | both                     ($SUITE)
  SET               screen | core                          ($SET)
                    swe screen = bench13/swe_screen.tsv (12), core = bench13/swe_core.tsv (4)
                    aider (both sets) = bench13/aider_hard.tsv (20)
  RUNS              runs per task (default: screen 1, core 2 for swe; 1 for aider)
  WORKROOT, TAG     results go to WORKROOT/TAG             ($WORKROOT/$TAG)
  SWE_TIMEOUT       agent wall clock per SWE task, s       ($SWE_TIMEOUT)
  SWE_EVAL_TIMEOUT  harness per SWE task, s                ($SWE_EVAL_TIMEOUT)
  AIDER_TIMEOUT     both tries of one exercise, s          ($AIDER_TIMEOUT)
  AIDER_EDIT_FORMAT aider edit format                      ($AIDER_EDIT_FORMAT)
  HEALTH_POLL, HEALTH_FAILS  aider /health watchdog: poll s, fails in a row ($HEALTH_POLL, $HEALTH_FAILS)
  BASE_URL          endpoint override (default: aillama env)
  NO_SWITCH=1       do not call 'aillama switch' (mock dry runs)
  APPEND=1          add rows to an existing TAG (default: refuse to reuse a TAG)
  SWE_TASKS / AIDER_TASKS  comma lists overriding the set files
  PUB               data root                              ($PUB)

Results: WORKROOT/TAG/results.tsv (model suite task run verdict score secs note)
  -> bench13/select_core.py builds the SWE core list from a screen TAG.
EOF
	exit 0
}

# ---------------------------------------------------------------- endpoint --

endpoint() {  # print the /v1 base URL
	if [[ -n "$BASE_URL" ]]; then echo "${BASE_URL%/}"; return; fi
	local b
	b=$("$AILLAMA_BIN" env | sed -n 's/^export OPENAI_BASE_URL="\(.*\)".*/\1/p')
	[[ -n "$b" ]] || die "cannot determine endpoint from '$AILLAMA_BIN env'"
	echo "$b"
}

server_alive() { curl -sf -m 10 "${API%/v1}/health" >/dev/null 2>&1; }

wait_health() {
	local i
	for i in $(seq 1 60); do
		server_alive && return 0
		sleep 5
	done
	die "endpoint ${API%/v1} not healthy after 300 s"
}

switch_model() {
	local model="$1" swlog
	if [[ "$NO_SWITCH" == 1 ]]; then
		echo "--- NO_SWITCH=1: using $API as-is for model name '$model' ---"
		# dry runs: the mock's "server restart" (what aillama switch does for real)
		[[ "${MOCK_REVIVE:-0}" == 1 ]] && { curl -sf -m 5 -X POST "${API%/v1}/mock/revive" >/dev/null 2>&1 || true; }
		server_alive || { echo "!!! endpoint not healthy before $model" >&2; SERVER_DEAD=1; return 0; }
		SERVER_DEAD=0
		return 0
	fi
	echo "--- switching llama-server to profile: $model ---"
	swlog=$(mktemp /tmp/bench13-switch.XXXXXX.log)
	"$AILLAMA_BIN" switch "$model" >"$swlog" 2>&1 || {
		tail -5 "$swlog" >&2
		die "aillama switch $model failed (full log: $swlog)"
	}
	command rm -f "$swlog"
	wait_health
	SERVER_DEAD=0
	# remember which llama-server process serves this profile (no pid/profile
	# file, e.g. a Strata profile: the same-server check is skipped)
	SERVER_PID=""
	local pid
	pid=$(cat "$AILLAMA_DIR/llama-server.pid" 2>/dev/null || true)
	if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
			&& [[ "$(cat "$AILLAMA_DIR/current-profile" 2>/dev/null)" == "$model" ]]; then
		SERVER_PID="$pid"
	fi
}

same_server() {  # same_server MODEL: same profile and same llama-server process as at switch time
	[[ "$NO_SWITCH" == 1 || -z "${SERVER_PID:-}" ]] && return 0
	[[ "$(cat "$AILLAMA_DIR/current-profile" 2>/dev/null)" == "$1" ]] || return 1
	[[ "$(cat "$AILLAMA_DIR/llama-server.pid" 2>/dev/null)" == "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null
}

# ------------------------------------------------------------------- lists --

swe_list() {
	if [[ -n "$SWE_TASKS" ]]; then tr ',' '\n' <<<"$SWE_TASKS"; return; fi
	local f="$B13/swe_$SET.tsv"
	[[ -f "$f" ]] || die "missing $f (SET=core needs: bench13/select_core.py on a screen run)"
	python3 "$B13/b13.py" ids "$f"
}

aider_list() {
	if [[ -n "$AIDER_TASKS" ]]; then tr ',' '\n' <<<"$AIDER_TASKS"; return; fi
	python3 "$B13/b13.py" ids "$B13/aider_hard.tsv"
}

# --------------------------------------------------------------------- swe --

DS_DIR=""
prepare_swe() {
	DS_DIR="$OUT/swe-dataset"
	local ids
	ids=$(swe_list) || die "cannot read the SWE task list"
	[[ -n "${ids//[[:space:]]/}" ]] || die "empty SWE task list (SET=$SET, SWE_TASKS='$SWE_TASKS')"
	printf '%s\n' $ids >"$OUT/swe-ids.tsv"
	"$VENV/bin/python" "$B13/b13.py" swe-dataset "$SWE_PARQUET" "$OUT/swe-ids.tsv" "$DS_DIR" >/dev/null \
		|| die "cannot build SWE dataset"
	local img missing=0
	while read -r img; do
		docker image inspect "$img" >/dev/null 2>&1 || { echo "missing image: $img" >&2; missing=1; }
	done < <("$VENV/bin/python" -c 'import json,sys; [print(json.loads(l)["docker_image"]) for l in open(sys.argv[1])]' "$DS_DIR/test.jsonl")
	(( missing == 0 )) || die "pull the images above first (nice -n 10 ionice -c3 docker pull ...)"
}

# reap_mini BEFORE: remove minisweagent-* task containers that are not in BEFORE
# (a killed mini leaves its `sleep 3h` container behind, with whatever the
# agent started inside it still running next to llama-server)
reap_mini() {
	comm -13 <(printf '%s\n' "$1") <(docker ps -aq --filter name=minisweagent- | sort) \
		| xargs -r docker rm -f >/dev/null 2>&1 || true
}

# run_swe MODEL ID RUN -> sets V_VERDICT V_SCORE V_NOTE V_LOGS
run_swe() {
	local model="$1" id="$2" run="$3" dir rc=0 st="" rid
	dir="$OUT/swe/$model/r$run/$id"
	rm -rf "$dir" && mkdir -p "$dir"
	MINI_BEFORE=$(docker ps -aq --filter name=minisweagent- | sort)
	V_LOGS="$dir/agent.log $dir/minisweagent.log"
	timeout -k 60 $(( SWE_TIMEOUT + 300 )) "$VENV/bin/mini-extra" swebench \
		--subset "$DS_DIR" --split test --filter "^${id}\$" -o "$dir" -w 1 \
		-m "openai/$model" \
		-c "$MINI_CFG" -c "$B13/swe_local.yaml" \
		-c "model.model_kwargs.api_base=$API" \
		-c "agent.wall_time_limit_seconds=$SWE_TIMEOUT" \
		>"$dir/agent.log" 2>&1 </dev/null || rc=$?
	reap_mini "$MINI_BEFORE"; MINI_BEFORE=""
	st=$("$VENV/bin/python" - "$dir/$id/$id.traj.json" <<-'PY' 2>/dev/null || true
	import json, sys
	t = json.load(open(sys.argv[1]))
	i = t.get("info", {})
	print(f'{i.get("exit_status", "?")}|{i.get("model_stats", {}).get("api_calls", "?")}')
	PY
	)
	local exit_status="${st%%|*}" calls="${st#*|}"
	if [[ $rc -eq 124 || $rc -eq 137 || "$exit_status" == TimeExceeded ]]; then
		V_VERDICT=TIMEOUT V_SCORE="-" V_NOTE="agent wall clock ${SWE_TIMEOUT}s (${calls:-?} calls)"
		return
	fi
	if [[ -z "$exit_status" ]]; then   # no trajectory: the env never started (docker), not a model failure
		V_VERDICT=ERROR V_SCORE="-" V_NOTE="HARNESS ERROR: no trajectory, rc $rc (see $dir/agent.log); task not scored"
		return
	fi
	if [[ ! -s "$dir/preds.json" ]]; then
		V_VERDICT=FAIL V_SCORE="-" V_NOTE="no prediction (agent exit '${exit_status:-?}', rc $rc)"
		return
	fi
	# unique run id per task-run: the eval container name is sweb.eval.<id>.<run_id>,
	# so an orphan of an earlier model/run can never collide (HTTP 409 -> no report);
	# the outer clock covers container start + patch + report, so it must exceed
	# the harness' own test timeout or a slow suite is always killed from outside
	rid="b13-$$-${model}-r${run}"; rid=${rid,,}; rid=${rid//[^a-z0-9_.-]/_}
	CUR_EVAL="sweb.eval.${id,,}.$rid"
	docker rm -f "$CUR_EVAL" >/dev/null 2>&1 || true
	(cd "$dir" && timeout -k 30 $(( SWE_EVAL_TIMEOUT + 300 )) "$VENV/bin/python" -m swebench.harness.run_evaluation \
		--dataset_name "$DS_DIR/test.jsonl" --predictions_path "$dir/preds.json" \
		--instance_ids "$id" --max_workers 1 --run_id "$rid" --namespace swerebench \
		--cache_level instance --timeout "$SWE_EVAL_TIMEOUT" --report_dir "$dir" \
		>"$dir/eval.log" 2>&1 </dev/null) || true
	docker rm -f "$CUR_EVAL" >/dev/null 2>&1 || true
	CUR_EVAL=""
	# a parser crash (truncated report.json ...) must not abort the night under set -e
	if ! IFS='|' read -r V_VERDICT V_SCORE V_NOTE < <("$VENV/bin/python" "$B13/b13.py" swe-verdict \
			"$dir/logs/run_evaluation/$rid" "$id" "$dir/preds.json" 2>>"$dir/eval.log"); then
		V_VERDICT=ERROR V_SCORE="-" V_NOTE="verdict parser crashed (see eval.log)"
	fi
	V_NOTE="${V_NOTE:+$V_NOTE; }exit ${exit_status:-?}, ${calls:-?} calls"
	# an unevaluated patch is a bench failure, not a model failure (bench_12: VERIFY-ERROR)
	if [[ "$V_VERDICT" == ERROR ]]; then V_NOTE="HARNESS ERROR (see eval.log); task not scored; $V_NOTE"; fi
	return 0
}

# ------------------------------------------------------------------- aider --

AIDER_SETTINGS=""
prepare_aider() {
	docker image inspect "$AIDER_IMAGE" >/dev/null 2>&1 || die "image $AIDER_IMAGE missing (build: cd $AIDER_SRC && docker build -f benchmark/Dockerfile -t $AIDER_IMAGE .)"
	[[ -d "$POLYGLOT/python/exercises/practice" ]] || die "missing $POLYGLOT"
	mkdir -p "$OUT/aider" "$PUB/cache/gradle"
	local t
	t=$(aider_list) || die "cannot read the aider task list"
	[[ -n "${t//[[:space:]]/}" ]] || die "empty aider task list (AIDER_TASKS='$AIDER_TASKS')"
	for t in $(aider_list); do
		[[ -d "$POLYGLOT/${t%%/*}/exercises/practice/${t#*/}" ]] || die "unknown exercise $t"
	done
}

# per-model aider settings: no temperature (server-side sampling like bench_05/
# qwen-code), strip <think> if a template leaks it into content
aider_settings() {
	local model="$1" f="$OUT/aider/$1/model-settings.yml"
	mkdir -p "$OUT/aider/$model"
	cat >"$f" <<-EOF
	- name: openai/$model
	  edit_format: $AIDER_EDIT_FORMAT
	  use_temperature: false
	  reasoning_tag: think
	  examples_as_sys_msg: true
	EOF
	echo "$f"
}

# run_aider MODEL LANG/EXERCISE RUN -> sets V_VERDICT V_SCORE V_NOTE V_LOGS
run_aider() {
	local model="$1" ex="$2" run="$3" lang name mdir rdir log res rc=0 cname
	lang="${ex%%/*}" name="${ex#*/}"
	mdir="$OUT/aider/$model"
	rdir="$mdir/r$run"
	# rdir must NOT exist before aider's first call: benchmark.py copies the
	# exercises only into a missing run dir
	mkdir -p "$mdir/logs-r$run"
	log="$mdir/logs-r$run/${lang}_${name}.log"
	res="$rdir/$lang/exercises/practice/$name/.aider.results.json"
	rm -f "$res"
	V_LOGS="$log"
	cname="b13-aider-$$-${lang}-${name}"
	local pid bad=0 tick=0 wd_dead=0 t0
	t0=$(date +%s)
	CUR_CNAME="$cname"   # the INT/TERM trap kills it (a background job ignores SIGINT)
	timeout -k 30 $(( AIDER_TIMEOUT + 120 )) docker run --rm --name "$cname" \
		--memory=12g --memory-swap=12g \
		-v "$AIDER_SRC":/aider -v "$mdir":/benchmarks -v "$POLYGLOT":/polyglot:ro \
		-v "$PUB/cache/gradle":/root/.gradle \
		-e OPENAI_API_KEY=none -e OPENAI_API_BASE="$API" \
		-e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks \
		"$AIDER_IMAGE" bash -c '
			set -u
			[ -e /benchmarks/polyglot-benchmark ] || ln -s /polyglot /benchmarks/polyglot-benchmark
			timeout -k 20 '"$AIDER_TIMEOUT"' python3 benchmark/benchmark.py /benchmarks/r'"$run"' \
				--model openai/'"$model"' --edit-format '"$AIDER_EDIT_FORMAT"' \
				--read-model-settings /benchmarks/model-settings.yml \
				--threads 1 --tries 2 --exercises-dir polyglot-benchmark \
				--keywords '"$lang/exercises/practice/$name"'
			rc=$?
			chown -R '"$(id -u):$(id -g)"' /benchmarks/r'"$run"' 2>/dev/null
			exit $rc' >"$log" 2>&1 </dev/null &
	pid=$!
	# watchdog: aider retries a dead endpoint for up to 24 h, so poll /health
	while kill -0 "$pid" 2>/dev/null; do
		sleep 1
		tick=$(( tick + 1 ))
		(( tick % HEALTH_POLL == 0 )) || continue
		kill -0 "$pid" 2>/dev/null || break
		if server_alive; then bad=0; else bad=$(( bad + 1 )); fi
		if (( bad >= HEALTH_FAILS )); then
			wd_dead=1
			echo "[bench13] /health failed ${bad}x in a row — killing $cname" >>"$log"
			docker kill "$cname" >/dev/null 2>&1 || true
			break
		fi
	done
	wait "$pid" || rc=$?
	docker kill "$cname" >/dev/null 2>&1 || true
	CUR_CNAME=""
	# a kill before the in-container chown leaves root-owned files (later rm -rf fails)
	if [[ -d "$rdir" && -n "$(find "$rdir" ! -user "$(id -u)" -print -quit 2>/dev/null)" ]]; then
		docker run --rm -v "$mdir":/benchmarks "$AIDER_IMAGE" \
			chown -R "$(id -u):$(id -g)" "/benchmarks/r$run" >/dev/null 2>&1 || true
	fi
	if (( wd_dead )); then
		local sig i
		# a /health that stalls under load is not a death: give it 60 s to answer
		for i in $(seq 1 12); do
			if server_alive; then
				V_VERDICT=ERROR V_SCORE="-" V_NOTE="watchdog false alarm (/health failed ${HEALTH_FAILS}x, answered again within 60 s); task not scored"
				return 0
			fi
			sleep 5
		done
		sig=$(server_died "$log") || sig="no transport error logged"
		V_VERDICT=DIED V_SCORE="-" V_NOTE="SERVER DIED (/health failed ${HEALTH_FAILS}x; $sig); task not scored"
		return 0
	fi
	if [[ ! -f "$res" ]]; then
		if [[ $rc -eq 124 ]] || (( rc == 137 && $(date +%s) - t0 >= AIDER_TIMEOUT )); then
			V_VERDICT=TIMEOUT V_SCORE="-" V_NOTE="both tries > ${AIDER_TIMEOUT}s"
		elif [[ $rc -eq 137 ]]; then
			V_VERDICT=FAIL V_SCORE="-" V_NOTE="killed (rc 137 before the time limit: container OOM at 12g?)"
		else
			V_VERDICT=ERROR V_SCORE="-" V_NOTE="HARNESS ERROR rc $rc, no results (see $log); task not scored"
		fi
		return 0
	fi
	if ! IFS='|' read -r V_VERDICT V_SCORE V_NOTE < <(python3 "$B13/b13.py" aider-verdict "$res" 2>>"$log"); then
		V_VERDICT=ERROR V_SCORE="-" V_NOTE="verdict parser crashed (see $log)"
	fi
	if [[ "$V_VERDICT" == ERROR ]]; then V_NOTE="HARNESS ${V_NOTE}; task not scored"; fi
	# aider's harness re-copies only the test files: a new conftest.py / package.json
	# "test" script / build file can turn the grader green -> never a PASS
	if [[ $V_VERDICT == PASS ]]; then
		local tamper=""
		if ! tamper=$(python3 "$B13/b13.py" aider-tamper "$rdir/$lang/exercises/practice/$name" \
				"$POLYGLOT/$lang/exercises/practice/$name" 2>>"$log"); then
			V_VERDICT=ERROR V_NOTE="HARNESS tamper check crashed (see $log); task not scored; $V_NOTE"
		elif [[ -n "$tamper" ]]; then
			V_VERDICT=FAIL V_NOTE="TAMPER: $tamper; graded $V_SCORE; $V_NOTE" V_SCORE="-"
		fi
	fi
	return 0
}

# ----------------------------------------------------------------- driver --

RESULTS=()
LOST=0
SERVER_DEAD=0
CUR_CNAME=""
CUR_EVAL=""
MINI_BEFORE=""
SERVER_PID=""
on_signal() {
	if [[ -n "$CUR_CNAME" ]]; then docker kill "$CUR_CNAME" >/dev/null 2>&1 || true; fi
	if [[ -n "$CUR_EVAL" ]]; then docker rm -f "$CUR_EVAL" >/dev/null 2>&1 || true; fi
	if [[ -n "$MINI_BEFORE" ]]; then reap_mini "$MINI_BEFORE"; fi
	echo "interrupted; partial results: ${OUT:-?}/results.tsv" >&2
	exit 130
}
trap on_signal INT TERM

record() {  # model suite task run verdict score secs note
	RESULTS+=("$1|$2|$3|$4|$5|$6|$7|$8")
	printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >>"$OUT/results.tsv"
}

run_one() {
	local model="$1" suite="$2" task="$3" run="$4" start dur sig="" label
	label="$model/$suite:$task@r$run"
	if (( SERVER_DEAD )); then
		echo "=== [$label] SKIPPED — server dead, task not run"
		record "$model" "$suite" "$task" "$run" SKIPPED - - "server dead; task not run"
		((LOST++)) || true
		return
	fi
	echo "=== [$label] running (timeout $([[ $suite == swe ]] && echo "$SWE_TIMEOUT" || echo "$AIDER_TIMEOUT")s) ==="
	start=$(date +%s)
	V_VERDICT=FAIL V_SCORE="-" V_NOTE="" V_LOGS=""
	"run_$suite" "$model" "$task" "$run"
	dur=$(( $(date +%s) - start ))
	# shellcheck disable=SC2086
	if [[ $V_VERDICT == DIED ]]; then
		((LOST++)) || true
	elif [[ $V_VERDICT != PASS ]] && sig=$(server_died $V_LOGS); then
		# a transport error is a death only if the server is really gone now
		# (agent logs may quote tool output such as "Connection refused")
		if server_alive; then
			if [[ $V_VERDICT == ERROR ]]; then
				V_NOTE="${V_NOTE:+$V_NOTE; }transport error '$sig' in logs, /health ok"
				((LOST++)) || true
			else
				V_NOTE="${V_NOTE:+$V_NOTE; }transport error '$sig' in logs but /health ok — scored"
			fi
		else
			V_VERDICT=DIED V_SCORE="-" V_NOTE="SERVER DIED ($sig); task not scored"
			((LOST++)) || true
		fi
	elif [[ $V_VERDICT == ERROR ]]; then
		((LOST++)) || true   # harness failure: rerun it, never a FAIL
	fi
	if ! same_server "$model"; then
		[[ $V_VERDICT == DIED || $V_VERDICT == ERROR ]] || ((LOST++)) || true
		V_VERDICT=DIED V_SCORE="-" V_NOTE="SERVER CHANGED (profile or llama-server pid differs from switch time); task not scored"
		SERVER_DEAD=1
	fi
	echo "=== [$label] $V_VERDICT (${dur}s) ${V_SCORE:+[$V_SCORE] }${V_NOTE:+— $V_NOTE}"
	record "$model" "$suite" "$task" "$run" "$V_VERDICT" "${V_SCORE:--}" "$dur" "$V_NOTE"
	if ! server_alive; then
		SERVER_DEAD=1
		echo "!!! endpoint does not answer /health after $label — remaining tasks of this model are SKIPPED" >&2
	fi
}

summary() {
	local r m s t k v sc d n
	echo
	echo "==================== SUMMARY ($TAG) ===================="
	printf '%-22s %-5s %-44s %-3s %-8s %-14s %-6s %s\n' MODEL SUITE TASK RUN VERDICT SCORE TIME NOTE
	for r in "${RESULTS[@]}"; do
		IFS='|' read -r m s t k v sc d n <<<"$r"
		printf '%-22s %-5s %-44s %-3s %-8s %-14s %-6s %s\n' "$m" "$s" "$t" "r$k" "$v" "$sc" "${d}s" "$n"
	done
	echo
	echo "Per model (lost = SERVER DIED + SKIPPED + harness ERROR; never counted as FAIL):"
	"$VENV/bin/python" "$B13/select_core.py" --summary "$OUT/results.tsv"
	echo
	echo "Paired per-task table (same tasks, same order; * = models disagree):"
	"$VENV/bin/python" "$B13/select_core.py" --paired "$OUT/results.tsv"
	if (( LOST )); then echo "$LOST task-runs lost (server death / harness ERROR): rerun them, do not count them as FAIL."; fi
	echo "results: $OUT/results.tsv"
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && help
[[ $# -eq 0 ]] || help
case "$SUITE" in swe|aider|both) ;; *) die "SUITE must be swe|aider|both" ;; esac
case "$SET" in screen|core) ;; *) die "SET must be screen|core" ;; esac
command -v docker >/dev/null || die "docker not found"
[[ -x "$VENV/bin/mini-extra" ]] || die "venv missing: $VENV (see night/research/24-official-adapter.md)"
[[ "$NO_SWITCH" == 1 ]] || command -v "$AILLAMA_BIN" >/dev/null || die "binary not found: $AILLAMA_BIN"
MINI_CFG="swebench.yaml"   # builtin: minisweagent/config/benchmarks/swebench.yaml
API=$(endpoint)
OUT="$WORKROOT/$TAG"
mkdir -p "$OUT"
if [[ -s "$OUT/results.tsv" ]]; then
	[[ "$APPEND" == 1 ]] || die "$OUT/results.tsv exists — pick a new TAG (reruns: TAG=<tag>-rerun), or APPEND=1 to add rows"
else
	: >"$OUT/results.tsv"
fi
SWE_RUNS="${RUNS:-$([[ $SET == core ]] && echo 2 || echo 1)}"
AIDER_RUNS="${RUNS:-1}"
[[ $SUITE == aider ]] || prepare_swe
[[ $SUITE == swe ]] || prepare_aider
{
	echo "TAG=$TAG SET=$SET SUITE=$SUITE MODELS=$MODELS API=$API SWE_RUNS=$SWE_RUNS AIDER_RUNS=$AIDER_RUNS"
	echo "SWE_TIMEOUT=$SWE_TIMEOUT AIDER_TIMEOUT=$AIDER_TIMEOUT AIDER_EDIT_FORMAT=$AIDER_EDIT_FORMAT"
	"$VENV/bin/python" -c 'import importlib.metadata as m; print("mini-swe-agent", m.version("mini-swe-agent"), "litellm", m.version("litellm"), "swebench", m.version("swebench"))'
	git -C "$AIDER_SRC" log -1 --format='aider %h %cs' 2>/dev/null || true
} | tee "$OUT/run-info.txt"

for model in ${MODELS//,/ }; do
	switch_model "$model"
	if [[ $SUITE != aider ]]; then
		for run in $(seq 1 "$SWE_RUNS"); do
			for id in $(swe_list); do run_one "$model" swe "$id" "$run"; done
		done
	fi
	if [[ $SUITE != swe ]]; then
		aider_settings "$model" >/dev/null
		for run in $(seq 1 "$AIDER_RUNS"); do
			for ex in $(aider_list); do run_one "$model" aider "$ex" "$run"; done
		done
	fi
done
summary
