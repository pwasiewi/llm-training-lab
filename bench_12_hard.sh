#!/bin/bash
# bench_12_hard.sh — HARD agentic benchmark via qwen-code (our workflow realities)
# Part of the bench_NN_* series. bench_05 (parser tier 4/4) and bench_07
# (20/20) are saturated for the serious-agentic tier; research/22 picked
# public sets (TB2-48, SR-21, AP-60) for generic ranking. bench_12 measures
# what those do not — the failure modes observed in OUR qwen-code work —
# with tasks sized so the top fleet lands at 30-70 % PASS:
#   ledger   (a) multi-file feature across a 12-file repo, hidden CLI
#                integration test (propagation to every call site)
#   chain6/chain10/chain14
#            (b) one task family, N dependent lookup+transform steps, each
#                output feeds the next; wrong values stay silent (total
#                range lookups). SCORE k = correct-prefix length -> per-step
#                reliability p (horizon fit in the summary). Data files are
#                identical across N (fill-matched: N is not confounded with
#                how much there is to read)
#   tracespec (c) bug hunt: README and unit tests disagree with recorded
#                production traces; the traces are the spec; hidden traces
#   reconcile (d) performance: make a slow matcher fast with byte-identical
#                output; time limit is relative to a reference run in the
#                same verify call (robust to llama-server CPU load)
#   ebuild   (e) Gentoo-flavoured packaging in a sandbox: fix a recipe for
#                a fake ebuild runner + patch a hostile Makefile, QA gates
#                (DESTDIR, libdir, CFLAGS, prestripped, doc dir), sources
#                protected by a Manifest
#   longctx  (f) ~75K-token incident dump; questions need facts COMBINED
#                across files (aliases, per-file timezones, override
#                precedence); one fixed-position retrieval control
#   rules    (g) tool discipline: ~450-line RULES.md, arbitrary literals at
#                three depths incl. the tail, an amending rule, salted
#                evidence token recomputed on the final artifact
#
# Every task ships prompt.md, a fixture, verify.py (ITEM lines + SCORE k/n,
# exit 0/1/2), reference/ and broken/ solutions under bench12/tasks/<task>;
# `--selftest` proves each verifier PASSes the reference and FAILs every
# broken variant. `--oracle` runs the whole driver path (setup, verify,
# summary, horizon fit) with the reference as the "agent" — no server.
#
# Server death: same rule as bench_05/07 — a non-PASS run whose agent.log
# carries an undici transport error is "SERVER DIED (<sig>); run not
# scored", counted as lost, never as FAIL; after each run /health must
# answer or the model's remaining runs are SKIPPED.
# Design + calibration plan: night/research/23-bench12-design.md
set -euo pipefail

MODELS="${MODELS:-ornith15-128k-mc}"
ALL_TASKS="ledger,chain6,chain10,chain14,tracespec,reconcile,ebuild,longctx,rules"
TASKS="${TASKS:-$ALL_TASKS}"
RUNS="${RUNS:-1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-}"      # empty = per-task BUDGET from task.conf
WORKROOT_SET="${WORKROOT:+1}"         # --oracle uses a fresh mktemp root unless WORKROOT is given
WORKROOT="${WORKROOT:-/tmp/bench12}"
AILLAMA_BIN="${AILLAMA_BIN:-aillama}"
QWEN_BIN="${QWEN_BIN:-qwen}"
B12_SEED="${B12_SEED:-12}"
export B12_SEED
export QWEN_CODE_SUPPRESS_YOLO_WARNING=1
# Task prompts run under the "bench" airag identity (AIRAG_SOURCE=bench at
# the QWEN_BIN call); ~/.qwen/hooks/airag-inject.sh skips that entity.

B12_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/bench12"
META_ROOT="${WORKROOT%/}.meta"        # pristine copies: a SIBLING of WORKROOT
# Advisory peek detector, matched against agent.log AND the run's qwen session
# transcript (headless `qwen -p` logs only the final answer; tool calls and
# their output live in ~/.qwen/projects/<cwd>/chats/*.jsonl). The real guard
# is the bwrap sandbox (sandbox_cmd).
PEEK_RE='bench12/(tasks|lib)|bench_12_hard|\.meta/|night/research/2[0-9]'
QWEN_PROJECTS="$HOME/.qwen/projects"
B12_OVERWRITE="${B12_OVERWRITE:-0}"
B12_SANDBOX="${B12_SANDBOX:-1}"       # 0 = run the agent without bwrap (results flagged UNSANDBOXED)

die() { echo "ERROR: $*" >&2; exit 1; }

# Same helper as bench_05/07 — keep identical.
SERVER_DEATH_RE='UND_ERR_SOCKET|ECONNREFUSED|fetch failed|other side closed'

server_died() {
	local sig
	[[ -f "${1:-}" ]] || return 1
	sig=$(grep -oE "$SERVER_DEATH_RE" "$1" | head -n1) || true
	[[ -n "$sig" ]] || return 1
	echo "$sig"
}

help() {
	cat <<EOF
Usage: ${0##*/} [-h|--help] [--selftest] [--oracle]

Hard agentic benchmark: for each aillama profile in MODELS, run each task in
TASKS through headless qwen-code RUNS times; each run is verified by a
script-side verify.py that the agent cannot see (protected-file sha256 vs a
pristine copy outside WORKROOT, hidden tests, recomputed answers).
Per-run line: === [model/task run r/R] VERDICT (Ns) [k/n (pct%)] — note
Summary: per-run table, per-model x task pass rate + mean partial score,
item-compliance matrix, and a horizon fit (per-step p) for chain6/10/14.

  --selftest   no model, no server: run every verifier on its reference
               solution (expect PASS) and on each broken/ variant (expect
               FAIL); print a table; exit 1 on any mismatch
  --oracle     no model, no server: full driver path with the reference
               solution standing in for the agent (driver smoke test);
               ORACLE_SOLUTION=broken/<name> substitutes a broken variant
               where a task has one (exercises FAIL rows and the fits)

Tasks: $ALL_TASKS
  (per-task budget seconds and family in bench12/tasks/<task>/task.conf)

Environment variables (current values):
  MODELS        aillama profiles to compare   ($MODELS)
  TASKS         task subset                    ($TASKS)
  RUNS          runs per model x task          ($RUNS)
  TASK_TIMEOUT  seconds per run; empty = task.conf BUDGET (${TASK_TIMEOUT:-per-task})
  WORKROOT      scratch directory root         ($WORKROOT)
                pristine copies go to          ($META_ROOT)
                --oracle without WORKROOT uses a fresh /tmp/bench12-oracle.XXXXXX
  B12_OVERWRITE 1 = reuse an existing WORKROOT/<model> (default: refuse)
  B12_SANDBOX   1 = agent runs under bwrap: /tmp, ~/Claude/llm, the git dir,
                WORKROOT, META_ROOT, other qwen transcripts and ~/.claude* are
                hidden; only the run's own dir is visible ($B12_SANDBOX)
  B12_SEED      generator seed                 ($B12_SEED)
  AILLAMA_BIN   aillama binary                 ($AILLAMA_BIN)
  QWEN_BIN      qwen-code binary               ($QWEN_BIN)
EOF
	exit 0
}

# ------------------------------------------------------------ task plumbing --

task_conf() {  # task_conf TASK KEY -> value
	local f="$B12_DIR/tasks/$1/task.conf"
	[[ -f "$f" ]] || die "no task.conf for task '$1'"
	sed -n "s/^$2=//p" "$f" | head -n1
}

setup_task() {  # setup_task TASK DIR PRISTINE
	local task="$1" dir="$2" pristine="$3" tdir="$B12_DIR/tasks/$1"
	[[ -d "$tdir" ]] || die "unknown task: $task"
	command rm -rf "$dir" "$pristine"
	mkdir -p "$dir" "$pristine"
	if [[ -d "$tdir/fixture" ]]; then command cp -a "$tdir/fixture/." "$dir/"; fi
	if [[ -f "$tdir/gen.py" ]]; then
		python3 "$tdir/gen.py" "$dir" >/dev/null || die "generator failed for $task"
	fi
	# never ship interpreter caches (stale bytecode of fixture code) to the agent
	find "$dir" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	command cp -a "$dir/." "$pristine/"
}

apply_solution() {  # apply_solution SRC DIR — overlay files, then run SRC/_solve.sh in DIR
	local src="$1" dir="$2"
	local f
	command cp -a "$src/." "$dir/"
	# files named _* are solver plumbing, not part of the solution
	for f in "$src"/_*; do if [[ -e "$f" ]]; then command rm -rf "$dir/${f##*/}"; fi; done
	if [[ -f "$src/_solve.sh" ]]; then
		(cd "$dir" && bash "$src/_solve.sh" "$dir") >/dev/null 2>&1 || return 1
	fi
	return 0
}

run_verify() {  # run_verify TASK DIR PRISTINE OUTFILE -> rc (0 PASS, 1 FAIL, 2 error)
	local rc=0
	timeout 900 python3 "$B12_DIR/tasks/$1/verify.py" "$2" "$3" >"$4" 2>&1 || rc=$?
	(( rc == 124 )) && { echo "VERIFY-ERROR verifier timed out" >>"$4"; rc=2; }
	(( rc > 2 )) && rc=2
	return $rc
}

score_of() {  # score_of VERIFY_OUT -> "k/n (pct%)" or empty
	local s k n
	s=$(grep -oE '^SCORE [0-9]+/[0-9]+' "$1" | tail -n1 | cut -d' ' -f2) || true
	[[ -n "$s" ]] || return 0
	k=${s%/*}; n=${s#*/}
	if (( n > 0 )); then echo "$k/$n ($(( 100 * k / n ))%)"; fi
}

failed_items() {
	grep -E '^ITEM [^ ]+ FAIL' "$1" | awk '{print $2}' | paste -sd, - || true
}

# ------------------------------------------------------------------ selftest --

selftest() {
	local tasks="${TASKS:-$ALL_TASKS}" task tdir case_dir name expect rc got score t0 dur
	local tmp mism=0 ncase=0
	tmp=$(mktemp -d /tmp/bench12-selftest.XXXXXX)
	# selftest exits the script: remove the scratch on die / Ctrl-C too
	# shellcheck disable=SC2064
	trap "command rm -rf '$tmp'" EXIT
	echo "selftest: verifiers vs reference (expect PASS) and broken/* (expect FAIL); scratch $tmp"
	printf '%-10s %-22s %-7s %-12s %-14s %-6s %s\n' TASK CASE EXPECT GOT SCORE TIME OK
	for task in ${tasks//,/ }; do
		tdir="$B12_DIR/tasks/$task"
		[[ -d "$tdir" ]] || die "unknown task: $task"
		for case_dir in "$tdir/reference" "$tdir"/broken/*/; do
			case_dir="${case_dir%/}"
			[[ -d "$case_dir" ]] || continue
			if [[ "$case_dir" == */reference ]]; then name=reference; expect=PASS
			else name="broken/${case_dir##*/}"; expect=FAIL; fi
			setup_task "$task" "$tmp/w" "$tmp/p"
			t0=$(date +%s)
			rc=0
			if ! apply_solution "$case_dir" "$tmp/w"; then
				echo "  ($task/$name: _solve.sh failed)" >&2
			fi
			run_verify "$task" "$tmp/w" "$tmp/p" "$tmp/out" || rc=$?
			dur=$(( $(date +%s) - t0 ))
			case $rc in 0) got=PASS ;; 1) got=FAIL ;; *) got=VERIFY-ERROR ;; esac
			score=$(score_of "$tmp/out")
			((ncase++)) || true
			if [[ $got == "$expect" ]]; then ok=ok
			else ok=MISMATCH; ((mism++)) || true
				sed 's/^/    | /' "$tmp/out" | tail -n 12 >&2
			fi
			printf '%-10s %-22s %-7s %-12s %-14s %-6s %s\n' "$task" "$name" "$expect" "$got" "${score:--}" "${dur}s" "$ok"
			if [[ $expect == FAIL && $got == FAIL ]]; then
				echo "           failed items: $(failed_items "$tmp/out")"
			fi
		done
	done
	command rm -rf "$tmp"
	echo
	echo "selftest: $ncase cases, $mism mismatches"
	(( mism == 0 ))
}

# --------------------------------------------------------------- server I/O --

server_alive() {
	(( ORACLE )) && return 0
	local base
	base=$("$AILLAMA_BIN" env | sed -n 's/^export OPENAI_BASE_URL="\(.*\)\/v1".*/\1/p')
	[[ -n "$base" ]] || die "cannot determine endpoint from '$AILLAMA_BIN env'"
	curl -sf -m 10 "$base/health" >/dev/null 2>&1
}

wait_health() {
	local i
	for i in $(seq 1 60); do
		server_alive && return 0
		sleep 5
	done
	die "llama-server not healthy after 300 s"
}

switch_model() {
	local model="$1" swlog
	SERVER_DEAD=0
	(( ORACLE )) && { echo "--- oracle mode: no server, reference solution stands in for the agent ---"; return 0; }
	echo "--- switching llama-server to profile: $model ---"
	swlog=$(mktemp /tmp/bench12-switch.XXXXXX.log)
	"$AILLAMA_BIN" switch "$model" >"$swlog" 2>&1 || {
		tail -5 "$swlog" >&2
		die "aillama switch $model failed (full log: $swlog)"
	}
	command rm -f "$swlog"
	wait_health
}

# ---------------------------------------------------------------- sandbox --

SANDBOX_OK=0
sandbox_init() {  # decide once per invocation whether the agent runs under bwrap
	(( ORACLE )) && return 0
	if [[ "$B12_SANDBOX" != 1 ]]; then
		echo "WARN: B12_SANDBOX=$B12_SANDBOX — agent runs WITHOUT the bwrap sandbox; rows are flagged UNSANDBOXED" >&2
		return 0
	fi
	if ! command -v bwrap >/dev/null; then
		echo "WARN: bwrap (sys-apps/bubblewrap) not installed — agent runs WITHOUT the sandbox: earlier runs," >&2
		echo "WARN: META_ROOT answer keys and bench12/ stay readable; rows are flagged UNSANDBOXED" >&2
		return 0
	fi
	if ! bwrap --dev-bind / / --unshare-pid --proc /proc --die-with-parent -- true 2>/dev/null; then
		echo "WARN: bwrap cannot create namespaces here — agent runs WITHOUT the sandbox; rows are flagged UNSANDBOXED" >&2
		return 0
	fi
	SANDBOX_OK=1
}

qwen_project() {  # qwen's per-cwd project dir name: every non-alphanumeric char -> '-'
	printf '%s' "$1" | sed 's/[^A-Za-z0-9]/-/g'
}

# sandbox_cmd DIR PROJDIR -> SBX array (empty without bwrap). The agent sees
# the host as-is (network, ~/.qwen config, toolchains) except: a private /tmp,
# no ~/Claude/llm (bench12/, the drivers, night/research), no git history of
# it, no WORKROOT/META_ROOT (earlier runs, .verify answer keys), no other qwen
# transcripts, no Claude notes; ~/QwenCode is a per-run tmpfs (no cross-run
# channel). Only DIR and its own transcript dir are bound read-write.
# --unshare-pid: whatever the agent left running dies with the run.
sandbox_cmd() {
	local dir="$1" projdir="$2" p
	SBX=()
	(( SANDBOX_OK )) || return 0
	SBX=(bwrap --dev-bind / / --unshare-pid --proc /proc --die-with-parent --tmpfs /tmp)
	for p in "$LLM_DIR" "$REPO_GIT_DIR" "$WORKROOT" "$META_ROOT" "$QWEN_PROJECTS" \
			"$HOME/.claude" "$HOME/.claude-mem" "$HOME/QwenCode"; do
		[[ -n "$p" && -d "$p" ]] && SBX+=(--tmpfs "$p")
	done
	SBX+=(--bind "$projdir" "$projdir" --bind "$dir" "$dir" --chdir "$dir" --)
}

# ------------------------------------------------------------------ runs ----

RESULTS=()
LOST=0
SERVER_DEAD=0
ORACLE=0
SUMMARY_DONE=0
SBX=()
LLM_DIR="$(readlink -f "$B12_DIR/..")"
REPO_GIT_DIR="$(git -C "$B12_DIR" rev-parse --absolute-git-dir 2>/dev/null || true)"
declare -A AGG_TOT AGG_PASS
ITEMS_SEEN=()

run_task() {
	local model="$1" task="$2" run="$3" dir pristine vout stamp rc=0 vrc=0 start dur verdict
	local score="" note="" sig="" budget lost=0 verr=0 pe="-" projdir=""
	dir="$WORKROOT/$model/$task-run$run"
	pristine="$META_ROOT/$model/$task-run$run"
	vout="$META_ROOT/$model/$task-run$run.verify"
	stamp="$META_ROOT/$model/$task-run$run.start"
	setup_task "$task" "$dir" "$pristine"
	budget="${TASK_TIMEOUT:-$(task_conf "$task" BUDGET)}"
	[[ "$budget" =~ ^[0-9]+$ ]] || die "bad budget for $task: '$budget'"
	echo "=== [$model/$task run $run/$RUNS] running (timeout ${budget}s, log: $dir/agent.log) ==="
	: >"$stamp"
	start=$(date +%s)
	if (( ORACLE )); then
		local sol="$B12_DIR/tasks/$task/${ORACLE_SOLUTION:-reference}"
		if [[ ! -d "$sol" ]]; then
			note="no ${ORACLE_SOLUTION} for $task: reference used; "
			sol="$B12_DIR/tasks/$task/reference"
		fi
		echo "oracle solution: ${sol#"$B12_DIR"/}" >"$dir/agent.log"
		apply_solution "$sol" "$dir" >>"$dir/agent.log" 2>&1 || rc=1
	else
		local prompt
		prompt=$(cat "$B12_DIR/tasks/$task/prompt.md")
		projdir="$QWEN_PROJECTS/$(qwen_project "$dir")"
		mkdir -p "$projdir"
		sandbox_cmd "$dir" "$projdir"
		(cd "$dir" && OPENAI_MODEL="$model" AIRAG_SOURCE=bench timeout -k 30 "$budget" \
			"${SBX[@]}" "$QWEN_BIN" -m "$model" --approval-mode yolo -p "$prompt" \
			>"$dir/agent.log" 2>&1) || rc=$?
	fi
	dur=$(( $(date +%s) - start ))
	# verify on whatever is left (TIMEOUT is scored too); agent.log stays out of it
	run_verify "$task" "$dir" "$pristine" "$vout" || vrc=$?
	if (( vrc > 1 )); then verdict=VERIFY-ERROR; verr=1; note="${note}bench bug, see $vout"
	elif (( rc == 124 || (rc == 137 && dur >= budget) )); then verdict=TIMEOUT
	elif (( vrc == 0 )); then verdict=PASS
	else verdict=FAIL; fi
	score=$(score_of "$vout")
	pe=$(sed -n 's/^PREFIX_END //p' "$vout" | tail -n1); pe=${pe:--}
	if (( ! verr )) && [[ $verdict != PASS ]]; then
		local f; f=$(failed_items "$vout")
		note="${note}${f:+failed: $f}"
	fi
	if (( ! ORACLE )); then
		if (( rc != 0 && rc != 124 && rc != 137 )); then   # e.g. HTTP 400 context overflow -> rc 1
			local apierr; apierr=$(grep -m1 -oE '\[API Error: [^]]{0,120}' "$dir/agent.log" || true)
			note="agent rc=$rc${apierr:+ ($apierr)}; ${note}"
		fi
		(( SANDBOX_OK )) || note="UNSANDBOXED; ${note}"
		local tr=() others=""
		if [[ -d "$projdir/chats" ]]; then
			mapfile -t tr < <(find "$projdir/chats" -name '*.jsonl' -newer "$stamp")
		fi
		if (( ${#tr[@]} == 0 )); then
			note="NO TRANSCRIPT (peek check impossible); ${note}"
		else
			if grep -qsE "$PEEK_RE" "$dir/agent.log" "${tr[@]}"; then
				note="PEEKED at bench internals; ${note}"
			fi
			# the transcript always carries the run's own cwd: report OTHER run dirs only
			others=$(grep -ohE "${WORKROOT_RE}/[^/\"' ]+/[^/\"' ]+-run[0-9]+" "${tr[@]}" \
				| sort -u | grep -vxF "$dir" | paste -sd, - || true)
			[[ -n "$others" ]] && note="READ OTHER RUN ($others); ${note}"
		fi
	fi
	if [[ $verdict != PASS && $verdict != VERIFY-ERROR ]] && sig=$(server_died "$dir/agent.log"); then
		lost=1; score=""; note="SERVER DIED ($sig); run not scored"
		((LOST++)) || true
	fi
	note="${note%; }"
	echo "=== [$model/$task run $run/$RUNS] $verdict (${dur}s) ${score:+[$score] }${note:+— $note}"
	RESULTS+=("$model|$task|$run|$verdict|${score:--}|${dur}|$pe|$note")
	printf '%s\n' "$model|$task|$run|$verdict|${score:--}|${dur}|$pe|$note" >>"$META_ROOT/results.psv"
	if ! server_alive; then
		SERVER_DEAD=1
		echo "!!! llama-server does not answer /health after $model/$task run $run — remaining runs of this model are SKIPPED; the next switch_model restarts it" >&2
	fi
	if (( ! lost && ! verr )); then
		local iname iv
		while read -r _ iname iv _; do
			[[ -n "${AGG_TOT["$task|$iname"]+x}" ]] || ITEMS_SEEN+=("$task|$iname")
			AGG_TOT["$model|$task|$iname"]=$(( ${AGG_TOT["$model|$task|$iname"]:-0} + 1 ))
			AGG_TOT["$task|$iname"]=1
			if [[ $iv == PASS ]]; then
				AGG_PASS["$model|$task|$iname"]=$(( ${AGG_PASS["$model|$task|$iname"]:-0} + 1 ))
			fi
		done < <(grep -E '^ITEM ' "$vout" || true)
	fi
}

summary() {
	local r m t run v s d pe n npass=0 nfail=0 ntmo=0 nverr=0 tsv key
	SUMMARY_DONE=1
	echo
	echo "=============================== SUMMARY ==============================="
	printf '%-20s %-10s %-4s %-12s %-14s %-7s %s\n' MODEL TASK RUN VERDICT SCORE TIME NOTE
	tsv=$(mktemp /tmp/bench12-results.XXXXXX.tsv)
	# shellcheck disable=SC2064
	trap "command rm -f '$tsv'; trap - RETURN" RETURN
	for r in "${RESULTS[@]}"; do
		IFS='|' read -r m t run v s d pe n <<<"$r"
		printf '%-20s %-10s %-4s %-12s %-14s %-7s %s\n' "$m" "$t" "$run" "$v" "$s" "${d}s" "$n"
		if [[ $v == VERIFY-ERROR ]]; then ((nverr++)) || true; continue; fi
		[[ $n == "SERVER DIED"* || $v == SKIPPED ]] && continue
		case "$v" in
			PASS) ((npass++)) || true ;;
			FAIL) ((nfail++)) || true ;;
			TIMEOUT) ((ntmo++)) || true ;;
		esac
		# model task family N verdict k n prefix_end
		local nn=""; nn=$(task_conf "$t" N 2>/dev/null) || true
		local kk="" tot=""
		if [[ $s =~ ^([0-9]+)/([0-9]+) ]]; then kk=${BASH_REMATCH[1]}; tot=${BASH_REMATCH[2]}; fi
		printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$t" "$(task_conf "$t" FAMILY)" "${nn:--}" "$v" "${kk:--}" "${tot:--}" "${pe:--}" >>"$tsv"
	done
	echo
	echo "PASS $npass  FAIL $nfail  TIMEOUT $ntmo  VERIFY-ERROR $nverr  —  $LOST runs lost to server death (not model failures)"
	if (( LOST )); then
		echo "Lost = 'SERVER DIED' and SKIPPED rows above; rerun them, do not count them as FAIL."
	fi
	if (( nverr )); then
		echo "VERIFY-ERROR = bench bug (see the .verify file); excluded from every count and fit — rerun after the fix."
	fi
	echo
	python3 - "$tsv" <<'PYEOF' || echo "!!! per-task table / horizon fit failed (python error above)" >&2
import math, sys
from collections import defaultdict
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
by = defaultdict(list)
for m, t, fam, N, v, k, n, pe in rows:
    by[(m, t)].append((v, k, n, fam, N))
print("============ PER MODEL x TASK (pass rate, mean partial score) ============")
print(f"{'MODEL':20} {'TASK':10} {'RUNS':>4} {'PASS':>6} {'RATE':>6} {'MEAN k/n':>9}")
models = sorted({m for m, _ in by})
for m in models:
    tot_pass = tot_runs = 0; part = []
    for (mm, t), lst in sorted(by.items()):
        if mm != m:
            continue
        p = sum(1 for v, *_ in lst if v == "PASS")
        fr = [int(k) / int(n) for v, k, n, *_ in lst if k != "-" and n != "-" and int(n) > 0]
        mean = sum(fr) / len(fr) if fr else float("nan")
        tot_pass += p; tot_runs += len(lst); part += fr
        print(f"{m:20} {t:10} {len(lst):4d} {p:6d} {100*p/len(lst):5.0f}% {100*mean:8.1f}%")
    if tot_runs:
        print(f"{m:20} {'ALL':10} {tot_runs:4d} {tot_pass:6d} {100*tot_pass/tot_runs:5.0f}% "
              f"{100*sum(part)/max(len(part),1):8.1f}%")
print()
print("============ HORIZON FIT (chain family: per-step reliability p) ============")
print("p_prefix = MLE from correct-prefix lengths (each step survives w.p. p); a run whose")
print("           first unmatched step was never WRITTEN (timeout / write-at-the-end) is")
print("           right-censored there, not a step failure;")
print("p_succ   = binomial MLE of success rate s_N = p^N over all N (n/a when every cell")
print("           is 0/n or n/n: pass rates alone carry no information about p then);")
print("N_50     = steps a model chains at 50 % success = ln 0.5 / ln p_prefix.")
print("Fewer than 10 runs per N: neither p is worth reading (flagged LOW-N).")
for m in models:
    ch = [(int(N), v, int(k), int(n), pe) for mm, t, fam, N, v, k, n, pe in rows
          if mm == m and fam == "chain" and k != "-" and N != "-"]
    if not ch:
        continue
    S = sum(k for _, _, k, _, _ in ch)                                    # steps survived
    F = sum(1 for _, _, k, n, pe in ch if k < n and pe != "missing")      # first-failure events
    cens = sum(1 for _, _, k, n, pe in ch if k < n and pe == "missing")   # censored runs
    p_pref = S / (S + F) if S + F else float("nan")
    per_n = defaultdict(list)
    for N, v, k, n, _ in ch:
        per_n[N].append(v == "PASS")
    cells = []; xs = []
    for N in sorted(per_n):
        cells.append(f"N={N}: {sum(per_n[N])}/{len(per_n[N])}")
        xs.append((N, sum(per_n[N]), len(per_n[N])))

    def ll(p):
        return sum(x * N * math.log(p) + (n - x) * math.log(max(1e-300, 1 - p ** N)) for N, x, n in xs)
    if all(x == 0 for _, x, _ in xs) or all(x == n for _, x, n in xs):
        p_succ = float("nan")
    else:
        lo, hi = 1e-6, 1 - 1e-9
        for _ in range(200):                # golden-section maximisation of ll on (0,1)
            a = hi - (hi - lo) / 1.618034; b = lo + (hi - lo) / 1.618034
            if ll(a) < ll(b):
                lo = a
            else:
                hi = b
        p_succ = (lo + hi) / 2
    n50 = (math.log(0.5) / math.log(p_pref) if 0 < p_pref < 1
           else (0.0 if p_pref == 0 else float("inf")))
    fmt = lambda x: "n/a" if x != x else f"{x:.3f}"
    low = "  LOW-N" if min(len(v) for v in per_n.values()) < 10 else ""
    print(f"{m:20} {'  '.join(cells)}  runs={len(ch)} censored={cens}  "
          f"p_prefix={fmt(p_pref)}  p_succ={fmt(p_succ)}  N_50={n50:.1f}{low}")
PYEOF
	echo
	echo "=========== ITEM-COMPLIANCE MATRIX (held / runs) ==========="
	printf '%-20s %-10s %-16s %s\n' MODEL TASK ITEM HELD
	for m in ${MODELS//,/ }; do
		for key in "${ITEMS_SEEN[@]}"; do
			t=${key%%|*}; r=${key#*|}
			[[ -n "${AGG_TOT["$m|$t|$r"]:-}" ]] || continue
			printf '%-20s %-10s %-16s %s/%s\n' "$m" "$t" "$r" "${AGG_PASS["$m|$t|$r"]:-0}" "${AGG_TOT["$m|$t|$r"]}"
		done
	done
	echo
	echo "Read with research/23: items holding in SOME runs = stochastic compliance;"
	echo "items failing in ALL runs = a hard gap on that axis."
}

# ------------------------------------------------------------------ main ----

case "${1:-}" in
	-h|--help) help ;;
	--selftest) selftest; exit $? ;;
	--oracle) ORACLE=1; MODELS="oracle" ;;
	"") ;;
	*) die "unknown argument: $1 (see --help)" ;;
esac
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"
if (( ! ORACLE )); then
	command -v "$QWEN_BIN" >/dev/null || die "binary not found: $QWEN_BIN"
	command -v "$AILLAMA_BIN" >/dev/null || die "binary not found: $AILLAMA_BIN"
	eval "$("$AILLAMA_BIN" env | grep '^export')"
fi
for task in ${TASKS//,/ }; do [[ -d "$B12_DIR/tasks/$task" ]] || die "unknown task: $task"; done
if (( ORACLE )) && [[ -z "$WORKROOT_SET" ]]; then
	# never leave reference solutions in the default WORKROOT next to later real runs
	WORKROOT=$(mktemp -d /tmp/bench12-oracle.XXXXXX)
fi
mkdir -p "$WORKROOT"
WORKROOT="$(cd "$WORKROOT" && pwd -P)"    # physical path: qwen transcripts record it
META_ROOT="${WORKROOT%/}.meta"
mkdir -p "$META_ROOT"
WORKROOT_RE=$(printf '%s' "${WORKROOT%/}" | sed 's/[][\.*^$+?(){}|]/\\&/g')
if [[ "$B12_OVERWRITE" != 1 ]]; then
	for model in ${MODELS//,/ }; do
		[[ ! -e "$WORKROOT/$model" ]] || die "$WORKROOT/$model exists (an earlier run; setup would wipe its evidence) — pick another WORKROOT or set B12_OVERWRITE=1"
	done
fi
sandbox_init
if (( ! ORACLE )) && [[ -f "$HOME/.qwen/QWEN.md" ]]; then
	# the user's global qwen memory is part of the measured environment (research/25 B12-09)
	echo "agent env: ~/.qwen/QWEN.md sha256 $(sha256sum "$HOME/.qwen/QWEN.md" | cut -c1-16), sandbox=$SANDBOX_OK"
fi
printf '# %s start MODELS=%s TASKS=%s RUNS=%s sandbox=%s\n' "$(date -Is)" "$MODELS" "$TASKS" "$RUNS" "$SANDBOX_OK" >>"$META_ROOT/results.psv"
# a die mid-night (switch failure, /health timeout) still prints what was measured
on_exit() {
	local rc=$?
	if (( rc != 0 && ! SUMMARY_DONE && ${#RESULTS[@]} > 0 )); then
		echo "!!! aborted (rc=$rc) — partial summary follows (rows also in $META_ROOT/results.psv)" >&2
		summary || true
	fi
}
trap on_exit EXIT

for model in ${MODELS//,/ }; do
	switch_model "$model"
	for task in ${TASKS//,/ }; do
		for run in $(seq 1 "$RUNS"); do
			if (( SERVER_DEAD )); then
				echo "=== [$model/$task run $run/$RUNS] SKIPPED — server dead, run not started"
				RESULTS+=("$model|$task|$run|SKIPPED|-|-|-|server dead; run not started")
				printf '%s\n' "$model|$task|$run|SKIPPED|-|-|-|server dead; run not started" >>"$META_ROOT/results.psv"
				((LOST++)) || true
				continue
			fi
			run_task "$model" "$task" "$run"
		done
	done
done
summary
