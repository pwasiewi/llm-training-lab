#!/bin/bash
# bench_10_research.sh — head-to-head of aillama profiles on the REAL aildr
# workload: LDR's own search loop (`aildr local`), local model only, no Claude
# in the path. Scores what deep research needs and what bench_05/bench_07 do
# not measure: factual correctness against an oracle, resistance to a confident
# wrong source, citation integrity, source diversity, and wall-clock cost.
#
# Each question carries its own oracle: MUST regexes (the correct answer) and
# MUST_NOT regexes (the tempting falsehood). Three of the ten are traps — the
# popular web answer is wrong, or the question itself contains a false premise.
#
# Models alternate order per question so search-engine throttling and
# time-of-day drift cannot favour whichever model runs first.

set -uo pipefail

MODELS="${MODELS:-ornith15-128k,ornith15-9b}"
ONLY="${ONLY:-}"                       # comma-separated question ids, empty = all
STRATEGY="${STRATEGY:-source-based}"
ITERATIONS="${ITERATIONS:-2}"
QPI="${QPI:-3}"                        # questions per iteration
TIMEOUT="${TIMEOUT:-1500}"
WORKROOT="${WORKROOT:-/var/tmp/bench10-$(date +%Y%m%d-%H%M)}"
AILDR="${AILDR:-$HOME/Claude/bin/aildr}"
AILLAMA="${AILLAMA:-$HOME/Claude/bin/aillama}"
RUN_LOG="${RUN_LOG:-$WORKROOT/bench_10.log}"

die() { echo "ERROR: $*" >&2; exit 1; }

help() {
	cat <<EOF
bench_10_research.sh — deep-research head-to-head on the aildr stack

  MODELS=a,b  ONLY=id1,id2  ITERATIONS=2  QPI=3  TIMEOUT=1500  WORKROOT=dir

Runs every question through \`aildr local\` once per model, scores each answer
against that question's oracle, and prints a per-model aggregate. Raw answers
are kept under WORKROOT/<model>/<id>.txt for a blind read afterwards.

Metrics per run: verdict (oracle), elapsed s, words, sources listed, unique
URLs, duplicate rate, citations used, orphan citations (a [n] with no source),
uncited sources.
EOF
	exit 0
}

# id @ topic @ MUST (;;-sep, ALL required) @ MUST_NOT (;;-sep, ANY fails) @ engine @ note
# '|' inside a field is regex alternation, so the field separator is '@'.
# The engine is pinned per question and identical for both models: open-web
# questions need the trap sources, factual anchors are cheaper and far more
# stable on wikipedia/arxiv, which do not CAPTCHA under load.
QUESTIONS=(
"tcpdefault@In mainline Linux 6.x, which TCP congestion control algorithm is the built-in default, and has BBR replaced CUBIC as the default?@cubic@bbr (is|became|has been|is now) (the )?default;;bbr replaced cubic;;default (is|to|changed to) bbr@searxng@TRAP: a forum post claims BBR is the default; it is not"
"gentoo@Why did the Gentoo project remove OpenRC as an init system in 2025?@(did not|no such|false premise|incorrect|still (supports|ships|offers|being used)|remains the|no evidence|cannot confirm|not (actually )?(contain|support|confirm|state|back)|never (explicitly )?state|no prior knowledge|does not support|question assumes|not (been )?removed|continues to be)@gentoo (removed|dropped|discontinued|abandoned) openrc@searxng@FALSE PREMISE: OpenRC was never removed"
"iouring@Is io_uring disabled by default in mainstream Linux distributions because of security vulnerabilities?@(google|chromeos|android|seccomp|docker|not disabled|remains enabled)@@searxng@NUANCE: Google disabled it in its own fleet, distros did not"
"pgvacuum@What does PostgreSQL's autovacuum_vacuum_scale_factor parameter default to and what does it control?@0\\.2@@searxng@docs anchor"
"fedora@Which filesystem does Fedora Workstation use by default, and since which release?@btrfs;;33@@wikipedia@factual anchor"
"rustlinux@In which mainline Linux kernel version was initial Rust support merged?@6\\.1@@wikipedia@factual anchor"
"armsoftbank@Which company acquired ARM Holdings in 2016 and for how much?@softbank;;(24\\.3|32)@@wikipedia@factual anchor"
"cudacores@How many CUDA cores does the NVIDIA GeForce RTX 5070 Ti have?@8960@@wikipedia@numeric anchor"
"gadgetbridge@Under which software licence is the Gadgetbridge Android app distributed?@(agpl|affero)@@wikipedia@factual anchor"
"mamba2@What is the main architectural contribution of Mamba-2 compared to Mamba-1?@(state space duality|ssd|duality)@@arxiv@scholarly"
)

SLEEP_BETWEEN="${SLEEP_BETWEEN:-60}"   # let engine suspensions lapse between runs
SEARCH_WAIT="${SEARCH_WAIT:-900}"      # max seconds to wait for SearXNG to recover
SEARXNG_URL="${SEARXNG_URL:-http://localhost:8081}"

# SearXNG recovers on its own after a burst (brave 429, duckduckgo/startpage
# CAPTCHA). Gate every open-web run on a control query so a suspended engine
# scores as the model's failure only when it really is one.
search_ok() {
	local n
	n=$(curl -s --max-time 30 "$SEARXNG_URL/search?q=linux+kernel&format=json" |
		python3 -c 'import sys,json
try: print(len(json.load(sys.stdin).get("results",[])))
except Exception: print(0)' 2>/dev/null)
	[[ "${n:-0}" -ge 10 ]]
}

wait_for_search() {
	local waited=0
	while ! search_ok; do
		if (( waited >= SEARCH_WAIT )); then
			echo "    WARNING: SearXNG still degraded after ${waited}s — running anyway" | tee -a "$RUN_LOG"
			return 0
		fi
		echo "    SearXNG degraded, waiting 120s (${waited}s so far)" | tee -a "$RUN_LOG"
		sleep 120; waited=$((waited + 120))
	done
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && help

# --rescore DIR... : apply the current oracle to answers already on disk, write
# DIR/results-rescored.tsv, print the aggregate. For oracle fixes after a run.
RESCORE=0
if [[ "${1:-}" == "--rescore" ]]; then
	RESCORE=1; shift
	(( $# )) || die "--rescore needs at least one WORKROOT"
fi
command -v "$AILDR" >/dev/null || die "not found: $AILDR"
command -v "$AILLAMA" >/dev/null || die "not found: $AILLAMA"

mkdir -p "$WORKROOT" || die "cannot create $WORKROOT"
trap 'rc=$?; echo "### ${0##*/} EXIT=$rc ###" | tee -a "$RUN_LOG"' EXIT

RESULTS="$WORKROOT/results.tsv"
printf 'model\tid\tverdict\telapsed_s\twords\tsources\tunique_urls\tdup_pct\tcites\torphans\tuncited\n' >"$RESULTS"

score_run() {
	# $1 raw answer, $2 must, $3 must_not, $4 model, $5 id, $6 elapsed -> full TSV row
	python3 - "$@" <<'PYSCORE'
import re, sys
path, must, must_not, model, qid, elapsed = sys.argv[1:7]
raw = open(path, encoding='utf-8', errors='replace').read()

idx = raw.rfind('\nSources:')
answer = raw[:idx] if idx > 0 else raw
srcblock = raw[idx:] if idx > 0 else ''

sources = re.findall(r'^\s*\[(\d+)\]\s+(.*?)\s+(https?://\S+)\s*$', srcblock, re.M)
urls = [u for _, _, u in sources]
uniq = len(set(urls))
dup = 0 if not urls else round(100 * (len(urls) - uniq) / len(urls))

cited = set(int(n) for n in re.findall(r'\[(\d+)\]', answer))
orphans = len([n for n in cited if n > len(sources)])
uncited = len([i for i in range(1, len(sources) + 1) if i not in cited])

def hits(spec):
    pats = [p for p in spec.split(';;') if p]
    return sum(1 for p in pats if re.search(p, answer, re.I)), len(pats)

def hits_in(spec, text):
    pats = [p for p in spec.split(';;') if p]
    return sum(1 for p in pats if re.search(p, text, re.I)), len(pats)

# MUST_NOT only counts in assertive sentences: not headings, not questions, and
# not sentences that negate ("no evidence that X", "X has not replaced Y").
NEG = re.compile(r"\b(not|no|none|nothing|neither|never|cannot|can't|isn't|hasn't|haven't|"
                 r"doesn't|didn't|don't|nor|without|false|incorrect|unsupported|unfounded|myth|"
                 r"assum\w*|premise|alleged|supposed|claim)\b", re.I)
sentences = re.split(r'(?<=[.!?])\s+|\n+', answer)
assertive = ' '.join(t for t in sentences
                     if t.strip() and not t.lstrip().startswith('#')
                     and not t.rstrip().endswith('?') and not NEG.search(t))

m_hit, m_tot = hits(must)
n_hit, _ = hits_in(must_not, assertive)

if n_hit:
    verdict = 'WRONG'
elif m_tot and m_hit == m_tot:
    verdict = 'CORRECT'
elif m_hit:
    verdict = 'PARTIAL'
else:
    verdict = 'MISS'

print('\t'.join(str(x) for x in [model, qid, verdict, elapsed, len(answer.split()),
                                 len(sources), uniq, dup, len(cited), orphans, uncited]))
PYSCORE
}

run_one() {
	local model="$1" id="$2" topic="$3" must="$4" mustnot="$5" engine="$6" attempt="${7:-1}"
	local dir="$WORKROOT/$model" out="$WORKROOT/$model/$id.txt" start dur tail_tsv
	mkdir -p "$dir"
	echo "=== [$model/$id] $(date '+%T') ===" | tee -a "$RUN_LOG"
	start=$(date +%s)
	timeout "$TIMEOUT" "$AILDR" local --strategy "$STRATEGY" --engine "$engine" \
		--iterations "$ITERATIONS" --questions "$QPI" --timeout "$TIMEOUT" \
		"$topic" >"$out" 2>"$dir/$id.err"
	dur=$(( $(date +%s) - start ))
	if [[ ! -s "$out" ]]; then
		printf '%s\t%s\tNOOUTPUT\t%s\t0\t0\t0\t0\t0\t0\t0\n' "$model" "$id" "$dur" >>"$RESULTS"
		echo "    NOOUTPUT (${dur}s)" | tee -a "$RUN_LOG"
		return
	fi
	if grep -qi "No sources were found" "$out"; then
		if (( attempt == 1 )); then
			echo "    NOSOURCE — search layer, not the model; waiting 300s and retrying once" | tee -a "$RUN_LOG"
			sleep 300; wait_for_search
			run_one "$model" "$id" "$topic" "$must" "$mustnot" "$engine" 2
			return
		fi
		printf '%s\t%s\tNOSOURCE\t%s\t0\t0\t0\t0\t0\t0\t0\n' "$model" "$id" "$dur" >>"$RESULTS"
		echo "    NOSOURCE (${dur}s) — excluded from the oracle score" | tee -a "$RUN_LOG"
		return
	fi
	row=$(score_run "$out" "$must" "$mustnot" "$model" "$id" "$dur")
	printf '%s\n' "$row" >>"$RESULTS"
	echo "    $(awk -F'\t' '{print $3}' <<<"$row") (${dur}s)" | tee -a "$RUN_LOG"
}

if (( RESCORE )); then
	for root in "$@"; do
		[[ -f "$root/results.tsv" ]] || { echo "skip $root: no results.tsv" >&2; continue; }
		out="$root/results-rescored.tsv"
		head -1 "$root/results.tsv" >"$out"
		tail -n +2 "$root/results.tsv" | while IFS=$'\t' read -r model id verdict elapsed _rest; do
			f="$root/$model/$id.txt"
			if [[ ! -s "$f" ]] || [[ "$verdict" == NOSOURCE || "$verdict" == NOOUTPUT ]]; then
				grep -P "^$model\t$id\t" "$root/results.tsv" >>"$out"; continue
			fi
			for rec in "${QUESTIONS[@]}"; do
				IFS='@' read -r qid _topic must mustnot _engine _note <<<"$rec"
				[[ "$qid" == "$id" ]] && { score_run "$f" "$must" "$mustnot" "$model" "$id" "$elapsed" >>"$out"; break; }
			done
		done
		echo "rescored: $out"
	done
	python3 - "${@/%//results-rescored.tsv}" <<'PYAGG'
import csv, sys, collections
rows = []
for p in sys.argv[1:]:
    rows += list(csv.DictReader(open(p), delimiter='\t'))
agg = collections.defaultdict(list)
for r in rows: agg[r['model']].append(r)
print(f"{'model':<16}{'CORRECT':>8}{'PARTIAL':>8}{'WRONG':>7}{'MISS':>6}{'med_s':>7}{'uniq_src':>9}{'orph':>6}")
for m, rs in agg.items():
    c = collections.Counter(r['verdict'] for r in rs)
    secs = sorted(int(r['elapsed_s']) for r in rs)
    sc = [r for r in rs if r['verdict'] not in ('NOSOURCE', 'NOOUTPUT')] or rs
    print(f"{m:<16}{c['CORRECT']:>8}{c['PARTIAL']:>8}{c['WRONG']:>7}{c['MISS']:>6}{secs[len(secs)//2]:>7}"
          f"{sum(int(r['unique_urls']) for r in sc)/len(sc):>9.1f}{sum(int(r['orphans']) for r in sc):>6}")
byq = collections.defaultdict(dict)
for r in rows: byq[r['id']].setdefault(r['model'], []).append(r['verdict'][:4])
for q, mv in byq.items():
    print(f"  {q:<14} " + "   ".join(f"{m}: {','.join(v)}" for m, v in mv.items()))
PYAGG
	exit 0
fi

echo "### bench_10 start $(date '+%F %T') models=$MODELS iterations=$ITERATIONS qpi=$QPI ###" | tee -a "$RUN_LOG"

qi=0
for rec in "${QUESTIONS[@]}"; do
	IFS='@' read -r id topic must mustnot engine note <<<"$rec"
	[[ -n "$ONLY" && ",$ONLY," != *",$id,"* ]] && continue
	qi=$((qi + 1))
	# alternate model order per question
	order="$MODELS"
	if (( qi % 2 == 0 )); then
		order="$(echo "$MODELS" | awk -F, '{for(i=NF;i>0;i--) printf "%s%s", $i, (i>1?",":"")}')"
	fi
	echo "--- question $qi: $id ($note) ---" | tee -a "$RUN_LOG"
	for model in ${order//,/ }; do
		"$AILLAMA" switch "$model" >>"$RUN_LOG" 2>&1 || { echo "switch $model failed" | tee -a "$RUN_LOG"; continue; }
		sleep 3
		[[ "$engine" == "searxng" ]] && wait_for_search
		run_one "$model" "$id" "$topic" "$must" "$mustnot" "$engine"
		sleep "$SLEEP_BETWEEN"
	done
done

echo | tee -a "$RUN_LOG"
echo "==================== PER-RUN ====================" | tee -a "$RUN_LOG"
column -t -s $'\t' "$RESULTS" | tee -a "$RUN_LOG"

echo | tee -a "$RUN_LOG"
echo "==================== PER-MODEL AGGREGATE ====================" | tee -a "$RUN_LOG"
python3 - "$RESULTS" <<'PY' | tee -a "$RUN_LOG"
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
agg = collections.defaultdict(list)
for r in rows:
    agg[r['model']].append(r)
print(f"{'model':<16}{'CORRECT':>8}{'PARTIAL':>8}{'WRONG':>7}{'MISS':>6}{'med_s':>7}{'tot_min':>8}{'src':>6}{'dup%':>6}{'orph':>6}")
for m, rs in agg.items():
    n = len(rs)
    c = sum(1 for r in rs if r['verdict'] == 'CORRECT')
    p = sum(1 for r in rs if r['verdict'] == 'PARTIAL')
    w = sum(1 for r in rs if r['verdict'] == 'WRONG')
    mi = sum(1 for r in rs if r['verdict'] in ('MISS', 'NOOUTPUT'))
    secs = sorted(int(r['elapsed_s']) for r in rs)
    med = secs[len(secs) // 2] if secs else 0
    tot = sum(secs) / 60
    src = sum(int(r['sources']) for r in rs) / n
    dup = sum(int(r['dup_pct']) for r in rs) / n
    orph = sum(int(r['orphans']) for r in rs)
    print(f"{m:<16}{c:>8}{p:>8}{w:>7}{mi:>6}{med:>7}{tot:>8.0f}{src:>6.1f}{dup:>6.0f}{orph:>6}")
print(f"\n{len(rows)} runs. CORRECT = every MUST anchor present and no MUST_NOT.")
print("WRONG = asserted the falsehood a trap question is built on.")
PY

echo "raw answers: $WORKROOT/<model>/<id>.txt" | tee -a "$RUN_LOG"
