#!/bin/bash
# bench_08_horizon.sh — controllable horizon p, context-fill decay p(fill) and
# vote-recoverability of a small model (design: memory project_bench08_design,
# 2026-07-18, refinements a-d). Part of the bench_NN_* series. NOTE: the name
# bench_08_spec_offload.sh was already taken (drafter bench) — this is the
# second bench_08, always called bench_08_horizon in BENCH.md.
#
# Why: bench_05/07 give pass rates per TASK; they cannot say how far a model
# can be trusted along a chain of dependent steps or how much of its context
# window is usable. This bench turns that into three numbers per model:
#
#   p        per-step reliability on a chain of N DEPENDENT steps, at a FIXED
#            context fill (factorial N x fill grid: refinement 07-18b — raising
#            N must not also raise fill, or p and fill decay contaminate each
#            other). Two estimates: p_step(direct) judges every step against
#            the model's OWN previous value (N trials per request), and
#            p_chain(MLE) fits chain success = p^N. Horizon N_max(S)=ln S/ln p.
#   p(fill)  the same per-step p at increasing prompt length L of NEUTRAL
#            ballast (foreign benign prose — llama.cpp docs + Gentoo news,
#            frozen once into $OUT/ballast/readme.txt). p0 = plateau at L=0
#            (measured, never assumed to be 1), L_eff = the fill where p drops
#            DELTA below p0, post-knee slope (refinement 07-18c). The task
#            sits at the END (optimistic bound) unless POSITION=start.
#            BALLAST=distractor (synthetic worked chains) is the sensitivity
#            check. Chain trials hit the RAW llama-server (/v1/chat/
#            completions), never qwen-code: its autoCompact 0.6 + truncation
#            would measure the harness's compaction policy, not the model.
#   vote     the nanoeuler ebuild polygon (class-B calibration task: the
#            committed-binary `emake` trap): VOTE_K independent headless
#            qwen-code create passes, each verified by bench08/qwen-ebuild-
#            oracle.sh (copy of ~/QwenCode/qwen-ebuild-run.sh: frozen local
#            mirror, pin gate, machine-readable gates). Per stage (header,
#            inherit, keywords, pin, compile, test, install) the proposal is a
#            coarse APPROACH label (b08.py features); majority-of-k either
#            lands on a verified-correct approach (class A: noise, voting
#            helps) or on a modal WRONG one (class B: voting amplifies).
#            ARBITER_K>0 adds arm C: the oracle's FAIL lines are fed back for
#            up to K fix iterations (iterations-to-pass = loop budget).
#
# Upstream pin: JustVugg/nanoeuler has no tags; the task ebuild must set
# EGIT_COMMIT=$PIN (gate `pin`) and the oracle's reference + the Portage
# build both read the local bare mirror $REF_REPO, so difficulty cannot
# drift between reps.
#
# Every trial is written as one JSONL (proposal, verified_outcome) row:
# $OUT/runs/<TAG>/chain.jsonl and vote.jsonl — usable as GRPO/distillation
# data. Server death (bench_07 rules): a vote attempt whose agent.log has an
# undici transport error, or a chain request that cannot connect, is LOST
# (not scored, counted separately) and the model's remaining work is skipped.
#
# Never pool POSITION/BALLAST populations: the report groups by them.
set -euo pipefail

MODELS="${MODELS:-gpt-oss20b-q8_0}"
PARTS="${PARTS:-chain,vote}"
NS="${NS:-1,2,4,8,16}"
FILLS="${FILLS:-0,8192,32768,65536,98304}"
REPS="${REPS:-10}"
BALLAST="${BALLAST:-readme}"
POSITION="${POSITION:-end}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
REQ_TIMEOUT="${REQ_TIMEOUT:-900}"
TEMP="${TEMP:-}"
DELTA="${DELTA:-0.05}"
VOTE_K="${VOTE_K:-5}"
VOTE_TIMEOUT="${VOTE_TIMEOUT:-1800}"
ARBITER_K="${ARBITER_K:-0}"
NO_SWITCH="${NO_SWITCH:-0}"
# B08_OUT, not OUT: the night queue lib exports OUT=llm/night, which sent REF_REPO and the runs
# there (q25b 10-16: "reference mirror missing", bench_08 never ran)
OUT="${B08_OUT:-/mnt/cc/strata/bench08}"
TAG="${TAG:-$(date +%Y%m%d-%H%M)}"
PIN="${PIN:-4cd90a51476266343c17331c52c1dd24b1b2e4a0}"
REF_REPO="${REF_REPO:-$OUT/nanoeuler.git}"
REPO_URL="${REPO_URL:-https://github.com/JustVugg/nanoeuler}"
OVERLAY="${OVERLAY:-/home/pwas/QwenCode/pwr}"
CATPN="${CATPN:-app-misc/nanoeuler}"
AILLAMA_BIN="${AILLAMA_BIN:-aillama}"
QWEN_BIN="${QWEN_BIN:-qwen}"
export QWEN_CODE_SUPPRESS_YOLO_WARNING=1
# Task prompts run under the "bench" airag identity (AIRAG_SOURCE=bench at the
# QWEN_BIN call), which ~/.qwen/hooks/airag-inject.sh skips — as in bench_07.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B08="$HERE/bench08/b08.py"
ORACLE="$HERE/bench08/qwen-ebuild-oracle.sh"
RUNDIR="$OUT/runs/$TAG"
PKGDIR="$OVERLAY/$CATPN"
EBUILD="$PKGDIR/$(basename "$CATPN")-9999.ebuild"

die() { echo "ERROR: $*" >&2; exit 1; }

# Same helper as bench_05/bench_07 — keep them identical.
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
Usage: ${0##*/} [-h] [--calibrate] [--report RUNDIR...]

Per aillama profile in MODELS: (chain) dependent-step chain trials over an
NS x FILLS grid against the raw llama-server -> horizon p, p(fill), p0,
L_eff; (vote) VOTE_K headless qwen-code nanoeuler-ebuild attempts verified
by the bench_08 oracle copy -> per-stage majority-of-k class A/B. All rows
go to JSONL under \$OUT/runs/\$TAG; the summary is printed and saved there.

  --calibrate   oracle self-test, no LLM: a known-good pinned ebuild must
                pass every gate, the committed-binary variant must fail
                exactly build,installed,noop,realcompile (passwordless sudo)
  --report DIR  re-print the summary from existing run directories

The server is left on the LAST profile in MODELS. NO_SWITCH=1 never
switches: it uses the running server and MODELS must be its profile.

Environment variables (current values):
  MODELS        aillama profiles               ($MODELS)
  PARTS         chain,vote                     ($PARTS)
  NS            chain lengths N                ($NS)
  FILLS         prompt fills, tokens           ($FILLS)
  REPS          chains per (N, fill)           ($REPS)
  BALLAST       readme|distractor (list)       ($BALLAST)
  POSITION      end|start (task vs ballast)    ($POSITION)
  MAX_TOKENS    generation cap per request     ($MAX_TOKENS)
  REQ_TIMEOUT   seconds per chain request      ($REQ_TIMEOUT)
  TEMP          temperature (empty = profile)  (${TEMP:-profile default})
  DELTA         p drop that defines L_eff      ($DELTA)
  VOTE_K        ebuild attempts per model      ($VOTE_K)
  VOTE_TIMEOUT  seconds per qwen-code pass     ($VOTE_TIMEOUT)
  ARBITER_K     fix iterations fed oracle FAILs ($ARBITER_K; 0 = off)
  NO_SWITCH     1 = never switch the server    ($NO_SWITCH)
  B08_OUT / TAG results root / run name        ($OUT / $TAG)
  PIN           nanoeuler commit               ($PIN)
  REF_REPO      local bare mirror              ($REF_REPO)
  OVERLAY       qwen ebuild overlay            ($OVERLAY; $CATPN is backed up and restored)
EOF
	exit 0
}

# ------------------------------------------------------------- plumbing ----
# (server_alive / wait_health / switch_model as in bench_07)

endpoint() {
	"$AILLAMA_BIN" env | sed -n 's/^export OPENAI_BASE_URL="\(.*\)\/v1".*/\1/p'
}

server_alive() {
	local base
	base=$(endpoint)
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

served_alias() {
	curl -sf -m 10 "$(endpoint)/v1/models" \
		| python3 -I -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
}

switch_model() {
	local model="$1" swlog alias
	if (( NO_SWITCH )); then
		server_alive || die "NO_SWITCH=1 but no server answers /health"
		alias=$(served_alias) || die "cannot read the served model alias"
		[[ "$alias" == "$model" ]] || die "NO_SWITCH=1: server runs '$alias', not '$model'"
		echo "--- using the running llama-server (profile $model), no switch ---"
		SERVER_DEAD=0
		return
	fi
	echo "--- switching llama-server to profile: $model ---"
	swlog=$(mktemp /tmp/bench08-switch.XXXXXX.log)
	"$AILLAMA_BIN" switch "$model" >"$swlog" 2>&1 || {
		tail -5 "$swlog" >&2
		die "aillama switch $model failed (full log: $swlog)"
	}
	wait_health
	SERVER_DEAD=0
}

# Freeze the neutral ballast once; the sha256 identifies the population.
ensure_ballast() {
	local f="$OUT/ballast/readme.txt"
	if [[ ! -s "$f" ]]; then
		mkdir -p "$OUT/ballast"
		local -a src=()
		mapfile -t src < <( { find /mnt/cc/strata/llama.cpp/docs -name '*.md' 2>/dev/null
			find /var/db/repos/gentoo/metadata/news -name '*.en.txt' 2>/dev/null; } | LC_ALL=C sort)
		(( ${#src[@]} )) || die "no ballast sources (llama.cpp docs / gentoo news) to build $f"
		cat "${src[@]}" >"$f.tmp" && mv "$f.tmp" "$f"
		sha256sum "$f" | cut -d' ' -f1 >"$OUT/ballast/readme.sha256"
		printf '%s\n' "${src[@]}" >"$OUT/ballast/readme.sources"
		echo "--- ballast frozen: $f ($(wc -c <"$f") bytes, ${#src[@]} files)"
	fi
	[[ "$(sha256sum "$f" | cut -d' ' -f1)" == "$(cat "$OUT/ballast/readme.sha256")" ]] \
		|| die "ballast $f changed since it was frozen (sha256 mismatch) — new population; move it aside deliberately"
	BALLAST_FILE="$f"
}

ensure_mirror() {
	[[ -d "$REF_REPO" ]] || die "reference mirror missing: git clone --bare $REPO_URL $REF_REPO"
	[[ "$(git -C "$REF_REPO" cat-file -t "$PIN" 2>/dev/null)" == commit ]] \
		|| die "pin $PIN not present in $REF_REPO"
}

# Back up the shared qwen overlay package dir once; restored on EXIT. The
# oracle convention is `rm -rf $PKGDIR` before every create pass.
PKG_BACKUP=""
backup_pkgdir() {
	[[ -n "$PKG_BACKUP" ]] && return
	PKG_BACKUP="$RUNDIR/pkgdir-backup"
	mkdir -p "$PKG_BACKUP"
	if [[ -d "$PKGDIR" ]]; then
		command cp -a "$PKGDIR/." "$PKG_BACKUP/"
		echo "--- backed up $PKGDIR -> $PKG_BACKUP (restored on exit)"
	fi
	trap restore_pkgdir EXIT
}
restore_pkgdir() {
	[[ -n "$PKG_BACKUP" ]] || return 0
	command rm -rf "${PKGDIR:?}"
	if [[ -n "$(ls -A "$PKG_BACKUP" 2>/dev/null)" ]]; then
		mkdir -p "$PKGDIR" && command cp -a "$PKG_BACKUP/." "$PKGDIR/"
	fi
	echo "--- restored $PKGDIR from $PKG_BACKUP"
}

# ----------------------------------------------------------------- vote ----

create_prompt() {
	echo "Create and validate the Gentoo ebuild for $REPO_URL as $CATPN-9999 in the overlay $OVERLAY. Pin the source to upstream commit $PIN with EGIT_COMMIT (do not track a branch). Follow the gentoo-ebuild skill end to end, including pkgcheck and real Portage build phases."
}

fix_prompt() { # $1 = gates file
	echo "The ebuild at $EBUILD was checked by an independent verifier and FAILED these gates:"
	grep '|FAIL|' "$1" | sed 's/^\([^|]*\)|FAIL|/- \1: /'
	echo "Fix the ebuild in place so that every gate passes. Keep EGIT_COMMIT=$PIN. Re-run pkgcheck and the real Portage build phases, then stop."
}

# verify_attempt DIR ITER -> rc 0 = all gates passed; gates in DIR/gates-ITER
verify_attempt() {
	local dir="$1" it="$2" rc=0
	local vdir="$dir/verify-$it"
	command rm -rf "${vdir:?}"; mkdir -p "$vdir"
	: >"$dir/gates-$it"
	QWEN_VERIFY_ONLY=1 QWEN_EBUILD_OVERLAY="$OVERLAY" QWEN_WORKROOT="$vdir" \
		QWEN_GATES_FILE="$dir/gates-$it" BENCH08_PIN="$PIN" BENCH08_REF_REPO="$REF_REPO" \
		bash "$ORACLE" "$REPO_URL" "$CATPN" >"$vdir/oracle.log" 2>&1 || rc=$?
	# keep build.log (in $vdir), drop the root-owned Portage build tree
	sudo -n rm -rf "${vdir:?}/ptmp/portage" 2>/dev/null || true
	return $rc
}

qwen_pass() { # $1 model, $2 log, $3 prompt
	(cd "$OVERLAY" && OPENAI_MODEL="$1" AIRAG_SOURCE=bench timeout "$VOTE_TIMEOUT" \
		"$QWEN_BIN" -m "$1" --approval-mode yolo -p "$3" >"$2" 2>&1)
}

vote_attempt() {
	local model="$1" a="$2" rc=0 start dur it status sig verdict failed eb
	local dir="$RUNDIR/vote/$model/attempt$a"
	command rm -rf "${dir:?}"; mkdir -p "$dir"
	command rm -rf "${PKGDIR:?}"
	echo "=== [$model/vote attempt $a/$VOTE_K] create pass (timeout ${VOTE_TIMEOUT}s, log: $dir/agent-0.log) ==="
	start=$(date +%s)
	qwen_pass "$model" "$dir/agent-0.log" "$(create_prompt)" || rc=$?
	it=0
	while :; do
		status=scored; eb=""
		if [[ -f "$EBUILD" ]]; then
			eb="$dir/ebuild-$it.ebuild"
			command cp -f "$EBUILD" "$eb"
			if verify_attempt "$dir" "$it"; then verdict=PASS; else verdict=FAIL; fi
		else
			verdict=FAIL; status=noebuild; : >"$dir/gates-$it"
		fi
		[[ $rc -eq 124 && $verdict != PASS ]] && verdict=TIMEOUT
		if [[ $verdict != PASS ]] && sig=$(server_died "$dir/agent-$it.log"); then
			status=lost; verdict="LOST (SERVER DIED: $sig)"; ((LOST++)) || true
		fi
		python3 -I "$B08" voterow --model "$model" --tag "$TAG" --attempt "$a" --iter "$it" \
			--status "$status" --ebuild "$eb" --gates "$dir/gates-$it" --out "$RUNDIR/vote.jsonl"
		dur=$(( $(date +%s) - start ))
		failed=$(grep '|FAIL|' "$dir/gates-$it" 2>/dev/null | cut -d'|' -f1 | paste -sd, || true)
		[[ $status == noebuild ]] && failed="no ebuild produced"
		echo "=== [$model/vote attempt $a/$VOTE_K iter $it] $verdict (${dur}s)${failed:+ — failed: $failed}"
		RESULTS+=("$model|vote|$a.$it|$verdict|${dur}s|$failed")
		if ! server_alive; then
			SERVER_DEAD=1
			echo "!!! llama-server does not answer /health after $model vote attempt $a — remaining work of this model is SKIPPED" >&2
			return
		fi
		[[ $verdict == PASS || $status != scored || $it -ge $ARBITER_K ]] && break
		it=$((it + 1)); rc=0
		echo "=== [$model/vote attempt $a/$VOTE_K] arbiter iteration $it: feeding the FAIL gates back"
		qwen_pass "$model" "$dir/agent-$it.log" "$(fix_prompt "$dir/gates-$((it - 1))")" || rc=$?
	done
}

# ----------------------------------------------------------- calibrate ----

calib_ebuild() { # $1 = good|bad (bad keeps the committed prebuilt binaries)
	cat <<'EOF'
# Copyright 2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

inherit git-r3 toolchain-funcs

DESCRIPTION="A GPT-2-class language model built entirely from scratch in C/CUDA"
HOMEPAGE="https://github.com/JustVugg/nanoeuler"
EGIT_REPO_URI="https://github.com/JustVugg/nanoeuler.git"
EOF
	echo "EGIT_COMMIT=\"$PIN\""
	cat <<'EOF'

LICENSE="MIT"
SLOT="0"
KEYWORDS=""

src_prepare() {
	default
EOF
	if [[ $1 == good ]]; then
		echo '	rm -f nanoeuler nanoeuler_check nanoeuler.bin || die'
	fi
	cat <<'EOF'
}

src_compile() {
	emake \
		CC="$(tc-getCC)" \
		CFLAGS="${CFLAGS} -fopenmp -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function" \
		nanoeuler nanoeuler_check
}

src_test() {
	./nanoeuler_check gradcheck || die "gradient check failed"
}

src_install() {
	dobin nanoeuler
	dodoc README.md
}
EOF
}

calibrate() {
	ensure_mirror
	sudo -n true 2>/dev/null || die "--calibrate needs passwordless sudo (the oracle's install phase)"
	local cdir="$OUT/calibrate/$TAG" ov kind rc fails want ok=1
	ov="$cdir/overlay"
	mkdir -p "$ov/metadata" "$ov/profiles" "$ov/$CATPN"
	chmod 755 "$OUT/calibrate" "$cdir"
	printf 'masters = gentoo\nthin-manifests = true\n' >"$ov/metadata/layout.conf"
	echo bench08calib >"$ov/profiles/repo_name"
	cat >"$ov/$CATPN/metadata.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE pkgmetadata SYSTEM "https://www.gentoo.org/dtd/metadata.dtd">
<pkgmetadata>
  <upstream>
    <remote-id type="github">JustVugg/nanoeuler</remote-id>
  </upstream>
</pkgmetadata>
EOF
	for kind in good bad; do
		calib_ebuild "$kind" >"$ov/$CATPN/$(basename "$CATPN")-9999.ebuild"
		rc=0
		mkdir -p "$cdir/$kind"; : >"$cdir/$kind/gates"
		QWEN_VERIFY_ONLY=1 QWEN_EBUILD_OVERLAY="$ov" QWEN_WORKROOT="$cdir/$kind" \
			QWEN_GATES_FILE="$cdir/$kind/gates" BENCH08_PIN="$PIN" BENCH08_REF_REPO="$REF_REPO" \
			bash "$ORACLE" "$REPO_URL" "$CATPN" >"$cdir/$kind/oracle.log" 2>&1 || rc=$?
		sudo -n rm -rf "${cdir:?}/${kind:?}/ptmp/portage" 2>/dev/null || true
		fails=$(grep '|FAIL|' "$cdir/$kind/gates" | cut -d'|' -f1 | LC_ALL=C sort | paste -sd, || true)
		# the committed prebuilt nanoeuler_check also fails gradcheck, so the
		# trap variant dies in src_test: build+installed fail with the trap gates
		want=""; [[ $kind == bad ]] && want="build,installed,noop,realcompile"
		if [[ "$fails" == "$want" ]]; then
			echo "CALIBRATE $kind: OK (oracle rc=$rc, failed gates: ${fails:-none})"
		else
			echo "CALIBRATE $kind: WRONG — expected failed gates '${want:-none}', got '${fails:-none}' (log $cdir/$kind/oracle.log)"
			ok=0
		fi
		if ! grep -q "file://$REF_REPO" "$cdir/$kind/build.log" 2>/dev/null; then
			echo "CALIBRATE $kind: WARNING — build.log shows no fetch from file://$REF_REPO (git-r3 override not applied?)"
		fi
	done
	(( ok )) || exit 1
	exit 0
}

# --------------------------------------------------------------- main ----

RESULTS=()
LOST=0
SERVER_DEAD=0

case "${1:-}" in
	-h|--help) help ;;
	--calibrate) calibrate ;;
	--report)
		shift; (( $# )) || die "--report needs run directories"
		ch=(); vo=()
		for d in "$@"; do ch+=("$d/chain.jsonl"); vo+=("$d/vote.jsonl"); done
		python3 -I "$B08" report --chain "${ch[@]}" --vote "${vo[@]}" --delta "$DELTA"
		exit 0 ;;
	"") ;;
	*) die "unknown argument: $1 (see -h)" ;;
esac

command -v "$AILLAMA_BIN" >/dev/null || die "binary not found: $AILLAMA_BIN"
[[ -f "$B08" && -f "$ORACLE" ]] || die "helpers missing under $HERE/bench08"
if [[ ",$PARTS," == *,vote,* ]]; then
	command -v "$QWEN_BIN" >/dev/null || die "binary not found: $QWEN_BIN"
	sudo -n true 2>/dev/null || die "vote part needs passwordless sudo (oracle install phase)"
	ensure_mirror
fi
if [[ ",$PARTS," == *,chain,* ]]; then ensure_ballast; fi
mkdir -p "$RUNDIR"
eval "$("$AILLAMA_BIN" env | grep '^export')"
{
	echo "bench_08_horizon $TAG  $(date -Is)"
	for v in MODELS PARTS NS FILLS REPS BALLAST POSITION MAX_TOKENS REQ_TIMEOUT TEMP DELTA VOTE_K VOTE_TIMEOUT ARBITER_K NO_SWITCH PIN REF_REPO; do
		printf '%s=%s\n' "$v" "${!v}"
	done
	if [[ -n "${BALLAST_FILE:-}" ]]; then echo "ballast_sha256=$(cat "$OUT/ballast/readme.sha256")"; fi
} >"$RUNDIR/params.txt"
echo "--- run dir: $RUNDIR"

for model in ${MODELS//,/ }; do
	switch_model "$model"
	curl -sf -m 10 "$(endpoint)/props" 2>/dev/null \
		| python3 -I -c 'import json,sys; d=json.load(sys.stdin); print("server_build['"$model"']=" + str(d.get("build_info","?")) + " n_ctx=" + str((d.get("default_generation_settings") or {}).get("n_ctx","?")))' \
		>>"$RUNDIR/params.txt" 2>/dev/null || true
	if [[ ",$PARTS," == *,chain,* ]]; then
		rc=0
		python3 -I "$B08" chain --base "$(endpoint)" --model "$model" --tag "$TAG" \
			--ns "$NS" --fills "$FILLS" --reps "$REPS" --ballast "$BALLAST" --position "$POSITION" \
			--corpus "$BALLAST_FILE" --max-tokens "$MAX_TOKENS" --req-timeout "$REQ_TIMEOUT" \
			${TEMP:+--temperature "$TEMP"} --out "$RUNDIR/chain.jsonl" || rc=$?
		if (( rc == 3 )); then
			SERVER_DEAD=1; ((LOST++)) || true
		elif (( rc )); then
			die "chain part failed for $model (rc=$rc)"
		fi
	fi
	if [[ ",$PARTS," == *,vote,* ]]; then
		backup_pkgdir
		for a in $(seq 1 "$VOTE_K"); do
			if (( SERVER_DEAD )); then
				echo "=== [$model/vote attempt $a/$VOTE_K] SKIPPED — server dead, attempt not started"
				RESULTS+=("$model|vote|$a|SKIPPED|-|server dead; attempt not started")
				((LOST++)) || true
				continue
			fi
			vote_attempt "$model" "$a"
		done
	fi
done

{
	echo
	echo "========================= bench_08_horizon SUMMARY ($TAG) =========================="
	if (( ${#RESULTS[@]} )); then
		printf '%-24s %-5s %-7s %-28s %-7s %s\n' "MODEL" "PART" "ATT.IT" "VERDICT" "TIME" "FAILED GATES"
		for r in "${RESULTS[@]}"; do
			IFS='|' read -r m p at v d n <<<"$r"
			printf '%-24s %-5s %-7s %-28s %-7s %s\n' "$m" "$p" "$at" "$v" "$d" "$n"
		done
	fi
	echo "$LOST unit(s) lost to server death (not model failures, not scored)"
	python3 -I "$B08" report --chain "$RUNDIR/chain.jsonl" --vote "$RUNDIR/vote.jsonl" --delta "$DELTA"
	echo
	echo "JSONL: $RUNDIR/chain.jsonl  $RUNDIR/vote.jsonl   params: $RUNDIR/params.txt"
} | tee "$RUNDIR/summary.txt"
