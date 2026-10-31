#!/usr/bin/env bash
# qwen ebuild workflow driver: create -> review -> verify, headless.
#
# bench_08 COPY of ~/QwenCode/qwen-ebuild-run.sh (2026-10-16). Never edit the
# original from bench_08; differences from it, all marked "bench08:":
#   - the committed-binary reference clone comes from a LOCAL bare mirror
#     pinned at BENCH08_PIN (BENCH08_REF_REPO), not a live `git clone` of
#     upstream, so the oracle's ground truth is frozen with the task;
#   - the Portage build fetches from the same mirror via git-r3's
#     EGIT_OVERRIDE_REPO_<NAME> (offline, identical bytes);
#   - extra gate `pin`: EGIT_COMMIT must equal BENCH08_PIN (and no
#     EGIT_BRANCH) — the task ebuild is pinned by commit;
#   - every gate is also written machine-readable to QWEN_GATES_FILE as
#     `gate|PASS/FAIL/SKIP|note` (bench_08 builds its JSONL from it);
#   - WORKROOT can be set (QWEN_WORKROOT) so logs land under the bench dir;
#   - the installed-binary smoke run is capped with `timeout 20`.
#
# Runs the small local model (via `qwen -p`) twice — a create pass and a
# dedicated self-review pass that is handed the exact ebuild path — then
# VERIFIES the artifact independently of anything the model claimed. The
# verifier is the sole source of truth: mechanical header/inherit/KEYWORDS
# checks plus a real Portage build gate that distinguishes a genuine compile
# from the committed-binary trap.
#
# Usage: qwen-ebuild-run.sh [REPO_URL] [CATEGORY/PN]
#   defaults: JustVugg/nanoeuler  app-misc/nanoeuler   (the lab polygon)
#
# Exit: 0 only if every verify gate passes. Non-zero + a per-gate report
# otherwise. Needs passwordless sudo for the install phase.
set -uo pipefail

REPO_URL="${1:-https://github.com/JustVugg/nanoeuler}"
CATPN="${2:-app-misc/nanoeuler}"
OVERLAY="${QWEN_EBUILD_OVERLAY:-/home/pwas/QwenCode/pwr}"
PV="${QWEN_EBUILD_PV:-9999}"
QWEN_TIMEOUT="${QWEN_TIMEOUT:-1800}"
# bench08: frozen reference + pin
BENCH08_PIN="${BENCH08_PIN:-4cd90a51476266343c17331c52c1dd24b1b2e4a0}"
BENCH08_REF_REPO="${BENCH08_REF_REPO:-/mnt/cc/strata/bench08/nanoeuler.git}"
GATES_FILE="${QWEN_GATES_FILE:-/dev/null}"
if [[ -n "${QWEN_WORKROOT:-}" ]]; then
	WORKROOT="$QWEN_WORKROOT"; mkdir -p "$WORKROOT"
else
	WORKROOT="$(mktemp -d /tmp/qwen-ebuild-run.XXXXXX)"
fi
# Portage drops to the unprivileged 'portage' user (userpriv), so its TMPDIR
# must be traversable by others — mktemp's 700 blocks it. Keep it separate
# from WORKROOT (which holds our logs).
PTMP="$WORKROOT/ptmp"; mkdir -p "$PTMP"; chmod 755 "$WORKROOT" "$PTMP"
PKGDIR="$OVERLAY/$CATPN"
EBUILD="$PKGDIR/$(basename "$CATPN")-$PV.ebuild"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
pass() { printf '  \033[1;32mPASS\033[0m %s\n' "$*"; }
fail() { printf '  \033[1;31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
# bench08: machine-readable gate record (name|verdict|note)
gate() { printf '%s|%s|%s\n' "$1" "$2" "${3:-}" | tr -d '\033' >>"$GATES_FILE"; }
cleanup() { sudo -n rm -rf "$PTMP/portage" 2>/dev/null || true; rm -rf "$WORKROOT/upstream" 2>/dev/null || true; }
trap cleanup EXIT

run_qwen() { ( cd "$OVERLAY" && timeout "$QWEN_TIMEOUT" qwen --approval-mode yolo -p "$1" ) 2>&1; }

# QWEN_VERIFY_ONLY=1 skips both model passes and verifies the existing ebuild
# in place — used to unit-test the gate against known good/bad artifacts.
if [[ "${QWEN_VERIFY_ONLY:-0}" == 1 ]]; then
	[[ -f "$EBUILD" ]] || { echo "no ebuild at $EBUILD"; exit 2; }
	FAILED=0
	goto_verify=1
fi

if [[ "${goto_verify:-0}" != 1 ]]; then
# ---------------------------------------------------------------- create
say "CREATE pass"
rm -rf "$PKGDIR"
run_qwen "Create and validate the Gentoo ebuild for $REPO_URL as $CATPN-$PV in the overlay $OVERLAY. Follow the gentoo-ebuild skill end to end, including pkgcheck and real Portage build phases." \
	| tee "$WORKROOT/create.log" | tail -4
[[ -f "$EBUILD" ]] || { echo "create produced no ebuild at $EBUILD"; exit 2; }
cp "$EBUILD" "$WORKROOT/after-create.ebuild"

# ---------------------------------------------------------------- review
say "REVIEW pass (path handed in explicitly)"
run_qwen "Self-review task. The ebuild at $EBUILD was just created. Using the gentoo-ebuild skill checklist (rules 1-10), audit THIS file and fix every violation in place — especially rule #2 (first two lines must be the EXACT Gentoo header: '# Copyright <years> Gentoo Authors' then '# Distributed under the terms of the GNU General Public License v2'), rule #3 (inherit git-r3 when EGIT_* is set), and rule #10 (src_test running the check/test target). After editing, run 'pkgcheck scan' on the package directory and report what you changed." \
	| tee "$WORKROOT/review.log" | tail -4
[[ -f "$EBUILD" ]] || { echo "review removed the ebuild"; exit 2; }
if diff -q "$WORKROOT/after-create.ebuild" "$EBUILD" >/dev/null; then
	echo "  (review made no change)"
else
	echo "  (review edited the ebuild)"
fi
fi   # end: skip create/review when QWEN_VERIFY_ONLY=1

# ---------------------------------------------------------------- verify
FAILED="${FAILED:-0}"
say "VERIFY — mechanical rules"
l1="$(sed -n '1p' "$EBUILD")"; l2="$(sed -n '2p' "$EBUILD")"
if [[ "$l1" =~ ^\#\ Copyright\ .*\ Gentoo\ Authors$ ]]; then
	pass "header line 1"; gate header1 PASS
else fail "header line 1: '$l1'"; gate header1 FAIL "header line 1: '$l1'"; fi
if [[ "$l2" == "# Distributed under the terms of the GNU General Public License v2" ]]; then
	pass "header line 2"; gate header2 PASS
else fail "header line 2: '$l2'"; gate header2 FAIL "header line 2: '$l2'"; fi
if grep -q '^EGIT_' "$EBUILD"; then
	if grep -qE '^inherit( .*)? git-r3( |$)' "$EBUILD"; then
		pass "inherit git-r3"; gate inherit PASS
	else fail "EGIT_* without inherit git-r3"; gate inherit FAIL "EGIT_* without inherit git-r3"; fi
else gate inherit SKIP "no EGIT_* variables"; fi
if [[ "$EBUILD" == *-9999.ebuild ]] && grep -qE '^KEYWORDS=' "$EBUILD"; then
	if grep -qE '^KEYWORDS=""' "$EBUILD"; then
		pass "KEYWORDS empty for 9999"; gate keywords PASS
	else fail "9999 must have KEYWORDS=\"\""; gate keywords FAIL "9999 must have KEYWORDS=\"\""; fi
else gate keywords SKIP "no KEYWORDS line"; fi
# bench08: the task ebuild is pinned by commit, not by branch
pin_val="$(sed -n 's/^EGIT_COMMIT="\{0,1\}\([0-9a-f]*\)"\{0,1\}.*/\1/p' "$EBUILD" | head -1)"
if [[ "$pin_val" == "$BENCH08_PIN" ]] && ! grep -qE '^EGIT_BRANCH=' "$EBUILD"; then
	pass "EGIT_COMMIT pinned to ${BENCH08_PIN:0:7}"; gate pin PASS
else
	fail "EGIT_COMMIT must be $BENCH08_PIN without EGIT_BRANCH (got '${pin_val:-none}')"
	gate pin FAIL "EGIT_COMMIT='${pin_val:-none}'$(grep -qE '^EGIT_BRANCH=' "$EBUILD" && echo ' + EGIT_BRANCH set')"
fi

say "VERIFY — build gate"
if ! ebuild "$EBUILD" manifest >"$WORKROOT/manifest.log" 2>&1; then
	fail "ebuild manifest (see $WORKROOT/manifest.log)"; gate manifest FAIL "$(tail -1 "$WORKROOT/manifest.log")"
else pass "ebuild manifest"; gate manifest PASS; fi

pk="$(pkgcheck scan "$PKGDIR" 2>&1)"
if [[ -z "$pk" ]]; then pass "pkgcheck scan clean"; gate pkgcheck PASS
else fail "pkgcheck findings:"; printf '%s\n' "$pk" | sed 's/^/      /'; gate pkgcheck FAIL "$(printf '%s' "$pk" | tr '\n' ' ' | cut -c1-200)"; fi

BLOG="$WORKROOT/build.log"
# LC_ALL=C keeps make's diagnostics in English so the no-op-compile gate
# below can match "is up to date" regardless of the host locale.
# bench08: git-r3 override = fetch the pinned bytes from the local mirror.
# Variable name per git-r3: repo URI minus scheme+host, uppercased, [^A-Z0-9]->_
egit_ovr="EGIT_OVERRIDE_REPO_$(basename "$(dirname "$REPO_URL")")_$(basename "${REPO_URL%.git}")"
egit_ovr="$(printf '%s' "$egit_ovr" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9_' '_')"
if sudo -n env LC_ALL=C PORTAGE_TMPDIR="$PTMP" "$egit_ovr=file://$BENCH08_REF_REPO" \
		ebuild "$EBUILD" clean test install >"$BLOG" 2>&1; then
	pass "clean test install (exit 0)"; gate build PASS
else
	fail "clean test install failed (tail below)"; tail -8 "$BLOG" | sed 's/^/      /'
	gate build FAIL "$(sed 's/\x1b\[[0-9;]*m//g' "$BLOG" | grep -m1 -E '^ \* ERROR|error:' | cut -c1-200)"
fi

# no-op compile trap: a committed prebuilt binary shadows the Makefile target,
# so `make` reports it already up to date and nothing is actually built. In a
# fresh (clean) build this can only mean an artifact was committed to the repo.
# Scope the scan to the compile phase so a genuinely no-op test phase does not
# trip it. Range stays open to EOF if the compile aborted before completion.
compile_log="$(sed -n '/>>> Compiling source/,/>>> Source compiled/p' "$BLOG")"
if printf '%s\n' "$compile_log" | grep -qiE "is up to date|Nothing to be done for"; then
	fail "no-op compile: make built nothing (committed binary shadows the Makefile target)"
	printf '%s\n' "$compile_log" | grep -iE "is up to date|Nothing to be done for" | sed 's/^/      /'
	gate noop FAIL "make built nothing (committed binary shadows the target)"
else
	pass "compile did real work (no up-to-date/no-op make lines)"; gate noop PASS
fi

WORKDIR="$PTMP/portage/$CATPN-$PV/work"
IMAGE="$PTMP/portage/$CATPN-$PV/image"

# test phase actually executed (not "Nothing to be done" / skipped)
if grep -qE '^>>> Test phase: ' "$BLOG" && ! grep -qiE 'Nothing to be done for .?(check|test)' "$BLOG"; then
	pass "test phase ran"; gate testran PASS
elif grep -qE 'Test phase \[not enabled\]' "$BLOG"; then
	fail "src_test absent — test phase not enabled"; gate testran FAIL "test phase not enabled"
else
	fail "test phase did not run a check/test target"; gate testran FAIL "no check/test target ran"
fi

# real compile vs committed-binary trap: compare every ELF in workdir against
# the same path in a fresh upstream clone; byte-identical = not rebuilt.
# NOTE: scan ALL regular files, not just -executable ones — upstreams commit
# prebuilt binaries with mode 0644 (nanoeuler ships two such), and an
# -executable filter would skip exactly the artifact this gate exists to catch.
# bench08: frozen reference — the local bare mirror at the pinned commit
rm -rf "${WORKROOT:?}/upstream"; ref_ok=1
if git clone --quiet "$BENCH08_REF_REPO" "$WORKROOT/upstream" 2>/dev/null \
		&& git -C "$WORKROOT/upstream" checkout --quiet "$BENCH08_PIN" 2>/dev/null; then :
else fail "reference mirror $BENCH08_REF_REPO unusable at $BENCH08_PIN"; gate realcompile FAIL "reference mirror unusable"; ref_ok=0; fi
trap_hit=0; checked=0
if (( ref_ok )) && sudo -n test -d "$WORKDIR"; then
	while IFS= read -r bin; do
		# work-dir files are portage-owned; read them via sudo.
		sudo -n head -c4 "$bin" 2>/dev/null | grep -q $'\x7fELF' || continue
		rel="${bin#"$WORKDIR"/}"; rel="${rel#*/}"          # strip work/<S>/ prefix
		up="$WORKROOT/upstream/$rel"
		[[ -f "$up" ]] || continue
		checked=$((checked+1))
		if sudo -n cmp -s "$bin" "$up"; then
			fail "committed-binary trap: $rel byte-identical to upstream (never recompiled)"
			trap_hit=1
		fi
	done < <(sudo -n find "$WORKDIR" -type f 2>/dev/null)
fi
if (( ! ref_ok )); then :
elif [[ $trap_hit -eq 0 ]]; then
	pass "real compile (no committed-binary matches, $checked ELF checked)"; gate realcompile PASS "$checked ELF checked"
else gate realcompile FAIL "committed binary byte-identical to upstream"; fi

# installed binaries smoke-run
smoked=0
if sudo -n test -d "$IMAGE"; then
	while IFS= read -r bin; do
		smoked=$((smoked+1))
		sudo -n timeout 20 "$bin" --help >/dev/null 2>&1 || sudo -n timeout 20 "$bin" >/dev/null 2>&1 || true
	done < <(sudo -n find "$IMAGE" -type f -executable 2>/dev/null)
fi
if [[ $smoked -gt 0 ]]; then pass "installed $smoked binary(ies)"; gate installed PASS "$smoked"
else fail "no binaries installed"; gate installed FAIL "no binaries installed"; fi

say "RESULT"
if [[ $FAILED -eq 0 ]]; then
	printf '  \033[1;32mALL GATES PASSED\033[0m — %s\n' "$EBUILD"; exit 0
else
	printf '  \033[1;31mVERIFY FAILED\033[0m — logs in %s\n' "$WORKROOT"; trap - EXIT; exit 1
fi
