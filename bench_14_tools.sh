#!/usr/bin/env bash
# bench_14_tools.sh — tool-protocol conformance for an aillama profile (2026-10-29).
#
# Why this exists: bench_05 drives qwen-code and can only report FAIL when a model
# never makes a tool call. Two different defects hide behind that one FAIL, and both
# have already cost a wrong decision here:
#   * the GGUF chat template has no tool section, so llama-server reports
#     supports_tools:false and SILENTLY DROPS the tools array — the model answers as
#     a chat bot (bielik11b, 2026-10-25);
#   * the model emits <tool_call> XML as assistant CONTENT instead of filling the
#     tool_calls field (llama.cpp #30490, LFM2.5-24B-A2B).
# Neither is a reasoning failure, both disqualify a model as an agent, and neither is
# visible in a speed number. This script separates protocol from reasoning: every
# check is an exact structural assertion on the server's JSON, never a judge.
#
# Scenarios (all objective):
#   preflight  the chat template mentions tools at all
#   simple     one tool, one required string arg -> correct name + parsed args
#   nested     object argument with a nested field -> structure survives
#   longarg    4 KB string argument -> arrives intact (no truncation)
#   parallel   two independent calls in one turn -> >= 2 tool_calls
#   recover    the tool returns an error -> model reacts, no raw XML
#   decline    required info missing -> must NOT fabricate the argument
#
# Usage: bench_14_tools.sh [-h] [--selftest]
#   MODELS=prof1,prof2   switch aillama to each profile first (default: whatever runs now)
#   URL=http://...       OpenAI-compatible base (default: the running aillama endpoint)
set -euo pipefail

URL="${URL:-http://10.20.0.1:8092}"
MODELS="${MODELS:-}"
TIMEOUT="${TIMEOUT:-180}"
OUT="${OUT:-/tmp/bench14}"

die() { echo "bench_14_tools: $*" >&2; exit 1; }

help() {
	sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
	exit 0
}

# ---------------------------------------------------------------- checkers (stdin = server JSON)
check() {  # SCENARIO — reads the response on stdin, prints "PASS|FAIL  note"
	python3 -I -c '
import json, sys
scn = sys.argv[1]
raw = sys.stdin.read()
try:
    r = json.loads(raw)
except Exception as e:
    print(f"FAIL  unparseable response ({e})"); sys.exit(0)
choice = (r.get("choices") or [{}])[0]
finish = choice.get("finish_reason")
msg = choice.get("message", {}) or {}
content = msg.get("content") or ""
calls = msg.get("tool_calls") or []

# the XML-as-content defect: tool syntax in the prose instead of the tool_calls field
if not calls and any(t in content for t in ("<tool_call", "<function", "```tool")):
    print("FAIL  PROTOCOL: tool syntax emitted as content, tool_calls empty (llama.cpp #30490 class)")
    sys.exit(0)

def args_of(i=0):
    try:
        return json.loads(calls[i]["function"]["arguments"])
    except Exception:
        return None

def bad_args():
    """invalid arguments: a harness cap is inconclusive, anything else is a real defect"""
    if finish == "length":
        print("INCONC  response truncated at max_tokens - raise it for this scenario, not a model defect")
    else:
        print("FAIL  arguments are not valid JSON")
    sys.exit(0)

if scn == "simple":
    if not calls: print(f"FAIL  no tool_calls (content: {content[:60]!r})"); sys.exit(0)
    name = calls[0]["function"]["name"]; a = args_of()
    if name != "get_weather": print(f"FAIL  wrong tool {name!r}"); sys.exit(0)
    if a is None: bad_args()
    city = a.get("city")
    ok = str(city or "").lower().startswith("krak")
    print(("PASS  " if ok else "FAIL  ") + f"city={city!r}")
elif scn == "nested":
    if not calls: print("FAIL  no tool_calls"); sys.exit(0)
    a = args_of()
    if a is None: bad_args()
    f = a.get("filter") or {}
    ok = isinstance(f, dict) and str(f.get("status", "")).lower() == "open" and f.get("priority") is not None
    print(("PASS  " if ok else "FAIL  ") + f"filter={f!r}")
elif scn == "longarg":
    if not calls: print("FAIL  no tool_calls"); sys.exit(0)
    a = args_of()
    if a is None: bad_args()
    body = a.get("body", "")
    n = body.count("NEEDLE")
    print(("PASS  " if n >= 8 else "FAIL  ") + f"{len(body)} chars, {n}/8 markers kept")
elif scn == "parallel":
    names = [c["function"]["name"] for c in calls]
    print(("PASS  " if len(calls) >= 2 else "FAIL  ") + f"{len(calls)} call(s): {names}")
elif scn == "recover":
    # after a tool error the model must either retry the tool or explain; it must not
    # pretend the call succeeded and it must not emit raw tool syntax
    retried = bool(calls)
    explained = any(w in content.lower() for w in ("error", "failed", "unavailable", "could not", "nie uda", "błąd"))
    print(("PASS  " if (retried or explained) else "FAIL  ") +
          ("retried the call" if retried else f"content: {content[:70]!r}"))
elif scn == "decline":
    # the city is not in the prompt: fabricating one is the failure
    if calls:
        city = (args_of() or {}).get("city")
        print(f"FAIL  called the tool anyway with city={city!r}")
    else:
        asked = "?" in content or any(w in content.lower() for w in ("which", "what city", "specify", "need"))
        print(("PASS  " if asked else "FAIL  ") + f"no call; content: {content[:70]!r}")
else:
    print("FAIL  unknown scenario")
' "$1"
}

# ---------------------------------------------------------------- payloads
WEATHER_TOOL='{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}}}'

payload() {  # SCENARIO -> request JSON on stdout
	python3 -I -c '
import json, sys
scn = sys.argv[1]
weather = json.loads(sys.argv[2])
def req(msgs, tools, **kw):
    # max_tokens must exceed what the scenario asks the model to emit: a truncated
    # response arrives as invalid JSON and looks like a protocol defect (2026-10-29)
    d = {"messages": msgs, "tools": tools, "max_tokens": 800}
    d.update(kw); return json.dumps(d)
if scn == "simple":
    print(req([{"role":"user","content":"What is the weather in Krakow right now? Use the tool."}], [weather]))
elif scn == "nested":
    t = {"type":"function","function":{"name":"search_issues","description":"Search the tracker",
         "parameters":{"type":"object","properties":{
            "query":{"type":"string"},
            "filter":{"type":"object","properties":{"status":{"type":"string"},"priority":{"type":"integer"}},
                      "required":["status","priority"]}},
          "required":["query","filter"]}}}
    print(req([{"role":"user","content":"Search the tracker for issues about memory leaks that are still open with priority 1. Use the tool."}], [t]))
elif scn == "longarg":
    body = "\n".join(f"line {i}: NEEDLE {i} some padding text to make this argument long" if i % 10 == 0
                     else f"line {i}: ordinary padding text that carries no marker at all"
                     for i in range(80))
    t = {"type":"function","function":{"name":"save_note","description":"Store a note verbatim",
         "parameters":{"type":"object","properties":{"body":{"type":"string"}},"required":["body"]}}}
    print(req([{"role":"user","content":"Save this note VERBATIM with the save_note tool, do not shorten or summarise it:\n\n" + body}], [t], max_tokens=4000))
elif scn == "parallel":
    print(req([{"role":"user","content":"I need the weather in Krakow AND in Gdansk. Issue both tool calls now, in this single turn."}], [weather]))
elif scn == "recover":
    print(req([
        {"role":"user","content":"What is the weather in Krakow?"},
        {"role":"assistant","content":None,"tool_calls":[{"id":"call_1","type":"function",
            "function":{"name":"get_weather","arguments":"{\"city\": \"Krakow\"}"}}]},
        {"role":"tool","tool_call_id":"call_1","content":"ERROR: weather service unavailable (HTTP 503)"}],
        [weather]))
elif scn == "decline":
    print(req([{"role":"user","content":"Check the weather for me with the tool."}], [weather]))
' "$1" "$WEATHER_TOOL"
}

SCENARIOS=(simple nested longarg parallel recover decline)

# ---------------------------------------------------------------- self-test (no server)
selftest() {
	local fails=0 got
	# a correct simple call
	got=$(printf '%s' '{"choices":[{"message":{"content":"","tool_calls":[{"function":{"name":"get_weather","arguments":"{\"city\":\"Krakow\"}"}}]}}]}' | check simple)
	[[ $got == PASS* ]] || { echo "selftest: simple should PASS, got: $got"; fails=1; }
	# the XML-as-content defect must be caught as a PROTOCOL failure
	got=$(printf '%s' '{"choices":[{"message":{"content":"<tool_call>{\"name\":\"get_weather\"}</tool_call>"}}]}' | check simple)
	[[ $got == FAIL*PROTOCOL* ]] || { echo "selftest: XML-as-content should be a PROTOCOL FAIL, got: $got"; fails=1; }
	# a chat answer with no call at all
	got=$(printf '%s' '{"choices":[{"message":{"content":"It is sunny in Krakow today."}}]}' | check simple)
	[[ $got == FAIL* ]] || { echo "selftest: chat answer should FAIL, got: $got"; fails=1; }
	# fabricating a city when none was given
	got=$(printf '%s' '{"choices":[{"message":{"tool_calls":[{"function":{"name":"get_weather","arguments":"{\"city\":\"Warsaw\"}"}}]}}]}' | check decline)
	[[ $got == FAIL* ]] || { echo "selftest: fabricated city should FAIL, got: $got"; fails=1; }
	# asking back is the correct decline
	got=$(printf '%s' '{"choices":[{"message":{"content":"Which city should I check?"}}]}' | check decline)
	[[ $got == PASS* ]] || { echo "selftest: asking back should PASS, got: $got"; fails=1; }
	# a response truncated by max_tokens is a harness limit, never a protocol verdict
	got=$(printf '%s' '{"choices":[{"finish_reason":"length","message":{"tool_calls":[{"function":{"name":"save_note","arguments":"{\"body\":\"cut off"}}]}}]}' | check longarg)
	[[ $got == INCONC* ]] || { echo "selftest: truncation should be INCONC, got: $got"; fails=1; }
	# every payload must be valid JSON
	local s
	for s in "${SCENARIOS[@]}"; do
		payload "$s" | python3 -I -c 'import json,sys; json.load(sys.stdin)' || { echo "selftest: payload $s is not valid JSON"; fails=1; }
	done
	(( fails )) && die "selftest FAILED"
	echo "selftest: 6 checker cases + ${#SCENARIOS[@]} payloads OK"
	exit 0
}

# ---------------------------------------------------------------- run
preflight() {  # -> prints the template verdict, returns 1 when tools are impossible
	local props tools_in_template
	props=$(curl -s -m 20 "$URL/props") || { echo "preflight: no /props on $URL"; return 1; }
	tools_in_template=$(python3 -I -c '
import json,sys
p = json.load(sys.stdin)
t = p.get("chat_template") or ""
print("yes" if "tools" in t else "no")' <<<"$props" 2>/dev/null) || tools_in_template=unknown
	echo "preflight: chat template mentions tools: $tools_in_template"
	[[ $tools_in_template == no ]] && {
		echo "  => llama-server will DROP the tools array for this GGUF; it is a chat model, not an agent"
		return 1; }
	return 0
}

run_profile() {  # PROFILE_LABEL
	local label=$1 s resp verdict pass=0 total=0
	mkdir -p "$OUT/$label"
	echo "=== $label ==="
	preflight || { echo "$label: preflight FAILED — scenarios skipped"; return 0; }
	for s in "${SCENARIOS[@]}"; do
		payload "$s" > "$OUT/$label/$s.req.json"
		resp=$(curl -s -m "$TIMEOUT" "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
			-d @"$OUT/$label/$s.req.json") || resp='{}'
		printf '%s' "$resp" > "$OUT/$label/$s.resp.json"
		verdict=$(printf '%s' "$resp" | check "$s")
		total=$((total + 1)); [[ $verdict == PASS* ]] && pass=$((pass + 1))
		printf '  %-9s %s\n' "$s" "$verdict"
	done
	echo "  --> $label: $pass/$total protocol scenarios passed (artifacts in $OUT/$label)"
}

case "${1:-}" in
	-h|--help) help ;;
	--selftest) selftest ;;
	"") ;;
	*) die "unknown argument: $1 (try -h)" ;;
esac

command -v curl >/dev/null || die "curl missing"
if [[ -n $MODELS ]]; then
	IFS=, read -ra PROFS <<<"$MODELS"
	for p in "${PROFS[@]}"; do
		aillama switch "$p" >/dev/null || die "aillama switch $p failed"
		run_profile "$p"
	done
else
	run_profile "$(aillama status 2>/dev/null | sed -n 's/.*profile: \([^,)]*\).*/\1/p' | head -1 || echo current)"
fi
