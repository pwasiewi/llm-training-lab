#!/bin/bash
# bench_11_depth_stream.sh — prompt-depth sweep measured from the CLIENT side,
# so two different engines can be compared on the same prompts.
#
# Why: llama.cpp reports its own timings, other engines (Strata, 2026-09-27)
# report theirs or none. This script times what a user sees, over the OpenAI
# streaming API, and applies the same arithmetic to every server:
#   pp  = prompt_tokens / time-to-first-token   (prefill rate, incl. queueing)
#   tg  = (completion_tokens - 1) / (t_last - t_first)   (decode rate)
# completion_tokens comes from the usage block (stream_options.include_usage);
# a server that sends none falls back to the count of content-bearing chunks,
# which UNDER-counts for engines that stream several tokens per chunk
# (speculative decoding) — the fallback is flagged in the output.
# llama.cpp's own timings (prompt_per_second / predicted_per_second) are
# printed beside the client numbers when present, as a cross-check.
#
# Prompts: an instruction + the start of a source corpus, cut to DEPTHS tokens
# (via the server's /tokenize when it has one, else ~3.3 chars per token; the
# real count is the server's usage.prompt_tokens, which is what pp uses).
# Every request starts with a unique nonce line, so no prefix cache can serve
# it: cache_prompt=false covers llama.cpp only, and Strata's conversation cache
# answered the second run of an identical prompt from memory (2026-09-27).
set -euo pipefail

MODEL="${MODEL:-}"                       # GGUF (server mode); first shard of a split file
URL="${URL:-}"                           # client-only mode: an already running endpoint
PORT="${PORT:-8095}"                     # 8090 = ntfy, which also answers /health
SERVER_BIN="${SERVER_BIN:-llama-server}"
SERVER_ARGS="${SERVER_ARGS:---fit on -c 40960 -fa on -ctk q8_0 -ctv q8_0 -t 16 -np 1 --jinja}"
DEPTHS="${DEPTHS:-1024,4096,32768}"
RUNS="${RUNS:-2}"
NGEN="${NGEN:-256}"
CORPUS="${CORPUS:-/mnt/cc/strata/llama.cpp/src}"   # any tree of text files; sorted, concatenated
THINK="${THINK:-off}"                    # off: enable_thinking=false (answer at once); on: model default
LOG="${LOG:-/var/tmp/bench11-$(date +%Y%m%d-%H%M).log}"
DRY_RUN="${DRY_RUN:-0}"
NEEDLE="${NEEDLE:-0}"                   # 1: retrieval check — a SECRET-CODE line at 25 % depth, answer must contain it

die() { echo "ERROR: $*" >&2; exit 1; }

help() {
	cat <<EOF
Usage: ${0##*/} [-h]
  server mode:  MODEL=/path/first-shard.gguf ${0##*/}    (starts SERVER_BIN, sweeps, stops it)
  client mode:  URL=http://127.0.0.1:8080 ${0##*/}       (sweeps an endpoint that is already up)
Environment (current values):
  MODEL / URL   see above                  (${MODEL:-unset} / ${URL:-unset})
  SERVER_ARGS   extra llama-server args    ($SERVER_ARGS)
  DEPTHS        prompt lengths in tokens   ($DEPTHS)
  RUNS / NGEN   repeats per depth / tokens generated ($RUNS / $NGEN)
  CORPUS        prompt text source         ($CORPUS)
  THINK         off|on                     ($THINK)
  NEEDLE=1      long-context retrieval check instead of the review prompt ($NEEDLE)
  PORT / SERVER_BIN / LOG / DRY_RUN=1 (print the server command only)
A 256-token warm-up request runs first and is not reported (page cache, graphs).
EOF
}

[[ ${1:-} == -h || ${1:-} == --help ]] && { help; exit 0; }
[[ -n $MODEL || -n $URL ]] || { help; exit 1; }
[[ -d $CORPUS || -f $CORPUS ]] || die "corpus not found: $CORPUS"

spid=""
cleanup() { if [[ -n $spid ]]; then kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; fi; }
trap cleanup EXIT

if [[ -z $URL ]]; then
	[[ -f $MODEL ]] || die "model not found: $MODEL"
	command -v "$SERVER_BIN" >/dev/null || die "binary not found: $SERVER_BIN"
	read -r -a extra <<<"$SERVER_ARGS"
	cmd=("$SERVER_BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" "${extra[@]}")
	echo "server: ${cmd[*]}" | tee -a "$LOG"
	(( DRY_RUN )) && exit 0
	curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && die "port $PORT already answers /health"
	"${cmd[@]}" >>"$LOG.server" 2>&1 &
	spid=$!
	for _ in $(seq 1 900); do
		curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
		kill -0 "$spid" 2>/dev/null || { tail -30 "$LOG.server"; die "server died — log: $LOG.server"; }
		sleep 1
	done
	curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || die "server not healthy after 900 s"
	grep -E 'fit|n_cpu_moe|override|offloaded|CUDA0 model buffer|KV buffer|compute buffer' "$LOG.server" |
		tail -12 | tee -a "$LOG" || true    # no match = model fully on the GPU, not an error
	nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/vram: /' | tee -a "$LOG"
	URL="http://127.0.0.1:$PORT"
fi

python3 - "$URL" "$DEPTHS" "$RUNS" "$NGEN" "$CORPUS" "$THINK" "$NEEDLE" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys, time, urllib.request, uuid

url, depths, runs, ngen, corpus, think, needle = sys.argv[1:8]
needle = needle == "1"
depths = [int(d) for d in depths.split(",")]
runs, ngen = int(runs), int(ngen)

def post(path, body, stream=False):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=7200)

files = []
if os.path.isdir(corpus):
    for root, _, names in os.walk(corpus):
        files += [os.path.join(root, n) for n in names]
    files.sort()
else:
    files = [corpus]
text = ""
for f in files:
    try:
        text += f"\n// ==== {os.path.relpath(f, corpus) if os.path.isdir(corpus) else f}\n" + open(f, errors="replace").read()
    except OSError:
        pass
    if len(text) > max(depths) * 6:
        break
head = "Review the following source code. List the three most serious defects you can find, with file and line.\n"

def cut(n):
    """The instruction + corpus prefix of about n tokens."""
    body = n - 64                                   # room for the chat template and the instruction
    try:
        toks = json.load(post("/tokenize", {"content": text[: body * 6]}))["tokens"][:body]
        return head + json.load(post("/detokenize", {"tokens": toks}))["content"], "tokenize"
    except Exception:
        return head + text[: int(body * 3.3)], "chars"

def one(prompt):
    prompt = f"[request {uuid.uuid4().hex}]\n" + prompt   # defeats every prefix cache
    body = {"model": "bench", "messages": [{"role": "user", "content": prompt}], "max_tokens": ngen,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "cache_prompt": False}
    if think == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["reasoning_effort"] = "none"
    t0 = time.time(); tfirst = tlast = None; chunks = 0; usage = None; timings = None; answer = ""
    with post("/v1/chat/completions", body, stream=True) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                answer += d.get("content") or ""
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time()
                    tfirst = tfirst or now; tlast = now; chunks += 1
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens")
    src = "usage" if ct else "chunks"
    ct = ct or chunks
    pp = pt / (tfirst - t0) if pt and tfirst else float("nan")
    tg = (ct - 1) / (tlast - tfirst) if tfirst and tlast and tlast > tfirst else float("nan")
    one.answer = answer
    return pt, ct, tfirst - t0 if tfirst else float("nan"), pp, tg, src, timings

print(f"endpoint {url}, NGEN {ngen}, thinking {think}")
one(cut(256)[0])                                    # warm-up, not reported
print(f"{'depth':>7} {'run':>3} {'prompt_t':>8} {'gen_t':>5} {'ttft_s':>7} {'pp_cli':>7} {'tg_cli':>6} {'cnt':>6} {'pp_srv':>7} {'tg_srv':>6}")
if needle:
    head = ("Somewhere in the source code below there is one line of the form 'SECRET-CODE: <word>'. "
            "Reply with that word only.\n")
for d in depths:
    prompt, how = cut(d)
    for i in range(runs):
        run_prompt, code = prompt, None
        if needle:                                   # a fresh code per run, planted at 25 % of the body
            code = uuid.uuid4().hex[:10]
            body = prompt[len(head):]; k = body.find("\n", len(body) // 4) + 1
            run_prompt = head + body[:k] + f"// SECRET-CODE: {code}\n" + body[k:]
        pt, ct, ttft, pp, tg, src, tm = one(run_prompt)
        pps = f"{tm['prompt_per_second']:7.1f}" if tm else "      -"
        tgs = f"{tm['predicted_per_second']:6.1f}" if tm else "     -"
        hit = "" if code is None else ("  needle OK" if code in one.answer else f"  needle MISS: {one.answer.strip()[:60]!r}")
        print(f"{d:>7} {i+1:>3} {pt or -1:>8} {ct:>5} {ttft:7.1f} {pp:7.1f} {tg:6.1f} {src:>6} {pps} {tgs}{hit}", flush=True)
PY
echo "log: $LOG"
