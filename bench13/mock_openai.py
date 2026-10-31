#!/usr/bin/env python3
"""mock_openai.py — CPU-only OpenAI-compatible stub for dry-running bench_13.

Never loads a model. Answers /v1/chat/completions deterministically:

  SWE-rebench (request carries `tools`, i.e. mini-swe-agent):
    the task is identified by matching the problem statement against the
    dataset jsonl; then turn 1 = one `bash` tool call, turn 2 = the submit
    command (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/patch.txt`).
      oracle  : turn 1 applies the GOLD patch (git apply --index) -> resolved
      garbage : turn 1 appends junk to the first file the gold patch touches
                -> patch applies, tests fail -> unresolved
  Aider Polyglot (no `tools`): the exercise is identified from the file names
    and instructions aider sends; the reply uses aider's "whole" edit format.
      oracle  : every solution file = the exercise's .meta reference solution
      garbage : every solution file = a junk line (compile/test failure)
      retry   : garbage until aider's second try (its test-failure prompt),
                then oracle (exercises aider's pass_2 accounting)
  Any harness, mode "die": the first completion request drops the TCP
    connection without an answer and /health turns 503 (simulated
    llama-server death) until POST /mock/revive.
  Mode "dielate" (death MID-task): the first turn of a task is answered
    like garbage (SWE: the bash edit runs; aider: try 1 fails its tests),
    the next request of that task dies as above.

The mode is taken from the requested MODEL NAME when it contains one of
oracle|garbage|retry|dielate|die (e.g. litellm "openai/mock-garbage"), else --mode.
So one mock serves a whole multi-model dry run: MODELS=mock-oracle,mock-garbage.

Usage:
  mock_openai.py [--mode oracle|garbage|retry] --swe-jsonl FILE --polyglot DIR
                 [--host 10.20.0.1] [--port 8199] [--dump DIR]
Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions,
POST /mock/revive.
"""
import argparse
import base64
import json
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/patch.txt"
LANG_EXT = {"py": "python", "go": "go", "rs": "rust", "js": "javascript",
            "cpp": "cpp", "h": "cpp", "hpp": "cpp", "java": "java"}


def text_of(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


class Oracle:
    def __init__(self, swe_jsonl: str | None, polyglot: str | None):
        self.swe = []
        if swe_jsonl:
            for line in Path(swe_jsonl).read_text().splitlines():
                if line.strip():
                    self.swe.append(json.loads(line))
        self.ex = {}  # (lang, name) -> {"solution": {relpath: ref_text}, "instr": str}
        if polyglot:
            for cfg in Path(polyglot).glob("*/exercises/practice/*/.meta/config.json"):
                d = cfg.parent.parent
                lang = d.parent.parent.parent.name
                c = json.loads(cfg.read_text())
                files = c.get("files", {})
                sol = files.get("solution", [])
                exm = files.get("example", [])
                given = {Path(e).name for e in files.get("editor", [])}
                pairs, used = {}, set()
                for i, s in enumerate(sol):
                    # match by basename first, then by position
                    cand = [e for e in exm if Path(e).name == Path(s).name] or exm[i:i + 1]
                    if cand and (d / cand[0]).exists():
                        pairs[s] = (d / cand[0]).read_text()
                        used.add(cand[0])
                # Java references sometimes spread helpers over extra files the
                # model may not create; a model can legally put them into the
                # solution file as package-private classes — so does the oracle
                extra = [e for e in exm if e not in used and Path(e).name not in given
                         and e.endswith(".java") and (d / e).exists()]
                if extra and len(pairs) == 1:
                    main_rel = next(iter(pairs))
                    pairs[main_rel] = merge_java(pairs[main_rel], [(d / e).read_text() for e in extra])
                instr = (d / ".docs/instructions.md")
                self.ex[(lang, d.name)] = {"solution": pairs,
                                           "instr": instr.read_text() if instr.exists() else ""}

    # ---- SWE
    def swe_instance(self, msgs: list[dict]) -> dict | None:
        blob = "\n".join(text_of(m) for m in msgs if m.get("role") == "user")
        for inst in self.swe:
            ps = inst["problem_statement"].strip()
            if ps[:300] and ps[:300] in blob:
                return inst
        return None

    # ---- Aider
    def aider_exercise(self, msgs: list[dict]):
        blob = "\n".join(text_of(m) for m in msgs)
        m = re.search(r"([a-z]+)/exercises/practice/([a-z0-9-]+)/", blob)
        if m and (m.group(1), m.group(2)) in self.ex:
            return (m.group(1), m.group(2))
        # fallback: solution-file basenames shown by aider + instructions text
        best, best_score = None, 0
        for key, ex in self.ex.items():
            names = [Path(s).name for s in ex["solution"]]
            if not names or not all(re.search(r"(^|[\s/`])" + re.escape(n) + r"\s*$", blob, re.M)
                                    for n in names):
                continue
            score = 1 + (2 if ex["instr"][:200] and ex["instr"][:200] in blob else 0)
            if score > best_score:
                best, best_score = key, score
        return best


def merge_java(main: str, extras: list[str]) -> str:
    """Append helper classes to the main Java file (imports hoisted, `public` dropped)."""
    imports, bodies = [], []
    for src in extras:
        for line in src.splitlines():
            if line.startswith("import "):
                imports.append(line)
            elif not line.startswith("package "):
                bodies.append(re.sub(r"^public\s+(?=(final\s+|abstract\s+)?(class|enum|interface|record)\b)", "", line))
    head = [ln for ln in imports if ln not in main]
    return "\n".join(dict.fromkeys(head)) + ("\n" if head else "") + main.rstrip("\n") + "\n\n" + "\n".join(bodies) + "\n"


def shown_name(blob: str, relpath: str) -> str:
    """The path aider displayed for this solution file (fallback: relpath)."""
    base = re.escape(Path(relpath).name)
    m = re.search(r"^(\S*" + base + r")\s*$", blob, re.M)
    return m.group(1) if m else relpath


def fence_for(text: str) -> str:
    return "````" if "```" in text else "```"


class Handler(BaseHTTPRequestHandler):
    server_version = "bench13-mock/1"

    def log_message(self, fmt, *args):  # quiet; the request log goes to --dump
        pass

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            if self.server.dead:
                return self._json(503, {"status": "dead"})
            return self._json(200, {"status": "ok"})
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") == "/mock/revive":
            self.server.dead = False
            return self._json(200, {"status": "revived"})
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if req.get("stream"):
            return self._json(400, {"error": {"message": "mock does not stream"}})
        S = self.server
        with S.lock:
            S.count += 1
            seq = S.count
        name = str(req.get("model", ""))
        mode = next((m for m in ("oracle", "garbage", "retry", "dielate", "die") if m in name),
                    S.default_mode)
        if mode == "dielate":
            # SWE: no assistant turn yet; aider: no "mock ..." answer yet
            done = [m for m in req.get("messages", []) if m.get("role") == "assistant"
                    and (req.get("tools") or "mock" in text_of(m)[:200])]
            mode = "garbage" if not done else "die"
        if mode == "die" or S.dead:
            S.dead = True
            print(f"[mock] #{seq} DIE ({name}) — dropping the connection", file=sys.stderr, flush=True)
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.close_connection = True
            return
        S.mode = mode  # read by the reply builders (per request; requests are serial per run)
        try:
            if req.get("tools"):
                msg, kind, note = self.swe_reply(req)
            else:
                msg, kind, note = self.aider_reply(req)
        except Exception as e:  # report, never hang the client
            msg, kind, note = {"role": "assistant", "content": f"mock error: {e}"}, "error", repr(e)
        if S.dump:
            (S.dump / f"{seq:05d}-{kind}.json").write_text(
                json.dumps({"note": note, "request": req, "reply": msg}, indent=1))
        print(f"[mock] #{seq} {kind} {note}", file=sys.stderr, flush=True)
        self._json(200, {
            "id": f"mock-{seq}", "object": "chat.completion", "created": int(time.time()),
            "model": req.get("model", "mock"),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })

    def swe_reply(self, req):
        S = self.server
        msgs = req.get("messages", [])
        inst = S.oracle.swe_instance(msgs)
        turn = sum(1 for m in msgs if m.get("role") == "assistant")
        if inst is None:
            cmd, note = "echo 'mock: unknown task' && " + SUBMIT, "swe ? (no instance matched)"
        elif turn == 0:
            if S.mode == "garbage":
                # an EXISTING file the gold patch modifies (prefer .py): non-empty, wrong patch
                files = [f for f in re.findall(r"^--- a/(\S+)", inst["patch"], re.M)]
                tgt = next((f for f in files if f.endswith(".py")), files[0] if files else "README.md")
                cmd = (f"printf '\\nthis is a mock garbage edit ((( \\n' >> {tgt} && "
                       "git diff > /tmp/patch.txt && wc -l /tmp/patch.txt")
            else:
                b64 = base64.b64encode(inst["patch"].encode()).decode()
                # image layers leave a stale stat cache: refresh, else --index refuses
                cmd = (f"echo {b64} | base64 -d > /tmp/gold.patch && git update-index -q --refresh; "
                       "git apply --index /tmp/gold.patch && "
                       "git diff --cached > /tmp/patch.txt && wc -l /tmp/patch.txt")
            note = f"swe {inst['instance_id']} turn 1 ({S.mode})"
        else:
            cmd, note = SUBMIT, f"swe {inst['instance_id']} turn {turn + 1} submit"
        return ({"role": "assistant", "content": "THOUGHT: mock step.",
                 "tool_calls": [{"id": f"call_{S.count}", "type": "function",
                                 "function": {"name": "bash",
                                              "arguments": json.dumps({"command": cmd})}}]},
                "swe", note)

    def aider_reply(self, req):
        S = self.server
        msgs = req.get("messages", [])
        key = S.oracle.aider_exercise(msgs)
        prior = sum(1 for m in msgs if m.get("role") == "assistant"
                    and "mock" in text_of(m)[:200])
        # aider's 2nd try starts with prompts.test_failures; lint reflections
        # inside one try do not, so "retry" yields pass_2, not a reflection fix
        second_try = any("The tests are correct, don't try and change them" in text_of(m)
                         for m in msgs if m.get("role") == "user")
        if key is None:
            return ({"role": "assistant", "content": "mock: exercise not identified."},
                    "aider", "aider ? (no exercise matched)")
        ex = S.oracle.ex[key]
        use_ref = S.mode == "oracle" or (S.mode == "retry" and second_try)
        blob = "\n".join(text_of(m) for m in msgs)
        parts = [f"mock {'reference' if use_ref else 'garbage'} answer.\n"]
        for rel, ref in ex["solution"].items():
            body = ref if use_ref else "this is a mock garbage edit (((\n"
            f = fence_for(body)
            parts.append(f"{shown_name(blob, rel)}\n{f}\n{body.rstrip(chr(10))}\n{f}\n")
        return ({"role": "assistant", "content": "\n".join(parts)}, "aider",
                f"aider {key[0]}/{key[1]} attempt {prior + 1} ({'ref' if use_ref else 'garbage'})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oracle", "garbage", "retry"], default="oracle")
    ap.add_argument("--swe-jsonl")
    ap.add_argument("--polyglot")
    ap.add_argument("--host", default="10.20.0.1")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--dump", help="directory for request/reply dumps")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.default_mode = srv.mode = a.mode
    srv.count, srv.lock, srv.dead = 0, threading.Lock(), False
    srv.oracle = Oracle(a.swe_jsonl, a.polyglot)
    srv.dump = Path(a.dump) if a.dump else None
    if srv.dump:
        srv.dump.mkdir(parents=True, exist_ok=True)
    print(f"[mock] {a.mode} on http://{a.host}:{a.port}/v1 — {len(srv.oracle.swe)} SWE tasks, "
          f"{len(srv.oracle.ex)} aider exercises", file=sys.stderr, flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
