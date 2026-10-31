#!/usr/bin/env python3
"""longctx verifier: ANSWERS.json vs the generator's answer key. 6 synthesis items +
1 fixed-position retrieval control + protected. PASS = all 8."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
sys.path.insert(0, str(HERE))
import b12  # noqa: E402
import gen  # noqa: E402


def norm(x):
    return x.strip().lower() if isinstance(x, str) else x


def as_int(x):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def body(work: Path, pristine: Path):
    files, truth = gen.build()
    for rel, text in files.items():
        if (pristine / rel).read_text() != text:
            raise b12.BenchError(f"pristine {rel} does not match the generator (seed drift?)")
    r = b12.Rubric()
    try:
        a = json.loads(b12.read_inside(work, "ANSWERS.json"))
        if not isinstance(a, dict):
            a = {}
    except (OSError, ValueError):
        a = {}
    g = a.get
    r.add("q1_deploy", norm(g("q1_deploy")) == truth["q1_deploy"], f"got {g('q1_deploy')!r}")
    d = g("q2_degraded_minutes") if isinstance(g("q2_degraded_minutes"), dict) else {}
    r.add("q2_degraded", {k: as_int(d.get(k)) for k in ("eu", "us")} == truth["q2_degraded_minutes"], f"got {d}")
    r.add("q3_timeout", as_int(g("q3_timeout_ms")) == truth["q3_timeout_ms"], f"got {g('q3_timeout_ms')!r}")
    r.add("q4_orders", as_int(g("q4_failed_orders")) == truth["q4_failed_orders"], f"got {g('q4_failed_orders')!r}")
    r.add("q5_key", norm(g("q5_changed_key")) == truth["q5_changed_key"], f"got {g('q5_changed_key')!r}")
    q6 = g("q6_ack") if isinstance(g("q6_ack"), dict) else {}
    r.add("q6_ack", norm(q6.get("who")) == truth["q6_ack"]["who"] and as_int(q6.get("minutes")) == truth["q6_ack"]["minutes"],
          f"got {q6}")
    r.add("control", norm(g("control_build")) == truth["control_build"], f"got {g('control_build')!r}")
    ok, why = b12.check_protected(work, pristine)
    r.add("protected", ok, why)
    r.finish()


b12.main_wrapper(body)
