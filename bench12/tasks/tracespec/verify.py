#!/usr/bin/env python3
"""tracespec verifier: hidden v1 production traces per behaviour group + unit tests + protected.

Items: one per trace group (baseline rounding dupes midnight freetotal open case money;
a group holds only if EVERY hidden trace of the group matches byte-for-byte), unit-tests
(the repo's own tests pass and none were deleted: >= 7 pass), protected. PASS = all 10.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
sys.path.insert(0, str(HERE))
import b12  # noqa: E402
import gen  # noqa: E402
import prod  # noqa: E402

ORIGINAL_TESTS = 7


def body(work: Path, pristine: Path):
    r = b12.Rubric()
    vis = sorted((pristine / "traces").glob("*.trace"))
    if [p.read_text() for p in vis] != [prod.trace_text(t) for t in gen.visible()]:
        raise b12.BenchError("pristine traces do not match the generator (seed drift?)")
    sc = b12.scratch(work)
    try:
        hdir = sc.parent / "hidden"
        hdir.mkdir()
        for group, trs in gen.hidden().items():
            bad = 0
            first = ""
            for i, lines in enumerate(trs):
                f = hdir / f"{group}-{i}.log"
                f.write_text("\n".join(lines) + "\n")
                res = b12.run([sys.executable, "-m", "meter", "bill", str(f)], sc, timeout=30)
                if res.returncode != 0 or res.stdout != prod.bill(lines):
                    bad += 1
                    if not first:
                        first = (res.stderr.strip().splitlines() or ["output differs"])[-1]
            r.add(group, bad == 0, f"{len(trs) - bad}/{len(trs)} hidden traces match" + (f"; {first}" if bad else ""))
        res = b12.run(b12.pytest_cmd("tests"), sc, timeout=120)
        passed, failed = b12.pytest_counts(res.stdout)
        tamper = b12.pytest_tamper(work, pristine)
        r.add("unit-tests", failed == 0 and passed >= ORIGINAL_TESTS and not tamper,
              f"TAMPER: {', '.join(tamper[:3])}" if tamper else
              f"{passed} passed, {failed} failed (need all passing, >= {ORIGINAL_TESTS})")
    finally:
        b12.cleanup(sc)
    ok, why = b12.check_protected(work, pristine)
    r.add("protected", ok, why)
    r.finish()


b12.main_wrapper(body)
