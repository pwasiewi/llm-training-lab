"""Shared verifier for chain6/10/14 (called from each task's verify.py)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b12  # noqa: E402
import chaingen  # noqa: E402


def verify(n: int):
    def body(work: Path, pristine: Path):
        r = b12.Rubric()
        _, steps, xs = chaingen.generate()
        # the pristine copy must be what this seed generates (bench sanity)
        if (pristine / "CHAIN.md").read_text() != chaingen.chain_md(steps, n):
            raise b12.BenchError("pristine CHAIN.md does not match the generator (seed drift?)")
        ok, why = b12.check_protected(work, pristine)
        got = {}
        for line in b12.read_inside(work, "chain.log").splitlines():
            m = re.match(r"\s*step\s*0*(\d+)\s*:\s*(\d{1,7})\s*$", line, re.I)
            if m:
                got.setdefault(int(m.group(1)), int(m.group(2)))  # first claim per step counts
        prefix = 0
        broken = False
        end = "complete"   # how the correct prefix ended: wrong value, or step never written
        for k in range(1, n + 1):
            want = xs[k]
            hit = got.get(k) == want
            if hit and not broken:
                prefix += 1
            elif not broken:
                broken = True
                end = "wrong" if k in got else "missing"
            r.add(f"step{k:02d}", hit, f"want {want:06d} got " + (f"{got[k]:06d}" if k in got else "-") +
                  f" [{steps[k-1]['trap']}/{steps[k-1]['op']}]")
        a = b12.read_inside(work, "ANSWER").strip()
        r.add("answer", a.isdigit() and int(a) == xs[n], f"ANSWER={a[:12] or '-'} want {xs[n]:06d}")
        r.add("protected", ok, why)
        # PREFIX_END missing = right-censored for the horizon fit (a timeout or a
        # write-at-the-end agent), not a step failure
        print(f"PREFIX_END {end}", flush=True)
        # SCORE = correct prefix length (horizon unit); PASS = every step + answer + inputs intact
        r.finish(passed=(prefix == n and r.ok("answer") and ok), k=prefix, n=n)
    b12.main_wrapper(body)
