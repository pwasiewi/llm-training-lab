#!/usr/bin/env python3
"""reconcile verifier: byte-identical output on hidden inputs + a relative time limit.

The time limit is RELATIVE: the reference implementation runs on the same
production-size input in the same verify call, so llama-server CPU load
slows both sides alike. Items: dense1..dense3 (tie-heavy small inputs),
sample (the workdir sample), big (hidden 100K-payment input), speed10x
(<= 10x reference), speed3x (<= max(3x reference, reference + 0.5 s)),
protected. PASS = all 8.
"""
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
sys.path.insert(0, str(HERE))
import b12  # noqa: E402
import gen  # noqa: E402
import recgen  # noqa: E402

REF = HERE / "reference" / "reconcile.py"
BIG_N = 100000


def timed(script, d, cwd, timeout):
    """One run in a fresh, empty HOME/TMPDIR/XDG_CACHE_HOME: an on-disk result
    cache from an earlier try (or an earlier verify call) is never visible."""
    fresh = tempfile.mkdtemp(prefix="b12home-")
    env = {"HOME": fresh, "TMPDIR": fresh, "XDG_CACHE_HOME": fresh}
    try:
        t = time.perf_counter()
        res = b12.run([sys.executable, str(script), str(d / "payments.csv"), str(d / "bank.csv")], cwd, timeout, env=env)
        return time.perf_counter() - t, res
    finally:
        shutil.rmtree(fresh, ignore_errors=True)


def ref_ok(res, what):
    """A reference failure (e.g. a timeout under heavy load) is a bench error, never a model FAIL."""
    if res.returncode != 0:
        raise b12.BenchError(f"reference failed on {what}: rc={res.returncode} {res.stderr.strip()[-120:]}")
    return res


def body(work: Path, pristine: Path):
    r = b12.Rubric()
    sc = b12.scratch(work)
    base = sc.parent
    try:
        agent = sc / "reconcile.py"
        cases = [("dense1", dict(n_pay=400, seed=gen.SEED * 7 + 1, skew=0.8, days=12)),
                 ("dense2", dict(n_pay=1500, seed=gen.SEED * 7 + 2, skew=0.6, days=25)),
                 ("dense3", dict(n_pay=3000, seed=gen.SEED * 7 + 3, skew=0.5, days=40))]
        for name, kw in cases:
            d = base / name
            recgen.write(d, kw.pop("n_pay"), kw.pop("seed"), **kw)
            want = ref_ok(timed(REF, d, base, 60)[1], name)
            _, got = timed(agent, d, sc, 120)
            r.add(name, got.returncode == 0 and got.stdout == want.stdout,
                  "identical" if got.stdout == want.stdout else f"differs (rc={got.returncode}) {got.stderr.strip()[-120:]}")
        want = ref_ok(timed(REF, pristine / "data", base, 60)[1], "sample")
        _, got = timed(agent, pristine / "data", sc, 300)
        r.add("sample", got.returncode == 0 and got.stdout == want.stdout,
              "identical" if got.stdout == want.stdout else f"differs (rc={got.returncode})")
        big = base / "big"
        # a fresh production-size input per verify call: no cross-run or cross-try cache hit
        recgen.write(big, BIG_N, secrets.randbits(31))
        trs = [timed(REF, big, base, 120) for _ in range(2)]
        for _, res in trs:
            ref_ok(res, "big")
        tref = min(t for t, _ in trs)
        want = ref_ok(timed(REF, big, base, 120)[1], "big")
        lim10, lim3 = 10 * tref, max(3 * tref, tref + 0.5)
        tag, got = timed(agent, big, sc, max(lim10 * 1.2, 5))
        if got.returncode == 0 and tag > lim3:      # one retry against scheduler noise
            t2, got2 = timed(agent, big, sc, max(lim10 * 1.2, 5))
            if got2.returncode == 0 and t2 < tag:
                tag, got = t2, got2
        fast_ok = got.returncode == 0 and got.stdout == want.stdout
        if got.returncode == 124:  # too slow for the speed items; still judge correctness
            _, got = timed(agent, big, sc, 60)
        r.add("big", got.returncode == 0 and got.stdout == want.stdout,
              "identical" if got.stdout == want.stdout else
              ("timed out after 60 s" if got.returncode == 124 else f"differs (rc={got.returncode})"))
        r.add("speed10x", fast_ok and tag <= lim10, f"agent {tag:.2f}s vs ref {tref:.2f}s (limit {lim10:.2f}s)")
        r.add("speed3x", fast_ok and tag <= lim3, f"agent {tag:.2f}s vs ref {tref:.2f}s (limit {lim3:.2f}s)")
    finally:
        b12.cleanup(sc)
    ok, why = b12.check_protected(work, pristine)
    r.add("protected", ok, why)
    r.finish()


b12.main_wrapper(body)
