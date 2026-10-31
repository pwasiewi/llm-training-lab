#!/usr/bin/env python3
"""ledger verifier: hidden integration run of the agent's CLI on hidden datasets,
compared byte-for-byte with an independent oracle (oracle.py), plus error paths,
backward compatibility, the repo's own tests and protected files. 11 items, PASS = all.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
sys.path.insert(0, str(HERE))
import b12  # noqa: E402
import hidden  # noqa: E402
import oracle  # noqa: E402


def run_cli(sc: Path, d: Path, orders, rates=True):
    d.mkdir(parents=True, exist_ok=True)
    (d / "orders.json").write_text(json.dumps(orders, indent=1))
    (d / "discounts.json").write_text(json.dumps(hidden.DISCOUNTS))
    with open(d / "rates.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "currency", "eur_per_unit"])
        w.writerows(hidden.RATES)
    out = d / "out"
    cmd = [sys.executable, "-m", "invoicer", "run", "--orders", str(d / "orders.json"),
           "--discounts", str(d / "discounts.json"), "--out", str(out)]
    if rates:
        cmd[6:6] = ["--rates", str(d / "rates.csv")]
    res = b12.run(cmd, sc, timeout=60)
    got = {}
    if out.is_dir():
        for p in out.rglob("*"):
            if p.is_file():
                got[p.relative_to(out).as_posix()] = p.read_text(errors="replace")
    return res, out, got


def body(work: Path, pristine: Path):
    r = b12.Rubric()
    sc = b12.scratch(work)
    base = sc.parent
    try:
        res, _, got = run_cli(sc, base / "main", hidden.MAIN)
        st, want = oracle.expected(hidden.MAIN, hidden.DISCOUNTS, hidden.RATES)
        assert st == "ok"
        num_of = {}
        for k, v in want.items():
            if k.startswith("invoices/"):
                num_of[v.splitlines()[1].split()[1]] = k
        crash = "" if res.returncode == 0 else f"rc={res.returncode} {res.stderr.strip()[-100:]}"
        for item, ids in hidden.GROUPS.items():
            bad = [i for i in ids if got.get(num_of[i]) != want[num_of[i]]]
            r.add(item, not bad and not crash, crash or (f"differs: {bad}" if bad else f"{len(ids)} invoices identical"))
        r.add("export", got.get("export.csv") == want["export.csv"], crash or "export.csv")
        r.add("summary", got.get("summary.txt") == want["summary.txt"], crash or "summary.txt")
        for item, ds, msg_part in (("err_rate", hidden.ERR_RATE, "no rate for USD on or before 2025-12-20"),
                                   ("err_ccy", hidden.ERR_CCY, "unknown currency SEK")):
            res, out, got = run_cli(sc, base / item, ds)
            ok = res.returncode == 3 and msg_part in res.stderr and not out.exists()
            r.add(item, ok, f"rc={res.returncode} out_exists={out.exists()} stderr={res.stderr.strip()[-80:]!r}")
        res, _, got = run_cli(sc, base / "compat", hidden.COMPAT, rates=False)
        st, want = oracle.expected(hidden.COMPAT, hidden.DISCOUNTS, [])
        r.add("compat", res.returncode == 0 and got == want,
              "EUR-only run without --rates identical" if got == want else f"rc={res.returncode} differs")
        res = b12.run(b12.pytest_cmd("tests"), sc, 120)
        passed, failed = b12.pytest_counts(res.stdout)
        tamper = b12.pytest_tamper(work, pristine)
        r.add("unit_tests", failed == 0 and passed >= 5 and not tamper,
              f"TAMPER: {', '.join(tamper[:3])}" if tamper else f"{passed} passed, {failed} failed")
    finally:
        b12.cleanup(sc)
    ok, why = b12.check_protected(work, pristine)
    r.add("protected", ok, why)
    r.finish()


b12.main_wrapper(body)
