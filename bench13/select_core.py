#!/usr/bin/env python3
"""select_core.py — from bench_13 screen results to the CORE task list, plus
the summary / paired tables printed by bench_13_public.sh.

results.tsv columns (written by bench_13_public.sh, no header):
  model  suite  task  run  verdict  score  secs  note
Lost rows (verdict SKIPPED, DIED or ERROR = harness failure, note "SERVER
DIED ...") never count as FAIL: they are dropped before any rate is computed.

Modes:
  select_core.py RESULTS.tsv [RESULTS2.tsv ...] [--suite swe] [--n 4] [--out FILE]
      Pick N tasks where the fleet DISAGREES (some model passes, another
      fails). Rank: disagreement (spread of per-model pass rates, max-min),
      then mean pass rate closest to 50 %, then screen order. If fewer than N
      tasks disagree, fill with the tasks whose mean is closest to 50 % and
      say so in the reason. Default --out: stdout. Write it to
      bench13/swe_core.tsv for SET=core. Tasks where some model has no scored
      run are skipped (rerun the lost rows first). With ONE scored run per
      model x task a "disagreement" is one coin flip: rank only on a set
      chosen WITHOUT these outcomes, and never pool screen + core rows in one
      ranking (selection on disagreement inflates the tested difference).
  select_core.py --summary RESULTS.tsv     per-model table
  select_core.py --paired RESULTS.tsv      per-task x model table + pairwise
                                           discordance (exact McNemar p)
"""
import argparse
import math
import sys
from collections import OrderedDict, defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                c = line.rstrip("\n").split("\t")
                c += [""] * (8 - len(c))
                m, s, t, r, v, sc, secs, note = c[:8]
                lost = v in ("SKIPPED", "DIED", "ERROR") or note.startswith("SERVER DIED")
                rows.append(dict(model=m, suite=s, task=t, run=r, verdict=v, score=sc,
                                 secs=int(secs) if secs.isdigit() else 0, note=note, lost=lost))
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_p(b, c):
    """exact two-sided binomial test on the discordant pairs"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summary(rows):
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["model"], r["suite"]), []).append(r)
    w = max([22] + [len(m) for m, _ in groups])
    print(f"{'MODEL':{w}} {'SUITE':5} {'PASS':>5} {'FAIL':>5} {'TMOUT':>5} {'LOST':>5} "
          f"{'RATE':>6} {'95% CI':>13} {'NO-TMO':>6} {'pass_1':>6} {'wall_min':>8}")
    for (m, s), rs in groups.items():
        sc = [r for r in rs if not r["lost"]]
        k = sum(r["verdict"] == "PASS" for r in sc)
        f = sum(r["verdict"] == "FAIL" for r in sc)
        t = sum(r["verdict"] == "TIMEOUT" for r in sc)
        n = len(sc)
        ci = f"{100 * wilson(k, n)[0]:4.0f}-{100 * wilson(k, n)[1]:3.0f}%" if n else "-"
        p1 = sum(r["score"] == "pass_1" for r in sc) if s == "aider" else None
        rate = f"{100 * k / n:5.1f}%" if n else "    -"
        # pass rate over runs that finished in time: separates speed (tok/s) from skill
        notmo = f"{100 * k / (n - t):5.1f}%" if n - t else "    -"
        print(f"{m:{w}} {s:5} {k:5} {f:5} {t:5} {len(rs) - n:5} {rate:>6} "
              f"{ci:>13} {notmo:>6} {('-' if p1 is None else p1):>6} "
              f"{sum(r['secs'] for r in sc) / 60:8.1f}")
    print("RATE = PASS / scored (TIMEOUT counts as not passed); NO-TMO = PASS / (scored - TIMEOUT); "
          "wall_min = scored rows only")


def per_task(rows, suite=None):
    """(suite, task) -> model -> [passes, scored, lost, pass_1]"""
    tab = OrderedDict()
    models = []
    for r in rows:
        if suite and r["suite"] != suite:
            continue
        if r["model"] not in models:
            models.append(r["model"])
        c = tab.setdefault((r["suite"], r["task"]), {}).setdefault(r["model"], [0, 0, 0, 0])
        if r["lost"]:
            c[2] += 1
            continue
        c[1] += 1
        if r["verdict"] == "PASS":
            c[0] += 1
            if r["score"] == "pass_1":
                c[3] += 1
    return tab, models


def cell(c, suite):
    if c is None:
        return "."
    k, n, lost, p1 = c
    if n == 0:
        return "L" * min(lost, 1) or "."
    if n == 1:
        if k == 1:
            return "P1" if suite == "aider" and p1 else ("P2" if suite == "aider" else "P")
        return "F"
    return f"{k}/{n}"


def paired(rows):
    tab, models = per_task(rows)
    if not tab:
        print("(no rows)")
        return
    w = max([12] + [len(m) for m in models])
    print(f"{'SUITE':5} {'TASK':44} " + " ".join(f"{m:>{w}}" for m in models) + "  ")
    for (s, t), per in tab.items():
        rates = [c[0] / c[1] for c in per.values() if c[1]]
        star = "*" if rates and max(rates) != min(rates) else " "
        print(f"{s:5} {t[:44]:44} " + " ".join(f"{cell(per.get(m), s):>{w}}" for m in models) + f" {star}")
    print("cells: P/F (1 run), P1/P2 = aider pass on try 1/2, k/n (several runs), L = lost, '.' = not run")
    if len(models) < 2:
        return
    print("Pairwise (tasks where one model's pass rate is higher; one pair per TASK; exact McNemar):")
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            only_a = only_b = 0
            for per in tab.values():
                ca, cb = per.get(a), per.get(b)
                if not ca or not cb or not ca[1] or not cb[1]:
                    continue
                # one pair per TASK (majority verdict): runs of one task are not
                # independent pairs, and run r1 of A is not paired with r1 of B
                pa, pb = ca[0] / ca[1], cb[0] / cb[1]
                if pa > pb:
                    only_a += 1
                elif pb > pa:
                    only_b += 1
            print(f"  {a} vs {b}: {a}-only {only_a}, {b}-only {only_b}, p = {mcnemar_p(only_a, only_b):.3f}")


def select(rows, suite, n, out):
    tab, models = per_task(rows, suite)
    if len(models) < 2:
        print(f"WARNING: only {len(models)} model(s) in the screen — 'disagreement' is meaningless",
              file=sys.stderr)
    if tab and max(c[1] for per in tab.values() for c in per.values()) < 2:
        print("WARNING: 1 scored run per model x task — a 'disagreement' is one coin flip; "
              "rank only on a set chosen WITHOUT these outcomes, never pool screen + core", file=sys.stderr)
    scored = []
    for order, ((s, t), per) in enumerate(tab.items()):
        rates = {m: c[0] / c[1] for m, c in per.items() if c[1]}
        if not rates:
            continue
        unscored = [m for m in models if m not in rates]
        if unscored:
            print(f"WARNING: {t}: no scored run for {','.join(unscored)} — rerun the lost rows first; task skipped",
                  file=sys.stderr)
            continue
        spread = max(rates.values()) - min(rates.values())
        mean = sum(rates.values()) / len(rates)
        lost = [m for m, c in per.items() if c[2]]
        scored.append((spread, abs(mean - 0.5), order, t, rates, mean, lost))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    picked = scored[:n]
    lines = [f"# bench_13 {suite} CORE set — select_core.py from {len(models)} model(s): {', '.join(models)}",
             "# columns: task<TAB>reason"]
    for spread, _, _, t, rates, mean, lost in picked:
        kind = "disagree" if spread > 0 else "FILLER (fleet agrees)"
        detail = ", ".join(f"{m} {100 * r:.0f}%" for m, r in rates.items())
        extra = f"; lost runs for {','.join(lost)}" if lost else ""
        lines.append(f"{t}\t{kind}: spread {100 * spread:.0f} pp, mean {100 * mean:.0f}% ({detail}){extra}")
    for spread, _, _, t, rates, mean, _ in scored[n:]:
        lines.append(f"#R {t}\tnot picked: spread {100 * spread:.0f} pp, mean {100 * mean:.0f}%")
    text = "\n".join(lines) + "\n"
    if out:
        open(out, "w").write(text)
        print(f"wrote {out}: {len(picked)} tasks "
              f"({sum(1 for x in picked if x[0] > 0)} disagreeing)", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--suite", default="swe")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = load(a.results)
    if a.summary:
        summary(rows)
        return 0
    if a.paired:
        paired(rows)
        return 0
    return select(rows, a.suite, a.n, a.out)


if __name__ == "__main__":
    sys.exit(main())
