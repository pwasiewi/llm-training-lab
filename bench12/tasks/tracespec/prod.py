"""tracespec: the v1 PRODUCTION billing behaviour (the spec) + bug/wrong-fix flags.

bill(lines, flags) renders exactly what `python -m meter bill` must print.
With flags empty it is production. Each flag injects one deviation:
  v2 bugs:   nearest lastdup nosplit freetotal dropopen case float
  wrong fixes an agent may plausibly make while repairing v2:
             ceilmin nodup startday openown firstspelling roundtotal
Trace generation uses the flags to make every trace of a group trigger
its own deviation(s) and nothing else (group isolation).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

BLOCK = 900
FREE = 4
STD = Decimal("0.125")
ENT = Decimal("0.085")

V2_BUGS = ("nearest", "lastdup", "nosplit", "freetotal", "dropopen", "case", "float")
GROUPS = {
    # group: (flags that EVERY trace of the group must trigger, flags allowed to co-trigger)
    "baseline": ((), ()),
    "rounding": (("nearest", "ceilmin"), ()),
    "dupes": (("lastdup", "nodup"), ("dropopen", "openown")),
    "midnight": (("nosplit", "startday"), ("freetotal",)),
    "freetotal": (("freetotal",), ("startday",)),
    "open": (("dropopen", "openown"), ()),
    "case": (("case", "firstspelling"), ()),
    "money": (("float", "roundtotal"), ()),
}
ALL_FLAGS = V2_BUGS + ("ceilmin", "nodup", "startday", "openown", "firstspelling", "roundtotal")


def pts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def blocks(seconds: float, flags) -> int:
    if "nearest" in flags:
        return max(1, round(seconds / BLOCK))
    if "ceilmin" in flags:
        return max(1, math.ceil(int(seconds // 60) / 15))
    return max(1, math.ceil(seconds / BLOCK))


def bill(lines, flags=frozenset()) -> str:
    flags = frozenset(flags)
    seen = {}
    evs = []
    spelling = {}
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ts, eid, acct, kind, sid = line.split()
        if "case" not in flags:
            spelling.setdefault(acct.lower(), acct)
            acct = acct.lower()
        ev = [pts(ts), i, eid, acct, kind, sid]
        if "nodup" in flags:
            evs.append(ev)
        elif "lastdup" in flags:
            seen[eid] = ev
        elif eid not in seen:
            seen[eid] = ev
    if "nodup" not in flags:
        evs = list(seen.values())
    evs.sort(key=lambda e: (e[0], e[1]))
    open_, sess = {}, []
    for ts, _, eid, acct, kind, sid in evs:
        key = (acct, sid)
        if kind == "start":
            open_.setdefault(key, ts)
        elif key in open_:
            sess.append((acct, open_.pop(key), ts))
    if open_ and "dropopen" not in flags:
        last_all = max(e[0] for e in evs)
        last_acct = defaultdict(lambda: None)
        for e in evs:
            last_acct[e[3]] = max(e[0], last_acct[e[3]] or e[0])
        for (acct, sid), st in open_.items():
            end = last_acct[acct] if "openown" in flags else last_all
            sess.append((acct, st, end))
    # per account per day blocks
    day_blocks = defaultdict(lambda: defaultdict(int))
    for acct, st, en in sess:
        if "nosplit" in flags:
            day_blocks[acct][st.date()] += blocks((en - st).total_seconds(), flags)
            continue
        cur = st
        while True:
            midnight = datetime.combine(cur.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            part_end = min(en, midnight)
            day = st.date() if "startday" in flags else cur.date()
            day_blocks[acct][day] += blocks((part_end - cur).total_seconds(), flags)
            if en <= midnight:
                break
            cur = midnight
    rows = []
    for acct in sorted(day_blocks):
        days = day_blocks[acct]
        total = sum(days.values())
        if "freetotal" in flags:
            billable = max(0, total - FREE)
        else:
            billable = sum(max(0, b - FREE) for b in days.values())
        rate = ENT if acct.lower().startswith("ent-") else STD
        name = spelling.get(acct, acct) if "firstspelling" in flags else acct
        rows.append((name, total, billable, rate * billable, float(rate) * billable))
    out = []
    if "float" in flags:
        for a, t, b, _, fc in rows:  # exactly v2: float(rate) * billable, f"{:.2f}"
            out.append(f"{a} {t} {b} {fc:.2f}")
        out.append(f"TOTAL {sum(fc for *_, fc in rows):.2f}")
    else:
        q = Decimal("0.01")
        rounded = [r[3].quantize(q, ROUND_HALF_UP) for r in rows]
        for (a, t, b, _, _), c in zip(rows, rounded):
            out.append(f"{a} {t} {b} {c}")
        tot = sum((r[3] for r in rows), Decimal(0)).quantize(q, ROUND_HALF_UP) if "roundtotal" in flags \
            else sum(rounded, Decimal(0))
        out.append(f"TOTAL {tot}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------ generation --

ACCTS = ["acme", "globex", "initech", "umbrella", "hooli", "ent-stark", "ent-wayne", "vandelay"]


def fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_log(rng: random.Random, group: str) -> list[str]:
    day0 = datetime(2026, 8, 3 + rng.randrange(20), tzinfo=timezone.utc)
    accts = rng.sample(ACCTS, rng.randrange(2, 4))
    sess = []  # (acct, start, end|None)
    ndays = 3 if group in ("freetotal",) else 1

    def dur():
        if group in ("rounding",):
            return timedelta(minutes=rng.randrange(1, 100), seconds=rng.choice([0, 0, 10, 20, 45]))
        if group in ("midnight", "case"):
            return timedelta(minutes=rng.randrange(5, 120), seconds=rng.choice([0, 30]))
        return timedelta(minutes=15 * rng.randrange(1, 8))

    for a in accts:
        for d in range(ndays):
            t = day0 + timedelta(days=d, hours=8 + rng.randrange(3), minutes=rng.randrange(60))
            for _ in range(rng.randrange(1, 5)):
                du = dur()
                sess.append((a, t, t + du))
                t = t + du + timedelta(minutes=rng.randrange(5, 90))
    if group == "midnight":
        a = rng.choice(accts)
        st = day0 + timedelta(hours=23, minutes=rng.randrange(20, 59), seconds=rng.choice([0, 30]))
        sess.append((a, st, st + timedelta(minutes=rng.randrange(10, 80))))
    if group == "open":
        for a in rng.sample(accts, rng.randrange(1, len(accts))):
            st = day0 + timedelta(hours=19, minutes=15 * rng.randrange(8))
            sess.append((a, st, None))
    if group == "money":
        sess.append(("ent-" + rng.choice(["stark", "wayne"]) if rng.random() < 0.3 else rng.choice(accts),
                     day0 + timedelta(hours=21), day0 + timedelta(hours=21, minutes=15 * rng.randrange(1, 9))))
    # events
    lines = []
    n = 0
    sids = defaultdict(int)
    for a, st, en in sess:
        sids[a] += 1
        sid = f"s{sids[a]}"
        disp = a
        if group == "case" and rng.random() < 0.5:
            disp = rng.choice([a.upper(), a.capitalize()])
        n += 1
        lines.append((st, f"ev{n:04d}", disp, "start", sid))
        if en is not None:
            n += 1
            disp2 = a if group != "case" or rng.random() < 0.5 else a.upper()
            lines.append((en, f"ev{n:04d}", disp2, "stop", sid))
    if group == "dupes":
        for _ in range(rng.randrange(1, 4)):
            src = rng.choice(lines)
            lag = timedelta(minutes=15 * rng.randrange(1, 16))
            lines.append((src[0] + lag, src[1], src[2], src[3], src[4]))
    lines.sort(key=lambda x: x[0])
    if group == "open":  # a late unrelated event so "last event in log" != "account's last event"
        last = max(lines[-1][0], day0 + timedelta(hours=22)) + timedelta(minutes=15 * rng.randrange(2, 8))
        last = last.replace(minute=(last.minute // 15) * 15, second=0)
        other = f"zz-probe{rng.randrange(9)}"
        lines.append((last - timedelta(minutes=15), f"ev{n + 1:04d}", other, "start", "s1"))
        lines.append((last, f"ev{n + 2:04d}", other, "stop", "s1"))
    out = [f"{fmt(t)} {eid} {a} {k} {s}" for t, eid, a, k, s in lines]
    if rng.random() < 0.3:
        out.insert(0, "# collector export")
    return out


def isolated(lines, group) -> bool:
    must, allowed = GROUPS[group]
    base = bill(lines)
    for f in ALL_FLAGS:
        diff = bill(lines, {f}) != base
        if f in must and not diff:
            return False
        if diff and f not in must and f not in allowed:
            return False
    return True


def traces(seed: int, group: str, count: int) -> list[list[str]]:
    rng = random.Random(seed * 1009 + sum(map(ord, group)))
    out = []
    tries = 0
    while len(out) < count:
        tries += 1
        if tries > 20000:
            raise RuntimeError(f"cannot isolate group {group}")
        lines = make_log(rng, group)
        stamps = [ln.split()[0] for ln in lines if not ln.startswith("#")]
        if len(set(stamps)) != len(stamps):  # no timestamp ties: tie order is unspecified
            continue
        if isolated(lines, group):
            out.append(lines)
    return out


def trace_text(lines) -> str:
    return "# input\n" + "\n".join(lines) + "\n# expected\n" + bill(lines)
