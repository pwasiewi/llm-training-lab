#!/usr/bin/env python3
"""longctx fixture: a ~350 KB (~85-95K token) incident dump + QUESTIONS.md.

Every synthesis answer needs facts from >= 2 files combined (codename alias,
per-file timezone header, layered config precedence + TTL, 5xx-only error
rate, dedupe). One CONTROL question is pure retrieval of a single line at a
fixed position (the exact middle line of the largest log).
truth() returns the answer key; it is recomputed by verify.py, never stored
in the workdir.
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = int(os.environ.get("B12_SEED", "12"))
DAY = datetime(2026, 9, 14, tzinfo=timezone.utc)
T0 = DAY + timedelta(hours=12)
T1 = DAY + timedelta(hours=18)
EU = timezone(timedelta(hours=2))
US = timezone(timedelta(hours=-7))

SERVICES = {"kestrel": "checkout-api", "kite": "checkout-web", "heron": "payment-gw",
            "osprey": "catalog", "plover": "auth"}
ENGINEERS = ["mpatel", "jkowalski", "aokafor", "lnguyen", "sbergstrom"]


def at(h, m, s=0):
    return DAY + timedelta(hours=h, minutes=m, seconds=s)


def build(seed: int = SEED):
    rng = random.Random(seed * 4241 + 17)
    files: dict[str, str] = {}
    truth: dict = {}

    # ---------------------------------------------------------- deploys --
    deploys = []
    used_ids = set()

    def did():
        while True:
            x = f"d-{rng.randrange(1000, 9999)}"
            if x not in used_ids:
                used_ids.add(x)
                return x

    for _ in range(34):
        t = T0 + timedelta(seconds=rng.randrange(6 * 3600))
        tgt = rng.choice(["osprey", "plover", "heron", "kite", "osprey", "plover"])
        deploys.append(dict(t=t, id=did(), target=tgt, region=rng.choice(["eu", "us"]), status="ok"))
    culprit = dict(t=at(13, 41, 5), id=did(), target="kestrel", region="eu", status="ok")
    decoy_kite = dict(t=at(13, 38, 40), id=did(), target="kite", region="eu", status="ok")
    early_k = dict(t=at(12, 55, 2), id=did(), target="kestrel", region="us", status="ok")
    late_k = dict(t=at(14, 2, 31), id=did(), target="kestrel", region="us", status="ok")
    rollback = dict(t=at(14, 20, 12), id=did(), target="kestrel", region="eu", status="ok",
                    note=f"rollback of {culprit['id']}")
    # keep any random deploy away from the kestrel window so the answer is unique
    deploys = [d for d in deploys if not (at(13, 30) <= d["t"] <= at(14, 30))]
    deploys += [culprit, decoy_kite, early_k, late_k, rollback]
    deploys.sort(key=lambda d: d["t"])
    lines = ["# deploys.log — deploy controller audit trail (all times UTC)",
             "# target = service codename, see services.yaml"]
    for d in deploys:
        lines.append(f"{d['t']:%Y-%m-%dT%H:%M:%SZ} deploy {d['id']} target={d['target']} region={d['region']} "
                     f"status={d['status']}" + (f" note=\"{d['note']}\"" if d.get("note") else ""))
    files["deploys.log"] = "\n".join(lines) + "\n"

    keys = ["checkout.pool_size", "checkout.retry_budget", "catalog.cache_ttl_s", "auth.jwt_skew_s",
            "payment.idempotency_window_s", "checkout.timeout_ms", "web.bundle_hash", "catalog.page_size"]
    for d in deploys:
        if d is culprit:
            ch = {"checkout.pool_size": {"from": 64, "to": 8}}
        elif d is rollback:
            ch = {"checkout.pool_size": {"from": 8, "to": 64}}
        elif d is late_k or d is early_k:
            ch = {"checkout.retry_budget": {"from": 3, "to": rng.choice([2, 4])}}
        elif d is decoy_kite:
            ch = {"web.bundle_hash": {"from": "9f1c2e", "to": "a07b3d"}}
        else:
            k = rng.choice(keys[2:5] + keys[6:])
            ch = {k: {"from": rng.randrange(1, 500), "to": rng.randrange(1, 500)}}
        body = [f"id: {d['id']}", f"target: {d['target']}", f"region: {d['region']}",
                f"started: {d['t']:%Y-%m-%dT%H:%M:%SZ}", "image: registry.local/" + d["target"] + f":{rng.randrange(100, 999)}",
                "changes:"]
        for k, v in ch.items():
            body.append(f"  {k}: {{from: {v['from']}, to: {v['to']}}}")
        files[f"deploy-manifests/{d['id']}.yaml"] = "\n".join(body) + "\n"
    truth["q1_deploy"] = culprit["id"]
    truth["q5_changed_key"] = "checkout.pool_size"

    svc = ["# services.yaml — codename registry", "services:"]
    for code, name in SERVICES.items():
        svc.append(f"  - codename: {code}\n    name: {name}\n    regions: [eu, us]")
    files["services.yaml"] = "\n".join(svc) + "\n"

    # --------------------------------------------- checkout-api logs ----
    eu_err = (at(13, 43, 17), at(14, 22, 40))
    us_err = (at(14, 10, 3), at(14, 25, 30))
    orders_failed = set()
    paths = ["/checkout", "/checkout/confirm", "/cart", "/checkout/quote"]

    def checkout_log(region, tz, win, rate_hi):
        out = [f"# checkout-api {region} — application log",
               f"# times are local wall-clock time, UTC{tz.utcoffset(None).total_seconds() / 3600:+03.0f}:00, no zone suffix",
               "# fields: time level status path order latency_ms"]
        t = T0
        first = None
        while t < T1:
            t += timedelta(milliseconds=rng.randrange(12000, 30000))
            in_win = win[0] <= t <= win[1]
            if t >= win[0] and first is None and in_win:
                t = win[0]
            if in_win and (rng.random() < rate_hi or t == win[0]):
                st = rng.choice([500, 502, 503, 503, 504])
                oid = f"ORD-{rng.randrange(100000, 999999)}"
                if orders_failed and rng.random() < 0.3:
                    oid = rng.choice(sorted(orders_failed))  # client retries of an already failed order
                orders_failed.add(oid)
                lvl = "ERROR"
                if first is None:
                    first = t
            else:
                st = rng.choice([200] * 30 + [201, 204, 400, 404, 409])
                oid = f"ORD-{rng.randrange(100000, 999999)}" if rng.random() < 0.6 else "-"
                lvl = "INFO" if st < 400 else "WARN"
            lt = t.astimezone(tz)
            out.append(f"{lt:%Y-%m-%d %H:%M:%S}.{t.microsecond // 1000:03d} {lvl} status={st} path={rng.choice(paths)} "
                       f"order={oid} latency_ms={rng.randrange(8, 900) if st < 500 else rng.randrange(2000, 30000)}")
        return out, first

    eu_lines, eu_first = checkout_log("eu", EU, eu_err, 0.35)
    us_lines, us_first = checkout_log("us", US, us_err, 0.2)
    files["logs/checkout-api-eu.log"] = "\n".join(eu_lines) + "\n"
    files["logs/checkout-api-us.log"] = "\n".join(us_lines) + "\n"
    first5xx = min(eu_first, us_first)
    assert first5xx == eu_err[0]
    truth["q4_failed_orders"] = len(orders_failed)

    # ------------------------------------------------------- metrics ----
    deg = {"eu": 0, "us": 0}
    meu = ["# metrics-eu.log — checkout-api per-minute counters, minute start in UTC",
           "# minute req err5xx err4xx"]
    mus = ["epoch_minute_start,requests,errors_4xx,errors_5xx"]
    t = T0
    while t < T1:
        for region, win in (("eu", eu_err), ("us", us_err)):
            req = rng.randrange(800, 1600)
            e4 = rng.randrange(0, int(req * 0.09))       # 4xx alone may exceed 5 % — must not count
            if win[0] - timedelta(minutes=1) < t <= win[1]:
                e5 = int(req * rng.choice([0.02, 0.06, 0.12, 0.3, 0.45, 0.04, 0.08]))
            else:
                e5 = rng.randrange(0, int(req * 0.01) + 1)
            if e5 / req > 0.05:
                deg[region] += 1
            if region == "eu":
                meu.append(f"{t:%Y-%m-%dT%H:%MZ} req={req} err5xx={e5} err4xx={e4}")
            else:
                mus.append(f"{int(t.timestamp())},{req},{e4},{e5}")
        t += timedelta(minutes=1)
    files["metrics/metrics-eu.log"] = "\n".join(meu) + "\n"
    files["metrics/metrics-us.csv"] = "\n".join(mus) + "\n"
    truth["q2_degraded_minutes"] = deg

    # -------------------------------------------------------- config ----
    files["config/defaults.yaml"] = ("checkout:\n  timeout_ms: 3000\n  pool_size: 64\n  retry_budget: 3\n"
                                     "catalog:\n  cache_ttl_s: 300\n  page_size: 50\n")
    files["config/region-us.yaml"] = "checkout:\n  timeout_ms: 2500\n"
    files["config/region-eu.yaml"] = "checkout:\n  timeout_ms: 2800\n  pool_size: 64\n"
    flags = ["# flags.log — feature-flag service changes (UTC)",
             f"{at(12, 30):%Y-%m-%dT%H:%M:%SZ} set catalog.page_size=40 scope=all",
             f"{at(14, 5):%Y-%m-%dT%H:%M:%SZ} set checkout.timeout_ms=1800 scope=us",
             f"{at(14, 9):%Y-%m-%dT%H:%M:%SZ} set checkout.timeout_ms=2000 scope=eu",
             f"{at(14, 45):%Y-%m-%dT%H:%M:%SZ} set checkout.timeout_ms=2200 scope=eu",
             f"{at(15, 30):%Y-%m-%dT%H:%M:%SZ} set checkout.timeout_ms=1500 scope=us"]
    files["config/flags.log"] = "\n".join(flags) + "\n"
    audit = ["# audit.log — admin API (UTC)"]
    for _ in range(40):
        tt = T0 + timedelta(seconds=rng.randrange(6 * 3600))
        audit.append((tt, f"{tt:%Y-%m-%dT%H:%M:%SZ} {rng.choice(ENGINEERS)} GET /admin/status region={rng.choice(['eu', 'us'])}"))
    audit.append((at(14, 12), f"{at(14, 12):%Y-%m-%dT%H:%M:%SZ} lnguyen SET checkout.timeout_ms=900 region=us ttl=30m"))
    audit.append((at(14, 14), f"{at(14, 14):%Y-%m-%dT%H:%M:%SZ} aokafor SET checkout.timeout_ms=950 region=eu ttl=2h"))
    audit = [audit[0]] + [s for _, s in sorted(audit[1:])]
    files["config/audit.log"] = "\n".join(audit) + "\n"
    truth["q3_timeout_ms"] = 1800  # at 14:50Z in us: runtime override expired 14:42, flag 14:05 = 1800

    # --------------------------------------------------------- pager ----
    ack = at(14, 6, 30)
    pg = ["# pager.log — on-call paging; times are local, UTC-07:00",
          f"{at(13, 20).astimezone(US):%Y-%m-%d %H:%M:%S} page P-7731 opened service=catalog summary=\"p99 latency\"",
          f"{at(13, 24).astimezone(US):%Y-%m-%d %H:%M:%S} page P-7731 ack by sbergstrom",
          f"{at(13, 46, 2).astimezone(US):%Y-%m-%d %H:%M:%S} page P-7740 opened service=checkout-api summary=\"5xx rate\"",
          f"{at(13, 58).astimezone(US):%Y-%m-%d %H:%M:%S} page P-7740 escalated to secondary",
          f"{at(14, 1).astimezone(US):%Y-%m-%d %H:%M:%S} page P-7741 opened service=checkout-web summary=\"JS errors\"",
          f"{at(14, 3).astimezone(US):%Y-%m-%d %H:%M:%S} page P-7741 ack by jkowalski",
          f"{ack.astimezone(US):%Y-%m-%d %H:%M:%S} page P-7740 ack by mpatel",
          f"{at(14, 31).astimezone(US):%Y-%m-%d %H:%M:%S} page P-7740 resolved"]
    files["pager.log"] = "\n".join(pg) + "\n"
    truth["q6_ack"] = {"who": "mpatel", "minutes": int((ack - first5xx).total_seconds() // 60)}

    # ------------------------------------------ ballast + control ------
    cat = ["# catalog eu — access log (UTC)"]
    n = 650
    mid = n // 2
    build_no = f"build-{rng.randrange(10000, 99999)}"
    for i in range(n):
        tt = T0 + timedelta(seconds=i * 33 + rng.randrange(30))
        if i == mid:
            cat.append(f"{tt:%Y-%m-%dT%H:%M:%SZ} INFO CONTROL-MARKER release-check passed for {build_no} (catalog)")
            continue
        cat.append(f"{tt:%Y-%m-%dT%H:%M:%SZ} INFO GET /v2/items/{rng.randrange(10**6)}?lang={rng.choice(['de', 'fr', 'pl', 'en'])} "
                   f"200 {rng.randrange(2, 90)}ms cache={rng.choice(['hit', 'hit', 'miss'])} "
                   f"build=build-{rng.randrange(10000, 99999)}")
    files["logs/catalog-eu.log"] = "\n".join(cat) + "\n"
    truth["control_build"] = build_no
    auth = ["# auth — token service log (UTC)"]
    for i in range(400):
        tt = T0 + timedelta(seconds=i * 54 + rng.randrange(50))
        auth.append(f"{tt:%Y-%m-%dT%H:%M:%SZ} INFO token issued sub=u{rng.randrange(10**7)} "
                    f"aud={rng.choice(['checkout', 'catalog', 'web'])} exp=900 kid=k{rng.randrange(3)}")
    files["logs/auth.log"] = "\n".join(auth) + "\n"
    chat = ["# incident-chat.txt — #inc-checkout export (times UTC)",
            "13:47 mpatel: seeing 5xx on checkout in eu, looking",
            "13:49 jkowalski: checkout-web deploy went out at 13:38, could be the bundle?",
            "13:55 aokafor: web errors are client-side only, api 5xx are separate",
            "14:12 lnguyen: dropping checkout timeout in us via admin api as a stopgap",
            "14:20 mpatel: rolling back the eu checkout deploy",
            "14:33 mpatel: recovering; postmortem owner = me"]
    files["incident-chat.txt"] = "\n".join(chat) + "\n"
    return files, truth


RUNBOOK = """# RUNBOOK.md — checkout incident conventions

## Services
Deploys and dashboards use codenames; services.yaml maps codename -> service.

## Time
Every log states its own clock in its header line. Convert before comparing.

## Error rate
Error rate for a minute = 5xx responses / requests. 4xx are client errors and
never count. A minute is "degraded" when the error rate is strictly above 5 %.

## Configuration precedence (highest first)
1. runtime overrides set through the admin API (config/audit.log, `SET`); an
   override expires when its ttl elapses (it is inactive from then on);
2. feature flags (config/flags.log; a later change replaces an earlier one
   for the same key and scope; scope=all applies to every region);
3. region file (config/region-<region>.yaml);
4. config/defaults.yaml.
"""

QUESTIONS = """# QUESTIONS.md

Answer from the material in this directory only (RUNBOOK.md defines the
conventions). Write `ANSWERS.json` with exactly these keys:

```json
{
  "q1_deploy": "d-NNNN",
  "q2_degraded_minutes": {"eu": 0, "us": 0},
  "q3_timeout_ms": 0,
  "q4_failed_orders": 0,
  "q5_changed_key": "section.key",
  "q6_ack": {"who": "name", "minutes": 0},
  "control_build": "build-NNNNN"
}
```

1. The incident began with the first HTTP 5xx returned by checkout-api in any
   region. Give the id of the last deploy of checkout-api (any region) that
   started before that moment.
2. For each region, how many minutes of the day were degraded for
   checkout-api?
3. What was the effective value of checkout.timeout_ms in region us at
   14:50:00 UTC?
4. How many distinct orders received at least one 5xx from checkout-api
   (both regions together)?
5. Which configuration key did the deploy from question 1 change?
6. Who acknowledged the page for checkout-api's 5xx rate, and how many whole
   minutes (rounded down) after the first 5xx of question 1 did that happen?
7. (control) Which build is named on the line containing CONTROL-MARKER?
"""


def write(out: Path) -> None:
    files, _ = build()
    for rel, text in files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (out / "RUNBOOK.md").write_text(RUNBOOK)
    (out / "QUESTIONS.md").write_text(QUESTIONS)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--truth":
        print(json.dumps(build()[1], indent=1))
    else:
        write(Path(sys.argv[1]))
