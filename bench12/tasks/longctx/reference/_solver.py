#!/usr/bin/env python3
"""Independent solver for longctx (selftest only). MODE: ok | localtime | with4xx | retrieval."""
import glob
import json
import re
import sys
from datetime import datetime, timedelta, timezone

MODE = sys.argv[1] if len(sys.argv) > 1 else "ok"


def offset_of(path):
    head = open(path).read(400)
    m = re.search(r"UTC([+-]\d\d):00", head)
    return timezone(timedelta(hours=int(m.group(1))))


codename = {}
cur = None
for line in open("services.yaml"):
    m = re.search(r"codename: (\w+)", line)
    if m:
        cur = m.group(1)
    m = re.search(r"name: ([\w-]+)", line)
    if m and "codename" not in line:
        codename[m.group(1)] = cur
ck = codename["checkout-api"]

first = first_key = None
orders = set()
for path in glob.glob("logs/checkout-api-*.log"):
    tz = offset_of(path)
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.split()
        st = int(f[3].split("=")[1])
        if 500 <= st < 600:
            naive = datetime.strptime(f"{f[0]} {f[1]}", "%Y-%m-%d %H:%M:%S.%f")
            t = naive.replace(tzinfo=tz).astimezone(timezone.utc)
            # localtime defect: earliest by wall-clock text, then converted correctly
            k = naive if MODE == "localtime" else t
            if first is None or k < first_key:
                first, first_key = t, k
            orders.add(f[5].split("=")[1])

deploys = []
for line in open("deploys.log"):
    if line.startswith("#"):
        continue
    f = line.split()
    kv = dict(x.split("=", 1) for x in f[3:] if "=" in x)
    deploys.append((datetime.strptime(f[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc), f[2], kv["target"]))
q1 = max((d for d in deploys if d[2] == ck and d[0] < first), key=lambda d: d[0])[1]
man = open(f"deploy-manifests/{q1}.yaml").read()
q5 = re.search(r"changes:\n  ([\w.]+):", man).group(1)

deg = {"eu": 0, "us": 0}
for line in open("metrics/metrics-eu.log"):
    if line.startswith("#"):
        continue
    kv = dict(x.split("=") for x in line.split()[1:])
    e = int(kv["err5xx"]) + (int(kv["err4xx"]) if MODE == "with4xx" else 0)
    deg["eu"] += e / int(kv["req"]) > 0.05
for line in open("metrics/metrics-us.csv").read().splitlines()[1:]:
    _, req, e4, e5 = map(int, line.split(","))
    e = e5 + (e4 if MODE == "with4xx" else 0)
    deg["us"] += e / req > 0.05

at = datetime(2026, 9, 14, 14, 50, tzinfo=timezone.utc)
val = None
for line in open("config/audit.log"):
    m = re.match(r"(\S+) \S+ SET checkout\.timeout_ms=(\d+) region=us ttl=(\d+)m", line)
    if m:
        t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if t <= at and (MODE == "with4xx" or at < t + timedelta(minutes=int(m.group(3)))):
            val = int(m.group(2))
if val is None:
    for line in open("config/flags.log"):
        m = re.match(r"(\S+) set checkout\.timeout_ms=(\d+) scope=(us|all)", line)
        if m and datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) <= at:
            val = int(m.group(2))
if val is None:
    val = int(re.search(r"timeout_ms: (\d+)", open("config/region-us.yaml").read()).group(1))

tzp = offset_of("pager.log")
who = mins = None
for line in open("pager.log"):
    m = re.match(r"(\S+ \S+) page (\S+) ack by (\w+)", line)
    if m:
        pid = m.group(2)
        opened = [l for l in open("pager.log") if f"page {pid} opened" in l and "service=checkout-api" in l]
        if opened:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tzp).astimezone(timezone.utc)
            who, mins = m.group(3), int((t - first).total_seconds() // 60)

ctl = None
for line in open("logs/catalog-eu.log"):
    if "CONTROL-MARKER" in line:
        ctl = re.search(r"(build-\d+)", line).group(1)

ans = {"q1_deploy": q1, "q2_degraded_minutes": deg, "q3_timeout_ms": val, "q4_failed_orders": len(orders),
       "q5_changed_key": q5, "q6_ack": {"who": who, "minutes": mins}, "control_build": ctl}
if MODE == "retrieval":   # gave up on synthesis, answered only the control
    ans = {"control_build": ctl}
json.dump(ans, open("ANSWERS.json", "w"), indent=2)
