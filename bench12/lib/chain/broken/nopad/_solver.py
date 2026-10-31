#!/usr/bin/env python3
"""Independent chain solver: reads only CHAIN.md / OPS.md conventions / tables.

MODE (argv[1]) selects deliberate defects for the broken/ variants:
  ok          correct solution
  headerblind ignore every header directive except #cols (OPS.md defaults only)
  nopad       digit ops on the unpadded decimal string
  firstwins   ignore #overlap directives (first match always)
"""
import csv
import io
import re
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "ok"
U = 1_000_000


def digits(v):
    return str(v) if MODE == "nopad" else f"{v:06d}"


def op(name, v, x):
    if name == "x := v":
        return v % U
    if name == "x := rev6(v)":
        return int(digits(v)[::-1]) % U
    if name == "x := dsub(v, x)":
        return (v - x) % U
    if name == "x := drot7(v)":
        return int("".join(str((int(c) + 7) % 10) for c in digits(v))) % U
    if name == "x := xor(v)":
        return (v ^ 0x5A5A5) % U
    if name == "x := mix(v, x)":
        return (v + 3 * x) % U
    if name == "x := swap3(v)":
        s = digits(v)
        h = len(s) // 2
        return int(s[h:] + s[:h]) % U
    raise SystemExit(f"unknown op {name}")


def lookup(path, x, field, want_hi=False):
    text = Path(path).read_text()
    d = {}
    cols = None
    data = []
    for line in text.splitlines():
        if line.startswith("#"):
            m = re.match(r"#\s*([\w-]+)\s*:\s*(.*)$", line)
            if m:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key == "cols":
                    cols = re.split(r"[,\s]+", val)
                else:
                    d[key] = val.lower()
            continue
        if line.strip():
            data.append(line)
    if MODE == "headerblind":
        d = {}
    if MODE == "firstwins":
        d.pop("overlap", None)
    if d.get("format") == "csv":
        rows = list(csv.reader(io.StringIO("\n".join(data))))
    else:
        rows = [ln.split("\t") for ln in data] if MODE != "headerblind" or "," not in data[0] else \
            [ln.split(",") for ln in data]
    kb = 16 if d.get("keys") == "hex" else 10
    vb = 16 if d.get("values") == "hex" else 10
    matches = []
    for r in rows:
        row = dict(zip(cols, r))
        if d.get("active-only") == "yes" and row.get("state") != "on":
            continue
        lo, hi = int(row["lo"], kb), int(row["hi"], kb)
        ok = lo <= x < hi if d.get("ranges") == "half-open" else lo <= x <= hi
        if ok:
            matches.append((lo, hi, row))
    if not matches:
        raise SystemExit(f"no match for {x} in {path}")
    ov = d.get("overlap", "first-wins")
    if ov == "last-wins":
        pick = matches[-1]
    elif ov == "narrowest-wins":
        pick = min(matches, key=lambda m: m[1] - m[0])
    else:
        pick = matches[0]
    v = int(pick[2][field], vb)
    return (v + pick[1]) % U if want_hi else v


def main():
    chain = Path("CHAIN.md").read_text()
    x = int(re.search(r"Start token: x = (\d+)", chain).group(1))
    out = []
    for m in re.finditer(r"^\| (\d+) \| (tables/\S+) \| ([\w-]+) \| `([^`]+)` \| (.*?) \|$", chain, re.M):
        k, table, field, opname, note = int(m.group(1)), m.group(2), m.group(3), m.group(4), m.group(5)
        key = x
        if note.startswith("look up swap3(x)"):
            key = int(f"{x:06d}"[3:] + f"{x:06d}"[:3])
        if note.startswith("ignore the field column"):
            field = "alpha" if x % 2 == 0 else "gamma"
        v = lookup(table, key, field, want_hi=note.startswith("v is (field + hi"))
        x = op(opname, v, x)
        out.append(f"step {k:02d}: {x:06d}")
        Path("chain.log").write_text("\n".join(out) + "\n")
    Path("ANSWER").write_text(f"{x:06d}\n")


main()
