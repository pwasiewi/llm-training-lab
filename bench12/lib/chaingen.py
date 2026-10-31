"""Generator for the bench_12 chain family (chain6 / chain10 / chain14).

One 14-step chain is generated from a fixed seed; variant N lists only the
first N steps in CHAIN.md. All 14 tables and OPS.md ship with EVERY
variant (fill-matched: the amount of material to read does not grow with N).

A step: look x up in a range table -> row; take one named field -> v;
x := OP(v, x). Every table covers 0..999999 completely, so a wrong x never
produces a "not found" — errors are silent, as in real pipelines.

Each table carries one convention trap placed exactly at the true x of the
previous step, so reading the table with the OPS.md defaults (or with the
bundled tools/rangeq.py, which ignores headers) yields a different row.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

U = 1_000_000
STEPS = 14
FIELDS = ("alpha", "beta", "gamma")
TRAPS = ("halfopen", "hexvalues", "lastwins", "colorder", "csvquoted", "activeonly", "narrowhex")
OPS = ("id", "rev6", "dsub", "drot7", "xor", "mix", "swap3")
START = 271828
# step-local notes (CHAIN.md "note" column), spread evenly so every prefix N
# carries a proportional share: 6 -> 2, 10 -> 3, 14 -> 4
NOTES = {3: "keyswap", 6: "parity", 9: "plushi", 12: "keyswap"}
NOTE_TEXT = {
    "keyswap": "look up swap3(x) instead of x; the operation still uses the unswapped x",
    "parity": "ignore the field column: use alpha when x is even, gamma when x is odd",
    "plushi": "v is (field + hi of the matched row) mod 1000000, hi read in the table's key base",
    "": "",
}


def seed() -> int:
    return int(os.environ.get("B12_SEED", "12"))


# ---------------------------------------------------------------- ops -----

def p6(n: int) -> str:
    return f"{n % U:06d}"


def apply_op(op: str, v: int, x: int) -> int:
    if op == "id":
        return v
    if op == "rev6":
        return int(p6(v)[::-1])
    if op == "dsub":
        return (v - x) % U
    if op == "drot7":
        return int("".join(str((int(c) + 7) % 10) for c in p6(v)))
    if op == "xor":
        return (v ^ 0x5A5A5) % U
    if op == "mix":
        return (v + 3 * x) % U
    if op == "swap3":
        s = p6(v)
        return int(s[3:] + s[:3])
    raise ValueError(op)


OP_TEXT = {
    "id": "x := v",
    "rev6": "x := rev6(v)",
    "dsub": "x := dsub(v, x)",
    "drot7": "x := drot7(v)",
    "xor": "x := xor(v)",
    "mix": "x := mix(v, x)",
    "swap3": "x := swap3(v)",
}


def op_ok(op: str, v: int) -> bool:
    """Value constraints that make the op's own trap bite."""
    if op == "rev6":  # leading/trailing zeros matter
        return v % 10 == 0 and v < 100000
    return True


# ------------------------------------------------------------- tables -----

def cuts(rng: random.Random, n: int, must: int | None = None) -> list[int]:
    pts = set(rng.sample(range(1, U), n - 1))
    if must is not None:
        pts.add(must)
    return [0] + sorted(pts) + [U]


def rand_val(rng: random.Random) -> int:
    return rng.randrange(U)


def digit_hex_val(rng: random.Random) -> int:
    """An int whose 6-hex-digit form uses only 0-9 (so it also parses as decimal)."""
    # leading 0 keeps the value below 1000000 (0x099999 = 629145)
    return int("0" + "".join(rng.choice("0123456789") for _ in range(5)), 16)


class RetryTable(Exception):
    pass


def build_table(rng: random.Random, trap: str, x: int, field: str, op: str):
    """Return (text, v_true, v_default_reading)."""
    nrows = rng.randrange(140, 170)
    hdr = []
    rows = []  # list of dicts: lo hi alpha beta gamma (+ note/state)
    sep = "\t"
    cols = ["lo", "hi", "alpha", "beta", "gamma"]
    keyfmt = lambda n: f"{n:06d}"  # noqa: E731
    valfmt = lambda n: f"{n:06d}"  # noqa: E731

    def pick_val(constraint=lambda v: True, gen=rand_val):
        while True:
            v = gen(rng)
            if op_ok(op, v) and constraint(v):
                return v

    if trap == "halfopen":
        hdr.append("#ranges: half-open")
        c = cuts(rng, nrows, must=x)
        for i in range(len(c) - 1):
            rows.append(dict(lo=c[i], hi=c[i + 1], alpha=rand_val(rng), beta=rand_val(rng), gamma=rand_val(rng)))
        tgt = next(r for r in rows if r["lo"] == x)
        tgt[field] = pick_val()
        wrong = next(r for r in rows if r["hi"] == x)  # inclusive + first-match reading
    elif trap == "hexvalues":
        hdr.append("#values: hex")
        valfmt = lambda n: f"{n:06x}"  # noqa: E731
        c = cuts(rng, nrows)
        for i in range(len(c) - 1):
            rows.append(dict(lo=c[i], hi=c[i + 1] - 1, alpha=digit_hex_val(rng), beta=digit_hex_val(rng),
                             gamma=digit_hex_val(rng)))
        tgt = next(r for r in rows if r["lo"] <= x <= r["hi"])
        tgt[field] = pick_val(gen=digit_hex_val)
        wrong = None  # same row, value parsed as decimal
    elif trap == "lastwins":
        hdr.append("#overlap: last-wins")
        c = cuts(rng, nrows)
        for i in range(len(c) - 1):
            rows.append(dict(lo=c[i], hi=c[i + 1] - 1, alpha=rand_val(rng), beta=rand_val(rng), gamma=rand_val(rng)))
        base = next(r for r in rows if r["lo"] <= x <= r["hi"])
        lo = max(0, x - rng.randrange(50, 3000))
        hi = min(U - 1, x + rng.randrange(50, 3000))
        tgt = dict(lo=lo, hi=hi, alpha=rand_val(rng), beta=rand_val(rng), gamma=rand_val(rng))
        tgt[field] = pick_val()
        # a few harmless overlap rows elsewhere, the trap row among them, all after the base
        extra = []
        for _ in range(6):
            a = rng.randrange(U - 5000)
            if a <= x <= a + 5000:
                continue
            extra.append(dict(lo=a, hi=a + rng.randrange(100, 5000), alpha=rand_val(rng), beta=rand_val(rng),
                              gamma=rand_val(rng)))
        extra.insert(rng.randrange(len(extra) + 1), tgt)
        rows.extend(extra)
        wrong = base
    elif trap == "colorder":
        cols = ["lo", "hi", "gamma", "alpha", "beta"]
        c = cuts(rng, nrows)
        for i in range(len(c) - 1):
            rows.append(dict(lo=c[i], hi=c[i + 1] - 1, alpha=rand_val(rng), beta=rand_val(rng), gamma=rand_val(rng)))
        tgt = next(r for r in rows if r["lo"] <= x <= r["hi"])
        tgt[field] = pick_val()
        wrong = "positional"
    elif trap == "csvquoted":
        hdr.append("#format: csv")
        sep = ","
        cols = ["lo", "hi", "note", "alpha", "beta", "gamma"]
        c = cuts(rng, nrows)
        words = ["cold", "storage", "batch", "eu", "rack", "spare", "legacy", "pool", "west", "tier"]
        for i in range(len(c) - 1):
            w = rng.sample(words, 2)
            note = f"{w[0]} {w[1]}" if rng.random() < 0.6 else f"\"{w[0]}, {rng.randrange(U):06d}\""
            rows.append(dict(lo=c[i], hi=c[i + 1] - 1, note=note, alpha=rand_val(rng), beta=rand_val(rng),
                             gamma=rand_val(rng)))
        tgt = next(r for r in rows if r["lo"] <= x <= r["hi"])
        tgt["note"] = f"\"{rng.choice(words)}, {rng.randrange(U):06d}\""
        tgt[field] = pick_val()
        wrong = "naivesplit"
    elif trap == "activeonly":
        hdr.append("#active-only: yes")
        cols = ["lo", "hi", "alpha", "beta", "gamma", "state"]
        c = cuts(rng, nrows)
        for i in range(len(c) - 1):
            rows.append(dict(lo=c[i], hi=c[i + 1] - 1, alpha=rand_val(rng), beta=rand_val(rng), gamma=rand_val(rng),
                             state="on"))
        idx = next(i for i, r in enumerate(rows) if r["lo"] <= x <= r["hi"])
        tgt = rows[idx]
        tgt[field] = pick_val()
        off = dict(lo=max(0, x - rng.randrange(10, 900)), hi=min(U - 1, x + rng.randrange(10, 900)),
                   alpha=rand_val(rng), beta=rand_val(rng), gamma=rand_val(rng), state="off")
        rows.insert(idx, off)
        # some other off rows for texture
        for _ in range(5):
            j = rng.randrange(len(rows))
            r = rows[j]
            if r["state"] == "on" and r is not tgt:
                rows.insert(j, dict(lo=r["lo"], hi=r["hi"], alpha=rand_val(rng), beta=rand_val(rng),
                                    gamma=rand_val(rng), state="off"))
        wrong = off
    elif trap == "narrowhex":
        hdr.append("#keys: hex")
        hdr.append("#overlap: narrowest-wins")
        keyfmt = lambda n: f"{n:05x}"  # noqa: E731
        c = cuts(rng, nrows)
        for i in range(len(c) - 1):
            rows.append(dict(lo=c[i], hi=c[i + 1] - 1, alpha=rand_val(rng), beta=rand_val(rng), gamma=rand_val(rng)))
        wide = dict(lo=max(0, x - rng.randrange(20000, 60000)), hi=min(U - 1, x + rng.randrange(20000, 60000)),
                    alpha=rand_val(rng), beta=rand_val(rng), gamma=rand_val(rng))
        tgt = dict(lo=max(0, x - rng.randrange(0, 8)), hi=min(U - 1, x + rng.randrange(0, 8)), alpha=rand_val(rng), beta=rand_val(rng),
                   gamma=rand_val(rng))
        tgt[field] = pick_val()
        if any(r["lo"] <= x <= r["hi"] and r["hi"] - r["lo"] <= tgt["hi"] - tgt["lo"] for r in rows):
            raise RetryTable()
        rows.insert(0, wide)
        rows.insert(rng.randrange(len(rows) // 2, len(rows)), tgt)
        wrong = wide
    else:
        raise ValueError(trap)

    hdr.append("#cols: " + " ".join(cols) if sep == "\t" else "#cols: " + ",".join(cols))
    lines = []
    for r in rows:
        f = []
        for cname in cols:
            if cname in ("lo", "hi"):
                f.append(keyfmt(r[cname]))
            elif cname in FIELDS:
                f.append(valfmt(r[cname]))
            else:
                f.append(str(r[cname]))
        lines.append(sep.join(f))
    v_true = tgt[field]
    # what a header-blind reader (inclusive, decimal, first match, fixed column order) gets
    if wrong is None:
        v_wrong = int(f"{v_true:06x}")
    elif wrong == "positional":
        v_wrong = tgt[cols[["lo", "hi", "alpha", "beta", "gamma"].index(field)]]
    elif wrong == "naivesplit":
        v_wrong = None
    else:
        v_wrong = wrong[field]
    return "\n".join(hdr + lines) + "\n", v_true, v_wrong, tgt


def generate(sd: int | None = None):
    """Return (tables{name:text}, steps[list of dict], xs[x0..x14])."""
    rng = random.Random(1_000_003 * (seed() if sd is None else sd) + 7)
    order = list(range(STEPS))
    rng.shuffle(order)  # table file number != step number
    tables = {}
    steps = []
    x = START
    xs = [x]
    for k in range(STEPS):
        trap = TRAPS[k % len(TRAPS)]
        op = OPS[(k * 3 + k // len(TRAPS)) % len(OPS)]  # pairing shifts in cycle 2
        field = rng.choice(FIELDS)
        note = NOTES.get(k + 1, "")
        if note == "parity":
            field = "alpha" if x % 2 == 0 else "gamma"
        key = int(p6(x)[3:] + p6(x)[:3]) if note == "keyswap" else x
        tname = f"t{order[k] + 1:02d}." + ("csv" if trap == "csvquoted" else "tsv")
        while True:
            try:
                text, v, vw, tgt = build_table(rng, trap, key, field, op)
                break
            except RetryTable:
                continue
        tables[tname] = text
        if note == "plushi":
            v = (v + tgt["hi"]) % U
        nx = apply_op(op, v, x)
        steps.append(dict(k=k + 1, table=tname, field=field if note != "parity" else "-", op=op, trap=trap,
                          note=note, v=v, x_in=x, x_out=nx, v_wrong=vw))
        x = nx
        xs.append(x)
    return tables, steps, xs


OPS_MD = """# OPS.md — range tables and step operations

## 1. Tokens

A token x is an integer 0 <= x <= 999999. In text it is written as a
6-digit zero-padded decimal ("004512"). Every chain starts from the start
token given in CHAIN.md.

## 2. Range tables (tables/*)

Default format, which applies unless the table's own header says
otherwise (a header directive ALWAYS overrides the defaults below):

- Lines starting with `#` are header/comment lines; blank lines are ignored.
- `#cols:` names the columns, in file order. Columns are tab-separated.
- `lo` and `hi` are written in decimal. A row matches x when lo <= x <= hi
  (both ends inclusive).
- Value columns (`alpha`, `beta`, `gamma`) are decimal integers.
- When several rows match, the FIRST matching row in file order wins.

Header directives that may appear in a table:

| directive | meaning |
|---|---|
| `#ranges: half-open` | a row matches when lo <= x < hi |
| `#keys: hex` | `lo` and `hi` are hexadecimal (no 0x prefix) |
| `#values: hex` | value columns are hexadecimal (no 0x prefix), even when they contain only digits |
| `#overlap: last-wins` | when several rows match, the LAST matching row wins |
| `#overlap: narrowest-wins` | when several rows match, the row with the smallest hi-lo wins (ties: first in file) |
| `#format: csv` | comma-separated, RFC 4180 quoting: a field in double quotes may contain commas |
| `#active-only: yes` | rows whose `state` column is not `on` are ignored entirely |

Every table covers the whole token space; a lookup never fails.

## 3. Operations

v is the value taken from the matched row, x is the current token. All
results are reduced modulo 1000000.

| op | result |
|---|---|
| `x := v` | v |
| `x := rev6(v)` | write v as 6-digit zero-padded decimal, reverse the 6 characters, read back as an integer ("004510" -> "015400" -> 15400) |
| `x := dsub(v, x)` | (v - x) mod 1000000 (never negative) |
| `x := drot7(v)` | write v as 6-digit zero-padded decimal, replace every digit d by (d + 7) mod 10 |
| `x := xor(v)` | (v XOR 0x5A5A5) mod 1000000 |
| `x := mix(v, x)` | (v + 3 * x) mod 1000000 |
| `x := swap3(v)` | write v as 6-digit zero-padded decimal and swap its two 3-digit halves ("123456" -> "456123") |

## 4. Recording

Write one line per step to `chain.log`, exactly `step NN: XXXXXX`
(NN = 2-digit step number, XXXXXX = the token AFTER the step, 6-digit
zero-padded). Write the final token alone (6 digits) to `ANSWER`.
"""

RANGEQ = '''#!/usr/bin/env python3
"""rangeq.py TABLE TOKEN [FIELD] — quick lookup helper for range tables.

Supports tables in the DEFAULT format of OPS.md section 2 only: tab-separated,
decimal keys and values, inclusive ranges, first match wins. It does not
interpret header directives other than #cols.
"""
import sys


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return 2
    path, x = sys.argv[1], int(sys.argv[2])
    field = sys.argv[3] if len(sys.argv) > 3 else None
    cols = ["lo", "hi", "alpha", "beta", "gamma"]
    for line in open(path):
        line = line.rstrip("\\n")
        if line.startswith("#cols:"):
            cols = line[6:].replace(",", " ").split()
            continue
        if not line or line.startswith("#"):
            continue
        f = line.split("\\t")
        if len(f) < 2:
            continue
        try:
            lo, hi = int(f[0]), int(f[1])
        except ValueError:
            continue
        if lo <= x <= hi:
            row = dict(zip(cols, f))
            print(row.get(field, "") if field else "\\t".join(f))
            return 0
    print("no match", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
'''

TOOLS_README = """tools/
  rangeq.py  quick lookup for range tables in the default format (OPS.md section 2).
"""


def chain_md(steps, n: int) -> str:
    out = [
        "# CHAIN.md — dependent lookup chain",
        "",
        f"Start token: x = {START:06d}",
        "",
        f"Apply the following {n} steps in order. Each step looks the CURRENT token x up in",
        "the named table (conventions: OPS.md section 2 plus the table's own header),",
        "takes the named field of the matched row as v, then applies the operation",
        "(OPS.md section 3). The result is the token for the next step.",
        "",
        "| step | table | field | operation | note |",
        "|---|---|---|---|---|",
    ]
    for s in steps[:n]:
        out.append(f"| {s['k']:02d} | tables/{s['table']} | {s['field']} | `{OP_TEXT[s['op']]}` | "
                   f"{NOTE_TEXT[s['note']]} |")
    out += ["", "Tables not listed above belong to other chains; ignore them.", ""]
    return "\n".join(out)


def write(outdir: str | Path, n: int) -> None:
    out = Path(outdir)
    tables, steps, _ = generate()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tools").mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        (out / "tables" / name).write_text(text)
    (out / "OPS.md").write_text(OPS_MD)
    (out / "CHAIN.md").write_text(chain_md(steps, n))
    (out / "tools" / "rangeq.py").write_text(RANGEQ)
    (out / "tools" / "README").write_text(TOOLS_README)


if __name__ == "__main__":
    t, st, xs = generate()
    for s in st:
        print(s["k"], s["table"], s["trap"], s["op"], s["field"], f"{s['x_in']:06d}->{s['x_out']:06d}",
              "v", s["v"], "vwrong", s["v_wrong"])
