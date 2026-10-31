#!/usr/bin/env python3
"""tracespec fixture: static meter v2 repo (fixture/) + generated visible traces."""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prod  # noqa: E402

SEED = int(os.environ.get("B12_SEED", "12"))
VISIBLE_PER_GROUP = 3


def visible():
    out = []
    for g in prod.GROUPS:
        out += prod.traces(SEED * 31 + 1, g, VISIBLE_PER_GROUP)
    random.Random(SEED).shuffle(out)  # names must not reveal the behaviour
    return out


def hidden():
    return {g: prod.traces(SEED * 31 + 2, g, 6) for g in prod.GROUPS}


if __name__ == "__main__":
    d = Path(sys.argv[1]) / "traces"
    d.mkdir(parents=True, exist_ok=True)
    for i, lines in enumerate(visible(), 1):
        (d / f"prod-{i:03d}.trace").write_text(prod.trace_text(lines))
