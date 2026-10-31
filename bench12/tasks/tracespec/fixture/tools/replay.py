#!/usr/bin/env python3
"""Replay recorded v1 production traces against `python -m meter bill`.

Each traces/*.trace file has a `# input` block (an event log) and a
`# expected` block (the exact output v1 produced in production).
Usage: python3 tools/replay.py [TRACE...]
"""
import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split(path):
    text = path.read_text()
    inp, exp = text.split("# expected\n", 1)
    return inp.replace("# input\n", "", 1), exp


def main(argv):
    paths = [Path(p) for p in argv] or sorted((ROOT / "traces").glob("*.trace"))
    bad = 0
    for p in paths:
        inp, exp = split(p)
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(inp)
        got = subprocess.run([sys.executable, "-m", "meter", "bill", f.name], cwd=ROOT,
                             capture_output=True, text=True)
        Path(f.name).unlink()
        if got.stdout == exp:
            print(f"ok    {p.name}")
        else:
            bad += 1
            print(f"DIFF  {p.name}")
            for d in difflib.unified_diff(exp.splitlines(), got.stdout.splitlines(), "expected", "got", lineterm=""):
                print("      " + d)
            if got.stderr.strip():
                print("      stderr: " + got.stderr.strip().splitlines()[-1])
    print(f"{len(paths) - bad}/{len(paths)} traces match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
