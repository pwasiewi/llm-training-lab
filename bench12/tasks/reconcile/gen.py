#!/usr/bin/env python3
"""reconcile fixture: static repo (fixture/) + a 20K-payment sample in data/."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recgen  # noqa: E402

SEED = int(os.environ.get("B12_SEED", "12"))
SAMPLE_N = 20000

if __name__ == "__main__":
    recgen.write(Path(sys.argv[1]) / "data", SAMPLE_N, SEED * 101 + 1)
