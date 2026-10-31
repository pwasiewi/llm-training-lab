#!/usr/bin/env python3
"""chain14 fixture: the shared 14-table chain, CHAIN.md lists the first 14 steps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import chaingen  # noqa: E402
chaingen.write(sys.argv[1], 14)
