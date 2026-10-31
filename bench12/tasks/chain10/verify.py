#!/usr/bin/env python3
"""chain10 verifier: SCORE = correct-prefix length of chain.log; PASS = all 10 steps + ANSWER + inputs intact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import chainverify  # noqa: E402
chainverify.verify(10)
