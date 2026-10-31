"""Shared task definition for the grpo_11 family: prompt, answer extraction and scoring.

Imported by the trainer (grpo_11_qwen3_17b_math.py) and the evaluator (grpo_11_qwen3_17b_math_test.py)
so the two can never drift apart (Kauldron idea H5, llm/research/grpo-improvements-20261020.md).
Extracted verbatim from the trainer on 2026-10-20; the evaluator's copy was logically identical
(int instead of float returns only). TASK_SHA identifies this file's content: the trainer records it in
run.json and the evaluator refuses an adapter trained under a different task definition.
"""
import hashlib
import re

from math_verify import LatexExtractionConfig, parse, verify

SYSTEM_PROMPT = "You are a careful math solver."
USER_TEMPLATE = ("/no_think\n\nSolve the problem step by step. "
                 "Put the final answer in \\boxed{}.\n\nProblem:\n{problem}")

_GOLD_CFG = [LatexExtractionConfig(boxed_match_priority=0)]

def last_boxed(s: str) -> str | None:
    """Return the content of the last \\boxed{...} in s (brace-balanced), or None."""
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None

def is_correct(text: str, gold_boxed: str) -> tuple[float, str]:
    """1.0 if math_verify says the completion's answer equals the gold boxed answer.
    Hedging gate: two DIFFERENT boxed answers -> 0 (math_verify would silently take the last).
    Returns (reward, status) with status in {ok, wrong, hedged, unparsed}."""
    boxed = re.findall(r"\\boxed\s*{", text)
    if len(boxed) > 1:
        contents = set()
        s = text
        while True:
            b = last_boxed(s)
            if b is None:
                break
            contents.add(b.replace(" ", ""))
            s = s[:s.rfind("\\boxed")]
        if len(contents) > 1:
            return 0.0, "hedged"
    try:
        pred = parse(text)
    except Exception:
        pred = []
    if not pred:
        return 0.0, "unparsed"
    try:
        gold = parse(f"$\\boxed{{{gold_boxed}}}$", extraction_config=_GOLD_CFG)
        ok = verify(gold, pred)
    except Exception:
        ok = False
    return (1.0, "ok") if ok else (0.0, "wrong")


def build_prompt(problem: str) -> list[dict]:
    """Chat messages for one problem, exactly as the trainer feeds them."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.replace("{problem}", problem)},
    ]


with open(__file__, "rb") as _fh:
    TASK_SHA = hashlib.sha256(_fh.read()).hexdigest()[:16]
