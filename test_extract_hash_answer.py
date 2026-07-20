#!/usr/bin/env python3
"""Audit extract_hash_answer() from grpo_02_qwen15b_gsm8k.py.

The function under test is extracted via AST from the training script itself
(importing the script would start training), so this always tests the live code.

Part 1: run it over every GSM8K ground-truth answer (train + test) and
compare against the canonical `split("####")[1]` value.
Part 2: synthetic model-output battery covering the formats a model may
emit inside <answer> tags ($10.000, $10 000, mixed fractions, newlines...).
Part 3: fallback quality — strip the "#### N" line and check how often the
last-number regex recovers the right answer from the reasoning alone.
"""
import ast
import re
import sys
from pathlib import Path

from datasets import load_dataset

SOURCE = Path(__file__).parent / "grpo_02_qwen15b_gsm8k.py"


def load_function(name: str = "extract_hash_answer"):
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"re": re}
            exec(compile(ast.Module([node], []), str(SOURCE), "exec"), ns)
            return ns[name]
    sys.exit(f"{name} not found in {SOURCE}")


extract_hash_answer = load_function()


def safe_float(x):
    try:
        return float(x.replace(",", "").strip())
    except Exception:
        return None


def canonical(answer_text: str) -> str:
    return answer_text.split("####")[1].strip().replace(",", "")


def part1_dataset():
    print("=" * 70)
    print("PART 1: ground-truth extraction over full GSM8K")
    failed = 0
    for split in ("train", "test"):
        data = load_dataset("openai/gsm8k", "main")[split]
        mismatch, unparseable, has_comma = [], [], []
        for i, row in enumerate(data):
            ext = extract_hash_answer(row["answer"])
            can = canonical(row["answer"])
            if ext is None or safe_float(ext) is None:
                unparseable.append((i, ext))
            if safe_float(ext) != safe_float(can):
                mismatch.append((i, ext, can))
            if ext is not None and "," in ext:
                has_comma.append((i, ext))
        print(f"\n[{split}] n={len(data)}")
        print(f"  value mismatches vs canonical : {len(mismatch)}")
        print(f"  unparseable extractions       : {len(unparseable)}")
        print(f"  extracted WITH comma kept     : {len(has_comma)}")
        for i, ext, can in mismatch[:10]:
            print(f"    MISMATCH idx={i}: extracted={ext!r} canonical={can!r}")
        failed += len(mismatch) + len(unparseable)
    return failed


SYNTHETIC = [
    # (model output, intended value, note)
    ("<answer>42</answer>", 42.0, "plain int in tags"),
    ("<answer>1,234,567</answer>", 1234567.0, "comma thousands in tags"),
    ("<answer>-5</answer>", -5.0, "negative in tags"),
    ("<answer>3.5</answer>", 3.5, "decimal in tags"),
    ("<answer>\n72\n</answer>", 72.0, "newlines inside tags (XML_COT_FORMAT emits this)"),
    ("<reasoning>\nHe pays 6*12=72.\n</reasoning>\n<answer>\n72\n</answer>", 72.0, "full trained format"),
    ("<answer>$72</answer>", 72.0, "currency symbol"),
    ("<answer>$10.000</answer>", 10.0, "dot = US decimal by convention (ground truths are all ints)"),
    ("<answer>$10 000</answer>", 10000.0, "space thousands"),
    ("<answer>1 000 000</answer>", 1000000.0, "double space thousands"),
    ("<answer>10 1/4</answer>", 10.25, "mixed fraction"),
    ("<answer>5 2/3</answer>", 5 + 2 / 3, "mixed fraction"),
    ("<answer>-10 1/4</answer>", -10.25, "negative mixed fraction"),
    ("<answer>1 0/0</answer>", None, "zero denominator must not crash"),
    ("<answer>72 dollars</answer>", 72.0, "number + unit"),
    ("<reasoning>She has 3 boxes with 8 each, 3*8=24, minus 4 eaten.</reasoning><answer>$20</answer>",
     20.0, "currency in tags, numbers in reasoning"),
    ("<reasoning>Total is 100. Half is 50.</reasoning><answer>fifty</answer>",
     50.0, "word answer -> falls to last number in WHOLE text"),
    ("The answer is 20000.", 20000.0, "no tags at all"),
]


def part2_synthetic():
    print("\n" + "=" * 70)
    print("PART 2: synthetic model outputs")
    print(f"{'note':<55} {'extracted':>12} {'intended':>10}  verdict")
    failed = 0
    for text, intended, note in SYNTHETIC:
        ext = extract_hash_answer(text)
        got = safe_float(ext) if ext is not None else None
        ok = (got == intended) if intended is not None else True
        failed += not ok
        print(f"{note:<55} {str(ext):>12} {str(intended):>10}  {'OK' if ok else 'WRONG'}")
    return failed


def part3_fallback():
    print("\n" + "=" * 70)
    print("PART 3: fallback (last-number regex) on reasoning text without '#### N'")
    data = load_dataset("openai/gsm8k", "main")["train"]
    good = sum(
        1 for row in data
        if (ext := extract_hash_answer(row["answer"].split("####")[0])) is not None
        and safe_float(ext) == safe_float(canonical(row["answer"]))
    )
    print(f"  last number in reasoning == final answer: {good}/{len(data)} ({100*good/len(data):.1f}%)")
    return 0


if __name__ == "__main__":
    failures = part1_dataset() + part2_synthetic() + part3_fallback()
    print("\n" + ("ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURES"))
    sys.exit(failures != 0)
