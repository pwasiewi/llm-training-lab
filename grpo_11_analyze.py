#!/usr/bin/env python3
"""Offline reward analysis for the grpo_11 family — no GPU needed.

  python grpo_11_analyze.py eval_math500_base.jsonl [more.jsonl ...]   # eval files
  python grpo_11_analyze.py grpo11_e0_run.log [more.log ...]           # training logs
  python grpo_11_analyze.py --diff a_run.log b_run.log                 # paired A/B of two runs

Answers the question a training run cannot answer cheaply: does a candidate PARTIAL-CREDIT
reward actually separate correct from wrong completions on this dataset? A shaping term
that scores wrong answers as highly as right ones adds noise, not signal, and burns
GPU-hours to find that out. Here it costs a second.

Input: an eval jsonl written by grpo_11_qwen3_17b_math_test.py (needs the `completion` and
`solution` fields; evals written before 2026-09-08 lack them and are skipped with a note).

Reports per file:
  * accuracy overall and per level, status split
  * step-credit reward (the `step_reward` of grpo_11): mean for correct vs wrong
    completions, the gap, AUC (probability that a random correct completion scores above a
    random wrong one; 0.5 = worthless, 1.0 = perfect), and how often it is exactly 0 or 1
  * the same for a length-normalised variant, to show whether the credit is just
    "longer answers mention more numbers"
  * gold-intermediate-count distribution (rows with none get no credit at all)
"""
import json
import re
import sys
from statistics import mean

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def last_boxed(s: str):
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


def gold_numbers(solution: str, final: str) -> list[str]:
    """Same rule as grpo_11_qwen3_17b_math.py: distinct numeric literals of the gold
    solution, minus the ones already in the final answer, minus 0/1/2."""
    final_nums = set(_NUM_RE.findall(final or ""))
    seen, out = set(), []
    for n in _NUM_RE.findall(solution):
        if n in final_nums or n in seen or n in ("0", "1", "2"):
            continue
        seen.add(n)
        out.append(n)
    return out


def step_credit(completion: str, gold: list[str]) -> float | None:
    if not gold:
        return None
    nums = set(_NUM_RE.findall(completion))
    hit = sum(1 for g in gold if g in nums) / len(gold)
    spray = len(nums) > 1.5 * len(gold) + 3
    return hit * (0.5 if spray else 1.0)


def raw_hit(completion: str, gold: list[str]) -> float | None:
    """Same, without the anti-spray penalty — isolates what the penalty is doing."""
    if not gold:
        return None
    nums = set(_NUM_RE.findall(completion))
    return sum(1 for g in gold if g in nums) / len(gold)


def auc(pos: list[float], neg: list[float]) -> float:
    """P(random positive > random negative), ties counted as half."""
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    neg_sorted = sorted(neg)
    import bisect
    for p in pos:
        wins += bisect.bisect_left(neg_sorted, p)
        ties += bisect.bisect_right(neg_sorted, p) - bisect.bisect_left(neg_sorted, p)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def describe(name: str, scorer, rows) -> None:
    pos = [v for r in rows if r["correct"] and (v := scorer(r)) is not None]
    neg = [v for r in rows if not r["correct"] and (v := scorer(r)) is not None]
    if not pos or not neg:
        print(f"  {name}: not enough data")
        return
    print(f"  {name}: correct mean {mean(pos):.3f} (n={len(pos)})  wrong mean {mean(neg):.3f} "
          f"(n={len(neg)})  gap {mean(pos) - mean(neg):+.3f}  AUC {auc(pos, neg):.3f}")
    for label, vals in (("correct", pos), ("wrong", neg)):
        zero = sum(1 for v in vals if v == 0.0) / len(vals)
        full = sum(1 for v in vals if v >= 0.999) / len(vals)
        print(f"      {label}: exactly 0 in {zero:.0%}, full credit in {full:.0%}")


def analyze(path: str) -> None:
    rows = [json.loads(l) for l in open(path)]
    print(f"\n=== {path}  ({len(rows)} items) ===")
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"accuracy {acc:.2%}")
    by_level: dict = {}
    status: dict = {}
    for r in rows:
        by_level.setdefault(r["level"], []).append(r["correct"])
        status[r["status"]] = status.get(r["status"], 0) + 1
    for lvl in sorted(by_level):
        v = by_level[lvl]
        print(f"  level {lvl}: {sum(v)}/{len(v)} = {sum(v) / len(v):.1%}")
    print(f"status: {status}")

    if "completion" not in rows[0] or "solution" not in rows[0]:
        print("no completion/solution fields — rerun the eval to analyse rewards")
        return

    for r in rows:
        r["_gold_nums"] = gold_numbers(r["solution"], last_boxed(r["solution"]))
    counts = [len(r["_gold_nums"]) for r in rows]
    empty = sum(1 for c in counts if c == 0)
    print(f"gold intermediate numbers: median {sorted(counts)[len(counts) // 2]} "
          f"mean {mean(counts):.1f} max {max(counts)}; {empty} rows ({empty / len(rows):.0%}) have none "
          f"-> step credit returns None there")

    print("step-credit discrimination (correct vs wrong completions):")
    describe("step_reward (as implemented)", lambda r: step_credit(r["completion"], r["_gold_nums"]), rows)
    describe("raw hit rate (no anti-spray)", lambda r: raw_hit(r["completion"], r["_gold_nums"]), rows)

    lens = {True: [], False: []}
    for r in rows:
        lens[bool(r["correct"])].append(r["tokens"])
    print(f"  length: correct mean {mean(lens[True]):.0f} tok, wrong mean {mean(lens[False]):.0f} tok "
          f"(AUC {auc(lens[True], lens[False]):.3f} — if this is far from 0.5, credit may just track length)")


def _step_rows(path: str) -> list[dict]:
    import ast
    rows = []
    for line in open(path):
        line = line.strip()
        if not line.startswith("{'loss'"):
            continue
        try:
            rows.append(ast.literal_eval(line))
        except Exception:
            continue
    return rows


def diff_logs(path_a: str, path_b: str) -> None:
    """Paired step-by-step comparison of two runs.

    Every grpo_11 run shuffles the dataset with the same seed and consumes 4 fresh prompts
    per step, so step k of run A and step k of run B score the SAME prompts. That makes the
    per-step reward difference a paired measurement: prompt difficulty cancels, which the
    bucket means of a single run cannot do. Far more sensitive than a 500-item greedy eval.
    """
    ra, rb = _step_rows(path_a), _step_rows(path_b)
    n = min(len(ra), len(rb))
    if n < 5:
        print("not enough overlapping steps")
        return

    def rew(d):
        # correct_reward FIRST: 'reward' is the aggregated multi-term reward, so it is not
        # comparable between a run with one reward function and a run with two. The share of
        # correct answers is the task metric both runs optimise.
        for k in ("rewards/correct_reward/mean", "reward"):
            if k in d:
                try:
                    return float(d[k])
                except (TypeError, ValueError):
                    return None
        return None

    pairs = [(x, y) for i in range(n) if (x := rew(ra[i])) is not None and (y := rew(rb[i])) is not None]
    d = [y - x for x, y in pairs]
    m = mean(d)
    if len(d) > 1:
        var = sum((v - m) ** 2 for v in d) / (len(d) - 1)
        se = (var / len(d)) ** 0.5
    else:
        se = float("nan")
    wins = sum(1 for v in d if v > 0)
    losses = sum(1 for v in d if v < 0)
    print(f"\n=== paired diff over {len(d)} steps ===")
    print(f"A {path_a}: mean reward {mean(x for x, _ in pairs):.4f}")
    print(f"B {path_b}: mean reward {mean(y for _, y in pairs):.4f}")
    print(f"B - A = {m:+.4f} +- {se:.4f} (SE)   t = {m / se if se == se and se > 0 else float('nan'):+.2f}")
    print(f"steps where B scored higher: {wins}, lower: {losses}, tied: {len(d) - wins - losses}")
    print("|t| > 2 means the difference is unlikely to be step noise; anything less is a null result.")
    half = len(pairs) // 2
    for label, sl in (("first half", pairs[:half]), ("second half", pairs[half:])):
        if sl:
            print(f"  {label}: A {mean(x for x, _ in sl):.3f}  B {mean(y for _, y in sl):.3f}  "
                  f"delta {mean(y - x for x, y in sl):+.3f}")


def analyze_log(path: str, buckets: int = 8) -> None:
    """Reward trend of a training run: the per-step dicts TRL prints, bucketed.

    A single step's reward is noise (4 prompts); what matters is whether the bucket means
    rise. `frac_zero_std` is the share of groups with no gradient signal at all, so it is
    the direct measure of whether the data is still in the productive band."""
    import ast
    rows = []
    for line in open(path):
        line = line.strip()
        if not line.startswith("{'loss'"):
            continue
        try:
            rows.append(ast.literal_eval(line))
        except Exception:
            continue
    if not rows:
        print(f"\n=== {path} ===\nno training-step lines found")
        return
    print(f"\n=== {path}  ({len(rows)} steps) ===")

    def col(d, *names):
        for n in names:
            if n in d:
                try:
                    return float(d[n])
                except (TypeError, ValueError):
                    return float("nan")
        return float("nan")

    n = len(rows)
    size = max(1, n // buckets)
    print(f"{'steps':>12} {'reward':>8} {'zero_std':>9} {'len':>7} {'clipped':>8} {'|grad|':>8} {'loss':>9}")
    for i in range(0, n, size):
        chunk = rows[i:i + size]
        if not chunk:
            continue
        def m(*names):
            vals = [v for c in chunk if (v := col(c, *names)) == v]
            return mean(vals) if vals else float("nan")
        print(f"{i + 1:>5}-{min(i + size, n):<6} "
              f"{m('reward', 'rewards/correct_reward/mean'):>8.3f} "
              f"{m('frac_reward_zero_std'):>9.2f} "
              f"{m('completions/mean_length'):>7.0f} "
              f"{m('completions/clipped_ratio'):>8.2f} "
              f"{m('grad_norm'):>8.4f} "
              f"{m('loss'):>9.4f}")
    nan_steps = sum(1 for r in rows if col(r, 'grad_norm') != col(r, 'grad_norm'))
    if nan_steps:
        print(f"WARNING: grad_norm is NaN on {nan_steps}/{n} steps")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--diff":
        if len(sys.argv) != 4:
            print("usage: grpo_11_analyze.py --diff a_run.log b_run.log")
            sys.exit(1)
        diff_logs(sys.argv[2], sys.argv[3])
    else:
        for p in sys.argv[1:]:
            (analyze_log if p.endswith(".log") else analyze)(p)
