#!/usr/bin/env python3
"""Pre-score MATH training prompts by difficulty, so GRPO stops paying for dead groups.

  python grpo_11_prescore.py [--model M] [--n 4] [--limit 3000] [--out math_l35_pass.jsonl]

Why: a GRPO group carries gradient only when its completions DISAGREE. Measured on f1
(Qwen3-1.7B, MATH L3-5, G=4): `frac_reward_zero_std` averages 0.67, i.e. **2.7 of the 4
prompts in every step produce zero signal** and in 21 of 93 steps all four were dead.
Two thirds of the rollout budget buys nothing. Prompts the model always solves and prompts
it never solves are both worthless; the informative band is in between.

This is DAPO's dynamic sampling done offline (TRL has no online version, and returning
`None` from a reward function does not resample — it only marks the row unscorable).
One pass with the base model, then every later run reuses the file.

Output: one json per line, `{"problem": <text>, "pass": <k>, "n": <N>}`. The training
script filters on it with GRPO_PASS_FILTER=<file>, keeping 0 < pass < n.

Sampling matches training (temperature 1.0, same prompt), so the pass rate estimates the
same distribution GRPO will sample from. Prompts are taken in the training script's own
shuffled order (seed 3407), so `--limit N` pre-scores exactly the prompts the next runs
will actually consume.
"""
import argparse
import json
import os
import re

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# flashinfer JIT: nvcc (CUDA 13.3) refuses host gcc > 15 and takes -ccbin from $CC
os.environ.setdefault("CC", "/usr/x86_64-pc-linux-gnu/gcc-bin/15/gcc")
os.environ.setdefault("CXX", "/usr/x86_64-pc-linux-gnu/gcc-bin/15/g++")

SYSTEM_PROMPT = "You are a careful math solver."
USER_TEMPLATE = ("/no_think\n\nSolve the problem step by step. "
                 "Put the final answer in \\boxed{}.\n\nProblem:\n{problem}")


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


def is_correct(text: str, gold_boxed: str) -> int:
    """Same rule as the training reward: math_verify, with the hedging gate."""
    from math_verify import parse, verify, LatexExtractionConfig
    if len(re.findall(r"\\boxed\s*{", text)) > 1:
        contents, s = set(), text
        while True:
            b = last_boxed(s)
            if b is None:
                break
            contents.add(b.replace(" ", ""))
            s = s[:s.rfind("\\boxed")]
        if len(contents) > 1:
            return 0
    try:
        pred = parse(text)
    except Exception:
        return 0
    if not pred:
        return 0
    try:
        gold = parse(f"$\\boxed{{{gold_boxed}}}$",
                     extraction_config=[LatexExtractionConfig(boxed_match_priority=0)])
        return int(verify(gold, pred))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=4, help="samples per prompt")
    ap.add_argument("--limit", type=int, default=3000, help="prompts to score (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--levels", default="3,4,5")
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--out", default="math_l35_pass.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    levels = {f"Level {l.strip()}" for l in args.levels.split(",")}
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval")["train"]
    ds = ds.filter(lambda x: x["level"] in levels)
    ds = ds.filter(lambda x: last_boxed(x["solution"]) is not None)
    ds = ds.shuffle(seed=3407)  # same order the training script consumes
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"scoring {len(ds)} prompts x {args.n} samples with {args.model}", flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gmu,
              max_model_len=512 + args.max_tokens, dtype="bfloat16",
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": USER_TEMPLATE.replace("{problem}", x["problem"])}],
        tokenize=False, add_generation_prompt=True) for x in ds]
    # Drop prompts longer than the training script's max_prompt_length. Without this the
    # engine rejects the batch outright ("maximum context length is 1536 tokens"), and the
    # scored set would not match what training actually consumes anyway.
    keep = [i for i, p in enumerate(prompts) if len(tok(p)["input_ids"]) <= 512]
    if len(keep) != len(prompts):
        print(f"dropping {len(prompts) - len(keep)} prompts longer than 512 tokens", flush=True)
        ds = ds.select(keep)
        prompts = [prompts[i] for i in keep]

    sp = SamplingParams(n=args.n, temperature=1.0, max_tokens=args.max_tokens, seed=3407)
    outs = llm.generate(prompts, sp)

    hist = {}
    with open(args.out, "w") as f:
        for x, o in zip(ds, outs):
            gold = last_boxed(x["solution"])
            k = sum(is_correct(c.text, gold) for c in o.outputs)
            hist[k] = hist.get(k, 0) + 1
            f.write(json.dumps({"problem": x["problem"], "pass": k, "n": args.n,
                                "level": x["level"]}) + "\n")
    total = sum(hist.values())
    live = total - hist.get(0, 0) - hist.get(args.n, 0)
    print(f"\npass-count histogram (of {args.n}): {dict(sorted(hist.items()))}")
    print(f"always wrong: {hist.get(0, 0)} ({hist.get(0, 0) / total:.0%})   "
          f"always right: {hist.get(args.n, 0)} ({hist.get(args.n, 0) / total:.0%})   "
          f"**informative: {live} ({live / total:.0%})**")
    print(f"written {args.out}")

    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    import gc
    gc.collect()
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
