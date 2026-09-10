#!/usr/bin/env python3
"""Greedy MATH-500 eval for the grpo_11 family (vLLM, math_verify), plus paired comparison.

  python grpo_11_qwen3_17b_math_test.py MODEL [--out results.jsonl] [--n N] [--max-tokens 1024]
  python grpo_11_qwen3_17b_math_test.py BASE --lora ADAPTER_DIR [--out results.jsonl]
  python grpo_11_qwen3_17b_math_test.py --compare base.jsonl trained.jsonl

MODEL is a HF id (Qwen/Qwen3-1.7B = base) or a merged model dir produced by grpo_merge.py.
The prompt is the training prompt of grpo_11 (no format reward, answer in \\boxed{}).
Scoring: math_verify(gold boxed, parse(completion)); two different boxed answers = wrong.

--compare runs an exact McNemar test on the discordant items of two result files:
a +2 pp gain is only visible in a PAIRED greedy comparison (binomial CI on 500 items
is +-3 pp), so never judge a run by two unpaired accuracies.

Everything executable sits under __main__: vLLM V1 spawns workers that re-import this file.
"""
import argparse, json, os, re, sys
import sys as _sys
from math import comb

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# flashinfer JIT-compiles its sampling kernels at engine start; nvcc (CUDA 13.3) refuses a
# host gcc newer than 15 and takes -ccbin from $CC. Without this the vLLM engine dies with
# "Ninja build failed ... unsupported GNU version" (grpo_11 eval, 2026-09-08).
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


def is_correct(text: str, gold_boxed: str):
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
            return 0, "hedged"
    try:
        pred = parse(text)
    except Exception:
        pred = []
    if not pred:
        return 0, "unparsed"
    try:
        gold = parse(f"$\\boxed{{{gold_boxed}}}$",
                     extraction_config=[LatexExtractionConfig(boxed_match_priority=0)])
        ok = verify(gold, pred)
    except Exception:
        ok = False
    return (1, "ok") if ok else (0, "wrong")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = A right & B wrong, c = A wrong & B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(path_a: str, path_b: str):
    ra = {json.loads(l)["id"]: json.loads(l) for l in open(path_a)}
    rb = {json.loads(l)["id"]: json.loads(l) for l in open(path_b)}
    ids = sorted(set(ra) & set(rb))
    a_only = sum(1 for i in ids if ra[i]["correct"] and not rb[i]["correct"])
    b_only = sum(1 for i in ids if rb[i]["correct"] and not ra[i]["correct"])
    acc_a = sum(ra[i]["correct"] for i in ids) / len(ids)
    acc_b = sum(rb[i]["correct"] for i in ids) / len(ids)
    p = mcnemar_exact(a_only, b_only)
    print(f"paired items: {len(ids)}")
    print(f"A {path_a}: {acc_a:.2%}")
    print(f"B {path_b}: {acc_b:.2%}   delta {100 * (acc_b - acc_a):+.2f} pp")
    print(f"discordant: A-only-right {a_only}, B-only-right {b_only}, exact McNemar p = {p:.4f}")
    for lvl in sorted({ra[i].get("level") for i in ids}):
        sub = [i for i in ids if ra[i].get("level") == lvl]
        print(f"  {lvl}: n={len(sub)} A {sum(ra[i]['correct'] for i in sub) / len(sub):.1%} "
              f"B {sum(rb[i]['correct'] for i in sub) / len(sub):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first N items")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gmu", type=float, default=0.8, help="vLLM gpu_memory_utilization")
    ap.add_argument("--lora", metavar="ADAPTER_DIR",
                    help="apply a LoRA adapter at runtime instead of using merged weights; "
                         "MODEL must then be the base the adapter was trained on")
    ap.add_argument("--lora-rank", type=int, default=16, help="max_lora_rank for --lora")
    ap.add_argument("--compare", nargs=2, metavar=("A.jsonl", "B.jsonl"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.model:
        ap.error("MODEL required unless --compare")

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    if args.n:
        ds = ds.select(range(args.n))
    out_path = args.out or f"eval_math500_{os.path.basename(args.model.rstrip('/'))}.jsonl"

    llm = LLM(model=args.model, gpu_memory_utilization=args.gmu, max_model_len=512 + args.max_tokens,
              dtype="bfloat16", enable_lora=bool(args.lora), max_lora_rank=args.lora_rank)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": USER_TEMPLATE.replace("{problem}", x["problem"])}],
        tokenize=False, add_generation_prompt=True) for x in ds]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    # The training-time dev eval applies the adapter at runtime, not merged into the base.
    # Both paths must be measurable to tell a real gain from a merge artefact (g1, 2026-09-09).
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("eval", 1, os.path.abspath(args.lora))
        print(f"applying LoRA at runtime: {lora_req.lora_path}", flush=True)
    outs = llm.generate(prompts, sp, lora_request=lora_req)

    n_ok = 0
    status_count = {}
    with open(out_path, "w") as f:
        for x, o in zip(ds, outs):
            text = o.outputs[0].text
            gold = x["answer"]
            c, st = is_correct(text, gold)
            n_ok += c
            status_count[st] = status_count.get(st, 0) + 1
            f.write(json.dumps({"id": x["unique_id"], "level": x["level"], "subject": x["subject"],
                                "correct": c, "status": st, "gold": gold,
                                "pred_boxed": last_boxed(text), "tokens": len(o.outputs[0].token_ids),
                                "finish": o.outputs[0].finish_reason,
                                # kept for offline reward analysis (grpo_11_analyze.py)
                                "completion": text, "solution": x["solution"]}) + "\n")
    print(f"model {args.model}: {n_ok}/{len(ds)} = {n_ok / len(ds):.2%}  status={status_count}")
    by_level = {}
    for l in open(out_path):
        r = json.loads(l)
        by_level.setdefault(r["level"], []).append(r["correct"])
    for lvl in sorted(by_level):
        v = by_level[lvl]
        print(f"  level {lvl}: {sum(v)}/{len(v)} = {sum(v) / len(v):.1%}")
    print(f"written {out_path}", flush=True)
    # vLLM V1 leaves its EngineCore child alive; without an explicit teardown this process
    # lingers holding the whole KV cache (13 GiB), which OOMs the next training run
    # (grpo11_e0_run.log, 2026-09-08). Shut the engine down, then exit hard.
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    import gc
    gc.collect()
    _sys.stdout.flush(); _sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
