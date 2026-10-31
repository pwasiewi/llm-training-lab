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

--k K (2026-10-20) samples K completions per item (--temperature 0.6, --top-p 0.95, --seed 0)
and records each item's success rate. One greedy pass is not reproducible: two identical
base runs flip 6/200 items and the Sept->Oct stack upgrade moved the g1-vs-base delta from
+2.2 pp to -1.0 pp. Two --k files are compared with a paired sign-flip permutation test on the
per-item rate differences plus a bootstrap CI. Use the same --seed for both models so they
face the same random draws (common random numbers tighten the paired test).

--item-seeds (2026-10-20) gives every item its own derived seed (grpo_runlib.derive_seed(seed, item_id)), so an
item's draws do not depend on batch composition. Files made with and without it must not be compared.
With --lora the adapter's run.json (written by the trainer) is read: an adapter trained under a different
grpo_11_task.py (TASK_SHA) is refused unless --allow-task-drift. Every eval also writes OUT.meta.json.

Everything executable sits under __main__: vLLM V1 spawns workers that re-import this file.
"""
import argparse, json, os, re, sys
import sys as _sys
from math import comb

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CC", "/usr/x86_64-pc-linux-gnu/gcc-bin/16/gcc")   # flashinfer JIT takes -ccbin from $CC (CUDA 13.4.3, gcc 16)
os.environ.setdefault("CXX", "/usr/x86_64-pc-linux-gnu/gcc-bin/16/g++")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Prompt, answer extraction and scoring are shared with the trainer (grpo_11_task.py, 2026-10-20),
# so train and eval can never drift; TASK_SHA is checked against the adapter's run.json.
from grpo_11_task import SYSTEM_PROMPT, USER_TEMPLATE, TASK_SHA, build_prompt, last_boxed, is_correct as _task_is_correct
import grpo_runlib as RL


def is_correct(text: str, gold_boxed: str):
    c, st = _task_is_correct(text, gold_boxed)
    return int(c), st


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = A right & B wrong, c = A wrong & B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare_k(ra: dict, rb: dict, ids: list, path_a: str, path_b: str):
    """Paired comparison of per-item success rates from two --k evals."""
    import random
    pa = [ra[i]["rate"] for i in ids]
    pb = [rb[i]["rate"] for i in ids]
    d = [b - a for a, b in zip(pa, pb)]
    n = len(d)
    mean = sum(d) / n
    rng = random.Random(12345)
    R = 20000
    obs = abs(mean)
    hits = 0
    for _ in range(R):
        if abs(sum(x if rng.random() < 0.5 else -x for x in d) / n) >= obs - 1e-12:
            hits += 1
    p = (hits + 1) / (R + 1)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    ka, kb = ra[ids[0]].get("k"), rb[ids[0]].get("k")
    print(f"paired items: {n}   k: A={ka} B={kb}")
    print(f"A {path_a}: mean pass@1 {sum(pa) / n:.2%}")
    print(f"B {path_b}: mean pass@1 {sum(pb) / n:.2%}   delta {100 * mean:+.2f} pp  "
          f"95% bootstrap CI [{100 * lo:+.2f}, {100 * hi:+.2f}] pp")
    print(f"items B better {sum(x > 0 for x in d)}, A better {sum(x < 0 for x in d)}, "
          f"tied {sum(x == 0 for x in d)}; paired sign-flip permutation p = {p:.4f}")
    for lvl in sorted({ra[i].get("level") for i in ids}):
        sub = [i for i in ids if ra[i].get("level") == lvl]
        print(f"  {lvl}: n={len(sub)} A {sum(ra[i]['rate'] for i in sub) / len(sub):.1%} "
              f"B {sum(rb[i]['rate'] for i in sub) / len(sub):.1%}")


def compare(path_a: str, path_b: str):
    ra = {json.loads(l)["id"]: json.loads(l) for l in open(path_a)}
    rb = {json.loads(l)["id"]: json.loads(l) for l in open(path_b)}
    ids = sorted(set(ra) & set(rb))
    if all("rate" in ra[i] and "rate" in rb[i] for i in ids):
        return compare_k(ra, rb, ids, path_a, path_b)
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
    ap.add_argument("--k", type=int, default=0,
                    help="sample K completions per item and record its success rate (0 = one greedy pass)")
    ap.add_argument("--temperature", type=float, default=0.6, help="sampling temperature for --k")
    ap.add_argument("--top-p", type=float, default=0.95, help="top-p for --k")
    ap.add_argument("--seed", type=int, default=0, help="per-request sampling seed for --k")
    ap.add_argument("--item-seeds", action="store_true",
                    help="with --k: one derived seed per item (draws independent of batch composition)")
    ap.add_argument("--allow-task-drift", action="store_true",
                    help="evaluate an adapter whose run.json records a different task_sha")
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
    run_manifest = None
    if args.lora:
        for cand in (os.path.join(args.lora, "run.json"), os.path.join(os.path.dirname(os.path.abspath(args.lora)), "run.json")):
            if os.path.isfile(cand):
                run_manifest = json.load(open(cand))
                break
        if run_manifest is None:
            print("NOTE: no run.json next to the adapter (trained before 2026-10-20): task drift cannot be checked", flush=True)
        elif run_manifest.get("task_sha") != TASK_SHA and not args.allow_task_drift:
            sys.exit(f"adapter trained with task_sha={run_manifest.get('task_sha')}, this eval has {TASK_SHA}: "
                     "prompt/scoring changed since training (use --allow-task-drift to override)")

    llm = LLM(model=args.model, gpu_memory_utilization=args.gmu, max_model_len=512 + args.max_tokens,
              dtype="bfloat16", enable_lora=bool(args.lora), max_lora_rank=args.lora_rank)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(build_prompt(x["problem"]), tokenize=False, add_generation_prompt=True)
               for x in ds]
    if args.k and args.item_seeds:
        sp = [SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                             seed=RL.derive_seed(args.seed, x["unique_id"]), max_tokens=args.max_tokens)
              for x in ds]
    elif args.k:
        sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                            seed=args.seed, max_tokens=args.max_tokens)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    # The training-time dev eval applies the adapter at runtime, not merged into the base.
    # Both paths must be measurable to tell a real gain from a merge artefact (g1, 2026-09-09).
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("eval", 1, os.path.abspath(args.lora))
        print(f"applying LoRA at runtime: {lora_req.lora_path}", flush=True)
    outs = llm.generate(prompts, sp, lora_request=lora_req)

    with open(out_path + ".meta.json", "w") as fh:
        json.dump({"model": args.model, "lora": args.lora, "k": args.k, "temperature": args.temperature,
                   "top_p": args.top_p, "seed": args.seed, "item_seeds": args.item_seeds,
                   "max_tokens": args.max_tokens, "n_items": len(ds), "task_sha": TASK_SHA,
                   "adapter_run": {k: run_manifest.get(k) for k in ("config", "overrides", "task_sha", "status")}
                   if run_manifest else None,
                   "environment": RL.environment_snapshot(os.path.dirname(os.path.abspath(__file__)))},
                  fh, indent=1, default=str)
    n_ok = 0
    status_count = {}
    with open(out_path, "w") as f:
        for x, o in zip(ds, outs):
            text = o.outputs[0].text
            gold = x["answer"]
            c, st = is_correct(text, gold)
            extra = {}
            if args.k:
                scored = [is_correct(s.text, gold) for s in o.outputs]
                extra = {"k": len(scored), "hits": [h for h, _ in scored],
                         "rate": sum(h for h, _ in scored) / len(scored),
                         "statuses": [t for _, t in scored]}
                n_ok += extra["rate"]
                for _, t in scored:
                    status_count[t] = status_count.get(t, 0) + 1
            else:
                n_ok += c
                status_count[st] = status_count.get(st, 0) + 1
            f.write(json.dumps({"id": x["unique_id"], "level": x["level"], "subject": x["subject"],
                                "correct": c, "status": st, "gold": gold, **extra,
                                "pred_boxed": last_boxed(text), "tokens": len(o.outputs[0].token_ids),
                                "finish": o.outputs[0].finish_reason,
                                # kept for offline reward analysis (grpo_11_analyze.py)
                                "completion": text, "solution": x["solution"]}) + "\n")
    what = f"mean pass@1 over k={args.k} (T={args.temperature}, seed={args.seed})" if args.k else "greedy"
    print(f"model {args.model} [{what}]: {n_ok:.2f}/{len(ds)} = {n_ok / len(ds):.2%}  status={status_count}")
    by_level = {}
    for l in open(out_path):
        r = json.loads(l)
        by_level.setdefault(r["level"], []).append(r.get("rate", r["correct"]))
    for lvl in sorted(by_level):
        v = by_level[lvl]
        print(f"  level {lvl}: {sum(v):.2f}/{len(v)} = {sum(v) / len(v):.1%}")
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
