#!/usr/bin/env python3
"""Evaluate the saved checkpoints of a grpo_11 run after training (Kauldron idea M4, 2026-10-20).

  python grpo_eval_ckpts.py outputs/lora-grpo-qwen3-17b-math-TAG-outputs [--every 100] [--k 4] [--n 500]

Walks RUN_DIR/checkpoint-*/ (plus adapter-final), evaluates each adapter through vLLM with the k-sample
evaluator of grpo_11_qwen3_17b_math_test.py (one process per checkpoint, so every vLLM engine is torn down),
and appends one row per checkpoint to RUN_DIR/eval_curve.jsonl:
  {"step", "adapter", "k", "n", "mean_pass1", "file"}
Idempotent: a step already in eval_curve.jsonl is skipped, so a killed run resumes. Per-item results stay in
RUN_DIR/eval_ckpt/step-NNNN_k{K}.jsonl for paired comparisons (--compare in the eval script).
Never pick the "best" checkpoint from this curve: with +-1.5 pp eval noise that is selection bias. The curve
is for seeing WHEN a run stops improving, and the final adapter is still judged on its own.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "grpo_11_qwen3_17b_math_test.py")


def help_exit():
    print(__doc__)
    sys.exit(0)


def main():
    if len(sys.argv) == 1:
        help_exit()
    ap = argparse.ArgumentParser(description="evaluate saved grpo_11 checkpoints (see module docstring)")
    ap.add_argument("run_dir")
    ap.add_argument("--every", type=int, default=100, help="evaluate checkpoints whose step is a multiple of this")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=500, help="MATH-500 items (500 = all)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--no-final", action="store_true", help="skip adapter-final")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        sys.exit(f"no such run dir: {run_dir}")
    targets = []
    for d in os.listdir(run_dir):
        if d.startswith("checkpoint-") and d.split("-")[1].isdigit():
            step = int(d.split("-")[1])
            if step % args.every == 0:
                targets.append((step, os.path.join(run_dir, d)))
    final = os.path.join(run_dir, "adapter-final")
    if not args.no_final and os.path.isdir(final):
        targets.append((10 ** 9, final))          # sorts last; recorded as step "final"
    targets.sort()
    if not targets:
        sys.exit(f"no checkpoints with step % {args.every} == 0 and no adapter-final in {run_dir}")

    curve = os.path.join(run_dir, "eval_curve.jsonl")
    done = set()
    if os.path.isfile(curve):
        for line in open(curve):
            if line.strip():
                r = json.loads(line)
                if r.get("k") == args.k and r.get("n") == args.n:
                    done.add(r["step"])
    os.makedirs(os.path.join(run_dir, "eval_ckpt"), exist_ok=True)

    for step, adapter in targets:
        label = "final" if step == 10 ** 9 else step
        if label in done:
            print(f"[ckpt] step {label}: already in eval_curve.jsonl, skipped", flush=True)
            continue
        out = os.path.join(run_dir, "eval_ckpt", f"step-{label}_k{args.k}.jsonl")
        cmd = [sys.executable, EVAL, args.base, "--lora", adapter, "--k", str(args.k), "--item-seeds",
               "--seed", str(args.seed), "--max-tokens", str(args.max_tokens), "--out", out,
               "--allow-task-drift"]          # checkpoints of one run share its task; drift is checked on final evals
        if args.n and args.n < 500:
            cmd += ["--n", str(args.n)]
        print(f"[ckpt] step {label}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(f"[ckpt] eval of step {label} failed rc={rc}; earlier steps are kept, re-run resumes")
        rates = [json.loads(l)["rate"] for l in open(out) if l.strip()]
        row = {"step": label, "adapter": adapter, "k": args.k, "n": args.n,
               "mean_pass1": round(sum(rates) / len(rates), 5), "file": out}
        with open(curve, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[ckpt] step {label}: mean pass@1 {row['mean_pass1']:.2%} (k={args.k}, n={len(rates)})", flush=True)


if __name__ == "__main__":
    main()
