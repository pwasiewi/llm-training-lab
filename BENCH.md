# Inference Runtime Benchmarks (`bench_NN_*.sh`)

Bash scripts benchmarking local LLM inference runtimes on the same GGUF model,
with MoE CPU/GPU hybrid offload tuned for 16 GB VRAM (RTX 5070 Ti) + 64 GB RAM.

| Script | Runtime | Binaries used |
|--------|---------|---------------|
| `bench_01_llamacpp.sh` | mainline llama.cpp (`sci-misc/llama-cpp::stuff`) | `llama-bench`, `llama-server` |
| `bench_02_ikllama.sh` | ik_llama.cpp fork (`sci-misc/ik-llama-cpp::pwr`) | `ik-llama-bench`, `ik-llama-server` |
| `bench_03_ollama.sh` | ollama (comparison add-on) | `ollama` |
| `bench_04_qwythos.sh` | mainline llama.cpp, Qwythos-9B-v2 (dense hybrid, no MoE) | `llama-bench`, `llama-server` |
| `bench_05_agentic.sh` | agentic coding capability (any aillama profile) | `qwen` (qwen-code), `aillama` |
| `bench_06_dense_generic.sh` | mainline llama.cpp, any dense (non-MoE) model — bench_04 generalized, per-depth loop so one OOM doesn't kill the sweep | `llama-bench`, `llama-server` |
| `bench_07_workflow.sh` | agentic **workflow discipline** (any aillama profile) — the failure axes bench_05 is blind to | `qwen` (qwen-code), `aillama` |

## The bench_05 agentic tasks (key to the fleet table below)

The "bench_05 weak spots" column in the fleet table refers to these 12
coding tasks, run through headless qwen-code against the live llama-server
and verified objectively by the script (protected-file checksums + its own
pytest — the agent's claims are never trusted). Verdict = PASS / FAIL /
TIMEOUT, each with a SCORE `passed/total (pct%)`; TIMEOUT is scored on
whatever the agent left at the cutoff.

| Task | Tier | Tests | What it probes |
|---|---|---|---|
| `bugfix` | easy | 4 | find & fix one bug in an existing module from a failing pytest |
| `scratch` | easy | agent-written | build a CLI tool + its own tests in an empty dir |
| `lru` | easy | 8 | implement LRU cache (capacity + TTL + injectable clock) from tests only |
| `multifile` | easy | 5 | three distinct bugs across three modules |
| `intervals` | mid | 12 | booking Scheduler: half-open overlap, exact cancel, free-slot search |
| `fsm` | mid | 13 | Order state machine: guarded transitions, atomic overpay rejection |
| `codec` | mid | 12 | LEB128 varint + XOR checksum codec; byte-literal assertions (bit/byte axis) |
| `toposort` | mid | 11 | lexicographically-smallest topological order (needs heapq; FIFO Kahn fails) |
| `template` | parser | 10 | mini template engine: `{{ var }}`, dotted lookup, `{% if/else %}`, nested `{% for %}` |
| `interp` | parser | 13 | expression evaluator: precedence, right-assoc `^`, lazy conditional, functions; **eval/exec/compile banned** (AST-verified) |
| `perf` | parser | 6 | EventLog range queries under a hard 10 s budget (400K adds + 100K queries — naive scan and insort both fail) |
| `regex` | parser | 14 | backtracking regex engine: classes, ranges, `\d \w \s`, greedy `* + ?`, alternation, nested capture groups; **import re/regex banned** (AST-verified) |

Reading the results: the easy+mid tier is a near-clean sweep for every
strong model (discriminates only weak ones); the parser tier discriminates
the top of the fleet but is slow and noisy — sampling non-determinism
regularly flips verdicts between same-day runs, so single-run rankings on
template/interp/regex are meaningless (repeat 2–3×). A FAIL splits into
*disqualification* (banned construct — judgment gap) vs *near-miss*
(e.g. 13/14 — capability edge); the NOTE column tells which.

## The bench_07 workflow-discipline task (added 2026-07-17)

bench_05 ranks coding capability but proved blind to how models fail on
real multi-step agentic workflows — the gap surfaced by seven live
`gpt-oss20b-q8_0` qwen-code sessions writing the nanoeuler ebuild
(2026-07-17): truncated instruction reading, per-run stochastic rule
compliance, metadata hallucinated from a suggestive name, thrashing on a
docs-vs-environment mismatch, and report embellishment. `bench_07`
reproduces exactly those five axes in one hermetic task (`relmeta`, a
release-metadata packaging job) with a 10-item mechanical rubric:

| Item | Rule | Axis | Trap it detects |
|---|---|---|---|
| `deliverable` | R1 | compliance | required file at required path |
| `name` | R2 | compliance | trivial control item |
| `version` | R3 | hallucination | README badge says `v2.5-dev`; `src/VERSION` says `2.4.1` |
| `license` | R4 | hallucination | README badge says `MIT`; `src/LICENSE` is BSD-2-Clause |
| `summary` | R5 | hallucination | project is named `nanoeuler` but is a language model — a summary riffing on Euler/numerics fails |
| `order` | R8 | tail-read | exact five-field order, rule past line 200 of RULES.md |
| `lastline` | R9 | tail-read | `Checked: manual` terminator, rule ~line 230 / char ~16K |
| `protected` | R6 | thrashing | any edit/delete under `src/`, `tools/`, RULES.md (sha256 manifest) |
| `no-strays` | R1 | thrashing | scratch files, backups, a recreated `tools/check.py` |
| `evidence` | R10 | evidence-gate | REPORT.md must quote a salted `VALIDATE-OK` token + sha256 recomputed against the FINAL file — unearned or stale claims fail |

Extra trap: RULES.md tells the agent to run `tools/check.py`, but the tool
is `tools/validate.py` (rename documented in `tools/README`) — recovering
from stale docs without "fixing" the inputs is the thrashing axis. The
validator is deliberately syntax-only so running it does not leak the tail
rules to an agent that never read them. Verdict = PASS only at 10/10;
`RUNS` (default 3) repetitions feed a RULE-COMPLIANCE MATRIX (held/runs per
item per model) — items that hold only in some runs are the stochastic-
compliance signal this bench exists for. Both the oracle solution (10/10)
and a seven-defect solution (3/10, each defect hitting its intended item)
were verified through the script's own setup/verify path before first use.
bench_07 scores are NOT comparable to bench_05 scores.

## Model fleet — single-table summary

One row per model/variant ever tested on this box, accepted and rejected
alike. Columns may be partially empty where a model was rejected before the
measurement was taken. Short per-model notes follow the table; full
evidence lives in the dated "Reference results" sections below (archive —
never rewritten, only appended). Status legend: **DEFAULT** = recommended
profile for its role, kept = usable niche/fallback, **REJECTED** = do not
use (GGUF possibly deleted), ref = kept only as a data point.

| Profile / model | Arch | Quant, file size | Fit @128K (`--n-cpu-moe`) | tg tok/s | bench_05 weak spots | Status / role |
|---|---|---|---|---|---|---|
| `gpt-oss20b-udq8kxl` (unsloth UD) | MoE 20.9B | Q8_K_XL, 12.29 GiB | whole (0) | ~193 | interp 3/3, regex 3/3 clean (bench_05); bench_07 11/20 again on 2026-09-19 (lastline 12/20, evidence 12/20 — base-model trait, stable since July) | **DEFAULT fast coding tier** (only fleet member that passes regex; 22–43 s per bench_07 run) |
| `gpt-oss20b-q8_0` | MoE 20.9B | Q8_0, 12.11 GiB | whole (0) | ~214 | regex 57%, interp 71% (14+ runs); template never failed | fallback (superseded by udq8kxl, still fastest tg) |
| `ornith-128k` (Ornith-1.0-35B) | qwen35moe A3B | Q4_K_M, 19.7 GiB | 20 (floor 18, was 24 until 2026-09-17) | ~54 @49K depth | regex never finishes (7/7 TIMEOUT); interp 2/4 at 900 s in the 2026-09-19 pairing | superseded by `ornith15-128k` (2026-09-19) — 20 GB hub blob, delete candidate |
| `qwythos` (Qwythos-9B-v2 +MTP) | qwen35 dense hybrid | Q8_0 | whole | 149 (MTP) | parser tier FAILs (9B ceiling); Xid 8 hang @69K with MTP | **RETIRED 2026-09-19** — dominated by `ornith15-9b` (same size, passes interp/perf, 95 % workflow); 18 GB (two files), delete candidate |
| `ornith-9b` (Ornith-1.0-9B) | qwen35 dense | Q8_0, 9.53 GiB | whole | | interp 12/13 near-miss @900s; perf anti-pattern | accepted; judgment pending |
| `ornith15-9b` (Ornith-1.5-9B) | qwen35 dense | Q8_0, 9.11 GiB | whole (12.0 GiB @128K) | 84 / 73 @49K | @900 s N=4: interp 13/13 scored every run but 3/4 by verdict, template 2/4, perf 3/4; **@1200 s N=3: template/interp/perf 3/3 each = 11/12 every run**, regex 0/3 (12/14 once). **bench_07 19/20 PASS** (every axis 20/20, evidence 19/20) | **DEFAULT workflow agent** (rule-heavy multi-step qwen-code jobs, whole-fit); with `TASK_TIMEOUT=1200` also a full coding agent minus regex; udq8kxl keeps the sprint tier (regex, 2.3× tg) |
| `ornith15-128k` (Ornith-1.5-35B-A3B) | qwen35moe | Q4_K_M, 20.22 GiB | 20 (floor 18) | ~53 @49K | parser tier N=7: template/interp/perf 7/7 each; **regex 2/7 PASS** (744 s and 670 s on 09-19 — first non-gpt-oss regex passes in this log; 14/14 scored-but-late once, 5/14 once); **bench_07 20/20 PASS, every rubric item 20/20** (2026-09-20) | **DEFAULT serious agentic** (2026-09-19; beats Ornith 1.0 on interp, same VRAM shape and speed; Tiel-Coder tied it at N=3) — **best on both axes**, the only perfect bench_07 in this log |
| `tiel-128k` (Tiel-Coder-35B-A3B, coder finetune of the Ornith 1.5 base) | qwen35moe | UD-Q4_K_XL, 20.82 GiB | 21 (14.4 GiB after a 51K request) | 50 @51K depth, pp 1885 | parser tier N=3: template/interp/perf 3/3 each, regex 1/3 PASS (340 s = fastest regex pass ever, then 2× TIMEOUT no file) | tied with `ornith15-128k` at N=3 — no measurable coder-finetune edge; 21 GB kept only as an alternate, delete candidate |
| `qwen38d-9b` (Qwen3.8-9B-Distill, empero-ai) | qwen35 dense | Q8_0, 9.11 GiB | whole (12.3 GiB @128K) | 71 @51K depth, pp 4429 | template FAIL 3/3 (1–4/10), interp 0/3 (9/13 then 0/13 ×2), regex 0/3, perf 3/3 (46 s); **bench_07 1/20** — `summary` 5/20 (riffs Euler/numerics off the project name), evidence 10/20, lastline 12/20 | **REJECTED** — a 9B that hallucinates from names and cannot hold a parser task; delete candidate |
| `qwen38d-128k` (Qwen3.8-35B-A3B-Distill, empero-ai) | qwen35moe | Q4_K_M, 20.22 GiB | 20 (14.1 GiB after a 51K request) | 53 @51K depth, pp 1865 | full suite ×3: easy/mid 24/24 clean and fast; template 2/3, interp 2/3, perf 3/3, regex 0/3 (11/14 once) → 11, 9, 11 of 12; **bench_07 15/20** (lastline 16, evidence 16, summary 19 — no hallucination trait) | accepted — general alternate, stronger than its base `qwen36-128k` on the parser tier and on bench_07 (udq8kxl 11/20), below `ornith15-128k` (regex, interp 3/3) at the same speed |
| `qwen36-128k` (HauhauCS 35B-A3B) | qwen35moe | IQ4_XS, 17.43 GiB | 16 (`--fit`: 16 + ffn_up) | ~82; 49K prompt 1850 pp / 75 tg | fastest template (118 s); insort-not-Fenwick on perf; 8/12 on 09-17 | general alternate — the speed pick (tg ~82); its distill `qwen38d-128k` is stronger at tg ~53 |
| `qwen36u-mxfp4-128k` (unsloth) | qwen35moe | MXFP4_MOE, 20.2 GiB | 16+ | | tied Ornith 7/8 | backup alternate |
| `glm-flash` (GLM-4.7-Flash) | deepseek2 MoE 30B-A3B | Q4_K_XL, 17.5 GiB | 12 @32K / 22 @128K (was 24) | 63 / 42 | 4/7 — gives up early on parser tier | 32K chat only |
| `dsv4flash` (Qwen3.5-9B-DSV4) | qwen35 dense | Q6_K | whole | ~93 | interp 0/13 — total parser failure | chat only, **NOT an agent** |
| `gpt-oss20b` (MXFP4-Aggressive) | MoE 20.9B | MXFP4, 11.27 GiB | whole (0) | ~214 | never passed template+interp across runs | fallback/ref (superseded by Q8_0) |
| `gpt-oss20b-f16` (unsloth F16) | MoE 20.9B | F16, 12.83 GiB | 2 | 2–6× slower | regex 6/14 | fallback for interp-style work |
| `gpt-oss20b-heretic` (DavidAU) | MoE 20.9B | IQ4_NL, 12.6 GiB | 1 (profile 2) | slowest of family | template 4/10, interp 0/13, regex disq. | **REJECTED** (conf ref) |
| `ornith-q5-128k` (Ornith Q5_K_M) | qwen35moe | Q5_K_M, 24 GiB | 22 | −13% vs Q4 | regex TIMEOUT, no file at all | ref — not for time-boxed work |
| `gemma4-fable5` (yuxinlu1 12B) | gemma4 dense | Q8_0, 12.7 GiB | ctx ≤32K only! | | 5/10; interp 1/13, regex 0/14; gen_n=1 red flag | **REJECTED** |
| `gemma4-qat` (HauhauCS 12B) | gemma4 dense | Q4_K_M, 7.38 GiB | | | 100% reproducible stream-hang after task 1 | **REJECTED** |
| `gemma4-12b-it` (unsloth base) | gemma4 dense | Q8_0, 12.7 GiB | ctx ≤65536 only! (131072 OOM) | ~55 | interp+regex 0% TIMEOUT (total parser-tier wipeout); easy/mid 10/12 clean | **REJECTED** |
| 27B dense family (5 finetunes) | qwen35 dense 65L | Q4, 16–19 GB | impossible @128K (any quant) | | | **REJECTED**, GGUFs deleted (113 GB) |
| `qwen36u-iq4xs` / `qwen36u-q4kxl` | qwen35moe | 16.5 / 20.8 GiB | 10 / 17 | | scratch FAIL / 2× TIMEOUT | **REJECTED**, deleted (38 GB) |
| GLM 5.2 (744B) | MoE | any | needs ≥245 GB RAM+VRAM | | | **REJECTED** — never fit |

### Per-model notes

- **All `--n-cpu-moe` profiles carry `--load-mode none` since 2026-09-17**
  (llama.cpp b11009): pp 1.6–3× on the CPU-resident experts, tg unchanged.
  The tg figures in this table predate it and still hold; pp figures from
  before that date are the mmap numbers — see "b11009 follow-ups".
- **`gpt-oss20b-udq8kxl`** — the fast-tier default since 2026-07-19,
  superseding `gpt-oss20b-q8_0`. unsloth's UD (dynamic) Q8_K_XL quant —
  finer per-tensor treatment of non-expert tensors than plain Q8_0, 0.18
  GiB bigger, fits whole at 131072 ctx (14.5/16.3 GiB, -np 1). 3-run repeat
  targeting the two noisy parser-tier tasks: interp 3/3 PASS, regex 3/3
  PASS, all 100% — q8_0 FAILed regex twice in the same 3 runs. ~10% slower
  tg (193 vs 214) is the tradeoff for the clean sweep. **bench_07
  (2026-07-19, 35 runs/model pooled)**: workflow-discipline is equivalent
  to q8_0 — udq8kxl 49% vs q8_0 54% PASS, per-axis within 1–2 runs. An
  apparent regression at N=10–15 dissolved at N=20 (the ranking flipped);
  the `lastline`/`evidence` weakness (~40–45% miss each) is a gpt-oss-20b
  base-model trait shared by both quants, to be mitigated in the harness,
  not by quant choice. bench_07 needs N≥20 before ranking on it.
- **`gpt-oss20b-q8_0`** — fast-tier default 2026-07-14 → 07-19, now
  fallback/reference (still the fastest tg in the family). gpt-oss keeps
  MoE experts at native ~4-bit regardless of quant label, so Q8_0 costs
  almost nothing over MXFP4 and still fits whole at 131072 ctx (15.0/16.3
  GiB). 14+ single-model runs: 6 easy tasks + template at 100%,
  codec/toposort 93%, interp 71%, regex 57%. Expect one parser-tier miss
  per run as the norm, not a clean 12/12 — confirmed again in the 07-19
  repeat (regex FAIL twice out of 3).
- **`ornith-128k`** — RL agentic-coding pedigree (SWE-bench 75.6%); best
  engineering judgment of the fleet (Fenwick tree on perf with 9× margin).
  Weak spot: regex — never converged in any budget tested (900/1200 s);
  over-verbose backtracking + self-verification loop. Q4_K_M beats both
  Q5_K_M and any bigger file (repeated "bigger file is not better" lesson).
- **`qwythos`** — dense hybrid Gated-DeltaNet, MTP variant gives +75% tg for
  −25% pp. 3–5× faster than 35B on easy tasks, hard ceiling on the parser
  tier. One known Xid 8 hang at ~69K ctx with MTP on (pkill -9, retry
  without `--spec-type draft-mtp`).
- **`ornith-9b`** — fits whole at full ctx; near-miss profile resembles its
  35B parent (needs time, not capability). Not yet fully ranked vs
  gpt-oss20b family — retest at `TASK_TIMEOUT=1200` pending.
- **`qwen36-128k` / `qwen36u-mxfp4-128k`** — general-purpose alternates.
  HauhauCS IQ4_XS is the speed pick; unsloth MXFP4_MOE was the only one of
  three official unsloth quants worth keeping (the other two deleted).
  mainline llama.cpp beats the ik fork on this arch at depth.
- **`glm-flash`** — pp champion on the ik fork (2×) but dominated as an
  agent (gives up instead of iterating); floor drifted 2 steps between
  llama.cpp builds (re-probe after rebuilds). 128K possible but 24% slower
  than ornith-128k — kept as a 32K chat profile.
- **`dsv4flash`** — fine as a fast chat model, catastrophic as an agent
  (interp 0/13 — looked identical to gpt-oss20b's "5/8" until the SCORE
  column existed; that contrast motivated scored verdicts).
- **gpt-oss20b variants** — MXFP4-Aggressive was the original default,
  demoted after it never passed template+interp in any later run; F16
  breaks the fits-whole property (floor 2) and pays 2–6× wall-clock for one
  extra interp pass; HERETIC failed the parser tier outright and was the
  slowest of the family.
- **gemma4 trio** — new arch loads fine on llama.cpp 9988, so all three
  rejects are model-level, each for a different reason: QAT variant has a
  100%-reproducible dead-stream bug after the first task; fable5 variant
  underperforms a 9B and can't hold >32K ctx compute buffers; the plain
  unsloth base-instruct variant (`gemma4-12b-it`, not a finetune, infra
  healthy, no stream-hang) is solid on easy/mid tier but totally wipes out
  on the parser tier — 0% TIMEOUT on both `interp` and `regex` in a clean
  re-run, vs `gpt-oss20b-q8_0` clearing all 12 tasks 100% in the same pass.
- **27B dense family** — architectural VRAM ceiling on 16 GB at 128K: whole
  model never fits (`-ngl 99` OOM even at 4K ctx), partial offload caps at
  ~32K ctx. Independent of finetune and quant; NVFP4 gave no Blackwell
  advantage. All five GGUFs deleted.

## What is measured

- **pp (prompt processing)** — tokens/s while ingesting the prompt; dominates
  agent/RAG workloads with long inputs.
- **tg (token generation)** — tokens/s while producing output; dominates chat.
- `bench_01`/`bench_02` run two tests: a `llama-bench` sweep (pp512/tg128 per
  `--n-cpu-moe` value) and a *real server request* (~1250-token prompt via
  `/completion`, timings parsed from the response).
- `bench_03` sends the same prompt to ollama's `/api/generate` twice — cold
  (includes model load) and warm — and prints ollama's whole-layer CPU/GPU
  split (`ollama ps`). Ollama has no `--n-cpu-moe` equivalent; quantifying
  that gap is the point of this script.
- `bench_05` measures capability, not throughput: it drives headless qwen-code
  (`--approval-mode yolo`) against each aillama profile on ten coding tasks
  of increasing difficulty — an easy tier (pytest bugfix, CLI tool from
  scratch, LRU+TTL cache, three-bug multifile hunt), a mid tier (interval
  Scheduler with free-slot search, Order state machine with guarded
  transitions), and a parser/compiler tier (template engine, expression
  interpreter, EventLog under a perf budget, backtracking regex engine).
  Verdicts are objective: protected-file checksums plus the script's own
  pytest/functional checks — the agent's claims and self-written tests are
  never trusted. Since 2026-07-14 every verdict carries a **SCORE** column
  (`passed/total (pct%)`); TIMEOUT is scored on whatever the agent left
  behind at the cutoff, so a 13/14 near-miss is distinguishable from an
  empty directory.
- `bench_04` replaces the `--n-cpu-moe` axis (meaningless for a dense model
  that fits fully in VRAM) with: a context-depth sweep (`-d 0,16384,65536,131072`
  — Gated-DeltaNet linear-attention pp scaling), a real server request at 128K
  context, and an MTP speculative-decoding on/off comparison (`--spec-type
  draft-mtp`, needs the `*-MTP-*.gguf` variant). Full results: see the
  2026-07-12 reference section below.
- `bench_07` measures workflow discipline, not coding: one long-rules
  packaging task (`relmeta`) through headless qwen-code, a 10-item
  mechanical rubric mapped to five failure axes (tail-read, compliance,
  hallucination, thrashing, evidence-gate), `RUNS` repetitions and a
  final per-item RULE-COMPLIANCE MATRIX. See its own section above.
- `bench_06` is `bench_04` generalized to **any** dense model (no hardcoded
  defaults; `MODEL=` required, `MTP_MODEL=`/`NGL=`/`CTX=` optional). Same
  three measurements — depth sweep, real server request, optional MTP
  comparison — but each depth value runs as a separate `llama-bench`
  invocation, so one OOM logs an error and the sweep continues instead of
  dying on its first value. Written for the 27B/12B/9B dense candidate
  round (2026-07-14).

## Usage

```bash
./bench_01_llamacpp.sh          # bench sweep + server test
./bench_01_llamacpp.sh -b      # sweep only
./bench_01_llamacpp.sh -s      # server test only
./bench_02_ikllama.sh           # same modes as bench_01
./bench_03_ollama.sh            # cold + warm query; -k keeps the daemon running
./bench_04_qwythos.sh           # depth sweep + server test + MTP comparison
./bench_04_qwythos.sh -b       # depth sweep only; -s server only; -m MTP only
NCMOE_LIST=6,8 CTX=16384 ./bench_01_llamacpp.sh   # env overrides
DEPTH_LIST=0,32768 CTX=32768 ./bench_04_qwythos.sh   # bench_04 env overrides
MODEL=~/models/foo/foo-Q8_0.gguf ./bench_06_dense_generic.sh   # any dense model; -b/-s/-m like bench_04
MODEL=... MTP_MODEL=... NGL=50 CTX=32768 ./bench_06_dense_generic.sh   # partial offload + MTP comparison
MODELS=gpt-oss20b-q8_0,ornith-128k TASK_TIMEOUT=1200 ./bench_05_agentic.sh   # agentic suite
MODELS=gpt-oss20b-q8_0 RUNS=5 ./bench_07_workflow.sh   # workflow discipline; matrix needs RUNS>1
```

## Model Storage

The benchmark scripts are configured so large model files live outside `/home`:

- Hugging Face cache: `HF_HOME=/mnt/db1/huggingface`
- GGUF model root: `MODEL_ROOT=~/models`, currently a symlink to
  `/mnt/db1/huggingface/models`
- Ollama model store: `OLLAMA_MODELS=/mnt/db1/ollama/models`

Current local symlinks:

```text
~/.cache/huggingface -> /mnt/db1/huggingface/
~/.ollama           -> /mnt/db1/ollama
~/models            -> /mnt/db1/huggingface/models
```

This keeps downloads, Ollama blobs, and GGUF files on `/mnt/db1`. If you run
Ollama manually, keep the same model store:

```bash
export OLLAMA_MODELS=/mnt/db1/ollama/models
ollama serve
```

Without that variable or the `~/.ollama` symlink, `ollama create`/`ollama pull`
will write large blobs back under `/home`.

## Parameters (env variables)

### bench_01 / bench_02

| Variable | Default | Meaning |
|----------|---------|---------|
| `HF_HOME` | `/mnt/db1/huggingface` | Hugging Face cache root exported for child tools |
| `MODEL_ROOT` | `~/models` | GGUF model root; symlinked to `/mnt/db1/huggingface/models` |
| `MODEL` | `$MODEL_ROOT/glm-4.7-flash/GLM-4.7-Flash-UD-Q4_K_XL.gguf` | GGUF path |
| `NCMOE_LIST` | `8,10,14` | `--n-cpu-moe` values for the bench sweep (comma-separated); lower = more experts on GPU = faster, until VRAM OOM |
| `SERVER_NCMOE` | `10` (01) / `18` (02) | `--n-cpu-moe` for the server test (needs headroom for KV cache at `CTX`); the fork's compute buffer scales with context (~1.7 GB at 32K vs ~0.9 GB mainline), hence the higher default |
| `THREADS` | `16` | CPU threads (physical cores of the 5950X) |
| `CTX` | `32768` | server context window |
| `PORT` | `8090` (01) / `8091` (02) | server port |
| `NPREDICT` | `256` | output tokens per request |
| `PROMPT_REPEATS` | `60` | sentence repetitions building the ~1250-token prompt |
| `CACHE_TYPE` | `q8_0` | KV cache quantization (`-ctk`/`-ctv`) |
| `BENCH_BIN` / `SERVER_BIN` | per script | binary override |
| `FA_FLAG` | `on` (01) / `1` (02) | flash-attention flag syntax differs between mainline and the fork |
| `EXTRA_SERVER_ARGS` | empty | extra raw args appended to the server command |

### bench_03

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL` | same GGUF as above | imported via Modelfile if `OLLAMA_MODEL` absent |
| `OLLAMA_MODEL` | `glm47-flash-q4` | model name inside ollama |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | API address |
| `OLLAMA_MODELS` | `/mnt/db1/ollama/models` | Ollama model store used by `ollama serve`, `list`, and `create` |
| `NPREDICT` / `PROMPT_REPEATS` | `256` / `60` | as above |

Flags: `-k` keeps the daemon running after the test (by default the script
stops the daemon only if it started it). The GGUF import creates a blob copy
(~model size) under `OLLAMA_MODELS`; remove with `ollama rm glm47-flash-q4`.

### bench_06

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL` | *(empty — required)* | main GGUF; no hardcoded default, unlike bench_04 |
| `MTP_MODEL` | empty | GGUF with MTP head; enables the `-m` speculative-decoding comparison |
| `NGL` | `99` | `-ngl` layers on GPU — the knob for dense models that don't fit whole (there is no `--n-cpu-moe` axis here) |
| `DEPTH_LIST` | `0,16384,65536,131072` | `llama-bench -d` sweep; each depth is a **separate** invocation, so one OOM logs and continues instead of killing the sweep |
| `CTX` | `65536` | server context for the real-request test |
| `REPS` | `2` | `llama-bench` repetitions per depth |
| `THREADS` / `PORT` / `NPREDICT` / `PROMPT_REPEATS` / `CACHE_TYPE` / `FA_FLAG` | as bench_01 (`PORT` 8090) | shared plumbing |

Flags like bench_04: `-b` depth sweep only, `-s` server test only, `-m` MTP
comparison only (requires `MTP_MODEL`); no flag = all (skips `-m` when
`MTP_MODEL` is unset).

## HF sweep — 2026-07-28: fleet files all current; new candidate GGUFs spotted (untested)

Checked every fleet repo's `lastModified` on HF against the local download
dates: **nothing to re-download**. The one scare — `deepreinforce-ai/
Ornith-1.0-35B-GGUF` shows a delete (07-15) + re-upload (07-18) of
`ornith-1.0-35b-Q4_K_M.gguf`, i.e. after our 07-13 download — resolved
clean: the LFS OID of the re-uploaded file (`ff25291b…`) is byte-identical
to the local HF-cache blob. All other fleet repos (unsloth gpt-oss-20b,
HauhauCS gpt-oss/qwen36, unsloth Qwen3.6/GLM-4.7-Flash, ornith-9b, Jackrong
dsv4flash, Qwythos v2, ggml-org embed/rerank) predate the local copies.

New candidates found in the same sweep, none benched yet. The first three
target current defaults directly:

1. **`unsloth/Ornith-1.0-35B-GGUF`** (new repo, 07-17) — official unsloth
   UD quants of the serious-agentic default. `UD-IQ4_XS` is 16.56 GiB vs
   the current Q4_K_M's 19.7 GiB → lower `--n-cpu-moe` floor → faster tg.
   Same move that made `udq8kxl` win on gpt-oss, BUT UD-IQ4_XS was the
   uniquely-failing quant on qwen36 — bench before switching.
2. **Eagle3 speculative decoding for gpt-oss-20b** — llama.cpp master
   (2026-07-28, PR #25794) adds eagle3-v3 support for gpt-oss, and a 20b
   draft exists: `RedHatAI/gpt-oss-20b-speculator.eagle3` (44k downloads,
   safetensors — needs GGUF conversion). First realistic shot at speeding
   up the fast-tier default's *decode* (bandwidth-bound, flat under every
   quant/build change so far). Requires a llama.cpp rebuild (build ≥ Jul 28).
3. **MTP variants of existing fleet models**: `SC117/Ornith-1.0-35B-MTP-
   APEX-GGUF`, `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, `Jackrong/Qwen3.5-9B-
   DeepSeek-V4-Flash-MTP-GGUF`. MTP gave qwythos +75% tg for −25% pp; for
   ornith-128k (~52 tok/s) this is potentially the biggest practical win.
   Known risk: the Xid 8 hang at ~69K ctx with `--spec-type draft-mtp`.
4. **`tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF`** — first sighting of the
   14B-A3B MoE size class (Q4_K_M 7.88 GiB, + vision mmproj): fits WHOLE
   in VRAM even at 131072 ctx. New shelf between qwythos and the 35Bs;
   community finetune, base-model quality unknown.
5. **`prism-ml/Ternary-Bonsai-27B-gguf`** — ternary (BitNet-style) 27B at
   6.67 GiB: the first 27B that fits whole (the dense-27B family was
   REJECTED on VRAM). Base repo has 2.3M downloads. Has a `dspark` draft
   variant (DSpark spec-decode landed in llama.cpp master 07-28 too).
   Agentic quality is a complete unknown — cheap to bench.
6. **`empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF`** — 1M-ctx sibling of
   the fleet's Qwythos v2, 3× its downloads (1.26M).

Rejected on size for 16 GB VRAM @128K: Laguna-S-2.1 (Q4 89 GiB),
Laguna-XS-2.1 (Q4 18.9 GiB dense — the 27B-class wall), KAT-Coder-V2.5
(~32B dense), MiniMax-M3, Kimi-K3.

## Reference results — 2026-08-18: Nemotron-3.5-Lightning-30B-A3B + Qwen3.8-27B — leader unbeaten; the VRAM-fit rule

Two fresh 2026-08 releases hunted after the host stack rebuild. Both needed a
llama.cpp master rebuild first (older builds reject the GGUFs outright).

### qwen38 (Qwen3.8-27B UD-Q4_K_XL 17.9G, dense, thinking) — FAILED TEST, model removed

The 27B-dense wall from the size-rejection list above, now measured instead
of assumed: 17.9G does not fit 16 GB VRAM, `-ngl 48` (16/64 layers on CPU)
gives **4.5 t/s decode**, and thinking mode multiplies the token bill.
Bench: **0/8 with every task dying at 810–854 s against the 900 s timeout**
— most tasks never even created the target file. Same failure shape as
gemma4-12b-it (timeout class), different cause (throughput, not parser).

**Rule going forward: a dense candidate whose quant does not fit whole in
VRAM (weights + KV, desktop running) is disqualified for the agentic tier UP
FRONT.** Partial-offload decode (~5 t/s) turns every task into a
near-timeout FAIL regardless of model quality — one 30-second curl speed
check settles it, not two hours of bench. MoE candidates are exempt (expert
offload keeps decode fast: the qwen36 / ornith / glm-flash / nemotron
pattern). GGUF deleted, profile removed from models.conf.

### nemotron (NVIDIA-Nemotron-3.5-Lightning-30B-A3B, MoE A3B Mamba2-hybrid, ggml-org Q4_0 18G) — 7/12 single run, leader NOT dethroned

`--n-cpu-moe 18`, 13.6G VRAM, **68 t/s decode** — fastest big candidate yet.

| task | verdict | time | score |
| --- | --- | --- | --- |
| bugfix | PASS | 62s | 4/4 |
| scratch | PASS | 104s | 6/6 |
| lru | PASS | 57s | 8/8 |
| multifile | FAIL | 211s | 4/5 (80%) |
| intervals | PASS | 128s | 12/12 |
| fsm | PASS | 169s | 13/13 |
| codec | PASS | 275s | 12/12 |
| toposort | PASS | 68s | 11/11 |
| template | FAIL | 172s | 9/10 (90%) |
| interp | FAIL | 331s | 0/13 |
| perf | FAIL | 409s | 5/6 (83%) |
| regex | FAIL | 324s | 0/14 |

**Verdict: 7/12 PASS (58%), mean score ~79%, zero timeouts** — vs the
leader gpt-oss20b-udq8kxl's pooled ~50% PASS. Promising, but by our own
bench_07 lesson a single run does not rank (N=20 minimum) — **the leader
stays unbeaten until a pooled run says otherwise.** The two 0% tasks are
the parser tier (interp, regex), the same axis every non-leader fails.
Unused reserves for the rematch: the separate `mtp-…-Q4_0.gguf` draft file
(`--spec-type draft-mtp`, untested) and a 128K-context profile.

Quant gotcha (cost an evening): unsloth's UD quants of this model embed the
`blk.*.nextn.*` MTP tensors in the main GGUF — mainline llama.cpp then dies
with `done_getting_tensors: wrong number of tensors; expected 417, got 408`
(it only consumes 408 for `nemotron_h_moe`; for qwen3.8 the same surplus is
merely a warning). The working pair is ggml-org's own conversion: main
Q4_0 + separate mtp file. The 25.5G UD-Q4_K_XL was deleted with nothing to
load it; re-download if llama.cpp ever learns embedded nextn for this arch.

## Reference results — 2026-07-19: gpt-oss20b-udq8kxl — new fast-tier default, beats q8_0 on parser-tier reliability

`gpt-oss-20b-UD-Q8_K_XL.gguf` (unsloth/gpt-oss-20b-GGUF, UD dynamic quant,
12.29 GiB actual — 0.18 GiB bigger than the current default `gpt-oss20b-q8_0`
at 12.11 GiB). Fits fully in VRAM at 131072 ctx like every other gpt-oss-20b
quant tested so far (experts stay native precision regardless of the
non-expert quant label): 14.5/16.3 GiB with `-np 1`, boot + real-request
clean, no first-decode OOM. Profile `gpt-oss20b-udq8kxl` in
`~/.aillama/models.conf`.

`bench_06_dense_generic.sh` depth sweep (F16 KV, no `-ctk`/`-ctv` — a
stricter VRAM test than the deployed server config below, hence the
d131072 OOM despite the server fitting fine at that depth with quantized
KV):

| depth | pp512 t/s | tg128 t/s |
|---|---|---|
| 0 | 10951.93 ± 3.66 | 207.16 ± 1.59 |
| 16384 | 8016.58 ± 61.78 | 186.51 ± 3.50 |
| 65536 | 4024.40 ± 25.27 | 145.78 ± 0.70 |
| 131072 | FAILED (OOM, unquantized KV) | FAILED (OOM, unquantized KV) |

Server test (quantized KV, matches the deployed profile) @ ctx 131072:
`prompt_n=683 pp=3437.1 tok/s`, `gen_n=256 tg=188.7 tok/s`. Real-request
follow-up hit 193.1 tok/s — about 10% below q8_0's ~213-214 tok/s.

**bench_05 agentic, three runs vs `gpt-oss20b-q8_0`** (run 1: full 12-task
suite; runs 2-3: `TASKS=interp,regex` only, the two tasks with documented
sampling noise for this model class — `WORKROOT` isolated per run):

| Run | q8_0 interp | q8_0 regex | udq8kxl interp | udq8kxl regex |
|---|---|---|---|---|
| 1 (full suite, other 10 tasks 100% both models) | PASS 100% (63s) | **FAIL** — real `SyntaxError` line 406 (43s) | PASS 100% (73s) | PASS 100% (146s) |
| 2 | PASS 100% (27s) | PASS 100% (467s) | PASS 100% (265s) | PASS 100% (85s) |
| 3 | PASS 100% (91s) | **FAIL** 12/14 85% — pytest failing (164s) | PASS 100% (156s) | PASS 100% (78s) |

`gpt-oss20b-udq8kxl`: **6/6 clean** across interp+regex over 3 runs, plus a
clean sweep of the other 10 tasks in run 1 (12/12 that run).
`gpt-oss20b-q8_0`: 4/6, failing regex twice — consistent with its
already-documented ~57% historical regex pass rate, not a fluke of this
session. Same pattern the project has used before to confirm a new default
(e.g. the two-run repeat that promoted q8_0 over MXFP4 on 2026-07-14).

**Verdict: `gpt-oss20b-udq8kxl` promoted to DEFAULT fast tier**, superseding
`gpt-oss20b-q8_0` (2026-07-14 → 07-19). `gpt-oss20b-q8_0` remains as
fallback/reference — still the raw-speed pick (~213 vs ~193 tok/s) when
regex/interp reliability doesn't matter for the task at hand.
`~/.aillama/models.conf` updated (CONFIRMED DEFAULT comment moved).

## Reference results — 2026-07-19: gemma4-12b-it (third gemma4 variant) — rejected, total parser-tier failure

`gemma-4-12b-it-Q8_0.gguf` (unsloth/Gemma-4-12B-It-GGUF, 12.67 GB) — confirmed
via GGUF header scan as the plain Google base instruct model, **not** a
finetune, so not covered by the `gemma4-fable5`/`gemma4-qat` rejection above.
Profile `gemma4-12b-it` in `~/.aillama/models.conf` (`-ngl 99 -c 65536 -np 1
-fa on -ctk q8_0 -ctv q8_0`).

`bench_06_dense_generic.sh` depth sweep (ngl=99, r=2):

| depth | pp512 t/s | tg128 t/s |
|---|---|---|
| 0 | 4598.96 ± 270.15 | 57.94 ± 0.15 |
| 16384 | 3216.47 ± 136.60 | 52.32 ± 0.33 |
| 65536 | 1811.89 ± 24.79 | 49.81 ± 0.84 |
| 131072 | FAILED (OOM) | FAILED (OOM) |

Server test @ ctx 65536: `prompt_n=683 pp=3157.1 tok/s`, `gen_n=256 tg=55.2
tok/s`. Same ctx ceiling as `gemma4-fable5` (32768–65536 reliable, 131072
OOMs the compute buffer) — architectural, not finetune-specific.

`bench_05_agentic.sh` full 12-task suite vs `gpt-oss20b-q8_0`, clean run (no
concurrent build this time, unlike the interrupted provisional run the day
before):

| Task | gemma4-12b-it | gpt-oss20b-q8_0 |
|---|---|---|
| bugfix | PASS 100% (35s) | PASS 100% (9s) |
| scratch | PASS 100% (95s) | PASS 100% (33s) |
| lru | PASS 100% (98s) | PASS 100% (20s) |
| multifile | PASS 100% (93s) | PASS 100% (18s) |
| intervals | PASS 100% (69s) | PASS 100% (14s) |
| fsm | PASS 100% (70s) | PASS 100% (15s) |
| codec | PASS 100% (123s) | PASS 100% (28s) |
| toposort | PASS 100% (72s) | PASS 100% (20s) |
| template | PASS 100% (697s) | PASS 100% (60s) |
| interp | **TIMEOUT 0% (900s)** | PASS 100% (146s) |
| perf | PASS 100% (112s) | PASS 100% (49s) |
| regex | **TIMEOUT 0% (900s)** | PASS 100% (113s) |

gemma4-12b-it: 10/12 clean, but `interp` and `regex` are a full 900s timeout
at **0%** — not a near-miss, no partial credit, nothing salvageable at the
cutoff. This is the same pair of tasks the earlier (VRAM-contaminated,
interrupted) provisional run flagged, that time dismissed as an infra
hiccup on `regex` — now reproduced clean, so it's a real, repeatable
parser-tier capability gap, not noise. `gpt-oss20b-q8_0` passed all 12 tasks
100% in the same pass, including clean runs of the same two tasks it
sometimes misses in its own noisy ~40%-FAIL tail (see 2026-07-15 section
below).

**Verdict: `gemma4-12b-it` REJECTED for agentic use.** Also easy/mid tier
parity with `gpt-oss20b-q8_0` doesn't offset a complete, reproducible
inability to finish parser-tier tasks — unlike gpt-oss's occasional tail
miss, this model never completes them at all. No further repeats planned;
two independent runs (provisional + clean) landing on the same two 0%
failures is enough signal. `gpt-oss20b-q8_0` remains the confirmed default
fast-tier profile.

## Reference results — 2026-07-17: bench_07 first live run, gpt-oss20b-q8_0, RUNS=5

```
MODELS=gpt-oss20b-q8_0 RUNS=5 ./bench_07_workflow.sh
```

All five runs fast (19–37 s). Raw first-pass output said 2/5 PASS with
`summary` held only 2/5 — but artifact inspection showed **all three
`summary` FAILs were verifier false-positives**: the model wrote factually
correct summaries ("tiny character-level transformer that predicts the
next ASCII token") that merely lacked the literal phrase "language model"
the check demanded. The grep-as-verifier trap from the CLAUDE.md pattern
table, caught by our own bench on day one. Fixed in `verify_relmeta()`
(required-phrase regex broadened to `language model|character-level|
transformer|next[- ]token`; the forbidden Euler/numerics list — the part
that actually detects the hallucination — is unchanged) and all five runs
re-scored from the preserved artifacts:

| Run | Verdict | Score | Failed items |
|---|---|---|---|
| 1 | PASS | 10/10 | |
| 2 | FAIL | 8/10 | `lastline`, `evidence` |
| 3 | PASS | 10/10 | |
| 4 | PASS | 10/10 | |
| 5 | PASS | 10/10 | |

Corrected matrix: every item 5/5 except `lastline` 4/5 and `evidence` 4/5.
Run 2 is the genuine bench_07 signal, and it is exactly the failure class
from the nanoeuler sessions: the model wrote `Checked: 0ab43e6a88e7` (the
validator token) instead of `Checked: manual` — conflating tail rules R9
and R10 — reduced REPORT.md to two lines of prose paraphrase with no
`## Evidence` section, no `VALIDATE-OK` line and no sha256, and then
**confidently declared** in its final message that the Checked field was
correctly "set to the validator hash". Stochastic tail-rule compliance +
report embellishment in one run, while runs 1/3/4/5 were flawless. Caveat:
a stray `^C` landed during run 2 (artifacts and agent.log look complete
and self-consistent, but a repeat is cheap). No hallucination on
version/license/summary in any run (README's stale `v2.5-dev`/`MIT` badges
ignored 5/5), no thrashing, `tools/check.py`→`validate.py` mismatch
recovered 5/5. Takeaway: gpt-oss20b-q8_0's workflow discipline is better
than the nanoeuler sessions suggested for a SHORT rules file read directly
— its weak spot is the report/attestation tail, ~1 in 5 runs. Next data
points wanted: `ornith-128k` (RL pedigree hypothesis) and RUNS=10 for a
tighter rate on the tail-rule slip.

## Reference results — 2026-07-19: bench_07 head-to-head, gpt-oss20b-q8_0 vs gpt-oss20b-udq8kxl (RUNS=10, then RUNS=20) — apparent udq8kxl regression dissolves at N=20; weakness is base-model-level

```
MODELS=gpt-oss20b-q8_0,gpt-oss20b-udq8kxl RUNS=10 ./bench_07_workflow.sh
```

Follows a standalone `MODELS=gpt-oss20b-udq8kxl RUNS=5` run earlier the same
day (2/5 PASS = 40%) that first raised the flag. This 10-run head-to-head
confirms it wasn't a fluke:

| Model | PASS rate | summary (halluc.) | lastline (tail-read) | evidence (evidence-gate) |
|---|---|---|---|---|
| `gpt-oss20b-q8_0` | 6/10 (60%) | 8/10 | 6/10 | 6/10 |
| `gpt-oss20b-udq8kxl` | 4/10 (40%) | **10/10** | 5/10 | **4/10** |

All other rubric items (deliverable, name, version, license, order,
protected, no-strays) held 10/10 for both models — the entire gap is
concentrated in three axes. `udq8kxl` fully fixed the hallucination axis
(no more paraphrase-triggered `summary` misses that plagued q8_0) but is
*worse* at reading rules near the end of the long RULES.md (`lastline`)
and *worse* at the evidence-gate (citing a real recomputed sha256/token
instead of an unearned claim) — combined `lastline,evidence` is the
dominant FAIL signature for udq8kxl (6 of its 6 FAILs across both sessions
carry both items together, vs isolated single-item misses for q8_0).

Combined across both udq8kxl sessions today: 6/15 PASS (40%), consistent
between the standalone 5-run and the head-to-head 10-run.

**CORRECTED the same day by a RUNS=20 head-to-head — the "udq8kxl
regression" was small-N sampling noise.** Third session, same command with
`RUNS=20`: udq8kxl **11/20 PASS (55%)** vs q8_0 **9/20 (45%)** — the
ranking *flipped* relative to the 10-run session. 20-run matrix: lastline
12/20 vs 11/20, evidence 12/20 vs 10/20, summary 17/20 vs 18/20 — every
axis within 1–2 runs of each other. One new single-occurrence miss:
q8_0 dropped `license` once (19/20), first hallucination-axis slip on
version/license for either model.

Pooled across ALL bench_07 sessions (35 runs per model: udq8kxl 5+10+20;
q8_0 5 from 07-17 + 10 + 20):

| Metric | udq8kxl | q8_0 |
|---|---|---|
| PASS | 17/35 (49%) | 19/35 (54%) |
| lastline (tail-read) | 20/35 (57%) | 21/35 (60%) |
| evidence (evidence-gate) | 19/35 (54%) | 20/35 (57%) |
| summary (hallucination) | 31/35 (89%) | 31/35 (89%) |

Statistically indistinguishable. Revised conclusions:

1. **The `lastline`+`evidence` weakness is a trait of the gpt-oss-20b BASE
   MODEL (~40–45% miss rate on each), not a quant-level difference** —
   both Q8_0 and UD-Q8_K_XL land on the same numbers once N is large
   enough. The dominant FAIL signature (`lastline,evidence` together) is
   identical for both.
2. **bench_07 session-level drift is large**: the same q8_0 scored 80% →
   60% → 45% PASS across three sessions with zero changes. Do not rank
   models on bench_07 with fewer than ~20 runs; treat 5–10-run deltas as
   noise. (Same lesson as bench_05 parser-tier, amplified.)
3. The earlier "udq8kxl fixed the summary axis" claim (10/10 in the 10-run
   session) also dissolved — pooled it's 89% for both.

**Decision stands: `gpt-oss20b-udq8kxl` remains DEFAULT fast-tier** — it
won bench_05 decisively and is now shown to be workflow-equivalent to
q8_0, so there is no trade-off left to weigh. The absolute weakness
remains real and shared: on long-rules workflows (qwen-code, ebuild
authoring) expect either quant to miss tail rules / skip the evidence
gate in roughly 2 of 5 runs — mitigate in the harness (checklist at top
of file, evidence-gate prompts), not by quant choice.

### Post-hoc artifact analysis (same day): the failure mechanism is "head read, tail lost"

Mining the 40 preserved run directories from the RUNS=20 session settled
*how* `lastline`/`evidence` fail. In **every** failing run the
`Checked:` field is PRESENT in `package.meta` — the model knows the field
exists from the field-order rule R8 (line 168, held 20/20 by both
models) — but carries an **invented** value: `Checked: OK`, `Checked:
true`, `Checked: VALID`, or the validator token hash, instead of the
literal `Checked: manual` demanded by R9 (line 231). And in every one of
those runs the `## Evidence` section required by R10 (line 235) is
**absent entirely** (0/13 failing runs have it). So the model is not
reading the tail rules and ignoring them — the content past ~line 230
never registers at all, while everything up to at least line 168 does.
Confound: the tail rules are also a distinct rule *type* (literal
attestation value + evidence section), so position vs type could not be
separated from this data alone.

### bench_07 extension: `LAYOUT=permuted` (added 2026-07-19)

New env var `LAYOUT=standard|permuted` in `bench_07_workflow.sh`,
designed to deconfound rule POSITION from rule TYPE. The permuted layout
swaps the tail rule pair (`lastline`, `evidence`) with two early rules
both models held 20/20 (`name`, `version`): the attestation and evidence
rules become R2 (line 21) and R3 (line 85), while name/version move to
the tail slots R9/R10 (lines 235/238). Rule wording, sections, pad
blocks and total length (280 lines) are identical — only slot assignment
and R-numbering change; the rubric is content-keyed so verification is
untouched, and the matrix AXIS column shows `moved-to-head` /
`moved-to-tail` for the four swapped items. Oracle-verified: standard
layout output is byte-identical to the pre-refactor generator, a perfect
solution scores 10/10 in both layouts, and the real-world defect
signature (invented `Checked:` value + missing Evidence section) fails
exactly `lastline,evidence`. Run dirs get a `-permuted` suffix; permuted
results are a **separate population** — never pool with standard runs.

Predictions: if `lastline`/`evidence` recover to ~100% at the head AND
`name`/`version` start failing at ~55–60% in the tail → the ~45% miss is
positional (truncated/decayed tail reading) and harness mitigations
(checklist at top) should work. If `lastline`/`evidence` keep failing at
the head → the rule type itself is hard and prompt-side mitigation needs
to target attestation/evidence semantics instead. Suggested first run:
`LAYOUT=permuted MODELS=gpt-oss20b-udq8kxl RUNS=20 ./bench_07_workflow.sh`.

### Permuted-layout results (2026-07-19, gpt-oss20b-udq8kxl, RUNS=20): position is causal, and tail probes must be unguessable

```
LAYOUT=permuted MODELS=gpt-oss20b-udq8kxl RUNS=20 ./bench_07_workflow.sh
```

**19/20 PASS (95%)** vs 55% on the standard layout the same day — and the
outcome split the two predictions in an instructive way:

| Item | Standard (tail) | Permuted (head) | | Item | Standard (head) | Permuted (tail) |
|---|---|---|---|---|---|---|
| lastline | 12/20 (60%) | **20/20** | | name | 20/20 | 20/20 |
| evidence | 12/20 (60%) | **20/20** | | version | 20/20 | 19/20 |

1. **Position is causal for the attestation/evidence failures.** Moved to
   the head, `lastline` and `evidence` held 40/40 — the model complies
   perfectly with the exact same rule text when it actually sees it. The
   ~40% miss rate in the standard layout is tail loss, not rule
   difficulty.
2. **But name/version did NOT collapse in the tail — because tail loss is
   only VISIBLE through unguessable rules.** `Name: nanoeuler` and
   `Version: 2.4.1` are inferable from the environment (R8's field list
   mid-file, `src/VERSION`, CHANGELOG) even if the tail rule is never
   read, so a lost tail usually produces the right answer anyway. The one
   `version` FAIL (run 20) is the exception that proves the mechanism:
   the model wrote `Version: v2.5-dev` — the stale README badge — i.e. a
   detected tail-loss event where the never-read tail rule ("use
   src/VERSION, ignore badges") was exactly the thing that would have
   prevented it. Same run had `Checked: manual` correct (head rule).
3. **Bench-design lesson: a tail-read probe must be an ARBITRARY literal
   that cannot be inferred from anything else in the environment.**
   `Checked: manual` is a strong probe (detects ~40% loss); name/version
   are weak probes (detect ~5% — only via the badge trap). The true
   tail-loss rate under the permuted layout is therefore underestimated
   by its rubric; the layout's purpose (causality test) is served, but it
   must not be read as "the tail is now fine".
4. **The harness mitigation is now validated quantitatively**: putting
   the arbitrary-literal rules at the top of the file is worth ~55%→95%
   PASS on this model. Direct consequence for real workflows (QWEN.md,
   SKILL.md, ebuild RULES): state attestation/format literals and
   evidence requirements in the FIRST screenful; keep only
   environment-inferable guidance in the tail. Permuted runs are also
   slightly faster (mean ~26 s vs ~30 s) — the needed rules arrive
   earlier.

Side observation: `summary` held 20/20 in permuted vs 17-18/20 standard
despite not moving (slot B in both layouts) — within noise, not
interpreted.

## Reference results — 2026-07-14: gpt-oss-20b quant/finetune variants (MXFP4-Aggressive vs F16 vs HERETIC)

Three GGUFs of the same base model (OpenAI gpt-oss-20b, MoE 20.91B params,
36 layers), full `bench_05` 10-task suite, `TASK_TIMEOUT=900`, run
back-to-back on one `aillama switch` chain so page-cache state is
comparable:

- `gpt-oss20b` — HauhauCS Uncensored-Aggressive, native MXFP4, 11.27 GiB.
  Fits fully in VRAM at 131072 ctx, no `--n-cpu-moe` (existing profile).
- `gpt-oss20b-f16` — unsloth F16, 12.83 GiB (non-MoE tensors upcast to F16,
  MoE experts stay native precision). Does **not** fit fully at 131072 ctx:
  floor `--n-cpu-moe 2` (`1` OOMs the compute buffer), 15.4/16.3 GiB.
- `gpt-oss20b-heretic` — DavidAU HERETIC-uncensored NEO-Imatrix finetune,
  IQ4_NL, 12.6 GiB. Also doesn't fit fully: floor `--n-cpu-moe 1` (`0`
  OOMs), profile set to `2` for the same margin as `-f16`.

| Task | gpt-oss20b (MXFP4) | gpt-oss20b-f16 | gpt-oss20b-heretic |
|------|---------------------|-----------------|----------------------|
| bugfix | PASS 4/4 (100%) 12s | PASS 4/4 (100%) 21s | PASS 4/4 (100%) 18s |
| scratch | PASS 4/4 (100%) 14s | PASS 4/4 (100%) 33s | PASS 4/4 (100%) 42s |
| lru | PASS 8/8 (100%) 18s | PASS 8/8 (100%) 52s | PASS 8/8 (100%) 62s |
| multifile | PASS 5/5 (100%) 16s | PASS 5/5 (100%) 32s | PASS 5/5 (100%) 43s |
| intervals | PASS 12/12 (100%) 13s | PASS 12/12 (100%) 25s | PASS 12/12 (100%) 38s |
| fsm | PASS 13/13 (100%) 47s | PASS 13/13 (100%) 25s | PASS 13/13 (100%) 334s |
| template | PASS 10/10 (100%) 40s | PASS 10/10 (100%) 214s | **FAIL 4/10 (40%)** 147s |
| interp | **FAIL — used eval/exec/compile** 37s | **PASS 13/13 (100%)** 350s | **FAIL 0/13 (0%)** 197s |
| perf | PASS 6/6 (100%) 55s | PASS 6/6 (100%) 39s | PASS 6/6 (100%) 31s |
| regex | PASS 14/14 (100%) 73s | **FAIL 6/14 (42%)** 222s | **FAIL — used re/regex/importlib** 40s |
| **verdicts** | **9/10** | **9/10** | **6/10** |

Easy+mid tier (6 tasks: bugfix/scratch/lru/multifile/intervals/fsm) is a
clean sweep for all three — still not discriminating at this capability
level, same pattern as every other strong-model pairing in this file.

**`gpt-oss20b-f16` fixed the MXFP4 quant's `interp` failure but broke
`regex`.** The MXFP4 baseline disqualified itself on `interp` by calling
banned builtins (`eval`/`exec`/`compile`) — a genuine capability/judgment
gap, not a close call. F16 solved the same task cleanly, 13/13, at the
cost of 350s (vs 37s to fail) — plausible that the extra precision on
attention/embedding tensors changed the model's judgment about which
approach to reach for. But F16 then regressed on `regex` (6/14, 42%) where
the MXFP4 quant went 14/14 clean. Net effect on this run: a wash in verdict
count (9/10 both) with the failures swapped to different tasks — not a
clear win for the bigger file, matching the standing lesson elsewhere in
this doc ("bigger file ≠ better," Q5_K_M/UD-Q4_K_XL entries above) but this
time with an offsetting win rather than a pure regression.

**`gpt-oss20b-heretic` is clearly the weakest of the three** — the only one
to fail the parser tier twice outright (`template` 4/10, `interp` 0/13) plus
disqualify itself on `regex` by importing a banned module (`re`/`regex`/
`importlib` — reaching for the standard library instead of implementing the
engine, the opposite failure mode from MXFP4's `interp` eval/exec use, but
the same category: shortcut instead of engineering). Also the slowest
wall-clock model in the trio on `fsm` (334s vs 25–47s) with no accuracy
payoff. The IQ4_NL quantization plus whatever the HERETIC finetune changed
did not help agentic coding capability on this hardware — no reason to
prefer it over the existing MXFP4-Aggressive profile.

**Verdict:** keep `gpt-oss20b` (MXFP4-Aggressive) as the default fast-tier
profile — it fits fully in VRAM (no `--n-cpu-moe`, fastest wall-clock by a
wide margin on every task) and its only failure is a known, narrow gap.
`gpt-oss20b-f16` is a reasonable fallback if a future task specifically
needs `interp`-style capability and can tolerate ~2-6x slower generation
from partial CPU offload. `gpt-oss20b-heretic` has no identified use case
so far; kept in `models.conf` for reference but not recommended.

## bench_05 changes — 2026-07-14 (evening): two fast trap-dense tasks (codec, toposort)

Motivation: the easy+mid tier (6 tasks) no longer discriminates strong
models at all, and the parser tier (template/interp/regex) discriminates
at the cost of 150–900 s per task *and* demonstrated run-to-run sampling
noise (see the Q8_0 section below and Ornith's regex saga). What was
missing: tasks that are **hard in reasoning but small in code** — solved
in well under two minutes by a capable model, failed on precise edge
cases (not on time) by a weaker one. Two new capability axes no existing
task touches:

- **`codec`** (12 tests) — binary frame codec: LEB128 varint length +
  payload + XOR checksum; `decode` must raise on truncated varint,
  truncated payload, bad checksum, dangling bytes. Several tests assert
  exact byte literals (`b"\x02AB\x03"`, `b"\xc8\x01"`). Byte/bit-level
  reasoning is exactly where quantization damage tends to show first.
- **`toposort`** (11 tests) — deterministic topological sort returning
  the lexicographically smallest valid order + `CycleError(ValueError)`.
  The determinism requirement is the trap: naive FIFO Kahn passes only
  9/11 (verified against a deliberately naive implementation); a correct
  solution needs a heap over the ready set. Graph reasoning + exception
  hierarchy in ~30 lines.

Both oracle-verified before use (reference solutions pass 12/12 and
11/11 through the script's own `setup_*`/`verify_*` path). Inserted into
the default `TASKS` between `fsm` and `template`.

First validation run (`TASKS=codec,toposort`,
`MODELS=gpt-oss20b,gpt-oss20b-q8_0`): all four cells clean PASS at 100%,
12–24 s each. So for the gpt-oss-20b capability class these two land in
the non-discriminating mid tier alongside intervals/fsm — they did NOT
separate MXFP4 from Q8_0, and gave no signal on the "was Q8_0's 9/10
lucky" question. They stay in the default set anyway: ~1 min of total
wall-clock per model buys coverage of two previously untested axes, and
like intervals/fsm they should start separating once capability drops
(weaker quants, smaller models) — that's where quantization damage on
byte-level reasoning would show. The top-tier discriminator remains the
parser tier, and given its demonstrated sampling noise, repeated runs
are the only honest way to rank models there.

**Comparability boundary #2: verdicts before this change are out of 10
tasks, after it out of 12.** (Boundary #1 was 8→10 earlier the same day —
see "pass-ratio SCORE + two mid-tier tasks" below.) Don't compare raw
verdict counts across either line.

## Reference results — 2026-07-14: gpt-oss-20b Q8_0 added — best of the small-file class, but baseline is noisier than first thought

Follow-up to the MXFP4/F16/HERETIC comparison above. The size table for
unsloth's `gpt-oss-20b-GGUF` repo is nearly flat from Q2_K (11.5 GB) to
Q8_0 (12.1 GB) — confirms the pattern already seen with F16/MXFP4: gpt-oss
keeps MoE expert tensors at native ~4-bit precision regardless of quant
label, only non-expert tensors (attention/embeddings/norms) actually
change size. Only `UD-Q8_K_XL` (13.2 GB) and `F16` (13.8 GB) break that
flat pattern. Picked `Q8_0` (12.11 GiB) as the best quality available
while still staying close to the already-fitting MXFP4-Aggressive
footprint (12.10 GB decimal — nearly identical file size).

Boot-tested at `-ngl 99 -c 131072`, no `--n-cpu-moe`: **fits fully**,
15.0/16.3 GiB — same class as MXFP4-Aggressive, unlike F16/HERETIC which
both needed CPU offload. Added as profile `gpt-oss20b-q8_0`.

`bench_05` vs `gpt-oss20b` (MXFP4-Aggressive), same-day re-run:

| Task | gpt-oss20b (this run) | gpt-oss20b-q8_0 |
|------|------------------------|-------------------|
| bugfix | PASS 4/4 (100%) 10s | PASS 4/4 (100%) 9s |
| scratch | PASS 5/5 (100%) 18s | PASS 4/4 (100%) 32s |
| lru | PASS 8/8 (100%) 21s | PASS 8/8 (100%) 20s |
| multifile | PASS 5/5 (100%) 19s | PASS 5/5 (100%) 21s |
| intervals | PASS 12/12 (100%) 14s | PASS 12/12 (100%) 15s |
| fsm | PASS 13/13 (100%) 48s | PASS 13/13 (100%) 15s |
| template | **FAIL 1/10 (10%)** 51s | PASS 10/10 (100%) 23s |
| interp | **FAIL 0/13 (0%)** 154s | PASS 13/13 (100%) 163s |
| perf | PASS 6/6 (100%) 49s | PASS 6/6 (100%) 24s |
| regex | PASS 14/14 (100%) 148s | FAIL 11/14 (78%) 100s |
| **verdicts** | **7/10** | **9/10** |

**The MXFP4 baseline is noisier than the earlier same-day run suggested.**
Three runs of `gpt-oss20b` today: 9/10 (`interp` FAIL via banned
eval/exec), 9/10 again isn't quite right — the *comparison* run against
f16/heretic gave 9/10 with `interp` FAIL, now this run gives **7/10** with
*different* tasks failing entirely (`template` 1/10, `interp` 0/13, and
`regex` flipped from FAIL→PASS). Same weights, same quant, same profile,
zero code changes between runs — this is sampling non-determinism on the
same scale already logged for Ornith's `regex` task
([[feedback-ncmoe-probe-order]] territory: measure repeatedly, don't trust
one run). **Do not read the `gpt-oss20b` column above as a regression** —
treat both 7/10 and 9/10 as within its normal noise band until more runs
accumulate.

`gpt-oss20b-q8_0`'s own result stands on its own regardless: **9/10**,
clean sweep through `perf`, only a near-miss on `regex` (11/14, 78% — not
a disqualification like HERETIC's banned-import FAIL). Wall-clock is
dramatically faster than `gpt-oss20b-f16` on every shared task (`interp`
163s vs 350s, `template` 23s vs 214s, `regex` 100s vs 222s) because it
needs zero CPU offload — same VRAM-fit class as the MXFP4 baseline, so no
speed penalty for the quality gain.

**Verdict: `gpt-oss20b-q8_0` looks like the strongest of the four
gpt-oss-20b variants tested today** — fits fully in VRAM like the MXFP4
baseline (no wall-clock penalty), and its one failure is a near-miss
rather than a disqualification. Worth promoting over `gpt-oss20b` as the
default fast-tier profile, but given the baseline's demonstrated run-to-run
noise, treat this as a lead, not a settled verdict, until `gpt-oss20b-q8_0`
itself gets a repeat run.

### Repeat run (same day, later): `MODELS=gpt-oss20b,gpt-oss20b-q8_0 TASKS=template,interp,regex`

The repeat this section called for. Parser-tier only (3/10 tasks):

| Task | gpt-oss20b (MXFP4) | gpt-oss20b-q8_0 |
|------|---------------------|-------------------|
| template | FAIL 8/10 (80%) 157s | PASS 10/10 (100%) 47s |
| interp | FAIL — used eval/exec/compile — 128s | PASS 13/13 (100%) 67s |
| regex | FAIL 12/14 (85%) 255s | FAIL 13/14 (92%) 271s |

`gpt-oss20b-q8_0` reproduced its exact `template`/`interp` clean-PASS result
from the first run and improved its `regex` near-miss (78%→92%). `gpt-oss20b`
(MXFP4) failed all three parser tasks this time, including a `regex` result
that flipped from a clean 14/14 (first run) to FAIL 12/14 — the baseline's
noise now covers *every* parser task, never a clean sweep across two runs.

**Verdict confirmed: `gpt-oss20b-q8_0` promoted to the default fast-tier
profile**, superseding `gpt-oss20b` (MXFP4-Aggressive). Across two runs
q8_0 never failed to pass `template`/`interp` and only ever near-missed
`regex`; MXFP4 has never passed `template` or `interp` and lost its one
`regex` clean sweep on the repeat. Same VRAM-fit class, same wall-clock
tier — no cost to the switch. MXFP4 profile kept in `models.conf` as a
fallback/reference, not recommended for new work.

## Reference results — 2026-07-15: 3× full 12-task sweep, gpt-oss20b-q8_0 vs ornith-128k, TASK_TIMEOUT=1200

Deliberate weak-spot hunt on the two current fleet leaders: full default
`TASKS` (all 12), 3 repeats each, `TASK_TIMEOUT` raised globally to 1200s
(motivation: `ornith-128k` is slow, `gpt-oss20b-q8_0` is fast but its
`interp`/`regex` reasoning chains run long). Command:

```bash
for i in 1 2 3; do
  MODELS=gpt-oss20b-q8_0,ornith-128k TASK_TIMEOUT=1200 ./bench_05_agentic.sh
done
```

### gpt-oss20b-q8_0 — 34/36 verdicts PASS

| Task | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| bugfix..perf (9 tasks) | PASS 100% (all 3 runs, all 9 tasks) | | |
| interp | PASS 126s | PASS 89s | **FAIL 0/13 (0%) 37s** |
| regex | FAIL 12/14 (85%) 139s | PASS 14/14 188s | PASS 14/14 95s |

The `interp` collapse in run 3 was not a reasoning failure: `agent.log` was
one line — `[API Error: The model produced output that does not match the
expected peg-native format]` — qwen-code choked on a malformed tool-call
response from the server, and the task died in 37s with nothing built. New
failure signature, added to the skill's catalogue. Otherwise fully
deterministic: 9 non-parser tasks clean 3/3, `template` clean 3/3, `regex`
2/3 clean (14/14) with one near-miss.

### ornith-128k — 30/36 verdicts PASS (4 more if you count the 100%-scored TIMEOUT)

| Task | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| bugfix..toposort, perf (9 tasks) | PASS 100% (all 3 runs, all 9 tasks; wall time 81–180s) | | |
| template | PASS 235s | **FAIL 4/10 (40%) 497s** | PASS 345s |
| interp | TIMEOUT 13/13 (100%!) 1200s | PASS 623s | TIMEOUT 10/13 (76%) 1200s |
| regex | TIMEOUT — (unscored) 1200s | TIMEOUT 11/14 (78%) 1200s | TIMEOUT 11/14 (78%) 1200s |

`interp` run 1's TIMEOUT actually scored 13/13 (100%) — the solution was
correct, the agent just didn't wrap up (extra polishing/verification) before
the 1200s cutoff. Genuine near-miss only in run 3 (76%).

**`regex` timed out in all 3 runs, even at the raised 1200s budget** — this
settles it: raising `TASK_TIMEOUT` does not fix `ornith-128k` on this task,
it's a reproducible capability/approach ceiling (verbose backtracking
implementation + heavy self-verification loop), not a clock problem. Run 1's
`regex` had no score at all — inspecting `/tmp/bench-agentic` showed the
same pattern as 07-14's fifth data point: model was mid-refactor of its own
AST classes when SIGTERM hit.

**Conclusion**: both models are near-clean on the easy+mid tier (18/18 and
27/27 respectively — fully non-discriminating, as expected). The parser
tier is where each model's specific weak spot lives: `gpt-oss20b-q8_0`'s is
an occasional infra-level tool-call parse error (rare, not a reasoning
gap); `ornith-128k`'s is a consistent, reproducible inability to finish
`regex` in any tested time budget. Neither finding changes the existing
role split (`gpt-oss20b-q8_0` = fast tier default, `ornith-128k` = serious
agentic work default) but both are now documented weak spots to watch for.

**Correction from the 5-run single-model repeat below: the `interp` collapse
was not rare** — see below, it recurred at a ~40% rate in a same-day,
same-condition batch.

### Follow-up: `gpt-oss20b-q8_0` alone, 5× repeat, `TASK_TIMEOUT=1200` (same day)

```bash
for i in 1 2 3 4 5; do
  MODELS=gpt-oss20b-q8_0 TASK_TIMEOUT=1200 ./bench_05_agentic.sh
done
```

| Task | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|------|-------|-------|-------|-------|-------|
| bugfix..perf, template (10 tasks) | PASS 100% (all 5 runs, all 10 tasks) | | | | |
| interp | PASS 173s | **FAIL 8/13 (61%) 66s** | **FAIL 0/13 (0%) 46s** | PASS 248s | PASS 233s |
| regex | **FAIL 10/14 (71%) 367s** | PASS 49s | PASS 50s | **FAIL 13/14 (92%) 86s** | PASS 165s |

50/50 clean on every non-parser task plus `template` — that part is fully
solid. But `interp` and `regex` are **both** noisy at roughly the same rate
(3 PASS / 2 FAIL each, 60%), not just `regex` as the earlier same-day
section framed it. The two `interp` FAILs are the interesting bit: **fast
failures** (46s, 66s) vs. its normal passing wall-time (173–248s) — the
model takes a quick wrong turn rather than grinding and running out of
budget, the opposite time signature from a near-miss. Whether these were
the same "malformed tool-call" API-format error as the run documented
above is unknown — `WORKROOT` is shared across iterations of the loop, so
each run's `/tmp/bench-agentic` overwrote the previous one's `agent.log`
before it could be inspected. **Practical fix for next time**: parameterize
`WORKROOT=/tmp/bench-agentic-run$i` per iteration to keep every run's
artifacts for postmortem.

**Revised verdict**: `gpt-oss20b-q8_0` is rock-solid (100%) on 10 of 12
tasks but has two comparably-noisy parser-tier weak spots, `interp` and
`regex`, each landing a real FAIL roughly 2 times in 5 — not a single rare
fluke. Still the right fast-tier default (non-parser-tier work is
unaffected, and `ornith-128k`'s own parser-tier failure rate on `regex` is
worse — 3/3 TIMEOUT), but don't expect a clean 12/12 sweep as the norm on
repeated runs; expect one parser-tier miss more often than not.

### Second follow-up: 9 more single-model runs (same day) — bug found in the verifier itself

```bash
for i in 1 2 3 4 5 6 7 8 9; do
  MODELS=gpt-oss20b-q8_0 TASK_TIMEOUT=1200 ./bench_05_agentic.sh
done
```

Combined with the 5-run batch above: **14 total single-model runs today.**
Per-task pass rate across all 14:

| Task | Pass rate | Notes |
|------|-----------|-------|
| bugfix, lru, multifile, intervals, fsm, perf | 14/14 (100%) | |
| **codec** | 13/14 (93%) | one FAIL 0/12 (0%, 21s) — first-ever failure seen on this task |
| **toposort** | 13/14 (93%) | one FAIL 0/11 (0%, 32s) — first-ever failure seen on this task |
| **template** | 14/14 (100%) | the only parser-tier task that has never failed |
| **interp** | 10/14 (71%) | 4 FAILs: 0%, 61%, 69%, 92% |
| **regex** | 8/14 (57%) | worst task: 6 FAILs across two batches |

Two new findings:

1. **The "rock-solid non-parser tier" claim needed correcting.** `codec` and
   `toposort` had been 100% in every prior comparison (2026-07-14 onward)
   but each failed completely (0%) once in this larger sample — still much
   rarer than the parser tier (~7% vs ~30-40%), but not literally
   deterministic. `template` is the only task with a perfect 14/14 record.

2. **A real bug in `bench_05_agentic.sh`'s verifiers, found via a raw
   traceback in run 4's `regex` FAIL**: the NOTE said "used re/regex/
   importlib" (implying a banned-import disqualification), but the actual
   stdout showed `ast.parse()` itself crashing with `IndentationError` —
   the generated `rx.py` had invalid syntax and never even parsed.
   `verify_regex()` and `verify_interp()` both ran `ast.parse()` unguarded,
   so any `SyntaxError` was caught by the same `||` handler that reports
   banned constructs, mislabeling "code doesn't parse" as "code uses a
   forbidden import/builtin". **Fixed**: both verifiers now catch
   `SyntaxError` explicitly and exit 2 (distinct from exit 1 for an actual
   banned construct), so the reported NOTE is now "rx.py/calc.py has
   invalid syntax" when that's the real cause. This means some past
   "disqualification" NOTEs (this run 4, and possibly HERETIC's `interp`
   FAIL on 2026-07-14) may have actually been syntax errors, not banned-
   construct violations — the practical verdict (FAIL either way) doesn't
   change, but the stated *reason* might be wrong in older BENCH.md
   entries. Not worth re-litigating old verdicts over, but trust the NOTE
   text going forward now that the fix is in.

## Reference results — 2026-07-14: gpt-oss20b vs Ornith-1.0-9B (new small Ornith)

`deepreinforce-ai/Ornith-1.0-9B-GGUF` (Q8_0, 9.53 GiB, arch `qwen35` dense,
32 layers — smaller sibling of the RL-agentic `ornith-1.0-35b` used
throughout this file). Boots fine at full `-ngl 99 -c 131072`, no
`--n-cpu-moe` needed (same "fits whole" class as Qwythos/dsv4flash).
Added as aillama profile `ornith-9b`. `bench_05` run right after aborting
the Gemma4 session, paired with `gpt-oss20b` per user's request (fast-tier
comparison, not the 35B-class ornith-128k):

| Task | gpt-oss20b | ornith-9b |
|------|-----------|-----------|
| bugfix | PASS 4/4 (100%) 9s | PASS 4/4 (100%) 19s |
| scratch | PASS 4/4 (100%) 64s | PASS 8/8 (100%) 69s |
| lru | PASS 8/8 (100%) 24s | PASS 8/8 (100%) 36s |
| multifile | PASS 5/5 (100%) 59s | PASS 5/5 (100%) 31s |
| intervals | PASS 12/12 (100%) 46s | PASS 12/12 (100%) 40s |
| fsm | PASS 13/13 (100%) 50s | PASS 13/13 (100%) 33s |
| template | PASS 10/10 (100%) 60s | **TIMEOUT 3/10 (30%)** 900s |
| interp | FAIL 9/13 (69%) 63s | **TIMEOUT 12/13 (92%)** 900s |
| perf | **TIMEOUT 0/6 (0%)** 900s | **TIMEOUT 0/6 (0%)** 900s |
| regex | FAIL 9/14 (64%) 716s | **TIMEOUT 2/14 (14%)** 900s |
| **verdicts** | **7/10** | **6/10** |

Both easy+mid tiers (6 tasks) are a clean sweep for both models — the new
`intervals`/`fsm` tasks still aren't discriminating this pair, same as the
07-14 gemma4-fable5 run (fsm/intervals only start separating models once
capability drops much further, e.g. gemma4-qat's stream-hang or a weaker
9B).

**perf: both models scored a hard 0/6, for two unrelated reasons** — dug
out of the generated code rather than left as opaque TIMEOUTs:

- `gpt-oss20b` wrote a Fenwick tree (the right data structure — same
  approach it used in an earlier successful run) but with a genuine
  initialization bug: `_ensure_size()` grows the backing array by doubling
  from `self._max` instead of `self._size`, and `_max` starts at `0` for
  the default (no-arg) constructor — `0 << 1` is `0` forever, so the loop
  never terminates. Every `add()` call hangs the process. Bad luck on
  sampling, not a repeat of a known capability gap — the model solved this
  exact task cleanly (PASS, 25s, correct Fenwick tree) earlier today.
- `ornith-9b` used `self._keys.insert(pos, ...)` — a sorted-list insertion,
  the *exact anti-pattern the prompt explicitly warns against* — giving
  O(N) per add and O(N²) total across 400K adds, genuinely too slow to
  finish, not a bug. This is a real regression vs. the 35B `ornith-128k`,
  which built a proper Fenwick tree on the same task with 10× time-budget
  margin (07-12 tiebreak round, see below) — the smaller RL-tuned model
  didn't transfer that engineering judgment down to 9B.

**template/interp/regex: ornith-9b is dramatically slower, but scoring
shows genuine partial progress, not stalls** — `interp` in particular is a
striking near-miss: **12/13 (92%)** at the 900s cutoff, i.e. one bug away
from a clean pass, just like `ornith-128k`'s recurring near-misses on
`regex` elsewhere in this file. The 9B model is clearly using the same
RL-trained "keep iterating until tests pass" strategy as its 35B parent,
it just needs more wall-clock per task at this size — `TASK_TIMEOUT=1200`
(the standard bump used for `interp`/`perf`-tier tasks on the 35B) is worth
a retry before writing off `ornith-9b` on the parser tier.

**Verdict:** `gpt-oss20b` still wins on verdict count (7/10 vs 6/10) and is
dramatically faster wall-clock everywhere except its own perf bug, but the
comparison is muddied by one-off failures on both sides (a bug, not a
capability gap, for gpt-oss20b's perf; a near-miss under time pressure, not
a wall, for ornith-9b's template/interp/regex). `ornith-9b` is worth a
second run at `TASK_TIMEOUT=1200` for the three timeout tasks before
concluding anything about its ceiling relative to `gpt-oss20b` — on this
run alone it looks like the RL-agentic training transfers to 9B scale, just
slower.

## Reference results — 2026-07-14: Gemma4-12B candidates — both rejected

New arch `gemma4` (dense, 48 layers, hybrid local/global attention like
gemma3), llama.cpp 9988 — confirmed loadable, not a compat problem. Two
variants pulled: `yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2`
(Q8_0, 12.7 GiB, claims an agentic finetune) and
`HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced` (Q4_K_M, 7.38 GiB,
usual uncensored-finetune family). Boot floors: `gemma4-fable5` OOMs the
compute buffer at 131072 ctx, 65536 is razor-thin (~960 MiB free, boots
inconsistently — one clean boot, one OOM under bench_06's server test with
nothing else running), 32768 is the only *reliable* ceiling; `gemma4-qat`
has much more margin (11.0/16.3 GiB at full 131072 ctx).

`bench_06_dense_generic.sh` throughput (both healthy at pp/tg, no red
flags): `gemma4-qat` pp 3827→1105 / tg 84.3→64.6 tok/s (depth 0→131072);
`gemma4-fable5` pp 4203→3178 / tg 53.3→50.9 tok/s (depth 0→16384, 65536
sweep point OOM'd). Real-request `/completion` test on `gemma4-fable5`
produced `gen_n=1` (model emits EOS almost immediately on raw non-chat
continuation of filler text) — a red flag for an "agentic" finetune, though
not necessarily disqualifying since real usage goes through the chat
template (see bench_05 below, which does use the chat template and still
failed).

**bench_05 agentic (interrupted mid-run, `MODELS=gemma4-fable5,gemma4-qat,gpt-oss20b`
— stopped by user judgement before finishing, gpt-oss20b never ran):**

| Task | gemma4-fable5 | gemma4-qat |
|------|---------------|------------|
| bugfix | PASS 4/4 (100%) 80s | PASS 4/4 (100%) 29s |
| scratch | FAIL 0/2 (0%) 350s | **FAIL (stream hang) 266s** |
| lru | FAIL 9/10 (90%) 196s | **FAIL (stream hang) 267s** |
| multifile | PASS 5/5 (100%) 101s | **FAIL (stream hang) 267s** |
| intervals | PASS 12/12 (100%) 57s | **FAIL (stream hang) 270s** |
| fsm | PASS 13/13 (100%) 86s | **FAIL (stream hang) 346s** |
| template | FAIL 4/10 (40%) 280s | **FAIL (stream hang) 271s** |
| interp | FAIL 1/13 (7%) 568s | not reached (killed) |
| perf | PASS 6/6 (100%) 161s | not reached |
| regex | FAIL 0/14 (0%) 453s | not reached |
| **verdicts** | **5/10** | **1/6 completed** |

`gemma4-qat` is disqualifying on infrastructure grounds, not capability:
every task after `bugfix` hit the identical client-side error —
`[API Error: No stream activity for 240000ms after N chunks (stream
lifetime: ~250-320s)]` — at wildly different chunk counts (231 to 5906),
meaning the model reliably drops into a dead/looping generation state
partway through *every* agentic task regardless of task shape or how far
along it was. 100% reproducible across 5/5 completed non-trivial tasks.
Not a timeout-budget problem (extending `TASK_TIMEOUT` would not help — the
stream is dead, not slow) and not this bench's harness (`gpt-oss20b` and
every other profile stream fine on the same qwen-code/llama.cpp stack).

`gemma4-fable5` at least produces results, but they're weak for a
same-class-size comparison — 5/10 verdicts, with two near-total capability
failures (`interp` 1/13, `regex` 0/14) worse than 9B `dsv4flash` (which at
least got partial credit) and no better than 20B `gpt-oss20b` despite being
in the same 12B weight class with a claimed "agentic" finetune. Combined
with the degenerate raw-completion behavior and the unreliable 65536 ctx
boot, not worth pursuing further.

**Verdict: both Gemma4-12B variants rejected.** `gemma4-qat` is unusable
for agentic work (stream-hang bug, not a capability question — worth a
quick check if it reproduces with `--jinja` off or a different chat
template, but not worth spending more bench time on before that). No
further testing planned unless the stream-hang is root-caused; `gemma4-fable5`
would need a second, less generous look only if the hang turns out to be
template-specific (i.e. affects fable5 too and was suppressing its true
capability) — on the numbers gathered here alone it doesn't beat the
existing fleet.

## bench_05 changes — 2026-07-14: pass-ratio SCORE + two mid-tier tasks

Two gaps showed up in the 07-14 sessions:

1. **Bare FAIL hides the margin.** `dsv4flash` and `gpt-oss20b` both scored
   5/8 verdicts, but on `interp` dsv4flash passed **0/13** tests (parser
   non-functional) while gpt-oss20b passed **10/13** (one operator bug) —
   the summary table showed the identical `FAIL`. Partial results had to be
   dug out of pytest logs by hand. Fix: `scored_pytest()` in
   `bench_05_agentic.sh` now records `passed/total (pct%)` for every task
   in a SCORE column; TIMEOUT verdicts run the verifier anyway and score
   the work-in-progress at cutoff (ornith's recurring "TIMEOUT but 13/14 on
   regex" is now visible without manual inspection).
2. **Difficulty was bimodal.** Five easy tasks (nearly every model passes,
   including 9Bs) and three parser-class tasks (small models fail
   wholesale) — nothing in between, so mid-size models separated only via
   anecdote. Two mid-tier tasks added, both oracle-verified against
   reference implementations before use (12/12, 13/13):
   - `intervals` — booking `Scheduler`: half-open `[start, end)` overlap
     rejection (adjacency allowed), exact-match cancel, earliest-free-slot
     search (`next_free(t, duration)` incl. exact-fit gaps). Algorithmic
     mid-tier: sorted-interval reasoning without a full parser.
   - `fsm` — `Order` state machine: guarded transitions raising
     `InvalidTransition`, accumulating `pay()` with *atomic* overpay
     rejection (`ValueError`, nothing applied), refunds from `cancel()`,
     state history that failed events must not touch. Spec-fidelity
     mid-tier — exactly where glm-flash-class models slipped (declare
     success after one weak pass).

Default `TASKS` is now 10 entries (`...multifile,intervals,fsm,template...`);
verdict totals in tables **before this section are out of 8**, later runs
are out of 10 — don't compare the raw counts across that boundary.

## Reference results — 2026-07-14: gpt-oss20b vs ornith-128k, full 10-task suite (first SCORE run)

First run of the expanded 10-task `bench_05` (adds `intervals`, `fsm`) with
the new SCORE column, `MODELS=gpt-oss20b,ornith-128k`, `TASK_TIMEOUT=900`.

| Task | gpt-oss20b | ornith-128k |
|------|-----------|-------------|
| bugfix | PASS 4/4 (100%) 10s | PASS 4/4 (100%) 89s |
| scratch | PASS 4/4 (100%) 29s | PASS 5/5 (100%) 114s |
| lru | PASS 8/8 (100%) 24s | PASS 8/8 (100%) 142s |
| multifile | PASS 5/5 (100%) 21s | PASS 5/5 (100%) 150s |
| intervals | PASS 12/12 (100%) 13s | PASS 12/12 (100%) 117s |
| fsm | PASS 13/13 (100%) 46s | PASS 13/13 (100%) 89s |
| template | PASS 10/10 (100%) 79s | PASS 10/10 (100%) 230s |
| interp | **FAIL 11/13 (84%)** 142s | PASS 13/13 (100%) 470s |
| perf | PASS 6/6 (100%) 49s | PASS 6/6 (100%) 240s |
| regex | **FAIL 8/14 (57%)** 201s | **TIMEOUT 0/14 (0%)** 900s |
| **verdicts** | **8/10** | **9/10** |

New mid-tier tasks (`intervals`, `fsm`) did **not** separate these two
models at all — both 100% in well under a minute (gpt-oss20b) or a few
minutes (ornith). They're doing their job as an easy/hard midpoint, just
not the axis that splits *this* pair; expect them to matter more against
weaker models (dsv4flash, glm-flash) where spec-fidelity was the failure
mode.

The SCORE column earns its keep on `interp`: gpt-oss20b's FAIL is now
visibly a near-miss (11/13, one operator bug) rather than indistinguishable
from a wholesale failure — consistent with the 07-14 dsv4flash comparison
where an "equivalent" FAIL was 0/13.

**Fifth regex data point for ornith-128k, and a new failure mode.** Unlike
the prior three TIMEOUTs (all "stuck retrying the same last unfixed case",
scoring 13/14 or 0/14-with-visible-progress), this run's killed `rx.py`
(328 lines, clearly mid-edit) scores a hard **0/14**: `count_groups()`
references a helper class `_Seq` that is never defined anywhere in the
file — the model was mid-rename/refactor of its AST node classes when the
900s SIGTERM landed, leaving the whole module non-importable-in-context
(a bare `NameError`, not a logic gap). Running tally across sessions:
PASS 225s (lucky) / FAIL 2122s 13/14 (07-13) / TIMEOUT 900s 13/14-in-progress
(07-14) / TIMEOUT 1200s 13/14 (07-14 retest) / **TIMEOUT 900s 0/14, broken
mid-refactor (07-14, this run)**. Confirms the earlier conclusion harder:
ornith's regex task duration/outcome is genuinely non-deterministic
per-session, now with two distinct failure shapes (near-miss stall vs.
mid-refactor crash) rather than one.

**Verdict:** gpt-oss20b's speed advantage holds (all easy+mid tasks in
under 50s vs ornith's 1–4 min) with one fewer verdict (8/10 vs 9/10) on
this run — still the best speed/capability tradeoff for iteration, still
not a drop-in for `ornith-128k` on the parser tier when correctness matters
most.

## Reference results — 2026-07-14: official unsloth Qwen3.6-35B-A3B-GGUF vs Ornith

Three quant variants of the **official** `unsloth/Qwen3.6-35B-A3B-GGUF`
(distinct from the `HauhauCS` uncensored fork used by the `qwen36*`
profiles) downloaded to compare against `ornith-128k`. `--n-cpu-moe` floors
via `bench_01 -b` sweep (descending probe, ~1.8 GB desktop VRAM in use):

| Variant | weight size | ncmoe floor | tg @ floor |
|---|---|---|---|
| UD-IQ4_XS | 16.50 GiB | 10 | 96.6 tok/s |
| MXFP4_MOE | 20.21 GiB | 16 | 68.1 tok/s |
| UD-Q4_K_XL | 20.81 GiB | 17 | 63.0 tok/s |

Profiles added to `~/.aillama/models.conf` as `qwen36u-{iq4xs,mxfp4,q4kxl}[-128k]`
(floor+2 @32K, floor+6/7 @128K, same margin convention as `ornith`).

### Agentic (bench_05), 128K profiles, TASK_TIMEOUT=900s

| Task | qwen36u-iq4xs | qwen36u-mxfp4 | qwen36u-q4kxl | ornith-128k |
|------|---------------|---------------|---------------|-------------|
| bugfix | PASS 57s | PASS 87s | PASS 90s | PASS 90s |
| scratch | **FAIL** 137s (file not created) | PASS 121s | PASS 95s | PASS 129s |
| lru | PASS 90s | PASS 142s | PASS 92s | PASS 87s |
| multifile | PASS 70s | PASS 118s | PASS 109s | PASS 112s |
| template | PASS 519s | PASS 187s | PASS 159s | PASS 147s |
| interp | PASS 378s | PASS 803s | **TIMEOUT** 900s | **TIMEOUT** 900s¹ |
| perf | PASS 102s | PASS 154s | PASS 149s | PASS 373s |
| regex | **TIMEOUT** 900s² | **TIMEOUT** 900s² | **TIMEOUT** 900s² | **PASS 225s**³ |
| **verdicts** | 6/8 | 7/8 | 6/8 | 7/8 (8/8 with ¹) |

¹ Re-run at `TASK_TIMEOUT=1200` (server already on `ornith-128k`, single
task): **PASS at 1067s**. Same pattern as the 07-13 tiebreak round — the
35B-class models need the 1200s budget on `interp`/`perf`-tier tasks, 900s
is tight but not always enough. Not a capability regression.

² None of the three `qwen36u-*` variants had even written `rx.py` by the
900s cutoff (`agent.log` = `"Operation cancelled."`, directory has only the
harness-provided `test_rx.py`) — a different failure mode than Ornith's
07-13 near-miss (13/14, stuck on one case): the vanilla Qwen3.6 family
doesn't get partway through a backtracking regex engine in this budget,
regardless of quant.

³ **Notable reversal of the 07-13 finding.** That session's `ornith-128k`
(same Q4_K_M weights) FAILED regex at 2122s/13/14 on a documented capability
gap — `fullmatch("(a+)(a+)", "aaa")` not backtracking across group
boundaries. This run's independently-generated solution handles that exact
case correctly (`fullmatch('(a+)(a+)', 'aaa') == ('aa', 'a')`, verified
directly) and passes all 14/14 in 225s — an order of magnitude faster than
the previous attempt's 2122s before failing. Sampling non-determinism, not
a re-test of the same solution: the gap isn't a hard architectural limit,
just inconsistently reached.

**Verdict:** none of the three official variants beats `ornith-128k`. The
biggest file (`UD-Q4_K_XL`, 20.81 GiB) is the *worst* of the three official
quants (double TIMEOUT) — same "bigger file ≠ better" lesson as the 07-13
Q5_K_M experiment. `qwen36u-mxfp4` ties Ornith on verdict count and is the
only one of the three worth keeping as a non-RL alternative; `UD-IQ4_XS`
uniquely fails an *easy* task (`scratch`) despite being fastest, and
`UD-Q4_K_XL` offers no advantage over either Ornith or the smaller unsloth
quants — both are candidates for deletion. `ornith-128k` stays the default
agentic profile.

## Reference results — 2026-07-14: 27B dense candidates — all rejected, VRAM ceiling

Five dense 27B finetunes were pulled from HF (all `arch qwen35`, 65
transformer layers) as agentic-coding candidates: `unsloth/Qwen3.6-27B-MTP-GGUF`
(UD-Q4_K_XL 17G, UD-Q3_K_XL 14G), `bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF`
(Q4_K_M 16G, no MTP head), `protoLabsAI/ThinkingCap-Qwen3.6-27B-MTP-GGUF`
(Q4_K_M-MTP 16G, NVFP4-Q4_K_M-MTP 15G), `DavidAU/...NEO-CODE...-GGUF`
(Q4_K_M 16G), `nerkyor/Qwen3.6-27B-DSV4Pro-GLM52-SFT-GPT55-RL-Coding-GGUF`
(Q4-LynnStyle 19G). Unlike Qwythos-9B-v2 (dense, fits fully at `-ngl 99`),
none of these fit whole on a 16 GB card — weight size alone (14–19 GB)
already exceeds the ~15.3 GB actually free after desktop overhead.

`-ngl 99` (full offload) OOMs on **every** variant, even at ctx 4096. Fit
requires partial CPU offload; bisected floors (`bench_06_dense_generic.sh`
probes, ctx 4096 unless noted):

| Model | size | ngl floor | max ctx @ floor-ish ngl |
|---|---|---|---|
| qwen36-27b-mtp UD-Q3_K_XL | 14G | 55–60 / 65 | 32768 (ngl 55); 65536 OOMs |
| thinkingcap-27b Q4_K_M (bottlecapai, no MTP) | 16G | 50 / 65 (55 OOMs) | 32768; 65536 OOMs |
| thinkingcap-27b-mtp Q4_K_M-MTP (protoLabsAI) | 16G | 55 / 65 | 32768 (ngl 50); 65536 OOMs |
| thinkingcap-27b-mtp NVFP4-Q4_K_M-MTP | 15G | 55 / 65 | 32768 (ngl 50); 65536 OOMs |
| qwen36-27b-neocode (DavidAU), qwen36-27b-dsv4pro-coding (nerkyor) | 16G / 19G | OOM at 99, not bisected further | — |

Surprise: the two protoLabsAI MTP files fit at `ngl 55` while the
same-labelled bottlecapai baseline OOMs at `ngl 55` and needs `ngl 50` —
same nominal "Q4_K_M", same 16 GB file size, different per-tensor bit
allocation between repos. NVFP4 (native Blackwell FP4 format) showed no
VRAM or load-time advantage over plain Q4_K_M on this llama.cpp build —
same ngl floor, same ctx ceiling, ~18s to fail vs ~4s for k-quants when it
doesn't fit (single large contiguous buffer allocation, not incremental).

**Verdict: the entire 27B dense family is rejected — hard ceiling ~32K
context on this 16 GB card, never reaches the 128K the agentic profiles
need.** All five GGUFs deleted (113 GB freed: `qwen36-27b-mtp/`,
`thinkingcap-27b/`, `thinkingcap-27b-mtp/`, `qwen36-27b-neocode/`,
`qwen36-27b-dsv4pro-coding/`). `qwen35-9b-dsv4flash` (Jackrong distill,
9B, fits fully at `-ngl 99` like Qwythos) was kept — see depth-sweep
numbers in the session this section summarizes. `ornith-128k` stays the
default 128K agentic profile; `bench_06_dense_generic.sh` (generic
MODEL/MTP_MODEL/NGL/CTX-parametrized dense bench, adapted from bench_04)
stays in the repo for future dense-model candidates that might actually
fit.

## Reference results — 2026-07-14: dsv4flash agentic (bench_05) + ornith-128k regex re-test

`qwen35-9b-dsv4flash` (Jackrong Qwen3.5-9B-DeepSeek-V4-Flash distill, dense,
fits fully at `-ngl 99`) added as aillama profile `dsv4flash` and run
through the full `bench_05` 8-task suite against `ornith-128k`
(`TASK_TIMEOUT=900`):

| Model | Task | Verdict | Time | Note |
|---|---|---|---|---|
| dsv4flash | bugfix | PASS | 21s | |
| dsv4flash | scratch | PASS | 122s | |
| dsv4flash | lru | PASS | 138s | |
| dsv4flash | multifile | PASS | 29s | |
| dsv4flash | template | FAIL | 691s | 4/10 tests — `{% for %}` never expanded, output is the literal template text |
| dsv4flash | interp | FAIL | 608s | 13/13(!) — parser non-functional, `ParseError` on nearly every construct |
| dsv4flash | perf | PASS | 28s | |
| dsv4flash | regex | FAIL | 810s | 11/14 — mostly `None` (no match at all), far worse than ornith's near-miss |
| ornith-128k | bugfix | PASS | 84s | |
| ornith-128k | scratch | PASS | 141s | |
| ornith-128k | lru | PASS | 82s | |
| ornith-128k | multifile | PASS | 96s | |
| ornith-128k | template | PASS | 474s | |
| ornith-128k | interp | PASS | 651s | |
| ornith-128k | perf | PASS | 251s | |
| ornith-128k | regex | TIMEOUT | 900s | see re-test below |
| **verdicts** | | **dsv4flash 5/8, ornith-128k 7/8 (+regex re-test)** | | |

**dsv4flash verdict:** same "classic small-model gap" as Qwythos-9B before
it — fast and solid on bugfix/scratch/lru/multifile/perf (the well-scoped
tasks), but the three parser/compiler-class tasks (template, interp, regex)
are not just slow, they're *substantively* wrong (interp: 13/13 tests
failing, not a near-miss). Confirms the pattern is architectural/scale, not
specific to Qwythos — a second 9B dense model hits the identical wall.
Not worth promoting over `ornith-128k` or `qwythos` for agentic work;
useful only where the well-scoped-task speed matters more (bugfix in 21s
vs ornith's 84s) and parser-class tasks won't come up.

**ornith-128k/regex re-test:** re-ran the single task at `TASK_TIMEOUT=1200`
(same profile already warm) hoping for the `interp`-style "just needed more
budget" outcome from 07-14 earlier today — instead: **TIMEOUT again at
1200s.** Inspected the in-progress `rx.py` manually: **13/14 tests passing**,
the *exact same* `test_backtracking_across_groups` failure
(`fullmatch("(a+)(a+)", "aaa")` not backtracking across capture-group
boundaries) as the 07-13 FAIL (2122s/13/14). Unlike `interp`, this is not a
timeout-budget problem — extending the budget doesn't help because the
model gets stuck retrying the same unfixed edge case rather than making
slow-but-steady progress. Fourth data point on this specific gap across
sessions: PASS 225s (07-14, lucky correct generation) / FAIL 2122s 13/14
(07-13) / TIMEOUT 900s (today) / TIMEOUT 1200s 13/14 (today, retest) — 3
of 4 runs hit the same capture-group backtracking bug; only extending the
timeout further would need to out-wait the retry loop, not just budget for
slower-but-correct generation. Treat ornith's regex capability as
genuinely non-deterministic per-session, not a solved gap.

## Reference results — 2026-07-14: GPT-OSS-20B-Uncensored-HauhauCS-Aggressive

`HauhauCS/GPT-OSS-20B-Uncensored-HauhauCS-Aggressive` (MXFP4, native
gpt-oss MoE quant, 20.91B params, 11.27 GiB) — fits **fully in VRAM at
`-ngl 99` with no `--n-cpu-moe`**, even at the full 131072 ctx (only the
`llama-bench -d 131072` sweep point OOMs; the real `llama-server` boot at
ctx 131072 works fine — sweep vs server allocate the compute buffer
differently). Added as aillama profile `gpt-oss20b`.

Depth sweep + real request (`bench_06_dense_generic.sh`):

| depth | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 0 | 10697.8 | 239.3 |
| 16384 | 7781.4 | 209.8 |
| 65536 | 3941.9 | 155.6 |
| real request (683 tok prompt) | pp 3437.7 | tg 213.8 |

**By far the fastest model in the fleet** — tg 213.8 tok/s vs dsv4flash's
~93 and ornith-128k's ~61 at comparable depth.

bench_05 agentic (`TASK_TIMEOUT=900`):

| Task | Verdict | Time | Note |
|---|---|---|---|
| bugfix | PASS | 9s | |
| scratch | PASS | 15s | |
| lru | PASS | 17s | |
| multifile | PASS | 17s | |
| template | FAIL | 57s | 8/10 — only `nested_loops`, `if_without_else` |
| interp | FAIL | 120s | 10/13 — `SyntaxError` parsing unary minus |
| perf | PASS | 25s | |
| regex | FAIL | 128s | 6/14 — code bug (`AttributeError: 'Matcher' object has no attribute 'index'`) plus the familiar capture-group backtracking miss |
| **verdict** | **5/8** | **~390s total** | vs dsv4flash 5/8 in ~2450s, ornith-128k 7/8(+) in ~2680s |

**Verdict:** same 5/8 as `dsv4flash` but qualitatively much stronger —
these are *near-misses* (8/10, 10/13, 6/14), not `dsv4flash`'s wholesale
failures (0/13 on interp). Combined with ~7× the throughput of ornith and
~2-6× that of dsv4flash, this is the best speed/capability tradeoff tested
today. Not a drop-in replacement for `ornith-128k` (misses the same
parser-class ceiling, plus a real bug in the regex `Matcher` class), but a
strong candidate for the "fast iteration" role Qwythos was filling — worth
a second bench_05 run to check the regex `AttributeError` isn't
introduced-vs-inherent (different agent transcript might avoid the buggy
code path), and worth probing whether the uncensored/aggressive finetune
variant matters here vs an official gpt-oss-20b quant.

## Reference results — 2026-07-13: GLM-4.7-Flash refresh + agentic

Re-run of `bench_01` on the current llama.cpp build (9999) plus the first
`bench_05` run for the `glm-flash` profile (GLM-4.7-Flash UD-Q4_K_XL,
30B-A3B MoE, 16.3 GiB), ~0.9 GB desktop VRAM in use.

**The `--n-cpu-moe` floor moved up by 2 since 2026-07-05:** bench ncmoe 8
now crashes (was the floor), and the server at 32K/q8_0 OOMs at ncmoe 10 on
the first decode (`cublasCreate: the resource allocation failed` — it loads
and answers `/health`, then dies mid-request; `-np 1` does not help). The
`glm-flash` profile in `~/.aillama/models.conf` was bumped 10 → 12.

| Config | pp | tg |
|--------|----|----|
| bench ncmoe 8 | crash (SIGABRT, was 1445/91.8 on 07-05) | — |
| bench ncmoe 10 | 1177 tok/s | 76.7 tok/s |
| bench ncmoe 14 | 905 tok/s | 64.5 tok/s |
| server 32K q8_0, ncmoe 10 | OOM on first decode | — |
| server 32K q8_0, ncmoe 12 | 977 tok/s | 63.0 tok/s |

### Agentic (bench_05), glm-flash: 4/7

| Task | glm-flash | qwen36-128k | ornith-128k |
|------|-----------|-------------|-------------|
| bugfix | PASS 63 s | PASS 58 s | PASS 76 s |
| scratch | FAIL 114 s (own tests fail) | PASS 184 s | PASS 147 s |
| lru | PASS 50 s | PASS 59 s | PASS 74 s |
| multifile | PASS 66 s | PASS 76 s | PASS 110 s |
| template | FAIL 282 s | PASS 118 s | PASS 294 s |
| interp | FAIL 130 s | PASS 642 s | PASS 384 s |
| perf | PASS 74 s | PASS 62 s | PASS 180 s |

The failures are capability, not infrastructure: `template` never parses
`{% for %}` blocks at all (8/10 tests fail, raw tags leak into output),
`interp` breaks on unary minus, precedence, lazy ternary and function-call
argument parsing (7/13 fail), and in `scratch` the agent's own CLI misreads
its spec (tie-breaking, `--top` validation; 6/9 fail). Note the FAIL times:
glm-flash gives up fast instead of iterating — it declares success after one
weak pass rather than running tests until green.

Takeaway: glm-flash matches the 35Bs on Qwythos-class micro-tasks (bugfix,
lru, multifile, perf — all PASS at qwen36-like wall-clock), but fails
exactly where the 35Bs win: parser/compiler-shaped problems and spec
fidelity. As an agentic daily driver it is dominated on both axes —
Qwythos is faster on the easy tier, qwen36/ornith are stronger on the hard
tier. Fine as a general 32K chat model (tg 63 at ncmoe 12).

### regex task (new bench_05 task) + Q4 vs Q5_K_M, Ornith-1.0-35B

New `bench_05` task: `rx.py`, a backtracking regex engine (`fullmatch` with
capture groups, char classes, `\d\w\s`, greedy `*+?`, alternation, nested
groups) from a 14-test suite; `import re`/`regex` forbidden, AST-verified
(no grep — see the 07-13 tiebreak lesson). Test suite cross-checked against
a `re`-based oracle (14/14) before use.

**`ornith-128k` (Q4_K_M, ncmoe 24)**: first attempt TIMEOUT at 1800s (still
mid-edit, not stuck); re-run at TASK_TIMEOUT=2400s → **FAIL at 2122s,
13/14 passing**. The one failure is a genuine capability gap, not noise:
`fullmatch("(a+)(a+)", "aaa")` — the engine doesn't backtrack *across*
group boundaries (first `+` group greedily eats all of "aaa", second group
has nothing left, and the engine never gives the first group back). Closest
any local model has come to solving a hard bench_05 task cleanly.

**`ornith-q5-128k` (Q5_K_M, 24 GiB, ncmoe 24)**: tried on the hypothesis
that less quantization loss might close that last gap. Floor probed
downward from 34: 22 is the floor (20 aborts at boot — same ncmoe as
Q4_K_M's floor at 128K, despite Q5 being 4.3 GiB heavier), tg is ~13-14%
slower than Q4_K_M at equal ncmoe (bigger expert tensors = more CPU-side
bytes read per token: 44.2 vs 51.3 tok/s at ncmoe 24, 48.0 vs 55.5 at 22).
Result: **TIMEOUT at 2700s (45 min) with `rx.py` never created** —
`agent.log` contains only `"Operation cancelled."` (the SIGTERM shutdown
message; qwen-code block-buffers stdout when not a TTY, so 45 minutes of
actual reasoning/tool-call activity produced no visible trace before the
kill). Worse outcome than Q4_K_M, not better.

Takeaway: **don't reach for a bigger quant to fix a capability gap on this
model class.** Q5_K_M's slower CPU-offloaded expert compute compounds with
Ornith's long RL-trained reasoning chains — the combination pushed total
wall-clock past 45 minutes without even a first draft, while Q4_K_M
produced a near-complete solution in 35. `ornith-128k` (Q4_K_M) stays the
recommended profile; the cross-group-backtracking gap is logged as a known
limitation, not something worth chasing with more precision. The Q5_K_M
GGUF (24 GiB) and the `ornith-q5-128k` profile were removed after this
result — not worth the disk space for a quant that performed worse.

### glm-flash at 128K context

GLM-4.7-Flash trains at **202 752 tokens** native ctx (`deepseek2` arch,
MLA attention, kv head_count 1), so 128K is in-distribution. It fits, but
the KV cache alone is **3.6 GiB at 128K/q8_0** — the ncmoe floor jumps
12 → 22 (20 still aborts at boot; probe *downward* from a known-good high
value, an upward probe just burns runs on OOM crashes):

| ncmoe @ 128K | pp | tg |
|--------------|----|----|
| 20 | abort at boot (KV alloc) | — |
| 22 (floor) | 609 tok/s | 42.3 tok/s |
| 24 (profile, floor+2) | 562 tok/s | 39.6 tok/s |
| 28 | 493 tok/s | 35.0 tok/s |

Profile `glm-flash-128k` added (ncmoe 24). Note the trade: at 128K
glm-flash generates 24% slower than ornith-128k (39.6 vs 52) — the MLA KV
advantage does not compensate for 10 extra expert layers pushed to CPU.

### ik_llama.cpp vs mainline — Ornith-1.0-35B @ 128K (2026-07-13)

ik-llama-cpp rebuilt from upstream HEAD the same day (version 4721,
7937465f — confirmed equal to `ls-remote` main, no newer patches exist).
All runs: Q4_K_M 19.7 GiB, KV q8_0, ~0.9 GB desktop VRAM in use.

ik bench sweep (`bench_02 -b`): ncmoe 18 → pp 857 ± 98 / tg **67.8**,
20 → 787 ± 73 / 62.6, 24 → 672 ± 63 / 54.0. tg runs ~2 tok/s above
mainline (65.7 / 60.6 at 18 / 20); pp variance is high as usual for the
fork.

Server at 128K ctx, both backends at ncmoe 24 (pp / tg tok/s):

| Prompt | mainline | ik 4721 |
|--------|----------|---------|
| short (1118 tok) | 459 / 51.3 | **595 / 50.3** (+30% pp) |
| deep (25 238 tok) | 714 / **48.9** | **730** / 46.6 (wash) |
| short, ncmoe 22 | 508 / 55.5 | 635 / 53.8 |

Takeaways:

- **The fork's +30% pp lead exists only on short prompts.** On an agentic
  ~25K-token prompt the two are equal at pp (730 vs 714) and mainline
  generates 5% faster — same pattern as qwen36 (arch `qwen35moe`), where
  the fork's GLM-era pp advantage also evaporated at depth. No reason to
  move `aillama`/`ornith-128k` off mainline.
- Unlike qwen36 at 32K, ik needs **no extra ncmoe step** on ornith at
  128K: ncmoe 24 (the mainline profile value) boots and decodes fine,
  and even 22 works on both backends (+4 tg). The profile stays at 24
  for margin — the verified-with-0.9-GB-desktop floor moves up once the
  desktop holds more VRAM (see the qwen36-128k 14→16 bump on 07-12).
- pp again *rises* with prompt depth on both backends (459→714 mainline,
  595→730 ik).

## Reference results — 2026-07-10: Qwen3.6-35B-A3B (llama.cpp vs ik fork)

Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive IQ4_XS (34.66B MoE, 40 layers,
17.43 GiB, native ctx 262K), same hardware as below.
Model: `~/models/qwen36-35b-a3b/` (symlink into the HF hub cache).

| Test | Config | pp | tg |
|------|--------|----|----|
| bench pp512/tg128 | `--n-cpu-moe 10` (floor; 9 OOMs) | 1107 tok/s | **109.4 tok/s** |
| bench pp512/tg128 | `--n-cpu-moe 12` | 966 tok/s | 93.2 tok/s |
| bench pp512/tg128 | `--n-cpu-moe 16` | 826 tok/s | 81.3 tok/s |
| server ctx 32K q8_0 | `--n-cpu-moe 10` | **1139 tok/s** | 98.4 tok/s |
| server ctx 64K q8_0 | `--n-cpu-moe 12` (10 segfaults on start) | 1009 tok/s | 88.4 tok/s |
| server ctx 128K q8_0 | `--n-cpu-moe 14` | 863 tok/s | 81.6 tok/s |

## Reference results — 2026-07-12: Qwythos-9B-v2 (dense hybrid Gated-DeltaNet)

Qwythos-9B-v2 Q8_0 (8.95B dense, arch `qwen35`, 8/32 full-attention layers,
8.86 GiB, native ctx 262K, YaRN to 1M), llama.cpp build 9924, fully on GPU
(`-ngl 99`, no offload), KV q8_0. Model: `~/models/qwythos-9b-v2/`.
Run: `./bench_04_qwythos.sh` (r=2).

| Test | Config | pp | tg |
|------|--------|----|----|
| bench pp512/tg128 | depth 0 | **5858 tok/s** | 87.2 tok/s |
| bench pp512/tg128 | depth 16K | 5050 tok/s | 82.2 tok/s |
| bench pp512/tg128 | depth 64K | 2891 tok/s | 67.9 tok/s |
| bench pp512/tg128 | depth 128K | 1963 tok/s | 58.1 tok/s |
| server ctx 128K q8_0 | baseline (no MTP) | 4580 tok/s | 85.3 tok/s |
| server ctx 128K q8_0 | MTP `--spec-type draft-mtp` | 3450 tok/s | **149.2 tok/s** |

Takeaways vs Qwen3.6-35B-A3B above:

- **pp is 4–5× faster** across the board (no CPU-offloaded experts + linear
  attention): 4580 vs 863 tok/s at server ctx 128K. Long-prompt/agent
  workloads are the clear win.
- **tg degrades gently with depth** (87 → 58 tok/s at 128K, −33%); only the
  8 full-attention layers pay the KV-scan cost.
- **MTP is a big deal: +75% tg** (85.3 → 149.2 tok/s) at the cost of ~25% pp.
  For chat/generation-heavy use, always run the `*-MTP-*.gguf` variant.
- Whole 128K-context server fits in 16 GB VRAM with no `--n-cpu-moe` juggling
  (weights 9.5 GB + KV ~2.2 GB + buffers).

## Reference results — 2026-07-12: agentic coding, qwen36-128k vs Qwythos

`./bench_05_agentic.sh` (qwen-code 0.19.9 headless, yolo, TASK_TIMEOUT 900 s).
Note: the qwen36-128k profile was bumped `--n-cpu-moe` 14 → 16 the same day —
14 no longer boots at 128K with ~1.8 GB of desktop VRAM in use.

| Task | qwen36-128k | qwythos (MTP) | ornith |
|------|-------------|---------------|--------|
| bugfix (failing pytest) | PASS 58 s | PASS **15 s** | PASS 76 s |
| scratch (CLI tool + own tests) | PASS 184 s | PASS **36 s** | PASS 147 s |
| lru (implement from test suite) | PASS 59 s | PASS **18 s** | PASS 74 s |
| multifile (3 bugs / 3 modules) | PASS 76 s | PASS **30 s** | PASS 110 s |
| template (mini engine from tests) | **PASS 118 s** | FAIL 537 s | PASS 294 s¹ |

¹ On the `ornith-128k` profile. On the 32K `ornith` profile the template task
FAILED with `400 request exceeds the available context size` — the qwen-code
session alone (code + test output + reasoning) outgrew 32K mid-task, aborting
the agent with a half-finished parser. **Agent profiles need ≥64K context.**

Ornith-1.0-35B (qwen35moe A3B, RL-trained for agentic coding, SWE-bench
Verified 75.6%): Q4_K_M 19.7 GiB, ncmoe floor 18 (tg 65.7), profiles
`ornith` ncmoe 20 (tg 60.6) / `ornith-128k` ncmoe 24 (tg ~52). It is the only
model here that solved all five tasks, but on micro-tasks it is consistently
slower than qwen36 (bigger file → more CPU offload → lower tg) and shows no
capability edge at this scale — its SWE-bench pedigree should matter on
real-repo, multi-step work, not toy tasks.

### Tiebreak round (same day): interp + perf, qwen36-128k vs ornith-128k

Two harder tasks added to bench_05 to separate the 35Bs (TASK_TIMEOUT 1200 s):
`interp` (expression evaluator: precedence, right-assoc `^`, short-circuit,
lazy ternary, functions; bare eval/exec/compile forbidden) and `perf`
(EventLog range counting, 400K adds + 100K queries under a 10 s budget).

| Task | qwen36-128k | ornith-128k |
|------|-------------|-------------|
| interp | PASS 642 s | PASS **384 s** |
| perf (verdict) | PASS 62 s | PASS 180 s |
| perf (solution runtime) | 9.17 s — **0.8 s from failing** | **1.05 s** (9× headroom) |

Verdicts stayed tied (7/7 both), but the *engineering quality* split:

- **perf**: qwen36 used `bisect.insort` per add — the approach the prompt
  warned against — and passed the 10 s budget by 0.8 s (would fail on a
  slower box or a 5 s budget). **Ornith built a Fenwick tree** (BIT over the
  timestamp domain, O(log N) add/count) with 10× margin. Ornith also solved
  interp 1.7× faster despite lower tg. The RL-agentic pedigree shows up as
  *better engineering judgment under constraints*, not more verdict-passes.
- **Verification lesson**: the first interp verdicts were false FAILs — the
  cheat-detector `grep -E '\b(eval|exec|compile)\s*\('` matched legitimate
  `re.compile()` tokenizers and a method named `eval()` on the model's own
  AST class. Both models had honestly passed 13/13. Fixed with a Python
  `ast`-walk detecting only *bare* builtin calls. Grep is not a code
  verifier.

On single-file tasks with a clear spec the two are equal in capability (both
produced identical-size, correct `OrderedDict` LRUs) and Qwythos is **3–5×
faster wall-clock** (MTP tg 149 tok/s + no MoE offload). Even the multi-file
bug hunt (mutable default + truncation rounding + checkout aliasing across
three modules) did not separate them.

**Where the 35B wins: parser/compiler-class problems.** The `template` task
(nestable `{% for %}`/`{% if %}` blocks — requires writing a real
tokenizer + recursive evaluator) qwen36-128k solved cleanly in 118 s
(207-line tokenizer/parser, 10/10). Qwythos produced a flat regex-based
substitutor that can't handle block structure (5/10) and burned 537 s.

Two failure modes compounded for Qwythos on that task:

1. **Capability**: no block-structure parsing — the classic small-model gap.
2. **Stability**: at ~69K tokens of accumulated agent context the server died
   with **CUDA "launch timed out" (Xid 8)** in
   `ggml_backend_cuda_synchronize` — a kernel (likely the Gated-DeltaNet
   linear-attention scan at deep context, MTP enabled) exceeded the ~2 s
   display-GPU watchdog. llama-server then hung inside
   `ggml_print_backtrace` (health up→NOT RESPONDING, needed `pkill -9`).
   Watch for recurrence in long qwythos agent sessions; if it repeats,
   test without `--spec-type draft-mtp` to isolate, and consider reporting
   upstream (arch `qwen35`, build 9924).

Practical split: **Qwythos for fast iteration on well-scoped tasks; qwen36
for anything parser-shaped, architectural, or requiring sustained deep
context.**

Setting sweeps at `--n-cpu-moe 10`:

- `-ub` (ubatch): 512 (default) is optimal — 256 kills pp (663 tok/s),
  1024/2048 change nothing (~1105 tok/s). Keep the default.
- KV `q8_0` vs `f16`: ~5% slower tg (102 vs 109), but halves KV VRAM —
  that headroom is what lets ctx 32K run at `--n-cpu-moe 10`.

Recommended configs:

```bash
# chat / default (32K ctx)
llama-server -m $MODEL -ngl 99 --n-cpu-moe 10 -c 32768 -fa on -ctk q8_0 -ctv q8_0 -t 16
# long context (64K → ncmoe 12, 128K → ncmoe 14)
llama-server -m $MODEL -ngl 99 --n-cpu-moe 12 -c 65536 -fa on -ctk q8_0 -ctv q8_0 -t 16
```

Each `--n-cpu-moe` step costs ~4–8% pp and tg; the ~0.44 GiB/layer expert
size means +2 ncmoe buys roughly the VRAM one context doubling needs.

### ik_llama.cpp comparison (same model)

All values pp / tg in tok/s.

| Test | mainline | ik fork |
|------|----------|---------|
| bench pp512/tg128, ncmoe 10 (floor for both; 9 OOMs) | 1107 / 109.4 | 1316 / 103.5 |
| bench pp512/tg128, ncmoe 12 | 966 / 93.2 | 1165 / 91.9 |
| bench pp512/tg128, ncmoe 14 | 898 / 89.1 | 1109 / 85.6 |
| server 32K q8_0, short prompt (1118 tok) | 1139 / 98.4 (ncmoe 10) | 1171 / 96.6 (ncmoe 11; 10 aborts) |
| server 32K q8_0, short prompt, ncmoe 12 | — | 913 / 90.1 |
| server 64K, short prompt | 1009 / 88.4 (ncmoe 12) | 1072 / 89.9 (ncmoe 12) |
| server 128K, short prompt | 863 / 81.6 (ncmoe 14) | 972 / 81.2 (ncmoe 14) |
| **server 64K, deep prompt (28 838 tok), ncmoe 12** | **1682 / 79.0** | 1389 / 76.4 |

ik notes: run via `bench_02_ikllama.sh` (`ik-llama-bench`/`ik-llama-server`,
`-fa 1` syntax). The fork's bench VRAM floor equals mainline (ncmoe 10),
but its server needs one step more at 32K (ncmoe 11) — the bigger compute
buffer, same as with GLM (which needed 18 vs mainline 10). ik bench pp
values had high variance (±86–146 tok/s) vs mainline (±5–13).

Takeaway: **the GLM-era "ik = 2× pp" advantage does not carry over to
qwen35moe** — on short prompts the fork leads pp by a mere 3–19%, and on a
real ~29K-token prompt mainline is 21% *faster* at pp with equal-or-better
tg. For this model use mainline llama.cpp at every context size. Note pp
*rises* with prompt depth (1682 vs 1139 tok/s) — batch pipelining dominates
until the KV-attention cost catches up.

## Reference results — 2026-07-05

GLM-4.7-Flash UD-Q4_K_XL (30B-A3B MoE, 17.5 GB), RTX 5070 Ti 16 GB
(driver 610.43.02), Ryzen 9 5950X, 64 GB RAM:

| Runtime | Config | pp | tg |
|---------|--------|----|----|
| ik_llama.cpp (4682) | bench, `--n-cpu-moe 10` | **2106 tok/s** | 71.2 tok/s |
| llama.cpp (b9869) | bench, `--n-cpu-moe 8` | 1445 tok/s | **91.8 tok/s** |
| llama.cpp (b9869) | bench, `--n-cpu-moe 10` | 1074–1199 tok/s | 74.6–81.2 tok/s |
| ik_llama.cpp (4682) | server, ctx 32K, `--n-cpu-moe 18` | 1257 tok/s | 48.2 tok/s |
| llama.cpp (b9869) | server, ctx 32K, `--n-cpu-moe 10` | 819–884 tok/s | 67.4–71.8 tok/s |
| ollama 0.31.1 | warm, ctx 4K, auto split ~19%/81% | 188–436 tok/s | 50–52 tok/s |

Takeaways:

- `--n-cpu-moe 6` OOMs on 16 GB; `8` is the floor for short context, `10`
  leaves room for 32K context with q8_0 KV (~15.4/16 GB VRAM). The ik fork
  needs `18` at 32K (bigger compute buffer); at `14` and below it OOMs.
- Numbers vary run-to-run with page cache state: the first benchmark after
  boot reads 17.5 GB from disk (ollama's cold pp dropped to 33 tok/s in one
  morning run); repeat measurements before drawing conclusions.
- ik_llama.cpp: ~2× mainline at prompt processing, slightly slower generation
  — prefer it for agents/RAG, mainline for pure chat generation.
- ollama offloads whole layers (attention included) to CPU instead of expert
  tensors only: 4–11× slower pp, ~30% slower tg, at 8× smaller context.
- ik_llama.cpp warning: do not use `-rtr` with experts on CPU (kills pp);
  benchmark flag changes with `ik-llama-sweep-bench` before adopting them.

## Literature — 2026-09-17: what the papers add to this log (from an aildr/Opus research run)

Source: `~/Claude/ldr/reports/2026-09-17-0000-best-llm-models-and-techniques-for-16gb-cards.md`
(Claude Opus driving local-deep-research over arXiv + SearXNG). Everything the
report presents as "the frontier" is already measured above — MoE expert
offload, MXFP4 vs Q8, dynamic quants, KV q8_0, MTP drafts — so this section
keeps only what the log did NOT have: the papers behind the knobs, read in full
where it mattered, and what each one changes for this box.

| Paper | What it establishes | Consequence here |
|---|---|---|
| SpecMoEOff, arXiv:2508.21706 | Speculative decoding pays off MOST when experts are offloaded: the GPU idles while expert weights stream over PCIe, and draft tokens raise the work per expert load. Throughput peaks at 5–6 draft tokens, then falls (CPU attention cost). EAGLE draft < 2 GB. Tested only on Mixtral-8x7B, A30 / 4090D, 190–250 GB RAM. | Our MTP drafts were only ever measured on a model that fits whole (Qwythos). The offloaded profiles (`qwen36-128k` ncmoe 16, `ornith-128k` 24, `glm-flash-128k` 24) are exactly the paper's regime and have no drafter — the MTP variants in the to-try list should be benchmarked THERE first. `bench_08_spec_offload.sh` sweeps `--spec-draft-n-max` with `NCMOE=` for that. |
| ATSInfer, arXiv:2607.10183 | Per-TENSOR placement (knapsack on measured cost per byte) beats per-layer: up to 3.29× decode and 1.94× prefill vs llama.cpp on an RTX 3060 6 GB with GPT-OSS-20B and Qwen3-30B-A3B; plus runtime re-planning when CPU speed drifts >15%. 15k lines of C++, not a flag. | The re-planning is out of reach, but the placement granularity is not: `--override-tensor` moves single expert tensors, `--n-cpu-moe` moves three per layer. `bench_09_ot_halfstep.sh` probes the 1/3 and 2/3 steps between two ncmoe values (`ffn_down_exps` alone, or `ffn_up`+`ffn_gate`), and `-f` shows what the new `--fit on` chooses against the hand-found floors. |
| Quantization meets reasoning, arXiv:2505.11574 (+2501.03035) | 4-bit weight-only is NOT safe for hard reasoning: MATH −15 pp average at W4A16, small models catastrophic (Qwen2.5-0.5B −38…−70 pp), 7B only −2…−3 pp. Errors are execution/method errors, not concept errors. W8A8 preserves far more. Repair: locate the first wrong step, 332 curated examples, 3–5 min of DPO → back near fp16. | Matches the log's own finding that Q4 breaks the gpt-oss router while Q8 does not, and the parser-tier failures of the IQ4_XS 35B MoEs. Judge quants on the hard tiers (interp/regex), never on chat. The DPO repair is a cheap experiment for `grpo_11`-style pipelines after quantising a trained model. |
| Parameter efficiency ≠ memory efficiency, arXiv:2604.22783 | In LoRA fine-tuning the peak is activation memory O(B·S·H·L), not optimizer state; checkpointing and FlashAttention shave constants only. Sequence length is the ceiling; their LARS pooling cuts ~33% at equal accuracy. | Explains why `lc_09` (gemma-12B, 4-bit, r=32) tops out on batch size at max_length 128 and why `grpo_11` is governed by `max_completion_length`. The lever that matters is S, not r. |
| Ladder side nets, arXiv:2512.14237 | Train a side network on frozen backbone activations, no backprop through the backbone: about half of QLoRA's peak memory, MATH-500 68.9 vs 70.4 for QLoRA on Qwen2.5-7B. RL/GRPO integration explicitly unsolved. | Candidate for the SFT classifiers (`lc_07`–`lc_09`) if a 12B run at longer sequences is ever needed; useless for the GRPO series. |
| WKVQuant 2402.12065, AnTKV 2506.19505 | Quantising weights AND KV cache together is where the memory goes; sub-4-bit KV needs anchor-token tricks to stay accurate. | Keeps `-ctk/-ctv q8_0` as the floor for the reasoning profiles; a `q4_0` KV trial only buys ncmoe room and must be scored on bench_05, not on tg. |
| PIPO 2504.03664, ExpertFlow 2410.17954, MoBiLE 2510.12357, HGCA 2507.03153 | Pipelined transfer/compute overlap (GPU util <40% → >90% on a 6 GB laptop card), predicted expert caching, big-little expert pairs, CPU-side attention for long context. | Research systems, not llama.cpp features; recorded so the next "why is pp slow with ncmoe 24" question has the references. |

### Measured the same day: MTP draft length on Qwythos-9B-v2 (dense, whole in VRAM)

`bench_08_spec_offload.sh`, ctx 32K, q8_0 KV, greedy, 384 generated tokens,
384-token coding prompt (server timings):

| `--spec-draft-n-max` | tg | acceptance | note |
|---|---|---|---|
| none | 85.2 tok/s | — | baseline |
| 3 (default) | 138.7 tok/s | 236/437 = 54 %, mean run 2.6 | **best** |
| 5 | 133.9 tok/s | 260/608 = 43 %, mean run 3.1 | more drafts rejected |
| 3, short prompt | 173 tok/s | 78 % | acceptance is prompt-dependent |
| 3, temp 0.7, short prompt | 150 tok/s | 62 % | sampling lowers acceptance |
| 5 + `--spec-draft-p-min 0.5` | 161 / 147 tok/s (greedy / 0.7) | | p-min trims the bad drafts |

On a dense model that fits, the paper's 5–6 is NOT the optimum: the default 3
wins, because every extra draft token here costs target compute, not idle
time. The paper's regime is the offloaded MoE — re-run with `NCMOE=` on an
MTP-capable 35B build before touching the `qwythos` profile. Nothing in
`models.conf` changed.

### Traps found while measuring (fixed in the scripts)

- **Port 8090 is the LAN ntfy server** and it answers `/health` with 200 — the
  bench scripts still defaulted to it from before ntfy existed, so a sweep
  "started" fine, posted its prompt into an ntfy topic and reported ntfy's
  reply. Defaults moved to 8095 in `bench_01/04/06/08/09`; the new scripts
  refuse a port that is already listening.
- **Backgrounding via `eval … &` kills the wrong process:** `$!` is the
  subshell, `kill $!` leaves the llama-server orphaned with 10 GB of VRAM, the
  next server cannot bind the port, and every later row measures the first
  server — five identical 85 tok/s rows. Use an args array and run the binary
  directly.

### 2026-09-17 01:10 — updated to b11009 (0.4.1-dev), host + amdkdehard + nvidiahard

`--fit on` probe (`bench_09_ot_halfstep.sh -f`, qwen36 IQ4_XS, ctx 131072, q8_0 KV,
target 512 MiB free) landed on **41 layers on CUDA0 with 16 overflowing to the CPU,
plus a fractional step: the `ffn_up` of one more layer** (`overflow_type=UP`, then
GATE tried and rejected) — 13 465 MiB used, 649 MiB free. That is the hand-found
`ncmoe 16` of the `qwen36-128k` profile, found automatically in 3 s, and the new
build already does the per-tensor fraction that bench_09's HALF sweep was written
to probe. Keep bench_09 -b for comparing tg between the two, but the floor search
itself is now `--fit`. MTP on Qwythos unchanged after the update (134 tok/s at
n-max 3, 51 % acceptance, 84 baseline).

New warning worth a benchmark: *"tensor overrides to CPU are used with mmap
enabled — consider using `--load-mode none` for better performance"*. Every ncmoe
profile triggers it. Try `EXTRA_SERVER_ARGS="--load-mode none" bench_01_llamacpp.sh -s`
on qwen36-128k and ornith-128k before adding it to models.conf.

### 2026-09-17 04:20–04:40 — b11009 follow-ups: `--load-mode none` doubles pp on every offload profile; `--fit` vs the hand floors; MTP on an offloaded MoE is a loss

**1. `--load-mode none` (no mmap) — adopted in every `--n-cpu-moe` profile.**
`bench_01_llamacpp.sh -s`, 1.1K-token prompt, 256 generated, ctx 131072, q8_0 KV,
two passes for qwen36 (first / second):

| profile | ncmoe | pp mmap (default) | pp `--load-mode none` | tg mmap | tg none |
|---|---|---|---|---|---|
| `qwen36-128k` | 16 | 707 / 770 | **1386 / 1293** | 73.4 / 78.6 | 77.0 / 78.6 |
| `ornith-128k` | 24 | 467 | **769** | 54.2 | 53.3 |
| `glm-flash-128k` | 24 | 353 | **1090** | 42.5 | 41.7 |

llama-bench, qwen36 ncmoe 16: pp512 802 ± 28 → **1818 ± 46** (2.27×), tg128 82.0 → 81.3.
The build's own warning ("tensor overrides to CPU are used with mmap enabled —
consider `--load-mode none`") was right: the CPU-resident experts are read through
file-backed page-cache mappings under mmap; with `none` they are copied into
anonymous memory once at load. Decode is single-token and bandwidth-bound either
way, prompt batches hammer the expert matmuls and pay the mapping overhead. Cost:
+5–10 s load (the copy; more when the page cache is cold), and the CPU part of the
model lives twice in RAM (anonymous + page cache) — irrelevant at 64 GB. Applied to
all 12 ncmoe profiles in `~/.aillama/models.conf` and the built-ins in `bin/aillama`.
Whole-in-VRAM profiles (gpt-oss, 9B dense) are untouched: nothing of theirs is on
the CPU.

**2. `--fit on` vs the hand floors** (`bench_09_ot_halfstep.sh -f`, ctx 131072,
`--fit-target 512`, ~0.5 GB desktop VRAM):

| profile | hand floor / profile value | `--fit` chose | free after fit |
|---|---|---|---|
| `qwen36-128k` | 16 / 16 | 41 layers, 16 overflowing + `UP` of one more | 649 MiB |
| `qwen36u-mxfp4-128k` | 16 @32K / 23 | 18 overflowing + `UP` | 611 MiB |
| `ornith-128k` | 18 / 24 | 17 overflowing + `GATE` | 516 MiB |
| `glm-flash-128k` | 22 (20 aborted) / 24 | 19 overflowing + `ATTN` | 596 MiB |
| `gpt-oss20b-f16` | 2 (1 OOMed) / 2 | nothing offloaded, "no changes needed" | 1036 MiB |

The hand floors were found with 0.9–1.8 GB of desktop VRAM in use; `--fit` sees
~0.5 GB today, hence the lower numbers — it is a measurement of the moment, not a
new floor (its own log says so: re-run with the real desktop load). Its fractional
step is cheap: bench_09 -b on qwen36 ncmoe 16 — none pp 802 / tg 82.0; + `ffn_up`
of layer 16: 815 / 81.0; + `ffn_down`: 813 / 79.9 — about 1.3 % tg for a third of
a layer, the same per-tensor rate as a whole layer.

Real 49K-token request at the `--fit` placement (qwen36, 16 + UP, `--load-mode
none`, ctx 131072): prompt 49 216 tokens at **1833 / 1853 tok/s**, tg **75.1 tok/s
at 49K depth**, VRAM 15.25 → 15.36 GiB of 16.3 across two requests, no OOM. That
is the number to quote for "how fast is a real long prompt on this box".

**3. MTP on an offloaded MoE — the SpecMoEOff regime, measured: a loss.**
Nemotron-3.5-Lightning-30B-A3B Q4_0 + ggml-org's separate `mtp-*.gguf` (19
tensors: `token_embd`, `output`, block 52 = the MTP head; loads with
`-md mtp.gguf --spec-type draft-mtp`, server log "loading draft model"), ncmoe 18,
ctx 32K, `--load-mode none`, greedy, 384 tokens, `bench_08_spec_offload.sh`:

| `--spec-draft-n-max` | tg | acceptance |
|---|---|---|
| none | 71.4 / 71.3 tok/s | — |
| 3 | 66.5 / 66.0 | 52 %, mean run 2.55 |
| 5 | 51.6 / 52.2 | 33 %, mean run 2.65 |

The paper's gain assumes experts *streamed* over PCIe per token, so the GPU idles
and verifying k+1 tokens costs nothing extra. Here the offloaded experts are
resident in RAM and computed on 16 CPU threads: verifying a 4-token batch through
CPU experts costs close to 4× a single token (compute-bound, not bandwidth-bound),
which eats the whole draft win. The 2026-08-18 "unused reserve" of the nemotron
`mtp-*.gguf` is closed — do **not** add `--spec-type draft-mtp` to an ncmoe
profile. MTP stays a whole-in-VRAM win (`qwythos`, n-max 3).

n-gram drafter (`--spec-type ngram-mod`, model-free, qwen36 ncmoe 16): first
request has nothing to draft from (78.5 tok/s = baseline), the *identical* second
request runs at 99.3 tok/s (56 % accepted, mean run 35.8 tokens) because the
drafter replays its own history. It pays only when the output repeats earlier
context (re-emitting a file after an edit, long tool-output echo) — a use-case
note, not a ranking number. `bench_08` now treats `ngram-*` types as one row
(`DRAFT_LIST=0,1`), since `--spec-draft-n-max` is not their knob.

**4. Lower floors from `--fit`, checked with the 49K request, and bench_05 on the
Gated-DeltaNet models after the #28068 normalisation fix** — see the next entry.

## HF sweep — 2026-09-17: Ornith 1.5 family is the candidate that matters; 27B-dense and >100B releases are out by rule

Done while bench_05 ran (HF API: trending + most-downloaded GGUF repos created
since 2026-07-28, plus targeted searches; sizes from `?blobs=true`). The 16 GB /
64 GB rules from the fleet table apply up front: MoE A3B class ≤ ~21 GiB at Q4
(offload), or a whole-fit file ≤ ~13 GiB at 128K; no dense 27B; nothing whose
Q4 does not fit in RAM.

**Worth benching (in this order):**

1. **`ornith-ai/Ornith-1.5-35B-A3B-GGUF`** (2026-08-18, 4.4 M downloads,
   `qwen35moe`, native ctx 262 144) — the successor of the serious-agentic
   default. Q4_K_M 20.22 GiB = the same shape as Ornith-1.0 Q4_K_M (19.7 GiB,
   floor 18 @128K, profile 20 since today). Card: self-improvement loop expanded
   from Ornith-1.0; Terminal-Bench 2.1, SWE-Bench Verified/Pro/Multilingual,
   MCP-Atlas, Toolathlon numbers on the card; `qwen3_coder` tool-call format
   (works with qwen-code). The direct A/B against `ornith-128k` on bench_05 +
   bench_07 is the single most valuable run in this list.
2. **`peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF`** (2026-08-19, 415 K, base =
   Ornith-1.5-35B-A3B, `qwen35moe`) — a coder finetune + unsloth-style dynamic
   requant; UD-IQ4_XS 16.51 GiB (lower floor than Q4_K_M — but remember the
   qwen36 UD-IQ4_XS lesson: it was the uniquely failing quant there), UD-Q4_K_XL
   20.82 GiB. Card claims SWE-bench-Live 12/25 vs Ornith-1.5's 8/25 and
   Claw-Eval 67.2 vs 65.3, self-reported. `Cyber-Tiel-Coder` (09-08) is its
   uncensored sibling. Ships a 1.39 GiB `mtp-*.gguf` — irrelevant here (see
   "b11009 follow-ups": MTP loses on an offloaded MoE).
3. **`ornith-ai/Ornith-1.5-9B-GGUF`** (2026-08-19, 5.3 M, `qwen35` dense) —
   successor of `ornith-9b`; Q8_0 9.11 GiB fits whole at 128K like Qwythos.
   Fast-tier challenger to `qwythos` / `gpt-oss20b-udq8kxl`.
4. **`empero-ai/Qwen3.8-9B-Distill-GGUF`** (2026-08-15, 699 K, `qwen35` dense,
   Q8_0 9.11 GiB) — Qwen3.8-27B distilled into the 9B shape; same class as 3,
   bench whichever of the two wins first against the other.
5. `DevQuasar/amd.Instella-MoE-16B-A3B-Think-GGUF` (`instella-moe`, native ctx
   32 768 only, Q6_K 13.2 GiB whole) and the July `tvall43/Qwen3.6-14B-A3B-
   FableVibes-GGUF` (Q8_0 13.65 GiB whole) — the "MoE that fits whole" shelf;
   curiosity tier, after 1–4.

**Out by rule (no download):** the whole Qwen3.8-27B GGUF wave (unsloth 8.9 M,
HauhauCS-MTP, DavidAU TURBO/TWIN-TURBO, ISTA-DASLab GSQ, Bucoid "16GB-VRAM" IQ4_XS-
MTP, z-lab DFlash2, Jackrong Qwopus3.8, cdiamond NVFP4-MTP …) — dense 27B, the
2026-08-18 `qwen38` result (4.5 t/s, 0/8) already settled the class; a drafter
does not fix partial-offload decode of a dense model, and the "16 GB VRAM" quants
leave no room for a 128K KV. `unsloth/Qwen3.8-Flash-Next` (`qwen4exp`, 177 B
total), `unsloth/GLM-5.3-Flash` (`glm5next`, 321 B), `DeepSeek-V4.1-Flash` (755 B,
Q2 = 341 GiB), `Ling-3.0-flash` (124 B), `Muse-Glimmer-30B` (dense 28 B + vision),
`MiniMax-H3` (odd pruned 20 B files), `Nemotron-3-Nano-30B-A3B` (older sibling of
the tested 3.5 Lightning). Still nothing new for the eagle3 draft of gpt-oss-20b
(RedHatAI repo untouched since 04-08, safetensors only).

**July list, re-checked:** all eight repos still exist. `unsloth/Ornith-1.0-35B-
GGUF` UD-IQ4_XS is superseded by Ornith 1.5; the three MTP variants
(Ornith-MTP-APEX, qwen36-MTP, dsv4flash-MTP) are dropped after today's measurement
(35B MoE = offloaded = MTP loses; dsv4flash is chat-only anyway).
`Ternary-Bonsai-27B` stays a curiosity (6.7 GiB ternary, updated 08-31).

**Blocker before any download: `/mnt/db1` has 30 GB free (95 %).** Ornith-1.5-35B
Q4_K_M + Ornith-1.5-9B Q8_0 need 29.3 GiB. Rejected files still on disk under
`~/models` (= `/mnt/db1/huggingface/models`): `gemma4-12B-fable5` 12 G,
`gemma4-12b-q8_0` 12 G, `gemma4-12b-qat` (hub blob 7.4 G), `gpt-oss-20b-heretic`
(hub blob 12.6 G), `gpt-oss-20b-neoplus` 12 G (never profiled), `qwen35-9b-
dsv4flash` 6.9 G (chat-only) — ~56 GB if all six go. Nothing was deleted; the
choice is yours.

### 2026-09-17 04:38–07:00 — lower floors verified, profiles changed; bench_05 on the Gated-DeltaNet models after b11009; Ornith 1.5 first boot

**Floors from `--fit`, checked with a real 49K-token request** (ctx 131072,
`--load-mode none`, `n_predict` 256, greedy; VRAM after the request):

| profile | ncmoe | VRAM after request | pp @49K | tg @49K depth | verdict |
|---|---|---|---|---|---|
| ornith-128k | 20 | 13.62 GiB | 1478 | 54.4 | **new profile value** (was 24) |
| ornith-128k | 18 | 14.55 GiB | 1602 | 57.1 | = hand floor, 1.75 GB left |
| glm-flash-128k | 22 | 14.51 GiB | 956 (41K tok) | 32.8 | **new profile value** (was 24) |
| glm-flash-128k | 20 | 15.15 GiB | 982 | 33.9 | boots and serves now (aborted in July) — 1.1 GB left, too tight |
| ornith15-128k (Ornith-1.5-35B Q4_K_M) | 20 | 13.84 GiB | 1416 | 52.7 | same shape as 1.0 — profile at 20 |
| ornith15-128k | 18 | 14.77 GiB | 1530 | 56.7 | floor confirmed |

floor + 2 everywhere, with ≥ 1.8 GB for the desktop. The July "20 aborts at boot"
for GLM was a build + desktop-VRAM artefact (the skill already says floors drift
per build). tg at 49K depth is within 5 % of the 1K-prompt figures.

**bench_05 after the GDN normalisation fix (#28068), `TASK_TIMEOUT=900`,
`--load-mode none` profiles, ornith at ncmoe 20** — one run each, so per the
house rule this is a regression check, not a ranking:

| task | qwythos | qwen36-128k | ornith-128k |
|---|---|---|---|
| bugfix | PASS 4/4 13 s | PASS 4/4 34 s | PASS 4/4 44 s |
| scratch | PASS 33 s | PASS 15/15 145 s | PASS 7/7 79 s |
| lru | PASS 18 s | PASS 40 s | PASS 43 s |
| multifile | PASS 33 s | PASS 50 s | PASS 70 s |
| intervals | PASS 66 s | PASS 34 s | PASS 86 s |
| fsm | PASS 20 s | PASS 43 s | PASS 47 s |
| codec | FAIL 16 s — file not created | PASS 37 s | PASS 60 s |
| toposort | PASS 16 s | PASS 40 s | PASS 60 s |
| template | FAIL 0/10 481 s | TIMEOUT 5/10 | PASS 10/10 238 s |
| interp | FAIL 0/13 634 s | TIMEOUT 10/13 (76 %) | TIMEOUT 11/13 (84 %) |
| perf | PASS 37 s | PASS 39 s | PASS 101 s |
| regex | FAIL — used re | TIMEOUT 10/14 (71 %) | TIMEOUT 10/14 (71 %) |
| **PASS** | **8/12** | **8/12** | **10/12** |

Reading: no regression anywhere on the easy/mid tier (24/24 clean across the
three). `ornith-128k` at ncmoe 20 with mmap off behaves exactly as its July
profile (template PASS, interp scored-but-late, regex never converges — the
documented signature), so the profile change stands. `qwythos` keeps its
parser-tier wall (all four FAIL, two of them instant — the 9B does not attempt
the file). `qwen36-128k` now TIMEOUTs template at 5/10 where it once was "the
fastest template (118 s)"; that is one run against N≈3 historical runs — flag
for a repeat, not a verdict. Nothing here says the GDN fix changed a verdict in
either direction; it did not break anything, which was the question.

**Ornith 1.5 (downloaded 05:23–05:33 after space was freed):** `ornith15-128k`
(35B Q4_K_M 20.22 GiB, ncmoe 20) and `ornith15-9b` (Q8_0 9.11 GiB, whole at
128K) are in models.conf; bench_05 on both is running — next entry.

### 2026-09-17 07:00–08:40 — Ornith 1.5: both sizes 11/12 on the first run; the 9B is the news

`bench_05_agentic.sh`, `TASK_TIMEOUT=900`, `WORKROOT=/tmp/bench-agentic-ornith15`,
llama.cpp b11009, one run each (regression-check rules apply: N=1 does not rank).

| task | `ornith15-9b` (Q8_0 9.11 GiB, whole @128K) | `ornith15-128k` (Q4_K_M 20.22 GiB, ncmoe 20, mmap off) |
|---|---|---|
| bugfix | PASS 4/4 22 s | PASS 4/4 44 s |
| scratch | PASS 6/6 84 s | PASS 6/6 125 s |
| lru | PASS 65 s | PASS 84 s |
| multifile | PASS 123 s | PASS 100 s |
| intervals | PASS 65 s | PASS 112 s |
| fsm | PASS 69 s | PASS 70 s |
| codec | PASS 58 s | PASS 82 s |
| toposort | PASS 60 s | PASS 47 s |
| template | PASS 10/10 488 s | PASS 10/10 319 s |
| interp | **PASS 13/13 312 s** | **PASS 13/13 275 s** |
| perf | PASS 616 s | PASS 375 s |
| regex | TIMEOUT 2/14 | TIMEOUT, no file |
| **PASS** | **11/12** | **11/12** |

Same day, same harness, same budget: `ornith-128k` (1.0) 10/12, `qwen36-128k` 8/12,
`qwythos` 8/12 — and no 9B in this log had ever passed `interp` (Qwythos 0/13,
dsv4flash 0/13, ornith-9b 12/13 near-miss at 900 s). `regex` stays the family's
wall (Ornith 1.0 never converged either; the 9B at least left 2/14).

Throughput, `ornith15-9b` alone on the GPU, ctx 131072, q8_0 KV, greedy, 256 out:
12.0 GiB after load (fits whole with 4 GB spare), **pp 4235 / tg 84.1 tok/s** on an
836-token prompt, **pp 4491 / tg 72.7 at 49K depth**. `ornith15-128k`: 13.84 GiB
at ncmoe 20, pp 1416 / tg 52.7 at 49K depth (see the floor table above) — the 35B
buys nothing over the 9B on this suite at 0.63× the speed.

**What changes:** nothing in the DEFAULT column yet (house rule, N≥20 before a
ranking claim), but the queue order is set: (1) `ornith15-9b` ×3 on the parser
tier + bench_07 `RUNS=10` against `gpt-oss20b-udq8kxl` — if it holds ~11/12 it
takes the fast tier and Qwythos retires; (2) `ornith15-128k` ×3 vs `ornith-128k`
— if it holds, Ornith 1.0 retires and 19.7 GB come back. Tiel-Coder (the coder
finetune of this base) only matters if (2) holds. The Ornith 1.5 profiles are in
`models.conf` with today's numbers in their comments.

## Reference results — 2026-09-19: the Ornith 1.5 queue — parser tier ×3 paired with Ornith 1.0, then bench_07 RUNS=20 vs gpt-oss20b-udq8kxl

```
# /var/tmp/bench-o15-queue.sh, log /var/tmp/bench-o15-queue-2026-09-18.log, 21:25 → 03:59
for i in 1 2 3; do MODELS=ornith15-9b,ornith15-128k,ornith-128k TASKS=template,interp,perf,regex \
  TASK_TIMEOUT=900 WORKROOT=/tmp/bench-o15-parser-run$i ./bench_05_agentic.sh; done
MODELS=ornith15-9b,gpt-oss20b-udq8kxl RUNS=20 WORKROOT=/tmp/bench-wf-o15 ./bench_07_workflow.sh
```

The queue set on 09-17 (N=1 does not rank). Only the parser tier repeats — the
easy/mid tier was 8/8 for all three the day before and discriminates nothing at
the top of the fleet. `ornith-128k` (Ornith 1.0) runs in the same pass as a paired
control instead of leaning on July numbers. bench_07 at `RUNS=20`, not the 10
written in the queue: N=10 already produced one false verdict in July. llama.cpp
b11009 throughout, ~1 h 50 per parser pass, bench_07 47 min for the 9B and 10 min
for udq8kxl.

**Parser tier, four runs per model (09-17 first run + tonight's three):** verdict,
`score` where the verdict is not a clean PASS, wall-clock seconds.

| task | `ornith15-9b` | `ornith15-128k` | `ornith-128k` (1.0) |
|---|---|---|---|
| template | PASS 488 · TIMEOUT – · TIMEOUT 6/10 · PASS 888 → **2/4** | PASS 319 · 216 · 162 · 326 → **4/4** | PASS 238 · 119 · 271 · 421 → **4/4** |
| interp | PASS 312 · PASS 341 · TIMEOUT 13/13 · PASS 628 → **3/4** (13/13 scored in all four) | PASS 275 · 467 · 376 · 404 → **4/4** | TIMEOUT 11/13 · TIMEOUT 12/13 · PASS 256 · PASS 358 → **2/4** |
| perf | PASS 616 · PASS 541 · TIMEOUT 5/6 · PASS 358 → **3/4** | PASS 375 · 132 · 322 · 178 → **4/4** | PASS 101 · 146 · 89 · 108 → **4/4** |
| regex | TIMEOUT 2/14 · – · – · – → **0/4** | TIMEOUT no file · – · 11/14 · 14/14 → **0/4** | TIMEOUT 10/14 · 5/14 · 12/14 · – → **0/4** |
| parser verdicts | 3, 2, 0, 3 of 4 | **3, 3, 3, 3 of 4** | 2, 2, 3, 3 of 4 |

**bench_07 `relmeta`, RUNS=20:**

| model | PASS | lastline | evidence | license | summary | order | other 5 items | s/run |
|---|---|---|---|---|---|---|---|---|
| `ornith15-9b` | **19/20 (95 %)** | 20/20 | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 | 92–222 |
| `gpt-oss20b-udq8kxl` | 11/20 (55 %) | 12/20 | 12/20 | 18/20 | 19/20 | 19/20 | 20/20 | 22–43 |

udq8kxl reproduces its July RUNS=20 result to the run (11/20 then, 11/20 now), with
the same `lastline,evidence` pair as the dominant FAIL signature — the base-model
trait is stable across two llama.cpp builds and two months. The 9B's single miss
was an `evidence` slip with everything else held.

**Reading.**

1. **`ornith15-128k` takes the serious-agentic role from Ornith 1.0.** Same file
   shape (20.22 vs 19.7 GiB), same floor (18) and speed (~53 tok/s at 49K depth),
   and 11/12 held on all four runs; the one task where the two differ is `interp`
   (4/4 vs 2/4 at 900 s — Ornith 1.0 scores 11–12/13 and runs out of time). regex is
   the family wall for both, but the 1.5 reached 14/14 scored inside the budget once
   (still TIMEOUT: it kept verifying instead of stopping). The pre-set rule held →
   Ornith 1.0 35B retires (20 GB hub blob under `models--deepreinforce-ai--Ornith-1.0-35B-GGUF`).
2. **`ornith15-9b` did NOT hold "~11/12" on verdicts** — parser verdicts 3, 2, 0, 3
   of 4 — so by the queue's own rule it does not take the fast tier from udq8kxl.
   The shape of its misses matters though: `interp` scored 13/13 in all four runs
   (one of them past the wall), `perf` 5/6 and `template` 6/10 at the cutoff. This
   is a model that gets there and is slow about it (template PASS at 888 s), not one
   that lacks the capability; at `TASK_TIMEOUT=1200` most of those TIMEOUTs would
   flip. It never touches regex (0/4, no file in 3 of 4) — udq8kxl is still the only
   fleet member that passes regex, and at 193 vs 84 tok/s.
3. **bench_07 is the news: 19/20 vs 11/20.** The `lastline`/`evidence` weakness that
   bench_07 exists to detect — the gpt-oss-20b trait behind the July nanoeuler
   ebuild sessions — is simply absent in Ornith 1.5 9B: every tail-read and
   hallucination item 20/20. For rule-heavy multi-step qwen-code jobs (ebuild
   authoring, long QWEN.md/RULES files) that is the axis that costs real sessions,
   and the harness mitigations (checklist at the top of the file) were written for
   the model that no longer needs to be the default there.
4. **Roles after tonight:** `gpt-oss20b-udq8kxl` = fast coding tier (scoped
   sprints, regex, raw speed); `ornith15-9b` = workflow agent (whole-fit 12 GiB,
   84 tok/s, follows rules); `ornith15-128k` = serious agentic. `qwythos` is
   dominated on every axis by `ornith15-9b` (same size class, parser tier, 95 %
   workflow, no Xid 8 hang) → retired; 18 GB (MTP + plain files) delete candidate.
   Tiel-Coder (coder finetune of this base) now has a reason to be benched.
5. Operational side effect worth a line: the 35B passes leave ~1.7 GB of VRAM to
   the desktop, and a Netflix tab in Chrome went black-with-audio during the run
   (Chromium `SharedImageManager::ProduceSkia … non-existent mailbox` at 21:31 —
   a video frame whose GPU texture never got allocated). Retest with the GPU free
   pending; if confirmed, queues want a lean desktop or a floor+4 profile.

## Reference results — 2026-09-19 (day): queue 2 — `ornith15-9b` at 1200 s, Tiel-Coder paired with Ornith 1.5, Qwen3.8-9B-Distill

```
# /var/tmp/bench-o15b-queue.sh, log /var/tmp/bench-o15b-queue-2026-09-19.log, 10:07 → 17:09
MODELS=ornith15-9b TASKS=template,interp,perf,regex TASK_TIMEOUT=1200 ×3
MODELS=tiel-128k,ornith15-128k TASKS=… TASK_TIMEOUT=900 ×3      (after a 51K-token preflight)
MODELS=qwen38d-9b TASKS=… TASK_TIMEOUT=900 ×3; then RUNS=20 ./bench_07_workflow.sh
```

New this queue: every model that had never booted here got a **preflight** — `aillama
switch`, one ~51K-token chat request at `-c 131072`, VRAM read after it, and an
automatic `--n-cpu-moe +2` retry if the server died on first decode. Neither newcomer
needed the retry: `tiel-128k` at ncmoe 21 → 14.4 GiB, pp 1885 / tg 50.3 at 51K depth;
`qwen38d-9b` whole → 12.3 GiB, pp 4429 / tg 71.2.

**1. `ornith15-9b`, parser tier at `TASK_TIMEOUT=1200`, three runs:**

| task | run 1 | run 2 | run 3 |
|---|---|---|---|
| template | PASS 770 s | PASS 370 s | PASS 394 s |
| interp | PASS 297 s | PASS 603 s | PASS 176 s |
| perf | PASS 158 s | PASS 1082 s | PASS 196 s |
| regex | TIMEOUT 12/14 | TIMEOUT no file | TIMEOUT 0/14 |

11/12-equivalent on all three runs. Yesterday's verdicts at 900 s (3, 2, 0, 3 of 4)
were the wall, not the model: nothing changed but the budget, and 9/9 non-regex tasks
pass. The cost is real though — two of the nine needed 770 and 1082 s, i.e. this 9B
spends up to 18 minutes on a task udq8kxl finishes in two. Reading for the roles: the
9B is a full coding agent minus regex if you can afford 1200 s per task; for sprints
udq8kxl stays.

**2. Tiel-Coder-35B-A3B (`tiel-128k`, UD-Q4_K_XL) paired with `ornith15-128k`, 900 s:**

| task | tiel run 1 · 2 · 3 | ornith15-128k run 1 · 2 · 3 |
|---|---|---|
| template | PASS 162 · 199 · 154 | PASS 192 · 176 · 170 |
| interp | PASS 325 · 313 · 293 | PASS 276 · 255 · 218 |
| perf | PASS 222 · 228 · 143 | PASS 412 · 182 · 214 |
| regex | TIMEOUT no file · **PASS 14/14 340 s** · TIMEOUT no file | **PASS 14/14 744 s** · TIMEOUT 5/14 · **PASS 14/14 670 s** |
| parser verdicts | 3, 4, 3 of 4 | 4, 3, 4 of 4 |

Two things here, and neither is "Tiel is better":

- **The regex wall is not a wall.** Ornith 1.5 35B passed regex twice today after 0/4
  yesterday (where it had already scored 14/14 once but kept verifying past the cutoff).
  Pooled N=7: 2 PASS, 4 more at ≥11/14 scored. Tiel 1/3, with the fastest regex pass in
  this log (340 s). So the family's true regex behaviour at 900 s is roughly a one-in-three
  pass with the solution usually complete-but-unconfirmed at the cutoff — not the
  "never converges" of Ornith 1.0. **12/12 is now on the table for the 35B**, and udq8kxl
  is no longer the only fleet member that passes regex.
- **Tiel = Ornith 1.5 at N=3.** Identical on template/interp/perf (Tiel a shade faster
  on perf), regex 1/3 vs 2/3 — inside noise. The card's SWE-bench-Live claim does not
  show up on this suite; 0.6 GiB more weights, one ncmoe step more, 5 % less tg. No reason
  to switch a default for a tie. Kept as an alternate for now; 21 GB delete candidate.

**3. Qwen3.8-9B-Distill (`qwen38d-9b`) — rejected on both benches:**

| task | run 1 | run 2 | run 3 |
|---|---|---|---|
| template | FAIL 2/10 | FAIL 4/10 | FAIL 1/10 |
| interp | TIMEOUT 9/13 | TIMEOUT 0/13 | TIMEOUT 0/13 |
| perf | PASS 46 s | PASS 515 s | PASS 136 s |
| regex | TIMEOUT 0/14 | FAIL 0/14 | FAIL 0/14 |

bench_07 RUNS=20: **1/20 PASS.** Matrix: `summary` 5/20 (the hallucination trap — it
writes about Euler/numerics because the project is called nanoeuler, 15 times in 20),
`evidence` 10/20, `lastline` 12/20, `version` 18/20; the five mechanical items 20/20.
Fast (27–52 s per run, perf in 46 s) and wrong. Same file size and shape as
`ornith15-9b`, which went 19/20 on the same rubric the night before — the distillation
target, not the 9B shape, is what decides this. Delete candidate (9.1 GB).

**After queue 2:** roles unchanged — `ornith15-128k` serious agentic (now with a real
regex chance), `ornith15-9b` workflow agent (and full coding agent at 1200 s),
`gpt-oss20b-udq8kxl` sprint tier. Queue 3 (Qwen3.8-35B-A3B-Distill, full suite ×3 +
bench_07) follows in the evening — the interesting question there is whether the
35B-A3B distill shares the 9B distill's hallucination trait.

## Reference results — 2026-09-19 (evening): queue 3 — Qwen3.8-35B-A3B-Distill, full suite ×3 + bench_07 RUNS=20

```
# /var/tmp/bench-q38d35-queue.sh, log /var/tmp/bench-q38d35-queue-2026-09-19.log, 17:10 → 20:07
MODELS=qwen38d-128k TASK_TIMEOUT=900 ./bench_05_agentic.sh   ×3 (all 12 tasks)
MODELS=qwen38d-128k RUNS=20 ./bench_07_workflow.sh
```

The sweep pick of the day: `empero-ai/Qwen3.8-35B-A3B-Distill` (published 09-16),
Qwen3.8-27B distilled into the Qwen3.6-35B-A3B shape — the same `qwen35moe` shape as
`qwen36-128k` (its base) and `ornith15-128k`, and byte-for-byte the Ornith 1.5 Q4_K_M
file size, so it inherited the ncmoe-20 recipe untouched. Preflight: 14.1 GiB after a
51K-token request, pp 1865 / tg 52.9 at that depth — identical to Ornith 1.5.

| task | run 1 | run 2 | run 3 |
|---|---|---|---|
| bugfix · scratch · lru · multifile | PASS 38 · 140 · 92 · 66 | PASS 40 · 194 · 40 · 68 | PASS 36 · 73 · 96 · 139 |
| intervals · fsm · codec · toposort | PASS 113 · 40 · 44 · 39 | PASS 44 · 39 · 41 · 64 | PASS 243 · 54 · 42 · 41 |
| template | PASS 733 s | TIMEOUT 8/10 | PASS 302 s |
| interp | PASS 408 s | TIMEOUT 2/13 | PASS 234 s |
| perf | PASS 54 s | PASS 159 s | PASS 80 s |
| regex | TIMEOUT 11/14 | TIMEOUT 4/14 | TIMEOUT 0/14 |
| **PASS** | **11/12** | **9/12** | **11/12** |

bench_07 RUNS=20: **15/20 PASS (75 %)** — `lastline` 16/20, `evidence` 16/20, `summary`
19/20, the seven others 20/20; FAIL signature `lastline,evidence` (4×) + one `summary`.
~75–125 s per run.

Reading:

- **The 9B distill's hallucination trait did not carry over.** `summary` 19/20 and
  version/license 20/20 here vs the 9B's summary 5/20 the same afternoon. So the trait
  was not "empero distillation" per se — the 35B-A3B target absorbed the 27B teacher, the
  9B target did not.
- **Better than its base, not better than Ornith 1.5.** Against `qwen36-128k` (8/12 on
  09-17, template/interp TIMEOUTs; "tied Ornith 7/8" in July) the distill is clearly
  stronger on the parser tier and the easy/mid tier is a clean, fast 24/24. Against
  `ornith15-128k` at the identical speed: interp 2/3 vs 7/7, regex 0/3 vs 2/7, workflow
  15/20 vs (9B) 19/20 — second place in the same shape. Between udq8kxl (11/20) and
  Ornith on the workflow axis.
- The price of the extra quality over `qwen36-128k` is speed: Q4_K_M at ncmoe 20 is
  tg ~53 against the IQ4_XS base's ~82. `qwen36-128k` stays the fast general profile;
  `qwen38d-128k` is the stronger general alternate. Neither is a default.

**Fleet after the three queues (09-18 → 09-19):** `gpt-oss20b-udq8kxl` sprint tier ·
`ornith15-9b` workflow agent (full coding agent at 1200 s) · `ornith15-128k` serious
agentic (12/12-capable) · `qwen38d-128k` general alternate · `qwen36-128k` fast general ·
`tiel-128k` tie-with-Ornith alternate. Rejected today: `qwen38d-9b`. Deleted today:
Ornith 1.0 35B + 9B, Qwythos. Delete candidates on the table: Tiel (21 GB), qwen38d-9b
(9.1 GB). Open experiment class that none of this touches: bench_08 arm C.

## Reference results — 2026-09-20: bench_07 for `ornith15-128k` — the hole the three queues left, closed at 20/20

```
# /var/tmp/bench-wf-o15-35b.sh, log /var/tmp/bench-wf-o15-35b.log, 09:52 → 11:14
MODELS=ornith15-128k RUNS=20 WORKROOT=/tmp/bench-wf-o15-35b ./bench_07_workflow.sh
```

Queue 1 sent only `ornith15-9b` into phase B, so the model holding the
serious-agentic role had no workflow-discipline number at all while the 9B below it
did — the one axis bench_07 exists to measure was missing from the model the fleet
table recommends. Filled today, same build (b11009), same `relmeta` task, standard
layout.

**20/20 PASS (100 %).** RULE-COMPLIANCE MATRIX: `deliverable`, `name`, `version`,
`license`, `summary`, `order`, `lastline`, `protected`, `no-strays`, `evidence` —
all **20/20**. Run time 190–313 s, mean ~240 s, no timeouts, no near-misses.

| model | bench_07 | tail-read (`lastline`) | evidence-gate | hallucination (`summary`) |
|---|---|---|---|---|
| **`ornith15-128k`** | **20/20** | **20/20** | **20/20** | **20/20** |
| `ornith15-9b` | 19/20 | 20/20 | 19/20 | 20/20 |
| `qwen38d-128k` | 15/20 | 16/20 | 16/20 | 19/20 |
| `gpt-oss20b-udq8kxl` | 11/20 | 12/20 | 12/20 | 19/20 |
| `qwen38d-9b` | 1/20 | 12/20 | 10/20 | 5/20 |

Reading:

- **First perfect bench_07 in this log.** The previous best was the 9B's 19/20 the
  day before; before Ornith 1.5 nothing had ever cleared 55 % (udq8kxl 11/20, pooled
  49 % over 35 July runs). A clean 20/20 also means zero stochastic compliance —
  the drift this bench was built to expose does not appear in this model at N=20.
- **The open question from the 09-19 write-up is answered the other way.** The
  hypothesis was that rule-discipline might be a trait the distillation put into the
  small model, since the 9B out-scored udq8kxl 19/20 to 11/20. It is a family trait,
  and the 35B has more of it: the parent is perfect where the 9B slipped once on
  `evidence`.
- **No trade-off left in the role assignment.** `ornith15-128k` is now first on the
  coding axis (only 12/12-capable model, only one that passes `regex`) *and* first on
  the workflow axis. `ornith15-9b` keeps its role on speed and footprint — whole-fit,
  tg 84 vs 53, ~240 s per relmeta run for the 35B — not on quality.

## Reference results — 2026-09-20: bench_10 — the aildr workload head-to-head, `ornith15-128k` vs `ornith15-9b`: a tie on every axis that matters, 27/30 each

```
# /var/tmp/bench10-queue.sh, log /var/tmp/bench10-queue.log, 11:30 → 14:45
for pass in 1 2 3; do MODELS=ornith15-128k,ornith15-9b ITERATIONS=2 QPI=3 TIMEOUT=1500 \
  SLEEP_BETWEEN=60 WORKROOT=/var/tmp/bench10-20260920-1130-p$pass ./bench_10_research.sh; done
./bench_10_research.sh --rescore /var/tmp/bench10-20260920-1130-p{1,2,3}
```

**Why a new bench.** bench_05 scores code, bench_07 scores rule-following; the aildr
loop does neither — it reads dozens of search snippets in a long context, decides
what to search next and writes a cited synthesis. Its failure modes are believing one
confident wrong source, fabricating citations, and being slow across many sequential
calls. `bench_10_research.sh` runs LDR's own loop through `aildr local` (local model
gathers *and* writes, no Claude in the path) on ten questions, each with an oracle:
MUST patterns for the right answer, MUST_NOT for the tempting falsehood. Three are
adversarial — the BBR-replaced-CUBIC trap (a VyOS forum post says yes; in the smoke
run the 35B believed it), a false premise ("why did Gentoo remove OpenRC in 2025?"),
and an io_uring nuance. Engine pinned per question and identical for both models
(searxng for the traps and docs, wikipedia for anchors, arxiv for the paper); model
order alternates per question; a SearXNG health gate waits out engine suspensions
and retries a source-less run once (fired once in 60 runs); three passes.

**Result (after the oracle fix below):**

| model | CORRECT | WRONG | MISS | median s | unique sources | dup % | orphan citations | uncited / run |
|---|---|---|---|---|---|---|---|---|
| `ornith15-128k` | **27/30** | 0 | 3 | 111 | 12.2 | 12 | 0 | 4.6 |
| `ornith15-9b` | **27/30** | 0 | 3 | 116 | 13.0 | 13 | 1 | 5.5 |

Per question both are 3/3 on nine of ten — including all three adversarial ones —
and 0/3 on `mamba2`, where neither run retrieved the Mamba-2 paper (arxiv appeared
in 5 of 6 source lists but never the right entry); both then said so explicitly and
filled from "previous knowledge" with a wrong description (the 35B credited Mamba-2
with S6 gating, which is Mamba-1's; the 9B called it a hybrid). A search failure
handled the same honest-but-wrong way by both.

**The oracle needed two fixes, both applied symmetrically and rescored offline.**
First scoring showed 24/23 CORRECT and 3/4 WRONG; every one of the eight WRONGs was
a false positive — the MUST_NOT phrase quoted inside a denial ("no evidence that
Gentoo removed OpenRC", "they do not support the claim that BBR is the default") or
a question heading ("Has BBR Replaced CUBIC as the Default?"). MUST_NOT now counts
only in assertive sentences (no heading, no `?`, no negation token). The
false-premise MUST list was then too narrow for the 9B's phrasing ("I cannot
confirm", "never explicitly state") — broadened; all six gentoo answers reject the
premise on reading. `--rescore` mode added so an oracle fix never costs a rerun.

**Reading.**

1. **Indistinguishable at the job.** Same correctness, same three misses, the same
   traps passed 3/3, near-identical source counts; the one hallucination-class signal
   in 60 runs is a single orphan citation, in the 9B's column.
2. **The 9B is not faster here.** Median 111 vs 116 s, 57 min total each — the LDR
   loop is search- and fetch-bound, so tg 84 vs 53 never shows. Speed would only
   separate them when the LLM share grows (more iterations, Claude-driven follow-ups).
3. **The blind read agrees**: the 9B writes longer, more structured answers (io_uring:
   680 vs 527 words, short answer / CVE list / bottom line); the 35B is terser with a
   sharper caveat about truncated excerpts. Neither is better research prose.
4. **So the decision falls to what surrounds the job.** The 9B leaves ~4 GB of VRAM at
   128K, enough for the embedder on the GPU at a small batch (measured below); the 35B
   leaves 1.5 GB and forces the embedder onto the CPU. The 35B's edges — 12/12 coding,
   20/20 workflow discipline — are real and irrelevant to LDR's loop.

**Decision for aildr: `ornith15-9b`** — equal research quality, the GPU has room for
the embedder, faster per call for the day the loop stops being search-bound.
`ornith15-128k` stays the serious-agentic default for coding and rule-heavy work.

**Embedder beside the 9B, measured right after (Qwen3-Embedding-0.6B Q8_0, `--gpu`):**
baseline `ornith15-9b` @128K + desktop 12 629 MiB; `-c/-ub/-b 1024` → +1 640 MiB (14 269
total) but a 1 202-token input is rejected (`exceed_context_size_error`); `-c/-ub/-b 2048`
→ +2 421 MiB (15 050 total, ~1.25 GB spare), 1 202-token input embeds fine (1024-dim).
The default 8192 batch is the 6.6 GB figure — buffers, not weights. So: 2048 during a
run (covers airag's 2000-char chunks and LDR snippets), 8192 only for bulk `airag index`
on the empty card. Left running: `ornith15-9b` + GPU embedder at 2048.

