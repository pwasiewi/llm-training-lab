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
| `gpt-oss20b-q8_0` | MoE 20.9B | Q8_0, 12.11 GiB | whole (0) | ~214 | regex 57%, interp 71% (14 runs); template never failed | **DEFAULT fast tier** |
| `ornith-128k` (Ornith-1.0-35B) | qwen35moe A3B | Q4_K_M, 19.7 GiB | 24 | ~52 | regex never finishes (3/3 TIMEOUT @1200s); rest near-clean | **DEFAULT serious agentic** |
| `qwythos` (Qwythos-9B-v2 +MTP) | qwen35 dense hybrid | Q8_0 | whole | 149 (MTP) | parser tier FAILs (9B ceiling) | fast iteration on scoped tasks |
| `ornith-9b` (Ornith-1.0-9B) | qwen35 dense | Q8_0, 9.53 GiB | whole | | interp 12/13 near-miss @900s; perf anti-pattern | accepted; judgment pending |
| `qwen36-128k` (HauhauCS 35B-A3B) | qwen35moe | IQ4_XS, 17.43 GiB | 14–16 | ~82 | fastest template (118 s); insort-not-Fenwick on perf | general alternate |
| `qwen36u-mxfp4-128k` (unsloth) | qwen35moe | MXFP4_MOE, 20.2 GiB | 16+ | | tied Ornith 7/8 | backup alternate |
| `glm-flash` (GLM-4.7-Flash) | deepseek2 MoE 30B-A3B | Q4_K_XL, 17.5 GiB | 12 @32K / 24 @128K | 63 / 39.6 | 4/7 — gives up early on parser tier | 32K chat only |
| `dsv4flash` (Qwen3.5-9B-DSV4) | qwen35 dense | Q6_K | whole | ~93 | interp 0/13 — total parser failure | chat only, **NOT an agent** |
| `gpt-oss20b` (MXFP4-Aggressive) | MoE 20.9B | MXFP4, 11.27 GiB | whole (0) | ~214 | never passed template+interp across runs | fallback/ref (superseded by Q8_0) |
| `gpt-oss20b-f16` (unsloth F16) | MoE 20.9B | F16, 12.83 GiB | 2 | 2–6× slower | regex 6/14 | fallback for interp-style work |
| `gpt-oss20b-heretic` (DavidAU) | MoE 20.9B | IQ4_NL, 12.6 GiB | 1 (profile 2) | slowest of family | template 4/10, interp 0/13, regex disq. | **REJECTED** (conf ref) |
| `ornith-q5-128k` (Ornith Q5_K_M) | qwen35moe | Q5_K_M, 24 GiB | 22 | −13% vs Q4 | regex TIMEOUT, no file at all | ref — not for time-boxed work |
| `gemma4-fable5` (yuxinlu1 12B) | gemma4 dense | Q8_0, 12.7 GiB | ctx ≤32K only! | | 5/10; interp 1/13, regex 0/14; gen_n=1 red flag | **REJECTED** |
| `gemma4-qat` (HauhauCS 12B) | gemma4 dense | Q4_K_M, 7.38 GiB | | | 100% reproducible stream-hang after task 1 | **REJECTED** |
| 27B dense family (5 finetunes) | qwen35 dense 65L | Q4, 16–19 GB | impossible @128K (any quant) | | | **REJECTED**, GGUFs deleted (113 GB) |
| `qwen36u-iq4xs` / `qwen36u-q4kxl` | qwen35moe | 16.5 / 20.8 GiB | 10 / 17 | | scratch FAIL / 2× TIMEOUT | **REJECTED**, deleted (38 GB) |
| GLM 5.2 (744B) | MoE | any | needs ≥245 GB RAM+VRAM | | | **REJECTED** — never fit |

### Per-model notes

- **`gpt-oss20b-q8_0`** — the fast-tier default since 2026-07-14/15. gpt-oss
  keeps MoE experts at native ~4-bit regardless of quant label, so Q8_0
  costs almost nothing over MXFP4 and still fits whole at 131072 ctx
  (15.0/16.3 GiB). 14 single-model runs: 6 easy tasks + template at 100%,
  codec/toposort 93%, interp 71%, regex 57%. Expect one parser-tier miss
  per run as the norm, not a clean 12/12.
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
- **gemma4 pair** — new arch loads fine on llama.cpp 9988, so the rejects
  are model-level: QAT variant has a 100%-reproducible dead-stream bug
  after the first task; fable5 variant underperforms a 9B and can't hold
  >32K ctx compute buffers.
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
