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
  (`--approval-mode yolo`) against each aillama profile on three coding tasks
  of increasing difficulty (pytest bugfix → CLI tool from scratch → implement
  an LRU+TTL cache from a provided test suite). Verdicts are objective:
  protected-file checksums plus the script's own pytest/functional checks —
  the agent's claims and self-written tests are never trusted.
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
