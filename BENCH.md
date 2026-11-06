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

## The bench_08_horizon task (added 2026-10-16)

`bench_08_horizon.sh` (the name `bench_08_spec_offload.sh` was already taken by
the drafter bench) turns "how much can this model be trusted along a chain and
across its window" into numbers per model (design: memory
`project_bench08_design`). Two parts, every trial one JSONL
(proposal, verified_outcome) row under `/mnt/cc/strata/bench08/runs/<TAG>/`:

| Part | Unit | Harness | Output |
|---|---|---|---|
| `chain` | arithmetic/conditional chain of N **dependent** steps, answer `S1 = …`, `FINAL = …` | **raw** llama-server `/v1/chat/completions` (qwen-code's autoCompact would measure the harness) | `p_step(direct)` (each step judged against the model's own previous value), `p_chain(MLE)` from success ≈ p^N, `N_max(0.5)`, per fill: `p0`, `L_eff` (first fill with p < p0 − `DELTA`), post-knee slope |
| `vote` | `VOTE_K` headless qwen-code create passes of the nanoeuler ebuild, pinned `EGIT_COMMIT=4cd90a5` | `bench08/qwen-ebuild-oracle.sh` (copy of `~/QwenCode/qwen-ebuild-run.sh`: local bare mirror `/mnt/cc/strata/bench08/nanoeuler.git`, git-r3 override for the build, `pin` gate, machine-readable gates) | per stage (header, inherit, keywords, pin, compile, test, install) the modal APPROACH label and whether it passes the oracle: **class A** = vote recovers, **class B** = modal answer is wrong (voting amplifies). `ARBITER_K>0` = arm C, FAIL gates fed back, iterations-to-pass |

The grid is factorial (`NS` × `FILLS`): fill is total prompt tokens, so a
short chain is padded to the same occupancy as a long one and p is fill-free
per row. Ballast = neutral prose frozen once (`ballast/readme.txt`, sha256
checked every run); `BALLAST=distractor` and `POSITION=start` are separate
populations — never pool them. Chain seeds depend on (N, rep) only, so every
fill level sees the same chains (paired). `--calibrate` runs the oracle on a
known-good pinned ebuild (all gates PASS) and the committed-binary variant
(fails exactly build, installed, noop, realcompile — the prebuilt
`nanoeuler_check` also fails gradcheck). Power: SE(p) ≈ √(p(1−p)/n); a
0.90 → 0.80 drop needs ~100 scored steps per fill — the default grid
(`NS=1,2,4,8,16`, `REPS=10`) gives 310 per fill.

First smoke (2026-10-16, `TAG=smoke1`, ornith15-9b, 2 reps): `--calibrate` OK; chain 40/40 steps at fill 0 and 16K (p0 = 1.000 — too small a grid to mean anything); vote 1 of 2 attempts passed every gate (457 s), the other timed out at 900 s and did not recover in one arbiter iteration. Code path check only; no reference results yet.

## Model fleet — current state

**This is the only live picture; it supersedes every dated section in
`BENCH-archive.md`.** One row per model/variant ever run on this box. The
evidence behind each verdict stays in the archive, which is append-only and
never rewritten. Columns are empty where a model was rejected before the
measurement was taken.

Roles in service today (2026-10-30):

| Role | Profile |
|---|---|
| serious agentic (default) | `ornith15-128k-mc` — reconfirmed by the user 2026-10-19 over `gsq35-mc` |
| fast coding tier (default), English/code | `groot-19b-iq4xs` — **switched 2026-11-05 on the user's word** ("switch roles") after the finals + Aider; `gpt-oss20b-udq8kxl-high` stays the fast tier for Polish or untimed work, see "Finalists in one session" |
| workflow agent (default) | `ornith15-9b` |
| serious agentic (alternate) | `tiel-128k-mc` |
| fast tier, speed pick | retired 2026-11-05 (`ornith15-gsq3` on the HDD, profile kept); Groot covers it |
| alternate, not promoted | `bolt` — lost the same-session N=20 run to `ornith15-128k-mc` (2026-10-30) |

## Finalists in one session — 2026-11-05 (`night/q-final-1105.sh`, 01:38 → 17:22, rc=0, 0 Xid)

Six profiles, every phase in the same session with the same llama.cpp b12001, so the columns are
comparable with each other and NOT with older dated runs. Groot arm = IQ4_XS (winner of the quant arms,
`night/q-groot-quant-1106` + `q-groot-q4km-1106`). The Aider column comes from `q-aider-1105` (17:23 → 22:55,
rc=0, 0 Xid): phase A of the finals ran on a dead venv (Python 3.13 removed 11-02) and was rerun on the
rebuilt one, same five models, same llama.cpp. Evidence: `BENCH-archive.md` "2026-11-05 — finalists".

**Table 1 — speed and fit** (128K ctx, q8_0 KV; 27Bs at 53K ctx; bench_11 ×2, needle on; VRAM = after the deepest prompt)

| | `ornith15-128k-mc` (leader) | `groot-19b-iq4xs` | `enru-27b-mtp` | `swift-27b-mtp2-kv4` | `gpt-oss20b-udq8kxl-high` | `ornith122-iq2` |
|---|---|---|---|---|---|---|
| file | Q4_K_M 20.2 GiB, ncmoe 28 + mc | IQ4_XS 10.9 GB, whole | IQ3_XXS 11.0 GB + MTP, whole | IQ3_S 12.1 GB + MTP, whole | Q8_K_XL 12.3 GiB, whole | UD-IQ2_XXS, ncmoe 40 |
| tg @1K / 16K / 49K / 98K | 93 / 106 / 101 / 88 | **187 / 175 / 157 / 136** | 107 / 100 / 92 / — | 102 / 92 / 86 / — | 185 / 159 / 122 / 88 | 21 / 22 / 22 / 21 |
| pp @49K / 98K | 2 758 / 2 363 | **7 563 / 5 239** | 1 316 / — | 1 327 / — | 7 385 / 4 994 | 1 050 / 957 |
| VRAM after deepest | 14 673 MiB | **13 611** | 14 483 | 14 976 | 15 036 | 14 038 |
| needle | 8/8 | 8/8 | 6/6 (to 49K) | 6/6 (to 49K) | 6/8 — **98K: empty answer, 256 tokens of reasoning** (format artifact, see Table 3) | 8/8 |

**Table 2 — code** (bench_12_hard RUNS=2, nine tasks, bwrap; `o122` RUNS=1; Aider hard 20 = `q-aider-1105`, one run, two tries per task)

| | leader | groot-iq4xs | enru-mtp | swift | gpt-oss-high | o122 |
|---|---|---|---|---|---|---|
| bench_12 PASS | **12/18** | 8/18 | 5/18 | 5/18 | 4/18 | 2/9 (6 TIMEOUT) |
| 2/2 held | ledger, chain6, chain10, rules | ledger, chain10, longctx | — | ledger, reconcile | tracespec | chain6, chain14 |
| 0/2 | ebuild | chain6, rules, reconcile, ebuild | chain6, chain10, rules, tracespec | chain6, chain10, chain14, rules, tracespec, ebuild | ledger, chain10, rules, longctx, reconcile, ebuild | ledger, chain10, tracespec, reconcile, ebuild, rules (all TIMEOUT) |
| mean PASS run | 522 s | **202 s** | 254 s | 203 s | 410 s | 1 518 s |
| groot quant arms (RUNS=2, 11-04/05) | — | IQ4_XS 13, Q4_K_M 13, Q4_K_S 11, Q4_0 10 | | | | |
| Aider hard 20 (`q-aider-1105`) | **15/20** (75 %, CI 53–89) | 9/20 (45 %) | 7/20 (35 %) | 10/20 (50 %) | **13/20** (65 %) | skipped (speed) |
| Aider pass on try 1 / wall for 20 | 7 / 48 min | 4 / 43 min | 1 / 33 min | 2 / 29 min | 3 / **176 min** | — |
| Aider paired vs leader (exact McNemar) | — | 0–6, **p 0.031** | 0–8, **p 0.008** | 1–6, p 0.125 | 1–3, p 0.625 | — |

Aider reading: the leader is the only model significantly ahead of Groot and enru; **gpt-oss-high is the
runner-up on pass rate** (13 vs 15, not separable at N=20) and beats Groot 5–1 on the tasks they split
(p 0.22), but its reasoning costs 4× the wall time (176 vs 43 min for the 20 tasks). Five tasks fell for
everyone (`go/poker`, `java/connect`, `java/word-search`, `cpp/spiral-matrix`, plus `rust/macros` for the
three non-27Bs); the JavaScript set and `rust/grade-school` passed for everyone.

bench_12 at RUNS=2 moves ±3 between sessions for the same model (groot 13 → 8 overnight on the same
file), so only the leader-vs-rest gap (12 vs ≤8) is a finding; the order of the 4–8 group is not.
`ebuild` is 0/2 for everyone: the ldflags trap holds against the whole fleet.

**Table 3 — long context** (bench_08_horizon chain part, NS 1/4/8/16, REPS 10, 290 steps per fill; p_step = per-step correctness)

| fill | leader | groot-iq4xs | enru-mtp | swift | gpt-oss-high | o122 |
|---|---|---|---|---|---|---|
| 0 | 0.903 ±.017 | 0.976 | **0.990** | **0.997** | 0.934 | 0.914 |
| 16K | 0.903 | 0.976 | 0.990 | **1.000** | 0.997 | 0.966 |
| 32K | 0.962 | 0.986 | **0.997** | 0.993 | 0.993 | 0.986 |
| 98K | 0.966 | 0.972 | — | — | **0.997** | 0.990 |
| N_max(S=0.5) @98K | 20 | 23 | — (173 @32K) | — (99 @32K) | **173** | 63 |

Nobody decays with fill on this chain; the leader is the LOWEST at every fill (its loss is at the
step level, not the context level) and groot holds a longer chain than the leader everywhere. The
gpt-oss 98K needle MISS is therefore a format artifact (reasoning consumed the 256-token answer
budget), not a context failure.

**Table 4 — language and tools** (8 Polish prompts, thinking off, graded by hand 0–2: 2 = usable as is, 1 = usable with edits, 0 = unusable; tools = bench_14, same session for groot arms, 11-03/04 hunt for the rest)

| | leader | groot-iq4xs | enru-mtp | swift | gpt-oss-high | o122 |
|---|---|---|---|---|---|---|
| Polish /16 | 12 | **0** | 9 | 11 | 14 | **15** |
| typical miss | case slips (`Fotosyntezę to`), `koń` miejscownik, Cyrillic token in an explanation | answers in English/Chinese, loops (3 418 tokens on one prompt) | invented words (`wiadność`, `opartowalność`), `koń` declension collapses, wrong aspect theory | `koń` dopełniacz `kon`, wrong aspect theory, Polish-Latin blends | `koń` celownik/wołacz, `niedokładny` for `niedokonany` | one 4-sentence answer where 3 were asked |
| tokens on the 8 prompts | 1 466 | 8 263 | 2 183 | 1 650 | 9 538 (hidden reasoning) | 1 344 |
| tools 6/6 | yes | yes | yes | yes | yes | yes |

**Table 5 — roles after the finals** (recommendation; the "Roles in service today" table above changes only on the user's word)

| Role | Today | Recommended | Why |
|---|---|---|---|
| serious agentic (default) | `ornith15-128k-mc` | **keep** | bench_12 12/18 vs 8/18 for the runner-up, same session; Polish 12/16 |
| fast coding tier (default), English/code, time-boxed | `gpt-oss20b-udq8kxl-high` | **`groot-19b-iq4xs`** | 1.3–1.5× gpt-oss tg at depth, bench_12 8 vs 4, bench_07 19/20 vs 17/20 and 15/20, parser 20/20, horizon ≥ gpt-oss to 32K; 13.6 GiB whole. Aider 9/20 in 43 min vs gpt-oss 13/20 in 176 min: Groot is the pick when the clock matters |
| fast tier, Polish or no time limit | — | `gpt-oss20b-udq8kxl-high` | Polish 14/16, 98K chain 0.997, Aider 13/20 (runner-up to the leader, 5–1 over Groot on split tasks); the fallback when Polish output or Aider-style pass rate matters more than speed |
| fast tier, speed pick | `ornith15-gsq3` — demoted to the HDD 2026-11-04 (parser 8/12 in the promotion run); retire once Groot is confirmed | **retire** (demoted to HDD 11-04) | parser 8/12, regex 0/3 in the promotion run; groot is faster and cleaner |
| fast tier, speed pick (11-06 addendum) | — | `groot-19b-iq4xs-nothink` | bench_12 12/18 at 108 s mean PASS run (think: 8/18 at 202 s), Aider 4/20 in 16 min; `--reasoning off` — see "Thinking arms — 2026-11-06" |
| slow-but-smart alternate | `ornith122-iq2` | keep as is | Polish 15/16, horizon 0.99, but 20 tok/s: bench_12 6 TIMEOUTs; not for time-boxed work |
| 27B alternates | `enru-27b-mtp`, `swift-27b-mtp2-kv4` | **neither promoted** | bench_12 5/18 each, Polish 9 and 11, horizon excellent to 32K; 53K ctx only. `enru-27b` plain stays the 27B candidate on parser evidence |

## Thinking arms — 2026-11-06 (`night/q-think-1106.sh`, TAG `think-1106r`, 00:32 → 07:23, rc=0, 0 Xid)

Same llama.cpp b12001, five arms in one session: Groot IQ4_XS thinking on / off (`--reasoning off`),
gpt-oss-20b Q8_K_XL at reasoning effort low / medium / high. Aider hard 20 for all five; bench_12_hard
RUNS=2 for the three arms without a number. Evidence: `BENCH-archive.md` "2026-11-06 — thinking arms".

| | `groot-19b-iq4xs` (think) | `groot-19b-iq4xs-nothink` | `gpt-oss20b-udq8kxl-low` | `gpt-oss20b-udq8kxl` (medium) | `gpt-oss20b-udq8kxl-high` |
|---|---|---|---|---|---|
| Aider hard 20 | 7/20 | 4/20 | 5/20 | 7/20 | 8/20 |
| Aider wall for 20 | 50 min | **16 min** | **8 min** | 34 min | 161 min |
| Aider paired (exact McNemar) | 4–1 over nothink, p 0.375 | — | 2–5 vs high, p 0.453 | 1–2 vs high, p 1.0 | 3–2 over groot-think, p 1.0 |
| bench_12 RUNS=2 | 8/18 (finals; 13/18 the night before) | **12/18** | 0/18 | 0/18 | 4/18 (finals) |
| bench_12 mean PASS run | 202 s | **108 s** | — | — | 410 s |

Reading: **thinking off costs Groot nothing on the multi-step own tasks** (12/18 = the leader's finals
count, 2/2 on chain6, chain10, tracespec, longctx, rules) and halves the run time, but on Aider it
drops 7 → 4 (not separable at N=20). With the ±3 bench_12 session drift this is "not worse, 2× faster",
not "better". **gpt-oss reasoning effort does not separate on Aider** (5 / 7 / 8 in 8 / 34 / 161 min)
and **low and medium are 0/18 on bench_12** — the multi-step tasks are beyond gpt-oss-20b at any effort
(high: 4/18). Session drift on Aider is large: gpt-oss-high 13/20 on 11-05 → 8/20 today, Groot 9 → 7 on
unchanged profiles; the 11-05 "runner-up" reading of gpt-oss-high is one session's result. Role
consequence (recommendation; the table above changes only on the user's word): fast tier stays
`groot-19b-iq4xs` with thinking on for exercise-style edits; `groot-19b-iq4xs-nothink` is the speed pick
for time-boxed multi-step agent runs (replaces the retired gsq3); `gpt-oss20b-udq8kxl-low` is not worth a
role, medium is the sane gpt-oss setting when Polish is not needed (same Aider count as high in 1/5 of
the time), high stays for Polish.

### In service, by role

| Profile / model | Arch | Quant, file size | Fit @128K (`--n-cpu-moe`) | tg tok/s | bench_05 weak spots | Status / role |
|---|---|---|---|---|---|---|
| `ornith15-128k-mc` (same GGUF + **expert cache**, PR #27861 user patch) | qwen35moe | Q4_K_M, 20.22 GiB | 28 + `--moe-cache-mib 3540` (upstream cache, b12001; floor 26), `-ub 2048` | **103 @49K, pp 2 757** (110 @16K; 256-token decode, 2026-10-30) | parser tier ×2: 8/8, **regex 2/2** (657, 353 s); **bench_07 20/20, every item 20/20** (2026-10-01) | **DEFAULT serious agentic** (reconfirmed by the user 2026-10-19 over gsq35-mc: needle 4/4 at 1K where gsq35-mc refuses short SECRET-CODE prompts) (2026-10-01, confirmed 2026-10-18 final head-to-head: 24/24 at 1200 s incl. regex 2/2, bench_07 10/10, the only one of four) — ornith quality at +31 % decode / ×1.9 prefill vs the 09-30 profile; since 2026-10-29 runs on upstream `--moe-cache-mib` (the old user patch was never installed); 2026-10-30 same-session vs `bolt`: parser ×3 11/12 (regex timeout once), **bench_07 20/20 every item**, tools 6/6 |
| `gpt-oss20b-udq8kxl-high` (same GGUF + `--reasoning-effort high`) | MoE 20.9B | Q8_K_XL, 12.29 GiB | whole (0) | ~193 | **bench_07 17/20 and 15/20** (two N=20 sessions; default 12/20, 9/20; low 4/20); median run 42 s | **DEFAULT fast coding tier** (2026-10-02) — the lastline/evidence weakness was effort-dependent |
| `ornith15-9b` (Ornith-1.5-9B) | qwen35 dense | Q8_0, 9.11 GiB | whole (12.0 GiB @128K) | 84 / 73 @49K | @900 s N=4: interp 13/13 scored every run but 3/4 by verdict, template 2/4, perf 3/4; **@1200 s N=3: template/interp/perf 3/3 each = 11/12 every run**, regex 0/3 (12/14 once). **bench_07 19/20 PASS** (every axis 20/20, evidence 19/20) | **DEFAULT workflow agent** (rule-heavy multi-step qwen-code jobs, whole-fit); with `TASK_TIMEOUT=1200` also a full coding agent minus regex; udq8kxl keeps the sprint tier (regex, 2.3× tg) |
| `tiel-128k-mc` (Tiel-Coder-35B-A3B + **expert cache**) | qwen35moe | UD-Q4_K_XL, 20.82 GiB | 29 + `--moe-cache-mib 3417` (floor 27), `-ub 2048` | 89 @49K, pp 2 660 (b12001) | parser tier 2 rounds: **4/4, 4/4** (regex 14/14 in 413 s, 455 s); **bench_07 20/20, every item 20/20** (2026-10-02) | **ALTERNATE serious agentic** (2026-10-02) — the only sibling that matches `ornith15-128k-mc` on both axes, ~6 % slower decode; falls back to `tiel-128k` without the user patch |
| `gsq35-mc` (Ornith-1.5-35B-A3B, IST-DASLab GSQ-RCO 3.5-bit; **same arch**, NextN + vision dropped) | qwen35moe | Q2_K/Q3_K/Q4_K experts + Q8_0 attention/GDN/shared, 3.50 bpw, 14.12 GiB | **15** (floor 13 on b12001; 12 boots but fails the 49K needle) + `--moe-cache-mib 1129`, `-ub 2048` | **116 @49K, pp 3 906** (125 @16K; b12001, 2026-10-30) | needle 6/6; parser tier ×2 at 900 s: **8/8, regex 2/2** (334, 497 s); **bench_07 20/20, every item 20/20**, median 158 s (at ncmoe 26) | **CANDIDATE successor of `ornith15-128k-mc` (2026-10-18)** — same scores at N=20, +28 % decode, +40 % prefill, 6 GiB smaller file; **confirmed at ncmoe 16 (2026-10-18 gsq35-confirm): parser 8/8, bench_07 20/20 (median 133 s vs 159 s default), bench_08 flat to 96K** — **FALLBACK (faster) to `ornith15-128k-mc`** (user 2026-10-19); 49K tg 104 on 0005 + CUDA 13.4.3 (night/research/40, 10-19); refused the 1K needle 2 of 5 runs |
| `ornith15-gsq3` (Ornith-1.5-35B-A3B, pfeifferj GSQ-RCO 3-bit) | qwen35moe A3B | GSQ-RCO 3-bit, 12.99 GB | **whole**, 14.2 GiB @49K | **127–132 @49K, pp ~3 380** (170–190 @1K) | needle OK 1K/16K/49K; template 2/2, perf 2/2, **interp 1/2** (11/13), **regex 0/2** (8/14, 13/14 at 209/233 s); bench_07 **19/20** (evidence 19/20, every other item 20/20), median run **88 s** (max 146 s); control `ornith15-128k-mc` same session 4/4 incl. regex 272 s | **FAST TIER 2026-10-17** - ~1.8x the decode of `ornith15-128k-mc`, whole on the card with 1.8 GB spare, but the 3-bit quant costs exact code (regex/interp); not the default |
| `swift-27b-mtp2` (Swift-1.5-Qwen3.8-27B, ukisai, MTP head in the GGUF) | qwen35 hybrid 27B + MTP | GSQ-RCO IQ3_S, 12.12 GB | whole, 15.5 GiB @boot | not measured with MTP (the plain profile: 35–36 @49K, pp ~1 230) | bench_05 = the plain `swift-27b` (same GGUF): template/interp/perf 2/2 each, **regex 1/2** (10/14 FAIL at 771 s, then 14/14 in 415 s); **bench_07 20/20, every item 20/20**, median run **102 s** (max 146 s) vs 215 s (max 304 s) without MTP, which also scored 20/20 | **ALTERNATE, fast 27B tier 2026-10-18** — run as `swift-27b-mtp2-kv4` (q4_0 KV: the q8_0 profile OOMs on the first request when the desktop holds >1 GiB); final head-to-head 23/24 (regex 1/2), bench_07 10/10 at 116 s median, fastest in this log; was CANDIDATE 10-17: best 27B so far: MTP draft-2 halves the bench_07 run time at no score cost; plain `swift-27b` dominated by it (profile removed 2026-10-17). `--spec-type draft-mtp --spec-draft-n-max 2 --ctx-checkpoints 4`, `-c 53248` |
| `q38ud-27b` (unsloth Qwen3.8-27B) | qwen35 hybrid 27B (MTP head not used) | UD-Q3_K_XL, 13.15 GB | whole, 15.4 GiB @49K | 38 @49K, pp ~1 350 (55 @1K) | needle OK 1K/16K/49K; template/interp/perf 2/2 each, **regex 1/2** (8/14 FAIL at 663 s, then 14/14 in 367 s); **bench_07 20/20, every item 20/20**, median run 172 s (max 470 s) | **alternate, dominated by swift-27b-mtp2 2026-10-18** (final head-to-head: 23/24 scored, regex 1/2 at 1147 s, bench_07 10/10 at 143 s, one OOM death at the VRAM edge); was CANDIDATE 10-17: same tier as `bonsai2-27b`/`swift-27b`; no MTP arm run yet. `-c 53248` (49152 rejects bench_11's 49K prompt with HTTP 400) |
| `enru-27b` (Igor282 qwen3.8-27b-pruned-enru: weights untouched, tokenizer 248k → 173k, CJK/Arabic/Indic scripts cut) | qwen35 hybrid 27B (MTP head present, not used) | IQ3_XXS imatrix, 10.99 GB (3.31 bpw; FFN 3.08, gated-delta 3.72, attention 3.02) | whole, 14.0 GiB @49K, q8_0 KV | 54 @49K, pp ~1 400 (63 @1K) | needle 6/6; bench_14 tools **6/6**; parser tier ×2 **8/8, regex 2/2** (249, 474 s); **bench_07 10/10**, median 179 s; Polish: very good (identical PL tokenisation to the original) | **CANDIDATE 2026-11-03** — first 27B with 8/8 on this card; same session control `swift-27b-mtp2-kv4` 4/4 (regex 130 s) at roughly half the run time (MTP); q8_0 KV with 2 GiB spare where swift needs q4_0. Next: an `enru-27b-mtp` arm (`--spec-type draft-mtp`; the card warns the draft head was not retrained after the vocab cut) |
| `saluki-27b` (ConwayResearch Underdog-Saluki-27B-1.0, tool-call tuned) | qwen35 hybrid 27B, MTP dropped | ISTA GSQ-RCO IQ2 mix, 7.90 GB (2.35 bpw; embeddings IQ1_M) | whole, 11.7 GiB @49K | 62 @49K, pp ~1 280 (70 @1K) | needle 6/6; tools **6/6**; parser ×2 **5/8** (interp 9/13 once, **regex 0/2**: 0/14, 11/14); bench_07 10/10, median 184 s; Polish good, minor errors | **alternate, not promoted 2026-11-03** — best of the 2-bit pair; loses exact code to the same-session control (swift 4/4) |
| `caphlon-27b` (ssuasd Caphlon-Model: unsloth IQ2_S + vocab cut to 100k merges + layers 12–15 removed) | qwen35 hybrid, **60** layers, MTP dropped | IQ2_S, 7.38 GB (2.47 bpw) | whole, 11.3 GiB @49K | **67 @49K** (79 @1K), pp ~1 400 | needle 6/6; tools 5/6 (`decline`: invents the city); parser ×2 **4/8** (pass 2: 1/4); bench_07 10/10, median 192 s; **Polish unusable** (wrong content, invented words; +17 % tokens per PL sentence) | **REJECTED 2026-11-03** — fastest 27B, weakest everywhere else; GGUF + profile deleted 2026-11-04 |
| `groot-19b-iq4xs` / `groot-19b` (GrootTeam Groot-19B-A3B-261023: Qwen3.6-35B-A3B with experts pruned 256 → 128 + <0.1B tokens post-training; bartowski IQ4_XS / Q4_0 / Q4_K_S / Q4_K_M on disk) | qwen35moe A3B, 128 experts | IQ4_XS 10.9 GB (Q4_K_M 12.5 GB = quality alternate, Q4_0 11.0 GB = fastest decode) | **whole** (ncmoe 0), 13.6 GiB after 98K (Q4_K_M 15.1 GiB) | **157 @49K, 136 @98K, pp 7 560** (Q4_0 167 @49K) | tools 6/6; parser **20/20** (11-03 → 11-04); **bench_07 19/20, median 49 s** (leader 138 s); bench_12 RUNS=2: 13 → 8 across two sessions (quant arms IQ4_XS 13, Q4_K_M 13, Q4_K_S 11, Q4_0 10; leader 12 same session); horizon p 0.97–0.99 flat to 98K; Polish **0/16** (unusable) | **RECOMMENDED fast coding tier (English/code only), 2026-11-05** — card sampling temp 0.6, top_p 0.95, top_k 20, presence 0.0; tables above |
| `ornith122-iq2` (unsloth Ornith-1.5-122B-A10B) | qwen35moe 122B-A10B | UD-IQ2_XXS | 40 (floor 38; 36 boot fail), 14.0 GiB after 98K | 21–22 at every depth, pp 1 050 | tools 6/6; parser ×2 **8/8, regex 2/2** (690, 705 s); **bench_07 10/10**, median 399 s; bench_12 RUNS=1 2/9 with **6 TIMEOUTs** (1 500 s budgets); horizon 0.99 @98K; Polish **15/16**, best of the fleet | **ALTERNATE, slow-but-smart (2026-11-05)** — the leader's quality at 1/5 the speed; fine for chat and Polish, out of budget for time-boxed agent work |
| `enru-27b-mtp` (`enru-27b` + `--spec-type draft-mtp`) | qwen35 hybrid 27B + MTP | as `enru-27b` | whole, 14.5 GiB @49K | **92 @49K** (54 without MTP), pp 1 316 | tools 6/6; parser pooled 10/12, **regex 1/3** (two 13/14 near-misses); bench_07 10/10, median 121 s; bench_12 5/18; horizon 0.99 to 32K; Polish 9/16 | **speed alternate of `enru-27b`, not promoted (2026-11-05)** — the draft head was not retrained after the vocab cut; plain `enru-27b` keeps the 27B candidate status (regex 2/2) |
| `adq4-mc` (Ornith-1.5-35B-A3B, AtomicChat AD-Q4_K-IQ4_XS imatrix) | qwen35moe | Q4_K + IQ4_XS, 18.74 GiB, KLD 0.0315 (Q4_K_M 0.0477) | 27 (floor 25 on b12001) + `--moe-cache-mib 2958`, `-ub 2048` | 99 @49K, pp 3 000 (b12001) | needle 6/6; parser tier ×2: **8/8, regex 2/2** (241, 763 s); **bench_07 20/20, every item 20/20**, median **194 s** | **alternate, no reason to switch (2026-10-18)** — the better KLD buys nothing on these benches, decode = leader at 49K, agent runs 22 % slower (IQ4_XS experts dequantize slower on the CPU) |
| `ornith15-128k` (Ornith-1.5-35B-A3B) | qwen35moe | Q4_K_M, 20.22 GiB | 20 (floor 18), `-ub 2048` since 10-01 | 55 @49K, pp 2 691 | parser tier N=7: template/interp/perf 7/7 each; **regex 2/7 PASS** (744 s and 670 s on 09-19 — first non-gpt-oss regex passes in this log; 14/14 scored-but-late once, 5/14 once); **bench_07 20/20 PASS, every rubric item 20/20** (2026-09-20) | no-patch fallback of `ornith15-128k-mc` (2026-10-01); was **DEFAULT serious agentic** 09-19 → 09-30 ( beats Ornith 1.0 on interp, same VRAM shape and speed; Tiel-Coder tied it at N=3) — **best on both axes**, the first perfect bench_07 in this log |
| `tiel-128k` (Tiel-Coder-35B-A3B, coder finetune of the Ornith 1.5 base) | qwen35moe | UD-Q4_K_XL, 20.82 GiB | 21 (14.4 GiB after a 51K request) | 50 @51K depth, pp 1885 | parser tier N=3: template/interp/perf 3/3 each, regex 1/3 PASS (340 s = fastest regex pass ever, then 2× TIMEOUT no file) | tied with `ornith15-128k` at N=3 — no measurable coder-finetune edge; 21 GB kept only as an alternate, delete candidate |
| `qwen38d-128k` (Qwen3.8-35B-A3B-Distill, empero-ai) | qwen35moe | Q4_K_M, 20.22 GiB | 20 (14.1 GiB after a 51K request) | 53 @51K depth, pp 1865 | full suite ×3: easy/mid 24/24 clean and fast; template 2/3, interp 2/3, perf 3/3, regex 0/3 (11/14 once) → 11, 9, 11 of 12; **bench_07 15/20** (lastline 16, evidence 16, summary 19 — no hallucination trait) | accepted — general alternate, stronger than its base `qwen36-128k` on the parser tier and on bench_07 (udq8kxl 11/20), below `ornith15-128k` (regex, interp 3/3) at the same speed |
| `qwen36-128k` (HauhauCS 35B-A3B) | qwen35moe | IQ4_XS, 17.43 GiB | 16 (`--fit`: 16 + ffn_up) | ~82; 49K prompt 1850 pp / 75 tg | fastest template (118 s); insort-not-Fenwick on perf; 8/12 on 09-17 | general alternate — the speed pick (tg ~82); its distill `qwen38d-128k` is stronger at tg ~53 |
| `nex25-mini` (nex-agi Nex-N2.5-mini, agentic-trained, bartowski Q4_K_M) | qwen35moe 34.7B | Q4_K_M, 22.3 GB | 21 (14.0 GiB) | 48 @49K, pp 1 035 | N=1: 10/12; parser rounds 2–3: regex 14/14 late, then 0/14; interp FAIL 12/13 once; **bench_07 20/20 every item** (10-01) | alternate (2026-10-01) — workflow discipline = ornith, parser tier slower and less reliable |
| `nex25u-mini` (orcarouter Nex-N2.5-mini-Uncensored, gated) | qwen35moe 34.7B | IQ4_XS, 18.7 GB (on /mnt/cc) | 16 (14.2 GiB) | **64 @49K, pp 1 452** | N=1: 10/12; round 3: 2/4 (interp, regex 100 % but late); **bench_07 18/20** (`summary` 18/20, invents a description) | alternate (2026-10-01) — uncensoring did not break it (unlike huihui), but below nex25-mini on bench_07 |
| `qwen36-128k-mc` / `nex25u-mini-mc` / `qwen38d-128k-mc` / `nex25-mini-mc` / `tiel-128k-mc` (sibling GGUFs + expert cache) | qwen35moe | as the base lines | ncmoe 22 · 21 · 26 · 31 · 29 + `--moe-cache-mib` 1874 · 1718 · 2669 · 4153 · 3417, `-ub 2048` (floor+2 on b12001, 2026-10-30) | 49K tg 112 · 112 · 100 · 96 · 89 (b12001, 256-token decode; 10-02 cache-64 build: 74.7 · 77.8 · 68.5 · 67.8 · 67.5) | needle 3/3 each; parser round 1 (pre-fix build) / round 2 (fixed): tiel **4/4 / 4/4**, nex25u 2/4 / 2/4, nex25-mini 2/4 / 2/4, qwen38d 1/4 (bug casualties) / 3/4 (regex 0/14), qwen36 3/4 / –; bench_07 (N=20): tiel **20/20**, nex25-mini 19/20, nex25u 17/20, qwen36 12/20, qwen38d 11/20 | **cache profiles of the siblings** (2026-10-02); each falls back to its base line |
| `strata-iq2xs` (Qwen3.8-Flash-Next 125B, ISTA GSQ-RCO IQ2_XS, **Strata engine**, not llama.cpp) | qwen4exp MoE, 512 experts | IQ2_XS, 63.4 GiB (incl. 26.8 GiB n-gram table) | own expert cache, 128K + `--vram-reserve-mib 1700` (39 GB RAM, 14.7 GB VRAM) | 78–82 @128K cfg (MTP, 10-01 head-to-head; 09-29: ~50), pp ~1 700 — llama.cpp + cache 64 on the same file: 27–35, pp 465 | template 4/4, interp 4/4, **regex 3/3** (279–553 s), perf 2/4 (time budget); bench_07 20/20 every item (N=1) | **candidate → tuned 2026-10-02**: 0.1.31 + combined config (pcie 0.35, min-p 0.7, reserve 1000, kv-resident, prefill auto) = **97 tg / pp 2 805 @49K**, parser N=6/6/5 all PASS (regex 210–553 s), perf 4/6; needs bench_07 RUNS=20 on the new config before it takes the serious-agentic role; one model only, nothing heavy beside it |
| `gpt-oss20b-udq8kxl` (unsloth UD) | MoE 20.9B | Q8_K_XL, 12.29 GiB | whole (0) | ~193 | interp 3/3, regex 3/3 clean (bench_05); bench_07 11/20 again on 2026-09-19 (lastline 12/20, evidence 12/20 — base-model trait, stable since July) | fallback of `gpt-oss20b-udq8kxl-high` since 2026-10-02 (template-default effort = medium; the only fleet member that passes regex; 22–43 s per bench_07 run) |
| `gpt-oss20b-q8_0` | MoE 20.9B | Q8_0, 12.11 GiB | whole (0) | ~214 | regex 57%, interp 71% (14+ runs); template never failed | fallback (superseded by udq8kxl, still fastest tg) |
| `qwen36u-mxfp4-128k` (unsloth) | qwen35moe | MXFP4_MOE, 20.2 GiB | 16+ | | tied Ornith 7/8 | backup alternate |
| `gpt-oss20b` (MXFP4-Aggressive) | MoE 20.9B | MXFP4, 11.27 GiB | whole (0) | ~214 | never passed template+interp across runs | fallback/ref (superseded by Q8_0) |
| `gpt-oss20b-f16` (unsloth F16) | MoE 20.9B | F16, 12.83 GiB | 2 | 2–6× slower | regex 6/14 | fallback for interp-style work |
| `ornith-q5-128k` (Ornith Q5_K_M) | qwen35moe | Q5_K_M, 24 GiB | 22 | −13% vs Q4 | regex TIMEOUT, no file at all | ref — not for time-boxed work |
| `glm-flash` (GLM-4.7-Flash) | deepseek2 MoE 30B-A3B | Q4_K_XL, 17.5 GiB | 12 @32K / 22 @128K (was 24) | 63 / 42 | 4/7 — gives up early on parser tier | 32K chat only |
| `dsv4flash` (Qwen3.5-9B-DSV4) | qwen35 dense | Q6_K | whole | ~93 | interp 0/13 — total parser failure | chat only, **NOT an agent** |
| `ornith-9b` (Ornith-1.0-9B) | qwen35 dense | Q8_0, 9.53 GiB | whole | | interp 12/13 near-miss @900s; perf anti-pattern | accepted; judgment pending |
| `bolt` (NVIDIA Nemotron 3.5 Bolt 74B-A9B) | nemotron_h_moe, 88 blocks, 496 experts / 9 used + 1 shared, Mamba2 hybrid | vendor mixed: IQ2_XXS experts + NVFP4 Mamba + Q8_0 attn, 20.50 GiB | 46 (llama-bench floor 44, 42 OOM); `--fit` picks 41 overflowing layers, 13.7 GiB | 52-57 @1K, **50 @72K**; pp 634 @512 / **1551 @72K** | **none — bench_05 12/12, every task 100%** (43 min; regex 14/14 in 538 s, interp 13/13, perf 6/6, template 10/10) | **ALTERNATE 2026-10-30 (was CANDIDATE 10-29, cleanest screen in this log)** — also bench_07 RUNS=3 3/3 with every rubric item 3/3 (incl. `lastline` + `evidence`), bench_14 tools 6/6. Needle at 72K; KV q8_0 @128K only 708 MiB. No MTP head. **Same-session N=20 vs `ornith15-128k-mc` (2026-10-30, q-night-1029): NOT promoted** — parser tier ×3 10/12 vs 11/12 (template over 900 s at 10/10, regex 6/14 once), 2–5× the wall time per task; **bench_07 17/20 vs 20/20** (`summary` 17/20 — invents a description, every other item 20/20); tools 6/6 both. Alternate. **LiLiCorr drafter measured 2026-10-29: +21 % (62-64 t/s) at `--n-cpu-moe 52 --spec-type draft-dflash --spec-draft-n-max 3`, 71 % acceptance** — it cannot load at all at ncmoe 46 (needs 990 MiB), which is why the published 16 GB review called it a slowdown. Vision mmproj still unbenched |
| `nemotron` (Nemotron-3.5-Lightning-30B-A3B) | nemotron_h_moe A3B Mamba2 hybrid | ggml-org Q4_0, 18 GB | 18 @32K | **68** | 7/12 single run (2026-08-18) | kept, ref — leader NOT dethroned at N=1; never re-run at N>=20. unsloth UD quants rejected by mainline (embed nextn/MTP tensors) |
| `sage-a15-mc` (InternScience Sage-A1.5, qwen3_5_moe 35.95B-A3B VLM; mmproj NOT loaded) | qwen35moe | Q4_K_M, 20.22 GiB | 28 + `--moe-cache-mib 3540` (the leader's split, not re-probed), `-ub 2048` | 81.5 @49K, pp 2 751 | needle 49K OK; bench_14 tools 6/6; bench_05 N=1 **11/12** (regex TIMEOUT); same-session parser tier ×3 **8/12 vs 11/12**: template/interp 3/3, perf 2/3, **regex 0/3** (13/14, 1/14, none) — leader regex 3/3; **bench_07 RUNS=20 20/20 every item** (median 138 s; leader 20/20, 134 s) | **NOT promoted, GGUF + profile DELETED 2026-10-31 (q-cand2-1030)** — equals the leader everywhere except regex (0/4 over all its runs = a hard gap, not noise), ~20 % slower on the parser tier. **Vision (untested here, model card only):** image input via `Sage-A1.5-mmproj.gguf` (0.9 GB, `--mmproj`; text-only works without it); no video; the only multimodal score published is OmniDocBench **91.5** (document parsing, BF16 weights, not Q4_K_M). Base = Qwen3.6-35B-A3B + Agents-A1; native ctx 262K; the GGUF also carries an **MTP head** (never tried: `--spec-type draft-mtp`). Re-download both files (21.7 + 0.9 GB) when `bench_15_vision.sh` exists — it is the first fleet-class (35B-A3B) VLM; profile = `ornith15-128k-mc` args + `--mmproj` (re-probe ncmoe, the projector takes VRAM) |
| `lfm25-24b` (LiquidAI LFM2.5-24B-A2B, 4/64 experts active) | lfm2_moe | Q4_K_M, 13.53 GiB | **9** (floor 7: 6 and 4 die at boot) | 90–103 @49K, pp ~3 200 | bench_14 tools **5/6** (`nested`: object arg arrives as a JSON string; #30490 XML-in-content NOT reproduced on b12001); bench_05 N=1 **8/12** — easy/mid 8/8 (24–237 s), template 0/10 TIMEOUT, interp 0/13 TIMEOUT, perf FAIL 5/6, regex 2/14 TIMEOUT | **REJECTED as an agent 2026-10-30, GGUF + profile deleted 10-31** — fast easy-tier coder, no parser tier. Always thinks (ignores `enable_thinking:false`): bench_11 `NEEDLE=1` reads an empty answer even at NGEN 1024 — use a ≥3K token budget for its needle |
| `mimo-9b` (MiMo-V2.6-Distill-Qwen-9B) | qwen35 hybrid dense, 8 full-attn layers | Q8_0, 9.53 GB | whole | 75 @49K, pp 5 165 | needle 49K OK; bench_14 tools 6/6; bench_05 N=1 (10-30, q-cand-1030): easy/mid 7/8 (**intervals TIMEOUT 11/12**), template FAIL 4/10, interp TIMEOUT 10/13, perf TIMEOUT, regex stopped mid-run | **REJECTED 2026-10-30, GGUF deleted** — below `ornith15-9b` on the parser tier and drops an easy-tier task; vendor SWE-bench claim not borne out here |
| `bielik11b` (Bielik-11B-v3.0) | mistral dense, depth-upscaled 50 layers | Q6_K, ~12.8 GiB @32K | whole @32K | not measured | n/a — harness incompatible | **POLISH CHAT ONLY (2026-10-25)** — the GGUF chat template has no tool section, llama-server reports `supports_tools:false` and silently DROPS the `tools` array: never in an agent harness |
| `gemma4-26b` (Gemma 4 26B-A4B, Google QAT) | gemma4 MoE 25.2B / 3.8B active, 5:1 sliding-window | QAT Q4_0, 13.45 GiB | profile written, not probed | not measured | not run | **profile written 2026-10-20, NEVER BENCHED** — never `--swa-full` (+13 GiB KV at 128K). Note the three rejected gemma4 12B variants below |

### Retired and rejected

Kept as evidence so the same candidate is not re-tried. "GGUF deleted" means
the weights are gone from disk; the row stays.

| Profile / model | Arch | Quant, file size | Fit @128K (`--n-cpu-moe`) | tg tok/s | bench_05 weak spots | Status / role |
|---|---|---|---|---|---|---|
| `ornith-128k` (Ornith-1.0-35B) | qwen35moe A3B | Q4_K_M, 19.7 GiB | 20 (floor 18, was 24 until 2026-09-17) | ~54 @49K depth | regex never finishes (7/7 TIMEOUT); interp 2/4 at 900 s in the 2026-09-19 pairing | superseded by `ornith15-128k` (2026-09-19) — 20 GB hub blob, delete candidate |
| `qwythos` (Qwythos-9B-v2 +MTP) | qwen35 dense hybrid | Q8_0 | whole | 149 (MTP) | parser tier FAILs (9B ceiling); Xid 8 hang @69K with MTP | **RETIRED 2026-09-19** — dominated by `ornith15-9b` (same size, passes interp/perf, 95 % workflow); 18 GB (two files), delete candidate |
| `qwen38d-9b` (Qwen3.8-9B-Distill, empero-ai) | qwen35 dense | Q8_0, 9.11 GiB | whole (12.3 GiB @128K) | 71 @51K depth, pp 4429 | template FAIL 3/3 (1–4/10), interp 0/3 (9/13 then 0/13 ×2), regex 0/3, perf 3/3 (46 s); **bench_07 1/20** — `summary` 5/20 (riffs Euler/numerics off the project name), evidence 10/20, lastline 12/20 | **REJECTED** — a 9B that hallucinates from names and cannot hold a parser task; delete candidate |
| `qwen25-14b-64k-kv4` (Qwen2.5-14B-Instruct, YaRN x2) | qwen2 dense | Q4_K_M | whole, 13.1 GiB @64K; q8_0 KV does NOT fit (compute pp buffers OOM) -> q4_0 KV | 34 @49K, pp 1 420 | needle OK 1K/16K/49K; **bench_05 0/12** (5 tasks never created the file); **bench_07 0/5** (2/10 items in 4 runs); bench_08 not run (B08_OUT bug, fixed) | **REJECTED 2026-10-16** - a 2024 model that does not drive the qwen-code tool loop; ornith15-9b 10/12 + 5/5 in the same pass |
| `bonsai2-27b` (Ternary-Bonsai-2-27B, prism fork quant) | qwen35 hybrid 27B (Qwen3.8-27B shape, no MTP) | Q2_0 ternary, 7.63 GB = 2.25 bpw (ggml type 42, block 64 x 2 bit + f16 scale) | whole, 13.6 GiB @49K | 50 @49K, pp 1 475 | needle OK 1K/16K/49K; parser tier x2: template 2/2, perf 2/2, interp PASS + TIMEOUT@13/13, **regex PASS 14/14** + TIMEOUT 7/14; **bench_07 20/20, every item 20/20** (median run 242 s, max 611 s; ornith15-9b same session 17/20, all fails = evidence) | **off the candidate list 2026-10-18** (final head-to-head: parser tier on time, regex 0/2 TIMEOUT, bench_07 10/10 at 227 s median) — was CANDIDATE 10-16: matches the 35B-mc on bench_07 while whole on the card, but ~2x slower per run; needs SERVER_BIN (scratch master build; installed cbb7d52ec cannot run Q2_0). Two negative forum reports (loops, 0/4 Next.js) not reproduced here |
| `apex-mini-mtp` (mudler Qwen3.6-35B-A3B APEX-MTP I-Mini) | qwen35moe A3B | APEX I-Mini, 14.3 GB | whole @40K; spills @128K | ~190 @40K no spec; 120–133 @16K / 79–82 @120K with MTP @128K | bench_05 9/12 (template TIMEOUT 9/10, interp no file, regex TIMEOUT); **bench_07 9/20** (lastline 9, evidence 9) | **REJECTED** (2026-09-29) — base-model quality at a lower bit width; udq8kxl is as fast and passes regex. GGUFs deleted 2026-09-29 (re-download: huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF) |
| `huihui35-9b` (huihui-ai Huihui-Qwen3.5-9B-abliterated, mradermacher Q8_0) | qwen35 dense | Q8_0, 9.53 GB | whole (12.2 GiB @128K) | 80 / 70 @49K, pp 4 090 | bench_05 @1200 s stopped after 8 tasks: **3/8 on the easy/mid tier** (lru TIMEOUT 6/8, multifile TIMEOUT 3/5, intervals 10/12, fsm 12/13, codec 7/12) | **REJECTED** (2026-09-29) — "almost right, never finished": abliteration cost edit precision; same speed as ornith15-9b, far below it; GGUF + profile deleted 2026-09-29 |
| `orcasaq2-27b` (orcarouter OrcaSAQ-2-Cyber-27B-Uncensored, gated) | qwen35 **dense** 27.3B (Qwen3.8-27B-Uncensored) | vendor "OrcaSAQ2" mixed quant, 15.68 GB | no — weights alone fill the card (14.6 GB used @32K via `--fit`) | **10–11** @32K, pp 65 | not run (speed gate) | **REJECTED** (2026-09-30) — the 27B-dense ceiling again; the card's 20.5 tok/s is a 24 GB GPU; GGUF deleted |
| `kataguru-35b` (Kataguru-Sceptic-35B-A3B v2.6 APEX-Good) | qwen35moe A3B | APEX-Good | 14 (floor 12; 10 boots, fails the needle) | 70 @49K | tools 6/6; parser ×2 5/8, **regex 0/2**; bench_07 9/10, 61 s; Polish terse | **REJECTED 2026-11-04** — nothing over the leader |
| `bolt-q3kxl` (Nemotron 3.5 Bolt 74B-A9B, unsloth) | nemotron_h_moe | UD-Q3_K_XL | 80 (floor 78) | 24 @49K | tools 6/6; parser ×2 7/8; bench_07 9/10, median 395 s | **REJECTED 2026-11-04** — slower quant of the `bolt` alternate that already lost to the leader |
| `q38-14b` (Qwen3.8-14B-Instruct-Turbo) | qwen3 dense 14B | Q5_K_M | whole | 66.5 @49K | empty content on every request: tools 0/6, parser 0/8, bench_07 0/10; Polish gibberish | **REJECTED 2026-11-04** — broken GGUF/template; GGUF + profile deleted 2026-11-04 |
| `memt-35b` (brishen MemT-1.0-Nano, 35B-A4B + memory module) | qwen35moe + 120 `memt_*` tensors | Q4_K_M, 21.6 GB | — | — | does not load on b12001: "wrong number of tensors; expected 853, got 733" | **NOT RUNNABLE 2026-11-04** — needs the author's llama.cpp fork; GGUF + profile deleted 2026-11-04 |
| `gpt-oss20b-heretic` (DavidAU) | MoE 20.9B | IQ4_NL, 12.6 GiB | 1 (profile 2) | slowest of family | template 4/10, interp 0/13, regex disq. | **REJECTED** (conf ref) |
| `gemma4-fable5` (yuxinlu1 12B) | gemma4 dense | Q8_0, 12.7 GiB | ctx ≤32K only! | | 5/10; interp 1/13, regex 0/14; gen_n=1 red flag | **REJECTED** |
| `gemma4-qat` (HauhauCS 12B) | gemma4 dense | Q4_K_M, 7.38 GiB | | | 100% reproducible stream-hang after task 1 | **REJECTED** |
| `gemma4-12b-it` (unsloth base) | gemma4 dense | Q8_0, 12.7 GiB | ctx ≤65536 only! (131072 OOM) | ~55 | interp+regex 0% TIMEOUT (total parser-tier wipeout); easy/mid 10/12 clean | **REJECTED** |
| 27B dense family (5 finetunes) | qwen35 dense 65L | Q4, 16–19 GB | impossible @128K (any quant) | | | **REJECTED**, GGUFs deleted (113 GB) |
| `qwen36u-iq4xs` / `qwen36u-q4kxl` | qwen35moe | 16.5 / 20.8 GiB | 10 / 17 | | scratch FAIL / 2× TIMEOUT | **REJECTED**, deleted (38 GB) |
| GLM 5.2 (744B) | MoE | any | needs ≥245 GB RAM+VRAM | | | **REJECTED** — never fit |

### Candidates not downloaded (HF sweep 2026-10-29)

Sage-A1.5 and LFM2.5-24B-A2B were downloaded and benched 2026-10-30 (rows above); none left from this sweep.

Rejected in the same sweep without downloading: DeepSeek-V4.1-Flash-1017
(763B despite the name), Ornith-1.5-122B (measured after all 2026-11-04: UD-IQ2_XXS 8/8 at 20 tok/s, see the fleet table), Tess-4.5-27B (27B dense ceiling),
Huihui-Qwen3.8-Flash-Next-abliterated-v2 (125B base). Detail: the dated sweep
in `BENCH-archive.md`.

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
MODELS=gpt-oss20b-q8_0 ./bench_08_horizon.sh   # horizon p, p(fill), L_eff + nanoeuler vote-recoverability; --calibrate first
MODEL=/path/first-shard.gguf ./bench_11_depth_stream.sh   # pp/tg at prompt depths, client-side timing
URL=http://127.0.0.1:8080 ./bench_11_depth_stream.sh      # same sweep against any running OpenAI endpoint
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


## Where the benches stopped discriminating

**The top tier has saturated bench_05 and bench_07.** Five profiles now score
`bench_07 20/20 with every rubric item 20/20` and clear the parser tier
(`ornith15-128k-mc`, `tiel-128k-mc`, `nex25-mini`, `gsq35-mc`, `adq4-mc`).
Among those five the only axis left in the standard suite is **speed**, which
is exactly how `gsq35-mc` came to be proposed as the default: it matched the
incumbent on every quality axis measured and was 28% faster. It was then
reverted by hand (2026-10-19) because it **refuses short SECRET-CODE prompts** —
a defect no bench in the suite looks for, found by using the model.

That is the gap to close. The benches that still separate the leaders:

| Bench | What it still separates | Status |
|---|---|---|
| `bench_12_hard.sh` | 9 harder own tasks — `ornith-mc` 67% vs `gpt-oss-high` 1/9 | works, needs RUNS>=3; run it on every leader candidate |
| `bench_08_horizon.sh` | dependent-step chains per context fill — `gpt-oss20b` p~0.99 flat to 98K vs `ornith15-128k-mc` 0.95 -> 0.88 | run **once** (2026-10-16). Under-used: it is the one bench that cleanly ranked two saturated models |
| `bench_13_public.sh` | SWE-rebench / Aider subsets, paired McNemar | calibrated 2026-10-03, rarely run |

### Proposed new benches (not yet written)

Each targets a failure class that has already cost a wrong decision here, and
none of them is a speed measurement.

1. **`bench_14_tools.sh` — tool-protocol conformance.** Pre-flight the server's
   `supports_tools` flag, then drive a fixed set of tool-call shapes: nested
   object arguments, a 4 KB string argument, two parallel calls in one turn, a
   tool that returns an error the model must recover from, and a call the model
   must *decline* because the required argument is missing. Score protocol
   compliance separately from reasoning.
   *Evidence it is needed:* `bielik11b` (2026-10-25) answers as a chat bot
   because its template has no tool section and llama-server silently drops the
   `tools` array; llama.cpp #30490 reports LFM2.5-24B-A2B emitting
   `<tool_call>` XML **as content**. Both are invisible to bench_05, which just
   records FAIL. Cheapest of the four and catches a whole class.
2. **`bench_15_vision.sh` — for `mmproj` models.** Needed now: `bolt` shipped a
   1.73 GB projector and `Sage-A1.5` is a VLM (image-only, OmniDocBench 91.5 per its card;
   text side benched 10-30 = leader minus regex, GGUF deleted until this bench exists). Fixed tasks with objective
   answers: read N values out of a screenshot of a monitoring table (the XDA
   Bolt test read 48 of them), transcribe a chart's axis labels, and turn a
   screenshot into a JSON schema — scored by exact match, not by a judge.
3. **Refusal / instruction-adherence probe.** A short fixed battery: a
   SECRET-CODE retrieval prompt, a "reply with exactly this token" prompt, and a
   system prompt that contradicts the user. Pure pass/fail, seconds to run.
   *Evidence:* the `gsq35-mc` refusal cost a default swap and was found by
   accident, not by measurement.
4. **Quantization-damage probe (knowledge, not reasoning).** The archive flags
   it as an open question: a 3.5-bit file "may lose on long-horizon or
   knowledge-heavy work that bench_08 horizon would catch". A closed-book
   factual set answered at two bit widths of the same base would settle
   3-bit-vs-4-bit, which currently rests on bench_05/07 alone.

Running order for the next leader candidate: tool conformance -> refusal probe
-> bench_05 N=1 (screen) -> `bench_12_hard` RUNS>=3 -> `bench_08_horizon`
paired with the incumbent -> bench_07 RUNS=20 only if it is still alive.


## Durable lessons (apply before the next candidate)

Distilled from the archive; each one cost at least one wasted run.

**Measurement**
- **Ranking two models needs N>=20 in the SAME session.** bench_07 drifted
  80% -> 60% -> 45% PASS across three unchanged sessions, and an apparent
  quant gap at N=10-15 flipped sign at N=20. RUNS=3-5 finds hard (all-runs)
  gaps only.
- **Never conclude from a single run on the parser tier** (template, interp,
  perf, regex) — the same profile has produced PASS/FAIL/TIMEOUT/TIMEOUT.
- **Put the incumbent in the same pass**, never compare against last month's
  table.
- **Order effect:** the first arm after the others stop runs ~6% faster.
  Rotate or interleave arms before believing a 3-6% delta.
- **Objective verdicts only** — protected-file sha256 + script-side pytest.
  The agent's own claims are not evidence.

**VRAM and placement**
- **Probe `--n-cpu-moe` top-down.** Every run at a working value is a
  measurement; probing upward from an OOM gives a series of crashes.
- **Bench floor != server floor** (the server's compute buffer is bigger), and
  both drift between llama.cpp builds — re-probe after every rebuild.
- Production profile sits at **floor + 1-2**: desktop VRAM load varies by
  1-2 GB between sessions and has retroactively invalidated exact floors.
- A server that loads and answers `/health` can still die on the **first
  decode** — send one real request, and one deep (~50K) request, before
  trusting a floor.
- `--load-mode none` on every CPU-offload profile (pp 1.6-3x, tg unchanged).

**Model choice**
- **A bigger file is not better.** Q5_K_M, F16 and bigger quants have
  repeatedly lost or tied against the smaller file that still fits.
- Prefer the largest quant that **fits whole**; anything that forces new
  offload is a separate candidate needing its own proof.
- **27B dense at 128K is an architectural ceiling on 16 GB** — independent of
  finetune and quant. Five finetunes rejected, 113 GB deleted.
- **>100B models are out by rule** unless the engine does its own expert
  paging (the one exception measured: Strata).
- **Check the chat template for a tool section before benching an "agentic"
  model** — `supports_tools:false` means llama-server drops the tools array
  and the model answers as a chat bot (Bielik-11B, 2026-10-25).

**Queue hygiene**
- A GPU queue never runs next to a heavy build (an ML emerge at -j12 beside a
  bench stalled the 64 GB host). Chain on EXIT sentinels instead.
- A model the installed llama.cpp cannot run goes on a scratch build via
  `SERVER_BIN=`, unset in the EXIT trap — the scratch build lacks the
  expert-cache user patch the `-mc` profiles need.

## Archive

Every dated run, oldest first in its original wording: **`BENCH-archive.md`**
(36 sections, 2026-07-05 -> 2026-10-18). Append new dated sections there;
update the fleet table above in place.
