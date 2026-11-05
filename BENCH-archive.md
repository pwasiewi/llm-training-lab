# BENCH archive — dated runs, append-only

Every dated section ever written for `BENCH.md`, oldest format preserved.
**Never rewritten, only appended.** Split out of `BENCH.md` on 2026-10-29 so the
main file can hold the current fleet picture instead of a stack of snapshots;
the content below is byte-identical to what `BENCH.md` carried before the split.

Current fleet, task definitions, how to run, durable lessons: `BENCH.md`.

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


## Reference results — 2026-09-27: Strata (one-model engine for Qwen3.8-Flash-Next) vs llama.cpp b11086 — code read, llama.cpp baseline measured; Strata measured 2026-09-29 (end of section)

```
# model: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ2_XS (2 shards, 63.4 GiB) in /mnt/cc/strata/models/IQ2_XS
MODEL=/mnt/cc/strata/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  ./bench_11_depth_stream.sh                                       # log /var/tmp/bench11-llama-fit.log
SERVER_ARGS="--fit on -c 40960 -fa on -ctk q8_0 -ctv q8_0 -t 16 -np 1 --jinja -b 2048 -ub 2048" RUNS=1 \
  MODEL=... ./bench_11_depth_stream.sh                             # log /var/tmp/bench11-llama-fit-ub2048.log
```

**What Strata is.** `github.com/Niko1221/Strata` (created 2026-09-24, 251 stars in 3 days,
no LICENSE file despite "free and open source" — legally all-rights-reserved until one
lands; not packageable into ::pwr as is). Not a llama.cpp fork: ~18 K lines of its own
C++/CUDA written for exactly one model (Qwen3.8-Flash-Next, `qwen4exp`: 48 layers, 36
Gated-DeltaNet + 12 QSA sparse-attention, 512 experts top-10 per layer, hyper-connections,
a 51 B-parameter n-gram embedding table), linking ggml only for the i-quant CPU dot
products and the vision encoder (llama.cpp pinned at `3cf0325`, 2026-09-20). Security read:
no network code in the engine; `serve/server.py` fetches image URLs it is handed; `setup.py`
by default downloads a prebuilt binary from the author's GitHub releases (build from source
instead) and `apt`-installs build tools on Ubuntu.

**The "177 B" in the 2026-09-17 HF sweep was 125 B + the 51 B n-gram table**, and the
"out by rule" call was wrong for this model: the table (26.8 GiB, shard 2) is only
row-indexed by the last three token ids, so it stays mmap'd in the page cache, and the
IQ2_XS experts are ~36 GB — the whole thing runs on 16 GB VRAM + 64 GB RAM under stock
llama.cpp (measured below).

**What Strata does that merged llama.cpp does not** (claims from its paper + code; the
code matches the paper where checked — `generate.cpp` `adapt_every=4`, `adapt_swaps=96`,
`pcie_frac` 0.55 native / 0.2 Q2_0, a doorbell spin loop in `session.cpp`):

| Strata mechanism | its claimed gain | llama.cpp status (2026-09-27) |
|---|---|---|
| **Per-EXPERT VRAM cache**, filled from a routing profile, then every 4 rounds up to 96 of the conversation's hot experts swapped in (async, no wait) | hit rate 0.50 static → 0.72–0.78 adaptive with ~4 500 slots on 12 GB | `--n-cpu-moe`/`-ot`/`--fit` place whole layers (see below: 24 % of routed experts on GPU). Open PR **#27861** (GPU LRU cache for host-offloaded experts, decode-only, +31 % on this model on 2×3090) — measured there: *no* exploitable static skew, strong temporal locality (LRU-128 ≈ 81 % hit) |
| GPU and CPU experts of one layer computed **at the same time** (router ids through a pinned "doorbell", CPU spins on it), part of the misses DMA'd over PCIe by the copy engine | IQ2_XS 70 → 78 tok/s from the DMA share alone | ggml's scheduler runs CPU and GPU splits one after another |
| **MTP self-speculation** (the model's own draft layer, fetched as 5 GB from the BF16 checkpoint), draft kept only at p ≥ 0.5 | ×1.6–1.8 in the offloaded regime (plain 47–57 → 82–92 tok/s @4K) | open PRs **#27836 / #28243** (M3 Max: +42 %). Our 2026-09-17 finding "MTP on an offloaded MoE is a loss" (nemotron) does not transfer: here the per-token GPU part is 3.5 GB of dense weights (mixers + 1.3 GB hyper-connections + head), which a 4-token verify window amortises |
| Gather-based QSA decode (attend only to the ≤ 2 048 selected positions) | flat prefill 4K → 262K | open PR **#28213** (+50 % tg at 130K) |
| KV streaming (full KV in RAM, 32K-position working set in VRAM) → more expert slots | Q2_0 @262K 50.9 → 62.6 tok/s | none |
| AVX-512 VNNI Q2_0 kernel | "near RAM bandwidth" | open PR **#26348**; **irrelevant here** — the 5950X (Zen 3) has AVX2 only, no AVX-512/VNNI |

**llama.cpp baseline on nvhard** (RTX 5070 Ti 16 GB, 5950X 16 threads, DDR4, b11086 =
2026-09-21, which already contains Strata's pinned commit and all merged qwen4exp work:
sparse FA, hc ops, state rollback). `--fit on` @40K/q8_0 KV put the experts of layers
13–47 (+ `ffn_down` of 12) on the CPU → **12 of 48 layers' experts on the GPU, 14.6 GiB
VRAM** (desktop included). Thinking off, 256 generated tokens, client-side timing
(`bench_11`, matches llama.cpp's own `timings` to ±1 %), warm runs:

| llama.cpp config | pp 1K | pp 4K | pp 32K | tg 1K | tg 4K | tg 32K |
|---|---:|---:|---:|---:|---:|---:|
| `--fit on`, `-ub 512` (default) | 210 | 225 | 224 | 29.8 | 28.4 | 23.3 |
| `--fit on`, **`-b 2048 -ub 2048`** | 266 | **362** | **434** | 26.3 | 25.4 | 23.2 |
| *Strata's claim, IQ2_XS, RTX 5070 12 GB, 6-core AVX-512, DDR5 (not measured here)* | *332* | *463* | *495* | *82.0* | *78.0* | *65.3* |

Answers are coherent (sanity prompt, tg 26.6). `--load-mode none` was NOT tried: it would
pull the 26.8 GiB n-gram table into anonymous memory on top of ~24 GiB of CPU experts
(46 GiB available) — mmap is the only sane mode for this model.

**Reading.**

1. **Prefill: llama.cpp is close once `-ub` is raised.** Every prompt micro-batch re-streams
   the CPU-side experts (~24 GiB) over PCIe from pageable memory; 2048-token batches cut that
   4×: pp ×1.6 at 4K, ×1.9 at 32K. Strata streams in 2048-token chunks too — so its prefill
   edge is ~15–30 %, not an engine class. The fleet rule "keep `-ub` at 512" holds for the
   35B-A3B profiles (little CPU-expert traffic); for a model whose experts are mostly on the
   CPU, `-ub 2048` is the lever (tg −1–3 at 1K/4K because the larger compute buffer costs
   VRAM, equal at 32K).
2. **Decode: the claimed gap is ~3× and it is structural, not tuning.** llama.cpp serves
   24 % of routed experts from VRAM by construction (whole layers), Strata claims 72–78 %
   from fewer slots, overlaps the CPU half with the GPU half, and adds MTP ×1.7. No
   llama.cpp flag reaches that today; the three open PRs above (#27861 + #28243 + #28213)
   together would cover most of it.
3. **Expected Strata on nvhard (estimate, ±30 %, unverified):** more VRAM (16 vs 12 GB ≈ +2 800
   expert slots) and 16 cores for the arithmetic-bound i-quants push up; DDR4 (≈ 35–40 GB/s
   vs 41–52) and no AVX-512 push down. IQ2_XS is the right file here (the paper: i-quants are
   CPU-arithmetic-bound at 20 GB/s on 6 cores, and Q2_0's fast path is AVX-512-only). Guess:
   decode ~60–85 tok/s at 4K, i.e. 2.5–3× llama.cpp.
4. **Caveats on the claims.** Every published number is ONE run per cell on one Windows
   box; three days old; mostly AI-assisted commits; the parity test tree (`tests/`,
   `bench/micro/`) is not in the published source. The exactness claim is well argued
   (teacher-forced KL 0.022 vs llama.cpp; cache on/off perplexity equal within 1 SE) but
   was not reproduced here.
5. **Quality is untested on our benches.** Speed alone does not make it a fleet candidate:
   IQ2_XS of a 125 B model vs `ornith15-128k` (Q4_K_M 35 B) on bench_05/07 is the question
   that decides a role, and it can run on llama.cpp today at ~25 tok/s (bench_05 at that
   speed needs TASK_TIMEOUT ≥ 1800).

**Build notes (done 2026-09-27, operator-authorised).** The first attempt was blocked by the
auto-mode permission classifier (external code); the user then authorised it. ccache broke the
first build ("File exists") — build with `CCACHE_DISABLE=1 -DGGML_CCACHE=OFF`. Prepared on /mnt/cc/strata (437 GB free
before, ~64 GB used now): the GGUF, a clean copy of the source (`src/`), llama.cpp at the
pinned `3cf0325` (`llama.cpp/`), a venv with numpy/jinja2/regex/pyyaml/tqdm/requests/pillow/
psutil. Remaining steps, by hand:
```
cmake -G Ninja -S /mnt/cc/strata/src -B /mnt/cc/strata/build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-15 -DSTRATA_GGML_DIR=/mnt/cc/strata/llama.cpp
cmake --build /mnt/cc/strata/build --target strata -j 16
# then (STRATA_GGUF_PY=/mnt/cc/strata/llama.cpp/gguf-py): tools/iq_pack.py (IQ2_XS pack),
# tools/mtp_fetch.py fetch (5 GB range reads from Qwen/Qwen3.8-Flash-Next) → mtp_pack.py → mtp_rt.py,
# serve/server.py with the args setup.py writes (setup.py lines 881-895), then
URL=http://127.0.0.1:8080 ./bench_11_depth_stream.sh
```
`bench_11` counts `completion_tokens` from the usage block; if Strata's server sends
none, the chunk-count fallback under-counts speculative output (flagged in the `cnt`
column) — compare its tg against the engine's own log line then.

**Licence update 2026-09-30:** upstream added an MIT `LICENSE` on 2026-09-28 (commit `c0107c3`, the day
after this review; ggml MIT, web font OFL-1.1, the speed-projection vector in `data/` Qwen Community
License 1.0). The "all-rights-reserved / not packageable" note above is superseded — an ::pwr ebuild
must pin a commit at or after `c0107c3` (the local /mnt/cc/strata/src copy predates it).

**Watch list (re-check trigger for the llama.cpp side):** #27861 (expert LRU cache),
#28243/#27836 (qwen4exp MTP), #28213 (QSA gather decode). Any one merged = re-run the
two llama.cpp rows above.

### Measured 2026-09-29: Strata on nvhard vs llama.cpp b11086, same prompts

```
cd /mnt/cc/strata && ./run_bench.sh            # Strata server per config + bench_11 client mode, RUNS=3
./run_llama_ngram.sh                           # llama.cpp -ub 2048 + --spec-type ngram-simple, RUNS=3
# logs: /mnt/cc/strata/bench11-{mtp,llama-ngram}.log, engine log strata-iq2_xs-mtp.log
```

`bench_11` now prefixes every request with a unique nonce line. Without it the 09-27 Strata
runs were invalid from run 2 on: its conversation cache served 902 of 909 prompt tokens
("902 reused + 7 read"). With the nonce every request logs "0 reused". Strata footprint:
39 GB RAM used (of 62), 15.7 GB VRAM, 6 981–7 608 resident expert slots, AVX2 kernels.
Strata **cannot be served without MTP**: `serve` requires `--mtp`, and an IQ pack requires
`--spec ≥ 2` — so there is no plain-decode Strata row. llama.cpp's MTP (`draft-mtp`, merged
in #22673) cannot run on this GGUF: it carries no NextN tensors (Strata fetches its own
draft layer, `mtp/rt`). The closest pairing is llama.cpp with prompt-lookup drafting.

| engine (RUNS=3, thinking off, 256 tokens) | pp 1K | pp 4K | pp 32K | tg 1K | tg 4K | tg 32K |
|---|---:|---:|---:|---:|---:|---:|
| Strata IQ2_XS, MTP `--spec 4 --spec-min-p 0.5`, run 1 (cold) | 376 | 545 | 727 | 80.4 | 77.0 | 81.2 |
| Strata, runs 2–3 (same text, new nonce) | 600 | 911 | 990 | 88–95 | 69–83 | 73–96 |
| llama.cpp `--fit on -ub 2048 --spec-type ngram-simple`, run 1 | 241 | 379 | 421 | 26.3 | 27.1 | 21.0 |
| llama.cpp, runs 2–3 | 326–335 | 413–522 | 531–534 | 29.0–29.5 | 25.7–28.4 | 18.6–23.1 |
| *Strata's claim (5070 12 GB, AVX-512, DDR5)* | *332* | *463* | *495* | *82.0* | *78.0* | *65.3* |

**Reading.**

1. **Decode: Strata's ~3× is real on this box — about 3× at 1K/4K and 3.5–4× at 32K**
   (median of 9 Strata runs 81 tok/s vs llama.cpp 18.6–29.5). It matches the paper's
   claimed numbers, not an optimistic reading of them: nvhard's extra VRAM (≈ +2 500 slots)
   and 16 cores offset DDR4 and the missing AVX-512. MTP drafts were accepted ~70 % of the time
   (e.g. 152–170 of 196–230 per 256 tokens).
2. **Prefill: 1.4–1.9× over llama.cpp at `-ub 2048`**, above the 09-27 "15–30 %" estimate.
   Both engines get faster on runs 2–3 of the same text with no prompt reuse (nonce, "0
   reused"), most likely page cache for llama.cpp's mmap'd experts, and Strata's adaptive cache swapping
   in the experts the text routes to. Real use sits between the rows: a new document is
   run 1, a follow-up in the same topic is closer to run 2.
3. **Decode variance is large and unexplained.** A first (discarded, 2-run) sweep an hour
   earlier had two cells at ~40 tok/s (1K and 32K, run 1) with the same acceptance rate. Do
   not quote a single Strata run.
4. **llama.cpp `ngram-simple` does nothing here**: draft acceptance 0–15 %, tg equal to the
   09-27 no-spec rows. Prompt lookup only pays when the answer copies the prompt; a
   code review does not.
5. Status unchanged: no LICENSE upstream (not packageable), quality on bench_05/07 not tested,
   one model only. For a llama.cpp user the levers today are `-ub 2048` (prefill) and
   #27861 (expert cache, draft). A Qwen3.5/3.6 MTP GGUF on llama.cpp's `draft-mtp` is the
   upstream MTP test — a different model, not this comparison.

### 2026-09-29 (continued): Strata on bench_05/07; llama.cpp forks and PRs on the same model; APEX-MTP

```
/mnt/cc/strata/queue.sh, queue2.sh   # logs queue.log, queue2.log, bench11-*.log, bench05-strata.log, bench07-strata.log
# bench_05/07 drive Strata through /mnt/cc/strata/aillama-strata (a stand-in for aillama: 'env' prints
# the endpoint, 'switch' only checks /health); Strata config strata-iq2_xs-mtp128k.json
AILLAMA_BIN=/mnt/cc/strata/aillama-strata MODELS=strata-iq2xs TASK_TIMEOUT=900 ./bench_05_agentic.sh
AILLAMA_BIN=/mnt/cc/strata/aillama-strata MODELS=strata-iq2xs RUNS=20 ./bench_07_workflow.sh
```

**Strata needs two settings for agent work.** At `--max-context 40960` every qwen-code request
is refused (prompt ~24K + qwen-code's `max_tokens` 32000). And `--expert-cache auto` leaves only
191 MiB of VRAM free at 128K, so the MTP verify step dies with `verify: instantiate: out of memory`
on the first real request. Working config: `--max-context 131072 --vram-reserve-mib 1700` →
1 190 MiB free, 5 147 expert slots (of ~5 835 at the default 700 MiB reserve), no OOM over both
benches.

**bench_05 (12 tasks, 900 s budget, N=1): 12/12, every task 100 %.** Times (s): bugfix 50,
scratch 139, lru 88, multifile 109, intervals 62, fsm 77, codec 123, toposort 74, template 186,
interp 203, perf 262, **regex 279 — PASS 14/14**. For comparison: `ornith15-128k` passes regex
2/7 (670–744 s), `gpt-oss20b-udq8kxl` 3/3 (the only fleet member that reliably does), the 9B
tier 0/N. N=1 does not rank (house rule) — Strata needs the parser tier ×3 before it goes into
the fleet table.

**bench_07 (relmeta, RUNS=20): 20/20 PASS, every rubric item held 20/20** (compliance,
hallucination, tail-read, thrashing, evidence gate); run 1 took 171 s. Same ceiling as
`ornith15-128k`, until now the only perfect bench_07 in this log.

Speed during the benches, for scale: Strata MTP decodes at ~81 tok/s on this model where
llama.cpp manages 20–30, so an agent turn costs about what the 35B-A3B fleet models cost — the
125 B model is usable as a daily agent on this box. Costs: 39 GB RAM + 14.7 GB VRAM while up, one
model only, no LICENSE file upstream (not packageable), Windows-first project three days old.

**llama.cpp PR #27861 (GPU LRU expert cache, draft, head bccbacdb8), same IQ2_XS file, RUNS=3:**

| config | pp 1K | pp 4K | pp 32K | tg 1K | tg 4K | tg 32K |
|---|---:|---:|---:|---:|---:|---:|
| PR build, `--fit on -ub 2048`, no cache | 224–314 | 352–491 | 420–504 | 25.6–28.1 | 25.3–27.3 | 20.2–21.3 |
| `--fit-target 4608 --moe-expert-cache 64 --moe-expert-cache-inserts 2` | 271–285 | 434–454 | 464–469 | **33.3–35.1** | **31.7–35.5** | **25.8–26.7** |

+25–27 % decode, as the PR claims (+31 %); prefill slightly lower at 1K/4K (4.5 GB kept free
= fewer layers on the GPU). Decode-only by design, bypassed under speculative decoding. Still
~2.5× behind Strata. When it merges: `--moe-expert-cache` is worth it for every CPU-heavy MoE.

**2026-09-30: the PR ported onto current master (25747b0) — +31–60 % decode, output identical.** The PR's
base is 08-28; on master `src/CMakeLists.txt` (indentation) and the `build_moe_ffn` hook conflicted —
master had grown new SILU branches (`ggml_swiglu_clamp` for MAPLE/DEEPSEEK4/GLM5_NEXT/HY_V4/DFLASH,
a MISTRAL4 precision flag) that the cache chain must mirror exactly. Ported patch:
`/mnt/cc/strata/llama-cpp-moe-expert-cache-pr27861.patch` (12 files, +646). Test build
`/mnt/cc/strata/master_build`, same `--fit on --fit-target 4608 -ub 2048` for both arms
(`/var/tmp/pr-verify.sh`): greedy 200-token answer **byte-identical** with `--moe-expert-cache 0` and
`64`; tg 23.9–25.3 → **35.6–39.4** at 1K/4K, 22.6–23.0 → **29.6–31.3** at 32K; pp unchanged
(305/491/498 vs 304/485/506 warm). Larger than the +25–27 % on the PR's own base.

**Installed 2026-09-30:** user patch `/etc/portage/patches/sci-misc/llama-cpp-9999/0001-moe-expert-cache-pr27861-ported-to-25747b0.patch`,
rebuilt with `sudo env EMERGE_DEFAULT_OPTS="--jobs=16 --load-average=32 --quiet-build=y"
EGIT_OVERRIDE_COMMIT_GGML_ORG_LLAMA_CPP=25747b08e7a0 emerge -1 --usepkg=n sci-misc/llama-cpp` (the installed 9999 is
::stuff's; the pin keeps the patch on the verified commit; a plain `emerge -1` does nothing — `--update` in the
defaults skips an installed live package). Now build 11284. The cache is opt-in (`--moe-expert-cache N`), no
profile uses it yet. Fleet check after the jump from b11086 (`/var/tmp/fleet-check.sh`, 1K + 49K, RUNS=1 cold):
all six main profiles boot and hold 49K, VRAM 11.9–14.5 GB. Same-day comparisons show no regression
(nex25-mini 1 035/47.7 → 1 069/49.7, nex25u-mini 1 452/64.4 → 1 462/64.5 pp/tg @49K; ornith15-9b 4 502/72.9 =
09-17). ornith15-128k (pp 1 163 vs 1 416) and qwen36-128k (1 562/67 vs ~1 850/75) are lower against 09-17
numbers from b11009 — different build and desktop load, single cold run: re-measure ×2 before calling it.

**`--moe-expert-cache` on a fleet profile (2026-09-30, `/var/tmp/cache-test.sh`, log `/var/tmp/cache-test.log`):**
`ornith15-128k` (Ornith-1.5-35B-A3B Q4_K_M), profile args at 128K, `--moe-expert-cache-inserts 2`, RUNS=2:

| config | VRAM | tg 1K | tg 16K | tg 49K | pp 49K |
|---|---:|---:|---:|---:|---:|
| ncmoe 20, cache 0 (current profile) | 14.0 GB | 45.0–45.7 | 57.9–58.4 | 45.6–50.8 | 1 127–1 139 |
| ncmoe 22, cache 16 | 13.9 GB | 61.9–62.4 | 56.4–61.5 | 51.4–55.5 | 1 085–1 087 |
| ncmoe 24, cache 32 | 13.6 GB | 63.3–67.5 | 57.7–61.3 | 55.2–63.5 | 1 014 |
| **ncmoe 26, cache 48** | 13.8 GB | **70.2–77.4** | **73.1–75.6** | **67.6–68.1** | 970 |

Trading 6 expert layers for 48 cache slots per CPU layer: decode **+30–50 %** at every depth, prefill −14 % at
49K (the cache is decode-only; prefill streams more CPU layers), VRAM no higher. Still rising at 48 slots —
try ncmoe 28 / cache 64 next. Same-session baseline only (the 1K baseline of 45 is below this morning's 63:
decode varies run to run here, so read the rows against each other). Not yet in models.conf: a profile change
needs bench_05/07 unchanged (the split is exact by construction — Strata model byte-identical — but the
harness is the proof).

**ik_llama.cpp (70bd424) cannot load this file:** SIGFPE in `gguf_init_from_file` — the ISTA
GSQ-RCO IQ2_XS carries 53 tensors of upstream type 42 (`Q2_0`), which ik's type table (ends at
41) does not know. ik has qwen4exp and Qwen 3.8 MTP (its PR 2369), but needs an ik-made quant; the
smallest published one (ji-farthing IQ3_KT) is 86 GB, larger than 62 GB RAM + 16 GB VRAM.
Upstream llama.cpp still has no qwen4exp MTP (PRs #28243, #27836 open), so on this model MTP
exists only in Strata. The EasiiX standalone head (`mtp-Qwen3.8-Flash-Next-hc-Q8_0.gguf`,
4.1 GB, `-md`) is downloaded for when either lands.

**Qwen3.6-35B-A3B APEX-MTP (mudler) on llama.cpp b11086, `--fit on`, q8_0 KV, 40K, RUNS=3**
(the "65–70 t/s on an RX 9070 XT" forum claim that started this):

| file | spec | pp 1K | pp 4K | pp 32K | tg 1K | tg 4K | tg 32K |
|---|---|---:|---:|---:|---:|---:|---:|
| I-Mini 14.3 GB (whole on GPU) | none | 2 630–2 870 | 3 330–3 380 | 3 350–3 430 | **191–192** | **185–187** | 147–148 |
| I-Mini | `draft-mtp`, n_max 3 | 1 420–1 600 | 1 800–1 880 | 1 930–2 010 | 152–172 | 152–164 | 162–166 |
| I-Compact 17.3 GB (partly CPU) | none | 660–720 | 740–780 | 700–760 | 105–109 | 106–107 | 69–92 |
| I-Compact | `draft-mtp`, n_max 3 | 920–980 | 1 020–1 040 | 1 020 | 88–104 | 94–115 | 97–103 |

MTP draft acceptance 73–83 %, and still: **on a model that fits the GPU, MTP LOSES** — tg −10–18 %
at 1K/4K, prefill −45 % (the draft head rides every prompt batch); it wins only at 32K (+11 %),
where the target's own step gets expensive. On Compact it is noise at 1K/4K and +15 % at 32K.
(Compact's higher prefill with MTP on is unexplained — probably a different `--fit`
placement; not investigated.) The 5070 Ti does 190 tok/s on the Mini without any fork — the
9070 XT claim is about a model fully in VRAM, which says nothing about hybrid engines. MTP
pays where the per-token dense GPU work is large and the experts are elsewhere (Strata's case),
not on a 3B-active model that already runs from VRAM.

### 2026-09-29 (continued): 128K context — Strata vs Qwen3.6-35B-A3B APEX-MTP, speed + needle

```
/mnt/cc/strata/queue3.sh     # log queue3.log; per-run logs ctx128-<engine>-{speed,needle}.log
URL=... DEPTHS=16384,65536,120000 RUNS=2 NGEN=256 ./bench_11_depth_stream.sh            # speed
URL=... DEPTHS=16384,65536,120000 RUNS=2 NGEN=48 NEEDLE=1 ./bench_11_depth_stream.sh     # retrieval
# Strata: strata-iq2_xs-mtp128k.json (--max-context 131072 --vram-reserve-mib 1700)
# llama.cpp b11086: --fit on -fa on -c 131072 -ctk q8_0 -ctv q8_0 -np 1 [--spec-type draft-mtp --spec-draft-n-max 3]
```

`NEEDLE=1` (new in bench_11) plants `// SECRET-CODE: <random hex>` a quarter of the way into the
prompt, fresh per run, and checks the answer contains it. **Needle: 30/30 OK** — every engine,
every depth up to 115–120K tokens, both runs. 128K works on all five configurations; VRAM
14.8–14.9 GB after load for the llama.cpp ones.

| engine @ `-c 131072` (runs 1–2) | pp 16K | pp 64K | pp 120K | tg 16K | tg 64K | tg 120K |
|---|---:|---:|---:|---:|---:|---:|
| Strata IQ2_XS 125 B, MTP | 509–949 | 747–913 | 746–895 | 32–50 | 47–50 | 50–55 |
| APEX I-Mini, no spec | 885–907 | 842–874 | 814–821 | 108–122 | 94–95 | 76.5 |
| APEX I-Mini, `draft-mtp` | 1 402–1 411 | 1 248–1 262 | 1 166–1 170 | 120–133 | 101–107 | 79–82 |
| APEX I-Compact, no spec | 620–627 | 601–606 | 581–586 | 82–84 | 69–70 | 58–59 |
| APEX I-Compact, `draft-mtp` | 772–803 | 751–755 | 715–720 | 90–92 | 78–83 | 72–75 |

Time to first token for a 120K prompt: Strata 129–154 s, Mini+MTP 103 s, Compact+MTP 168 s.

**Reading.**

1. **At 128K the Mini no longer fits the GPU**: `--fit` spills experts to make room for the KV
   cache, so prefill drops from 2 600–3 400 (40K context) to 815–905 and decode from ~190 to
   76–122. Context size, not model file size, decides "whole in VRAM" here.
2. **With experts on the CPU, MTP wins again**: tg +5–10 % on the Mini, +10–25 % on Compact,
   growing with depth — the same rule as Strata's case (MTP pays when the target step is expensive).
3. **MTP raises prefill** (+55 % Mini, +23 % Compact), as it did for Compact at 40K. Unexplained;
   the server log does not print `--fit`'s placement, so a different layer split is the guess,
   not a finding. Worth one run with the placement pinned (`-ncmoe`) before trusting it.
4. **Strata at 128K decodes ~50 tok/s**, down from ~81 at 40K (fewer expert slots after the
   1 700 MiB reserve, larger KV working set — likely, not isolated). A 35B-A3B at 120K on plain
   llama.cpp with MTP (72–82) now decodes faster than the 125 B Strata; prefill is comparable.
   The trade is model size (125 B vs 35 B) against speed — bench_05/07 at N≥3 decide whether the
   bigger model earns it.

### 2026-09-29 evening: Strata parser tier ×3 (incomplete) + APEX Mini+MTP agentic

```
/mnt/cc/strata/queue5.sh   # Strata 128K config, TASKS=template,interp,perf,regex TASK_TIMEOUT=900, 3 runs
/mnt/cc/strata/queue4.sh   # APEX I-Mini + draft-mtp @131072: bench_05 full x1, bench_07 RUNS=20 (driven via aillama-strata, STRATA_URL=:8120)
```

**Strata, parser tier (logs parser-strata-run{1,2,3}.log):**

| task | run 1 | run 2 | run 3 | + 09-29 full run | total |
|---|---|---|---|---|---|
| template | PASS 219 s | PASS 159 s | PASS 126 s | PASS 186 s | **4/4** |
| interp | PASS 205 s | PASS 297 s | PASS 249 s | PASS 203 s | **4/4** |
| perf | TIMEOUT (6/6 scored) | PASS 415 s | FAIL 5/6 (440 s) | PASS 262 s | **2/4** |
| regex | PASS 553 s | PASS 349 s | — machine froze | PASS 279 s | **3/3** |

Against `ornith15-128k` (N=7: template/interp/perf 7/7, regex 2/7): Strata is clearly stronger on
regex (3/3, 279–553 s vs 2/7 at 670–744 s) and weaker on perf (the hard time budget; one TIMEOUT
with every test green = too slow, one near-miss). Too few runs for a ranking; the pattern is
"best reasoner in the log, slowest finisher".

**The run-3 regex task never finished: the host froze at 19:21.** Strata's generation stalled at
2 809 thinking tokens ~60 s into the task (token count flat for 250 s) and the whole machine
hung 5 min later; the previous boot's journal has no OOM, NVRM or hung-task line. Contributing
for certain: a 16-job `cargo build` started at ~19:20 beside Strata's 39 GB with no swap (an
operator-side mistake — nothing heavy runs beside Strata). Root cause of the stall itself: not
determined. Rule for the next Strata run: nothing else heavy on the box, and watch `free -g`.

**Earlier queue bug (fixed):** `cd $S/src && python server.py … &` backgrounds the whole `cd && …`
list, so `$!` is a subshell and `kill $spid` never reaches the server. queue3's Strata server
stayed up orphaned, queue4's Strata start then failed on "port 8080 already in use", and all 12
parser tasks "failed" in 22 s against the dead engine (discarded). Now `cd` first, then start;
queue5 refuses to start when :8080 is busy.

**APEX Qwen3.6-35B-A3B I-Mini + `draft-mtp` @131072 (base Qwen3.6, not a finetune):** bench_05
**9/12** — easy/mid 8/8 in 43–72 s, perf PASS 288 s, template TIMEOUT 9/10, interp FAIL (no file),
regex TIMEOUT. bench_07 **9/20 PASS**: lastline 9/20, evidence 9/20, everything else 20/20
(runs ~82 s). One better than `qwen36-128k` (8/12) at ~1.5× its speed → at most a faster
general alternate; not a candidate against Ornith 1.5 or `gpt-oss20b-udq8kxl`.

### 2026-09-29 night: Huihui-Qwen3.5-9B-abliterated — rejected on the easy tier

```
# profile huihui35-9b = ornith15-9b's shape (-ngl 99 -c 131072 q8_0 KV), GGUF ~/models/huihui35-9b
MODELS=huihui35-9b TASK_TIMEOUT=1200 WORKROOT=/tmp/bench-agentic-huihui ./bench_05_agentic.sh   # stopped after task 8
```

Most-downloaded abliterated Qwen 3.5 (mradermacher GGUF 61K downloads, original 143 likes),
tested because of good community opinions. Preflight fine: whole in VRAM at 128K (12.2 GiB),
49K prompt at pp 4 090 / tg 70 (ornith15-9b: 4 491 / 73), 49K needle found. bench_05 at the
1200 s budget where ornith15-9b scores 11/12: bugfix PASS 21 s, scratch PASS 49 s, toposort
PASS 34 s — then lru TIMEOUT (6/8), multifile TIMEOUT (3/5), intervals FAIL 10/12 (1 100 s),
fsm FAIL 12/13, codec FAIL 7/12 (1 082 s). **3/8 on the tier every strong model sweeps**; stopped
there (operator's call — the parser tier and bench_07 could only add hours). Failure shape:
correct first drafts, then fix-test loops that never close the last one or two tests — the
known cost of abliteration (removing the refusal direction also blunts precise editing).
Community "good" opinions are about uncensored chat, not agentic coding.

### 2026-09-30: OrcaSAQ-2-Cyber-27B-Uncensored — rejected on speed (dense 27B, again)

Gated repo (`hf` login 42n4 + "Agree"), one file, 15.68 GB, vendor's proprietary "OrcaSAQ2"
sensitivity-aware mixed quant of `orcarouter/Qwen3.8-27B-Uncensored` (claims +0.8 % PPL vs BF16 —
unverified). The forum figure "20.5 tok/s on 16 GB" is, per the card, a **24 GB GPU** with 14.9 GB
peak at its example `-c 32768`. Here, `llama-server --fit on -c 32768 -fa on -ctk/-ctv q8_0`:
14.6 GB VRAM, **pp 65 / tg 11.4 at 1K, pp 65 / tg 10.0 at 16K**, 16K needle OK. The 64K run was
cancelled by the operator — a 27B dense whose weights fill the card cannot reach agent context
at a usable speed (fleet rule: 27B dense @128K is architecturally impossible on 16 GB).
orcarouter's other GGUFs: GLM-5.3-Flash 321B, DeepSeek-V4-Flash 284B, Qwen3.8-Flash-Next-Uncensored
177B (too big) and **Nex-N2.5-mini-Uncensored** (qwen35moe 34.7B) — tested next, beside the
original `nex-agi/Nex-N2.5-mini`.

### 2026-09-30: Nex-N2.5-mini, original vs uncensored — bench_05 N=1: 10/12 each

```
/var/tmp/nex-queue.sh   # log /var/tmp/nex-queue.log; bench_05 logs /var/tmp/nex{,u}-bench05.log
MODELS=nex25-mini  TASK_TIMEOUT=900 WORKROOT=/tmp/bench-agentic-nex  ./bench_05_agentic.sh
MODELS=nex25u-mini TASK_TIMEOUT=900 WORKROOT=/tmp/bench-agentic-nexu ./bench_05_agentic.sh
```

`nex-agi/Nex-N2.5-mini` (850 likes, Apache-2.0) is a 34.7B qwen35moe post-trained for long-horizon
agentic work (coding, tools, self-correction). Original = bartowski Q4_K_M 22.3 GB, profile ncmoe 21
(ornith15-128k's 20 + one step for the larger file; 14.0 GiB loaded); uncensored = orcarouter IQ4_XS
18.7 GB, ncmoe 16 (qwen36-128k's shape; 14.2 GiB). Preflight 49K: original pp 1 035 / tg 47.7,
uncensored pp 1 452 / tg 64.4; needle OK on both.

| task | nex25-mini | nex25u-mini |
|---|---|---|
| easy/mid 8 | 8/8, 57–186 s | 8/8, 37–126 s |
| template | FAIL 4/10 (761 s) | **PASS 10/10** (725 s) |
| interp | PASS 13/13 (688 s) | TIMEOUT, 13/13 scored |
| perf | PASS (204 s) | PASS (643 s) |
| regex | TIMEOUT, **14/14 scored** | TIMEOUT 5/14 |

Both 10/12 at N=1, in ornith15-128k's first-run range (11/12 on 09-17), and both misses on the
parser tier include a "solved but late" (regex 14/14 at the cutoff; interp 13/13) — the kind a
repeat can flip. Unlike huihui35-9b, the orcarouter uncensoring did not damage editing precision;
the IQ4_XS/ncmoe 16 variant is also ~35 % faster. N=1 does not rank: next is the parser tier ×3
and bench_07 RUNS=20 for both, with ornith15-128k in the same session.

**Stage 2 (09-30 08:51 → stopped 11:31 by the operator, to continue at night)** — parser tier,
`/var/tmp/nex-stage2.sh`, logs `/var/tmp/nex-parser-run{1,2}.log`, TASK_TIMEOUT=900:

| task | nex25-mini | nex25u-mini | ornith15-128k |
|---|---|---|---|
| template | PASS 432 s, PASS 518 s | PASS 810 s, PASS 571 s | PASS 307 s |
| interp | PASS 824 s, PASS 434 s | PASS 336 s, (run 2 cut) | PASS 354 s |
| perf | PASS 130 s, TIMEOUT 5/6 | PASS 187 s | PASS 135 s |
| regex | TIMEOUT 14/14, TIMEOUT 14/14 | TIMEOUT 14/14 | **PASS 588 s** |

Operator's reading: "no revolution but quite solid". Correctness equals the incumbent (every task
100 % scored), completion time does not: both Nex variants solve regex every time (14/14 ×4 incl.
stage 1) but never inside 900 s, and the "faster" uncensored IQ4_XS (tg 64 vs 53) is not faster
per task — it thinks longer. Still to run: round 3 and bench_07 RUNS=20 for all three.

### 2026-10-01 night queue: `-ub 2048` + expert cache on ornith, Strata head-to-head, Nex finished

```
~/Claude/llm/night/night-queue.sh   # log + summary in ~/Claude/llm/night/, agent dirs /mnt/cc/strata/night-work
# llama.cpp b11284 + ported PR #27861 (user patch), 128K, q8_0 KV, --load-mode none; bench_11 RUNS=2
```

**1. ornith 35B (Ornith-1.5-35B-A3B Q4_K_M) sweep, same session, @49K (mean of 2 runs):**

| ncmoe / cache / ubatch | pp 1K | tg 1K | pp 49K | tg 49K |
|---|---:|---:|---:|---:|
| 20 / 0 / 512 (profile until 09-30) | 1 020 | 60.0 | 1 152 | 54.3 |
| **20 / 0 / 2048** → `ornith15-128k` | 1 616 | 60.9 | **2 691** | 55.1 |
| 22 / 0 / 2048 | 1 556 | 57.5 | 2 590 | 51.0 |
| 26 / 48 / 512 | 877 | 78.7 | 958 | 66.6 |
| 26 / 48 / 2048 | 1 371 | 77.5 | 2 348 | 67.8 |
| 28 / 64 / 512 | 819 | 84.2 | 908 | 73.5 |
| **28 / 64 / 2048** → `ornith15-128k-mc` | 1 271 | 77.7 | **2 239** | **71.3** |

`-b 2048 -ub 2048` is the largest single gain this fleet has seen from a flag since `--load-mode none`:
prefill ×2.3 at 49K, decode unchanged, and it still fits at the same ncmoe. It also removes the expert cache's
prefill penalty (the cache is decode-only; more CPU layers used to cost −14 % pp). Net, the cache profile vs the
09-30 profile: **pp ×1.9, tg +31 %** at 49K. Both picks written into models.conf and boot-checked
(backup `models.conf.bak-night-20261001`).

**2. Strata vs llama.cpp + PR on the Strata model (Qwen3.8-Flash-Next IQ2_XS), 128K, same session:**

| engine | pp 16K | pp 49K | tg 1K | tg 16K | tg 49K | needle @49K | VRAM / RAM free |
|---|---:|---:|---:|---:|---:|---|---|
| llama.cpp `--fit-target 4608`, cache 0 | 236 | 263 | 18.1 | 16.5 | 16.6 | OK | 11.1 GB / 52 GB |
| llama.cpp `--fit-target 4608`, cache 64 | 478 | 465 | 34.8 | 31.7 | 27.2 | OK | 15.7 GB / 52 GB |
| **Strata 0.1.27 (packaged), MTP** | **1 676** | **1 777** | **79.9** | **82.4** | **78.0** | OK | 14.7 GB / 22 GB |

Strata keeps a 2.3–2.9× decode and 3.5–3.8× prefill lead over the best llama.cpp setting. Caveat: the cache-0 arm
shared `--fit-target 4608` and left 4.5 GB VRAM idle, so it understates plain llama.cpp (09-30: 23–25 tg at
`--fit on`); the cache's real gain on this model is the 09-30 +31–60 %, not ×2.

**3. Does the cache profile keep ornith's scores? Yes.** Parser tier ×2, same pass:

| profile | template | interp | perf | regex |
|---|---|---|---|---|
| `ornith15-128k-mc` | 2/2 (107, 328 s) | 2/2 (191, 154 s) | 2/2 (201, 159 s) | **2/2** (657, 353 s) |
| `ornith15-128k` (ub 2048) | 2/2 | 2/2 | 1/2 (TIMEOUT 6/6 scored) | **2/2** (602, 583 s) |

bench_07 RUNS=20: `ornith15-128k-mc` **20/20, every rubric item 20/20** — the second perfect bench_07 in this
log, same GGUF as the first. → **`ornith15-128k-mc` is the serious-agentic default**; `ornith15-128k` stays as the
no-patch fallback (llama.cpp without the user patch rejects `--moe-expert-cache`).

**4. Nex-N2.5-mini finished.** Round 3 parser tier: `nex25-mini` 2/4 (interp FAIL 12/13, regex TIMEOUT 0/14),
`nex25u-mini` 2/4 (interp + regex 100 % scored but late). bench_07 RUNS=20: `nex25-mini` **20/20 every item**,
`nex25u-mini` 18/20 (`summary` 18/20 — invents a project description twice). Verdict: alternates; workflow
discipline equals ornith, parser tier slower and less reliable.

**5. qwen36-128k re-measure (×2):** pp 1 548 / tg 67.2 @49K (1K: 1 383 / 79.3) — the 09-17 1 850 / 75 is not
reproduced; the drop after b11009 → b11284 is real for this profile (see the ub 2048 sweep below).

**6. `-b/-ub 2048` on the other CPU-offload profiles (`night/ub-sweep.sh`, same session, @49K, mean of 2):**

| profile | before (ncmoe, ub 512) pp / tg | after (ub 2048) pp / tg | ncmoe |
|---|---:|---:|---|
| `qwen36-128k` | 1 544 / 67.1 | **3 062** / 63.1 | 16 → 18 (16 + ub 2048 dies on the first request) |
| `qwen38d-128k` | 1 198 / 55.9 | **2 608** / 51.8 | 20 → 22 (same) |
| `nex25-mini` | 1 074 / 51.0 | **2 596** / 49.6 | 21 |
| `nex25u-mini` | 1 456 / 67.1 | **3 159** / 66.7 | 16 |

Prefill ×2.1–2.4 everywhere. The ub 2048 compute buffer is ~0.9 GB larger: two profiles needed +2 ncmoe and pay
−6–7 % decode for it; an agent turn (a few K new tokens + a few hundred generated) still comes out ahead, and a
first 24K-token qwen-code prompt drops from ~16 s to ~8 s. All four written to models.conf (backup
`models.conf.bak-ub-20261001`). A server that boots at ub 2048 can still die on the first request — the sweep's
1K row is the check, `/health` is not.

### 2026-10-01: PR #27861 thread review, fence added, branch on the fork; cache output is NOT byte-identical on ornith

Thread review (42 comments): no reported correctness bug reaches our setup (single CUDA GPU, 1-token decode, no
draft model). The "tables mutate mid-prefill" bug needs the cache to serve prefill batches (`MAX_TOKENS` > 1
variants); SYCL/Vulkan/draft-model/duplicate-slot bugs are other backends or batch > 8. One applies in principle:
a missing fence before `llama_moe_cache_step()` publishes uploads → added pjsgsy's
`ggml_backend_sched_synchronize(sched)` (his v2/v3). All posted variants: `/mnt/cc/strata/pr27861-variants/`.

Final patch = port + fence: branch https://github.com/pwasiewi/llama.cpp/tree/moe-expert-cache-pr27861-port
(commit e1295c7 on 25747b0), file `llm/patches/llama-cpp-moe-expert-cache-pr27861-fence.patch`, installed as the
llama-cpp-9999 user patch (`…-ported-to-25747b0-fence.patch`; the unfenced v1 kept as `…-v1-nofence.patch`).
Speed unchanged (ornith 28/64/2048 @49K: pp 2 272–2 300, tg 62.6–67.6 vs installed unfenced 2 254 / 68.8).

**Correction to 09-30 "output byte-identical":** that held for the Strata model (200 greedy tokens). On ornith with
the same `--n-cpu-moe 28` in both arms, cache 0 vs 64 diverges at byte ~226 into another plausible answer, and the
cache-on answer even changes between two identical requests (the cache fills in between). CPU vs GPU expert math
rounds differently (also reported in the thread). Quality evidence is the bench results, not identity.

**Intermittent CUDA fault with `-ub 2048`:** Xid 31 "illegal memory access", FAULT_PDE at the same address
0x3_03200000 three times — qwen36 ncmoe 16 + ub 2048 (cache off), qwen38d ncmoe 20 + ub 2048 (cache off), and the
fenced build at 28/64/2048 on its 2nd request (then 2/2 clean repeats; the installed build 1/1 clean; the whole night
of ub 2048 runs clean). The two cache-off cases are why the ub sweep reported "dies on the first request" and moved
those profiles up 2 ncmoe — it happens near the VRAM limit. Not cache-related; watch for it in daily use, and drop to
`-ub 1024` on a profile where it shows up.

**A/B: our port + fence vs pjsgsy v3 (admission gate), same session, RUNS=3 (`night/v3-ab.sh`, log `night/v3-ab.log`).**
v3 = `pr27861-variants/rtx3060_v3_0928.patch` on 25747b0 (fence + TacoTakumi draft fix + windowed admission gate:
an expert is uploaded only after `--moe-expert-cache-admit` uses within a `--moe-expert-cache-window` half-life;
defaults 3 / 16). Same args otherwise (`--moe-expert-cache 64 --moe-expert-cache-inserts 2`, `-ub 2048`).

| tg (mean of 3) | ornith 1K | ornith 16K | ornith 49K | Flash-Next 1K | Flash-Next 16K |
|---|---:|---:|---:|---:|---:|
| our port + fence | 79.4 | 73.1 | 66.0 | 28.9 | 26.5 |
| **v3, admit 3 (default)** | **92.4** | **85.3** | **74.0** | 29.5 | 26.5 |
| v3, admit 1 (gate off) | 76.5 | 75.1 | 67.9 | 25.4 | 23.6 |

Prefill identical across arms (ornith ~2 300 @49K; Flash-Next first-run cold-cache rows aside). The gate is worth
**+12–17 % decode on ornith** and nothing on the 512-expert model; v3 without the gate is ~12 % slower than our
port on Flash-Next, so keep the default. No illegal-memory faults in any of the six servers. → **v3 is now the
llama-cpp-9999 user patch** (`…-pjsgsy-v3-0928.patch`); our fenced port stays on the fork branch as the reference.
Cache hit/upload stats are printed only at clean shutdown, so none were captured here.

## Night queue 2 — 2026-10-01 → 10-02: research fan-out, the expert cache on every sibling, the Xid 31 root cause, gpt-oss effort

Plan and reports: `night/research/01…09-*.md` (Strata upstream, bench gaps, cache code, prototypes, llama.cpp upstream,
PR #28414 port, ik_llama, **Xid 31 root cause**, forks/engines). Drivers: `night/night2-queue.sh` (P0–P6),
`night2b-queue.sh` (P7–P13, resumed after a derail), `post-queue.sh`, `day-queue.sh`, `fix-queue.sh`, `prefetch-queue.sh`,
`forks-queue.sh`, `prefetch2-queue.sh`, `tail-queue.sh` (chained on EXIT sentinels, all detached). Helpers:
`night2-lib.sh`. Running log of every number: `night/night2-results-draft.md`. Same-session rule, pick rules and
bench_07 RUNS=20 as before.

### 1. The "dies on the first request" fault is an upstream bug, fixed with a second user patch
CUDA MMQ `mul_mat_id` (ggml-cuda/mmq.cu) pads the quantized-activation scratch by `J_max(ne11)` rows; in the MoE path
`ne11 == 1`, so the pad is **zero** and the kernel's last J-wide tile reads up to 18 KB past the allocation. It is fatal
only when that pool allocation ends within 18 KB of the VMM pool's mapped end, which on the 35B-A3B class is exactly a
**1020-token ubatch** (bench_11 prompts are 1019–1028 tokens → "~1 start in 8"). All 20 Xid 31 faults in dmesg are
`FAULT_PDE VIRT_READ` at pool base + 18 MiB. Upstream: issue #27792, fix PR #27044 (open since 08-13). Our patch
`/etc/portage/patches/sci-misc/llama-cpp-9999/0002-cuda-mmq-mul-mat-id-pad-src1-q8-1-issue27792.patch` (PR #27044's
hunk + `ids_dst` pad, clamped to 8 rows; copy in `llm/patches/`). **Proof** (`fix-queue.sh`): deterministic reproducer
`-b 2048 -ub 907` (128 B of slack) + one ~1K prompt: 3/3 cold starts died on build 11284, **0/3 after the rebuild,
0/8 at the production `-ub 2048`**; speed unchanged (49K pp 2292 / tg 70.4). Not the ubatch size (probe: `-ub 2048`
1/8, `-ub 1024` 1/8), not the cache (cache-off configs died), not threads. Consequence for results: every fast
bench_05 FAIL "<file> not created" whose agent.log ends with `API Error: terminated (UND_ERR_SOCKET)` was the server
dying, not the model (qwen38d-128k-mc round 1, P7 default round 2 interp, D3 arms) — reruns on the fixed build in tail-queue.

### 2. Expert cache on every CPU-offload sibling (P3, same session, incumbent = profile + ub 2048, 1K/49K ×2)

| profile | incumbent 49K pp / tg | cache 48 (ncmoe+6) | cache 64 (ncmoe+8) | written `-mc` line | needle 16K/64K/120K |
|---|---|---|---|---|---|
| qwen36-128k (nc 18) | 3024 / 58.7 | **2655 / 74.7** | died (bug) | nc 24, 48 slots | 3/3 |
| nex25u-mini (nc 16) | 3131 / 64.2 | **2720 / 77.8** | 2605 / 78.1 | nc 22, 48 | 3/3 (90 / 86 / 81 tg) |
| qwen38d-128k (nc 22) | died (bug) | **2283 / 68.5** | died | nc 28, 48 | 3/3 |
| nex25-mini (nc 21) | 2583 / 48.8 | 2282 / 63.0 | **2200 / 67.8** | nc 29, 64 | 3/3 |
| tiel-128k (nc 21; +ub 2048) | 2593 / 49.5 | 2272 / 63.0 | **2190 / 67.5** | nc 29, 64 | 3/3 |
| glm-flash-128k (nc 22) | OOM at load | 1107 / 27.2 at nc 28 | — | no (09-17: 42 tg) | — |

+21–39 % decode for −12–16 % prefill, the same trade accepted for Ornith. The driver's pick rule (pp ≥ 0.90×) is
wrong across cache-on/off groups and rejected all of them; lines were written by hand. Rule for next time: a cache arm
wins at tg ≥ +15 % with pp ≥ 0.80×. Quality so far (parser tier ×1, pre-fix build): **tiel-128k-mc 4/4** (regex 14/14
in 413 s); nex25u-mini-mc 2/4 (interp 13/13 + regex 13/14 scored but late) and **bench_07 17/20**; nex25-mini-mc 2/4
(perf TIMEOUT 0/6, regex late); qwen38d-128k-mc 1/4 (two bug casualties); qwen36-128k-mc 3/4 (regex TIMEOUT 3/14) and
bench_07 12/20 → qwen36 stays the speed pick. Rounds 2 run in tail-queue on the fixed build.

### 3. Default `ornith15-128k-mc`: nothing beats 28/64 at ub 2048 (P2, 1K/16K/49K ×2)
Incumbent 2240 / 71.3 (16K tg 80, 1K 85). Within noise: inserts 4 (70.3), admit 2/4 (67.6/67.3), window 32 (67.8),
`-tb 32` (68.7), KV q4_0 (69.9), `--cpu-range 0-15 --cpu-strict 1` (70.4), `--poll 0` (68.3). OOM at load: 28/96,
32/128, KV f16; 30/96 crashed in cublas at the first request. Dead (bug): inserts 1, -t 12/24/32, ub 4096 @nc 32,
GGML_CUDA_GRAPH_OPT=1. Prototypes (branch `night2-proto`, /mnt/cc/strata/llama-proto, research/04): A batched cache
table writes = neutral (70.8 / 2274); **B D2D source for the prefill expert copies = pp +9 % (2438) but tg −9 % (64.6)**;
B with `LLAMA_MOE_CACHE_VERIFY=2` completed every row (exact). PR #28414 prefetch port (`night2-prefetch`, research/06,
built as build 6 with the mmq fix), clean A/B on the fixed build (n=3, same session): incumbent 2268 / 72.8;
B alone 2483 / 74.0; **B + `--prefetch-experts-slots 3` = 2782 / 74.8 at 49K, 2912 / 86.7 at 16K (+23 % prefill, decode
equal)**; 2 slots 2790 / 72.4; 3 slots at ncmoe 30 2706 / 70.0. The earlier "B costs −9 % tg" was bug-era noise.
Candidate user patch 0003 = `llm/patches/llama-cpp-moe-cache-b1-e-prefetch-pr28414-night2.patch` (applies on 0001),
gated by `pf-queue.sh` (parser ×1 + bench_07 RUNS=20 on that binary via aillama `SERVER_BIN`). Speculative decoding on
the offloaded 35B is closed: 0.8B draft with the cache 86 → 32–40, Qwen3.6 MTP head 53 → 47–48 (forks-queue K3b). Parser round 3 for the default: **4/4, regex 14/14 in 396 s** → N=3 = 12/12 on the parser tier.

### 4. Quality levers
- **gpt-oss20b-udq8kxl `--reasoning-effort`**, bench_07 RUNS=20 ×2 sessions: low 4/20 (median 15 s), default 12/20 →
  9/20 (26 s), **high 17/20 → 15/20** (42 s, max 70 s). The "base-model trait" (lastline/evidence) is effort-dependent.
  `gpt-oss20b-udq8kxl-high` = **DEFAULT fast coding tier** (2026-10-02); the plain line stays as the fallback.
- **qwen-code sends no sampling fields** to this host, so every bench in this file ran at llama-server's defaults
  (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05), never greedy. `night/qwen-sampling.sh on|off` writes
  `model.generationConfig.samplingParams` into `~/.qwen/settings.json` (verified via `/slots`). A/B on the default
  profile (parser tier): server default 4/4 then 2/4 (one bug casualty, one regex TIMEOUT), Qwen card 0.6/20 4/4,
  greedy 4/4 → no separation at N=1; more rounds in tail-queue. Thinking stays on by default.
- **`--reasoning off`** on the default: parser 3/4, **bench_07 11/20** (20/20 with thinking) → never for agents.
- **`--no-reasoning-preserve`**: parser 4/4 with regex in 273 s, compactions unchanged (0) → round 2 in tail-queue before adoption.
- Ornith GGUFs carry only the four NextN norm/projection tensors → `draft-mtp` cannot load. ggml-org's Qwen3.5-0.8B as a
  draft for the 9B (same vocab): 83.8 → 65–69 tok/s at 51–60 % acceptance → no. 35B + draft/MTP arms rerun in forks-queue.

### 5. Strata 0.1.27 → 0.1.31 (sci-ml/strata::pwr) and its knobs
Bump (ebuild unchanged but the version; cancel-fix #183/#205, truncated tool calls #211/#231, short-prompt prefill
streaming, AVX2 expert prefetch, `reasoning_budget_tokens`); smoke 78 / 68 tok/s at 1K / 16K. Knob sweep (two sessions,
1K / 16K, pkg = 78–80 / 81–83 tg, pp 737 / 1676): per-request `strata_tune` **pcie_frac 0.35 + spec_min_p 0.7 → 82–83 tg
vs 70–72 at the 0.55/0.5 default** (min-p 0.3 hurts); `--kv-resident 32768` 84–86 / 82–85; `--vram-reserve-mib 1000`
85–87 / 82–85; `--prefill auto` 75 / 80 but **16K pp 2822–2828 (+69 %)**; `--spec 3` 77 / 79, `--spec 6` 74 / 76;
`--pool-workers 12`, `STRATA_GR_V3=1`, `STRATA_PREFILL_RING=8` within noise (ring8 pp −15 %). Cache stayed 4926–4930
slots in every arm. **Combined config** (`--pcie-frac 0.35 --spec-min-p 0.7 --vram-reserve-mib 1000 --kv-resident 32768 --prefill auto`)
vs stock, same session: 49K **pp 2805 / tg 97.2 vs 1767 / 76.2**, 16K tg 96.7 vs 83.6; parser rounds 3–4 on it **8/8**
(regex 210 s and 229 s) → written into `strata-pkg.json` 10-02 (backup kept). `strata-iq2xs`: N=6/6/5 template/interp/regex
all PASS, perf 4/6; the fastest serious-agentic option, pending a bench_07 RUNS=20 on the new config.

### 6. Dropped after research
ik_llama.cpp (no cache equivalent, Flash-Next unloadable: Q2_0; decode −20–30 % expected); KTransformers, exllamav3,
vLLM/SGLang offload, academic systems (research/09). Two llama.cpp forks built on the same cache PR (neurall `release`,
thecodacus `perf`) get a build + A/B in `forks-queue.sh`. Rebasing the pin is neutral for the 35B class; worth it only
for Flash-Next (#29751 qwen4exp fix +13–51 % tg, #29761 MTP).

### 7. Process lessons
A sed range `/^fn()/,/^}/` over a file with one-line functions ran the driver's wrap-up inside my shell and derailed
the queue at 00:05 (the control model's regex FAIL at 397 s is that, not a model result). Helpers now live in
`night/night2-lib.sh` (definitions only). Chained detached scripts (`setsid nohup`, waiting on EXIT sentinels) survive
session limits and are invisible to the statusline (`ps -eo pid,args | grep queue.sh`). **Never SIGTERM a waiting
queue script**: bash defers the signal until `sleep` returns and then runs the EXIT trap, which started the 9B server
beside a running A/B and killed its arms at load; use `kill -9` on a script that is only waiting.

## bench_12 / bench_13 first calibration (2026-10-03)

New harder benches (research/22-26 in `llm/night/research/`): `bench_12_hard.sh` (9 own tasks, bwrap sandbox) and
`bench_13_public.sh` (SWE-rebench screen 12 + Aider Polyglot hard 20). bench_12: ornith15-128k-mc 18/27 (3 runs),
tiel-128k-mc 7/9 run 1, gpt-oss20b-udq8kxl-high 1/9. Aider hard 20: ornith 13/20, tiel 11/20 (p = 0.69). In band;
ornith vs tiel = tie within noise, ornith stays default on speed. tracespec/reconcile budgets to raise to ~2400 s.
Details: `llm/night/REPORT-night2-2026-10-02.md` §14.

## 3-LLM queue — 2026-10-16 → 10-17: Qwen3.8-27B UD-Q3_K_XL, Swift-1.5 27B (+MTP), Ornith-1.5 GSQ 3-bit

`llm/night/qwen38-queue.sh` (logs `llm/night/qwen38-*.log`, work dirs `/mnt/cc/strata/qwen38-work/`). Per model:
download + sha256, deep preflight (bench_11 needle at 1K/16K/49K ×2), bench_05 parser tier ×2 (TASK_TIMEOUT 1200
for the 27Bs, 900 for the MoE), bench_07 RUNS=20; then a bench_07 arm for `swift-27b-mtp2`. All profiles `-c 53248`,
Qwen sampling (`--reasoning-effort medium --temp 1.0 --top-p 0.95 --top-k 20`). Ran 23:09 → 04:52, rc=0.

| | `q38ud-27b` | `swift-27b` | `swift-27b-mtp2` | `ornith15-gsq3` | `ornith15-128k-mc` (control) |
|---|---|---|---|---|---|
| file / VRAM | 13.15 GB / 15.4 GiB | 12.12 GB / 15.0 GiB | same / 15.5 GiB | 12.99 GB / **14.2 GiB** | 20.2 GiB, 28 layers' experts in RAM |
| tg / pp @49K | 38 / ~1 350 | 35–36 / ~1 230 | – | **127–132 / ~3 380** | 71 / 2 239 |
| needle 1K/16K/49K | OK | OK | – | OK | OK |
| template / interp / perf | 2/2 · 2/2 · 2/2 | 2/2 · 2/2 · 2/2 | – | 2/2 · **1/2** · 2/2 | 1/1 ×3 this session |
| regex | 1/2 (367 s) | 1/2 (415 s) | – | **0/2** | 1/1 (272 s) |
| bench_07 | **20/20** | **20/20** | **20/20** | 19/20 (evidence) | 20/20 (10-01) |
| bench_07 run median / max | 172 / 470 s | 215 / 304 s | **102 / 146 s** | **88 / 146 s** | – |

Reading: both dense 27Bs reach the bench_07 ceiling and pass regex half the time (the `bonsai2-27b` tier), whole on
the card but at about half the decode of the expert-cached 35B. MTP draft-2 halves Swift's per-run time with no
score change, so `swift-27b-mtp2` is the 27B to keep. The 3-bit Ornith is the fastest serious-ish model in this log
(whole on the card, ~1.8x the default's decode), but loses exact-code tasks: a fast tier, not the default.
**`ornith15-128k-mc` stays the serious-agentic default** — still the only one with regex 2/2 here and every bench_07
item at 20/20. Open: a q38ud MTP arm, `swift-27b-mtp2` decode t/s at depth, bench_12/13 for the 27Bs.

## Final head-to-head — 2026-10-18: the three CANDIDATEs against `ornith15-128k-mc`

`llm/night/final-queue.sh` (02:29 → 08:01, rc=0) and `final2-queue.sh` (08:06 → 09:41, rc=0; logs `llm/night/final-*.log`,
work dirs `/mnt/cc/strata/final-work/`). One session, one timeout: bench_05 **full 12-task suite ×2** with
TASK_TIMEOUT **1200 for every model** (the leader's usual 900 would have been unfair to the 27Bs), bench_07 RUNS=10 as the
same-session run-time control. The easy/mid tier had never been run for the 27Bs nor for the `-mc` leader.

| Profile | b05 pass 1 | b05 pass 2 | regex (s) | bench_07 | median run | VRAM at boot | lost to server death |
|---|---|---|---|---|---|---|---|
| `ornith15-128k-mc` (leader) | **12/12** | **12/12** | PASS 399, PASS 607 | 10/10, every item | 159 s | 14.3 GiB | 0 |
| `swift-27b-mtp2-kv4` (q4_0 KV) | 11/12 | **12/12** | FAIL 8/14 @768, **PASS 184** | 10/10, every item | **116 s** | 14.8 GiB | 0 |
| `q38ud-27b` | 8/8 + 4 lost → parser rerun 3/4 | **12/12** | TIMEOUT 5/14, PASS 1147 | 10/10, every item | 143 s | 15.4 GiB | 4 tasks |
| `bonsai2-27b` (scratch master build) | 11/12 | 10/12 | TIMEOUT 7/14, TIMEOUT 11/14 | 10/10, every item | 227 s | 13.6 GiB | 0 |
| `swift-27b-mtp2` (q8_0 KV, the 10-17 profile) | void | void | — | 10 runs lost | — | 15.6 GiB | 22 of 22 requests |

Every model passed all eight easy/mid tasks in every pass (bugfix … toposort: 32 of 32 per model, except the four q38ud
tasks never reached). The suite therefore separates the four only on the parser tier and on time.

**Server deaths = the 16 GiB edge, not the model.** `swift-27b-mtp2` (q8_0 KV) loaded at 15.6 GiB and died on every
first request with `CUDA error: out of memory` in `ggml_cuda_graph_evaluate_and_capture` (reproduced by hand after the
queue, `~/.aillama/llama-server.log`). On 10-17 it had 0.8 GiB of margin; on 10-18 the desktop held 1.2 GiB (Chrome
GPU process, Spotify, EasyEffects). `q38ud-27b` at 15.4 GiB lost one request the same way. New profile
**`swift-27b-mtp2-kv4`** (q4_0 KV, same MTP args): 14.8 GiB at boot, 15.0 GiB after a request, no death in 34 tasks and
10 runs. A dense 27B at `-c 53248` on this card needs q4_0 KV whenever the desktop is up; the expert-cached 35B does not.

**Verdict.**
- **`ornith15-128k-mc` stays the serious-agentic DEFAULT**: the only profile with 24/24 at 1200 s, regex now 5/5 across
  10-01, 10-17 and 10-18, bench_07 every item at 20/20 + 10/10.
- **`swift-27b-mtp2` → ALTERNATE, fast 27B tier** (run as `-kv4`): same easy/mid and template/interp/perf record as the
  leader, bench_07 10/10 at the fastest median in this log (116 s vs 159 s), but regex is a coin flip (2/4 over two
  sessions; the 184 s pass is the fastest regex pass ever recorded here, the 768 s fail is the typical 27B overthink).
  Whole on the card with 1.3 GiB spare only with q4_0 KV. Dominates `q38ud-27b` and `bonsai2-27b`.
- **`q38ud-27b` → alternate, dominated by swift**: identical scores (23/24 scored, regex 2/4 over two sessions, one of
  them at 1147 s), slower bench_07 (143 s), no MTP, and it dies at the VRAM edge without `-kv4`. Keep the GGUF for an MTP
  arm (the head is in the file, never enabled); otherwise a delete candidate.
- **`bonsai2-27b` → off the candidate list**: parser tier on time (regex 0/2 TIMEOUT, interp TIMEOUT 4/13 once),
  bench_07 clean but at 227 s median, and it needs the scratch master `llama-server` the installed build cannot replace.
  Workflow discipline intact, exact-code work not; 7.6 GB kept as the ternary reference.

Open: swift decode t/s at depth with MTP (`bench_11` on `-kv4`), a q38ud MTP arm, bench_08 horizon for swift vs the
leader. Lesson for the queue tooling: `qrun watch` pushes `FAILED` lines; a bench's "SERVER DIED" lines are pushed only
when the queue script greps for them itself (`final2-queue.sh deaths()`), the first queue did not and 22 silent deaths
went unpushed for 5 hours.

### Architecture of the four (GGUF headers, `llm/ggufmeta.py`; gguf-dump cannot read the Q2_0 file)

| | ornith15-128k-mc | swift-27b-mtp2(-kv4) | q38ud-27b | bonsai2-27b |
|---|---|---|---|---|
| GGUF arch / type | `qwen35moe`, sparse MoE hybrid | `qwen35`, dense hybrid | `qwen35`, dense hybrid | `qwen35`, dense hybrid, ternary |
| Params total / active | 35.5 B / ~3 B | 27.3 B / 27.3 B | 27.3 B / 27.3 B | 26.9 B / 26.9 B |
| Layers (main + MTP) | 40 + 1 | 64 + 1 | 64 + 1 (unused) | 64, no head |
| Attention (interval 4) | 10 full + 30 GDN | 16 full + 48 GDN | 16 full + 48 GDN | 16 full + 48 GDN |
| d_model / FFN | 2048 / 512 per expert | 5120 / 17408 | 5120 / 17408 | 5120 / 17408 |
| Experts | 256, 8 routed + 1 shared | — | — | — |
| Heads q/kv, head dim | 16/2, 256 | 24/4, 256 | 24/4, 256 | 24/4, 256 |
| GDN inner / state / groups | 4096 / 128 / 16 | 6144 / 128 / 16 | 6144 / 128 / 16 | 6144 / 128 / 16 |
| Vocab, native ctx | 248 320, 262 144 | same | same | same |
| KV layers × kv_heads × head_dim | 10 × 2 × 256 | 16 × 4 × 256 | 16 × 4 × 256 | 16 × 4 × 256 |
| KV/token q8_0 / q4_0 | 10.6 / 5.6 KiB | 34 / 18 KiB | 34 / 18 KiB | 34 / 18 KiB |
| KV at profile ctx | 1.33 GiB @128K | 1.73 GiB @52K, 0.92 q4_0 | 1.73 GiB @52K | 1.73 GiB @52K |
| Quant, bpw | Q4_K_M 4.89 (Q4_K 29 B, Q6_K 6.4 B) | GSQ-RCO IQ3_S mix 3.55 (IQ3_S 8.5 B, IQ4_XS 5.7 B, IQ3_XXS 5.4 B, Q4_K 2.8 B) | UD-Q3_K_XL 3.85 (IQ4_XS 9.4 B, IQ3_S 8.2 B, IQ3_XXS 2.4 B, Q3_K 2.0 B) | Q2_0 2.27 (26.87 B one type) |
| Placement | experts of 28/40 layers on CPU + 64-slot cache, 14.3 GiB | whole, 15.6 GiB q8 KV / 14.8 q4 | whole, 15.4 GiB | whole, 13.6 GiB |
| tg @49K | 71 | 35–36 (+MTP draft-2) | 38 | 50 |

Same Qwen3.8-27B skeleton for the three dense files: they differ only by quant (3.55 / 3.85 / 2.27 bpw), hence the
tie on easy/mid and the split on exact code. 16 full-attention layers × 4 KV heads = 34 KiB/token, 3× the leader's;
at 53K that is 1.73 GiB of KV on top of 12–13 GiB of weights, which is why the desktop's 1.2 GiB tipped them over and
q4_0 KV (−0.8 GiB) is the whole margin. ~3 B active parameters per token is why the offloaded 35B decodes 2× faster.


## `ornith15-128k-mc` tuning — 2026-10-18: host knobs, two upstream PRs, thread placement, two re-quants

Question (user, 10-18): can the default's serving still be improved? Two research agents (research/38 upstream engines + PRs,
research/39 forums + quants), then measurements in one serial GPU chain (`epp-ab`, `pin-ab`, `fa30202`, `ncmoe20596`,
`fa30202-quality`, `ornq-bench`, `fa30202-regex`, `ornq-floor` queues, 10:52–18:17; working notes research/40). Profile unchanged
unless stated: `--n-cpu-moe 28 --moe-expert-cache 64 -ub 2048 -c 131072 -ctk/-ctv q8_0`, installed llama.cpp build 11438 + user
patches 0001–0004. Speed = bench_11 (49K prompt, 256 generated), two samples per arm per round unless stated.

### 1. Scorecard

| lever | 49K tg (installed → changed) | verdict |
|---|---|---|
| **PR #30202** (am17an draft: FA MMA dequantizes q8_0/q4_0 KV on the fly) | 71.2 → **78.8 (+10.6 %)**; 16K 82.8 → 85.8; pp unchanged | **ACCEPT**: quality gate below; patch `llm/patches/0005-cuda-fa-mma-q-dequant-pr30202.patch` ready to install |
| PR #20596 (ggml-cpu fused MUL_MAT_ID+GLU at n_tokens=1, work-stealing) | 72.9 → **41.8 (−43 %)**, pp +8 % | **REJECT**: the fused CPU path recomputes the experts that patch 0001's GPU cache would serve |
| threads: `-t 16 --cpu-range 0-15 --cpu-strict 1` (one per physical core) | 69.9 → 72.0 (+3 %, overlapping) | noise-edge, not applied |
| threads: `-t 8 --cpu-range 0-7 --cpu-strict 1` (one CCD) | 69.9 → 71.3 | **same speed with half the threads**: decode is memory-bandwidth-bound; frees a CCD for the desktop/qwen-code — candidate, measure bench_07 run time first |
| threads: `-t 32` (SMT) | 69.9 → 46 (−35…−40 %) | never |
| amd-pstate EPP performance vs balance_performance | within noise | closed |
| hugepages / THP | n/a: kernel has no THP, experts live in CUDA pinned host memory | closed |
| **re-quant GSQ-RCO 3.5-bit (`gsq35-mc`)** | 71 → **91–92 (+28 %)** at ncmoe 16, pp 2.9K → **4.1K (+40 %)** | **CANDIDATE successor** (section 3) |
| re-quant AtomicChat AD-Q4_K-IQ4_XS (`adq4-mc`) | 71 → 68–71, pp +6 %; bench_07 median 159 → 194 s | quality-equal, slower in agent runs: alternate only |

### 2. PR #30202 quality gate (patched binary, same profile)

Parser tier ×2 at 900 s: 4/4, then 3/4 with a regex TIMEOUT (model still iterating at the cutoff, no server fault). bench_07 RUNS=10:
10/10, every item 10/10, median 166 s. Because regex is the noisiest task, a **paired regex-only gate** followed (alternating ×4 at
TASK_TIMEOUT 1200, 0 server deaths):

| round | installed | + #30202 |
|---|---|---|
| 1 | **TIMEOUT 0/14** | PASS 405 s |
| 2 | PASS 973 s | PASS 982 s |
| 3 | PASS 312 s | PASS 700 s |
| 4 | PASS 594 s | PASS 722 s |

Patched 4/4, installed 3/4: the morning timeout was model noise and the same noise hit the installed stack. Verdict rule (patched ≥
installed − 1, no deaths) met with margin → **install 0005**: copy to `/etc/portage/patches/sci-misc/llama-cpp-9999/` and run the
pinned rebuild (research/35 evening A recipe; needs sudo = the user's window). Expected: +10 % decode at 49K for every `-mc` profile
with q8_0 KV, including `gsq35-mc`.

### 3. Two re-quants of the same Ornith-1.5-35B-A3B (research/39; files in `/mnt/cc/strata/ornith-quants`, sha256 = HF LFS oid)

Both are the same `qwen35moe` graph as the default (40 blocks, 10 full-attn + 30 GDN, 256 experts 8+1 shared); GSQ-RCO drops the
NextN/MTP block and the vision tower (not used here) and re-quantizes: Q2_K/Q3_K/Q4_K experts, Q8_0 attention/GDN/shared expert,
3.50 bpw vs the default's Q4_K_M 4.89 (`llm/ggufmeta.py`). Profiles = the default's args with only `-m` and `--n-cpu-moe` changed.

**ncmoe floor sweep (49K request at every step, top-down; VRAM = nvidia-smi used incl. ~1 GiB desktop):**

| ncmoe | gsq35 VRAM GiB | gsq35 tg @49K | adq4 VRAM | adq4 tg @49K |
|---|---|---|---|---|
| 28 | | | 14.4 | 69.6 |
| 26 | 12.6 | 78.4 | 15.0 | 72.9 |
| 24 | 13.2 | 77.8 | **OOM on the request** (15.5) | — |
| 22 | 13.6 | 86.2 | | |
| 20 | 14.0 | 87.0 | | |
| 18 | 14.4 | 88.4 | | |
| 16 | **14.8** | **91.4** | | |
| 14 | 15.3 | 89.5 | | |
| 12 | 15.4 | 98.2 (floor) | | |
| 10 | does not boot | | | |

Record run at the chosen value (1K/16K/49K ×2, 256 generated):

| | 1K tg | 16K tg | 49K tg | pp @49K | pp @1K |
|---|---|---|---|---|---|
| `ornith15-128k-mc` (ncmoe 28, 14.3 GiB; 10-18 samples) | 85–87 | 80–89 → 82.8 | 67–75 → 71.2 | 2 820–2 950 | — |
| `gsq35-mc` ncmoe 14 (15.3 GiB) | 117.3, 120.7 | 108.0, 105.6 | **92.1, 92.2** | 4 054–4 097 | 2 356–2 431 |
| `adq4-mc` ncmoe 28 (14.4 GiB) | 81.2, 86.4 | 77.4, 82.3 | 68.1, 71.4 | 3 123–3 143 | 1 643–1 655 |

`gsq35-mc` profile set to **ncmoe 16** (not the queue's floor+2 = 14): 14 loads at 15.66 GiB, exactly where the 27Bs died on the
first request with a loaded desktop (section "Final head-to-head"); 16 keeps 1.1 GiB headroom at −2 % tg. The 12 → 98 t/s point
shows what a fully resident expert set would do; 10 does not fit.

**Quality (both at their first serving ncmoe, 26 and 28; placement does not change the math):** needle 6/6 each; parser tier ×2 at
900 s **8/8 with regex 2/2 for both** (gsq35: 334, 497 s; adq4: 241, 763 s); **bench_07 RUNS=20: 20/20, every item 20/20 for both**,
0 server deaths; median run gsq35 **158 s** (min 106, max 235) = default 159 s; adq4 **194 s** (min 133, max 335).

**Reading.** The 3.5-bit GSQ file matches the Q4_K_M default on every quality axis this log measures (N=20 workflow, 2 parser passes
incl. regex) while being 6 GiB smaller — 14 more expert layers fit on the GPU, hence +28 % decode and +40 % prefill (the extra layers
also leave the expert cache serving fewer CPU layers). AtomicChat's better KLD (0.0315 vs 0.0477) buys nothing measurable, and its
IQ4_XS expert tensors dequantize slower on the CPU, which shows as a 22 % longer bench_07 median at equal 49K tg. Caveat on gsq35:
bench_05/07 are coarse; a 3.5-bit file may lose on long-horizon or knowledge-heavy work that bench_08 horizon would catch, and the
two speed samples at ncmoe 16 are two, not twenty.

**Decision:** `ornith15-128k-mc` stays DEFAULT until `gsq35-mc` is confirmed at ncmoe 16 (bench_07 ×20 + parser ×2 at the production
value, plus a bench_08 horizon pair); then swap. `adq4-mc` stays in models.conf as a documented alternate. Next queue:
`gsq35-confirm` (bench_07 ×20, parser ×2, bench_11 ×2 at ncmoe 16, ideally on the 0005 binary once installed).

### 4. Process lessons (details research/40)

- `qrun stop` on a *waiting* queue ran its restore-at-exit and restarted the server under another queue's measurement (one sample lost,
  twice). Rule: the restore is guarded by `STARTED=1` set after the last `q_after`; `qrun stop` waits `QRUN_STOP_WAIT` 90 s before -9;
  never stop/cancel an old-shape waiting queue while another queue measures.
- A probe that takes the *first* serving ncmoe understates a new file by up to 2.3 GiB of VRAM (gsq35: 26 vs floor 12): keep stepping
  down, stop at the first failure after a success (fixed in `ornq-bench-queue.sh`; `ornq-floor-queue.sh` did the sweep).
- bench_11 with 10 generated tokens (needle mode) mis-ranks decode (adq4 looked −14 %); rank on 256-token runs.

## HF sweep — 2026-10-29: two real candidates (Sage-A1.5, LFM2.5-24B-A2B), DeepSeek "Flash" is 763B

Swept `huggingface.co/api/models?filter=gguf` by creation date and by downloads,
window 2026-10-10 -> 10-29 (the previous sweep in this log is 2026-09-17; the
candidates tested since came from it). Checked each hit's `config.json` for
arch/size before judging the name.

**Nothing new in the base families we run.** No Qwen3.9, no Ornith 1.6/2.0, no
new Nemotron Lightning. The Ornith/Qwen3.8 churn on HF is all re-quants and
abliterations of bases already in the fleet table.

| Repo | What it really is | Verdict |
|---|---|---|
| `InternScience/Sage-A1.5-Q4_K_M-GGUF` (249k dl) | **qwen3_5_moe 35.95B**, tags `vlm vision agentic code tool-use`, arxiv:2610.28578; official GGUF 21 GB + `mmproj` | **CANDIDATE — the strongest since Ornith 1.5.** Same arch as the whole fleet (so llama.cpp support is certain), same size class as `ornith15-128k-mc` (20.22 GiB), and it is the first fleet-class model that is *also* a VLM |
| `LiquidAI/LFM2.5-24B-A2B-GGUF` (210k dl) | `lfm2_moe` 23.97B, 64 experts / **4 active (A2B)**, 40 layers, 131K ctx | **CANDIDATE, second priority** — a lower active-parameter class than anything in the fleet (A2B vs A3B), so potentially much faster. llama.cpp has the arch (several closed issues), but **#30490: "LFM2.5-24B-A2B emits Qwen3-Coder-style `<tool_call>` XML as content"** — a tool-protocol defect, test that first |
| `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` (1.58M dl) | `deepseek_v41`, **763B params** — "Flash" is a product name, not a size | **REJECTED by the >100B rule** (no engine-side expert paging in llama.cpp) |
| `ornith-ai/Ornith-1.5-122B-GGUF` (443k dl) | Ornith 1.5 at 122B | **REJECTED by the >100B rule** |
| `migtissera/Tess-4.5-27B-GGUF` (219k dl) | 27B dense finetune | **REJECTED** — the 27B-dense-at-128K ceiling, already paid for five times |
| `DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF` (315k dl) | third-party re-quant of the LiquidAI model | skip — prefer the vendor GGUF; the one DavidAU file ever tested here (gpt-oss20b-heretic) was the slowest of its family and failed the parser tier |
| `AMAImedia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` | speech-transcription variant | out of scope |
| `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` (275k dl) | abliteration of the 125B Flash-Next | out by the >100B rule; the abliterated 9B sibling was rejected 2026-09-29 |

**Action:** `Sage-A1.5` is worth a download slot as soon as the Bolt screen
finishes — it is the first candidate that would exercise both the agentic
benches and a vision bench that does not exist yet (see "Where the benches
stopped discriminating" in BENCH.md).

## Reference results — 2026-10-29: NVIDIA Nemotron 3.5 Bolt 74B-A9B — 12/12 on bench_05 at the first attempt

First run of `bolt` (profile added the same day: `--n-cpu-moe 46`, 131072 ctx,
q8_0 KV, 15.5 GiB VRAM). Installed llama.cpp `f64a49819` (b11835) already
carries Bolt support (#30520, merged 10-15) and the pixtral->Tekken tokenizer
fix (#30960) — **no rebuild was needed**; the b11927 in the XDA article is just
the build its author happened to use.

**Placement.** `--fit` at 128K put all 89 layers on the GPU with the experts of
41 layers overflowing to system memory (13.7 GiB VRAM, 8.7 GiB RAM).
`llama-bench` floor is 44 (42 OOM); the profile sits at 46 per the floor+2 rule.
GGUF header settles the block count the research left open: **88 blocks**, 496
experts, 9 used per token + 1 shared, hidden 4096, ctx 262144.

**Speed.** 52-57 t/s decode short, pp 634 @512. At a 72K-token prompt: **pp 1551
t/s, tg 50 t/s**, and the planted access code was retrieved correctly. The KV
cache is unusually cheap — only ~8 of 88 blocks are attention, 2 KV heads at
head_dim 128, so q8_0 KV at 128K is **708 MiB** (an A3B sibling pays ~4 GiB).
For reference the XDA article measured 22-32 t/s and 110 t/s pp on a 4070 Ti
Super with 32 GB RAM.

**bench_05, N=1, TASK_TIMEOUT=900 — 12/12, every task at 100%:**

| task | time | task | time | task | time |
|---|---|---|---|---|---|
| bugfix 4/4 | 40 s | intervals 12/12 | 98 s | template 10/10 | 136 s |
| scratch 5/5 | 719 s | fsm 13/13 | 77 s | interp 13/13 | 374 s |
| lru 8/8 | 116 s | codec 12/12 | 53 s | perf 6/6 | 296 s |
| multifile 5/5 | 112 s | toposort 11/11 | 57 s | **regex 14/14** | 538 s |

`regex` is the task that discriminates the top tier — `ornith15-128k` passes it
roughly 2 runs in 7, `gpt-oss20b-q8_0` 57 %, and most of the fleet has never
passed it at all. Bolt passed it on the first attempt, with a perfect score on
every other task. The only model that previously reached 12/12 here is
`ornith15-128k` (2026-09-19), and not on a first run.

**bench_07, RUNS=3 — 3/3 PASS, 10/10 every run** (119, 160, 186 s). Every
rubric item held 3/3, including `lastline` and `evidence`, the pair that is a
known ~40-45 % miss for the gpt-oss family. Median run 160 s is level with
`gsq35-mc` (158 s) and `ornith15-128k-mc` (159 s) despite 50 vs 71 t/s decode —
Bolt spends fewer tokens per task.

**bench_14_tools (new, same day) — 6/6**: preflight OK, simple, nested object,
a 4892-char argument round-tripped with all 8 markers, two parallel calls in one
turn, graceful recovery from a tool error, and a correct refusal to invent a
missing argument.

**Reading, with the house caveat.** This is the cleanest screen in this log, but
it IS a screen: bench_05 at N=1 and bench_07 at RUNS=3 find hard gaps, they do
not rank (same-session drift has flipped rankings at N=15). Nothing changes role
until `bolt` and `ornith15-128k-mc` run **RUNS=20 in one session**, plus a
parser-tier repeat and a `bench_08_horizon` pair. Two further unknowns: the
1.73 GB vision projector and the LiLiCorr drafter are downloaded and unbenched,
and users on the HF discussions report runaway thinking (8K and 16K budgets both
exhausted on one task) — the profile carries `--reasoning-budget 8192` for that
reason.

## Reference results — 2026-10-29: LiLiCorr (dflash) speculative decoding on Bolt — a WIN on 16 GB, against the published report

`bench_08_spec_offload.sh`, installed build b11835, ctx 32768, NPREDICT 384,
REPEATS 2. Drafter = `nvidia/...-24GB-LiLiCorr-GGUF`, 998 MB, arch `dflash`
(5 blocks, embedding 4096, 32 heads / 2 KV, block_size 8). Bolt has no MTP
head, so `draft-dflash` is its only drafter.

**First attempt failed at the production split, and the failure is the finding.**
At `--n-cpu-moe 46` the server dies during start:
`failed to allocate CUDA0 buffer of size 990362624` — the drafter wants 990 MiB
and the experts already hold it. The XDA review reported the drafter as a
slowdown on a 16 GB 4070 Ti Super; this is the mechanism, seen directly.

**Second attempt: free the room, then compare like for like.** Raising the split
to `--n-cpu-moe 52` lets the drafter load. Both arms then share ncmoe 52:

| ncmoe | draft n-max | tg t/s (2 runs) | acceptance |
|---|---|---|---|
| 46 | none (best no-draft) | 50.5, 53.1 | — |
| 52 | none | 48.4, 51.2 | — |
| 52 | 2 | 55.5, 57.2 | 220/324 = 68 % |
| 52 | **3** | **62.3, 64.3** | 260/366 = **71 %** |

**Reading.** Drafting at n-max 3 beats the best no-draft configuration by
**~21 %** (62-64 vs 50-53) and the same-split baseline by ~25 %, at 71 %
acceptance. The six extra CPU expert layers cost ~2 t/s; the drafter wins that
back several times over. This is the SpecMoEOff prediction holding on our box
(`BENCH.md` -> Literature): with experts on the CPU the GPU idles waiting on
PCIe, so a draft batch is nearly free. The published "drafter is slower on
16 GB" result is an artifact of leaving the expert split where it was — the
drafter cannot even load there.

**n-max 3 is a floor, not the peak.** 3 beat 2, and the paper's throughput peaks
at 5-6. The sweep stopped at 3 deliberately: llama.cpp **#31350** (spec rollback
corrupts recurrent state for drafts longer than `n_rs_seq`) is NOT in b11835 and
Bolt is a Mamba2 hybrid. Higher n-max needs the build that carries the fix —
pinned at `7df8b876b`, measurement pending.

**Not yet done:** a quality check with the drafter on. Speculative decoding is
supposed to be output-preserving, but that is an assumption until a parser-tier
run says so.

### Addendum, same evening — the sweep on the fixed build (7df8b876b = b11996, patch-free scratch)

Same script and settings, `SERVER_BIN` = the scratch build that carries #31350:

| build | n-max | tg t/s (2 runs) | acceptance |
|---|---|---|---|
| b11996 | none | 50.1, 50.5 | — |
| b11996 | **3** | **58.7, 59.6** | 252/393 = 64 % |
| b11996 | 5 | 56.2, 57.1 | 277/527 = 53 % |
| b11996 | 6 | server died at first decode (VRAM — a longer draft batch needs a bigger compute buffer at ncmoe 52) | — |
| b11835 (above) | 3 | 62.3, 64.3 | 260/366 = 71 % |

**n-max 3 is the peak on both builds**, not the 5-6 the SpecMoEOff paper
predicts: acceptance falls from 64 % to 53 % between 3 and 5, faster than the
longer draft helps, and 6 does not fit at all at this split. So the rollback fix
does NOT unlock a higher useful n-max here — 3 was already the right value.

**Unresolved: the new build accepts fewer drafts at the same n-max (64 % vs 71 %)**,
and is correspondingly slower with drafting (59 vs 63 t/s) while the no-draft
baseline is unchanged (~50 t/s on both). Two candidate causes, neither tested:
(1) #31086 switched `ggml_softplus` to `log1pf` after b11835, and open #31220
reports exactly this — a lower DFlash accept rate — on Qwen; the update-scope
research thought Bolt's Mamba2 kernels were immune (they compute softplus inline),
and this measurement says otherwise or points elsewhere; (2) #31350 itself changes
which recurrent checkpoint is restored after a rejected draft — if the old build's
state was being corrupted, its higher acceptance is not something to want back.

**The second cause is why a quality check is now mandatory, not optional**: if
b11835 at n-max 3 is in fact hitting the rollback bug, its extra speed is bought
with wrong output. Until a parser-tier run with the drafter on settles it, the
drafter is a measured speed result only, on either build.

## Reference results — 2026-10-29: llama.cpp update gate — master `eb4e244be` + rebased 0005 vs installed, on `ornith15-128k-mc`

Profile `ornith15-128k-mc` on upstream `--moe-cache-mib 3540` (migrated the same evening: our
0001/0003 were never in the installed build). Queue `llm/night/q-llama-update-ab.sh`, arms in the
order **A B C C B A**: A = installed b11835 (f64a49819, which DOES carry 0005 — `FA_Q_MMA` strings
present in its libggml-cuda); B = scratch master b12001 + rebased 0005; C = B with
`FA_Q_MMA_MIN=99` (0005 path off at runtime). bench_11, NGEN 256, RUNS 2, default corpus.

| arm | 16K tg | 49K tg | 49K pp | needle 1K / 49K |
|---|---|---|---|---|
| A installed | 107.4 | 98.5 | 2 622 | 4/4 OK |
| **B master + 0005** | **107.9** | **101.5** | **2 708** | 4/4 OK |
| C master, 0005 off | 103.1 | 88.4 | 2 670 | 4/4 OK |

(means of 4 runs per arm, two per round; per-run spread ~±5 %). **Xid lines in the kernel log over
the whole run: 0** (4 cold starts of the new binary, 2 of the old).

**Reading.**
- **The update is level-to-slightly-better** (B vs A: +3 % decode and +3 % prefill at 49K, same sign
  in both rounds). Not a significant speed win on its own; it is worth taking for #31350 and the
  merged upstream patches, not for speed.
- **0005 is worth +15 % decode at 49K** (B 101.5 vs C 88.4) and +5 % at 16K — a decode-at-depth win,
  exactly as its gate `Q->ne[1]` 1..8 predicts. It must stay in the series.
- **The needle gate passed at both depths on both builds**, so the non-mmap loader rewrite of 10-29
  (`0ac1a7fd3`) did not corrupt weights on this model.
- **Indicative, different session and build:** 98-101 t/s decode at 49K on upstream's cache vs the
  71-74 recorded for this profile on 10-01 with a build that carried our 0001. If it holds, the
  upstream cache is much faster than our patch was — needs its own same-session check before quoting.

**Gates still open before installing:** the bench_05 parser tier + bench_14 tools on arm B (the
update includes `a71c00f0b`, a global PEG-parser semantics change — the qwen-code path breaks there
first). Install = delete 0001-0004 from `/etc/portage/patches/sci-misc/llama-cpp-9999/`, replace 0005
with the rebased file, pin the commit, `emerge -1 sci-misc/llama-cpp`, re-probe floors.

### Addendum — parser gate on the new build (same evening): PASS

`llm/night/q-llama-update-parser.sh`, profile `ornith15-128k-mc` (`--moe-cache-mib 3540`):
bench_14 tools **6/6 on the installed build and 6/6 on master+0005** (same session); bench_05
parser tier on master+0005 **4/4: template 10/10 (100 s), interp 13/13 (260 s), perf 6/6 (135 s),
regex 14/14 (328 s)**. The global PEG-parser change `a71c00f0b` did not break the qwen-code path.
All gates for installing master `eb4e244be` + rebased 0005 have passed: needle 24/24, Xid 0 over
6 cold starts, speed level-to-better, tools and parser tier clean. (regex at N=1 is not a rate —
this profile historically passes it ~2 runs in 7; here it is only evidence of no breakage.)

## 2026-10-30 — b12001: -mc floors on `--moe-cache-mib`, joint ncmoe/MiB tuning, Bolt vs ornith N=20

Build: llama.cpp b12001 (eb4e244be) + rebased 0005, installed 10-29 20:37. Queues: `night/q-night-1029.sh`
(floors at 2-step strides + Bolt), `q-night-1029b.sh` / `q-floor-1030.sh` (HDD floors), `q-tune-1030.sh`
(arms `PROFILE:NCMOE:MIB`, `TUNE_ARMS=` override), `q-bootcheck-1030.sh`. Tables:
`night/q-night-1029.floors.tsv`, `night/q-tune-1030.tsv`, `night/q-tune-1030b.tsv`.

**Probe method.** Each arm: llama-server started directly with the profile's args, one 49K needle request
(wrong answer = DEEP-FAIL), then bench_11 at 16K and 49K, RUNS=3, NGEN=256. The current setting ran first
and last per profile: drift within 1-2 %. Measured in daytime with the desktop running (~1.5 GB VRAM).

**Trading cache MiB for fixed expert layers is always slower** (ornith 28/3540 103.4 tg @49K vs 24/2600
100.9, 22/1700 94.5, 20/800 82.1; nex25-mini 29/4153 99.9 vs 27/3200 96.2, 25/2300 91.3; tiel 29/3417 89.3
vs 25/2500 87.0). Keep the MiB, move ncmoe only.

| profile | old ncmoe | floor (first fail) | new | tg @49K old → new | pp @49K new |
|---|---|---|---|---|---|
| ornith15-128k-mc (+ -n3 variants) | 28 | 26 (25 boot) | 28 kept | 103.4 | 2757 |
| adq4-mc | 28 | 25 (24 boot) | 27 | 97.3 → 98.8 | 2999 |
| nex25u-mini-mc | 22 | 19 (18 boot) | 21 | 107.0 → 111.9 | 3348 |
| gsq35-mc | 16 | 13 (12 wrong needle) | 15 | 113.5 → 115.6 | 3906 |
| qwen36-128k-mc | 24 | 20 (19 wrong needle, 18 boot) | 22 | 107.1 → 111.6 | 3308 |
| qwen38d-128k-mc | 28 | 24 (23 boot) | 26 | 94.2 → 100.1 | 2823 |
| nex25-mini-mc | 29 | 29 (28 boot at 4153 MiB) | 31 | 99.9 → 95.5 | 2619 |
| tiel-128k-mc | 29 | 27 (26 boot) | 29 kept | 89.3 | 2660 |

New = floor+2 (desktop headroom). nex25-mini ran AT its floor before; 31/4153 beats every MiB trade.
`ornith15-128k-mc-budget` removed: its quoted `--reasoning-budget-message` was split by aillama and it never
booted (known since 10-02; `-n3-budget` is the working arm, N=1 3/4).
Two below-floor arms booted but answered the 49K needle wrongly (qwen36 19, gsq35 12) — a boot check alone
would have accepted them.

aillama's fixed 120 s health wait killed every cold db5-HDD load in the night queues (the warm load is 9-12 s);
it is now `LLAMA_HEALTH_WAIT` (default 300 s).

**Bolt vs ornith15-128k-mc, same session (q-night-1029, 0 Xid).**

| | bolt | ornith15-128k-mc |
|---|---|---|
| parser tier ×3 | 10/12 — template TIMEOUT at 10/10 (900 s), regex TIMEOUT 6/14 | 11/12 — regex TIMEOUT once |
| task wall time | 265-900 s | 58-323 s |
| bench_07 RUNS=20 | 17/20 (`summary` 17/20, every other item 20/20) | **20/20, every item** |
| bench_14 tools | 6/6 | 6/6 |

Verdict: Bolt stays an alternate; the default does not change.

## 2026-10-30 — candidate screen: mimo-9b, Sage-A1.5, LFM2.5-24B-A2B (q-cand-1030, q-cand2-1030, 0 Xid)

llama.cpp b12001. Screen = 49K needle + bench_14 tools + bench_05 all 12 tasks N=1 (TASK_TIMEOUT 900).

| | mimo-9b (Q8_0, whole) | sage-a15-mc (Q4_K_M, ncmoe 28 + cache 3540) | lfm25-24b (Q4_K_M, ncmoe 9) |
|---|---|---|---|
| 49K tg / pp | 75 / 5 165 | 81.5 / 2 751 | 90–103 / ~3 200 (ncmoe 7–8) |
| bench_14 tools | 6/6 | 6/6 | 5/6 (`nested` object as a JSON string) |
| bench_05 easy/mid | 7/8 (intervals TIMEOUT 11/12) | 8/8 | 8/8 |
| template / interp / perf / regex | FAIL 4/10 / TIMEOUT 10/13 / TIMEOUT / stopped by the user | PASS / PASS / PASS / TIMEOUT | TIMEOUT 0/10 / TIMEOUT 0/13 / FAIL 5/6 / TIMEOUT 2/14 |
| verdict | REJECTED, GGUF deleted (user stop mid-regex) | → same-session tier | REJECTED (8/12 < 10 gate) |

LFM floor: bench_11 `NEEDLE=1` reported "49K FAIL" at ncmoe 16 — the model always reasons first and ignores
`enable_thinking:false`, so the needle read an empty answer even at NGEN 1024. A hand needle (48.7K prompt, 3K
token budget) passed at 16, 8, 7; 6 and 4 died at boot. Profile at 9 (floor+2).

**Sage-A1.5 vs ornith15-128k-mc, same session.**

| | sage-a15-mc | ornith15-128k-mc |
|---|---|---|
| parser tier ×3 | 8/12 — template 3/3, interp 3/3, perf 2/3 (TIMEOUT at 6/6), **regex 0/3** (13/14, 1/14, unscored) | 11/12 — perf TIMEOUT once at 6/6, **regex 3/3** (271–696 s) |
| task wall time | 108–384 s (passes) | 107–696 s (passes); template/interp/perf 107–187 s |
| bench_07 RUNS=20 | **20/20, every item**, median 138 s | **20/20, every item**, median 134 s |

Verdict: Sage stays an alternate; the default does not change. Regex 0/4 across every Sage run is a hard gap.
Vision (mmproj 0.84 GiB, downloaded) is untested.

## 2026-11-03 — small 27B builds: enru, Saluki, Caphlon vs swift-27b-mtp2-kv4 (night/small27-queue-1103.sh, 0 server deaths)

Three Qwen3.8-27B builds that fit whole with room to spare, same profile shape as `q38ud-27b`
(`-c 53248`, q8_0 KV, reasoning medium, Qwen sampling); control `swift-27b-mtp2-kv4` in the same
session. Installed llama.cpp b12001. Polish = 3 fixed prompts, thinking off, read by hand
(`night/small27-*-polish.txt`). Architecture comparison and the tokenizer test:
`~/Claude/info/research-gguf-arch-quality-20261103.md`.

| | enru-27b | swift-27b-mtp2-kv4 | saluki-27b | caphlon-27b |
|---|---|---|---|---|
| file | IQ3_XXS 10.99 GB | GSQ-RCO IQ3_S 12.12 GB | IQ2 mix 7.90 GB | IQ2_S 7.38 GB |
| VRAM @49K | 14.0 GiB | ~15 GiB (q4_0 KV) | 11.7 GiB | 11.3 GiB |
| tg @1K / @49K | 63 / 54 | MTP, not re-measured | 70 / 62 | 79 / 67 |
| needle 1K/16K/49K | 6/6 | – | 6/6 | 6/6 |
| bench_14 tools | 6/6 | 6/6 | 6/6 | 5/6 (decline) |
| parser pass 1 | 4/4 (regex 249 s) | **4/4** (regex 130 s) | 2/4 | 3/4 |
| parser pass 2 | 4/4 (regex 474 s) | not run | 3/4 | 1/4 |
| bench_07 RUNS=10 | 10/10, 179 s | (10-18: 10/10, 116 s) | 10/10, 184 s | 10/10, 192 s |
| Polish | very good | very good | good | unusable |

Verdict: enru-27b = CANDIDATE (8/8 incl. regex 2/2, tools 6/6, q8_0 KV fits with 2 GiB spare);
saluki-27b = alternate, not promoted (regex 0/2); caphlon-27b = REJECTED. The control scoring 4/4
rules out session drift as the cause of the 2-bit losses. bench_07 is 10/10 for all four: it no
longer separates this tier. A prediction written before the result (enru ~6/8 because its
attention sits at 3.02 bpw) was wrong: the "keep attention high" rule holds for MoE files, not
for this dense hybrid with an imatrix IQ3_XXS.

## 2026-11-03/04 — GGUF hunt queue: 7 candidates + same-session control (night/q-hunt-1103.sh, 13 h 45 min, rc=0)

Candidates from the 16 GB hunt (`~/Claude/info/ggufhunt-16g-20261103.md`,
`info/scratch-27b-arch-hunt-20261103.md`). Per model: top-down ncmoe floor probe (each step
must answer the 49K needle; production = last pass + 2), tg at 49K, bench_14 tools,
3 Polish prompts (thinking off, read by hand, `night/hunt-*-polish.txt`), bench_05 parser
tier ×2 (`TASK_TIMEOUT=1200`), bench_07 RUNS=10. Control `ornith15-128k-mc` at the end:
tools + Polish + parser ×1. Installed llama.cpp b12001. Summary: `night/q-hunt-1103.summary.tsv`.

| | enru-27b-mtp | kataguru-35b | bolt-q3kxl | groot-19b | q38-14b | ornith122-iq2 | memt-35b | **ctrl ornith15-128k-mc** |
|---|---|---|---|---|---|---|---|---|
| file | IQ3_XXS 10.99 GB + MTP draft | APEX-Good | UD-Q3_K_XL | Q4_0 (13.8 GB, whole) | Q5_K_M | UD-IQ2_XXS | Q4_K_M 21.6 GB | Q4_K_M 20.2 GiB |
| ncmoe (floor) | whole | **14** (12; 10 boots, needle FAIL) | **80** (78; 74 boot fail) | **0** | whole | **40** (38; 36 boot fail) | — | 28 + mc |
| VRAM after 49K | 14.9 GiB | 14.6 | 13.8 | 13.8 | 13.5 | 13.9 | — | — |
| tg @49K | **88.8** | 70.3 | 24.0 | **168.9** | 66.5 | 20.1 | — | (103, 10-30) |
| tools | 6/6 | 6/6 | 6/6 | 6/6 | **0/6** | 6/6 | — | 6/6 |
| parser pass 1 | 3/4 (regex TIMEOUT 10/14) | 3/4 | 3/4 | **4/4** | 0/4 | **4/4** (regex 690 s) | — | **4/4** (133/149/165/640 s) |
| parser pass 2 | 0/4 = **infra** (server never healthy, 300 s, disk contention from downloads) | 2/4 (regex never) | 4/4 | **4/4** | 0/4 | **4/4** (regex 705 s) | — | not run |
| bench_07 RUNS=10 | 10/10, med 121 s | 9/10, 61 s | 9/10, 395 s | **10/10, 48 s** | 0/10 (7 FAIL, 3 TIMEOUT) | **10/10**, 399 s | — | not run |
| Polish | very good | correct, terse | good, minor slips ("volatile", typo) | **unusable** (mixed languages) | gibberish | very good | — | very good |

Verdicts:
- **groot-19b = CANDIDATE (fast tier)** — 8/8 parser incl. regex 2/2, bench_07 10/10 at the
  fastest median in this log (48 s), fits whole at 169 tok/s. Polish unusable (expert pruning
  256→128, probably dropped multilingual experts — inference). Promotion needs RUNS=20 +
  parser ×3 against `ornith15-128k-mc` and `ornith15-gsq3` in one session, with the card's
  coding sampling (temp 0.6, top_p 0.95, top_k 20, presence 0.0) added to the profile.
- **ornith122-iq2 = ALTERNATE, "slow but smart"** — 8/8 parser, 10/10 bench_07, very good
  Polish, but 20 tok/s and 399 s median runs (≈ 8× groot). First measurement of a >100B model on
  this card; the 10-29 ">100B rule" rejection was wrong on quality, right on speed.
- **enru-27b-mtp = unresolved** — MTP lifts tg 54 → 89 and halves the bench_07 median
  (179 → 121 s) but pass 1 lost regex (the card warns the draft head was not retrained after the
  vocab cut); pass 2 was lost to infra. Rerun parser ×2 before a verdict.
- **kataguru-35b = REJECTED** — 5/8 parser (regex 0/2), Polish terse; nothing over the leader.
- **bolt-q3kxl = REJECTED** — 7/8 but 24 tok/s and 395 s median; the existing `bolt` alternate
  lost to the leader already (10-30), a slower quant of it earns nothing.
- **q38-14b = REJECTED** — empty content on every request, 0/6 tools; broken GGUF/template. Delete candidate.
- **memt-35b = NOT RUNNABLE on b12001** — `wrong number of tensors; expected 853, got 733`:
  `general.architecture` says `qwen35moe` but the file carries 120 extra `blk.N.memt_*` tensors
  (12 kinds × 10 layers: q/k/v/w/b/f/g, decay, o_norm — a memory module) that mainline does not
  know. Needs the author's llama.cpp fork; the queue's "floor FAIL" is not a VRAM result.
- Control 4/4 in the same session rules out drift for the losses above.

## 2026-11-05 — finalists in one session (night/q-final-1105.sh, 01:38 → 17:22, rc=0, 0 Xid) + Groot quant arms (q-groot-quant-1106, q-groot-q4km-1106)

Live tables: `BENCH.md` "Finalists in one session — 2026-11-05". This section keeps the per-task
evidence. Installed llama.cpp b12001. Logs `night/final-*.log`, `night/gq-*.log`, summaries
`night/q-final-1105.summary.tsv`, `night/q-groot-quant-1106.summary.tsv`.

### Groot quant arms (same args and card sampling; depth 1K/16K/49K/98K ×2 needle, tools, parser ×1, bench_12 RUNS=2)

| | IQ4_XS 10.94 GB | Q4_K_S 11.53 GB | Q4_0 11.04 GB | Q4_K_M 12.48 GB |
|---|---|---|---|---|
| tg @1K / 16K / 49K / 98K | 178 / 173 / 151 / 135 | 185 / 174 / 157 / 139 | **205 / 190 / 167 / 146** | 173 / 163 / 147 / 127 |
| pp @49K | 7 368 | 7 354 | 7 226 | 7 391 |
| VRAM after 98K | **13 753 MiB** | 14 399 | 13 941 | 15 076 (ncmoe 0 held, the `-n2` fallback not needed) |
| needle / tools / parser | 8/8, 6/6, 4/4 | 8/8, 6/6, 4/4 | 8/8, 5/6 (`longarg` INCONC), 4/4 | 8/8, 6/6, 4/4 |
| bench_12 RUNS=2 | **13/18** | 11/18 | 10/18 | **13/18** |
| 2/2 | chain6, chain10, chain14, longctx, tracespec | chain6, chain14, ledger, tracespec | chain6, chain14, ledger, longctx | chain6, chain10, chain14, ledger, rules |
| 0/2 | ebuild | ebuild, reconcile | reconcile, rules, tracespec | ebuild |

Winner by the queue rule (boot OK, no needle miss, highest bench_12, tie → smaller file): **IQ4_XS**.
All four are bartowski imatrix files; the legacy Q4_0 layout is the fastest decoder and the weakest
on the hard set. Q4_K_M is the quality alternate (1.2 GB spare at 128K).

### bench_12_hard per task, finals (RUNS=2; o122 RUNS=1, per-task budgets 1 200–2 400 s)

| task | leader | groot-iq4xs | enru-mtp | swift | gpt-oss-high | o122 |
|---|---|---|---|---|---|---|
| ledger | PP | PP | PF | PP | FF | T |
| chain6 | PP | FF | FF | FF | PF | P 729 s |
| chain10 | PP | PP | FF | FF | FF | T |
| chain14 | FP | PF | PF | FF | PF | P 2 306 s |
| tracespec | PT | PF | FF | FF | PP | T |
| reconcile | TP | FF | FP | PP | FF | T |
| ebuild | FF | FF | PF | FF | FF | T |
| longctx | PF | PP | FP | PF | FF | F 797 s |
| rules | PP | FF | FF | FF | FF | T |
| **PASS** | **12/18** | 8/18 | 5/18 | 5/18 | 4/18 | 2/9 |
| mean PASS / FAIL run | 522 / 493 s | 202 / 170 s | 254 / 227 s | 203 / 148 s | 410 / 354 s | 1 518 / — |

P = PASS, F = FAIL, T = TIMEOUT. Groot held chain6 2/2 and rules 1/2 the night before on the same
file (13/18): RUNS=2 session drift is ±3 on this bench; rank only on gaps larger than that.

### bench_08_horizon chain (NS 1,4,8,16; REPS 10; 290 steps per fill)

| fill | leader | groot | enru-mtp | swift | gpt-oss | o122 |
|---|---|---|---|---|---|---|
| 0 | 0.903 ±.017 (N_max 11) | 0.976 (25) | 0.990 (63) | 0.997 (173) | 0.934 (46) | 0.914 (63) |
| 16 384 | 0.903 (9) | 0.976 (26) | 0.990 (63) | 1.000 (inf) | 0.997 (231) | 0.966 (63) |
| 32 768 | 0.962 (15) | 0.986 (49) | 0.997 (173) | 0.993 (99) | 0.993 (99) | 0.986 (49) |
| 98 304 | 0.966 (20) | 0.972 (23) | — | — | 0.997 (173) | 0.990 (63) |

Reads the same as 10-16: no fill decay in this family; the leader's per-step p is the lowest, which
matches its bench_12 profile (wins by persistence, not by per-step accuracy).

### Polish, 8 prompts, thinking off, graded 0–2 by hand (night/final-*-polish.txt)

| prompt | leader | groot | enru-mtp | swift | gpt-oss | o122 |
|---|---|---|---|---|---|---|
| RAM vs storage, 5 sentences | 2 | 0 (English with Polish words) | 1 (`opartowalność`) | 2 (`volatylową`) | 2 | 2 |
| translate: linker error | 2 | 0 (Chinese) | 2 | 2 | 2 | 2 |
| e-mail to a neighbour, ≤80 words | 1 (`Prośb`, `relak`, gender mix) | 0 (English) | 2 | 1 (`Sąsiedzku`, `cichość`) | 2 | 2 |
| photosynthesis, 3 sentences | 1 (`Fotosyntezę to`, `Fotosyntezę jest`) | 0 (English) | 2 | 2 | 2 | 1 (4 sentences) |
| translate: rollback | 2 | 0 (restarts, never finishes) | 0 (`wdroż wstępną wersję`, `wiadność`) | 1 (`wycofaj się do`) | 2 | 2 |
| decline `koń` | 1 (msc. `koni`) | 0 | 0 (1 179 tokens, `konio`) | 1 (dop. `kon`, woł. `koń`) | 1 (cel. `koni`, woł. `koń`) | 2 |
| `wziąć` vs `brać` | 1 (`aspectem`, Cyrillic, broken example) | 0 | 0 (aorist theory, `Zajmę wziąć`) | 0 (claims `brać` is archaic) | 1 (`niedokładny`) | 2 |
| OS update, 4 steps | 2 | 0 (English) | 2 | 2 | 2 | 2 |
| **total /16** | **12** | **0** | **9** | **11** | **14** | **15** |

### Phase A (Aider hard 20) — no data

All five models reported 0/20 in ten seconds, rc=127: `/mnt/cc/strata/bench-public/venv/bin/python`
pointed at Python 3.13, removed by the 2026-11-02 ML stack bump. Venv rebuilt on 3.14 (same pins)
2026-11-05 14:48; rerun `night/q-aider-1105.sh` chained after this queue, same TAG `final-1105-aider`.
Lesson: a phase that ends in seconds with a round 0/N for every model is `blocked`, not a result
(memory `reference_venv_python_bump`).
