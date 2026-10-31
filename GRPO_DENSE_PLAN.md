# GRPO redesign: stop learning format, learn faster from partial credit

Status: PLAN, written 2026-09-08. Nothing below is implemented yet.
Target scripts: new `grpo_11_*` family (do not retrofit grpo_01–10).
Research basis: two literature sweeps (2025–2026 papers) cross-checked against
the local TRL 1.10.0.dev0 source and the grpo_07 / grpo_07_2 run logs.

## 1. Diagnosis (why the current recipe learns slowly)

Current reward stack (grpo_07/07_2/10, inherited from the willccbb GSM8K notebook):

| reward | max | what it rewards |
|---|---|---|
| starts_with_reasoning_tag | 1.0 | format |
| xmlcount | 0.5 | format |
| soft_format | 0.5 | format |
| strict_format | 0.5 | format |
| int | 0.5 | answer is an integer |
| correctness | 2.0 | exact numeric match |

Three compounding problems, all measured on our own runs:

1. **Format outweighs correctness (3.0 vs 2.0).** With `scale_rewards="group"` the
   group std-normalisation lets the highest-variance component dominate the
   advantage (MO-GRPO 2509.22047, GDPO 2601.05242). Result of grpo_07: format
   +33 pp, accuracy +1.4 pp. The optimiser did what the reward asked.
2. **`num_generations=2` is a coin flip.** With base accuracy p=0.86 the chance
   that both completions agree on correctness is p²+(1-p)² = 0.76 at G=2,
   0.55 at G=4, 0.30 at G=8. Measured: `frac_reward_zero_std` 0.30→0.79 over the
   epoch. In those groups the only remaining signal is format/int noise.
   Every published ablation says G=4 is the minimum and G=2 underperforms
   (2512.07611 on GSM8K 1.5B; BiasGRPO 2606.04807).
3. **GSM8K is saturated for Phi-4-mini** (86.3% base). Even a perfect reward
   cannot buy much accuracy on this set; the detectable headroom is ~2–4 pp
   and needs a paired full-test eval to see. Dense reward pays off on data
   where the base model is in the 20–80% band.

Side finding: `beta=0.001` with zero-std groups still leaks a KL gradient
(trl#5588); `beta=0` fixes it and skips the reference-model forward entirely.

## 2. Design decisions

### 2.1 Reward = answer only, format as a negative gate
- One binary term `correct` (1.0 / 0.0). Extraction stays lenient
  (`<answer>` tag → `####` → last number), so **no format reward is needed**:
  an unparseable answer scores 0 by construction (Open-Reasoner-Zero
  2503.24290, SimpleRL-Zoo 2503.18892, Dr. GRPO 2503.20783; unsloth's own
  Qwen3 notebook now SFTs the format first "so GRPO stops learning
  formatting").
- Prompt still *asks* for a final `#### <number>` line (cheap, no reward).
- Anti-hacking gate: more than one `<answer>` tag or more than one `####`
  line → 0.0 (not `None`; `None` removes the row from the group and wastes it).

### 2.2 Partial credit from GSM8K's own step annotations
GSM8K gold solutions carry `<<a*b=c>>` calculator annotations: median 3,
mean 3.17 per problem, only 95/7473 without any. "Reward Granularity in RLVR"
(2607.02869) used exactly this: `R_step = matched_gold_values / total_gold_values`
(match within 1e-5 relative), process-only 63.7 vs outcome-only 53.8 on
Qwen2.5-0.5B. Caveats: single seed, cold-start regime (33% base), and a
small process weight (λ=0.1) was *worse* than outcome-only — the mix ratio is
an experiment, not a constant.

Design:
```python
# dataset map adds: answer (final), steps (list[float] of gold intermediate values, deduped, in order)
def step_reward(completions, steps, **kw):
    body = completion text before the final answer line
    nums = ordered-unique floats found in body
    hit  = |{v in steps : some n in nums with |n-v| <= 1e-5*max(1,|v|)}| / |steps|
    return hit * (1.0 if len(nums) <= 1.5*len(steps)+3 else 0.5)   # anti number-spray
```
Rows with empty `steps` return `None` for this term (TRL skips it).

### 2.3 Optional bounded closeness (third term, small weight)
Quadratic tolerance `(1-d/τ)²` for relative error d<τ=0.05, else 0, and
`None` when |gold| < 20 (for small integers "close" is semantically wrong;
2509.04457 got +4.3 pp over binary on chart QA with τ=0.02). Off by default;
A/B it after 2.1+2.2 are measured.

### 2.4 Combine with GDPO, not a raw sum
TRL 1.10 field: `multi_objective_aggregation="normalize_then_sum"` (per-reward
group normalisation, then weighted sum, then batch normalisation). With it,
`reward_weights` are the real mixing ratios. Start `[1.0, 0.5]` for
`[correct, step]`; sweep 0.25 / 0.5 / 1.0.

### 2.5 Group size and loss
```python
GRPOConfig(
    num_generations=8,              # G=4 as the fallback if VRAM says no
    generation_batch_size=8,        # keep == num_generations: the unsloth logp path scores the
                                    # whole generation batch (matmul_lora OOM fix, 2026-07-20)
    unsloth_grpo_mini_batch=1, unsloth_logit_chunk_multiplier=16,
    max_completion_length=768,      # GSM8K completions average ~160 tokens; 2048 was wasted KV
    loss_type="dapo", epsilon=0.2, epsilon_high=0.28,
    beta=0.0,                       # no KL leak from zero-std groups, no ref forward
    scale_rewards="batch",          # Dr.GRPO/PPO-Lite: no per-question 1/std amplification
    mask_truncated_completions=True,
    multi_objective_aggregation="normalize_then_sum",
    reward_weights=[1.0, 0.5],
    num_iterations=2,               # reuse each rollout twice; clip-higher makes this safe
    temperature=1.0,
)
```
Do NOT use `importance_sampling_level="sequence"` with dapo (TRL refuses).
Do NOT use `UNSLOTH_VLLM_STANDBY` (corrupts CuMemAllocator; see memory).

### 2.6 LoRA
Keep `lora_alpha == lora_rank` (bug (c) workaround, s=1) but raise the
learning rate: LoRA-Without-Regret and Tina put RL LoRA at ~10× full-FT lr;
our 4e-6 at s=1 is 2–5× low. Plan: 1e-5, watch `clip_ratio`, back off to
5e-6 above ~0.2. Rank 8 suffices for RL (rank-1 matches full FT in Tina).

### 2.7 Data: offline dynamic sampling, then harder data
- **Pre-score** the 7,473 train prompts once with the base model in vLLM
  alone (`n=8, temperature=1.0, max_tokens=768, prefix caching`), ~1 GPU-hour.
  Keep prompts with 1 ≤ correct ≤ 7 of 8. Expect ~1.5–2k survivors at 86%
  base. This is DAPO dynamic sampling done offline; TRL has no online
  version and `None` rewards do not resample. Re-score after ~300 steps.
- **Next dataset** once GSM8K signal is exhausted: MATH train Level 3–5
  (`\boxed{}`, fits 2k ctx, MIT) or Skywork-OR1-RL-Data filtered by the
  1.5B difficulty column 4–12 (ready-made pass-rate band). Both need
  `math-verify` (ebuild in overlay `stuff`, not installed). Keep
  GSM8K-Platinum (1,209, cleaned labels) and GSM-Plus as eval only —
  GSM-Plus is built from the GSM8K *test* set, never train on it.

### 2.8 Evaluation that can see the gain
Binomial half-widths at p≈0.86: ±3.0 pp (n=500), ±1.9 pp (n=1,319).
A +2 pp gain is visible only with a **paired** greedy eval on the full test
set (McNemar on discordant items). Sampled pass@8 on a 200-item slice is a
collapse/diversity monitor, not the decision metric. Eval every 100 steps;
Tina found best checkpoints at 30–60% of an epoch, unsloth says judge after
step 300.

## 3. Model choice
- **Develop on grpo_10 lineage (Qwen3-1.7B, bf16 both sides, healthy):**
  fastest iteration, lower base accuracy → more informative groups.
- **Confirm on Phi-4-mini (grpo_07 lineage, 4-bit, healthy).**
- Qwen3-4B stays parked: bug (a) needs bf16 on both sides = 2×8 GB, no fit.

## 4. Phases

| # | step | output | gate |
|---|---|---|---|
| 0 | `grpo_prescore.py`: base pass@8 per train prompt → `gsm8k_train_pass.jsonl` | pass-rate histogram | ≥1k prompts in 1–7/8 band |
| 0 | `grpo_eval.py`: greedy, format-agnostic, paired McNemar vs base, GSM8K test + Platinum | baseline numbers for Qwen3-1.7B and Phi-4-mini | — |
| 1 | `grpo_rewards.py` shared module: `correct`, `step_reward`, `closeness`, extractor; `test_grpo_rewards.py` incl. hacking cases (number spray, double tag) | unit tests green | — |
| 2 | `grpo_11_qwen3_17b_dense.py`: config §2.5, rewards `[correct]` only, filtered data, G=8 | smoke 5 steps: VRAM, non-finite=0, |B|max sane | fits at G=8 else G=4 |
| 3 | A/B on 300 steps: (a) correct only, (b) +step 0.5, (c) +step 1.0, (d) +closeness | paired eval curves | pick weights |
| 4 | Transfer winner to Phi-4-mini; full run; paired eval | accuracy vs 87.7% (old GRPO) | +2 pp paired significant |
| 5 | Switch data to MATH L3–5 / Skywork band; install math-verify | first non-GSM8K run | — |
| 6 | Optional loss-level credit: GRPO-λ (2510.00194) or NSR (2506.01347) via `_compute_loss` subclass | — | only if 3–5 plateau |

Expected effect per GPU-hour (from the two sweeps, order of magnitude):
G=8 + batch-8 rollouts 3–5× wall-clock and real advantages; offline filter
3–5× fewer dead groups; `num_iterations=2` + `beta=0` ~1.6–2× gradient
steps per rollout and no ref forward; lr fix 1.5–2× faster reward rise.
Partial credit itself: +? on GSM8K (unknown in the 86% regime), the
documented win is in the cold-start regime.

## 4b. EXECUTION LOG (2026-09-08)

**Decision taken by the user: skip the GSM8K A/B, go straight to MATH L3-5, build the
new family from scratch, iterate by experiment rather than by up-front design.**

Done:
- `math-verify-0.9.0` + `latex2sympy2-extended-1.11.0` installed from overlay `stuff`
  (needed `/etc/portage/package.accept_keywords/math-verify`, both are `~amd64`).
- `grpo_11_qwen3_17b_math.py` — Qwen3-1.7B bf16, MATH-lighteval Level 3-5, 5,525 train
  prompts after filtering (`\boxed` present, prompt <= 512 tok). ONE mandatory reward
  (`correct`, math_verify), everything else behind env knobs so one run differs from the
  previous in exactly one variable.
- `grpo_11_qwen3_17b_math_test.py` — greedy MATH-500 eval + exact McNemar paired compare.
  Helper unit tests pass (hedging, unparseable, LaTeX equivalence, `\sqrt{25}` vs `5`).

Three runtime traps found and fixed while smoking the script:
1. `str.format` on a prompt template containing a literal `\boxed{}` raises IndexError.
   Use `.replace("{problem}", ...)`.
2. `max_completion_length=1024` truncated 25-50% of completions (Qwen3-1.7B no-think
   writes ~900-1000 tokens on L3-5). Default is now 2048.
3. **NaN loss with `loss_type="dapo"` (the real blocker).** Unsloth's patched GRPOConfig
   FORCES `mask_truncated_completions=True` whenever loss is dapo ("Unsloth: The DAPO
   paper recommends ... we will set it"), and the compiled dapo branch normalises by
   `num_items_in_batch` with NO clamp. A micro-batch whose completions are all truncated
   gets its mask zeroed -> 0 valid tokens -> 0/0 -> NaN loss -> 196/196 non-finite LoRA-B
   grads. Proven by the NAN-WATCH probe (`GRPO_NANWATCH=1`): the two failing calls are
   exactly `lens=[0, 0]`. Not the packing path (disabling `UNSLOTH_GRPO_SEQ_PACKING`
   changed nothing) and not autograd anomaly mode (that one dies on
   GuardOnDataDependentSymNode inside the compiled forward — useless here).
   **Default loss is now `bnpo`**: same token-level aggregation, `clamp(min=1.0)` on the
   denominator, no forced masking.

Smoke 3/3 GREEN (`grpo11_smoke8_bnpo.log`): loss finite (0, -0.24, +0.026), grad_norm
0.028-0.039, non-finite B grads 0/196 on every step, |B|max 1.5e-07, peak VRAM 11.7 GiB
of 16, 132 s for 3 steps (~44 s/step, vs 98 s/step while the dapo path was NaN-ing).
Reward signal is alive at G=4: `frac_reward_zero_std` 1.0 / 0.5 / 0.75, correctness mean
0.5 / 0.25 / 0.69 — i.e. MATH L3-5 puts Qwen3-1.7B in the productive band, unlike GSM8K.
`hedged=3` of 12 groups: the hedging gate is doing real work.

### Baseline (2026-09-08, `eval_math500_base.jsonl`)

Qwen3-1.7B, no training, greedy, 2048 max tokens, training prompt, math_verify scoring:

| level | n | correct |
|---|---|---|
| 1 | 43 | 95.3% |
| 2 | 90 | 83.3% |
| 3 | 105 | 84.8% |
| 4 | 128 | 67.2% |
| 5 | 134 | 47.8% |
| **all** | **500** | **71.0%** |

Status split: 355 ok, 134 wrong, 11 hedged, **0 unparsed** — the model always emits a
parseable answer, which confirms the design call: no format reward is needed, the
verifier alone carries the signal. 47/500 completions hit the 2048-token cap and only
1 of those was correct, so truncation ~= wrong; a length penalty is redundant for now.
Median completion 537 tokens.

Training is on L3-5, where the base sits at 47.8-84.8% — the productive band. MATH-500
overlaps the MATH test split, not the train split, so it is a clean held-out set.

### Trap: a finished vLLM eval keeps the GPU (2026-09-08)

`grpo_11_..._test.py` printed its results and wrote the jsonl, but the process stayed
alive — vLLM V1's `EngineCore` runs as a child and nothing tears it down at the end of
`main()`. It sat there holding 13 GiB, and the next training run died with
`CUDA out of memory ... 200 MiB free` while loading the model. Fixed in the eval script:
`llm.llm_engine.shutdown()` + `del llm` + `gc.collect()` + `os._exit(0)` after the last
print. Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before
blaming a config for an OOM; a `VLLM::EngineCore` with a long ELAPSED is a leftover.

### e0 result — answer-only reward, lr 4e-6, 200 steps (2026-09-08)

Training: 200 steps, 7,159 s (~36 s/step), 0 NaN, |B|max grows monotonically, 800 groups
scored, 116 completions (3.6%) hit the hedging gate. Reward per 25-step bucket:
0.540, 0.568, 0.608, 0.542, 0.630, 0.578, 0.650, 0.593 — a weak upward drift swamped by
which prompts each bucket happened to draw.

Paired MATH-500 (greedy, same 500 items, exact McNemar):

| | base | e0 |
|---|---|---|
| accuracy | 70.20% | 71.60% |
| discordant | 23 base-only | 30 e0-only |
| | delta +1.40 pp | **p = 0.41 — not significant** |

**Null result, as designed to be readable.** 200 steps at lr 4e-6 move nothing measurable.
This is the evidence for reordering the queue: testing a reward change on top of a
learning rate that produces no learning compares two flat curves. Next runs are
e2 (lr 1e-5, answer-only) and e3 (lr 1e-5 + step credit).

### Offline reward analysis (free, no GPU) — `grpo_11_analyze.py`

Measured on the 500 evaluated completions of each model, correct vs wrong:

| signal | base AUC | e0 AUC | reading |
|---|---|---|---|
| step credit (with anti-spray) | 0.656 | 0.689 | real signal, worth training on |
| raw hit rate (no anti-spray) | 0.629 | 0.658 | the anti-spray term adds ~0.03 |
| completion length | 0.138 | 0.157 | wrong answers are 2.5x LONGER |

Three conclusions:
1. **Step credit is not noise** (AUC 0.66-0.69 vs 0.5 for a worthless term), so e3 is
   justified. Note it is measured *against* a length confound: wrong completions are
   longer and therefore mention more numbers, which should have pushed this AUC toward
   0.5. The signal survives that.
2. **Do NOT add a length penalty**, and keep `GRPO_OVERLONG=0`. Wrong completions average
   1,303 tokens vs 509 for correct ones, but that is the model rambling and looping when
   it cannot solve the problem. Rewarding shortness would train it to bail out early on
   exactly the hard problems we are trying to teach. The correlation is real and the
   causal direction makes it a trap.
3. **The hedging gate is cheap and stays.** It fires on 9-11 of 500 completions at eval
   (3.6% during training); of those, 27% (base) and 67% (e0) would have been scored
   correct by their last boxed answer. So the gate costs ~1-2% of completions their
   reward, in exchange for not teaching the model to spray candidate answers.

### BUG FOUND 2026-09-08: the adapter never reached the rollout engine

**Every GRPO run with `fast_inference=True` on this stack sampled its rollouts from the
frozen BASE model.** The gradient still updated the adapter, so nothing looked wrong —
tripwires clean, loss finite, |B| growing, reward plausible — but the behaviour policy
never improved, which is the whole point of the algorithm.

How it was caught: e2 (lr 1e-5) produced a **bit-identical** reward sequence to e0
(lr 4e-6) for all 106 steps. Two learning rates cannot give identical samples unless the
sampling model is identical. Both were sampling the untrained base.

Root cause: unsloth ships the adapter to its colocated engine by REWRITING the TRL
trainer source with a regex (`unsloth/models/rl.py` ~2225) that appends
`lora_request = self.model.load_lora(..., load_tensors=True)` to `self.llm.generate(`
calls. TRL 1.10 moved generation into `trl/generation/vllm_generation.py`, so the regex
matches nothing: `grep -c lora_request unsloth_compiled_cache/UnslothGRPOTrainer.py` = 0.
Unsloth even logs its own failure and continues anyway:

    [unsloth_zoo.log|WARNING] Unsloth: Warning - regex did not match,
    VLLMGeneration.generate patch may have failed

Fix (`GRPO_LORA_ROLLOUT=1`, default on): wrap the engine's own `LLM.generate` and attach
the live adapter via `model.load_lora(dir, load_tensors=True)` on every call. It raises
instead of continuing if `trainer.vllm_generation.llm` is missing, so a future TRL
refactor fails loudly rather than silently reverting to base sampling. `=0` reproduces
the broken behaviour on purpose.

Validation (`grpo11_lorafix_smoke.log`, lr 1e-5, 6 steps):

| step | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| e0 (broken) | 0.500 | 0.250 | 0.6875 | 0.875 | 0.4375 | 0.8125 |
| fix armed | 0.500 | 0.375 | 0.6875 | 0.750 | 0.3125 | 0.750 |

Identical at step 1 (the adapter is still zero, so it *must* match) and divergent
afterwards — the signature of a working sync. Cost ~39 s/step vs 36 s/step.

**Consequence: e0 and e2 measured a frozen-policy pipeline and are void as GRPO results.**
They stay on record as the broken baseline. Reruns with the fix: f1 (lr 1e-5,
answer-only) and f2 (lr 1e-5 + step credit). e2's 106 logged steps are still useful:
paired against f1 they isolate the value of the fix itself, same lr, same prompts.

### f1 / f2 — first runs with working rollouts (2026-09-08)

**f1 (lr 1e-5, answer-only, fix ON), 200 steps.** Stable: 0 non-finite gradients, no
tripwire, |B|max 1.3e-3, ~39 s/step. Paired MATH-500 vs base: 70.20% -> 71.20%,
**p = 0.58 — null again.** Per level the changes are noise in both directions
(L5 44.8->49.3, L3 81.9->85.7, L4 69.5->67.2, L1 97.7->95.3).

**Three 200-step runs, three nulls on the 500-item eval.** That instrument has a +-3 pp
half-width, so a run that moves accuracy by 1-2 pp cannot be resolved by it. The problem
is the measurement, not only the treatment — see the dev-eval curve added below.

### THE REAL BOTTLENECK: two thirds of the rollout budget buys nothing

A GRPO group produces gradient only if its completions disagree. Measured on f1 (G=4):

| dead prompts in the step | share of steps |
|---|---|
| 1 of 4 | 13% |
| 2 of 4 | 28% |
| 3 of 4 | 37% |
| 4 of 4 | 23% |

Mean `frac_reward_zero_std` = **0.672 — 2.7 of every 4 prompts per step are dead**, and
23% of steps are entirely wasted. This dwarfs any reward-scale tuning.

### f2 — partial credit: the mechanism is CONFIRMED

f2 = f1 + step credit (`GRPO_STEP=1 GRPO_W_STEP=0.5`, GDPO per-term group normalisation).
First 31 steps:

| | f1 (answer only) | f2 (+ step credit) |
|---|---|---|
| mean `frac_reward_zero_std` | 0.672 | **0.298** |
| steps with NO dead prompt | 0% (min was 0.25) | 32% |

**Dead groups more than halved.** This is exactly the predicted mechanism: when all four
completions are wrong, binary correctness gives them identical zeros and no gradient,
while step credit separates the completion that reproduced three gold intermediate values
from the one that reproduced none. GDPO normalisation makes it work: the zero-variance
correctness term drops out of the group and the advantage comes purely from partial credit.

Task metric so far (paired per step on correctness only, identical prompts):
f2 - f1 = +0.010 +- 0.016, t = 0.64 over 31 steps — too early to call. The effect of
denser signal is cumulative, so it needs the full run and probably more steps than 200.

### Instrument fix: in-training dev eval (`GRPO_DEVEVAL=N`)

Every N steps, greedy-score a fixed MATH-500 slice with the CURRENT adapter on the engine
that is already resident (~2-4 min for 200 items). Successive points are paired: the slice
and decoding are fixed, only the policy changes. Turns each run into a learning curve
instead of one endpoint, which is what three null endpoints showed was needed.

### Difficulty pre-scoring done (2026-09-08 18:19) — only 30% of the data can teach anything

`grpo_11_prescore.py`, 5525 MATH L3-5 prompts x 4 samples at the training temperature,
~55 min. Base Qwen3-1.7B pass-count histogram: `{0: 1727, 1: 539, 2: 489, 3: 628, 4: 2142}`.

| band | prompts | share | value to GRPO |
|---|---|---|---|
| pass 0 (always wrong) | 1727 | 31% | none — group agrees, advantage 0 |
| pass 1-3 (informative) | **1656** | **30%** | the only prompts that can produce gradient |
| pass 4 (always right) | 2142 | 39% | none |

Operational notes: `gpu_memory_utilization=0.85` OOMs while the KDE session holds ~2 GB, use
0.75; prompts over 512 tokens must be dropped or the engine rejects the whole batch
("maximum context length is 1536 tokens") — 59 dropped. The file is reusable by every
later run via `GRPO_PASS_FILTER`.

### g1 — the first run designed to be readable (in flight)

`grpo_11_phase2.sh`: 800 steps, lr 1e-5, partial credit (w=0.5) + pass filter, dev-eval
every 50 steps, then a final paired eval. Filter applied: `5525 -> 1656 prompts`.
At step 50: dev-eval 72.00%, `zero_std` ~0.19 (f1 0.68, f2 0.31), clip ratio 0.03 vs
0.10-0.15 in f1/f2, ~40 s/step. Both levers compose as intended — three quarters of the
rollout budget now lands on prompts that can move the policy, against one third before.


### g1 mid-run readout (step 550/800, 2026-09-09)

Dev-eval on MATH-500[:200], greedy, every 50 steps:

| step | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| acc | 72.0 | 71.5 | 72.5 | 71.5 | 75.5 | 72.0 | 73.5 | 74.0 | 76.0 | 74.5 | 72.0 |

**Base on the same 200-item slice: 71.0%** (142/200, from `eval_math500_base.jsonl`; the
full-500 base is 70.2%). Mean of the curve +2.2 pp, peak +5.0 pp, 10 of 11 points at or
above base. Suggestive only — the points share a trajectory and a slice, so they are not
independent samples; the paired McNemar on the full 500 at the end decides.

Training health across all 550 steps: reward 0.84-0.90, `frac_reward_zero_std` 0.17-0.21
(f1 was 0.68), clip ratio ~0.0007 (corrected 10-20), |grad| ~0.05 flat, no NaN. The two levers are holding
for the whole run, not just the first steps.

Instrument fixes applied for the NEXT run (the live one kept the old code): dev-eval now
also fires at step 0 (= the base model on this slice, since the adapter is still zero) and
writes per-item hits to `<model_path>-outputs/deveval_items.jsonl`, so any two points can be
compared with an exact paired test instead of two bare accuracies.


### g1 FINAL (800 steps, 2026-09-09)

Training was healthy for all 800 steps: `frac_reward_zero_std` 0.17-0.21, clip_ratio/region_mean ~0.0007 (corrected 10-20; "0.03-0.05" was wrong),
0/196 non-finite grads, `|B|max` 3.554e-03 and flat after ~step 700 (converged, not cut
short).

The first reading of this run was a null (+0.6 pp, p = 0.80) taken from the **merged**
model. That reading was an artefact — see the next section. Measured through the adapter,
which is what the training curve was measuring all along:

| metric | base | g1 (adapter) |
|---|---|---|
| MATH-500 greedy, paired | 70.20% | **72.40%** (+2.2 pp, McNemar p = 0.18) |
| discordant | 23 base-only | 34 g1-only |
| L1 / L2 / L3 / L4 / L5 | 97.7 / 82.2 / 81.9 / 69.5 / 44.8 | 97.7 / 83.3 / 83.8 / 73.4 / 47.0 |
| dev slice MATH-500[:200] | 142/200 | 150/200 — the curve's 75.0% reproduces exactly |
| same run, merged weights | 70.20% | 70.80% (the artefact) |

**What this settles.** The dev curve was not lying; the merge was. Gains at every level 2-5,
no band redistribution (the earlier "L5 up, L3/L4 down" was merge noise). Still not
significant on 500 items at p = 0.18 — +2.2 pp needs ~1500 items or a repeat seed to clear
the bar, and the noise floor below is 2 pp wide.

**What it opens.**
1. A repeat seed or a bigger eval set to push +2.2 pp past significance. Cheapest real test.
2. Levers not yet pulled: LoRA rank/lr (rank 16 at lr 1e-5 converged by step 700), G=8
   instead of 4, and a harder base band (the filter left 1656 prompts).


### THE MERGE WAS EATING THE RESULT: bf16 cannot store a small LoRA update (2026-09-09)

Every `grpo_11` verdict before this date was read off a **merged** bf16 model. Measured
against the base checkpoint, tensor by tensor:

| quantity | value |
|---|---|
| `|BA|` max, per adapted tensor | ~5e-5 … 1e-4 |
| `|W|` rms | ~3.3e-2 → bf16 ULP ~1.2e-4 |
| median `|BA|` / half-ULP | **0.13** |
| fraction of the intended update that survived the merge | **6.1%** |
| weights left bit-identical to base | **93%** |
| sign agreement where nonzero | 100.0% |

bf16 keeps 8 mantissa bits. This adapter's per-weight update is ~1/8 of the rounding
threshold, so `W + BA` rounds back to `W` almost everywhere. The merge arithmetic is
correct — 100% sign agreement on the entries that moved — the result is simply not
representable. **Saving fp32 does not fix it**: any bf16 runtime re-quantises at load.
For updates this small the merged model is not a model of the run.

Re-measured through the adapter path (`--lora`), all four runs:

| run | merged | adapter | d(merged) | d(adapter) |
|---|---|---|---|---|
| e0 | 71.60% | 70.20% | +1.40 | +0.00 |
| f1 | 71.20% | 69.40% | +1.00 | -0.80 |
| f2 | 69.60% | 70.40% | -0.60 | +0.20 |
| g1 | 70.80% | **72.40%** | +0.60 | **+2.20** |

e0/f1/f2 stay null either way, so no earlier conclusion reverses. But note what the merged
column is: four measurements of (approximately) the base model, scattered over
69.6-71.6%. **That 2 pp spread is the unpaired greedy-decode noise floor on 500 items** —
useful calibration, and the reason a bare accuracy pair is never enough.

**Why the old tripwire missed it.** It was built for the opposite failure, the 2026-07-24
aliasing bug where `|B|` exploded to ~1e13. It checked for values too *large* and said
nothing about values too small to record.

**Fix shipped.** `grpo_merge.py:merge_fidelity()` samples 8 adapted tensors after the merge,
reports the surviving fraction, and warns with the ready-made `--lora` command when it is
under 50%. Verified on g1: 5.7% on the sample vs 6.1% across all 196 tensors.
`grpo_11_qwen3_17b_math_test.py` gained `--lora ADAPTER_DIR` (engine `enable_lora=True` +
a `LoRARequest`), which is now the path that matches the in-training dev eval.

## 5. Open decisions for the user
1. Shared `llm/grpo_rewards.py` vs the self-contained-script convention
   (21 copies of the extractor already hurt: the 2026-07-19 fix touched all).
2. Whether to spend a GPU-hour on pre-scoring or reuse the frontier estimate
   from a first short run (pre-scoring is cleaner; recommended).
3. Whether to move to MATH L3–5 right after phase 2 instead of finishing the
   GSM8K A/B — GSM8K headroom is small; the A/B mostly validates the
   reward code, not the final gain.

## 6. Key sources
DAPO 2503.14476 · Dr. GRPO 2503.20783 · GDPO 2601.05242 · MO-GRPO 2509.22047 ·
Reward Granularity (GSM8K step credit) 2607.02869 · closeness 2509.04457 ·
G ablation 2512.07611 · BiasGRPO 2606.04807 · SPEED-RL 2506.09016 ·
Prompt Curriculum Learning (Gao et al., ICLR 2026) · Open-Reasoner-Zero 2503.24290 ·
SimpleRL-Zoo 2503.18892 · NSR 2506.01347 · GRPO-λ 2510.00194 · Tina 2504.15777 ·
LoRA Without Regret (thinkingmachines.ai/blog/lora) · trl#5588 (KL leak) ·
trl#4764 (no dynamic sampling) · Skywork/Skywork-OR1-RL-Data · madrylab/gsm8k-platinum.

## Status 2026-10-19 (stack: trl 1.10 / unsloth 2026.10.6 / torch 2.15 / CUDA 13.4.3)

Measured stack: torch 2.15.0a0+gitdd1fefe, transformers 5.20.1, trl 1.10.0.dev0 (GRPOConfig
193 dataclass fields incl. the 3 unsloth_* ones), peft 0.20.1.dev0, unsloth / unsloth_zoo
2026.10.6, vllm 9999 (pin 7a100bb6), gcc 16. Evidence and logs:
`night/upgrade-check-grpo-1019/` (REPORT.md there has the per-script table).

### What of this plan grpo_11 already implements

| plan item | grpo_11 today | knob |
|---|---|---|
| 2.1 answer-only reward, format as a gate | yes: math_verify `correct`, hedging gate (2 different `\boxed{}` -> 0) | always on |
| 2.2 partial credit | yes: fraction of gold intermediate numbers, anti-spray halving | `GRPO_STEP=1 GRPO_W_STEP=0.5` |
| 2.4 GDPO aggregation | yes | `GRPO_AGG=normalize_then_sum` (default) |
| 2.5 group size | G=4 (G=8 never tried on 16 GB) | `GRPO_G` |
| 2.5 beta=0 | yes (default) | `GRPO_BETA` |
| 2.5 scale_rewards=batch | yes (default) | `GRPO_SCALE` |
| 2.5 loss | bnpo, not dapo (Sept NaN finding) | `GRPO_LOSS` |
| 2.6 lr 1e-5, alpha == rank | yes in g1 (script default still 4e-6) | `GRPO_LR` |
| 2.7 offline dynamic sampling | yes: `math_l35_pass.jsonl`, 1656 informative prompts | `GRPO_PASS_FILTER` |
| 2.8 paired eval + curve | yes: `--compare` exact McNemar, `GRPO_DEVEVAL`, `--lora` eval path | — |
| 6 GRPO-λ / NSR | no | — |

### g1 re-measured on the new stack: the +2.2 pp is NOT reproduced

Same adapter (`outputs/lora-grpo-qwen3-17b-math-g1-outputs/adapter-final`), same script,
greedy, 2048 tokens, MATH-500:

| eval | Sept stack | today |
|---|---|---|
| base, MATH-500[:200] | 142/200 (71.0%) | 147/200 (73.5%), repeat 147/200 |
| g1 adapter, MATH-500[:200] | 150/200 (75.0%) | 141/200 (70.5%) |
| base, full 500 | 70.20% | 71.60% |
| g1 adapter, full 500 | 72.40% (+2.2, p=0.18) | **70.60% (-1.0 pp, 33 base-only / 28 g1-only, p=0.61)** |

Greedy vLLM output does not reproduce across the upgrade, and is not fully stable even within one
stack:
- Sept base vs today's base: 121/500 completions byte-identical (54 of the first 200).
- Two identical base runs today, back to back: 162/200 completions identical, with 6/200 items
  flipping correct/wrong (3 each way). Greedy decoding with batched vLLM kernels is not
  batch-invariant.
- The adapter is applied: today's base and g1 outputs share only 21/200 identical completions,
  well below the 162/200 a same-stack repeat gives.

Reading: g1's +2.2 pp sits inside the run-to-run spread of the instrument itself. The cross-stack
swing (base +1.4 pp, g1 -1.8 pp, both n.s.) is about as large as the effect we tried to measure.
**g1 is no longer evidence of a gain, and on this stack it is a null.** One greedy pass over 500 items
cannot settle +2 pp. The decision metric must average over decoding noise:
(a) several greedy passes over the full set, or a sampled pass@1 averaged over k=4-8 seeds per item
(each item then gets a success rate instead of a 0/1), paired per item; and (b) a bigger item pool
(MATH test L3-5 outside MATH-500 adds ~4.5k items).

### Stack changes that matter for this plan (installed code, not changelogs)

1. **The "LoRA never reached the rollouts" bug is fixed upstream.** `unsloth/models/rl_replacements.py`
   `vllm_generation_init_patch` no longer regex-patches TRL's generate. It wraps the vLLM engine's
   `generate`/`chat` for the duration of each `VLLMGeneration.generate` call and injects
   `load_lora(..., load_tensors=True)`. Probe with `GRPO_LORA_ROLLOUT=0` (grpo_11's own fix off):
   `[LORA-PROBE] total engine generate calls=12 with lora_request=12`. As a consequence the old
   detector (`grep -c lora_request unsloth_compiled_cache/UnslothGRPOTrainer.py`) now reads **0 on
   a healthy run** and is obsolete. grpo_11 keeps `GRPO_LORA_ROLLOUT=1`; upstream skips a call that
   already carries a request, so the adapter is not loaded twice.
2. **The dapo NaN mechanism is gone in the installed code**, both halves of it.
   `unsloth_zoo/rl_replacements.py` now floors the normaliser ("Floor at 1 like TRL: a fully masked
   batch ... is 0/0 = nan otherwise"). `unsloth/models/rl.py` forces `mask_truncated_completions=True`
   only for an EXPLICIT `loss_type='dapo'` ("Only an explicit dapo gets the paper's settings").
   Also, **bnpo now normalises by the global token count, the same as dapo** ("bnpo too: a per
   micro-batch token mean changes with the GPU / accumulation split"). So grpo_11's `bnpo` default
   is now numerically the dapo aggregation. Untested: an explicit dapo run on grpo_11.
3. Unsloth's loss path implements `importance_sampling_level` (token/sequence), `cispo`, `sapo`,
   `luspo`, `vespo`, `delta`, `off_policy_mask_threshold` and the vLLM IS correction.
   **It ignores `top_entropy_quantile`, `entropy_coef` and `use_adaptive_entropy`**
   (`_warn_unsupported_grpo_options`: "will be ignored (training proceeds as standard GRPO)").
   The "Beyond the 80/20 rule" token mask is therefore not available through unsloth.
4. Defaults moved: TRL `loss_type` defaults to `'dapo'` and `beta` to `0.0`. grpo_01-10 set neither,
   so they now train with implicit dapo without forced masking. All three smokes were finite.
   `multi_objective_aggregation` still defaults to `sum_then_normalize`; grpo_11 sets it explicitly.
5. Config-time validation no longer refuses `importance_sampling_level='sequence'` together with
   `loss_type='dapo'` (checked: the GRPOConfig constructs). The 2.5 warning above may be stale.
   Untested at trainer init.

### New installed features mapped onto the open items (all UNTESTED here)

Quoted help text is from the installed `trl.GRPOConfig` / `peft.LoraConfig` metadata.

| open item | feature | installed help (abridged) | proposal |
|---|---|---|---|
| decoding mismatch, vLLM vs trainer (greedy is not reproducible, see above) | `vllm_importance_sampling_mode` (default `sequence_mask`), `vllm_importance_sampling_clip_max` (3.0) | "Importance Sampling (IS) to correct for the mismatch between vLLM completion logprobs and recomputed training logprobs ... '*_mask' modes, ratios larger than C are set to zero" | keep it on (`GRPO_VLLM_IS=1`). Log how many tokens get masked before changing the mode; `token_truncate` is the gentler choice if masking turns out to discard many rows |
| sequence-level reward vs token-level ratio | `importance_sampling_level='sequence'` (GSPO) | "averages the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper shows that sequence-level sampling often yields more stable training" | only meaningful OFF-policy: with `num_iterations=1` the ratio is exp(0)=1 and, after the bnpo token sum, the gradient equals token-level GRPO exactly (zoo `grpo_compute_loss`; smoke `e_grpo11_gspo_smoke3.log` step-2 `grad|max=4.137e-04` = the token-level runs). So h1 needs `GRPO_ITERS=2` |
| clip/lr headroom (g1 clip_ratio/region_mean ~0.0007, corrected 10-20 = the update is almost never clipped) | `loss_type='cispo'` | "Clips the importance sampling weights instead of the advantage scaled importance weights" — with `epsilon_high` "the recommended value is 5.0" (unsloth sets 5.0 itself if unset) | after h1. `GRPO_LOSS=cispo GRPO_EPS_HIGH=5.0`. Not a cure for the old dapo NaN (that is fixed, see 2.) |
| soft gate instead of hard clip | `loss_type='sapo'` (+ `sapo_temperature_pos/neg`) | "Replaces hard clipping with a smooth, temperature-controlled gate" | low priority; same reason as cispo |
| MO-GRPO concern (high-variance term dominates) | `scale_rewards` | "`'batch'`: rewards are scaled by the standard deviation across the entire batch ... The Dr. GRPO paper recommends not scaling rewards" | keep `batch` + `normalize_then_sum`: GDPO already normalises each term per group before the sum, and batch scaling then avoids the per-question 1/std blow-up on near-dead groups. `none` is the next A/B only if step credit starts to dominate `rewards/correct_reward/mean` |
| eval cost of curves | `num_generations_eval` | "Number of generations to sample during evaluation" | not useful as-is: our dev eval is a callback on the vLLM engine, not TRL's `evaluate()`. Better: give the DevEval callback k sampled passes (see the decision metric above) |
| LoRA capacity (|B|max flat after step ~700 in g1) | `use_rslora` (`get_peft_model` has it natively) | "sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`" | `GRPO_RSLORA=1` multiplies the scale by 4 at r=16 (alpha == r). Equivalent to 4x lr on B, so pair it with `GRPO_LR=2.5e-6`; it only pays when rank goes up (r=64) |
| bf16 merge trap (small |BA|) | `use_dora` | "decomposes the updates ... into magnitude and direction" | do not adopt: unsloth's vLLM LoRA path and our merge/fidelity tooling are LoRA-only. Untested whether `load_lora` ships DoRA magnitude vectors |
| fp32 head | `cast_lm_head_to_fp32` | "only supported when the model has untied word embedding" | n/a: Qwen3-1.7B ties its embeddings |
| sleep mode | `vllm_enable_sleep_mode` | "offload weights/cache during the optimizer step" | candidate for G=8 on 16 GB. unsloth patches `sync_weights` to wake the engine ("TRL >= 1.x only wakes the engine from sleep mode inside this method"). Smoke it before any long run |

### Proposed next run (untested)

1. **Fix the instrument first (no training).** Re-evaluate base and g1 with k=4 sampled passes at
   T=0.6 (or 4 greedy passes) on full MATH-500, paired per item. If g1 is still within ±1 pp of base,
   g1 is a null and the next run must change something bigger than seed noise.
2. **h1 = g1 + 2 PPO epochs per rollout + GSPO**, 800 steps:
   `GRPO_TAG=h1 GRPO_LR=1e-5 GRPO_STEP=1 GRPO_W_STEP=0.5 GRPO_PASS_FILTER=math_l35_pass.jsonl
   GRPO_DEVEVAL=50 GRPO_DEVN=200 GRPO_ITERS=2 GRPO_IS_LEVEL=sequence GRPO_MAX_STEPS=800 python grpo_11_qwen3_17b_math.py`
   (loss stays bnpo, now dapo-like aggregation). GSPO alone at `GRPO_ITERS=1` is mathematically
   identical to token level (see the table), so the sequence ratio only matters once the second epoch
   reuses a rollout. A clean A/B needs h1a = `GRPO_ITERS=2` with token level as well. The GSPO code
   path ran 3 finite steps here (`e_grpo11_gspo_smoke3.log`, 0/196 non-finite); `GRPO_ITERS=2` has
   never run on this host.
3. Only if h1 also fails to beat base under the averaged metric: h2 = h1 with `GRPO_G=8` and
   `vllm_enable_sleep_mode` (VRAM probe first), then cispo.

Repeat seed and the 1500-item eval stay open. The finding above makes them more urgent: without
them no run can be called, in either direction.

## Results 2026-10-20 (queue `night/grpo-dense-queue.sh`, evidence `night/grpo-dense-1020/`)

Instrument: `grpo_11_qwen3_17b_math_test.py --k 8 --seed 0` = 8 samples per item at T=0.6,
top-p 0.95, 2048 tokens, full MATH-500; per-item success rate; paired sign-flip permutation
test + bootstrap CI (`--compare` on two `--k` files). Same seed for every model.

| model | mean pass@1 | vs base | 95% CI | p |
|---|---|---|---|---|
| base Qwen3-1.7B | 69.40% | | | |
| g1 (Sept, ITERS=1, token) | 70.95% | +1.55 pp | [+0.45, +2.67] | 0.008 |
| h1 (ITERS=2, GSPO sequence) | 70.05% | +0.65 pp | [-0.47, +1.75] | 0.26 |
| h1a (ITERS=2, token) | 70.38% | +0.97 pp | [-0.03, +2.02] | 0.063 |

h1 vs h1a (GSPO effect): -0.33 pp, CI [-1.38, +0.80], p = 0.58. h1 vs g1: -0.90 pp, p = 0.17.
All three trainings healthy (0/196 non-finite every step); h1 260 min, h1a 269 min for 800 steps.

Reading:
- **g1 IS a real gain (+1.5 pp)**. The 10-19 "null" was the single greedy pass: its run-to-run
  noise hid an effect the averaged metric resolves. Greedy single-pass verdicts are retired.
- **A second PPO epoch per rollout (ITERS=2) did not help** and may cost a little (h1a +0.97 vs
  g1 +1.55, not significant either way). **GSPO adds nothing measurable** over token level.
- Dev curves (greedy, 200 items) stayed flat 70-76% in both runs: they cannot see a 1 pp effect.

Next candidates (untested): repeat g1 with a second seed to size seed-to-seed spread; G=8 with
sleep mode; a harder / larger train pool. Not: more GSPO / ITERS variants.

## Run hygiene implemented 2026-10-20 (before g3; user: fix the method first)

From `research/grpo-improvements-20261020.md` section 3 (Kauldron ideas), in `grpo_11_qwen3_17b_math.py`:
- `grpo_runlib.py`: run.json manifest (config, overrides, env, git, package versions incl. torch.__version__,
  file hashes, status, smoke result), metrics.jsonl (train logs, tripwire, bf16, deveval rows), strict
  `GRPO_*` check, derived seed streams, live bf16 survival statistic, smoke assertions.
- `grpo_11_task.py`: prompt + scoring shared by train and eval (copies were logically identical); TASK_SHA.
- New knobs: GRPO_SMOKE, GRPO_RESUME=auto (unverified), GRPO_DEVK (k-sample dev eval with per-item seeds),
  GRPO_STATS_EVERY, GRPO_SEED_INIT/DATA/TRAINER. GRPO_SEED is now a ROOT seed (streams derived); unset = g1 values.
- Eval: `--item-seeds`, task_sha check against the adapter's run.json, OUT.meta.json (installed after the seed
  queue ends; staged copy `night/staged/`). `grpo_eval_ckpts.py`: post-hoc checkpoint curve.
- CPU-tested (runlib units, bf16 stat on a real PEFT layer, scorer, compare output identical). GPU verification
  = `night/grpo-hygiene-queue.sh` after the LLM benches. g3 waits for it.


## Results 2026-10-21 (queue `night/grpo-seed-queue.sh` cancelled after g2; hygiene `night/grpo-hygiene-1021/`)

| run | recipe | mean pass@1 (k=8) | vs base | 95% CI | p |
|---|---|---|---|---|---|
| base | - | 69.40% | | | |
| g1 (Sept stack) | ITERS=1, token, partial credit, pass filter | 70.95% | +1.55 pp | [+0.45, +2.67] | 0.008 |
| **g2** (g1 recipe + seeds, stack of 10-19) | same | **70.03%** | **+0.62 pp** | [-0.57, +1.88] | **0.34** |
| g2 vs g1 | | | -0.92 pp | [-2.05, +0.20] | 0.12 |

Reading: same seeds do not give the same trajectory on a new stack (vLLM sampling), so g1 and g2 are two draws of one
recipe. The recipe is worth roughly +0.5 to +1.5 pp and a 500-item eval cannot resolve that: **M5 (larger eval pool)
is the next measurement step, before any recipe change.** g3 (seed 1) postponed by the user.

Run hygiene (H1-H4, H6, M2, M4 of the backlog) verified on the GPU: smoke PASSED (12/12 rollouts carried the adapter —
all injected by unsloth 2026.10 itself), typo guard works, item-seeded eval + task_sha check work, checkpoint curve
written (step 400 69%, 800 66%, final 74% at n=50 k=2: noise, as the docstring warns). Bugs found by the smoke and
fixed: metrics_callback kwarg collision; the adapter-call counter counted only its own injections.
