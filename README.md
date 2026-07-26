# LLM Experiments

Experiments with training and fine-tuning LLM models. GPU: NVIDIA Blackwell (RTX 5070 Ti), CUDA 12.x.

Naming scheme: `grpo_NN_*` = GRPO/RL experiments (increasing model size) · `lc_NN_*` = fine-tuning classifiers · `bench_NN_*` = inference runtime benchmarks (bash)

---

## File Overview

---

### 1. Mini-LLM from Scratch in PyTorch

Goal: learn transformer fundamentals — custom BPE tokenizer, custom architecture, training on text.
Dataset: `wikimedia/wikipedia`. Config: vocab=5000, hidden=256, 4 heads, 4 layers.

| File | Description |
|------|-------------|
| `simple_llm03.py` | transformer from scratch + BPE + `TOKENIZERS_PARALLELISM` |

---

### 2. IMDB Classification from Scratch — Pure Transformer

Goal: custom transformer (no LSTM/CNN) for IMDB sentiment analysis.
Config: hidden=512, 6 layers, max_seq=512.

| File | Description |
|------|-------------|
| `simple_imdb.py` | base transformer |
| `simple_imdb_extended.py` | extended (more data / epochs) |
| `simple_imdb_augmented.py` | + data augmentation |
| `simple_imdb_trf.py` | architecture variant |
| `simple_imdb_check.py` | saved model evaluation |

---

### 3. IMDB Classification from Scratch — LSTM+CNN+Transformer Hybrid

Goal: custom hybrid architecture (Transformer → LSTM → CNN) on IMDB.

| File | Architecture | hidden / filters | train |
|------|-------------|-----------------|-------|
| `simple_lstm_cnn_trf_res.py` | Transformer + 2×LSTM + residual + CNN | 512 / 200 | 23k |
| `simple_lstm_cnn_trf_res4.py` ★ | 3×LSTM + residual + CNN filters (3,4,5,7,9) | 512 / 50 | 23k |
| `simple_lstm_cnn_check.py` | saved model evaluation | — | — |

> `_res4` best: 3 LSTM layers with skip connections + CNN filters at 5 sizes (3,4,5,7,9).

---

### 4. Fine-tuning Pre-trained Models on IMDB with LoRA (`lc_NN_*`)

Goal: IMDB sentiment classification via fine-tuning pre-trained models with LoRA/QLoRA.

#### Tools and BART

| File | Description |
|------|-------------|
| `lc_01_bart.py` | BART — translation / summarization |
| `lc_01_bart_v2.py` | BART v2 |
| `lc_02_show_distilbert.py` | DistilBERT structure inspection |

#### Encoder Models (best for classification)

| File | Model | Params | Epochs | lr |
|------|-------|--------|--------|----|
| `lc_03_distilbert_imdb.py` | distilbert-base-uncased | 66M | 15 | 2e-4 |
| `lc_03_distilbert_imdb_optuna.py` | distilbert + Optuna HPO | 66M | auto | auto |
| `lc_04_electra_imdb.py` | electra-large-discriminator | 335M | 30 | 3e-4 |
| `lc_05_roberta_imdb.py` ★ | roberta-large | 355M | 30 | 2e-4 |
| `lc_06_modernbert_imdb.py` ★ | ModernBERT-large | 395M | 10 | 2e-4 |

#### Generative Models (Gemma)

| File | Model | Params | Epochs | lr |
|------|-------|--------|--------|----|
| `lc_07_gemma2b_imdb.py` | gemma-2-2b-it | 2B | 10 | 1e-4 |
| `lc_08_gemma9b_imdb.py` | gemma-2-9b-it | 9B | 3 | 1e-4 |
| `lc_09_gemma12b_imdb.py` ★ | gemma-3-12b-pt | 12B | 3 | 1e-4 |

> **Ranking:** `ModernBERT-large` ≥ `roberta-large` ≈ `electra-large` > `gemma3-12b` > `gemma2-9b` > `gemma2-2b` > `distilbert`
>
> ModernBERT (Dec 2024): rotary embeddings, Flash Attention 2, 8192-token context, ~24% faster than RoBERTa.

---

### 5. GRPO + LoRA Fine-tuning on GSM8K (`grpo_NN_*`)

Goal: improve mathematical reasoning via GRPO (reinforcement learning).
Dataset: GSM8K (math problems). Framework: unsloth + trl.
Scheme: no suffix = training · `_cont` = continuation · `_test` = evaluation on 500 examples.

#### Models (sorted by increasing size)

| File | Model | Params | VRAM 4-bit | lr | LoRA rank | Notes |
|------|-------|--------|------------|----|-----------|----|
| `grpo_01_gemma1b_gsm8k.py` | gemma-3-1b-it | 1B | ~1GB | 3e-6 | 32 | small model baseline |
| `grpo_02_qwen15b_gsm8k.py` | Qwen2.5-1.5B-Instruct | 1.5B | ~2GB | 5e-6 | 64 | math-specialist |
| `grpo_02_qwen15b_gsm8k_cont.py` | Qwen2.5-1.5B | — | ~2GB | 5e-6 | 64 | continuation |
| `grpo_03_llama8b_gsm8k.py` | Llama-3.1-8B-Instruct | 8B | ~5GB | 5e-6 | 32 | FA2, `starts_with_reasoning_tag` |
| `grpo_03_llama8b_gsm8k_cont.py` | Llama-3.1-8B | — | ~5GB | 5e-6 | 32 | continuation from checkpoint-2400 |
| `grpo_04_deepseek_r1_8b_gsm8k.py` ★ | DeepSeek-R1-Distill-Llama-8B | 8B | ~5GB | 3e-6 | 32 | CoT baked into weights via RL |
| `grpo_05_qwen3_8b_gsm8k.py` ★ | Qwen3-8B | 8B | ~5GB | 5e-6 | 32 | hybrid `/think`, `<think>` format |
| `grpo_06_dapo_llama8b_gsm8k.py` | Llama-3.1-8B | 8B | ~5GB | 5e-6 | 32 | DAPO: beta=0, overlong penalty, gen=8 |
| `grpo_07_phi4_14b_gsm8k.py` ★ | Phi-4 | 14B | ~9GB | 2e-6 | 16 | strongest reasoning/math per param |
| `grpo_07_1_phi4_gsm8k.py` ★ | Phi-4-mini-instruct | 3.8B | ~5GB | 2e-6 | 16 | no-vLLM variant: rollouts via HF `generate`, `GRPO_ATTN_IMPL=flash_attention_2` |
| `grpo_08_qwen3_14b_gsm8k.py` ★ | Qwen3-14B | 14B | ~9GB | 2e-6 | 16 | largest fitting in 16GB |

#### Baseline Tests (no fine-tuning)

| File | Model |
|------|-------|
| `gsm8k_openai_1_test_500.py` | GPT-4o (OpenAI API) |
| `gsm8k_grokai_1_test_500.py` | Grok (xAI API) |

#### Quality Hierarchy on GSM8K after GRPO

```
Qwen3-14B ≈ Phi-4  >  DeepSeek-R1-Distill-8B  >  Qwen3-8B  >  Llama-3.1-8B  >  Qwen2.5-1.5B  >  Gemma-3-1b
~9GB 4-bit            ~5GB (pre-trained CoT)      ~5GB          ~5GB              ~2GB              ~1GB
★★★★                       ★★★★                    ★★★           ★★★               ★★                ★
```

> **This hierarchy predates the bug (a)/(b)/(c)/(d) investigation below and is aspirational,
> not measured post-fix.** See "Current status" immediately below for which scripts are
> actually confirmed to train as of 2026-07-24.

#### Current status (2026-07-24 evening) — which scripts are confirmed to actually train

**grpo_01 (gemma-3-1b) and grpo_07 (Phi-4-mini) are confirmed working.** Every other
script sits in one of three buckets:

| Script | Status | Why |
|---|---|---|
| **grpo_07** (Phi-4-mini) | ✅ confirmed working | Full epoch (935/935 steps) + GSM8K eval showing a real improvement over the base model (format 52%→85%, accuracy 86.3%→87.7%) |
| **grpo_07_1** (Phi-4-mini, no-vLLM, fa2) | ✅ confirmed working (2026-07-26) | Full epoch (935/935 steps, ~25h), healthy KL/grad_norm throughout, in-training merge clean (no live-vLLM alias possible). GSM8K eval: format 80.97% (1068/1319), accuracy 88.25% (1164/1319) — matches/slightly beats grpo_07's vLLM-trained result (87.72% acc), confirming the bug (e) `UNSLOTH_DISABLE_FAST_GENERATION` bypass yields a fully functional training path, not just smoke-clean |
| **grpo_01** (gemma-3-1b) | ✅ confirmed working (smoke) | Re-checked 2026-07-24: the script's `use_gradient_checkpointing=True` fix had been left **commented out** (bug (b) was never actually applied here — unsloth's default is `"unsloth"` GC, the broken mode), which is why the only pre-existing log (2026-07-20) showed `grad_norm: nan/inf` every step. Fixed and re-run (`GRPO_MAX_STEPS=5`): `non-finite(B grads)=0/182` on **all 5 steps**, `\|B\|max` grows monotonically (0→2.2e-5→7.8e-5→1.1e-4, nowhere near the ×256 bug-(c) explosion), `grad_norm` finite every step, KL stays 0.0008–0.0015. Only a 5-step smoke, not a full epoch — but the signature matches grpo_07's validated pattern exactly |
| **grpo_08** (Qwen3-4B) | ❌ confirmed blocked | Smoke 2026-07-24 (post-fix): `\|B\|max=0` on all 5 steps, `non-finite(B grads)=252/252` every step, KL up to 1.36e6 — bug (a) blocks training, not just cosmetic |
| **grpo_06** (DAPO Llama-8B) | ❌ confirmed blocked | `non-finite=448/448` from step 1 in both compiled and eager mode — DAPO loss math itself (beta=0), independent of bug (a)/(b) |
| grpo_02/03/04/05/09 | ❓ not re-tested post-fix | Pre-fix KL audit flagged 02/04/05 "sick" (bug (a), Qwen/DeepSeek-R1 family) and 03 inconclusive (resumes from checkpoint); none have been rerun with the GC fix or a `\|B\|max` tripwire — and given the grpo_01 lesson, **check each script's actual `get_peft_model` call, not just memory/README claims, before trusting any "fix applied" note** |

GSM8K test-set eval (1319 examples, `grpo_07_phi4_14b_gsm8k_test.py`, greedy decoding), base model in both cases `microsoft/Phi-4-mini-instruct`:

| Model | Format compliance | Accuracy |
|---|---|---|
| base `Phi-4-mini-instruct` | 52.24% (689/1319) | 86.28% (1138/1319) |
| grpo_07 (vLLM rollouts) | 85.29% (1125/1319) | 87.72% (1157/1319) |
| grpo_07_1 (fa2, no-vLLM) | 80.97% (1068/1319) | 88.25% (1164/1319) |

Bottom line: don't trust any script's old "healthy" KL verdict as proof it trains, and
don't trust a changelog saying a fix was "applied to all scripts" without grepping the
actual file — grpo_01's fix silently regressed to a comment. The only trustworthy signal
is a **fresh** smoke run with the `\|B\|max`/non-finite tripwire (see grpo_01/07/08 for
the pattern) confirming finite grads and a moving, non-exploding adapter.

#### Training Health Audit (2026-07-23) — KL/grad_norm pathology across all GRPO runs

Triggered by the recurring CUDA device-side assert in `grpo_04` (DeepSeek-R1-8B). The Triton
A/B test (3.6.0 → 3.7.1, `dev-python/triton-bin` in the pwr overlay) changed the symptom, not
the cause: under 3.6.0 the crash was deterministic at step 84; under 3.7.1 the run survived to
step 675 and the assert became readable: `index out of bounds: 0 <= tmp0 < 128256` — 128256 is
exactly the Llama-3/DeepSeek vocab size, so a compiled kernel gathers/scatters with a token id
outside the vocabulary. The `fast_lora.py → fast_dequantize` frame in the traceback is async
reporting noise; the real culprit launched earlier (most likely the compiled
`chunked_selective_log_softmax` logp gather).

The crash turned out to be a secondary symptom. Sweeping every `grpo*.log`:

| Script | Model | First KL values (step 1→) | Verdict |
|---|---|---|---|
| grpo_01 | gemma-3-1b | 0 / 0.001 / 0 | healthy |
| grpo_06 | Llama-3.1-8B (DAPO) | 0 / 0 / 0 | healthy (caveat: beta=0 may make KL trivially 0) |
| grpo_07 | Phi-4-mini | 0.0008 / 0.0008 / 0.001 | healthy |
| grpo_02 | Qwen2.5-1.5B | 5807 → **1.4e6** | sick (fresh run — no adapter load in log) |
| grpo_04 | DeepSeek-R1-8B | 45 → 4261 | sick |
| grpo_05 | Qwen3-4B | 2.5e5 | sick |
| grpo_08 | Qwen3-4B | 1.6e4 | sick |
| grpo_03 | Llama-3.1-8B | 72 → 1392 | inconclusive — resumes from checkpoint-2400, KL>0 partly legitimate |

Key facts:

1. **With a fresh (zero) LoRA, KL at step 1 must be ~0.** Values in the tens to millions mean
   the trainer-side per-token logps are garbage from the very first step — the sick runs never
   trained. The 675-step grpo_04 run confirms it: correctness reward flat at 1.47–1.56 in every
   50-step window (zero learning, pure GPU burn).
2. **`grad_norm` is broken in every script, healthy or sick.** 100% `nan` in all logs except
   grpo_01, whose "best" log has 27× nan, 12× inf and two absurd finite values (8.8e8, 3.6e5).
   This is a separate, stack-wide bug (unsloth/TRL/bnb grad-norm path), independent of the KL
   split.
3. **Model split:** all Qwen models (Qwen2.5-1.5B, Qwen3-4B) + DeepSeek-R1-distill are sick;
   Gemma-3-1b, fresh Llama-3.1-8B and Phi-4-mini are healthy. The earlier "`<think>`
   chat-template" hypothesis is weakened: Qwen2.5-1.5B uses plain ChatML (no think block) and
   is the sickest of all. The discriminating factor (chat template handling, pad/eos config in
   the unsloth repos, or a per-architecture unsloth code path) is not yet identified.
4. Eliminated suspects: Triton version (symptom shifter only), byte-level tokenizer corruption
   (`grpo-fix-hf-tokenizer scan` reports all cached tokenizers OK, and the in-script guard in
   grpo_04 stayed silent), NaN weights / sampler ids / vLLM input ids (all five WATCH probes
   from the earlier investigation were clean).

**Update 2026-07-23 (late evening): `grad_norm=nan` is NOT a display artifact — no GRPO run
has ever trained.** Proof from grpo_06 `checkpoint-50` (fresh run, 50 optimizer steps):

- all 224 `lora_B` matrices are **exactly zero** (LoRA-B initializes to zero; any applied
  update would move them), adapter tensors all finite;
- the bnb `paged_adamw_8bit` state shows `step=50` for every param but `absmax1=absmax2=0.0`
  — both Adam moments are exactly zero, so the optimizer ran 50 times and saw **exactly-zero
  gradients** every step (real NaN grads would have poisoned moments and weights);
- meanwhile the logged loss was huge (49.7 with beta=0) and `clip_ratio/region_mean≈0.43`,
  so the trainer/vLLM logp mismatch is present in grpo_06 too — its `kl=0` is trivial
  (beta=0), not evidence of health. Genuinely healthy logp measurements remain gemma
  (grpo_01) and Phi-4-mini (grpo_07) only.

So there are (at least) two distinct bugs: (a) trainer-side logp mismatch (the KL split
above), and (b) a backward/step-path bug shared by ALL scripts — displayed loss is finite,
`grad_norm` logs nan, yet the gradients reaching the optimizer are zero → every GRPO run to
date was a no-op (also explains the flat 675-step reward curve and why weights never went
NaN). Prime suspect for (b): the shared `unsloth_compiled_cache/UnslothGRPOTrainer.py`
chunked-logp path (`unsloth_grpo_mini_batch` / `unsloth_logit_chunk_multiplier`, part of the
local patch set) — a misplaced `detach()`/`no_grad` would produce exactly this signature.

**RESOLVED 2026-07-24 — bug (b) root cause: unsloth-zoo offloading gradient checkpointing.**
Probe chain on grpo_07 (Phi-4-mini, GRAD-WATCH instrumentation = per-tensor backward hooks +
global optimizer step pre-hook, both trainer-agnostic):

| Probe | Config | Result |
|---|---|---|
| GRAD-WATCH on normal run | compiled + GC `"unsloth"` | backward reaches LoRA params, but ~85% of micro-batches produce **non-finite** grads → accumulated grad NaN → optimizer sees 128/128 NaN every step |
| #2 anomaly | `TORCHDYNAMO_DISABLE=1` + GC `"unsloth"` | `AddmmBackward0 returned nan` — NaN persists in eager → compiled Triton kernels exonerated; forward trace hidden inside the unsloth-zoo checkpointed segment |
| #3 anomaly | eager + GC off (`GRPO_NO_GC`) | **clean**: grad_norm=0.071, non-finite=0 |
| #4 | compiled + GC `True` (HF per-layer) | **clean**: grad_norm=0.063, non-finite=0 — first real grad_norm ever logged |

Mechanism: `use_gradient_checkpointing="unsloth"` (unsloth-zoo's CPU-offload checkpointing,
`unsloth_zoo/gradient_checkpointing.py`) corrupts recomputed activations during backward →
NaN gradients in most micro-batches; one poisoned micro-batch NaNs the whole gradient
accumulation; bnb `paged_adamw_8bit` then effectively skips the update (moments stay zero) →
`lora_B` never leaves zero. The grpo_04 device-side assert (OOB gather in backward,
`fast_dequantize` frame) is most likely the same corruption crossing into an
assert-compiled kernel.

**Fix applied 2026-07-24:** all `grpo_0*.py` (01–09 + `_cont`) switched to
`use_gradient_checkpointing=True` (HF per-layer). grpo_07 keeps a `GRPO_GC=unsloth|hf|off`
env gate (default `hf`) plus `GRPO_ANOMALY=1` / `GRPO_MAX_STEPS=N` for future probes.
Costs: HF GC keeps layer-boundary activations on GPU (no CPU offload) — slightly higher
VRAM than unsloth GC; watch the first full runs on the 8B models. Still open: bug (a)
logp mismatch (Qwen*/DeepSeek-R1), and grpo_07's LoRA only covering `o_proj`+`down_proj`
(Phi-4-mini's fused `qkv_proj`/`gate_up_proj` don't match the target regex — add them
explicitly after retraining starts working).

**Bug (c), found 2026-07-24 right after the GC fix — progressive vLLM-engine corruption
once real updates flow.** First genuine training ever (grpo_07 steps 1–4: grad_norm
0.063→0.049, KL 0.002–0.007, all finite), then generations rot progressively: batches
34–36 coherent → 37 malformed → 38–39 gibberish → 41+ single-token loops
(`AddAddAdd…`) — with the adapter CONSTANT between optimizer steps, and update
magnitudes (warmup lr ~1e-7, clip 0.1) mathematically incapable of altering
generations. loss 2.8e24 / KL 2.8e27 / grad_norm inf at step 5 are downstream of
scoring garbage text (clip 0.1/inf then zeroes all grads). Ruled out: all 7 pwr
overlay patches live (audit clean); vLLM standby/sleep default OFF. Leading
hypothesis: corruption confined to the vLLM engine side (own weight copy + LoRA
hot-load slots), triggered by the first-ever nonzero `lora_B` sync — or stray OOB
writes into the shared VRAM pool (same defect family as the grpo_04 backward assert).

Probe status at session close (2026-07-24 night):
- No-vLLM control (`GRPO_NO_VLLM=1`, rollouts via HF generate) is BLOCKED by two
  independent issues: (i) `attn_implementation="flash_attention_2"` requires the
  `flash_attn` package, absent on this host → script now falls back to `sdpa` for the
  no-vLLM path (TODO: package a `flash-attn` ebuild in the pwr overlay — several HF
  paths won't work without it); (ii) after that, unsloth's `unsloth_base_fast_generate`
  → transformers `_sample` crashes with `multinomial: prob_dist must be 1 or 2 dim`
  (separate unsloth/transformers drift bug, unfixed).
- Integrity probe VERDICT (`grpo07_integrity1.log`, 10 steps): `base-drift: none`
  (frozen weights untouched — vLLM engine and OOB-writer theories eliminated),
  gradients clean, but **`|B|max` grows exactly ×256 per step**
  (8.5e-4 → 0.219 → 56 → 1.4e4 → … → 6.2e13). 256 = 2⁸ = the LoRA scale
  s = alpha/r = 2 (grpo_07 uses alpha=2·rank) applied **in-place 8×/step** (once per
  generation round). Culprit: `unsloth_zoo/vllm_utils.py` `load_lora_directly()`
  (~line 2846): `vllm_lora_B.copy_(model_lora_B); vllm_lora_B *= s` — when the vLLM
  buffer aliases the training tensor (zero-copy colocation), `copy_` is a self-copy
  no-op and `*= s` compounds on the TRAINING weights. Invisible historically because
  (i) with the GC bug B was always zero, and (ii) with alpha=rank (grpo_01/02/03/06
  and most unsloth examples) s=1. Only surfaces when real training ∧ alpha≠rank.
  Next: prove aliasing via `data_ptr()` comparison; immediate workaround = set
  `lora_alpha=lora_rank` in grpo_07/08 (do NOT rerun grpo_07 without this — it
  explodes again by step ~4); proper fix = pwr overlay patch scaling into a temp
  (upstream-PR candidate).

Next steps:

- Do **not** rerun sick scripts as-is — hours of GPU with zero learning, and the assert will
  return.
- Decisive probe (cheap, no training): dump `prompt_ids`/`completion_ids` right before the logp
  gather for one batch and compare against what vLLM generated (lengths, `max(id)`, `min(id)`,
  whether the trainer prompt includes the template-appended `<think>`). Run it pairwise: one
  sick model (Qwen2.5-1.5B — smallest, most extreme) vs one healthy (gemma-1b or Phi-4-mini),
  and diff structurally.
- Resolve `grad_norm=nan` separately on a healthy script (grpo_07): logging artifact vs real
  non-finite gradients.
- Permanent guard for all scripts: assert KL < 1.0 at step 1 with a fresh LoRA, so a broken run
  dies after a minute instead of a day.

---

### 6. Other

| File | Description |
|------|-------------|
| `intutor_pw_02.py` | English tutor CEFR A1-C1, OpenAI API, topics by level |

---

### 7. Inference Runtime Benchmarks (`bench_NN_*.sh`)

Goal: compare local inference runtimes (llama.cpp vs ik_llama.cpp vs ollama)
on the same GGUF with MoE CPU/GPU offload tuned for 16 GB VRAM.
**Parameters, usage and reference results: [BENCH.md](BENCH.md).**

| File | Description |
|------|-------------|
| `bench_01_llamacpp.sh` | mainline llama.cpp: `llama-bench` sweep over `--n-cpu-moe` + timed `llama-server` request |
| `bench_02_ikllama.sh` | ik_llama.cpp fork (`ik-llama-*` binaries): same two tests |
| `bench_03_ollama.sh` | ollama add-on: cold/warm `/api/generate`, auto CPU/GPU split shown via `ollama ps` |
| `bench_04_qwythos.sh` | Qwythos-9B-v2 (dense hybrid): context-depth sweep + server request + MTP on/off comparison |
| `bench_05_agentic.sh` | agentic coding capability: 12 tasks via headless qwen-code against any aillama profile, objective verdicts + SCORE |
| `bench_06_dense_generic.sh` | bench_04 generalized to any dense model (`MODEL=` required); per-depth loop survives OOM at deeper values |
| `bench_07_workflow.sh` | agentic workflow discipline: long-rules packaging task via headless qwen-code, 10-item rubric on 5 failure axes (tail-read, compliance, hallucination, thrashing, evidence-gate), `RUNS`× repetitions + per-item compliance matrix |

> 2026-07-05 (GLM-4.7-Flash 30B-A3B Q4): ik_llama.cpp pp 2106 tok/s (2× mainline),
> mainline tg 92 tok/s best, ollama far behind (whole-layer offload). Details in BENCH.md.

---

## Tests

### GSM8K — GRPO Model Evaluation

All `*_test.py` files load a saved model (merged 16-bit) and evaluate it on the full GSM8K test set (~1319 examples).

| Test file | Tests model (directory) | Response format |
|-----------|------------------------|-----------------|
| `grpo_02_qwen15b_gsm8k_test.py` | `outputs/lora-grpo-qwen3` | `<reasoning>` + `<answer>` |
| `grpo_03_llama8b_gsm8k_test.py` | `outputs/lora-grpo-lama4` | `<reasoning>` + `<answer>` |
| `grpo_04_deepseek_r1_8b_gsm8k_test.py` | `outputs/lora-grpo-deepseek-r1-llama8b` | `<reasoning>` + `<answer>` |
| `grpo_05_qwen3_8b_gsm8k_test.py` | `outputs/lora-grpo-qwen3-8b` | `<think>` + `<answer>` |
| `grpo_07_phi4_14b_gsm8k_test.py` | `outputs/lora-grpo-phi4` | `<reasoning>` + `<answer>` |
| `grpo_07_phi4_14b_gsm8k_test.py` (reused) | `outputs/lora-grpo-phi4-mini-novllm` | `<reasoning>` + `<answer>` (grpo_07_1 no-vLLM/fa2 run) |
| `grpo_08_qwen3_14b_gsm8k_test.py` | `outputs/lora-grpo-qwen3-14b` | `<think>` + `<answer>` |
| `gsm8k_openai_1_test_500.py` | GPT-4o | baseline (API) |
| `gsm8k_grokai_1_test_500.py` | Grok (xAI) | baseline (API) |

```bash
python grpo_03_llama8b_gsm8k_test.py
python grpo_04_deepseek_r1_8b_gsm8k_test.py
python grpo_05_qwen3_8b_gsm8k_test.py
```

Qwen3 models (`_qwen3_8b_`, `_qwen3_14b_`) use `temperature=0.6` (`/think` mode);
others use `temperature=0.1` (deterministic).

### IMDB — Classification Model Evaluation

| Test file | Tests model (directory) |
|-----------|------------------------|
| `lc_03_distilbert_imdb_test.py` | `outputs/lora-distilbert` |
| `lc_04_electra_imdb_test.py` | `outputs/lora-electra-imdb` |
| `lc_05_roberta_imdb_test.py` | `outputs/lora-bert-roberta2` |
| `lc_06_modernbert_imdb_test.py` | `outputs/lora-modernbert-imdb` |
| `lc_07_gemma2b_imdb_test.py` | `outputs/lora-gemma2-imdb` |
| `lc_08_gemma9b_imdb_test.py` | `outputs/lora-gemma2-5-imdb` |
| `lc_09_gemma12b_imdb_test.py` | `outputs/lora-gemma3-imdb` |

```bash
python lc_05_roberta_imdb_test.py     # best encoder
python lc_06_modernbert_imdb_test.py  # new SOTA (Dec 2024)
```

IMDB files print a full `classification_report` (precision/recall/F1 per class) and final accuracy.

---

## Output Directory Structure

All model checkpoints, saved models, and logs are written to `outputs/` (excluded from git via `.gitignore`):

```
outputs/
  lora-grpo-phi4/                  # merged 16-bit model (grpo_07)
  lora-grpo-phi4-outputs/          # training checkpoints
  lora-grpo-phi4-mini-novllm/          # merged 16-bit model (grpo_07_1, no-vLLM/fa2)
  lora-grpo-phi4-mini-novllm-outputs/  # training checkpoints
  lora-grpo-qwen3-14b/
  lora-grpo-qwen3-14b-outputs/
  ...
  lora-bert-roberta2/              # lc_ fine-tuned models
  lora-modernbert-imdb/
  ...
  best_hybrid_double_light_res.pt  # simple_lstm_cnn best weights
  best_hybrid_double_light_res4.pt
  best_sentiment_model.pt          # simple_imdb best weights
  logs/                            # HuggingFace Trainer logs
  optuna-distilbert-*/             # Optuna trial checkpoints
```

---

## Installation — Blackwell (CUDA 12.x)

### Blackwell compatibility
- https://github.com/unslothai/unsloth/issues/1679#issuecomment-2776622643

### TRITON
```
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -r python/requirements.txt # build-time dependencies
cd python
MAX_JOBS=2 python setup.py bdist_wheel
pip install dist/xxx
```

### PYTORCH
```
git clone https://github.com/pytorch/pytorch
cd pytorch
export CFLAGS+=" -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"
export CXXFLAGS+=" -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"
git submodule sync
git submodule update --init --recursive -j 8
pip install -r requirements.txt
pip install mkl-static mkl-include wheel
# Build PyTorch (will take a long time)
export CUDA_HOME=/opt/cuda
export CUDA_PATH=$CUDA_HOME
export TORCH_CUDA_ARCH_LIST=Blackwell
MAX_JOBS=2 python setup.py bdist_wheel
pip install dist/xxx
```

### VLLM
```
git clone https://github.com/vllm-project/vllm.git
cd vllm
export CUDA_HOME=/opt/cuda
export CUDA_PATH=$CUDA_HOME
#export TORCH_CUDA_ARCH_LIST=Blackwell
export TORCH_CUDA_ARCH_LIST='12.0'
# trying to set other than only cuda libs
export USE_CUDNN=1
export USE_CUSPARSELT=1
export USE_CUFILE=1
export USE_CUDSS=0
export CMAKE_ARGS="-DUSE_CUDNN=1 -DUSE_CUSPARSELT=1 -DUSE_CUDSS=0 -DUSE_CUFILE=1"
# Build vllm (will take a long time)
export CUDA_HOME=/opt/cuda
python use_existing_torch.py
pip install -r requirements/build.txt
pip install setuptools_scm
MAX_JOBS=1 python setup.py bdist_wheel
pip install dist/xxx
```

### UNSLOTH, XFORMERS, FA2
```
pip install ninja bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
export TORCH_CUDA_ARCH_LIST='12.0'
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
MAX_JOBS=4 pip install flash-attn --upgrade --no-build-isolation
```

### Other not working recipes for only binary installation
```
python -m pip uninstall torch torchvision
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Links to issues
- https://github.com/unslothai/unsloth/issues/1679
- https://github.com/vllm-project/vllm/issues/14452
- https://github.com/pytorch/pytorch/issues/145949
- https://github.com/comfyanonymous/ComfyUI/issues/7127

### LED / RGB (Blackwell)
- https://gitlab.com/CalcProgrammer1/OpenRGB/-/issues/4710

### NVIDIA suspend
- https://forums.developer.nvidia.com/t/rtx-5070-ti-with-570-124-04-won-t-resume-monitor-from-suspend-to-ram/327297
