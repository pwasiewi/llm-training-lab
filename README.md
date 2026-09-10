# LLM Experiments

Experiments with training and fine-tuning LLM models. GPU: NVIDIA Blackwell (RTX 5070 Ti), CUDA 13.x.

Naming scheme: `simple_NN_*` = teaching lesson series (from-scratch fundamentals) · `grpo_NN_*` = GRPO/RL experiments (increasing model size) · `lc_NN_*` = fine-tuning classifiers · `bench_NN_*` = inference runtime benchmarks (bash)

---

## File Overview

---

### 1. From-Scratch Fundamentals — Lesson Series (`simple_NN_*`)

Goal: learn how LLMs work from first principles — seven standalone teaching
scripts with the theory in the comments, GPU-verified 2026-07-30.
**Full learning path, per-lesson concepts and the old→new file mapping:
[simple_README.md](simple_README.md).**

| File | Lesson |
|------|--------|
| `simple_01_gpt_tinystories.py` | pretrain a tiny GPT (BPE, causal mask, perplexity, sampling) |
| `simple_02_imdb_encoder.py` | encoder classifier, `--arch classic\|modern` (RoPE, GeGLU, sliding-window attention) |
| `simple_03_imdb_library.py` | same encoder via `nn.TransformerEncoder` + bf16 autocast |
| `simple_04_vram_math.py` | training VRAM estimation by hand (no GPU needed) |
| `simple_05_augmentation.py` | label-preserving text augmentation (WordNet + EDA) |
| `simple_06_hybrid_pretransformer.py` | transformer+BiLSTM+CNN hybrid, `--lstm-depth 1\|3`, data-leakage lesson |
| `simple_07_scaling_ablation.py` | measured cost-vs-seq-length scaling exponents |

> Replaces the old `simple_llm03.py`, `simple_imdb*.py` and `simple_lstm_cnn*.py`
> scripts (deleted 2026-07-30) — the mapping table is in simple_README.md.

---

### 2. Fine-tuning Pre-trained Models on IMDB with LoRA (`lc_NN_*`)

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
| `lc_03_distilbert_imdb_optuna.py` | distilbert + Optuna HPO | 66M | auto | auto | superseded by `lc_10` |
| `lc_04_electra_imdb.py` | electra-large-discriminator | 335M | 30 | 3e-4 | **diverges — use `lc_04_2`** |
| `lc_04_1_electra_imdb_bf16.py` | same, bf16 instead of fp16 | 335M | 30 | 3e-4 | negative control, still diverges |
| `lc_04_2_electra_imdb_lr.py` ★ | same, lr 1e-4 | 335M | 30 | 1e-4 | the fix |
| `lc_05_roberta_imdb.py` ★ | roberta-large | 355M | 30 | 2e-4 |
| `lc_05_1_roberta_imdb_targets.py` | same, LoRA on 6 modules not 2 | 355M | 30 | 2e-4 | negative control, no change |
| `lc_06_modernbert_imdb.py` ★ | ModernBERT-large | 395M | 10 | 2e-4 |
| `lc_11_neobert_imdb.py` | NeoBERT | 250M | 10 | 2e-4 | needs `sci-ml/xformers` |
| `lc_10_hpo_lora.py` | Optuna harness for all of the above | — | — | searched |

#### Generative Models (Gemma)

| File | Model | Params | Epochs | lr |
|------|-------|--------|--------|----|
| `lc_07_gemma2b_imdb.py` | gemma-2-2b-it | 2B | 10 | 1e-4 |
| `lc_08_gemma9b_imdb.py` | gemma-2-9b-it | 9B | 3 | 1e-4 |
| `lc_09_gemma12b_imdb.py` ★ | gemma-3-12b-pt | 12B | 3 | 1e-4 |

`lc_07_1`/`lc_08_1`/`lc_09_1_*_liger.py` — identical hyperparameters, but the Gemma
module classes are patched with [Liger](https://github.com/linkedin/Liger-Kernel)
fused Triton kernels (RMSNorm, RoPE, GeGLU) before loading, for a step-time and
peak-VRAM comparison against the plain runs (FusedLinearCrossEntropy is inert for
SEQ_CLS heads). Outputs go to separate `*-liger` dirs; peak VRAM is printed after
training. Results: see "Liger kernel comparison" below — 9–15% faster epochs,
accuracy unchanged.

> **Ranking (2026-07-27, full 25k test set; electra and NeoBERT measured 2026-09-10):** `ModernBERT-large` (96.2%) ≳ `NeoBERT` (95.9%) > `gemma3-12b` ≈ `gemma2-9b` (94.6%) > `gemma2-2b` (93.5%) > `electra-large` (93.4%, at lr 1e-4 — see below) > `roberta-large` (92.7%) > `distilbert` (87.1%)
>
> ModernBERT (Dec 2024): rotary embeddings, Flash Attention 2, 8192-token context, ~24% faster than RoBERTa.
>
> The top two are 0.24 pp apart on the saved adapters (96.16 vs 95.92), against a
> single-proportion SE of ≈0.12 pp at n=25000 — so roughly 1–2 SE, from one run
> each, with no seed replication and no McNemar test on the paired predictions.
> Read it as "indistinguishable on this benchmark", not as a ranking. What is
> not ambiguous: NeoBERT reaches it with 145M fewer parameters and a 24%
> shorter epoch. IMDB is saturated (see the 9B→12B note below), so it cannot
> separate these two — a harder task would be needed to.

#### Newer encoder-only releases (HF sweep 2026-07-28; NeoBERT since tested)

The encoder-only class is still alive, just lower-volume and more
specialized. Spotted in a HF sweep, potential additions to the comparison
above (and candidates for the Gentoo-log DAPT/triage-encoder idea):

| Model | Released | Params | Notes |
|-------|----------|--------|-------|
| `chandar-lab/NeoBERT` | 2025-02 | 250M | **TESTED 2026-09-10 → `lc_11`, 95.92%.** Deep-and-narrow (28 × 768), SwiGLU, RMSNorm, RoPE, 4096 ctx; claims to beat ModernBERT-large on GLUE/MTEB with 145M fewer params. On IMDB it lands 0.24 pp below it in 42 min vs 55 — indistinguishable at this saturation level. Requires `sci-ml/xformers`; see the NeoBERT section below for four traps, one of them silent |
| `LiquidAI/LFM2.5-Encoder-350M` | 2026-07-27 | 350M | Bidirectional MLM on the hybrid LFM2 arch (conv+attention); 15 languages **incl. Polish**; positioned as a fine-tune base for classification/NER/retrieval/rerank, on-device focus; claims "ahead of every model its size or smaller". License `lfm1.0` (custom, "other") — review before commercial use. Same size shelf as ModernBERT-large (395M) |
| `LiquidAI/LFM2.5-Encoder-230M` | 2026-07-27 | 230M | Lightweight sibling for latency/memory budgets |
| `jhu-clsp/mmBERT-base` | 2025 | 140M+ | Multilingual ModernBERT descendant (JHU), ~550k downloads; the Ettin suite is from the same team |
| `nvidia/Nemotron-3-Embed-1B` / `-8B` | 2026-07-14 | 1B/8B | Embedding class (not MLM); BF16 + NVFP4 variants |
| `google/embeddinggemma-300m` | 2025 | 300M | Embedding class, on-device |

Trend note: most new "encoders" are embedding models (often distilled from
decoders, e.g. Qwen3-Embedding); pure MLM encoders like LFM2.5-Encoder and
the ModernBERT family remain the go-to for cheap task-specific classifiers.
For the log-triage DAPT idea, `LFM2.5-Encoder-350M` is the main new
counter-candidate to ModernBERT-large (fresher pretrain, multilingual —
mixed Polish/English log lines).

#### VRAM tuning (2026-07-27, RTX 5070 Ti 16 GB, transformers 5.12 / torch 2.14)

Batch sizes probed empirically (4 train steps + full eval step per script); effective
batch (train bs × accumulation) kept identical to the original configs, so training
dynamics and lr stay comparable — only GPU utilization changes. Throughput measured on
the same 4-step smoke (includes warmup, so real epochs run slightly faster).

| Script | train/eval bs | accum | effective bs | grad ckpt | peak VRAM | train samples/s |
|---|---|---|---|---|---|---|
| `lc_03` distilbert | 128 / 256 | 1 | 128 | off | 5.3 GiB | ~342 |
| `lc_04` electra-large | 32 / 128 | 1 | 32 | off | 6.0 GiB | ~167 |
| `lc_05` roberta-large | 32 / 128 | 1 | 32 | off | 5.8 GiB | ~129 |
| `lc_06` modernbert (seq 512) | 16 / 64 | 2 | 32 | off | 12.6 GiB | ~33 |
| `lc_11` neobert (seq 512) | 16 / 64 | 2 | 32 | off | 10.5 GiB | ~61 † |
| `lc_07` gemma-2-2b nf4 | 16 / 32 | 2 | 32 | off | 13.5 GiB | ~33 |
| `lc_08` gemma-2-9b nf4 | 32 / 64 | 4 | 128 | on | 12.5 GiB | ~8.1 |
| `lc_09` gemma-3-12b nf4 | 8 / 16 | 2 | 16 | on | 11.9 GiB | ~6.0 |

Notes:
- † `lc_11`'s figure is derived from the real training run, not from the 4-step
  smoke used for every other row, so it is the *higher-quality* number and not
  directly comparable to them (the smoke includes warmup and understates).
  Derivation: 42.1 min wall − 4 evals × 217.8 s = 27.5 min of training for
  4 × 25000 samples. **Do not read `train_samples_per_second` out of the log for
  this**: HF reports 99.0, because it divides by the *configured* `num_train_epochs`
  (10) rather than the 4 epochs early stopping actually ran — an over-report by
  exactly the ratio of configured to run epochs, which bites any early-stopped run.
- Batch scales with **activation memory**, not model size: small models with grad
  checkpointing OFF store the full backward graph (lc_07: 2B weights ≈ 2.5 GB but
  ~10 GB activations at bs=16), while the 9B/12B keep checkpointing ON and afford a
  wider batch (only layer inputs stored, rest recomputed).
- OOM ceilings found: lc_06 at bs=32/seq 512 (>15 GiB), lc_07 at bs=32 with ckpt off,
  lc_09 at train bs=16 (evals OOM at ~13.4 GiB train peak). lc_09 additionally sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` against fragmentation near the limit.
- Eval batch can always be wider than train batch: forward-only passes free activations
  layer by layer.
- Critical fix (all gemma scripts): `use_cache=False` (both config levels on gemma-3) +
  `keys_to_ignore_at_inference=["past_key_values"]` — otherwise Trainer eval crashes with
  `TypeError: Unsupported types (DynamicCache)` at the first epoch boundary under
  transformers 5.

#### Final results (2026-07-27 overnight run, tuned configs above)

Training times deduced from checkpoint timestamps (runs were sequential; each epoch
includes a full 25k-example eval). Test accuracy = the saved adapter re-loaded by
`lc_NN_*_test.py` and evaluated on the full 25k test set — matching the Trainer's
best-epoch eval confirms the saved artifact is intact. Untrained baseline (pretrained
backbone + freshly initialized head) is chance level ≈ 50% for every model.

| Script | Model | Epochs run | Time/epoch | Total wall | Best eval acc | Test acc (saved) |
|---|---|---|---|---|---|---|
| `lc_03` | distilbert-base | 8 (best ep3) | ~40 s | ~7 min | — (eval_loss 0.398) | 87.14% |
| `lc_04` | electra-large | 6 | 129 s | 13 min | 50.0% | **50.00% — failed** |
| `lc_04_2` | electra-large @ lr 1e-4 | 4 (best ep3) | 134 s | 9 min | 93.32% | 93.36% |
| `lc_05` | roberta-large | 4 (best ep2) | 120 s | 8 min | 92.84% | 92.72% |
| `lc_05_1` | roberta-large, 6 LoRA modules | 4 (best ep2) | 141 s | 14 min | 92.80% | 92.77% |
| `lc_06` | ModernBERT-large | 4 (best ep2) | 13.8 min | 55 min | 96.24% | 96.16% |
| `lc_11` | NeoBERT (2026-09-10) | 4 (best ep2) | 10.5 min | 42 min | 95.94% | 95.92% |
| `lc_07` | gemma-2-2b nf4 | 4 (best ep2) | 13.8 min | 55 min | 93.50% | 93.50% |
| `lc_08` | gemma-2-9b nf4 | 3 (best ep1) | 65.5 min | 3 h 17 min | 94.56% | 94.55% |
| `lc_09` | gemma-3-12b nf4 | 3 (best ep2) | 82 min | 4 h 08 min | 94.60% | 94.60% |

Notes:
- **lc_04 (electra-large) diverged**: eval_loss pinned at ln 2 = 0.693 and accuracy at
  exactly 0.5 for all 6 epochs — the classic electra-large instability at lr 3e-4.
  Retry with lr ≈ 5e-5–1e-4 (and optionally warmup); not a loader or eval artifact.
- **RESOLVED 2026-09-10, and the diagnosis above was correct.** `lc_04_2` is `lc_04`
  with one number changed, lr 3e-4 → 1e-4: **93.36%** (Trainer best 93.32% at epoch 3,
  early-stopped at 4, adapter re-loaded and re-scored on the full 25k to confirm).
  electra-large therefore belongs above roberta-large in the ranking, not at the bottom.

  Getting there took two wrong turns worth recording, because both are traps that
  will recur:

  | run | lr | train | precision | bias | prepare_* | result |
  |---|---|---|---|---|---|---|
  | `lc_04` | 3e-4 | 25k | fp16 | lora_only | yes | 0.5000 |
  | `lc_04_1` | 3e-4 | 25k | **bf16** | lora_only | yes | 0.5000 |
  | `lc_10` trial 0 | 3e-4 | **5k** | bf16 | none | no | **0.9250** |
  | `lc_10` trial 0 | 3e-4 | 25k | bf16 | none | no | 0.5000 |

  Rows 1–2 kill the precision hypothesis: fp16's GradScaler is not the cause, and
  `lc_04_1` is kept as a labelled negative control rather than deleted. Rows 3–4 are
  the same code with only the training-set size moved, which isolates the real
  variable — the **number of optimizer steps near lr 3e-4** (782/epoch at 25k vs
  157/epoch at 5k). Weight dtype, `bias="lora_only"` and the (spurious)
  `prepare_model_for_kbit_training` call are all held constant across that pair and
  are all innocent.

  **The trap: an HPO study run on a subset never enters the regime containing the
  bug it is meant to explain.** All 30 trials of the electra study below ran at 5k,
  where this failure cannot happen, so trial 0's healthy 0.9250 actively misled the
  diagnosis. Anything searched on a subset must be re-checked at full scale before
  its conclusions are trusted.
- **Test-file loader bug fixed (2026-07-27)**: the old `lc_*_test.py` loaded the adapter
  directory directly via `AutoModelForSequenceClassification.from_pretrained(adapter_dir)`,
  which silently instantiates a **randomly initialized classification head** (measured:
  0.46–0.63 accuracy on models the Trainer scored at 92–93%). All test files now load the
  base model explicitly and apply the adapter via `PeftModel.from_pretrained`, with batched
  inference on the full 25k test set (previously 500 examples one-by-one).
- Compute-for-accuracy is brutal at the top: ModernBERT-large gets the best accuracy at
  55 min total, while gemma-3-12b burns 4 h for −1.6 pp. The 9B→12B step gains nothing
  (94.55% vs 94.60%); IMDB@128 tokens saturates around ~94.6% for decoder LoRA.

#### NeoBERT (2026-09-10, `lc_11_neobert_imdb.py`) — and a silent corruption bug

[NeoBERT](https://huggingface.co/chandar-lab/NeoBERT) (Chandar Lab, Feb 2025) is
the one entry in the "newer encoders" table below with a real claim on lc_06's
spot: 250M params, reported to beat ModernBERT-large on GLUE/MTEB. Deep and
narrow (28 × 768) where ModernBERT is wide, SwiGLU instead of GeGLU, RMSNorm
pre-norm, RoPE, 4096-token context. Result: **95.92%** vs ModernBERT's 96.16%,
in 42 min instead of 55 — see the ranking note above for why that gap does not
establish an ordering.

Getting it to run at all took four fixes, and **three of the four fail silently
or point somewhere else**, which is the transferable part of this entry:

| # | Symptom | Actual cause | How it fails |
|---|---|---|---|
| 1 | `ModuleNotFoundError: xformers` | remote code does an unguarded `from xformers.ops import SwiGLU` at module scope | loud, obvious |
| 2 | `TypeError: forward() got an unexpected keyword argument 'inputs_embeds'` | `PeftModelForSequenceClassification.forward` passes it unconditionally; NeoBERT has no such parameter | loud, but the naive fix causes #3 |
| 3 | `ValueError: You should supply an encoding ... you provided ['label']` | the shim for #2 written without `functools.wraps` destroys the forward signature that `Trainer._remove_unused_columns` inspects (it unwraps PeftModel via `get_base_model()` to do so), so `input_ids`/`attention_mask` are dropped as "unused columns" | loud, message points nowhere near the cause |
| 4 | *nothing* — trains fine, scores plausibly | `freqs_cis` is registered `persistent=False`, so it is absent from the checkpoint; transformers 5 materializes the model from the state dict on a meta device and leaves the buffer as **uninitialized memory** | **silent** |

Bug 4 is the one worth remembering. `torch.polar(ones_like(f), f)` guarantees
`|freqs_cis| == 1.0` exactly, so any other magnitude is corruption. Measured
across repeated loads of the *same* checkpoint, same input, same device:

```
|freqs_cis| absmax = 1.658e-39   (denormal noise)
|freqs_cis| absmax = 3.251e+35   (huge)
NaN                              (2 of 6 loads in one sample)
```

The NaN loads are the **lucky** ones: they crash. The rest run happily with
rotary embeddings multiplied by garbage — the model loses all positional
information, and still produces plausible activations, a decreasing loss, and a
finished run. Had the first attempt not happened to draw NaN, this would have
been written up as "NeoBERT underperforms on IMDB". The non-determinism is also
what made it hard to find: eight loads gave three different behaviours, so every
hypothesis that blamed a fixed thing (padding, PEFT wrapping, head init) was
consistent with some of the evidence.

Both shims live in `lc_neobert_compat.py` rather than being copy-pasted, and
`patch_neobert()` asserts unit magnitude so #4 can never regress quietly. **The
patch is required on the inference path too**: a saved adapter re-loaded without
it scores at chance, which would look like a corrupted save rather than a
corrupted load. `lc_11_neobert_imdb_test.py` applies it before
`PeftModel.from_pretrained`.

Two smaller traps, both specific to this checkpoint:
- The classification head is **two** layers, `dense` (768→768) then `classifier`
  (768→2), and the load report lists both as MISSING. PEFT's `SEQ_CLS` task type
  adds only `classifier`/`score` to `modules_to_save` by default, so `dense`
  would stay frozen at its random init — a random projection wired in front of
  the classifier, training nothing. Both are named explicitly.
- The revision is **pinned** (`5424c8ef`). This repo ships its model definition
  as remote code, which transformers silently re-downloads when upstream edits
  it; it pulled a new `rotary.py` mid-session while this script was being
  written. `trust_remote_code=True` without a revision pin means the
  architecture can change under a rerun.

Packaging: `sci-ml/xformers-0.0.35` imported into `::pwr` from `::stuff`
2026-09-10. The `::stuff` ebuild has a bug — its patch guards only the first
disjunct of setup.py's three-way accelerator test, so the `TORCH_CUDA_ARCH_LIST`
this host sets globally in `make.conf` for pytorch/caffe2 re-enables the CUDA
build behind `XFORMERS_DISABLE_ACCELERATOR=1`, which then dies on
`unsupported GNU version` (gcc 16.2 vs CUDA 13.3). The `::pwr` copy neutralizes
`TORCH_CUDA_ARCH_LIST` and `FORCE_CUDA` with `local -x` in `python_compile`.
Only `xformers.ops.SwiGLU` is needed, and it falls back to an eager PyTorch
implementation that still runs on CUDA tensors, just unfused.

#### Optuna HPO harness (2026-09-10, `lc_10_hpo_lora.py`)

One harness for every backbone in the tables above (`--model distilbert|electra|
roberta|modernbert|neobert`), replacing `lc_03_distilbert_imdb_optuna.py`. Studies
are SQLite-backed so a run is resumable, the TPE sampler is seeded, and the
published `lc_0N` config is enqueued as trial 0 so a study starts from the known
result. `--final` retrains the winner on the full 25k and saves the adapter.

The per-model registry carries whatever that backbone needs beyond the search
space: `attn` (ModernBERT's FA2), `revision` + `trust_remote_code` +
`modules_to_save` + a `patch` hook (NeoBERT — the pin, the two-layer head, and
`patch_neobert`, all explained in the NeoBERT section above). Batch sizes are
registry constants, not searched: they change the effective batch and therefore
the meaning of `lr`, which would confound the one axis that matters.

Five defects in the old script, worth knowing because four of them are silent:

1. **Learning rate was fixed at 1e-4 and never searched** while r/alpha/dropout were.
2. **The pruner was inert.** `pruner=None` disabled it, and the commented-out
   `MedianPruner` above it would not have worked either — a pruner does nothing
   unless something calls `trial.report()`/`should_prune()`, which with HF Trainer
   needs a callback that did not exist. All 20 trials ran their full 20 epochs.
   The new `OptunaPruningCallback` prunes 19 of 30 on electra.
3. **Optimized `eval_loss` with no `compute_metrics`** — accuracy was never computed,
   and the two diverge precisely when dropout is in the search space, which it was.
4. **`lora_alpha` was suggested over a range depending on `lora_r`**, making the
   search space dynamic; TPE models that badly. Search the alpha/r *ratio* instead.
5. **`prepare_model_for_kbit_training()` on an unquantized model** — that helper is
   for 4/8-bit backbones; here it mainly switched gradient checkpointing on and paid
   recompute for nothing. (`lc_04` has the same spurious call.)

Also: the dataset was re-tokenized inside every trial, there was no `storage=` so a
crash lost the study, the sampler was unseeded, and each trial left an unreaped
checkpoint dir.

**transformers 5 gotcha:** `warmup_ratio` was removed — only `warmup_steps` survives,
so a ratio must be resolved against the real optimizer-step count (`warmup_steps_for`).

**`enqueue_trial()` gotcha:** every enqueued value must be a member of the
distribution the objective declares, or the trial raises on the first `suggest_*()`
call — *after* the dataset is tokenized. The published baselines are r=128/40/36/32
with ratios 1, 3, 5/3, 2, so `r` and `alpha_ratio` are continuous here, not
categorical. `--selftest` round-trips all five baselines through the real search
space on CPU in about a second, and also checks that each `baseline["targets"]`
names a tier that actually exists (a missing tier passes the distribution check and
then `KeyError`s inside `build_model`). Run it after touching the registry.

**Study inventory (2026-09-10):** one real study exists — electra, 5k subset,
30 trials (11 complete, 19 pruned), best 0.9295 — plus a 1-trial diagnostic at
full 25k that reproduced the 0.5000 divergence. distilbert, roberta, modernbert
and neobert have never been searched. That is deliberate, not a backlog: see the
coverage experiment below for why more searching on IMDB is expected to measure
noise.

**What the search actually bought on electra (5k subset, 30 trials): nothing.**
Best 0.9295 vs baseline 0.9250 = +0.45 pp against a ±0.59 pp standard error on a
2000-example eval, and that best is a maximum over 11 completed trials, so it is
biased upward. Best-to-worst spread across all completed trials was 0.8 pp.
Parameter importance flipped between the 3-trial smoke (`lr` 0.79, `r` 0.01) and the
30-trial run (`r` 0.51, `lr` 0.25) — with a 0.8 pp spread it is partitioning noise,
so **neither ranking should be quoted**. Optuna's value here was diagnostic, not
optimizing: running the published config as trial 0 is what exposed the lc_04 story
above. See the subset warning there before trusting any of these numbers.

#### Does LoRA coverage matter more than the tuned knobs? (2026-09-10) — no

`target_modules` was the one axis the harness held constant while searching five
others, and it varies wildly across these scripts — which made it the obvious
suspect for the "HPO bought nothing" result:

| script | adapted modules | count |
|---|---|---|
| `lc_05` roberta-large | `query`, `key` | 2 |
| `lc_04_2` electra-large | `query`, `key`, `value` | 3 |
| `lc_06` ModernBERT-large | `Wqkv`, `Wo` | 2 names, **4 modules** |
| `lc_03` distilbert | full attention + both FFN linears | 6 |
| `lc_11` NeoBERT | `qkv`, `wo`, `w12`, `w3` | 4 names, full coverage |

Note `lc_06`: ModernBERT names *both* `attn.Wo` and `mlp.Wo` the same, and PEFT
matches by **name suffix**, so `["Wqkv", "Wo"]` is not "attention only" as its
comment says — it already adapts the MLP output projection, leaving only
`mlp.Wi` untouched. Verified against the real module tree. Worth checking on any
model before assuming a target list means what it reads like.

`lc_05` is the narrowest coverage in the series *and* roberta-large is the worst
large encoder in the table (92.72%, below electra-large at 20M fewer params), so
"the adapter never touches V, the attention output projection, or the FFN" was a
plausible cause. `lc_05_1` tests it with one variable moved, r held at 36:

| | trainable params | best eval | test (saved) | wall |
|---|---|---|---|---|
| `lc_05` (`query`, `key`) | 4.6M | 92.84% | 92.72% | 8 min |
| `lc_05_1` (6 modules) | 17.0M | 92.80% | **92.77%** | 14 min |

**3.7× the trainable parameters, +0.05 pp.** Same best epoch, same early stop.
Refuted.

The broader reading is what makes this worth keeping: `r`, `alpha` and target
coverage are all **capacity** knobs, and tripling capacity changed nothing, so
capacity is not what limits this task. That is the cleanest explanation for why
the Optuna search bought nothing — it was searching capacity knobs almost
exclusively. The one knob that ever mattered in this series was `lr`, and it
mattered as a **stability threshold**, not a capacity setting: `lc_04` scores
50% at 3e-4 and 93.4% at 1e-4. Threshold effects are exactly what a smooth
sampler like TPE is worst at finding and what a subset study hides entirely.

#### Liger kernel comparison (2026-07-28, `lc_0X_1_*_liger.py` vs plain runs above)

Same hyperparameters, seed and data; the only change is patching the Gemma module
classes with Liger fused Triton kernels (RMSNorm, RoPE, GeGLU) before loading.
Times from checkpoint mtimes (same methodology as above); test accuracy = saved
adapter re-evaluated on the full 25k test set by `lc_0X_1_*_liger_test.py`
(vanilla classes, no Liger — also proves the artifact is portable).

| Script | Model | Time/epoch plain → liger | Speedup | Best eval acc plain → liger | Test acc (saved) |
|---|---|---|---|---|---|
| `lc_07_1` | gemma-2-2b nf4 | 13.8 → 11.9 min | **−13.6%** | 93.50 → 93.41% | 93.41% |
| `lc_08_1` | gemma-2-9b nf4 | 65.5 → 55.7 min | **−15.0%** | 94.56 → 94.63% | 94.59% |
| `lc_09_1` | gemma-3-12b nf4 | 82 → 74.4 min | **−9.3%** | 94.60 → 94.80% | 94.80% |

Notes:
- Verdict: **9–15% faster epochs at unchanged accuracy** — all metric deltas are
  ≤0.2 pp, within single-run sampling noise (do not read the 12B "+0.2 pp" as a win
  without a multi-seed comparison).
- The speedup profile matches the mechanism: the 9B gains most (wide batch 32 with
  grad checkpointing ON → lots of RMSNorm/GeGLU recompute that Liger fuses), the 12B
  least (bs=8, proportionally more time in attention and nf4 quant/dequant, which
  Liger does not touch). FusedLinearCrossEntropy stays inert for SEQ_CLS heads.
- Total wall clock for the three runs: ~7 h 18 min vs ~8 h 20 min plain.

---

### 3. GRPO + LoRA Fine-tuning on GSM8K (`grpo_NN_*`)

Goal: improve mathematical reasoning via GRPO (reinforcement learning).
Dataset: GSM8K (math problems). Framework: unsloth + trl.
Scheme: no suffix = training · `_cont` = continuation · `_test` = evaluation on 500 examples.

#### Models (sorted by increasing size)

| File | Model | Params | VRAM 4-bit | lr | LoRA rank | Notes |
|------|-------|--------|------------|----|-----------|----|
| `grpo_01_gemma1b_gsm8k.py` | gemma-3-1b-it | 1B | ~1GB | 3e-6 | 32 | small model baseline |
| `grpo_02_qwen15b_gsm8k.py` | Qwen2.5-1.5B-Instruct | 1.5B | ~3GB bf16 | 5e-6 | 64 | math-specialist; **bf16 since 2026-07-26** (bug (a) fix) |
| `grpo_02_qwen15b_gsm8k_cont.py` | Qwen2.5-1.5B | — | ~2GB | 5e-6 | 64 | continuation |
| `grpo_03_llama8b_gsm8k.py` | Llama-3.1-8B-Instruct | 8B | ~5GB | 5e-6 | 32 | FA2, `starts_with_reasoning_tag` |
| `grpo_03_llama8b_gsm8k_cont.py` | Llama-3.1-8B | — | ~5GB | 5e-6 | 32 | continuation from checkpoint-2400 |
| `grpo_04_deepseek_r1_8b_gsm8k.py` ★ | DeepSeek-R1-Distill-Llama-8B | 8B | ~5GB | 3e-6 | 32 | CoT baked into weights via RL |
| `grpo_05_qwen3_8b_gsm8k.py` ★ | Qwen3-8B | 8B | ~5GB | 5e-6 | 32 | hybrid `/think`, `<think>` format |
| `grpo_06_dapo_llama8b_gsm8k.py` | Llama-3.1-8B | 8B | ~5GB | 5e-6 | 32 | DAPO: beta=0, overlong penalty, gen=8 |
| `grpo_07_phi4_14b_gsm8k.py` ★ | Phi-4 | 14B | ~9GB | 2e-6 | 16 | strongest reasoning/math per param |
| `grpo_07_1_phi4_gsm8k.py` ★ | Phi-4-mini-instruct | 3.8B | ~5GB | 2e-6 | 16 | no-vLLM variant: rollouts via HF `generate`, `GRPO_ATTN_IMPL=flash_attention_2` |
| `grpo_08_qwen3_4b_gsm8k.py` ★ | Qwen3-4B-Instruct-2507 | 4B | ~3GB | 4e-6 | 16 | non-thinking 2507 refresh; **blocked in 4-bit by bug (a)** — needs ≥24 GB for bf16 |
| `grpo_10_qwen3_17b_gsm8k.py` ★ | Qwen3-1.7B | 1.7B | ~7GB bf16 (2 copies) | 4e-6 | 16 | best small Qwen that actually trains here; `/no_think`, bf16 both sides (bug (a) fix) |

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
| **grpo_10** (Qwen3-1.7B, bf16) | ✅ confirmed working (smoke, 2026-07-26) | New script born from the bug (a) resolution (see below): bf16 on both sides, 2-step smoke clean (`non-finite=0/196`, KL 0, clip 0) + full pipeline incl. offline merge validated end-to-end |
| **grpo_02** (Qwen2.5-1.5B, bf16) | ✅ smoke-clean via probe (2026-07-26) | `grpo_02_buga_bf16_smoke.py` (same model/config skeleton, bf16): `non-finite=0/196`, KL ~1e-4 — flipped the production script to `load_in_4bit=False`; full epoch not yet run |
| **grpo_08** (Qwen3-4B-Instruct-2507) | ❌ blocked in 4-bit — root cause known | Smoke 2026-07-24 + 2026-07-26 (checkpoint swapped to 2507): `\|B\|max=0`, `non-finite=252/252` every step, KL up to 5.6e6 — bug (a), now root-caused (see "Bug (a) RESOLVED" below); 4B bf16 needs two ~8 GB copies → does not fit 16 GB |
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

**Bug (a) RESOLVED 2026-07-26 — root cause: two INDEPENDENTLY bnb-4bit-quantized copies
of the model (trainer vs colocated vLLM engine) numerically disagree.** GRPO compares
per-token logps across the two copies; on sharp-logit checkpoints (the whole Qwen family
+ DeepSeek-R1-distill) the quantization noise blows the per-token logp gap up by orders
of magnitude, and `exp()` of that gap in the importance ratio poisons every LoRA-B
gradient (non-finite from step 1 → optimizer skips → `|B|` stays 0 → the run silently
never trains) or crashes backward with an OOB gather assert (the grpo_02/grpo_04
`index out of bounds: 0 <= tmp0 < <vocab>` family). gemma/llama/phi have milder logit
distributions and tolerate the same noise — which is exactly the "weights, not
architecture" discriminator observed on 2026-07-24.

Probe chain (logs `/var/tmp/grpo08_2507_*.log`, `/var/tmp/grpo02_buga_*.log`):

| Probe | Config | Result |
|---|---|---|
| checkpoint swap | grpo_08 → `Qwen/Qwen3-4B-Instruct-2507` (different weights, same dense arch) | still sick (252/252 non-finite, KL 7475→5.6e6) → whole Qwen family, not one checkpoint |
| compiled kernels | `UNSLOTH_COMPILE_DISABLE=1` | still sick → kernels exonerated |
| KL term | `GRPO_BETA=0` | `kl=0` but grads still non-finite → KL estimator exonerated; telltale left standing: `clip_ratio ~6%` and loss 15–20 at step 1 with B=0 (healthy models: ratio≈1, loss≈0) = trainer-vs-engine logp divergence |
| **A/B decider** | `grpo_02_buga_bf16_smoke.py` (Qwen2.5-1.5B, ONLY `load_in_4bit` toggled) | **bf16 clean** (0/196 non-finite, KL ~1e-4, clip ~3e-4) vs **4-bit sick** (backward OOB crash) |

**Fix: `load_in_4bit=False` — one bf16 numerical identity shared by both sides.** The
important nuance: 4-bit per se is NOT broken, the *pair of divergent quantized copies* is:

| Configuration | Qwen / R1-distill (sharp logits) | gemma / llama / phi (mild logits) |
|---|---|---|
| 4-bit + colocated vLLM (2 copies) | **sick** (quant noise × sharp logits) | healthy (noise tolerated) |
| bf16 + colocated vLLM (2 copies) | healthy (copies numerically identical) | healthy |
| 4-bit, no vLLM (1 copy, HF-generate rollouts) | expected healthy — **untested** | healthy (grpo_07_1 validated on Phi) |

Consequences on 16 GB (bf16 needs TWO full weight copies — trainer + engine):
- ≤2B models train at full vLLM rollout speed: **grpo_10** (Qwen3-1.7B, new script,
  smoke + merge validated) and **grpo_02** (Qwen2.5-1.5B) flipped to bf16.
- Qwen3-4B (grpo_08, now on Instruct-2507) stays blocked in 4-bit: bf16 would need
  2×8 GB. Escape hatches: a ≥24 GB GPU, a future matched-quantization scheme, or the
  third row above — a grpo_07_1-style no-vLLM variant (`use_vllm=False` +
  `UNSLOTH_DISABLE_FAST_GENERATION=1`, single 4-bit copy, no divergence possible) at the
  cost of much slower rollouts (~3–5× longer epoch). Candidate: `grpo_08_1`, not written yet.
- Qwen3.5 small series (March 2026: 0.8B/2B/4B/9B) evaluated and rejected for now:
  Gated-DeltaNet + sparse-MoE hybrid, multimodal, requires transformers main — outside
  what unsloth `FastLanguageModel` + vLLM LoRA hot-load support.
- Upstream issue candidate (unsloth): "GRPO fast_inference + load_in_4bit silently never
  trains on sharp-logit checkpoints (trainer and engine quantized independently)".

---

### 4. Other

| File | Description |
|------|-------------|
| `intutor_pw_02.py` | English tutor CEFR A1-C1, OpenAI API, topics by level |

---

### 5. Inference Runtime Benchmarks (`bench_NN_*.sh`)

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
| `grpo_08_qwen3_4b_gsm8k_test.py` | `outputs/lora-grpo-qwen3-4b-2507-r16` | `<answer>` only (non-thinking Instruct-2507) |
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
| `lc_11_neobert_imdb_test.py` | `outputs/lora-neobert-imdb` |

```bash
python lc_06_modernbert_imdb_test.py  # best result, 96.16%
python lc_11_neobert_imdb_test.py     # 95.92% from 145M fewer params
```

IMDB files print a full `classification_report` (precision/recall/F1 per class) and final accuracy.

All IMDB test files (rewritten 2026-07-27) load the base model explicitly and apply the
adapter via `PeftModel.from_pretrained`, then run batched inference over the **full 25k
test set**. Do not load the adapter directory directly with `AutoModelForSequenceClassification`
— it silently attaches a randomly initialized classification head (see "Final results"
notes in section 2). Results: section 2, "Final results (2026-07-27)".

`lc_11` additionally calls `patch_neobert()` from `lc_neobert_compat.py` on the base
model **before** applying the adapter. This is not optional on the inference path:
without it NeoBERT's `freqs_cis` buffer is uninitialized memory and a perfectly good
adapter scores at chance, which reads as a corrupted save rather than a corrupted
load. Section 2, "NeoBERT", has the details.

---

## Output Directory Structure

All model checkpoints, saved models, and logs are written to `outputs/` (excluded from git via `.gitignore`):

```
outputs/
  lora-grpo-phi4/                  # merged 16-bit model (grpo_07)
  lora-grpo-phi4-outputs/          # training checkpoints
  lora-grpo-phi4-mini-novllm/          # merged 16-bit model (grpo_07_1, no-vLLM/fa2)
  lora-grpo-phi4-mini-novllm-outputs/  # training checkpoints
  lora-grpo-qwen3-4b-2507-r16/     # (grpo_08 — blocked in 4-bit, see bug (a))
  lora-grpo-qwen3-17b-r16/         # (grpo_10 — Qwen3-1.7B bf16)
  ...
  lora-bert-roberta2/              # lc_ fine-tuned models
  lora-modernbert-imdb/
  ...
  best_encoder_classic.pt          # simple_02 checkpoints (also _modern, _library)
  best_hybrid_depth1.pt            # simple_06 checkpoint (also _depth3)
  imdb_train_augmented.csv         # simple_05 output
  scaling_ablation.png             # simple_07 plot
  logs/                            # HuggingFace Trainer logs
  optuna-distilbert-*/             # Optuna trial checkpoints
```

---

## Installation — Blackwell (legacy Python venv recipes, CUDA 12.x era)

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

### Current install method: Gentoo ebuilds (pwr overlay)

The Python venv recipes above are **legacy** and kept for reference only. The whole
training/inference stack is now installed system-wide via Gentoo ebuilds from the
[`pwr` overlay](https://github.com/pwasiewi/pwr) — PyTorch (`sci-ml/caffe2`),
`dev-python/vllm`, `dev-python/triton-bin`, `sci-ml/unsloth` + `sci-ml/unsloth-zoo`,
`dev-python/flash-attn`, `dev-python/bitsandbytes`, plus TRL/PEFT/accelerate.
Portage tracks current upstream versions (live `-9999` ebuilds where needed), keeps
the ABI of the whole stack in lock-step, and carries the local patches (Blackwell
sm_120 fixes, Unsloth GRPO fixes) that a venv build would lose on every reinstall.

```sh
sudo emaint sync -r pwr
sudo emerge sci-ml/caffe2 dev-python/vllm sci-ml/unsloth
```

### LED / RGB (Blackwell)
- https://gitlab.com/CalcProgrammer1/OpenRGB/-/issues/4710

### NVIDIA suspend
- https://forums.developer.nvidia.com/t/rtx-5070-ti-with-570-124-04-won-t-resume-monitor-from-suspend-to-ram/327297
