# simple_* — a hands-on lesson series on how LLMs work

Seven standalone PyTorch scripts, each one lesson. They share no code on
purpose: every file is readable top-to-bottom, and the comments carry the
theory. All run on a single GPU (or CPU, slower) in minutes with their
default configs; every script ends with a "Things to try" list.

## Learning path

| # | Script | Lesson | Key concepts |
|---|--------|--------|--------------|
| 1 | `simple_01_gpt_tinystories.py` | Pretrain a tiny GPT from scratch | BPE, causal mask, next-token loss, perplexity, weight tying, temperature/top-k sampling |
| 2 | `simple_02_imdb_encoder.py` | Encoder classifier, classic vs modern (`--arch classic\|modern`) | bidirectional attention, padding mask, masked mean-pool, RoPE, GeGLU, bias-free pre-norm, alternating global/local sliding-window attention (ModernBERT) |
| 3 | `simple_03_imdb_library.py` | Same encoder with `nn.TransformerEncoder` | `batch_first`, `norm_first`, `src_key_padding_mask` (inverted convention!), bf16 autocast without GradScaler |
| 4 | `simple_04_vram_math.py` | Training memory on a napkin (no GPU needed) | 16 bytes/param AdamW rule, fp32 vs bf16, quadratic attention memory, sliding window at seq 8192, FlashAttention caveat |
| 5 | `simple_05_augmentation.py` | Label-preserving text augmentation | WordNet+POS synonym replacement, EDA random swap/delete, per-op change stats, when augmentation is UNsafe |
| 6 | `simple_06_hybrid_pretransformer.py` | Inductive biases: transformer + BiLSTM + CNN (`--lstm-depth 1\|3`) | recurrence vs attention, n-gram detectors, max-over-time pooling, residual projections, **tokenizer data-leakage post-mortem** |
| 7 | `simple_07_scaling_ablation.py` | Measure cost vs sequence length | log-log scaling exponents, quadratic vs linear, CUDA timing hygiene (warmup, synchronize, peak memory), token-length percentiles |

Suggested order is 1 → 7. Lessons 1–3 build the transformer twice (by hand,
then with the library); 4 is the budgeting interlude; 5–6 are the
data-and-architecture history lessons; 7 closes the loop by measuring what
1–4 claimed.

## Old → new file mapping (2026-07-30 rework)

| Old file | Became | Notes |
|----------|--------|-------|
| `simple_llm03.py` | `simple_01_gpt_tinystories.py` | + PAD `ignore_index` fix, perplexity/tok/s logging, sampling demo, weight tying |
| `simple_imdb.py` | `simple_02_imdb_encoder.py` | + padding mask & masked mean-pool (old version attended to PAD), + `--arch modern` with ModernBERT fragments |
| `simple_imdb_trf.py` | `simple_03_imdb_library.py` | + `batch_first=True` (drops the permute dance), padding mask, bf16 autocast |
| `simple_imdb_check.py` | `simple_04_vram_math.py` | rewritten estimator: dtype-aware, AdamW states, global-vs-window comparison, 512 vs 8192 ctx |
| `simple_imdb_augmented.py` + `simple_imdb_extended.py` | `simple_05_augmentation.py` | merged; + swap/delete ops, graceful no-nltk mode, per-op stats |
| `simple_lstm_cnn_trf_res.py` + `simple_lstm_cnn_trf_res4.py` | `simple_06_hybrid_pretransformer.py` | merged behind `--lstm-depth`; res4's train+test tokenizer leakage fixed and documented |
| `simple_lstm_cnn_check.py` | `simple_07_scaling_ablation.py` | seaborn dropped, tokenizer trained once, arbitrary "optimal length" heuristic replaced with fitted scaling exponents |

## Requirements

`torch` (CUDA optional), `datasets`, `tokenizers`, `scikit-learn`, `tqdm`,
`pandas`, `numpy`, `matplotlib` (lesson 7), `nltk` (lesson 5's synonym op
only — the script degrades gracefully without it). Checkpoints, CSVs and
plots land in `./outputs/`.
