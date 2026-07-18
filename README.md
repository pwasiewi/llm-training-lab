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
