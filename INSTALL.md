python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-deps -r requirements-lc07.txt

Why:

  - --system-site-packages lets the venv use the global Gentoo packages: torch, transformers, accelerate, scikit-learn, datasets, the CUDA stack.
  - --no-deps stops pip from pulling in its own torch + CUDA wheels, which could break/shadow the Gentoo build for the RTX 5070 Ti Blackwell.

Full sequence for llm:

cd /home/pwas/Codex/llm

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --no-deps -r requirements-lc07.txt

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python lc_07_gemma2b_imdb.py

# Reproducing BENCH.md results

Exact invocations behind each "Reference results" section of BENCH.md.
Older sections (before 2026-07-13) are reconstructed from the parameters
documented there plus script defaults. General rules:

- Free the GPU first: `aillama stop`; expect <1 GB desktop VRAM
  (`nvidia-smi --query-gpu=memory.used --format=csv`). Results shift with
  desktop VRAM load and page cache (first run after boot reads the whole
  GGUF from disk).
- When probing an `--n-cpu-moe` floor, go from a known-good HIGH value
  downward (every run measures something); probing upward burns runs on
  OOM crashes.
- All commands run from `~/Claude/llm/`.

## 2026-07-13 — GLM-4.7-Flash refresh (bench_01 defaults = this model)

```bash
./bench_01_llamacpp.sh                    # sweep NCMOE_LIST=14,10,8 (8 now crashes)
SERVER_NCMOE=10 ./bench_01_llamacpp.sh -s # → OOM on first decode (was the default then;
                                          #   script default bumped to 12 after this run)
SERVER_NCMOE=12 EXTRA_SERVER_ARGS="-np 1" ./bench_01_llamacpp.sh -s   # pp 977 / tg 63
```

## 2026-07-13 — glm-flash agentic (bench_05, 4/7)

```bash
MODELS=glm-flash ./bench_05_agentic.sh    # default TASKS=bugfix,...,perf; TASK_TIMEOUT=900
# failure inspection: cd /tmp/bench-agentic/glm-flash/<task> && python -m pytest -q
```

## 2026-07-13 — glm-flash at 128K (floor probe, downward)

```bash
for n in 28 24 22 20; do
    CTX=131072 SERVER_NCMOE=$n EXTRA_SERVER_ARGS="-np 1" ./bench_01_llamacpp.sh -s
done                                      # 22 = floor (20 aborts at boot); profile uses 24
```

## 2026-07-13 — ik_llama.cpp (4721) vs mainline, Ornith @ 128K

```bash
sudo emerge -1 sci-misc/ik-llama-cpp      # live 9999 rebuild; version check:
ik-llama-server --version                 # 4721 (7937465f) = upstream main HEAD

M=$HOME/models/ornith-1.0-35b/ornith-1.0-35b-Q4_K_M.gguf
MODEL=$M NCMOE_LIST=24,20,18 ./bench_02_ikllama.sh -b            # ik bench sweep

# server A/B, short prompt (~1118 tok):
MODEL=$M CTX=131072 SERVER_NCMOE=24 ./bench_02_ikllama.sh -s     # ik    595/50.3
MODEL=$M CTX=131072 SERVER_NCMOE=24 EXTRA_SERVER_ARGS="-np 1" ./bench_01_llamacpp.sh -s  # main 459/51.3

# server A/B, deep prompt (~25K tok):
MODEL=$M CTX=131072 SERVER_NCMOE=24 PROMPT_REPEATS=1400 ./bench_02_ikllama.sh -s         # ik    730/46.6
MODEL=$M CTX=131072 SERVER_NCMOE=24 PROMPT_REPEATS=1400 EXTRA_SERVER_ARGS="-np 1" ./bench_01_llamacpp.sh -s  # main 714/48.9

# floor probe at 128K (both work at 22, kept 24 for margin):
MODEL=$M CTX=131072 SERVER_NCMOE=22 ./bench_02_ikllama.sh -s
MODEL=$M CTX=131072 SERVER_NCMOE=22 EXTRA_SERVER_ARGS="-np 1" ./bench_01_llamacpp.sh -s
```

## 2026-07-12/13 — agentic: qwen36 / qwythos / ornith + tiebreak (reconstructed)

```bash
MODELS=qwen36-128k,qwythos ./bench_05_agentic.sh                 # 5-task round
MODELS=ornith ./bench_05_agentic.sh          # template FAILs: 32K ctx exceeded mid-task
MODELS=ornith-128k ./bench_05_agentic.sh     # agent profiles need >=64K ctx
MODELS=qwen36-128k,ornith-128k TASKS=interp,perf TASK_TIMEOUT=1200 ./bench_05_agentic.sh  # tiebreak
```

## 2026-07-12 — Qwythos-9B-v2 depth/MTP (reconstructed)

```bash
./bench_04_qwythos.sh          # DEPTH_LIST=0,16384,65536,131072; -b/-s/-m for single parts
```

## 2026-07-10 — Qwen3.6-35B-A3B, mainline vs ik (reconstructed)

```bash
M=$HOME/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf
MODEL=$M NCMOE_LIST=14,12,10,9 ./bench_01_llamacpp.sh -b         # floor 10 (9 OOMs)
MODEL=$M NCMOE_LIST=14,12,10  ./bench_02_ikllama.sh -b

# server at each context size (mainline; ik: bench_02 + its own floor, 32K needs 11):
MODEL=$M CTX=32768  SERVER_NCMOE=10 ./bench_01_llamacpp.sh -s
MODEL=$M CTX=65536  SERVER_NCMOE=12 ./bench_01_llamacpp.sh -s
MODEL=$M CTX=131072 SERVER_NCMOE=14 ./bench_01_llamacpp.sh -s

# deep prompt (~29K tok) at 64K:
MODEL=$M CTX=65536 SERVER_NCMOE=12 PROMPT_REPEATS=1400 ./bench_01_llamacpp.sh -s
MODEL=$M CTX=65536 SERVER_NCMOE=12 PROMPT_REPEATS=1400 ./bench_02_ikllama.sh -s
```

## 2026-07-05 — GLM-4.7-Flash, llama.cpp vs ik vs ollama (reconstructed)

```bash
./bench_01_llamacpp.sh                    # defaults; ncmoe 8 was still the floor then
SERVER_NCMOE=18 ./bench_02_ikllama.sh     # ik server needs 18 at 32K on GLM
./bench_03_ollama.sh                      # imports GGUF as glm47-flash-q4 (extra ~17 GB!)
ollama rm glm47-flash-q4                  # cleanup after
```
