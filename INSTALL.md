python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-deps -r requirements-lc07.txt

Why:

  - --system-site-packages lets the venv use the global Gentoo packages: torch, transformers, accelerate, scikit-learn, datasets, the CUDA stack.
  - --no-deps stops pip from pulling in its own torch + CUDA wheels, which could break/shadow the Gentoo build for the RTX 5070 Ti Blackwell.

Full sequence for llm:

cd $HOME/Codex/llm

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --no-deps -r requirements-lc07.txt

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python lc_07_gemma2b_imdb.py

# Reproducing BENCH.md results

First get models like that
```bash
hf download hf://unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf  --local-dir ~/models/gpt-oss20b-q8_0
hf download hf://unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf --local-dir ~/models/gemma4-12b-q8_0
```

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

## 2026-07-15 — 9× single-model repeat, gpt-oss20b-q8_0 only (extends the 5x below to 14 total)

```bash
for i in 1 2 3 4 5 6 7 8 9; do
  MODELS=gpt-oss20b-q8_0 TASK_TIMEOUT=1200 ./bench_05_agentic.sh
done
# Combined with the 5x batch below: 14 total runs. Found codec/toposort can
# also fail (13/14 each, first time ever seen) -- the "rock solid" tier
# isn't literally 100%, just far more reliable than the parser tier. Also
# surfaced a real bug in bench_05_agentic.sh: verify_regex()/verify_interp()
# ran ast.parse() unguarded, so a genuine SyntaxError in the model's file
# was mislabeled as a banned-import/eval disqualification. Fixed in this
# script (SyntaxError now caught explicitly, distinct NOTE). Details:
# BENCH.md 2026-07-15 "Second follow-up" section.
```

## 2026-07-15 — 5× single-model repeat, gpt-oss20b-q8_0 only (interp/regex noise check)

```bash
for i in 1 2 3 4 5; do
  MODELS=gpt-oss20b-q8_0 TASK_TIMEOUT=1200 ./bench_05_agentic.sh
done
# Isolates gpt-oss20b-q8_0 alone (no ornith-128k in the mix) to measure its
# own run-to-run noise cleanly. Result: interp and regex both FAIL ~2/5
# runs -- comparably noisy, not one rare fluke (see BENCH.md 2026-07-15
# follow-up section). NOTE: shared WORKROOT across the loop overwrites
# each run's agent.log -- use WORKROOT=/tmp/bench-agentic-run$i per
# iteration next time to keep evidence from failing runs.
```

## 2026-07-15 — 3× full 12-task sweep, gpt-oss20b-q8_0 vs ornith-128k (weak-spot hunt)

```bash
for i in 1 2 3; do
  MODELS=gpt-oss20b-q8_0,ornith-128k TASK_TIMEOUT=1200 ./bench_05_agentic.sh
done
# default TASKS (all 12); TASK_TIMEOUT raised globally 900->1200 for the
# slow parser tier (ornith-128k) and long-reasoning-chain tasks on the
# fast model (gpt-oss20b-q8_0) alike. Result: ornith-128k/regex TIMEOUT
# in all 3 runs even at 1200s (see BENCH.md 2026-07-15 section) -- confirms
# it's a capability ceiling, not a budget problem.
```

## 2026-07-14 — official unsloth Qwen3.6-35B-A3B-GGUF vs Ornith

```bash
# ncmoe floor sweep per quant (descending probe)
M=$HOME/models/qwen36-unsloth/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
MODEL=$M NCMOE_LIST=12,10,9   ./bench_01_llamacpp.sh -b   # floor 10 (9 OOMs)

M=$HOME/models/qwen36-unsloth/Qwen3.6-35B-A3B-MXFP4_MOE.gguf
MODEL=$M NCMOE_LIST=20,18,16 ./bench_01_llamacpp.sh -b
MODEL=$M NCMOE_LIST=15,14,13 ./bench_01_llamacpp.sh -b    # all OOM → floor 16

M=$HOME/models/qwen36-unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
MODEL=$M NCMOE_LIST=24,20,18 ./bench_01_llamacpp.sh -b
MODEL=$M NCMOE_LIST=16,15,14 ./bench_01_llamacpp.sh -b    # all OOM
MODEL=$M NCMOE_LIST=17       ./bench_01_llamacpp.sh -b    # floor 17

# agentic comparison @128K, TASK_TIMEOUT=900 (default)
# profiles from ~/.aillama/models.conf at test time — qwen36u-iq4xs*/
# qwen36u-q4kxl* were deleted afterward (worst 2 of 3, see BENCH.md);
# qwen36u-mxfp4* is the survivor.
MODELS=qwen36u-iq4xs-128k,qwen36u-mxfp4-128k,qwen36u-q4kxl-128k,ornith-128k \
TASKS=bugfix,scratch,lru,multifile,template,interp,perf,regex \
./bench_05_agentic.sh

# ornith-128k/interp TIMEOUTed at 900s (10/13 tests, unary minus unfinished);
# re-run alone at the 07-13-tiebreak budget to check if it was budget, not capability:
MODELS=ornith-128k TASKS=interp TASK_TIMEOUT=1200 ./bench_05_agentic.sh   # PASS 1067s
```

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


```bash
#bench_06 (dense, jeden model na wywołanie, surowa ścieżka GGUF z models.conf):
MODEL=$HOME/models/gemma4-12b-q8_0/gemma-4-12b-it-Q8_0.gguf ~/Claude/llm/bench_06_dense_generic.sh

MODEL=$HOME/models/gpt-oss-20b-q8_0/gpt-oss-20b-Q8_0.gguf CTX=131072 ~/Claude/llm/bench_06_dense_generic.sh
#(CTX=131072 bo gpt-oss20b-q8_0 ma w models.conf -c 131072, domyślne CTX w bench_06 to 65536 — zgodne z gemma4-12b-it, więc tam bez override.)

#bench_05 (agentic, oba modele jednym runem — aillama switch per profil):
MODELS=gemma4-12b-it,gpt-oss20b-q8_0 ~/Claude/llm/bench_05_agentic.sh

#bench_07 (workflow-discipline, domyślnie TASKS=relmeta RUNS=3):
MODELS=gemma4-12b-it,gpt-oss20b-q8_0 ~/Claude/llm/bench_07_workflow.sh

MODELS=gpt-oss20b-q8_0,gpt-oss20b-udq8kxl RUNS=5 ~/Claude/llm/bench_07_workflow.sh
```
