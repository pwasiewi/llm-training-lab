# --- Silence cosmetic third-party startup noise (must run before heavy imports) ---
import os, warnings, logging
os.environ.setdefault("GLOG_minloglevel", "2")          # caffe2/glog: hide INFO+WARNING (GroupedMMUtils fallback, InitGoogleLogging)
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")  # drop vLLM INFO banner; keep warnings/errors
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut VRAM fragmentation on the 16 GB card
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# The "[ERROR] ... not documented" lines are unsloth-zoo docstring checks, not real errors:
logging.getLogger("unsloth_zoo").setLevel(logging.CRITICAL)
# ----------------------------------------------------------------------------------

import torch
torch.backends.cuda.enable_flash_sdp(True)
# GRPO_ANOMALY=1: autograd anomaly mode — pinpoints the backward op that creates the
# NaN gradients found by GRAD-WATCH (2026-07-24: non-finite grads in most micro-batches,
# optimizer sees 128/128 NaN, lora_B stays zero -> every run so far was a no-op).
# ~3x slower; combine with GRPO_MAX_STEPS=1. If the report only names a compiled-function
# node, rerun with TORCHDYNAMO_DISABLE=1 as well — and if the NaN disappears in eager
# mode, the bug lives in the compiled Triton kernels (triton/torch pin mismatch trail).
if os.environ.get("GRPO_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly(True)
    print("GRPO_ANOMALY=1: autograd anomaly detection ON", flush=True)
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
import re
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm

os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
os.environ["VLLM_USE_V1"] = "0"
os.environ["FLASH_ATTENTION_USE_FA2"] = "1"
os.environ["XFORMERS_MEM_EFF_ATTN"] = "0"

# grpo_07_2: same validated colocated-vLLM setup as grpo_07 (all known bugs (b)/(c)/(d)
# fixed/worked around, confirmed live in the installed unsloth-zoo 2026-07-26 — see the
# README "Current status" table). NOTE (2026-07-26 smoke): grpo_07 was already fast
# (train_runtime 1.917e4s = ~5.3h for a full epoch, ~20s/step steady state) — the 25h
# run was grpo_07_1 (no-vLLM control), not this script. A first attempt here at 4x
# batching (num_generations/generation_batch_size/per_device_train_batch_size=4,
# gpu_memory_utilization=0.65) OOM'd at step 2 (~2.3 GiB short) — the HF-side logp
# scoring forward scales directly with generation_batch_size (see grpo_07's "TRL scores
# the whole generation batch in one logp forward" note), it isn't vLLM-side throughput
# that was the bottleneck. So this version keeps grpo_07's batch shape and only spends
# the ~5 GiB of previously-unused desktop headroom (measured idle use ~1 GiB, not the
# ~6-8 GiB the old comment assumed) on a slightly larger vLLM KV-cache budget.
model_name = "microsoft/Phi-4-mini-instruct"
model_path = "outputs/lora-grpo-phi4-mini-v2"
max_seq_length = 2048
max_prompt_length = 512
lora_rank = 16   # smaller rank: 14B already has high capacity, saves activations memory

# GRPO_NO_VLLM=1: control probe (2026-07-24, see grpo_07_1) — rollouts via HF generate
# instead of the colocated vLLM engine. Kept here only as an emergency fallback switch;
# the no-vLLM+fa2 path was already validated end-to-end in grpo_07_1 (full epoch, GSM8K
# format 80.97%/acc 88.25%). flash_attn is now installed (2.8.3.post1, confirmed
# importable 2026-07-26), so the no-vLLM branch can use flash_attention_2 too.
_no_vllm = os.environ.get("GRPO_NO_VLLM") == "1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
    device_map="auto",
    fast_inference=not _no_vllm,
    # Mild bump from 0.5 (not 0.65 — that OOM'd the HF-side trainer step at batch 4x,
    # see note above). Idle desktop holds ~1 GiB (measured 2026-07-26), so 0.55 still
    # leaves headroom for the trainer while giving vLLM's KV cache a bit more room.
    gpu_memory_utilization=0.55,
    max_lora_rank=lora_rank,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    # Phi-4-mini fuses attention into qkv_proj and the MLP into gate_up_proj; the
    # unfused Llama-style names match nothing, so the old list put LoRA only on
    # o_proj + down_proj. Fused-module LoRA in vLLM needs the pwr patches
    # (fused-packed-module wrap + no-stacked-name-mapping) — verified live 2026-07-26
    # (unsloth_zoo/vllm_utils.py:1207-1215 handles gate_up_proj without the up_proj dup).
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    # alpha MUST equal rank (s=1): unsloth_zoo load_lora_directly applies s=alpha/r
    # IN-PLACE on an aliased vLLM buffer 8x per step, so any s>1 compounds —
    # s=2 grew |B|max x256/step and rotted rollouts into gibberish (bug (c),
    # integrity probe 2026-07-24). The installed unsloth-zoo now hard-raises
    # RuntimeError instead of silently corrupting if s != 1 on an aliased buffer
    # (verified live 2026-07-26) — but alpha=rank is still required to train at all.
    lora_alpha=lora_rank,
    # GRPO_GC selects the gradient-checkpointing implementation (A/B probes 2026-07-24):
    #   hf (default) — plain torch/HF per-layer checkpointing; clean grads confirmed
    #     (grad_norm=0.063, GRAD-WATCH non-finite=0, compiled mode);
    #   unsloth — unsloth-zoo offloading GC; PROVEN to produce NaN gradients in most
    #     micro-batches -> optimizer gets NaN -> zero updates (no-op training);
    #   off — disabled entirely (clean grads confirmed; VRAM-heavy, probes only).
    use_gradient_checkpointing={"unsloth": "unsloth", "hf": True, "off": False}[
        os.environ.get("GRPO_GC", "hf")],
    random_state=3407,
)

def extract_hash_answer(text: str) -> str:
    # Try the <answer> tag first; DOTALL because the trained format puts newlines inside the tags
    tag_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if tag_match:
        content = tag_match.group(1).strip()
        # Mixed fraction like "10 1/4"
        frac = re.fullmatch(r'(-?\d+)\s+(\d+)\s*/\s*(\d+)', content)
        if frac and int(frac.group(3)) != 0:
            whole, num, den = frac.groups()
            value = abs(int(whole)) + int(num) / int(den)
            return str(-value if whole.startswith('-') else value)
        # Strip currency symbols and whitespace (joins space-grouped thousands like "10 000"),
        # then drop comma thousands separators
        norm = re.sub(r'[$€£\s]', '', content).replace(',', '')
        if re.fullmatch(r'-?\d+(\.\d+)?', norm):
            return norm
    if "####" in text:
        return text.split("####")[1].strip().replace(",", "")
    # Fallback: last number in the whole text (comma- or space-grouped thousands accepted)
    matches = re.findall(r"-?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?", text)
    return matches[-1].replace(",", "").replace(" ", "") if matches else None

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    return data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': (
                "You are a helpful assistant that always responds using <reasoning> and <answer> tags."
            )},
            {'role': 'user', 'content': (
                "Please solve the following problem and respond in this format:\n"
                "<reasoning>...</reasoning>\n"
                "<answer>...</answer>\n\n"
                "Start your response with the <reasoning> tag and include all calculation steps inside it.\n"
                "Problem:\n"
                f"{x['question']}"
            )}
        ],
        'answer': extract_hash_answer(x['answer'])
    })

dataset = get_gsm8k_questions()

def safe_float(x):
    try:
        return float(x.replace(",", "").strip())
    except:
        return None

_gen_batch = 0  # one reward call = one generation round (num_generations completions of one prompt)

def _progress():
    global _gen_batch
    _gen_batch += 1
    st = trainer.state if "trainer" in globals() else None
    s = f"batch {_gen_batch}/{len(dataset)}"
    if st is not None and st.epoch is not None:
        s += f" | step {st.global_step}/{st.max_steps} | left {st.max_steps - st.global_step} steps | epoch {st.epoch:.3f}"
    return s

def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [c[0]['content'] for c in completions]
    extracted = [extract_hash_answer(r) for r in responses]
    print('-'*20, f"[{_progress()}]", f"Question:\n{prompts[0]}", f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted[0]}", f"\nAnswer:\n{answer[0]}", flush=True)
    return [2.0 if safe_float(r) == safe_float(a) else 0.0 for r, a in zip(extracted, answer)]

def is_integer_like(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [c[0]['content'] for c in completions]
    extracted = [extract_hash_answer(r) for r in responses]
    return [0.5 if is_integer_like(r) else 0.0 for r in extracted]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>\s*$"
    responses = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def count_xml(text: str) -> float:
    return sum(0.125 for tag in ["<reasoning>", "</reasoning>", "<answer>", "</answer>"] if tag in text)

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c[0]["content"]) for c in completions]

def starts_with_reasoning_tag(completions, **kwargs):
    return [1.0 if c[0]["content"].strip().startswith("<reasoning>") else 0.0 for c in completions]

training_args = GRPOConfig(
    use_vllm=not _no_vllm,
    learning_rate=4e-6,   # 2e-6 doubled to offset LoRA scale s: 2 -> 1 (alpha=rank workaround)
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    # Same shape as grpo_07 (2/2/2) — smoke-tested 2026-07-26: 4/4/4 OOM'd at step 2.
    # The logp-scoring forward (HF side) scales with generation_batch_size regardless
    # of vLLM's own batching, so this is not free to raise on a 16 GB card at this
    # model/seq-length. Left at grpo_07's validated values; the actual speedup vs
    # grpo_07_1 already comes from being on vLLM at all (~20s/step vs ~90s+/step).
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=2,
    generation_batch_size=2,
    unsloth_grpo_mini_batch=1,
    unsloth_logit_chunk_multiplier=16,
    # TRL 1.8 removed max_prompt_length from GRPOConfig; prompts are no longer
    # truncated by the config. vllm_max_model_length sets the vLLM context window
    # (>= max prompt length in the dataset + max_completion_length).
    max_completion_length=max_seq_length - max_prompt_length,
    vllm_max_model_length=max_seq_length,
    num_train_epochs=1,
    # GRPO_MAX_STEPS=N caps the run for diagnostics (anomaly probe); -1 = full epoch.
    max_steps=int(os.environ.get("GRPO_MAX_STEPS", "-1")),
    save_strategy="steps",
    save_steps=50,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"{model_path}-outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        starts_with_reasoning_tag,
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# --- GRAD-WATCH (2026-07-24): grpo_06 checkpoint-50 proved the optimizer receives
# exactly-zero gradients every step (lora_B all zero, Adam moments absmax=0) while
# grad_norm logs nan. This probe pins down WHERE the gradient dies:
#   * per-tensor backward hooks fire during loss.backward() itself — if their call
#     count stays 0, the autograd graph never reaches the LoRA params (detached graph);
#     if they fire with zeros, backward computes zero contributions;
#     if they fire with non-finite values, real NaN grads get zeroed later;
#   * a global optimizer step pre-hook reports what the optimizer actually sees.
# Trainer-agnostic on purpose: works regardless of unsloth's exec'd inner loop.
_watch_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
_bwd_stats = {"calls": 0, "nonzero": 0, "nonfinite": 0}

# Integrity probes (bug (c), 2026-07-24): frozen base weights must NEVER change during
# training; the LoRA magnitudes must stay tiny (~lr x steps). If a base checksum drifts
# -> something writes into frozen weight memory (OOB write). If |B|max explodes ->
# the optimizer path is at fault after all.
_frozen_pool = [(n, p) for n, p in model.named_parameters()
                if not p.requires_grad and p.numel() > 1_000_000]
_frozen_probes = [_frozen_pool[i] for i in
                  sorted({0, len(_frozen_pool) // 2, len(_frozen_pool) - 1})] if _frozen_pool else []

def _checksum(p):
    d = p.data
    if d.dtype in (torch.uint8, torch.int8):
        return int(d.view(torch.uint8).long().sum().item())
    return float(d.float().abs().sum().item())

_frozen_base = {n: _checksum(p) for n, p in _frozen_probes}

def _adapter_mags():
    a = max((p.data.abs().max().item() for n, p in _watch_params if "lora_A" in n), default=0.0)
    b = max((p.data.abs().max().item() for n, p in _watch_params if "lora_B" in n), default=0.0)
    return a, b

def _mk_bwd_hook():
    def _hook(grad):
        _bwd_stats["calls"] += 1
        if not torch.isfinite(grad).all():
            _bwd_stats["nonfinite"] += 1
        elif grad.abs().max().item() > 0:
            _bwd_stats["nonzero"] += 1
        return grad
    return _hook

for _n, _p in [(n, p) for n, p in _watch_params if "lora_B" in n][:8]:
    _p.register_hook(_mk_bwd_hook())

_gw_step = {"n": 0}

def _grad_watch(optimizer, *args, **kwargs):
    _gw_step["n"] += 1
    if _gw_step["n"] > 10 and _gw_step["n"] % 25 != 0:
        return
    present = absent = nonzero = nonfinite = 0
    absmax = 0.0
    for _, p in _watch_params:
        if p.grad is None:
            absent += 1
            continue
        present += 1
        if not torch.isfinite(p.grad).all():
            nonfinite += 1
        m = p.grad.abs().max().item()
        if m > 0:
            nonzero += 1
        if m == m and m != float("inf"):  # finite
            absmax = max(absmax, m)
    drifted = [n.split("model.layers.")[-1][:30] for n, p in _frozen_probes
               if _checksum(p) != _frozen_base[n]]
    a_max, b_max = _adapter_mags()
    # Tripwire for bug (c): with alpha=rank the in-place s-scaling is a no-op and |B|
    # stays ~lr x steps. Anything near 1.0 means the aliased-buffer scaling is back
    # (e.g. someone raised alpha again) — abort instead of wasting a rotten run.
    if b_max > 1.0:
        raise RuntimeError(
            f"[GRAD-WATCH] |B|max={b_max:.3e} exploded at optimizer step {_gw_step['n']} "
            "— bug (c) aliased-buffer LoRA scaling regression; check lora_alpha == lora_rank "
            "and unsloth_zoo vllm_utils.load_lora_directly.")
    print(f"[GRAD-WATCH step {_gw_step['n']}] optimizer sees: {present} grads "
          f"({absent} None), nonzero={nonzero}, non-finite={nonfinite}, absmax={absmax:.3e} | "
          f"backward hooks (8x lora_B): calls={_bwd_stats['calls']}, "
          f"nonzero={_bwd_stats['nonzero']}, non-finite={_bwd_stats['nonfinite']} | "
          f"base-drift: {drifted if drifted else 'none'} | |A|max={a_max:.3e} |B|max={b_max:.3e}",
          flush=True)

from torch.optim.optimizer import register_optimizer_step_pre_hook
_gw_handle = register_optimizer_step_pre_hook(_grad_watch)
# --- end GRAD-WATCH ---

def check_lengths(dataset, tokenizer, max_prompt_length, max_seq_length):
    too_long = 0
    for sample in tqdm(dataset):
        prompt = tokenizer.apply_chat_template(sample["prompt"], tokenize=False)
        ids = tokenizer(prompt)["input_ids"]
        if len(ids) > max_prompt_length:
            too_long += 1
    print(f"Samples exceeding max_prompt_length ({max_prompt_length}): {too_long}")

check_lengths(dataset, tokenizer, max_prompt_length, max_seq_length)

# vLLM engine init (attention backend selector) permanently lowers
# torch._dynamo.config.recompile_limit to 16, clobbering unsloth's 1024.
# Worse: config writes land in a ContextVar, so other threads (autograd
# engine recomputing checkpointed forwards during backward) fall back to
# the DEFAULT of 8 -> FailOnRecompileLimitHit on unsloth's fullgraph=True
# compiled RMSNorm. Restore the main-thread override AND raise the default
# so every thread sees 1024.
import torch._dynamo
torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config._config["recompile_limit"].default = 1024

trainer.train()
# bug (d), 2026-07-24: save_pretrained_merged while the vLLM engine is still alive
# corrupted o_proj/down_proj (~1e13 values, aliased-memory garbage reads). Keep this
# call for convenience/inspection but DO NOT TRUST its output — always re-merge offline
# from the last checkpoint instead: python grpo_07_merge.py [checkpoint-NNN] (CPU-only,
# never touches the GPU/vLLM, so it cannot alias anything). Point it at
# outputs/lora-grpo-phi4-mini-v2-outputs/checkpoint-<highest>.
model.save_pretrained_merged(model_path, tokenizer, save_method="merged_16bit")
