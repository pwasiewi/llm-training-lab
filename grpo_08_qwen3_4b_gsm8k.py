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

# Qwen3-4B-Instruct-2507 — the July-2025 refresh of Qwen3-4B: same dense Qwen3
# architecture (unfused q/k/v/o + gate/up/down, target_modules unchanged) but
# NON-thinking (no <think> blocks; prompt and rewards below are answer-tag only)
# and much stronger than the April base. Swapped in 2026-07-26: the original
# Qwen3-4B checkpoint is a bug (a) carrier (non-finite LoRA-B grads from step 1,
# KL -> 1e6 by step 4; the discriminator is WEIGHTS, not architecture — gemma/
# llama/phi train clean on the same stack), so a sibling checkpoint of the same
# arch is both the fix candidate and the cleanest test of that hypothesis.
# Qwen3.5 small (DeltaNet+MoE hybrid, multimodal) rejected for now: unsloth
# FastLanguageModel + vLLM LoRA hot-load do not cover that architecture.
# 14B/8B do not fit on 16 GB with colocated vLLM (grpo_05 lesson); rank 16 here.
model_name = "Qwen/Qwen3-4B-Instruct-2507"
model_path = "outputs/lora-grpo-qwen3-4b-2507-r16"
max_seq_length = 2048  # 4096 OOMs vLLM LoRA-manager init on 16 GB (grpo_05 lesson)
max_prompt_length = 512
lora_rank = 16   # rank-16 variant (grpo_05 uses rank 32 on the same base)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
    device_map="auto",
    fast_inference=True,
    gpu_memory_utilization=0.5,  # 16 GB shared with desktop; higher leaves too little for the training step
    max_lora_rank=lora_rank,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    # Qwen3 keeps attention/MLP projections UNFUSED at the HF level, so these
    # unfused names are correct (unlike Phi-4-mini/grpo_07, which fuses into
    # qkv_proj/gate_up_proj). Do NOT switch to fused names here.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    # alpha=rank (s=1) + doubled lr: the validated-equivalent workaround for
    # bug (c). The dealias pwr patch (2026-07-26, upstream PR unsloth-zoo#951)
    # makes alpha != rank legal again, but this pairing is kept to minimize
    # variables while the checkpoint swap is under test.
    lora_alpha=lora_rank,
    use_gradient_checkpointing=True,  # HF per-layer GC; "unsloth" offload GC produces NaN grads -> no-op training (probes 2026-07-24, see README)
    random_state=3407,
)

def extract_answer(text: str) -> str:
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
        norm = re.sub(r'[$\u20ac\u00a3\s]', '', content).replace(',', '')
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
                "You are a helpful math assistant. "
                "Think through the problem carefully, then give the final numerical answer in <answer>...</answer> tags."
            )},
            # Instruct-2507 is non-thinking: no /think flag, no <think> blocks.
            {'role': 'user', 'content': (
                f"Solve this problem step by step. "
                f"End your response with <answer>NUMBER</answer>.\n\nProblem:\n{x['question']}"
            )}
        ],
        'answer': extract_answer(x['answer'])
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
    extracted = [extract_answer(r) for r in responses]
    print('-'*20, f"[{_progress()}]", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted[0]}", f"\nAnswer:\n{answer[0]}", flush=True)
    return [2.0 if safe_float(r) == safe_float(a) else 0.0 for r, a in zip(extracted, answer)]

def is_integer_like(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [c[0]['content'] for c in completions]
    extracted = [extract_answer(r) for r in responses]
    return [0.5 if is_integer_like(r) else 0.0 for r in extracted]

def has_answer_tag(completions, **kwargs) -> list[float]:
    pattern = r"<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.search(pattern, r, flags=re.DOTALL) else 0.0 for r in responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    # Non-thinking format: free-form reasoning, response ENDS with the answer tag
    # (no trailing chatter after </answer>).
    pattern = r"^.*?<answer>\s*.*?\s*</answer>\s*$"
    responses = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=4e-6,   # 2e-6 doubled to offset LoRA scale s: 2 -> 1 (alpha=rank workaround, bug (c))
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=2,
    # 16 GB: TRL scores the whole generation batch (batch x steps_per_generation)
    # in one logp forward -> OOM in matmul_lora (grpo_03 lesson, 2026-07-20).
    # Cap at num_generations and chunk the logp forward per sequence.
    generation_batch_size=2,
    unsloth_grpo_mini_batch=1,
    unsloth_logit_chunk_multiplier=16,
    # TRL 1.8 removed max_prompt_length from GRPOConfig; prompts are no longer
    # truncated by the config. vllm_max_model_length sets the vLLM context window
    # (>= max prompt length in the dataset + max_completion_length).
    max_completion_length=max_seq_length - max_prompt_length,
    vllm_max_model_length=max_seq_length,
    num_train_epochs=1,
    # GRPO_BETA=0 drops the KL term (no ref adapter, DAPO-style). Probe for
    # bug (a): with |B|=0 policy == ref, yet kl logs ~1e3-1e6 from step 1 on
    # Qwen-family checkpoints — the k3 estimator exp(d)-d-1 then overflows and
    # is the suspected source of the non-finite LoRA-B grads.
    beta=float(os.environ.get("GRPO_BETA", "0.001")),
    # GRPO_MAX_STEPS=N caps the run for a smoke test; -1 = full epoch.
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
        has_answer_tag,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# --- Integrity tripwire (bug (c), grpo_07 validated 2026-07-24) ---
# With lora_alpha == lora_rank the in-place s=alpha/r scaling in unsloth_zoo
# load_lora_directly is a no-op, so |B| grows slowly (~lr x steps) and stays well
# below 1.0. If |B|max ever nears 1.0 the aliased-buffer scaling is back (someone
# raised alpha, or the pwr patch regressed) -> rollouts rot into gibberish within a
# few steps. Abort instead of burning an epoch. Also prints a first-steps health line
# so the run can be confirmed live (nonzero grads, no non-finite).
_lora_B = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
_tw_step = {"n": 0}

def _integrity_tripwire(optimizer, *args, **kwargs):
    _tw_step["n"] += 1
    b_max = max((p.data.abs().max().item() for _, p in _lora_B), default=0.0)
    if b_max > 1.0:
        raise RuntimeError(
            f"[TRIPWIRE] |B|max={b_max:.3e} at step {_tw_step['n']} — bug (c) aliased-buffer "
            "LoRA scaling regression; check lora_alpha == lora_rank and "
            "unsloth_zoo vllm_utils.load_lora_directly.")
    if _tw_step["n"] <= 10 or _tw_step["n"] % 25 == 0:
        nonfinite = sum(1 for _, p in _lora_B if p.grad is not None and not torch.isfinite(p.grad).all())
        gmax = max((p.grad.abs().max().item() for _, p in _lora_B
                    if p.grad is not None and torch.isfinite(p.grad).all()), default=0.0)
        print(f"[TRIPWIRE step {_tw_step['n']}] |B|max={b_max:.3e} grad|max={gmax:.3e} "
              f"non-finite(B grads)={nonfinite}/{len(_lora_B)}", flush=True)

from torch.optim.optimizer import register_optimizer_step_pre_hook
register_optimizer_step_pre_hook(_integrity_tripwire)
# --- end tripwire ---

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

# bug (d), 2026-07-24: unsloth's in-process save_pretrained_merged reads base
# weights through memory aliased with the live vLLM pool -> o_proj/down_proj
# garbage (~1e13). Save only the adapter (small LoRA tensors, safe to read),
# then merge in a clean CPU-only subprocess that loads base + adapter from
# disk and cannot alias the engine.
# Manual re-run: python grpo_merge.py outputs/lora-grpo-qwen3-4b-2507-r16 adapter-final --base Qwen/Qwen3-4B-Instruct-2507
final_adapter = os.path.join(f"{model_path}-outputs", "adapter-final")
model.save_pretrained(final_adapter)
tokenizer.save_pretrained(final_adapter)
import subprocess, sys
subprocess.run(
    [sys.executable,
     os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_merge.py"),
     model_path, "adapter-final", "--base", model_name],
    check=True,
)
