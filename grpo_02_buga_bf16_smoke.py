# Bug (a) discriminating probe — quantization-mismatch hypothesis (2026-07-26).
#
# Established: Qwen2.5-1.5B / Qwen3-4B / Qwen3-4B-Instruct-2507 / DeepSeek-R1-distill
# all show non-finite LoRA-B grads from step 1 (|B|max stays 0, KL -> 1e6) while
# gemma/llama/phi train clean on the identical stack. Ruled out: compiled kernels
# (UNSLOTH_COMPILE_DISABLE=1 probe), KL-term overflow (beta=0 probe: kl=0, grads
# still non-finite). Remaining suspect: trainer-vs-vLLM logp divergence — the two
# INDEPENDENTLY bnb-4bit-quantized copies (training model vs colocated engine)
# disagree; on sharp-logit checkpoints the per-token logp gap explodes
# (clip_ratio ~6% and loss ~15-20 at step 1 with B=0, where healthy models sit
# near ratio=1 / loss~0), and exp() of that gap poisons the PG term.
#
# This probe runs the SAME sick checkpoint (Qwen2.5-1.5B-Instruct, grpo_02's)
# with load_in_4bit=False: one bf16 numerical identity on both sides.
#   - finite grads, |B| growing  -> hypothesis CONFIRMED; Qwen fix = bf16 (small
#     models) or matched quantization; grpo_02 trainable again.
#   - still non-finite           -> quant mismatch refuted; suspect the shared
#     bf16 forward itself (logit magnitude overflow) next.
#
# GRPO_LOAD_4BIT=1 re-runs the 4-bit control (expected sick).
# GRPO_MAX_STEPS (default 2), GRPO_BETA (default 0.001) as in grpo_08.
# Probe only: no adapter save, no merge.

import os, warnings, logging
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("unsloth_zoo").setLevel(logging.CRITICAL)

import torch
torch.backends.cuda.enable_flash_sdp(True)
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
import re
from datasets import load_dataset, Dataset

os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
os.environ["VLLM_USE_V1"] = "0"
os.environ["FLASH_ATTENTION_USE_FA2"] = "1"
os.environ["XFORMERS_MEM_EFF_ATTN"] = "0"

LOAD_4BIT = os.environ.get("GRPO_LOAD_4BIT", "0") == "1"
print(f"[PROBE] load_in_4bit={LOAD_4BIT}", flush=True)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_path = "outputs/probe-buga-bf16"  # scratch; nothing durable is written
max_seq_length = 2048
max_prompt_length = 512
lora_rank = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=LOAD_4BIT,
    attn_implementation="flash_attention_2",
    device_map="auto",
    fast_inference=True,
    gpu_memory_utilization=0.5,
    max_lora_rank=lora_rank,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing=True,  # HF per-layer GC; "unsloth" offload GC produces NaN grads
    random_state=3407,
)

def extract_answer(text: str) -> str:
    tag_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if tag_match:
        content = tag_match.group(1).strip()
        norm = re.sub(r'[$€£\s]', '', content).replace(',', '')
        if re.fullmatch(r'-?\d+(\.\d+)?', norm):
            return norm
    if "####" in text:
        return text.split("####")[1].strip().replace(",", "")
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

def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [c[0]['content'] for c in completions]
    extracted = [extract_answer(r) for r in responses]
    return [2.0 if safe_float(r) == safe_float(a) else 0.0 for r, a in zip(extracted, answer)]

def has_answer_tag(completions, **kwargs) -> list[float]:
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.search(r"<answer>.*?</answer>", r, flags=re.DOTALL) else 0.0 for r in responses]

training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=4e-6,
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
    generation_batch_size=2,
    unsloth_grpo_mini_batch=1,
    unsloth_logit_chunk_multiplier=16,
    max_completion_length=max_seq_length - max_prompt_length,
    vllm_max_model_length=max_seq_length,
    num_train_epochs=1,
    beta=float(os.environ.get("GRPO_BETA", "0.001")),
    max_steps=int(os.environ.get("GRPO_MAX_STEPS", "2")),
    save_strategy="no",
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"{model_path}-outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[has_answer_tag, correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
)

# Same tripwire as grpo_08: prints |B|max / grad|max / non-finite count per step.
_lora_B = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
_tw_step = {"n": 0}

def _integrity_tripwire(optimizer, *args, **kwargs):
    _tw_step["n"] += 1
    b_max = max((p.data.abs().max().item() for _, p in _lora_B), default=0.0)
    nonfinite = sum(1 for _, p in _lora_B if p.grad is not None and not torch.isfinite(p.grad).all())
    gmax = max((p.grad.abs().max().item() for _, p in _lora_B
                if p.grad is not None and torch.isfinite(p.grad).all()), default=0.0)
    print(f"[TRIPWIRE step {_tw_step['n']}] |B|max={b_max:.3e} grad|max={gmax:.3e} "
          f"non-finite(B grads)={nonfinite}/{len(_lora_B)}", flush=True)

from torch.optim.optimizer import register_optimizer_step_pre_hook
register_optimizer_step_pre_hook(_integrity_tripwire)

import torch._dynamo
torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config._config["recompile_limit"].default = 1024

trainer.train()
print("[PROBE] done", flush=True)
