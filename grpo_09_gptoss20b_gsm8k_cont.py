# GRPO continuation for gpt-oss-20b: resumes from a saved LoRA checkpoint
# of grpo_09_gptoss20b_gsm8k.py (adjust checkpoint_path below).
#
# Key differences vs the qwen/llama scripts in this suite (16 GB VRAM budget):
# - fast_inference=False: colocated vLLM cannot fit a 20B next to the training
#   step on 16 GB (bnb-4bit weights alone are ~12 GB). Unsloth's own generate
#   path is the one their "gpt-oss-20b GRPO in ~15 GB" claim uses.
# - LoRA on attention only: the MoE expert tensors are quantized (MXFP4/bnb)
#   and touching experts/router degrades routing (see project_gptoss_mxfp4_distill);
#   vLLM/unsloth LoRA on fused-MoE weights is also the historically broken path.
# - gpt_oss is on unsloth's FORCE_FLOAT32 list (no fp16); bf16 autocast is fine
#   on this GPU (Blackwell).
# - First run downloads ~13 GB of base weights into the HF cache.
# - Stop aillama (`aillama stop`) and check `nvidia-smi` before starting:
#   the desktop already holds ~1.6-2 GiB of the 16 GB.

# --- Silence cosmetic third-party startup noise (must run before heavy imports) ---
import os, warnings, logging
os.environ.setdefault("GLOG_minloglevel", "2")          # caffe2/glog: hide INFO+WARNING
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut VRAM fragmentation on the 16 GB card
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# The "[ERROR] ... not documented" lines are unsloth-zoo docstring checks, not real errors:
logging.getLogger("unsloth_zoo").setLevel(logging.CRITICAL)
# gpt-oss MoE: the default "grouped" path dequantizes ALL 32 experts into
# torch._grouped_mm stacks (~1.6 GB transient PER MLP layer) - the actual cause
# of the 16 GB OOMs regardless of sequence length. Force the per-expert eager
# loop instead (~33 MB transients, slower forward).
os.environ.setdefault("UNSLOTH_GPTOSS_GROUPED", "0")  # headless: override with UNSLOTH_GPTOSS_GROUPED=1 (fast grouped path)
# ----------------------------------------------------------------------------------

from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
import re
from datasets import load_dataset, Dataset

# Load model & tokenizer
model_name = "unsloth/gpt-oss-20b"  # load_in_4bit=True routes to the -unsloth-bnb-4bit repo
model_path = "outputs/lora-grpo-gptoss20b"
max_seq_length = 768    # OOM at 1536/1024/896 (2026-07-19): logprob buffers scale with completion
                        # tokens, and the desktop's VRAM share fluctuates 0.7-1.1 GiB on top
max_prompt_length = 384  # GSM8K prompt + harmony template overhead is ~250 tokens
lora_rank = 8           # attention-only LoRA needs far less rank than the full-mlp 64 used on 1.5B

checkpoint_path = f"{model_path}-outputs/checkpoint-300"  # Path to your last checkpoint

# Load the model and tokenizer from the checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,     # no colocated vLLM: 20B does not fit next to training on 16 GB
    offload_embedding=True,   # moves embed_tokens/lm_head off-GPU; part of the ~15 GB recipe
    adapter_name=checkpoint_path,  # This loads the LoRA weights from the checkpoint
)

# Apply LoRA — attention projections only (see header for why experts stay frozen)
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# Function to extract final answer
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
            # "Keep your reasoning brief": harmony's analysis channel must fit the tight
            # 384-token completion budget, or truncation drops the final answer entirely
            {'role': 'system', 'content': "You are a helpful assistant that always responds with reasoning and answer tags. Keep your reasoning brief."},
            {'role': 'user', 'content': f"Please solve the following problem and respond in this format:\n<reasoning>...</reasoning>\n<answer>...</answer>\n\nProblem:\n{x['question']}"}
        ],
        'answer': extract_hash_answer(x['answer'])
    })

dataset = get_gsm8k_questions()

# Reward functions
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
    print('-'*20, f"[{_progress()}]", flush=True)
    responses = [c[0]['content'] for c in completions]
    extracted_responses = [extract_hash_answer(r) for r in responses]
    return [2.0 if safe_float(r) == safe_float(a) else 0.0 for r, a in zip(extracted_responses, answer)]

def is_integer_like(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_hash_answer(r) for r in responses]
    return [0.5 if is_integer_like(r) else 0.0 for r in extracted_responses]

# No strict (^...$-anchored) format reward here: gpt-oss emits its Harmony
# analysis channel before the final message, so the decoded completion never
# STARTS with <reasoning>. The soft reward still shapes the tags in the final
# channel; xmlcount rewards partial tag emission.
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text: str) -> float:
    count = 0.0
    if "<reasoning>" in text:
        count += 0.125
    if "</reasoning>" in text:
        count += 0.125
    if "<answer>" in text:
        count += 0.125
    if "</answer>" in text:
        count += 0.125
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# Training arguments
training_args = GRPOConfig(
    use_vllm=False,  # matches fast_inference=False above
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=False,  # gpt_oss is FORCE_FLOAT32: fp16 produces NaN grad norms
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=2,
    max_completion_length=max_seq_length - max_prompt_length,  # analysis channel + final must fit in 512
    # Logprob computation materializes (tokens/chunks, 201088-vocab) logit buffers,
    # where chunks = rows_per_forward x multiplier. The autotuner (80% of free VRAM
    # at first step) picked multiplier=4 and OOMed ~140 MiB short on 16 GB; pin the
    # sizing explicitly instead of letting it guess.
    unsloth_grpo_mini_batch=1,
    unsloth_logit_chunk_multiplier=16,
    # Generate 2 sequences per call instead of the default batch x steps_per_generation
    # (=8): the eval-mode MoE path computes ALL experts over ALL prefill tokens, so
    # generation transients scale with this. Same total work, 4x smaller peaks.
    generation_batch_size=2,
    # Another bounded slice on top of the checkpoint loaded above.
    max_steps=300,
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
        xmlcount_reward_func,
        soft_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# Raise the dynamo recompile limit (default 8) and its ContextVar DEFAULT so
# autograd-engine threads see it too. Needed even without vLLM: the per-expert
# MoE loop recompiles unsloth's fullgraph=True kernels for varying per-expert
# token counts and blows past 8 within the first steps (FailOnRecompileLimitHit).
import torch._dynamo
torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config._config["recompile_limit"].default = 1024

trainer.train()

# Save the LoRA adapter only. merged_16bit of a 20B is ~40 GB and does not fit
# the current free disk; for serving, export to GGUF (experts stay MXFP4, ~14 GB)
# and register it in ~/.aillama/models.conf instead.
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
