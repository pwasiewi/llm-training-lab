# --- Silence cosmetic third-party startup noise (must run before heavy imports) ---
import os, warnings, logging, importlib.util
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

# ============================================================================
# grpo_07_1 — flash_attn no-vLLM CONTROL, derived from grpo_07 (2026-07-24)
# ============================================================================
# Purpose: a clean training run that DELIBERATELY drops the colocated vLLM
# rollout engine and generates rollouts through HF `model.generate()` with
# `flash_attention_2`. This isolates bug (c)/(d) — the aliased-buffer LoRA
# scaling in unsloth_zoo `load_lora_directly` that only exists on the vLLM
# hot-load path. If this run stays healthy (|B|max grows off 0, KL bounded,
# rollouts stay coherent) while the vLLM run (grpo_07) rots within ~5 steps,
# the defect is confined to the vLLM colocation, not the optimizer/loss.
#
# Why a SEPARATE file (not just GRPO_NO_VLLM=1 on grpo_07): grpo_07's no-vLLM
# branch falls back to `sdpa` because the flash_attn package was absent. That
# changes the training-side attention kernel too, so a difference between the
# runs could be "vLLM" OR "sdpa vs fa2". This file pins flash_attention_2 on
# BOTH the (now-added) HF path, leaving vLLM colocation as the ONLY variable.
#
# PREREQUISITE — the flash_attn package:
#   * Not shipped by default on this host (flex/sdpa stack). Build the pwr
#     ebuild dev-python/flash-attn (template: /var/db/repos/stuff/dev-python/
#     flash-attn/flash-attn-2.8.3_p1.ebuild) with Blackwell + low-RAM flags:
#       CUDAARCHS="120"  (no dot; sm_120 SASS-only)
#       MAKEOPTS="-j4"   (cicc OOMs ~7 GB each; use /etc/portage/env/lowmem-j4)
#   * Until then this script auto-falls back to sdpa and prints a loud notice
#     (the run is still a valid vLLM-vs-noVLLM control, just not fa2-matched).
#
# KNOWN SECOND BLOCKER (not fixed by flash_attn): unsloth's HF generation path
# `unsloth_base_fast_generate` has a `multinomial` bug that can surface once
# vLLM is off. If generation crashes there, that is the next patch to land —
# see the unsloth-grpo-patching skill. Watch the first rollout closely.
# ============================================================================

import torch
torch.backends.cuda.enable_flash_sdp(True)
# GRPO_ANOMALY=1: autograd anomaly mode — pinpoints the backward op that creates the
# NaN gradients found by GRAD-WATCH. ~3x slower; combine with GRPO_MAX_STEPS=1.
if os.environ.get("GRPO_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly(True)
    print("GRPO_ANOMALY=1: autograd anomaly detection ON", flush=True)
# fa2 rollouts REQUIRE bypassing unsloth's generate wrapper (must be set BEFORE the
# unsloth import — the wrapper is installed at patch time). unsloth_base_fast_generate
# corrupts fa2 generation two ways: it forces cache_implementation="static" (fa2 cannot
# mask the preallocated-but-empty slots -> attends over garbage), and even with
# UNSLOTH_DISABLE_STATIC_GENERATION=1 a second wrapper defect still garbles output.
# Plain HF generate with fa2 is verified clean (2026-07-25). sdpa keeps the wrapper
# (its paged path is the validated-healthy baseline).
if os.environ.get("GRPO_ATTN_IMPL") == "flash_attention_2":
    os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm

os.environ["FLASH_ATTENTION_USE_FA2"] = "1"
os.environ["XFORMERS_MEM_EFF_ATTN"] = "0"

# flash_attn gate — DEFAULT IS NOW sdpa (2026-07-25). With flash_attn installed,
# unsloth's hybrid HF-generate path corrupts rollouts for compiled archs (Phi3 via
# FastModel): prefill is split across unsloth's paged path and the transformers fa2
# interface, and during decode the KV length never grows (each token overwrites the
# same cache slot) -> token-loop gibberish from step 0 with LoRA B still zero.
# Isolated 2026-07-25: fa2 kernel itself is numerically clean (synthetic GQA/causal
# probes pass; pure-transformers fa2 generation clean); sdpa/eager on the SAME
# unsloth-loaded model clean; only unsloth generate + fa2 interface breaks (kv
# frozen at prompt length). Upstream unsloth bug, not the sm_120 build.
# Opt back in with GRPO_ATTN_IMPL=flash_attention_2 once upstream is fixed.
_have_flash_attn = importlib.util.find_spec("flash_attn") is not None
attn_impl = os.environ.get("GRPO_ATTN_IMPL", "sdpa")
if attn_impl == "flash_attention_2":
    if not _have_flash_attn:
        raise SystemExit("[grpo_07_1] GRPO_ATTN_IMPL=flash_attention_2 but flash_attn is not installed.")
    assert os.environ.get("UNSLOTH_DISABLE_FAST_GENERATION") == "1", \
        "fa2 requires the pre-import UNSLOTH_DISABLE_FAST_GENERATION=1 gate (see top of file)"
    print("[grpo_07_1] fa2 rollouts: unsloth generate wrapper BYPASSED "
          "(UNSLOTH_DISABLE_FAST_GENERATION=1) — plain HF generate, verified clean 2026-07-25.", flush=True)
else:
    print(f"[grpo_07_1] using attn_implementation='{attn_impl}' "
          f"(flash_attn installed: {_have_flash_attn}; fa2 opt-in via GRPO_ATTN_IMPL).", flush=True)

# Phi-4-mini (3.8B) in 4-bit: fits on 16 GB comfortably without a colocated vLLM
# copy (no rollout engine here, so all VRAM goes to the training step).
model_name = "microsoft/Phi-4-mini-instruct"
model_path = "outputs/lora-grpo-phi4-mini-novllm"
max_seq_length = 2048
max_prompt_length = 512
lora_rank = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    attn_implementation=attn_impl,
    device_map="auto",
    # No colocated rollout engine in this control: HF generate() does the rollouts.
    fast_inference=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    # Phi-4-mini fuses attention into qkv_proj and the MLP into gate_up_proj.
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    # alpha == rank (s=1). Kept identical to grpo_07 so the control differs from
    # the vLLM run ONLY in the rollout engine. NB: bug (c)'s in-place s-scaling
    # lives on the vLLM hot-load path, so it cannot fire here regardless of s —
    # but we keep s=1 to hold every other variable constant.
    lora_alpha=lora_rank,
    # GRPO_GC: hf (default, clean grads) | unsloth (proven NaN no-op) | off (VRAM-heavy).
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
    use_vllm=False,       # control: rollouts via HF generate(), no colocated vLLM engine
    learning_rate=4e-6,   # kept == grpo_07 (alpha=rank workaround doubled lr from 2e-6)
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
    # Keep the logp forward chunked per sequence (grpo_03 OOM lesson): the logp
    # forward still scores the whole generation batch even without vLLM.
    generation_batch_size=2,
    unsloth_grpo_mini_batch=1,
    unsloth_logit_chunk_multiplier=16,
    # TRL 1.8 removed max_prompt_length from GRPOConfig. No vLLM here, so
    # vllm_max_model_length is irrelevant; only the completion budget matters.
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=1,
    # GRPO_MAX_STEPS=N caps the run for diagnostics (smoke); -1 = full epoch.
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

# --- GRAD-WATCH (2026-07-24): per-tensor backward hooks on 8 lora_B params + an
# optimizer step pre-hook. Reports, per step: grads seen / None / non-finite /
# nonzero, frozen-base checksum drift, and |A|max / |B|max. Read the smoke log:
#   |B|max growing off 0, non-finite=0, nonzero>0  -> training (healthy control);
#   |B|max=0.0, non-finite=<all> every step        -> real no-op;
#   base-drift non-empty                            -> a write into frozen weights.
_watch_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
_bwd_stats = {"calls": 0, "nonzero": 0, "nonfinite": 0}

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
    # Tripwire: with no vLLM the aliased in-place s-scaling cannot fire, so |B|
    # must stay ~lr x steps. A blow-up here would mean the defect is NOT confined
    # to vLLM after all — a stronger finding than the vLLM run's tripwire.
    if b_max > 1.0:
        raise RuntimeError(
            f"[GRAD-WATCH] |B|max={b_max:.3e} exploded at optimizer step {_gw_step['n']} "
            "WITHOUT vLLM — bug (c) is NOT confined to the vLLM hot-load path; "
            "re-open the optimizer/loss hypothesis.")
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

# The recompile-limit override is needed even without vLLM: unsloth's fullgraph
# compiled forwards recompile across varying shapes, and the autograd engine
# recomputing checkpointed forwards during backward reads the DEFAULT (8) from a
# ContextVar. Raise both the live value and the cross-thread default.
import torch._dynamo
torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config._config["recompile_limit"].default = 1024

trainer.train()
model.save_pretrained_merged(model_path, tokenizer, save_method="merged_16bit")
