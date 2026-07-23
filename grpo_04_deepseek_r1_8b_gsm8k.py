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

# DeepSeek-R1-Distill-Llama-8B — distilled from DeepSeek-R1 with RL-trained reasoning baked in.
# GRPO starts from a stronger baseline than raw Llama-3.1-8B.
# Uses <reasoning> tags for consistency with existing test infra (model adapts via GRPO rewards).
# Native R1 format uses <think>...</think> — swap tags below if you prefer it.
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_path = "outputs/lora-grpo-deepseek-r1-llama8b"
max_seq_length = 2048   # R1-distill reasoning chains are longer than vanilla Llama
max_prompt_length = 512
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
    device_map="auto",
    fast_inference=True,
    gpu_memory_utilization=0.5,  # 16 GB shared with desktop; 0.8 leaves too little for the training step
    max_lora_rank=lora_rank,
)

def ensure_bytelevel_tokenizer(tok):
    # transformers 5.x rebuilds LlamaTokenizerFast in sentencepiece style, ignoring the
    # ByteLevel decoder in tokenizer.json. On this byte-level BPE vocab that corrupts both
    # directions: encode drops spaces from prompts, decode leaks raw Ġ/Ċ markers into the
    # reward functions (format rewards go to 0). Reload verbatim from tokenizer.json when
    # the round-trip fails. This only heals the HF-side tokenizer — run
    # `grpo-fix-hf-tokenizer fix` to repair the cached config that vLLM loads on its own.
    probe = "a b,\nc d"
    out = tok.decode(tok(probe, add_special_tokens=False).input_ids, skip_special_tokens=True)
    if out.strip() == probe:
        return tok
    from transformers import PreTrainedTokenizerFast
    print(f"WARNING: broken byte-level tokenizer from {tok.name_or_path}; "
          "reloading via PreTrainedTokenizerFast — run 'grpo-fix-hf-tokenizer fix' to repair the HF cache")
    fixed = PreTrainedTokenizerFast.from_pretrained(tok.name_or_path)
    if fixed.pad_token is None:
        fixed.pad_token = tok.pad_token or fixed.eos_token
    return fixed

tokenizer = ensure_bytelevel_tokenizer(tokenizer)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
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
    print('-'*20, f"[{_progress()}]", f"\nQuestion:\n{prompts[0]}", f"\nResponse:\n{responses[0]}",
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
    use_vllm=True,
    learning_rate=3e-6,   # lower than Llama baseline — R1-distill already RL-trained
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_generations=4,
    # 16 GB: TRL scores the whole generation batch (batch x steps_per_generation)
    # in one logp forward -> OOM in matmul_lora (grpo_03 lesson, 2026-07-20).
    # Cap at num_generations and chunk the logp forward per sequence.
    generation_batch_size=4,
    unsloth_grpo_mini_batch=1,
    unsloth_logit_chunk_multiplier=16,
    # TRL 1.8 removed max_prompt_length from GRPOConfig; prompts are no longer
    # truncated by the config. vllm_max_model_length sets the vLLM context window
    # (>= max prompt length in the dataset + max_completion_length).
    max_completion_length=max_seq_length - max_prompt_length,
    vllm_max_model_length=max_seq_length,
    num_train_epochs=1,
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

# Validate token ids coming out of generation before they reach the compiled
# logprob gather. An out-of-vocab id there dies as an async device-side assert
# ("index out of bounds: 0 <= tmp0 < 128256") with no python traceback, so fail
# early here with the actual offending values instead.
_vocab_size = model.config.vocab_size

# Weight probe for the NaN causal-chain hypothesis (NaN logps -> NaN LoRA
# weights -> NaN vLLM logits -> garbage sampled ids). Scans the training model
# (4-bit base is uint8 and skipped; LoRA A/B and norms are floating and
# checked) plus the colocated vLLM weight copy before every generation round,
# i.e. right after the previous optimizer step. The vLLM copy still holds the
# adapter synced for the *previous* round at that point, which is fine: the log
# ordering vs NAN-WATCH shows which link breaks first. Attribute paths match
# unsloth_zoo.vllm_utils (vLLM V1 and V0 layouts).
def _find_vllm_runner():
    eng = getattr(model, "vllm_engine", None)
    if eng is None:
        return None
    for path in ("engine_core.engine_core.model_executor.driver_worker.model_runner",
                 "model_executor.driver_worker.model_runner"):
        obj = eng.llm_engine
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None

_vllm_runner = _find_vllm_runner()
_vllm_model = getattr(_vllm_runner, "model", None)

def _scan_weights(mod, tag):
    bad = [n for n, p in mod.named_parameters()
           if p.is_floating_point() and not torch.isfinite(p).all().item()]
    if bad:
        print(f"WEIGHT-WATCH: non-finite weights in {tag} at step "
              f"{trainer.state.global_step if trainer.state is not None else '?'}: "
              f"{bad[:5]}{' ...' if len(bad) > 5 else ''} ({len(bad)} tensors)",
              flush=True)
    return bool(bad)

print("WEIGHT-WATCH armed: train-side LoRA"
      + (" + vLLM weight copy" if _vllm_model is not None
         else " (vLLM internals NOT reachable — vLLM copy unchecked)"),
      flush=True)

# SAMPLER-WATCH: the crash is an async inductor-kernel assert (bound = vocab
# size) that surfaces at an unrelated bitsandbytes launch, so the failing site
# is invisible. WEIGHT-WATCH ruled out non-finite weights on both sides, which
# leaves runtime state: non-finite logits reaching the sampler (torch.
# multinomial on NaN probs can return garbage/out-of-range ids), or the sampler
# emitting an id >= vocab that the next decode step's embedding gather trips
# on. Wrap the vLLM V1 sampler to turn either case into a synchronous
# RuntimeError with the logits dumped for post-mortem.
# The gpu-worker Sampler (vllm/v1/worker/gpu/sample/sampler.py) is a plain
# class invoked via __call__(logits, input_batch), not an nn.Module with
# forward — special methods resolve on the type, so patch the class.
_sampler = getattr(_vllm_runner, "sampler", None)
if _sampler is not None:
    _SamplerCls = type(_sampler)
    _orig_sampler_call = _SamplerCls.__call__
    def _checked_sampler_call(self, logits, *a, **kw):
        # .item() would abort CUDA graph capture (warmup does a dummy
        # sampler run while capturing) — pass through untouched there.
        if torch.cuda.is_current_stream_capturing():
            return _orig_sampler_call(self, logits, *a, **kw)
        step = trainer.state.global_step if trainer.state is not None else "?"
        if torch.is_tensor(logits) and not torch.isfinite(logits).all().item():
            torch.save(logits, "/tmp/grpo04_bad_logits.pt")
            n = (~torch.isfinite(logits)).sum().item()
            raise RuntimeError(
                f"SAMPLER-WATCH: {n} non-finite logits entering vLLM sampler "
                f"at step {step}; dumped to /tmp/grpo04_bad_logits.pt")
        out = _orig_sampler_call(self, logits, *a, **kw)
        ids = getattr(out, "sampled_token_ids", None)
        if torch.is_tensor(ids) and ids.numel():
            mn, mx = ids.min().item(), ids.max().item()
            if mn < 0 or mx >= _vocab_size:
                torch.save({"logits": logits, "ids": ids}, "/tmp/grpo04_bad_logits.pt")
                raise RuntimeError(
                    f"SAMPLER-WATCH: sampler emitted out-of-vocab id "
                    f"min={mn} max={mx} shape={tuple(ids.shape)} "
                    f"(vocab={_vocab_size}) at step {step}; "
                    f"dumped to /tmp/grpo04_bad_logits.pt")
        return out
    _SamplerCls.__call__ = _checked_sampler_call
print(f"SAMPLER-WATCH {'armed: vLLM sampler wrapped' if _sampler is not None else 'NOT armed: sampler not reachable'}",
      flush=True)

# LOGP-WATCH: with weights and the vLLM sampler exculpated, the only gathers
# over the vocab dim (the assert bound == vocab_size) are in unsloth's compiled
# logp functions (unsloth_compiled_cache/UnslothGRPOTrainer.py). They run
# INSIDE _prepare_inputs (old/ref logps) and in the loss — before the
# _prepare_inputs validator sees anything, which is why it stayed silent.
# Wrap both module-level functions: validate the gather index (completion ids)
# eagerly — the .item() sync here also surfaces any assert pending from
# EARLIER kernels, so a traceback in the pre-check means the crash was born
# upstream — and synchronize after the call so an assert from THIS gather
# becomes a synchronous error naming the true site. Call sites read these
# names from module globals at call time, so patching the attributes covers
# every path (including where the function is passed as an argument).
import sys as _sys
_ugrpo_mod = next((m for n, m in _sys.modules.items()
                   if n.endswith("UnslothGRPOTrainer")
                   and hasattr(m, "chunked_selective_log_softmax")), None)

def _wrap_logp_fn(fn, fname, index_pos):
    def wrapped(*args, **kwargs):
        idx = kwargs.get("index", args[index_pos] if len(args) > index_pos else None)
        step = trainer.state.global_step if trainer.state is not None else "?"
        if torch.is_tensor(idx):
            mn, mx = idx.min().item(), idx.max().item()
            if mn < 0 or mx >= _vocab_size:
                torch.save(idx, "/tmp/grpo04_bad_index.pt")
                raise RuntimeError(
                    f"LOGP-WATCH: OOB gather index into {fname}: min={mn} max={mx} "
                    f"shape={tuple(idx.shape)} (vocab={_vocab_size}) at step {step}; "
                    f"dumped to /tmp/grpo04_bad_index.pt")
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        return out
    return wrapped

if _ugrpo_mod is not None:
    _ugrpo_mod.chunked_selective_log_softmax = _wrap_logp_fn(
        _ugrpo_mod.chunked_selective_log_softmax,
        "chunked_selective_log_softmax", 1)
    _ugrpo_mod.chunked_hidden_states_selective_log_softmax = _wrap_logp_fn(
        _ugrpo_mod.chunked_hidden_states_selective_log_softmax,
        "chunked_hidden_states_selective_log_softmax", 2)
print(f"LOGP-WATCH {'armed: compiled logp gathers wrapped' if _ugrpo_mod is not None else 'NOT armed: UnslothGRPOTrainer module not found'}",
      flush=True)

# EMBED-WATCH: all three earlier probes stayed silent yet the process still
# died with the bare assert and no traceback — because bitsandbytes'
# CUDA_CHECK calls exit(1) the moment it sees the sticky assert, before any
# python-side sync can raise. The only vocab-bound gather interleaved with
# bnb kernels is the input embedding of the 4-bit training forward: if the
# ids assembled by TRL (prompt + completion + padding) contain an
# out-of-range value, embed_tokens asserts and layer-0's bnb dequant kills
# the process microseconds later. Validate ids eagerly at the embedding
# boundary — the one place no previous probe could see. (The vLLM-side
# embedding runs inside replayed CUDA graphs where hooks don't execute, so
# it cannot be armed this way; train-side is the prime suspect anyway given
# the crash lands right after the reward print, i.e. at the logp forward.)
_embed = model.get_input_embeddings()
def _embed_pre_hook(mod, args):
    ids = args[0] if args else None
    if torch.is_tensor(ids) and not ids.is_floating_point() and ids.numel():
        mn, mx = ids.min().item(), ids.max().item()
        if mn < 0 or mx >= _vocab_size:
            step = trainer.state.global_step if trainer.state is not None else "?"
            torch.save(ids, "/tmp/grpo04_bad_embed_ids.pt")
            raise RuntimeError(
                f"EMBED-WATCH: OOB input_ids entering embedding: min={mn} "
                f"max={mx} shape={tuple(ids.shape)} (vocab={_vocab_size}) "
                f"at step {step}; dumped to /tmp/grpo04_bad_embed_ids.pt")
    return None
_embed.register_forward_pre_hook(_embed_pre_hook)
print("EMBED-WATCH armed: train-side input embedding hooked", flush=True)

# VLLM-IDS-WATCH: with the whole training side exculpated (weights, sampler
# I/O, logp gather index, embedding ids all clean), the assert by elimination
# fires inside vLLM's compiled forward — also bnb-quantized, so its CUDA_CHECK
# is what exit(1)s the process. Hooks don't run inside replayed CUDA graphs;
# the last python-visible point is prepare_inputs() in execute_model, which
# fills the persistent input_ids buffer (host copies + a Triton kernel that
# stores each round's last sampled token directly into the buffer — prime
# suspect for an off-by-one at a specific batch shape). Validate the FULL
# buffer right after prepare_inputs, before the forward launches: the graph
# runs on shape-padded token counts, so the stale padding region is read too
# and is covered by checking the whole buffer.
if _vllm_runner is not None and hasattr(_vllm_runner, "prepare_inputs"):
    _vllm_ids_buf = getattr(getattr(_vllm_runner, "input_buffers", None), "input_ids", None)
    _orig_vllm_prepare_inputs = _vllm_runner.prepare_inputs
    def _checked_vllm_prepare_inputs(*args, **kwargs):
        ib = _orig_vllm_prepare_inputs(*args, **kwargs)
        ids = _vllm_ids_buf if _vllm_ids_buf is not None else getattr(ib, "input_ids", None)
        if (torch.is_tensor(ids) and ids.numel()
                and not torch.cuda.is_current_stream_capturing()):
            mn, mx = ids.min().item(), ids.max().item()
            if mn < 0 or mx >= _vocab_size:
                step = trainer.state.global_step if trainer.state is not None else "?"
                torch.save(ids, "/tmp/grpo04_bad_vllm_ids.pt")
                raise RuntimeError(
                    f"VLLM-IDS-WATCH: OOB id in vLLM input buffer before "
                    f"forward: min={mn} max={mx} (vocab={_vocab_size}) at "
                    f"step {step}; dumped to /tmp/grpo04_bad_vllm_ids.pt")
        return ib
    _vllm_runner.prepare_inputs = _checked_vllm_prepare_inputs
    print(f"VLLM-IDS-WATCH armed: prepare_inputs wrapped "
          f"({'full buffer' if _vllm_ids_buf is not None else 'batch view only'})",
          flush=True)
else:
    print("VLLM-IDS-WATCH NOT armed: runner/prepare_inputs not reachable", flush=True)

_orig_prepare_inputs = trainer._prepare_inputs
def _checked_prepare_inputs(*args, **kwargs):
    _scan_weights(model, "train model")
    if _vllm_model is not None:
        _scan_weights(_vllm_model, "vLLM copy")
    out = _orig_prepare_inputs(*args, **kwargs)
    if isinstance(out, dict):
        for key, t in out.items():
            if not torch.is_tensor(t):
                continue
            if t.is_floating_point():
                # NaN logps precede NaN LoRA weights -> NaN vLLM logits -> garbage
                # sampled ids; warn early so the causal chain is visible in the log.
                if torch.isnan(t).any().item():
                    print(f"NAN-WATCH: NaN in '{key}' at step {trainer.state.global_step}")
            elif "ids" in key:
                mn, mx = t.min().item(), t.max().item()
                if mn < 0 or mx >= _vocab_size:
                    torch.save({k: v for k, v in out.items() if torch.is_tensor(v)},
                               "/tmp/grpo04_bad_batch.pt")
                    raise RuntimeError(
                        f"out-of-vocab token id in '{key}': min={mn} max={mx} "
                        f"(vocab={_vocab_size}) at step {trainer.state.global_step}; "
                        f"batch dumped to /tmp/grpo04_bad_batch.pt")
    return out
trainer._prepare_inputs = _checked_prepare_inputs

def check_lengths(dataset, tokenizer, max_prompt_length, max_seq_length):
    too_long_prompt = 0
    for sample in tqdm(dataset):
        prompt = tokenizer.apply_chat_template(sample["prompt"], tokenize=False)
        ids = tokenizer(prompt)["input_ids"]
        if len(ids) > max_prompt_length:
            too_long_prompt += 1
    print(f"Samples exceeding max_prompt_length ({max_prompt_length}): {too_long_prompt}")

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
model.save_pretrained_merged(model_path, tokenizer, save_method="merged_16bit")
