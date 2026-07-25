# Diagnostic probe for the grpo_07_1 no-vLLM crash (2026-07-24):
#   RuntimeError: prob_dist must be 1 or 2 dim  (torch.multinomial in transformers _sample)
# Reproduces the exact generation path of grpo_07_1 (PatchFastRL + get_peft_model +
# unsloth_base_fast_generate) with shape instrumentation on the model forward output
# and on torch.multinomial, to find where the extra logits dimension is born.
import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("unsloth_zoo").setLevel(logging.CRITICAL)

import torch
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-4-mini-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    attn_implementation="sdpa",
    device_map="auto",
    fast_inference=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing=True,
    random_state=3407,
)

# --- Instrumentation ---------------------------------------------------------
import functools, inspect
_calls = {"fwd": 0, "mult": 0, "gen": 0, "prep": 0}

base = model.base_model.model  # peft -> lora wrapper -> causal LM
print(f"[SIG] base={type(base).__name__} forward params:",
      list(inspect.signature(base.forward).parameters), flush=True)

_orig_forward = base.forward
@functools.wraps(_orig_forward)  # preserve signature: unsloth + transformers inspect it
def fwd_probe(*args, **kwargs):
    out = _orig_forward(*args, **kwargs)
    if _calls["fwd"] < 6:
        _calls["fwd"] += 1
        lg = getattr(out, "logits", None)
        print(f"[FWD {_calls['fwd']}] logits.shape={tuple(lg.shape) if lg is not None else None} "
              f"class={type(out).__name__} "
              f"logits_to_keep={kwargs.get('logits_to_keep', kwargs.get('num_logits_to_keep', 'absent'))}",
              flush=True)
    return out
base.forward = fwd_probe

# Point 1: kwargs entering unsloth_base_fast_generate (base.generate is the patched one)
_unsloth_gen = base.generate
def gen_probe(*args, **kwargs):
    _calls["gen"] += 1
    print(f"[GEN {_calls['gen']}] entry kwargs keys={sorted(kwargs)} "
          f"logits_to_keep={kwargs.get('logits_to_keep', 'absent')}", flush=True)
    return _unsloth_gen(*args, **kwargs)
base.generate = gen_probe

# Point 2: kwargs reaching prepare_inputs_for_generation (mirrors model_kwargs in generate)
_orig_prep = base.prepare_inputs_for_generation
@functools.wraps(_orig_prep)
def prep_probe(*args, **kwargs):
    if _calls["prep"] < 3:
        _calls["prep"] += 1
        print(f"[PREP {_calls['prep']}] kwargs keys={sorted(kwargs)} "
              f"logits_to_keep={kwargs.get('logits_to_keep', 'absent')}", flush=True)
    out = _orig_prep(*args, **kwargs)
    if _calls["prep"] <= 3:
        print(f"[PREP {_calls['prep']}] model_inputs logits_to_keep="
              f"{out.get('logits_to_keep', 'absent')}", flush=True)
    return out
base.prepare_inputs_for_generation = prep_probe

# Point 3: what unsloth's arch-detection cache holds after the call
import unsloth.models.vision as _uv

_orig_mult = torch.multinomial
def mult_probe(probs, *a, **k):
    if _calls["mult"] < 6:
        _calls["mult"] += 1
        print(f"[MULTINOMIAL {_calls['mult']}] probs.shape={tuple(probs.shape)} dtype={probs.dtype}",
              flush=True)
    if probs.dim() > 2:
        print(f"[MULTINOMIAL] >2-dim probs! full shape={tuple(probs.shape)} — squeezing to continue probe",
              flush=True)
        probs = probs.reshape(-1, probs.shape[-1])
    return _orig_mult(probs, *a, **k)
torch.multinomial = mult_probe
# -----------------------------------------------------------------------------

msgs = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
# Two identical prompts = generation_batch_size=2, mirroring the GRPO call;
# do_sample=True + temperature mirrors TRL's generation_config.
from transformers import GenerationConfig
gc = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=None,
                      max_new_tokens=8, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
batch = ids.repeat(2, 1)
print("[PROBE] calling model.generate, input shape:", tuple(batch.shape), flush=True)
try:
    out = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch), generation_config=gc)
    print("[PROBE] generate returned:", tuple(out.shape), flush=True)
    print(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True), flush=True)
finally:
    print("[NUM_LOGITS_TO_KEEP cache]:", _uv.NUM_LOGITS_TO_KEEP, flush=True)
    print("[SUPPORTS] _supports_logits_to_keep:", base._supports_logits_to_keep(), flush=True)
