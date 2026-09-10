#!/usr/bin/env python3
# grpo_11: GRPO on MATH (Level 3-5) with an ANSWER-ONLY reward — no format reward.
#
# Why a new family (see GRPO_DENSE_PLAN.md, 2026-09-08): grpo_01-10 spend most of
# the gradient on XML tags (format terms sum to 3.0 vs 2.0 for correctness) and
# GSM8K is saturated for the healthy models (~86% base), so groups carry no
# signal. Here the only mandatory reward is `correct` (math_verify on the boxed
# answer; an unparseable completion scores 0 by construction, which is the
# whole "format" signal the model needs). Everything else is an EXPERIMENT
# switched on by environment variables so that one run differs from the
# previous one in exactly one knob:
#
#   GRPO_TAG=e0            run name -> outputs/lora-grpo-qwen3-17b-math-<tag>
#   GRPO_G=4               num_generations (== generation_batch_size; 16 GB rule)
#   GRPO_LOSS=bnpo         grpo | dr_grpo | dapo | bnpo | cispo | sapo. NOT dapo by default: unsloth's
#                          patched GRPOConfig FORCES mask_truncated_completions=True for dapo ("we will
#                          set it"), and its dapo normaliser (num_items_in_batch) has no clamp, so a
#                          group whose G completions are all truncated = 0 valid tokens = 0/0 -> NaN
#                          loss + non-finite LoRA grads (NAN-WATCH, smoke7 2026-09-08). bnpo = the same
#                          token-level loss normalised per micro-batch WITH clamp(min=1), no forced mask.
#   GRPO_BETA=0            KL coefficient (0 = no ref forward, no zero-std KL leak)
#   GRPO_SCALE=batch       scale_rewards: group | batch | none
#   GRPO_LR=4e-6           learning rate (proven healthy at alpha == rank)
#   GRPO_ITERS=1           num_iterations (PPO epochs per rollout; 2 with clip-higher)
#   GRPO_EPS_HIGH=0.28     epsilon_high (DAPO clip-higher); "" = symmetric clip
#   GRPO_MAXLEN=2048       max_completion_length (Qwen3-1.7B no-think averages ~700-900 tokens
#                          on MATH L3-5; at 1024 25-50% of completions were truncated)
#   GRPO_MASK_TRUNC=0      1 = mask_truncated_completions. Off = truncated completions simply score 0
#                          (unparseable), which is also a length signal. Ignored for loss=dapo (forced on).
#   GRPO_LEVELS=3,4,5      MATH levels used for training
#   GRPO_PASS_FILTER=      path to a grpo_11_prescore.py jsonl; keeps only prompts the base model
#                          solves SOMETIMES (0 < pass < n). Measured need: at G=4, 2.7 of the 4
#                          prompts per step give zero gradient (frac_reward_zero_std 0.67).
#   GRPO_STEP=0            1 = add partial-credit reward: fraction of the gold
#                          solution's intermediate numbers reproduced (weight GRPO_W_STEP)
#   GRPO_W_STEP=0.5        weight of the step reward relative to correct=1.0
#   GRPO_OVERLONG=0        1 = add DAPO soft overlong penalty (weight 1.0)
#   GRPO_AGG=normalize_then_sum   multi-reward aggregation (GDPO) | sum_then_normalize
#   GRPO_VLLM_IS=1         vllm_importance_sampling_correction (TRL default on)
#   GRPO_CHUNK=16          unsloth_logit_chunk_multiplier
#   GRPO_ANOMALY=0         1 = torch.autograd.set_detect_anomaly (USELESS here: collides with the
#                          unsloth torch.compile path, GuardOnDataDependentSymNode at step 1)
#   GRPO_LORA_ROLLOUT=1    1 = hand the live LoRA adapter to the vLLM rollout engine (BUG FIX, see below).
#                          0 = reproduce the broken behaviour (rollouts from the frozen base model).
#   GRPO_NANWATCH=0        1 = wrap the compiled grpo_accumulated_loss: on a non-finite loss dump the
#                          batch to /tmp/grpo11_nan_<n>.pt and re-run a clean no-grad forward to
#                          locate the first non-finite per-token logp (row, position, token)
#   GRPO_MAX_STEPS=-1      cap for smoke tests
#   GRPO_PRINT_EVERY=20    print one sample completion every N reward calls
#   GRPO_DEVEVAL=0         every N steps, greedy-score a fixed MATH-500 slice with the CURRENT
#                          adapter on the training engine -> a learning CURVE instead of one
#                          endpoint. 200 items costs ~2-4 min. A 500-item endpoint eval has a
#                          +-3 pp binomial half-width, too blunt for a 200-step run.
#   GRPO_DEVN=200          dev slice size
#
# Inherits every runtime fix of grpo_10 (Qwen3-1.7B bf16 both sides = bug (a) fix,
# alpha == rank + HF gradient checkpointing, generation_batch_size == G, chunked
# logp path, recompile-limit restore, adapter-only save + offline merge).
#
# Smoke:  GRPO_MAX_STEPS=3 python grpo_11_qwen3_17b_math.py
# Eval:   python grpo_11_qwen3_17b_math_test.py outputs/lora-grpo-qwen3-17b-math-e0

# --- Silence cosmetic third-party startup noise (must run before heavy imports) ---
import os, warnings, logging
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CC", "/usr/x86_64-pc-linux-gnu/gcc-bin/15/gcc")   # flashinfer JIT: nvcc rejects gcc-16
os.environ.setdefault("CXX", "/usr/x86_64-pc-linux-gnu/gcc-bin/15/g++")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("unsloth_zoo").setLevel(logging.CRITICAL)
# ----------------------------------------------------------------------------------

import re
import torch
torch.backends.cuda.enable_flash_sdp(True)
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from tqdm import tqdm
from math_verify import parse, verify, LatexExtractionConfig

os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
os.environ["VLLM_USE_V1"] = "0"
os.environ["FLASH_ATTENTION_USE_FA2"] = "1"
os.environ["XFORMERS_MEM_EFF_ATTN"] = "0"

# ---------------- experiment knobs ----------------
def _env(name, default, cast=str):
    v = os.environ.get(name, "")
    return cast(v) if v != "" else default

TAG          = _env("GRPO_TAG", "e0")
G            = _env("GRPO_G", 4, int)
LOSS         = _env("GRPO_LOSS", "bnpo")
BETA         = _env("GRPO_BETA", 0.0, float)
SCALE        = _env("GRPO_SCALE", "batch")
LR           = _env("GRPO_LR", 4e-6, float)
ITERS        = _env("GRPO_ITERS", 1, int)
_eh          = os.environ.get("GRPO_EPS_HIGH", "0.28")
EPS_HIGH     = float(_eh) if _eh else None
MAXLEN       = _env("GRPO_MAXLEN", 2048, int)
MASK_TRUNC   = _env("GRPO_MASK_TRUNC", 0, int) == 1
LEVELS       = {f"Level {l.strip()}" for l in _env("GRPO_LEVELS", "3,4,5").split(",")}
PASS_FILTER  = _env("GRPO_PASS_FILTER", "")
USE_STEP     = _env("GRPO_STEP", 0, int) == 1
W_STEP       = _env("GRPO_W_STEP", 0.5, float)
USE_OVERLONG = _env("GRPO_OVERLONG", 0, int) == 1
AGG          = _env("GRPO_AGG", "normalize_then_sum")
VLLM_IS      = _env("GRPO_VLLM_IS", 1, int) == 1
CHUNK        = _env("GRPO_CHUNK", 16, int)
ANOMALY      = _env("GRPO_ANOMALY", 0, int) == 1
NANWATCH     = _env("GRPO_NANWATCH", 0, int) == 1
LORA_ROLLOUT = _env("GRPO_LORA_ROLLOUT", 1, int) == 1
MAX_STEPS    = _env("GRPO_MAX_STEPS", -1, int)
PRINT_EVERY  = _env("GRPO_PRINT_EVERY", 20, int)
DEVEVAL      = _env("GRPO_DEVEVAL", 0, int)
DEVN         = _env("GRPO_DEVN", 200, int)

model_name = "Qwen/Qwen3-1.7B"
model_path = f"outputs/lora-grpo-qwen3-17b-math-{TAG}"
max_prompt_length = 512
max_seq_length = max_prompt_length + MAXLEN
lora_rank = 16

print(f"[grpo_11] tag={TAG} G={G} loss={LOSS} beta={BETA} scale={SCALE} lr={LR} iters={ITERS} "
      f"eps_high={EPS_HIGH} maxlen={MAXLEN} levels={sorted(LEVELS)} passfilter={PASS_FILTER or "-"} step={USE_STEP}(w={W_STEP}) "
      f"overlong={USE_OVERLONG} agg={AGG} mask_trunc={MASK_TRUNC} vllm_is={VLLM_IS} chunk={CHUNK} anomaly={ANOMALY} lora_rollout={LORA_ROLLOUT} max_steps={MAX_STEPS}", flush=True)

# ---------------- model ----------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # bf16 both sides — bug (a) fix (grpo_10 header)
    attn_implementation="flash_attention_2",
    device_map="auto",
    fast_inference=True,
    gpu_memory_utilization=0.5,  # 16 GB shared with the desktop
    max_lora_rank=lora_rank,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,             # s=1: bug (c) era invariant, kept deliberately
    use_gradient_checkpointing=True,  # HF per-layer GC; "unsloth" offload GC -> NaN grads
    random_state=3407,
)

# ---------------- data ----------------
_GOLD_CFG = [LatexExtractionConfig(boxed_match_priority=0)]

def last_boxed(s: str) -> str | None:
    """Return the content of the last \\boxed{...} in s (brace-balanced), or None."""
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def gold_numbers(solution: str, final: str) -> list[str]:
    """Distinct numeric literals in the gold solution, minus the ones in the final answer.
    These are the 'intermediate results' used by the optional step reward."""
    final_nums = set(_NUM_RE.findall(final or ""))
    seen, out = set(), []
    for n in _NUM_RE.findall(solution):
        if n in final_nums or n in seen or n in ("0", "1", "2"):  # trivially common digits carry no credit
            continue
        seen.add(n)
        out.append(n)
    return out

SYSTEM_PROMPT = "You are a careful math solver."
USER_TEMPLATE = ("/no_think\n\nSolve the problem step by step. "
                 "Put the final answer in \\boxed{}.\n\nProblem:\n{problem}")

def build_dataset() -> Dataset:
    data = load_dataset("DigitalLearningGmbH/MATH-lighteval")["train"]
    data = data.filter(lambda x: x["level"] in LEVELS)
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.replace("{problem}", x["problem"])},
        ],
        "answer": last_boxed(x["solution"]),
        "gold_nums": gold_numbers(x["solution"], last_boxed(x["solution"])),
    })
    data = data.filter(lambda x: x["answer"] is not None)
    # Drop prompts longer than max_prompt_length (59 of 5586 at 512 tokens, 2026-09-08).
    def _fits(x):
        s = tokenizer.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=True)
        return len(tokenizer(s)["input_ids"]) <= max_prompt_length
    data = data.filter(_fits)
    data = data.shuffle(seed=3407)
    if PASS_FILTER:
        import json as _json
        keep = {}
        with open(PASS_FILTER) as fh:
            for line in fh:
                r = _json.loads(line)
                keep[r["problem"]] = (r["pass"], r["n"])
        before = len(data)
        # Drop prompts the base model always solves or never solves: their groups have
        # zero reward variance, so they contribute no gradient at any group size.
        data = data.filter(lambda x: (pk := keep.get(x["problem"])) is not None
                           and 0 < pk[0] < pk[1])
        print(f"[grpo_11] pass-filter {PASS_FILTER}: {before} -> {len(data)} prompts", flush=True)
        if len(data) == 0:
            raise RuntimeError("pass filter left no prompts — wrong file or wrong levels?")
    return data

dataset = build_dataset()
print(f"[grpo_11] train prompts: {len(dataset)}", flush=True)

# ---------------- rewards ----------------
_calls = {"n": 0, "hedged": 0, "unparsed": 0}

def _progress():
    _calls["n"] += 1
    st = trainer.state if "trainer" in globals() else None
    s = f"call {_calls['n']}"
    if st is not None and st.epoch is not None:
        s += f" | step {st.global_step}/{st.max_steps} | epoch {st.epoch:.3f}"
    return s

def is_correct(text: str, gold_boxed: str) -> tuple[float, str]:
    """1.0 if math_verify says the completion's answer equals the gold boxed answer.
    Hedging gate: two DIFFERENT boxed answers -> 0 (math_verify would silently take the last).
    Returns (reward, status) with status in {ok, wrong, hedged, unparsed}."""
    boxed = re.findall(r"\\boxed\s*{", text)
    if len(boxed) > 1:
        contents = set()
        s = text
        while True:
            b = last_boxed(s)
            if b is None:
                break
            contents.add(b.replace(" ", ""))
            s = s[:s.rfind("\\boxed")]
        if len(contents) > 1:
            return 0.0, "hedged"
    try:
        pred = parse(text)
    except Exception:
        pred = []
    if not pred:
        return 0.0, "unparsed"
    try:
        gold = parse(f"$\\boxed{{{gold_boxed}}}$", extraction_config=_GOLD_CFG)
        ok = verify(gold, pred)
    except Exception:
        ok = False
    return (1.0, "ok") if ok else (0.0, "wrong")

def correct_reward(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    out, status = [], []
    for r, a in zip(responses, answer):
        v, st = is_correct(r, a)
        out.append(v)
        status.append(st)
    _calls["hedged"] += status.count("hedged")
    _calls["unparsed"] += status.count("unparsed")
    tag = _progress()
    if _calls["n"] % PRINT_EVERY == 1:
        print("-" * 20, f"[{tag}]\nResponse:\n{responses[0][-600:]}\nGold: {answer[0]}  status={status[0]}\n"
              f"group rewards={out}  hedged/unparsed so far={_calls['hedged']}/{_calls['unparsed']}", flush=True)
    return out

def step_reward(completions, gold_nums, **kwargs):
    """Partial credit: fraction of the gold solution's intermediate numbers that appear
    in the completion. None (skipped) when the gold has no intermediate numbers.
    Anti number-spray: halve the credit when the completion has > 1.5x+3 as many
    distinct numbers as the gold."""
    out = []
    for c, gold in zip(completions, gold_nums):
        if not gold:
            out.append(None)
            continue
        text = c[0]["content"]
        nums = set(_NUM_RE.findall(text))
        hit = sum(1 for g in gold if g in nums) / len(gold)
        spray = len(nums) > 1.5 * len(gold) + 3
        out.append(hit * (0.5 if spray else 1.0))
    return out

def overlong_reward(completions, completion_ids, **kwargs):
    """DAPO soft overlong punishment: linear from 0 to -1 over the last 25% of MAXLEN."""
    cache = int(0.25 * MAXLEN)
    out = []
    for ids in completion_ids:
        L = len(ids)
        out.append(-max(0.0, (L - (MAXLEN - cache)) / cache))
    return out

reward_funcs, reward_weights = [correct_reward], [1.0]
if USE_STEP:
    reward_funcs.append(step_reward); reward_weights.append(W_STEP)
if USE_OVERLONG:
    reward_funcs.append(overlong_reward); reward_weights.append(1.0)

# ---------------- trainer ----------------
training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=LR,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=G,
    # 16 GB rule (grpo_03 lesson): the unsloth logp path scores the whole generation
    # batch in one forward -> keep generation_batch_size == num_generations.
    generation_batch_size=G,
    unsloth_grpo_mini_batch=1,
    unsloth_logit_chunk_multiplier=CHUNK,
    vllm_importance_sampling_correction=VLLM_IS,
    max_completion_length=MAXLEN,
    vllm_max_model_length=max_seq_length,
    temperature=1.0,
    loss_type=LOSS,
    beta=BETA,
    epsilon=0.2,
    epsilon_high=EPS_HIGH,
    scale_rewards=SCALE,
    mask_truncated_completions=MASK_TRUNC,
    num_iterations=ITERS,
    reward_weights=reward_weights,
    multi_objective_aggregation=AGG,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    save_strategy="steps",
    save_steps=50,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"{model_path}-outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
)

# --- Integrity tripwire (bug (c), grpo_07 validated 2026-07-24) ---
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

# vLLM engine init lowers torch._dynamo recompile_limit to 16 (and other threads
# see the default 8) -> FailOnRecompileLimitHit in checkpointed backward. Restore.
import torch._dynamo
torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config._config["recompile_limit"].default = 1024

# --- LoRA-in-rollouts fix (GRPO_LORA_ROLLOUT=1, default on) ---
# Unsloth ships the trained adapter to its colocated vLLM engine by REWRITING the TRL
# trainer source: a regex appends `lora_request = self.model.load_lora(..., load_tensors=True)`
# to every `self.llm.generate(` / `self.llm.chat(` call (unsloth/models/rl.py ~2225).
# TRL 1.10 moved generation out of the trainer into `trl/generation/vllm_generation.py`
# (`self.vllm_generation.generate(...)` -> `VLLMGeneration.llm.generate(...)`), so the regex
# no longer matches anything: `grep -c lora_request unsloth_compiled_cache/UnslothGRPOTrainer.py`
# returns 0 and **every rollout is sampled from the frozen BASE model**. The gradient still
# updates the adapter, but the behaviour policy never improves, which is silent: the run looks
# healthy and simply does not learn. Detected 2026-09-08 because two runs at lr 4e-6 and
# lr 1e-5 produced BIT-IDENTICAL reward sequences for 106 steps (identical rollouts).
# Fix: wrap the engine's own `generate` and attach the live adapter on every call.
if LORA_ROLLOUT:
    _vg = getattr(trainer, "vllm_generation", None)
    _llm = getattr(_vg, "llm", None)
    if _llm is None:
        raise RuntimeError("GRPO_LORA_ROLLOUT=1 but trainer.vllm_generation.llm is missing — "
                           "TRL layout changed again; re-check the rollout path before trusting a run.")
    if not hasattr(model, "load_lora"):
        raise RuntimeError("model.load_lora missing (unsloth _utils.py attaches it) — cannot ship the adapter")
    _lora_dir = os.path.join(f"{model_path}-outputs", "_vllm_lora")
    os.makedirs(_lora_dir, exist_ok=True)
    _orig_generate = _llm.generate
    _lora_calls = {"n": 0}

    def _generate_with_lora(*a, **kw):
        if kw.get("lora_request") is None:
            # load_tensors=True reads the LIVE state_dict and ships clones (dealias patch),
            # and bumps the vLLM LoRA request id each call so nothing is served from cache.
            kw["lora_request"] = model.load_lora(_lora_dir, load_tensors=True)
            _lora_calls["n"] += 1
        return _orig_generate(*a, **kw)

    _llm.generate = _generate_with_lora
    print(f"[grpo_11] LoRA-in-rollouts fix armed on {type(_llm).__name__}.generate", flush=True)
else:
    print("[grpo_11] WARNING: GRPO_LORA_ROLLOUT=0 — rollouts come from the BASE model (broken baseline)", flush=True)
# --- end LoRA-in-rollouts fix ---

# --- NAN-WATCH probe (GRPO_NANWATCH=1) ---
if NANWATCH:
    import sys as _sys
    _mods = [m for m in list(_sys.modules.values())
             if getattr(m, "__file__", "") and str(getattr(m, "__file__", "")).endswith("UnslothGRPOTrainer.py")
             and hasattr(m, "grpo_accumulated_loss")]
    assert _mods, "compiled UnslothGRPOTrainer module not found in sys.modules"
    _orig_gal = _mods[0].grpo_accumulated_loss
    _nw = {"n": 0}

    def _probe_forward(tr, input_ids, attention_mask, logits_to_keep, completion_mask):
        """Clean no-grad bf16 forward; returns (first non-finite (row,pos,token) or None, min logp)."""
        m = tr.model
        with torch.no_grad():
            out = m(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1)
            x = out.logits if hasattr(out, "logits") else out[0]
            x = x[:, :-1, :]
            if x.shape[-1] != len(tr.processing_class):
                # hidden states were returned (UNSLOTH_RETURN_HIDDEN_STATES); apply the (possibly tied) lm_head
                w = m.get_output_embeddings().weight
                x = torch.nn.functional.linear(x.to(w.dtype), w)
            x = x.float()
            print(f"[NAN-WATCH] logits finite={bool(torch.isfinite(x).all())} max|logit|={x.abs().max().item():.3e}", flush=True)
            ids = input_ids[:, -logits_to_keep:]
            lp = torch.log_softmax(x, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            lp = lp * completion_mask
            bad = (~torch.isfinite(lp)).nonzero()
            first = tuple(bad[0].tolist()) + (int(ids[bad[0][0], bad[0][1]]),) if len(bad) else None
            return first, lp[torch.isfinite(lp)].min().item() if torch.isfinite(lp).any() else float("nan")

    import inspect as _inspect
    _gal_names = list(_inspect.signature(_orig_gal).parameters)

    def _gal_watch(*a, **kw):
        _nw["n"] += 1
        _b = dict(zip(_gal_names, a)); _b.update(kw)
        tr, input_ids, attention_mask, logits_to_keep = _b["trainer"], _b["input_ids"], _b["attention_mask"], _b["logits_to_keep"]
        completion_mask, advantages, old_logps, ref_logps = _b["completion_mask"], _b["advantages"], _b["old_logps"], _b["ref_logps"]
        lens = completion_mask.sum(1).tolist()
        adv_ok = bool(torch.isfinite(advantages).all())
        old_ok = old_logps is None or bool(torch.isfinite(old_logps).all())
        ref_ok = ref_logps is None or bool(torch.isfinite(ref_logps).all())
        res = _orig_gal(*a, **kw)
        loss = res[0]
        ok = bool(torch.isfinite(loss).all())
        print(f"[NAN-WATCH call {_nw['n']}] bsz={input_ids.shape[0]} qlen={input_ids.shape[1]} keep={logits_to_keep} "
              f"lens={lens} adv={[round(float(v),3) for v in advantages.flatten().tolist()]} adv_ok={adv_ok} old_ok={old_ok} ref_ok={ref_ok} "
              f"loss={loss.item() if loss.numel()==1 else loss} finite={ok}", flush=True)
        if not ok:
            dump = f"/tmp/grpo11_nan_{_nw['n']}.pt"
            torch.save({"input_ids": input_ids.cpu(), "attention_mask": attention_mask.cpu(), "logits_to_keep": logits_to_keep,
                        "completion_mask": completion_mask.cpu(), "advantages": advantages.cpu(),
                        "old_logps": None if old_logps is None else old_logps.cpu(),
                        "ref_logps": None if ref_logps is None else ref_logps.cpu(), "kw": {k: (v if not torch.is_tensor(v) else v.cpu()) for k, v in kw.items() if k != "trainer"}}, dump)
            try:
                first, mn = _probe_forward(tr, input_ids, attention_mask, logits_to_keep, completion_mask)
                print(f"[NAN-WATCH] dumped {dump}; clean-forward first non-finite logp (row,pos,token)={first} min finite logp={mn:.3f}", flush=True)
                if first is not None:
                    r, pos, t = first
                    print(f"[NAN-WATCH] token {t!r} = {tr.processing_class.decode([t])!r}; context: "
                          f"{tr.processing_class.decode(input_ids[r, -logits_to_keep:][max(0,pos-20):pos+5].tolist())!r}", flush=True)
            except Exception as e:
                print(f"[NAN-WATCH] probe forward failed: {e!r}", flush=True)
        return res

    _mods[0].grpo_accumulated_loss = _gal_watch
    print(f"[NAN-WATCH] armed on {_mods[0].__name__}", flush=True)
# --- end NAN-WATCH ---

if ANOMALY:
    torch.autograd.set_detect_anomaly(True)
# --- in-training dev eval (GRPO_DEVEVAL=N) ---
# Greedy-scores a FIXED slice of MATH-500 with the current adapter, on the engine that is
# already resident, so a run reports a learning curve rather than a single endpoint. The
# slice is fixed and greedy, so successive points are paired: only the policy changes.
if DEVEVAL > 0:
    from transformers import TrainerCallback
    from vllm import SamplingParams as _SP

    _dev = load_dataset("HuggingFaceH4/MATH-500")["test"].select(range(DEVN))
    _dev_prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": USER_TEMPLATE.replace("{problem}", x["problem"])}],
        tokenize=False, add_generation_prompt=True) for x in _dev]
    _dev_gold = list(_dev["answer"])
    _dev_hist = []

    class DevEval(TrainerCallback):
        def _run(self, step):
            engine = trainer.vllm_generation.llm
            # Step 0 runs before any optimizer step, so the adapter is still zero and the
            # point IS the base model on this exact slice — without it the curve floats and
            # every later point has to be compared against a number from a different run.
            lr_req = (model.load_lora(_lora_dir, load_tensors=True)
                      if LORA_ROLLOUT and step > 0 else None)
            outs = engine.generate(_dev_prompts,
                                   _SP(temperature=0.0, max_tokens=MAXLEN),
                                   lora_request=lr_req, use_tqdm=False)
            hits = [is_correct(o.outputs[0].text, g)[0] for o, g in zip(outs, _dev_gold)]
            ok = sum(hits)
            acc = ok / len(_dev_gold)
            _dev_hist.append((step, acc))
            # Per-item hits, so two points can be compared with an exact PAIRED test later
            # (aggregate accuracies only support the far weaker unpaired comparison).
            import json as _j
            with open(f"{model_path}-outputs/deveval_items.jsonl", "a") as f:
                f.write(_j.dumps({"step": step, "acc": acc,
                                  "hits": [int(x) for x in hits]}) + "\n")
            print(f"[DEVEVAL step {step}] MATH-500[:{DEVN}] greedy acc = "
                  f"{ok}/{len(_dev_gold)} = {acc:.2%}   history={[(s, round(a, 4)) for s, a in _dev_hist]}",
                  flush=True)

        def on_train_begin(self, args, state, control, **kwargs):
            self._run(0)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % DEVEVAL == 0:
                self._run(state.global_step)

    trainer.add_callback(DevEval())
    print(f"[grpo_11] dev eval armed: every {DEVEVAL} steps on {DEVN} MATH-500 items", flush=True)
# --- end dev eval ---

trainer.train()
print(f"[grpo_11] reward-call summary: calls={_calls['n']} hedged={_calls['hedged']} unparsed={_calls['unparsed']}", flush=True)

# bug (d): never merge in-process while vLLM is alive. Save the adapter, merge offline.
# Manual re-run: python grpo_merge.py <model_path> adapter-final --base Qwen/Qwen3-1.7B
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
