"""Base-weight precision switch shared by the Unsloth LoRA / GRPO scripts.

LORA_PRECISION=4bit|fp8|bf16 picks how the frozen base weights are loaded under
the LoRA adapter; unset = the script's own default, so existing runs, their
_cont continuations and adapter evaluations keep the base they were trained on.

Measured on the RTX 5070 Ti (llm/research/fp8-lora-test, 2026-10-25, Qwen3-1.7B,
LoRA r=16): fp8 0.501 s/step vs bf16 0.555-0.592 and 4bit ~0.57 (+11-18 %),
2.85 vs 4.17 GiB. FP8 tensor cores are not half-rated on GeForce Blackwell,
BF16 ones are (llm/research/tensor-cap-5070ti).

fp8 + colocated vLLM (fast_inference=True) fails at load with
"vLLM weight ... unexpected weight shape [4096, 2048] and scale [4096, 4] and
block size [128, 128]" (unsloth-zoo FP8 block scales vs vLLM's block loader),
so that combination stops here instead of after the model download.
"""
from __future__ import annotations

import os
import sys

CHOICES = ("4bit", "fp8", "bf16")


def base_precision(default: str, fast_inference: bool = False) -> dict:
    """Return the load_in_* kwargs for FastLanguageModel.from_pretrained."""
    p = os.environ.get("LORA_PRECISION", default).strip().lower()
    if p not in CHOICES:
        sys.exit(f"LORA_PRECISION={p!r}: expected one of {', '.join(CHOICES)}")
    # VLLM_USE_DEEP_GEMM=0 lets the probe through: the rejected scale layout ([N, K/512] packed UE8M0 for a
    # [128, 128] block grid) is what vLLM's DeepGEMM path chooses on sm_120; without it vLLM may land on the
    # CUTLASS block-FP8 kernel and keep float32 [N/128, K/128] scales (update review 2026-11-02, vLLM 2.2).
    if p == "fp8" and fast_inference and os.environ.get("VLLM_USE_DEEP_GEMM") != "0":
        sys.exit("LORA_PRECISION=fp8 does not work with the colocated vLLM engine "
                 "(unsloth-zoo FP8 block scales rejected by vLLM). Use 4bit/bf16, "
                 "or GRPO_NO_VLLM=1 where the script supports HF-generate rollouts "
                 "(VLLM_USE_DEEP_GEMM=0 = the untested probe that may avoid the rejected scale layout).")
    print(f"[precision] base weights: {p}"
          f"{'' if p == default else f' (script default {default})'}", flush=True)
    return {"load_in_4bit": p == "4bit", "load_in_fp8": p == "fp8"}


def attn_impl(preferred: str = "flash_attention_2") -> str:
    """attn_implementation for from_pretrained: FA2 only when flash_attn imports, else sdpa.

    flash_attn 2.8.3 was a py3.13-era install (2026-07) and is gone since the py3.14 move; transformers raises
    ImportError for "flash_attention_2" on every HF-forward path (GRPO_NO_VLLM=1, TRL's own vLLM in grpo_01),
    while unsloth's fast_inference path never enforced it. sdpa is the transformers default."""
    import importlib.util
    if preferred == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("[precision] flash_attn not installed: attn_implementation=sdpa", flush=True)
        return "sdpa"
    return preferred
