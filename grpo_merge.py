# Universal CPU-only LoRA merge for grpo_* runs — replaces the per-script
# grpo_0X_merge.py copies.
#
# Why offline/CPU: unsloth's in-training save_pretrained_merged(fast_inference=True)
# corrupted the non-fused LoRA modules (o_proj/down_proj hit ~1e13 — garbage reads
# from the live vLLM memory pool, bug (d), aliasing family, 2026-07-24). A PEFT merge
# from a checkpoint touches neither the GPU nor vLLM, so it cannot alias anything.
# Safe to run while a training job occupies the GPU (~2x model size in RAM, bf16).
#
# Usage:
#   python grpo_merge.py MODEL_PATH [CHECKPOINT] [--base NAME] [--outputs DIR]
#
#   MODEL_PATH  merged output dir, e.g. outputs/lora-grpo-phi4-mini-v2
#               (what the _test.py vLLM scripts load)
#   CHECKPOINT  dir name under OUTPUTS (e.g. checkpoint-935 or adapter-final),
#               or a full path; default: highest-numbered checkpoint-N
#   --base      base model override; default: base_model_name_or_path from the
#               checkpoint's adapter_config.json
#   --outputs   checkpoints dir; default: MODEL_PATH-outputs
import os, sys, glob, json, argparse, warnings, logging

# Hide the GPU entirely: transformers 5.x from_pretrained warms up the CUDA
# caching allocator when an accelerator is visible, which OOMs (and would
# defeat the whole point) while a training job holds the VRAM.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


def die(msg):
    print(f"grpo_merge: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="CPU-only PEFT merge of a grpo_* LoRA checkpoint.")
    ap.add_argument("model_path", help="merged output dir (e.g. outputs/lora-grpo-phi4-mini-v2)")
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="checkpoint dir name under outputs, or full path (default: newest checkpoint-N)")
    ap.add_argument("--base", default=None, help="base model override")
    ap.add_argument("--outputs", default=None, help="checkpoints dir (default: MODEL_PATH-outputs)")
    args = ap.parse_args()

    outputs = args.outputs or f"{args.model_path}-outputs"

    if args.checkpoint:
        ckpt = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.join(outputs, args.checkpoint)
    else:
        cands = [p for p in glob.glob(os.path.join(outputs, "checkpoint-*"))
                 if p.rsplit("-", 1)[1].isdigit()]
        if not cands:
            die(f"no checkpoint-N dirs in {outputs} — pass CHECKPOINT explicitly")
        ckpt = max(cands, key=lambda p: int(p.rsplit("-", 1)[1]))

    adapter_cfg = os.path.join(ckpt, "adapter_config.json")
    if not os.path.isfile(adapter_cfg):
        die(f"{ckpt} has no adapter_config.json — not a LoRA checkpoint")

    with open(adapter_cfg) as f:
        base_name = args.base or json.load(f).get("base_model_name_or_path")
    if not base_name:
        die("base model unknown: not in adapter_config.json, pass --base")
    # unsloth rewrites base_model_name_or_path to its pre-quantized mirror
    # (e.g. unsloth/phi-4-mini-instruct-unsloth-bnb-4bit). Merging into a 4-bit
    # base loses precision and diverges from the validated bf16 merge — the
    # caller must name the original 16-bit model instead.
    if not args.base and "4bit" in base_name.lower():
        die(f"adapter_config points at a quantized repo ({base_name}) — "
            f"pass --base <original 16-bit model>, e.g. --base microsoft/Phi-4-mini-instruct")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Merging adapter: {ckpt}  (base: {base_name})  ->  {args.model_path}")
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ckpt)
    model = model.merge_and_unload()

    # A previous merge may have left a different shard layout; stale shards or a
    # stale model.safetensors.index.json make vLLM raise FileNotFoundError.
    for old in glob.glob(os.path.join(args.model_path, "model*.safetensors*")):
        os.remove(old)

    model.save_pretrained(args.model_path)
    try:
        tok = AutoTokenizer.from_pretrained(ckpt)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_name)
    tok.save_pretrained(args.model_path)
    print(f"Done. vLLM can now load {args.model_path}")


if __name__ == "__main__":
    main()
