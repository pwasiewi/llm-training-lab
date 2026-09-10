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


def merge_fidelity(ckpt, before, after, sample=8):
    """How much of W += BA actually landed in the saved bf16 weights?

    bf16 keeps 8 mantissa bits, so a weight of magnitude 3e-2 has a ULP of ~1.2e-4.
    A small LoRA update (|BA| ~ 5e-6 after a short low-lr run) is far below half a
    ULP and round-to-nearest discards it *entirely*: g1 kept 6% of its update and
    left 93% of the individual weights bit-identical to the base (2026-09-09).
    The merged model then benchmarks as the base model and the run looks like a null.

    The older tripwire only caught the opposite failure (|B| exploding to ~1e13,
    the 2026-07-24 aliasing bug), so this direction went unnoticed for four runs.
    """
    import torch
    from safetensors.torch import safe_open
    with open(os.path.join(ckpt, "adapter_config.json")) as f:
        cfg = json.load(f)
    scale = cfg["lora_alpha"] / cfg["r"]
    lo = {}
    with safe_open(os.path.join(ckpt, "adapter_model.safetensors"), "pt") as h:
        for k in h.keys():
            lo[k] = h.get_tensor(k)
    names = sorted({k.split(".lora_")[0] for k in lo if ".lora_A" in k})
    if not names:
        return None
    step = max(1, len(names) // sample)
    kept_num = kept_den = 0.0
    zeros = []
    for n in names[::step]:
        key = n.replace("base_model.model.", "") + ".weight"
        if key not in before:
            continue
        a, b = lo[n + ".lora_A.weight"].float(), lo[n + ".lora_B.weight"].float()
        intended = (b @ a) * scale
        actual = after[key].float() - before[key]
        kept_num += (actual - intended).pow(2).sum().item()
        kept_den += intended.pow(2).sum().item()
        zeros.append((actual == 0).float().mean().item())
    if kept_den == 0:
        return None
    return 1 - (kept_num / kept_den) ** 0.5, sum(zeros) / len(zeros), len(zeros)


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

    from safetensors.torch import safe_open
    with safe_open(os.path.join(ckpt, "adapter_model.safetensors"), "pt") as _h:
        _adapted = {k.split(".lora_")[0].replace("base_model.model.", "") + ".weight"
                    for k in _h.keys() if ".lora_A" in k}
    sd = base.state_dict()
    before = {k: sd[k].detach().float().clone() for k in sorted(_adapted)[::max(1, len(_adapted) // 8)]
              if k in sd}

    model = PeftModel.from_pretrained(base, ckpt)
    model = model.merge_and_unload()

    fid = merge_fidelity(ckpt, before, model.state_dict())
    if fid:
        kept, zero, n = fid
        print(f"merge fidelity ({n} sampled tensors): {kept:.1%} of the update survived bf16, "
              f"{zero:.1%} of weights unchanged")
        if kept < 0.5:
            print(f"grpo_merge: WARNING — the bf16 merge discarded {1 - kept:.0%} of this adapter.\n"
                  f"  The update is below bf16 resolution, so {args.model_path} will benchmark\n"
                  f"  close to the base model no matter how well training went. Saving in fp32\n"
                  f"  does not help: any bf16 runtime re-quantises it away at load.\n"
                  f"  Evaluate through the adapter instead:\n"
                  f"    python grpo_11_qwen3_17b_math_test.py {base_name} --lora {ckpt}",
                  file=sys.stderr)

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
