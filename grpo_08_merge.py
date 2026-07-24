# Merge a grpo_08 LoRA checkpoint into a standalone 16-bit model that vLLM (_test.py) can load.
# CPU-only on purpose: unsloth's in-training save_pretrained_merged(fast_inference=True)
# corrupts LoRA modules by aliasing the live vLLM memory pool (bug (d), grpo_07
# proved this 2026-07-24 — o_proj/down_proj hit ~1e13). A PEFT merge from the
# checkpoint touches neither the GPU nor vLLM, so it cannot alias anything.
# Usage: python grpo_08_merge.py [checkpoint-935]   (default: newest checkpoint)
# NOTE: if re-merging into an existing dir, delete a stale model.safetensors.index.json
# left by the old sharded save, or vLLM raises FileNotFoundError on the missing shards.
import os, sys, glob, warnings, logging
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_name  = "Qwen/Qwen3-4B"
model_path = "outputs/lora-grpo-qwen3-4b-r16"          # merged output dir (what _test.py reads)
outputs    = f"{model_path}-outputs"

# Pick checkpoint: CLI arg, else the highest-numbered one.
if len(sys.argv) > 1:
    ckpt = os.path.join(outputs, sys.argv[1])
else:
    cands = glob.glob(os.path.join(outputs, "checkpoint-*"))
    ckpt = max(cands, key=lambda p: int(p.rsplit("-", 1)[1]))
print(f"Merging adapter: {ckpt}  ->  {model_path}")

# Drop a stale sharded index so vLLM does not look for missing shard files.
stale = os.path.join(model_path, "model.safetensors.index.json")
if os.path.exists(stale) and not os.path.exists(os.path.join(model_path, "model-00001-of-00002.safetensors")):
    os.remove(stale)

base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ckpt)          # loads adapter_config.json + adapter_model.safetensors
model = model.merge_and_unload()                       # fold LoRA into base weights
model.save_pretrained(model_path)
AutoTokenizer.from_pretrained(ckpt).save_pretrained(model_path)
print(f"Done. vLLM can now load {model_path}")
