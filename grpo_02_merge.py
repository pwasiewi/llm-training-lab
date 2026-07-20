# Merge a GRPO LoRA checkpoint into a standalone 16-bit model that vLLM (_test.py) can load.
# Run AFTER stopping the training that holds the GPU (16 GB shared).
# Usage: python grpo_02_merge.py [checkpoint-650]   (default: newest checkpoint)
import os, sys, glob, warnings, logging
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut VRAM fragmentation on the 16 GB card
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_name  = "Qwen/Qwen2.5-1.5B-Instruct"
model_path = "outputs/lora-grpo-qwen3"                 # merged output dir (what _test.py reads)
outputs    = f"{model_path}-outputs"

# Pick checkpoint: CLI arg, else the highest-numbered one.
if len(sys.argv) > 1:
    ckpt = os.path.join(outputs, sys.argv[1])
else:
    cands = glob.glob(os.path.join(outputs, "checkpoint-*"))
    ckpt = max(cands, key=lambda p: int(p.rsplit("-", 1)[1]))
print(f"Merging adapter: {ckpt}  ->  {model_path}")

base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ckpt)          # loads adapter_config.json + adapter_model.safetensors
model = model.merge_and_unload()                       # fold LoRA into base weights
model.save_pretrained(model_path)
AutoTokenizer.from_pretrained(ckpt).save_pretrained(model_path)
print(f"Done. vLLM can now load {model_path}")
