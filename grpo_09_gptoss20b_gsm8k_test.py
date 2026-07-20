# Evaluate the GRPO-trained gpt-oss-20b LoRA adapter on GSM8K test (first 100).
#
# Unlike the other *_test.py scripts this cannot use vLLM's LLM(model_path):
# model_path holds only a LoRA adapter, and a merged bf16 20B (~40 GB) fits
# neither disk nor VRAM. Instead: unsloth loads the bnb-4bit base + adapter
# and generates through its inference path (same as training-time generation).
# Stop aillama and check nvidia-smi first — the base alone needs ~12 GB.

# --- Silence cosmetic third-party startup noise (must run before heavy imports) ---
import os, warnings, logging
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut VRAM fragmentation on the 16 GB card
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("unsloth_zoo").setLevel(logging.CRITICAL)
# gpt-oss MoE: the default "grouped" path dequantizes ALL 32 experts into
# torch._grouped_mm stacks (~1.6 GB transient PER MLP layer) - the actual cause
# of the 16 GB OOMs regardless of sequence length. Force the per-expert eager
# loop instead (~33 MB transients, slower forward).
os.environ.setdefault("UNSLOTH_GPTOSS_GROUPED", "0")  # headless: override with UNSLOTH_GPTOSS_GROUPED=1 (fast grouped path)
# ----------------------------------------------------------------------------------

from unsloth import FastLanguageModel
from datasets import load_dataset
import re
import torch

seed = 42
model_name = "unsloth/gpt-oss-20b"
model_path = "outputs/lora-grpo-gptoss20b"  # adapter dir saved by grpo_09_gptoss20b_gsm8k.py
max_seq_length = 1536

# Function to extract final numerical answer
def extract_final_answer(text: str) -> str:
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

def safe_float(x):
    try:
        return float(x.replace(",", "").strip())
    except:
        return None

def generate_answer(question, model, tokenizer):
    inputs = tokenizer.apply_chat_template([
        {'role': 'system',
         'content': "You are a helpful assistant that always responds with reasoning and answer tags."},
        {'role': 'user',
         'content': f"Please solve the following problem and respond in this format:\n<reasoning>...</reasoning>\n<answer>...</answer>\n\nProblem:\n{question}"}
    ], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=1024,
            # Low temperature for a stable benchmark, matching the other *_test.py
            # scripts (note: OpenAI's recommended sampling for gpt-oss is temp=1.0).
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )
    # Decode only the completion; harmony channel markers are special tokens and are skipped
    return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,  # PEFT adapter dir; unsloth resolves the bnb-4bit base from it
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )
    FastLanguageModel.for_inference(model)

    dataset = load_dataset("gsm8k", "main")
    test_dataset = list(dataset["test"].shuffle(seed=seed))
    print(test_dataset[0])

    correct = 0
    total = 0
    for example in test_dataset[:100]:  # First 100 examples
        question = example["question"]
        ground_truth = extract_final_answer(example["answer"])

        model_answer = generate_answer(question, model, tokenizer)
        model_answer_final = extract_final_answer(model_answer)

        print(f"Question: {question}")
        print(f"Model Answer: {model_answer}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Extracted Model Answer: {model_answer_final}")

        if model_answer_final is not None and ground_truth is not None and safe_float(model_answer_final) == safe_float(ground_truth):
            correct += 1
        total += 1
        print(f"--{correct}/{total}--")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
