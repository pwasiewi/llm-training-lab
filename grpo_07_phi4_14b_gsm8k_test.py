# GSM8K test-set evaluation for grpo_07 (Phi-4-mini) — batched vLLM.
#
# Usage:
#   python grpo_07_phi4_14b_gsm8k_test.py [model_path_or_hf_id]
#
# Default model: outputs/lora-grpo-phi4-mini (merged 16-bit written at the end of
# grpo_07_phi4_14b_gsm8k.py). Run twice to compare trained vs base:
#   python grpo_07_phi4_14b_gsm8k_test.py outputs/lora-grpo-phi4-mini
#   python grpo_07_phi4_14b_gsm8k_test.py microsoft/Phi-4-mini-instruct
#
# Greedy decoding (temperature=0) for a deterministic, standard GSM8K comparison.
# gpu_memory_utilization=0.8 + max_model_len=2048: no colocated trainer here, but
# the KDE/Wayland desktop holds ~1-2 GiB (see the unsloth-grpo-patching skill).
import os
import sys
import re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL = "microsoft/Phi-4-mini-instruct"

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
    except (ValueError, AttributeError):
        return None

def build_prompt(tokenizer, question: str) -> str:
    # Same system/user prompt as training, so both models are scored on the task
    # the trained one was optimized for.
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': "You are a helpful assistant that always responds using <reasoning> and <answer> tags."},
        {'role': 'user', 'content': (
            "Please solve the following problem and respond in this format:\n"
            "<reasoning>...</reasoning>\n"
            "<answer>...</answer>\n\n"
            "Start your response with the <reasoning> tag and include all calculation steps inside it.\n"
            f"Problem:\n{question}"
        )},
    ], tokenize=False, add_generation_prompt=True)

def main():
    # vLLM V1 spawns worker processes that re-import this module — all executable
    # code must stay under __main__ or the engine init recurses.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import load_dataset

    model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/lora-grpo-phi4-mini"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        # merged dir may lack tokenizer files; the chat template is the base one anyway
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    data = load_dataset("openai/gsm8k", "main")["test"]
    prompts = [build_prompt(tokenizer, ex["question"]) for ex in data]
    truths = [extract_final_answer(ex["answer"]) for ex in data]

    llm = LLM(model=model_path, dtype="bfloat16",
              gpu_memory_utilization=0.8, max_model_len=2048)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))

    correct = fmt = 0
    fmt_pattern = re.compile(r'<reasoning>.*?</reasoning>\s*<answer>.*?</answer>', re.DOTALL)
    for i, (gt, out) in enumerate(zip(truths, outs)):
        resp = out.outputs[0].text
        pred = extract_final_answer(resp)
        if fmt_pattern.search(resp):
            fmt += 1
        if pred is not None and gt is not None and safe_float(pred) == safe_float(gt):
            correct += 1
        if i < 3:  # spot-check the first few completions in the log
            print(f"--- sample {i} ---\n{resp}\nGT: {gt} | Extracted: {pred}", flush=True)

    n = len(truths)
    print(f"\nModel: {model_path}")
    print(f"Format compliance (<reasoning>+<answer>): {fmt}/{n} = {fmt/n*100:.2f}%")
    print(f"Final Accuracy: {correct}/{n} = {correct/n*100:.2f}%")

if __name__ == "__main__":
    main()
