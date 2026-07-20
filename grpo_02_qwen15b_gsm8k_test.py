import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut VRAM fragmentation; must precede first CUDA alloc
# Import necessary libraries
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
import re

seed = 42
model_path = "outputs/lora-grpo-qwen3"

# Define system prompt
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Sampling parameters (no num_beams)
sampling_params = SamplingParams(
    temperature=0.1,           # Lower for deterministic outputs (GRPO prefers stability)
    top_p=0.9,                 # Slightly narrower than before to focus on high-probability paths
    top_k=30,                  # Moderate filtering to balance diversity and precision
    presence_penalty=0.1,      # Stronger penalty to avoid repetitive reasoning loops
    max_tokens=2048,           # GRPO may need more tokens for detailed step-by-step answers
)

# Function for inference
def generate_answer(question, model, tokenizer, max_length=512):
    text = tokenizer.apply_chat_template([
        {'role': 'system',
         'content': "You are a helpful assistant that always responds with reasoning and answer tags."},
        {'role': 'user',
         'content': f"Please solve the following problem and respond in this format:\n<reasoning>...</reasoning>\n<answer>...</answer>\n\nProblem:\n{question}"}
    ], tokenize=False, add_generation_prompt=True)

    output = model.generate(text, sampling_params=sampling_params)[0].outputs[0].text
    return output  # No need to decode

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
        norm = re.sub(r'[$\u20ac\u00a3\s]', '', content).replace(',', '')
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

# vLLM's V1 engine spawns worker processes; on spawn the module is re-imported,
# so all executable code must sit under __main__ or the workers recurse and crash
# ("An attempt has been made to start a new process before ... bootstrapping").
def main():
    # Load tokenizer & model (vLLM)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 0.9 (vLLM default) tries to grab ~14.3/15.5 GiB and fails when the KDE/Wayland
    # desktop + browsers already hold ~1.6-2 GiB; 0.8 leaves ~3 GiB headroom.
    # max_model_len=2048 matches training and shrinks the KV cache (GSM8K needs << 32k).
    model = LLM(model_path, gpu_memory_utilization=0.8, max_model_len=2048)

    # Load test dataset
    dataset = load_dataset("gsm8k", "main")  # Standard GSM8K dataset
    test_dataset = dataset["test"].shuffle(seed=seed)
    test_dataset = list(test_dataset)  # Convert to list
    print(test_dataset[0])

    # Evaluate model
    correct = 0
    total = 0

    for example in test_dataset[:100]:  # First 100 examples
        question = example["question"]
        ground_truth = extract_final_answer(example["answer"])

        model_answer = generate_answer(question, model, tokenizer)
        model_answer_final = extract_final_answer(model_answer)

        # Debugging output
        print(f"Question: {question}")
        print(f"Model Answer: {model_answer}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Extracted Model Answer: {model_answer_final}")

        # Compare model answer with ground truth
        if model_answer_final is not None and ground_truth is not None and safe_float(model_answer_final) == safe_float(ground_truth):
            correct += 1
        total += 1

        print(f"--{correct}/{total}--")

    # Compute accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
