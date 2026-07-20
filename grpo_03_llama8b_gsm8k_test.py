import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut VRAM fragmentation; must precede first CUDA alloc
# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig
from vllm import LLM, SamplingParams
from datasets import load_dataset
import re
import os

os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
seed = 43

# Load tokenizer & model (vLLM)
model_path = "outputs/lora-grpo-lama4"
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = LLM(model_path)  # No need for .to(device) or .eval()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


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
# sampling_params = SamplingParams(
#     temperature=0.2,           # Low but not zero to allow some flexibility
#     top_p=0.92,                # Slightly higher to include more possible reasoning paths
#     top_k=40,                  # Filter unlikely tokens
#     presence_penalty=0.05,     # Light penalty to avoid repetition
#     max_tokens=1536            # Increased to accommodate detailed reasoning
# )
sampling_params = SamplingParams(
    temperature=0.1,           # Lower for deterministic outputs (GRPO prefers stability)
    top_p=0.9,                 # Slightly narrower than before to focus on high-probability paths
    top_k=30,                  # Moderate filtering to balance diversity and precision
    presence_penalty=0.1,      # Stronger penalty to avoid repetitive reasoning loops
    max_tokens=2048,           # GRPO may need more tokens for detailed step-by-step answers
)
# Load test dataset
dataset = load_dataset("gsm8k", "main")  # Standard GSM8K dataset
test_dataset = dataset["test"].shuffle(seed=seed)
test_dataset = list(test_dataset)  # Convert to list
print(test_dataset[0])

# Function for inference
def generate_answer(question, model, tokenizer, max_length=512):
    text = tokenizer.apply_chat_template([
        {'role': 'system',
         'content': "You are a helpful assistant that always responds using <reasoning> and <answer> tags."},
        {'role': 'user',
         'content':
             "Please solve the following problem and respond in this format:\n"
                "<reasoning>...</reasoning>\n"
                "<answer>...</answer>\n\n"
                "Start your response with the <reasoning> tag and include all calculation steps inside it.\n"
                "Problem:\n"
                f"{question}"},
    ], tokenize=False, add_generation_prompt=True)

    #output = model.generate(text, sampling_params=sampling_params)[0].outputs[0].text
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        top_k=30,
        max_new_tokens=2048,
        repetition_penalty=1.1  # equivalent to presence_penalty
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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

# Evaluate model
correct = 0
total = 0

for example in test_dataset:  # First 100 examples
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

    print(f"-- {correct}/{total} = {correct/total:.2f} --")

# Compute accuracy
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")
