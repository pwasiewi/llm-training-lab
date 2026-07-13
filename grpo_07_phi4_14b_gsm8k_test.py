from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import re
import os

os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
seed = 43

model_path = "outputs/lora-grpo-phi4"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Phi-4 prefers bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = load_dataset("gsm8k", "main")
test_dataset = list(dataset["test"].shuffle(seed=seed))

def generate_answer(question, model, tokenizer):
    text = tokenizer.apply_chat_template([
        {'role': 'system', 'content': "You are a helpful assistant that always responds using <reasoning> and <answer> tags."},
        {'role': 'user', 'content': (
            "Please solve the following problem and respond in this format:\n"
            "<reasoning>...</reasoning>\n"
            "<answer>...</answer>\n\n"
            "Start your response with the <reasoning> tag and include all calculation steps inside it.\n"
            f"Problem:\n{question}"
        )},
    ], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        top_k=30,
        max_new_tokens=2048,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_final_answer(answer):
    tag_match = re.search(r'<answer>(.*?)</answer>', answer)
    if tag_match:
        content = tag_match.group(1).strip()
        if content.replace(',', '').replace('.', '', 1).replace('-', '', 1).isdigit():
            return content.replace(',', '')
    matches = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer)
    return matches[-1].replace(",", "").strip() if matches else None

def safe_float(x):
    try:
        return float(x.replace(",", "").strip())
    except:
        return None

correct = 0
total = 0
for example in test_dataset:
    question = example["question"]
    ground_truth = extract_final_answer(example["answer"])
    model_answer = generate_answer(question, model, tokenizer)
    model_answer_final = extract_final_answer(model_answer)
    print(f"Question: {question}")
    print(f"Model Answer: {model_answer}")
    print(f"Ground Truth: {ground_truth} | Extracted: {model_answer_final}")
    if model_answer_final is not None and ground_truth is not None and safe_float(model_answer_final) == safe_float(ground_truth):
        correct += 1
    total += 1
    print(f"-- {correct}/{total} = {correct/total:.2f} --")

print(f"Final Accuracy: {correct/total*100:.2f}%")
