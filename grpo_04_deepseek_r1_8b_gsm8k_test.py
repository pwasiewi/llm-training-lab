import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut VRAM fragmentation; must precede first CUDA alloc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import re
import os

os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
seed = 43

model_path = "outputs/lora-grpo-deepseek-r1-llama8b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def ensure_bytelevel_tokenizer(tok):
    # transformers 5.x rebuilds LlamaTokenizerFast in sentencepiece style, ignoring the
    # ByteLevel decoder in tokenizer.json: encode drops spaces, decode leaks Ġ/Ċ markers.
    # The merged checkpoint inherits the broken tokenizer_class, so guard here too.
    probe = "a b,\nc d"
    out = tok.decode(tok(probe, add_special_tokens=False).input_ids, skip_special_tokens=True)
    if out.strip() == probe:
        return tok
    from transformers import PreTrainedTokenizerFast
    print(f"WARNING: broken byte-level tokenizer from {tok.name_or_path}; "
          "reloading via PreTrainedTokenizerFast — run 'grpo-fix-hf-tokenizer fix' to repair the HF cache")
    fixed = PreTrainedTokenizerFast.from_pretrained(tok.name_or_path)
    if fixed.pad_token is None:
        fixed.pad_token = tok.pad_token or fixed.eos_token
    return fixed

tokenizer = ensure_bytelevel_tokenizer(tokenizer)

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
