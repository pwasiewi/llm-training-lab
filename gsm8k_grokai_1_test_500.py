import re
import random
import os
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

# Fix random seed
seed = 42
random.seed(seed)

# Load test dataset
dataset = load_dataset("gsm8k", "main")
test_dataset = dataset["test"].shuffle(seed=seed)
test_dataset = list(test_dataset)  # Convert to list

# Function to call GPT-4o
def generate_answer_openai(question):
    system_prompt = "You are a helpful assistant that always responds using <reasoning> and <answer> tags."
    user_prompt = f"""Please solve the following problem and respond in this format:
<reasoning>...</reasoning>
<answer>...</answer>

Start your response with the <reasoning> tag and include all calculation steps inside it.
Problem:
{question}"""

    response = client.chat.completions.create(
        model="grok-3-mini-beta",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
        top_p=0.9,
    )
    return response.choices[0].message.content

# Extract numerical final answer
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

# Evaluate
correct = 0
total = 0

for example in test_dataset:
    question = example["question"]
    ground_truth = extract_final_answer(example["answer"])

    model_answer = generate_answer_openai(question)
    model_answer_final = extract_final_answer(model_answer)

    print(f"Question: {question}")
    print(f"Model Answer: {model_answer}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Extracted Model Answer: {model_answer_final}")

    if model_answer_final and ground_truth and safe_float(model_answer_final) == safe_float(ground_truth):
        correct += 1
    total += 1

    print(f"-- {correct}/{total} = {correct/total:.2f} --")

# Final accuracy
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")
