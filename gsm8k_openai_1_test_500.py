import re
import random
import os
from datasets import load_dataset
import openai

client = openai.OpenAI()

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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0.1
    )
    return response.choices[0].message.content

# Extract numerical final answer
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
