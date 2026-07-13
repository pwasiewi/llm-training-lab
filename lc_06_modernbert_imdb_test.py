from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
import datasets
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "outputs/lora-modernbert-imdb"
base_model_name = "answerdotai/ModernBERT-large"

base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=2,
    torch_dtype=torch.float16,
).to(device)

model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = datasets.load_dataset("imdb")
test_data = dataset["test"].shuffle(seed=42)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

test_dataset = test_data.map(preprocess_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(test_dataset), 32):
        batch = test_dataset[i:i+32]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        acc = accuracy_score(all_labels, all_preds)
        print(f"[{i+len(preds)}/{len(test_dataset)}] Accuracy: {acc:.4f}")

print("\n" + classification_report(all_labels, all_preds, target_names=["negative", "positive"]))
print(f"Final Accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")
