import logging
# torchao 0.18 still calls the deprecated register_constant() on its Enums at
# import time; silence the resulting torch.utils._pytree warnings (emitted via
# logging, so a warnings filter would not catch them).
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
import datasets
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "outputs/lora-bert-roberta2"
base_model_name = "FacebookAI/roberta-large"

# The adapter dir holds only LoRA weights + the saved classifier head; the base
# model must be loaded explicitly and the adapter applied via PeftModel —
# loading the adapter dir directly with AutoModel silently yields a random head.
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=2,
).to(device)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = datasets.load_dataset("imdb")
test_data = dataset["test"].shuffle(seed=42)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

test_dataset = test_data.map(preprocess_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

BATCH = 64
all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(test_dataset), BATCH):
        batch = test_dataset[i:i+BATCH]
        outputs = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
        all_preds.extend(outputs.logits.argmax(-1).cpu().tolist())
        all_labels.extend(batch["label"].tolist())
        if (i // BATCH) % 20 == 0:
            print(f"[{len(all_preds)}/{len(test_dataset)}] Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

print("\n" + classification_report(all_labels, all_preds, target_names=["negative", "positive"]))
print(f"Final Accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")
