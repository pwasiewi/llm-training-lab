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

from lc_neobert_compat import patch_neobert

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "outputs/lora-neobert-imdb"
base_model_name = "chandar-lab/NeoBERT"
# Same pin as training. The remote code defines the model, so an unpinned
# reload can score the adapter against a different architecture than the one it
# was trained on.
REVISION = "5424c8efeea6491b151d62dee55a752165407430"

base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    revision=REVISION,
    num_labels=2,
    trust_remote_code=True,
    dtype=torch.bfloat16,  # `torch_dtype` deprecated; bf16 matches training
).to(device)

# The same two shims the training script applies (lc_neobert_compat) are needed
# HERE TOO, and this is the easy one to forget. The freqs_cis fix in particular
# is not a training detail: without it the rotary buffer is uninitialized
# memory, so a perfectly good adapter is evaluated against random positional
# phases and scores at chance — which reads as a corrupted SAVE rather than a
# corrupted LOAD, and sends you looking in the wrong place entirely.
#
# Must run before PeftModel.from_pretrained.
patch_neobert(base_model)

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
