from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datasets
import torch
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("TENSORBOARD_LOGGING_DIR", "outputs/logs")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

SEED = 42
model_path = "outputs/lora-modernbert-imdb"

# ModernBERT-large (HuggingFace / Answer.AI, December 2024)
# Key improvements over RoBERTa: rotary embeddings, Flash Attention 2, 8192-token context,
# alternating local/global attention, ~24% faster inference, new SOTA on many classification tasks.
# No quantization needed — 400M params fits comfortably in fp16 on 16GB VRAM.
model_name = "answerdotai/ModernBERT-large"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ModernBERT attention uses Wqkv (combined QKV) and Wo (output projection)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["Wqkv", "Wo"],   # ModernBERT attention layer names
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

dataset = datasets.load_dataset("imdb")
train_data = dataset["train"].shuffle(seed=SEED)
test_data = dataset["test"].shuffle(seed=SEED)

# ModernBERT supports up to 8192 tokens — use 512 for speed, increase if needed
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_data.map(preprocess_function, batched=True)
test_dataset = test_data.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=model_path,
    report_to=[],
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=True,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    load_best_model_at_end=True,
    save_total_limit=2,
    dataloader_num_workers=0,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

trainer.train()

lora_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
