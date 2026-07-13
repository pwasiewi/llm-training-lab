from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datasets
import torch
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("TENSORBOARD_LOGGING_DIR", "outputs/logs")
# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

SEED = 42
#torch.manual_seed(seed)
model_path = "outputs/lora-gemma2-imdb"

# Load pre-trained model and tokenizer
model_name = "google/gemma-2-2b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Configure LoRA
lora_config = LoraConfig(
    r=32,  # Low-rank dimension
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Adapted for LLaMA architecture
    #target_modules=["q_proj", "k_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

# Prepare model for int8 training to save memory
model = prepare_model_for_kbit_training(
    model,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
# model.gradient_checkpointing_enable()
lora_model = get_peft_model(model, lora_config)

# 3. Prepare data (IMDb dataset)
dataset = datasets.load_dataset("imdb")
# train_data = dataset["train"].shuffle(seed=SEED).select(range(2000))  # Smaller subset for testing
# test_data = dataset["test"].shuffle(seed=SEED).select(range(500))
train_data = dataset["train"].shuffle(seed=SEED)  # full dataset
test_data = dataset["test"].shuffle(seed=SEED)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,
padding="max_length", max_length=128)

train_dataset = train_data.map(preprocess_function, batched=True)
test_dataset = test_data.map(preprocess_function, batched=True)

# 4. Training configuration
training_args = TrainingArguments(
    output_dir=model_path,
    report_to=[],  # Disable W&B logging
    per_device_train_batch_size=16,  # Smaller batch size for 8GB VRAM
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Compensate for smaller batch size
    fp16=True,  # Mixed precision to save memory
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-4,
    load_best_model_at_end=True,
    save_total_limit=2,  # Limit checkpoints to save disk space
    dataloader_num_workers=0  # Avoid multiprocessing re-importing this script.
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    #label_names=["negative", "positive"]  # Add label names for the IMDB dataset
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

# 5. Start training
trainer.train()

# 6. Save fine-tuned model
lora_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
