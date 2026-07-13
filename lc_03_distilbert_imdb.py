from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import datasets
import torch
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("TENSORBOARD_LOGGING_DIR", "outputs/logs")

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load pre-trained BERT model and tokenizer
model_name = "distilbert/distilbert-base-uncased"
lora_path = "outputs/lora-distilbert"
seed = 43
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Prepare model for use with LoRA
lora_config = LoraConfig(
    r=128,  # Low-rank dimensions
    lora_alpha=128,
    target_modules=["attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin", "ffn.lin1", "ffn.lin2"],  # Target layers
    lora_dropout=0.25,
    bias="lora_only",
    task_type="SEQ_CLS",  # Sequence classification
    modules_to_save=["pre_classifier"]
)

# Prepare model for int8 training to save memory
model = prepare_model_for_kbit_training(
    model,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
lora_model = get_peft_model(model, lora_config)

# 3. Prepare data (IMDb dataset)
dataset = datasets.load_dataset("imdb")
# train_data = dataset["train"].shuffle(seed=seed).select(range(2000))  # Smaller subset for testing
# test_data = dataset["test"].shuffle(seed=seed).select(range(500))
train_data = dataset["train"].shuffle(seed=seed)  # full dataset
test_data = dataset["test"].shuffle(seed=seed)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,
padding="max_length", max_length=128)

train_dataset = train_data.map(preprocess_function, batched=True)
test_dataset = test_data.map(preprocess_function, batched=True)

# 4. Training configuration
training_args = TrainingArguments(
    output_dir=lora_path,
    report_to=[],  # Disable W&B logging
    per_device_train_batch_size=16,  # Smaller batch size for 8GB VRAM
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,  # Compensate for smaller batch size
    fp16=True,  # Mixed precision to save memory
    num_train_epochs=15,
    weight_decay=0.01,
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    warmup_steps=500,  # Add warmup steps
    load_best_model_at_end=True,
    save_total_limit=2,  # Limit checkpoints to save disk space
    dataloader_num_workers=0  # Avoid multiprocessing re-importing this script.
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

# 5. Start training
trainer.train()

# 6. Save fine-tuned model
lora_model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
