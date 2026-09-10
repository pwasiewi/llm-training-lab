# lc_04_1 — lc_04 (electra-large) with fp16 swapped for bf16. ONE variable.
#
# RESULT: NEGATIVE. Kept as a ruled-out control, not as a fix.
# Ran 2026-09-10 on the full 25k and diverged exactly like lc_04 —
# eval_loss 0.6992, accuracy 0.5000, and f1 = precision = recall = 0, i.e. one
# class predicted for all 25k examples. Precision is NOT the cause.
#
# What the cause turned out to be: the NUMBER OF OPTIMIZER STEPS at lr 3e-4.
# The lc_10 study scored 0.9250 on this configuration using a 5000-example
# subset (157 steps/epoch); the identical harness on the full 25k (782
# steps/epoch) collapses to 0.5000. So README section 4 was right the first
# time — electra-large really is unstable at lr 3e-4 — and the subset simply
# ends before the collapse. Model construction (bias="lora_only",
# prepare_model_for_kbit_training, fp32 vs bf16 weights) was eliminated by that
# same run and explains nothing.
#
# Methodological note worth keeping: an HPO study run on a subset never
# entered the regime that contains the failure it was meant to diagnose.
#
# --- original hypothesis, now refuted, kept for the record ---
#
# README section 4 records lc_04 as diverged: eval_loss pinned at ln 2 = 0.693
# and accuracy at exactly 0.500 for all six epochs, attributed there to
# "the classic electra-large instability at lr 3e-4", with a suggested retry at
# lr 5e-5..1e-4.
#
# That attribution looks wrong. The lc_10 Optuna study (2026-09-10) ran this
# exact configuration — lr 3e-4, r=40, alpha=120, dropout 0.2 — as its enqueued
# trial 0 and scored 0.9250, no divergence at all. The study differed from
# lc_04 in several ways, but the one that plausibly explains a hard failure is
# precision: it loads the backbone in bf16 and trains with bf16=True, whereas
# lc_04 keeps fp32 weights and runs fp16 autocast with a GradScaler.
#
# fp16 has ~5 exponent bits against bf16's 8. When the scaler cannot find a
# scale that keeps electra-large's gradients representable it repeatedly skips
# steps, the LoRA weights never move, and the classifier sits at chance —
# which is exactly the reported signature: not a noisy 50%, but 0.500 and
# ln 2 held flat for every epoch.
#
# This script changes ONLY the autocast dtype: the backbone still loads in
# fp32, exactly as lc_04 does. That is deliberate and is the tighter test —
# switching the stored weights to bf16 as well would move two variables at
# once, and it is the GradScaler path, not weight storage, that is on trial.
# `bf16=True` needs no scaler because bf16 carries fp32's exponent range.
#
# Everything else is lc_04 byte for byte (verified by diff: two lines differ,
# this one and the output dir). A pass confirms precision as the cause and
# makes the "retry with lr 5e-5..1e-4" advice in README section 4 wrong.
# Output goes to a separate dir so the original adapter survives.
import logging
# torchao 0.18 still calls the deprecated register_constant() on its Enums at
# import time; silence the resulting torch.utils._pytree warnings (emitted via
# logging, so a warnings filter would not catch them).
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
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
model_path = "outputs/lora-electra-imdb-bf16"

# Load pre-trained Mistral model and tokenizer
model_name = "google/electra-large-discriminator"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA for Mistral architecture
lora_config = LoraConfig(
    r=40,  # Low-rank dimension
    lora_alpha=120,
    target_modules=["query", "key", "value"],
    #target_modules=["query", "key"],
    lora_dropout=0.2,
    bias="lora_only",
    task_type="SEQ_CLS"
)

# Freeze base weights; gradient checkpointing OFF — 335M-param model, activations are cheap
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=False,
)
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
    per_device_train_batch_size=32,  # 335M model at seq 128 — fits easily
    per_device_eval_batch_size=128,  # eval stores no activations for backward
    gradient_accumulation_steps=1,  # keep effective batch at 32 (was 16*2)
    bf16=True,  # THE ONLY CHANGE vs lc_04 (see header)
    num_train_epochs=30,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=3e-4,
    load_best_model_at_end=True,
    save_total_limit=2,  # Limit checkpoints to save disk space
    dataloader_num_workers=0,  # Avoid multiprocessing re-importing this script.
    label_names=["labels"]  # Silence Trainer warning for PEFT-wrapped models
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
