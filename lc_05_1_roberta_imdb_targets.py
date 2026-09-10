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

# lc_05 with ONE variable changed: target_modules.
#
# lc_05 adapts `query` and `key` only — the narrowest LoRA coverage of any
# script in the series, while the others range from q/k/v (electra) through
# qkv+output-proj (modernbert) to full attention + both FFN linears (distilbert,
# neobert). roberta-large is also the WORST large encoder in the table (92.72%,
# below electra-large's 93.36% at 20M fewer params), which makes "the adapter
# never touches V, the attention output projection, or the FFN" a more
# plausible explanation than "roberta is just weaker".
#
# This matters beyond roberta: `target_modules` is the one axis the lc_10
# Optuna harness holds CONSTANT while searching lr/r/alpha/dropout/warmup. If
# it moves accuracy more than those four do, the searches were tuning the wrong
# knobs.
#
# r stays at 36, so per-module rank is unchanged; only the number of adapted
# module types grows (2 -> 5). Trainable parameters therefore rise, which is
# part of the hypothesis, not a confound to control away.
#
# RESULT: NEGATIVE (2026-09-10). Kept as a labelled negative control, like
# lc_04_1.
#
#     trainable params   4.6M -> 17.0M   (3.7x)
#     best eval acc      92.84% -> 92.80%
#     saved adapter      92.72% -> 92.77%
#
# Same epoch count, same best epoch (2), same early stop at 4. The difference
# is 0.05 pp against a ~0.16 pp standard error — nothing. Adapter coverage is
# NOT why roberta-large trails electra-large on IMDB.
#
# The useful reading is broader than roberta. r, alpha and target coverage are
# all CAPACITY knobs, and tripling capacity here changed nothing, so capacity
# is not what limits this task. That is also the cleanest explanation for why
# the lc_10 search bought nothing (README, "Optuna HPO harness"): it was
# searching capacity knobs almost exclusively. The one knob that ever mattered
# in this series was lr, and it mattered as a STABILITY threshold, not a
# capacity setting — lc_04 at 3e-4 scores 50%, at 1e-4 scores 93.4%.
SEED = 42
#torch.manual_seed(seed)
model_path = "outputs/lora-roberta-imdb-targets"

# 1. Load pre-trained BERT model and tokenizer
model_name = "FacebookAI/roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name,
num_labels=2).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Prepare model for use with LoRA
lora_config = LoraConfig(
    r=36,  # Low-rank dimensions
    lora_alpha=60,
    # Full coverage. PEFT matches by name suffix, so "output.dense" catches both
    # attention.output.dense and the block's own output.dense, and
    # "intermediate.dense" catches the FFN's first linear. Deliberately NOT a
    # bare "dense": that would also match classifier.dense in the head, which
    # belongs to modules_to_save, not to the adapter.
    target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
    lora_dropout=0.25,
    bias="none",
    task_type="SEQ_CLS"  # Sequence classification
)

# Freeze base weights; gradient checkpointing OFF — 355M-param model, activations are cheap
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=False,
)
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()  # compare against lc_05's q/k-only count

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
    per_device_train_batch_size=32,  # 355M model at seq 128 — fits easily
    per_device_eval_batch_size=128,  # eval stores no activations for backward
    gradient_accumulation_steps=1,  # keep effective batch at 32 (was 16*2)
    fp16=True,  # Mixed precision to save memory
    num_train_epochs=30,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
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
