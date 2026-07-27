import logging
# torchao 0.18 still calls the deprecated register_constant() on its Enums at
# import time; silence the resulting torch.utils._pytree warnings (emitted via
# logging, so a warnings filter would not catch them).
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
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
model_path = "outputs/lora-gemma2-imdb-liger"

# Load pre-trained model and tokenizer
model_name = "google/gemma-2-2b-it"

# Liger variant of lc_07: patch the Gemma-2 module classes with fused Triton
# kernels (RMSNorm, RoPE, GeGLU) BEFORE from_pretrained so the model is built
# on the patched classes. FusedLinearCrossEntropy targets the CausalLM head
# only, so it is inert for SEQ_CLS — expect activation-memory/step-time gains,
# not logits-memory gains. Compare against lc_07_gemma2b_imdb.py.
# The liger import eagerly imports VL modeling modules whose incomplete
# docstrings make transformers-HEAD auto_docstring lint emit "[ERROR] ...
# not documented" lines via bare print() (not logging) — swallow stdout for
# the import only; the apply call below stays unredirected.
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from liger_kernel.transformers import apply_liger_kernel_to_gemma2
apply_liger_kernel_to_gemma2()
print("Liger kernels applied to Gemma-2 classes")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # Gemma is trained in bf16; fp16 compute can overflow
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto"  # already places the model on GPU; .to(device) on a 4-bit model is redundant
)
# Classification needs no KV cache; a returned DynamicCache breaks Trainer eval gathering.
# Gemma-3 nests the text config, so disable the cache at both levels and tell the
# Trainer to drop past_key_values from eval outputs.
model.config.use_cache = False
if hasattr(model.config, "text_config"):
    model.config.text_config.use_cache = False
model.config.keys_to_ignore_at_inference = ["past_key_values"]
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

# Prepare model for kbit training; gradient checkpointing OFF — the 2B model at
# seq_len=128 uses ~6 GB of 16 GB VRAM, so trading memory for recompute only slows steps down
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=False,
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
    per_device_train_batch_size=16,  # bs=16 used only 6/16 GB — larger batch improves GPU utilization
    per_device_eval_batch_size=32,  # eval has no activations to store; go wide
    gradient_accumulation_steps=2,  # keep effective batch at 32 (was 16*2)
    bf16=True,  # Mixed precision to save memory (bf16: Gemma numerics, no GradScaler)
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-4,
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
print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

# 6. Save fine-tuned model
lora_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
