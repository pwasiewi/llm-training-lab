import logging
# torchao 0.18 still calls the deprecated register_constant() on its Enums at
# import time; silence the resulting torch.utils._pytree warnings (emitted via
# logging, so a warnings filter would not catch them).
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datasets
import torch
import os

from lc_neobert_compat import patch_neobert

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("TENSORBOARD_LOGGING_DIR", "outputs/logs")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

SEED = 42
model_path = "outputs/lora-neobert-imdb"

# NeoBERT (Chandar Lab, Feb 2025) — 250M-param encoder, the main candidate to
# displace lc_06 ModernBERT-large (395M, 96.16%). Deep and narrow: 28 layers x
# 768 hidden. SwiGLU feed-forward, RMSNorm pre-norm, RoPE, 4096-token context.
# Reported to beat ModernBERT-large on GLUE/MTEB with 145M fewer parameters.
#
# RESULT (2026-09-10): 95.92% on the full 25k, 4 epochs / 42 min, best at
# epoch 2. That is 0.24 pp below ModernBERT-large in 13 min less wall time.
# Do NOT read that as a ranking: it is 1-2 SE from a single run of each with no
# seed replication, and IMDB is saturated enough that it cannot separate them
# (the 9B -> 12B decoder step bought 0.05 pp). What the number does support is
# the parameter-efficiency claim. Separating the two needs a harder task.
#
# REQUIRES sci-ml/xformers (in ::pwr since 2026-09-10). The remote code does an
# unguarded `from xformers.ops import SwiGLU` at module scope, so the model
# cannot be imported without it. Its flash_attn import IS guarded and falls
# back to scaled_dot_product_attention, which is the path taken here.
model_name = "chandar-lab/NeoBERT"
# Pin the revision. This repo ships custom code (model.py, rotary.py) that
# transformers re-downloads whenever upstream edits it — it silently pulled a
# new rotary.py mid-session while this script was being written. Unpinned
# trust_remote_code means the model definition can change under a rerun.
REVISION = "5424c8efeea6491b151d62dee55a752165407430"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    revision=REVISION,
    num_labels=2,
    trust_remote_code=True,
    dtype=torch.bfloat16,  # `torch_dtype` deprecated in transformers v5; bf16 avoids GradScaler issues with half-precision trainable params
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=REVISION,
                                          trust_remote_code=True)

# Two things break when NeoBERT's remote code meets transformers 5 + PEFT, and
# only one of them announces itself:
#
#   1. PEFT passes inputs_embeds= unconditionally; NeoBERT has no such
#      parameter, so every forward raises TypeError. Loud.
#   2. freqs_cis is registered persistent=False, so it is not in the checkpoint,
#      and transformers 5 materializes the model from the state dict — leaving
#      the rotary buffer as UNINITIALIZED MEMORY. This does not crash (usually).
#      The model trains happily on garbage positions and scores plausibly.
#
# Both fixes live in lc_neobert_compat.py, shared with the test script and the
# lc_10 harness, because a fix whose absence is silent must not be copy-pasted.
# Read that module before changing anything here — the details matter, and (2)
# in particular has to be applied on the INFERENCE path too.
#
# Must run before get_peft_model.
patch_neobert(model)

# Every nn.Linear in a block: qkv + wo (attention) and w12 + w3 (the SwiGLU
# feed-forward — xformers stores w1/w2 packed into a single w12, shape
# (2*intermediate, hidden)). 28 of each. `dense` and `classifier` are excluded
# on purpose: they are the head, and they are handled below.
#
# THE HEAD IS A TRAP HERE. NeoBERTForSequenceClassification has TWO head
# layers, dense (768->768) then classifier (768->2), and the checkpoint
# contains neither — the load report lists dense.weight, dense.bias,
# classifier.weight and classifier.bias as newly initialized. PEFT's SEQ_CLS
# task type only adds "classifier"/"score" to modules_to_save by default, so
# `dense` would stay frozen at its random init: a random 768x768 projection
# wired in front of the classifier, training nothing. Name both explicitly.
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["qkv", "wo", "w12", "w3"],
    modules_to_save=["dense", "classifier"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

dataset = datasets.load_dataset("imdb")
train_data = dataset["train"].shuffle(seed=SEED)
test_data = dataset["test"].shuffle(seed=SEED)

# 512 to match lc_06, since ModernBERT-large is the model this is being
# compared against and sequence length dominates on IMDB (the seq-128 encoders
# top out at ~93%). NeoBERT itself supports 4096.
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_data.map(preprocess_function, batched=True)
test_dataset = test_data.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=model_path,
    report_to=[],
    per_device_train_batch_size=16,  # 250M at seq 512; same shelf as lc_06, which OOMs at 32
    per_device_eval_batch_size=64,  # eval stores no activations for backward
    gradient_accumulation_steps=2,  # effective batch 32, matching lc_06
    bf16=True,  # matches the bf16 model dtype; fp16 GradScaler cannot unscale half-precision trainable params
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-4,  # lc_06's value, for comparability
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    dataloader_num_workers=0,
    label_names=["labels"],  # Silence Trainer warning for PEFT-wrapped models
    seed=SEED,
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
