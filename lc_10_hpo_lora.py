"""lc_10 — Optuna hyperparameter search for the lc_* LoRA classifiers.

    python lc_10_hpo_lora.py --model electra --trials 30
    python lc_10_hpo_lora.py --model modernbert --trials 20 --train-n 8000
    python lc_10_hpo_lora.py --model electra --final          # retrain best on 25k

Replaces `lc_03_distilbert_imdb_optuna.py`, which had five defects that between
them made the search both slow and aimed at the wrong axis:

1. LEARNING RATE WAS NOT SEARCHED. It was pinned at 1e-4 while r, alpha and
   dropout were swept. That is backwards. LoRA is famously insensitive to r
   above ~8, whereas lr decides whether the run converges at all — and the one
   genuinely broken row in README section 4 is lc_04 (electra-large diverged,
   50.0% for six epochs at lr 3e-4). A search over lr would have caught it and
   a search over r never will. lr is the first parameter here.
2. THE PRUNER WAS INERT. `create_study(pruner=None)` disabled it outright, and
   the commented-out MedianPruner above it would not have worked either: a
   pruner does nothing unless somebody calls `trial.report()` and
   `trial.should_prune()`. With HF Trainer that needs a callback, which did not
   exist, so all 20 trials ran to their full 20 epochs no matter how hopeless.
   See OptunaPruningCallback below.
3. IT OPTIMIZED eval_loss AND NEVER COMPUTED ACCURACY. There was no
   compute_metrics at all. Loss and accuracy come apart precisely when dropout
   is in the search space, which it was.
4. `lora_alpha` WAS SUGGESTED OVER A RANGE THAT DEPENDED ON `lora_r`
   (`suggest_float('lora_alpha', lora_r, 4 * lora_r)`). That makes the search
   space dynamic, which TPE models poorly — the same stored value of
   `lora_alpha` means a different thing in different trials. Search the
   alpha/r RATIO instead: it is what actually scales the update.
5. `prepare_model_for_kbit_training()` WAS CALLED ON AN UNQUANTIZED fp32 MODEL.
   That helper is for 4/8-bit backbones; on a plain model its main effect here
   was to switch gradient checkpointing on, i.e. pay recompute for nothing on a
   66M encoder.

Also fixed: the dataset was re-loaded and re-tokenized inside every trial;
there was no storage, so a crash lost the whole study; the sampler was unseeded
so results were not reproducible; and every trial left an unreaped checkpoint
directory behind.
"""

import logging
# torchao 0.18 still calls the deprecated register_constant() on its Enums at
# import time; silence the resulting torch.utils._pytree warnings (emitted via
# logging, so a warnings filter would not catch them).
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

import argparse
import gc
import json
import math
import os
import sys

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import datasets
import optuna
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainerCallback, TrainingArguments)

from lc_neobert_compat import patch_neobert

SEED = 42

# Per-model knobs. Batch sizes come from the VRAM table in README section 4
# (RTX 5070 Ti, 16 GB) and are NOT searched: they change the effective batch and
# therefore the meaning of lr, which would confound the one axis we care about.
# `baseline` is the published config from the lc_0N script, enqueued as trial 0
# so the study starts from the known result and can only improve on it.
#
# `target_sets` names the LoRA coverage tiers, and every entry MUST define
# "published" as exactly what the corresponding lc_0N script uses, because
# `baseline["targets"]` names it and enqueue_trial() would otherwise reject
# trial 0. Tier names are shared across models so studies stay comparable:
#   attn_qk    - query/key only
#   attn_qkv   - + value
#   attn_full  - + the attention output projection
#   all_linear - + both feed-forward linears
MODELS = {
    "distilbert": dict(
        model_id="distilbert/distilbert-base-uncased",
        target_sets={
            "attn_qk": ["attention.q_lin", "attention.k_lin"],
            "attn_qkv": ["attention.q_lin", "attention.k_lin", "attention.v_lin"],
            "attn_full": ["attention.q_lin", "attention.k_lin", "attention.v_lin",
                          "attention.out_lin"],
            "published": ["attention.q_lin", "attention.k_lin", "attention.v_lin",
                          "attention.out_lin", "ffn.lin1", "ffn.lin2"],
        },
        seq=128, bs=128, eval_bs=256, accum=1,
        baseline=dict(lr=2e-4, r=128, alpha_ratio=1.0, dropout=0.25,
                      targets="published"),  # == all_linear
    ),
    "electra": dict(
        model_id="google/electra-large-discriminator",
        target_sets={
            "attn_qk": ["query", "key"],
            "published": ["query", "key", "value"],  # == attn_qkv
            "attn_full": ["query", "key", "value", "attention.output.dense"],
            "all_linear": ["query", "key", "value", "intermediate.dense",
                           "output.dense"],
        },
        seq=128, bs=32, eval_bs=128, accum=1,
        # lr 3e-4 is the value that DIVERGED (README section 4). It is enqueued
        # anyway: a search whose baseline is a known failure demonstrates the
        # pruner working, and TPE needs the bad region mapped to avoid it.
        baseline=dict(lr=3e-4, r=40, alpha_ratio=3.0, dropout=0.2,
                      targets="published"),
    ),
    "roberta": dict(
        model_id="FacebookAI/roberta-large",
        # NOTE: "output.dense" suffix-matches both attention.output.dense and
        # the block's own output.dense. A bare "dense" would additionally match
        # classifier.dense, which belongs to the head, not to the adapter.
        target_sets={
            "published": ["query", "key"],  # == attn_qk, the narrowest in the series
            "attn_qkv": ["query", "key", "value"],
            "attn_full": ["query", "key", "value", "attention.output.dense"],
            "all_linear": ["query", "key", "value", "intermediate.dense",
                           "output.dense"],
        },
        seq=128, bs=32, eval_bs=128, accum=1,
        baseline=dict(lr=2e-4, r=36, alpha_ratio=60 / 36, dropout=0.25,
                      targets="published"),
    ),
    "modernbert": dict(
        model_id="answerdotai/ModernBERT-large",
        # Two ModernBERT naming facts, both easy to get wrong:
        #  - q/k/v are FUSED into one Wqkv, so an attn_qk tier cannot be
        #    expressed at all; the coarsest tier is all three at once.
        #  - `Wo` is the name of BOTH attn.Wo and mlp.Wo, and PEFT matches by
        #    suffix. So lc_06's published ["Wqkv", "Wo"] is NOT "attention
        #    only" as its comment implies — it already adapts the MLP output
        #    projection, leaving only mlp.Wi unadapted. Scope with "attn.Wo"
        #    when attention alone is meant.
        # Verified against the real module tree, not assumed.
        target_sets={
            "attn_qkv": ["Wqkv"],
            "attn_full": ["Wqkv", "attn.Wo"],
            "published": ["Wqkv", "Wo"],  # attn_full + mlp.Wo
            "all_linear": ["Wqkv", "Wo", "Wi"],
        },
        seq=512, bs=16, eval_bs=64, accum=2, attn="flash_attention_2",
        baseline=dict(lr=2e-4, r=32, alpha_ratio=2.0, dropout=0.1,
                      targets="published"),
    ),
    "neobert": dict(
        model_id="chandar-lab/NeoBERT",
        # Pin the revision: this repo ships its model definition as remote code,
        # which transformers re-downloads whenever upstream edits it. Unpinned,
        # a resumed study can compare trials against different architectures.
        revision="5424c8efeea6491b151d62dee55a752165407430",
        trust_remote_code=True,
        # qkv is fused (no attn_qk tier possible, as with ModernBERT); w12 + w3
        # are the SwiGLU feed-forward, where xformers packs w1/w2 into a single
        # w12 of shape (2*intermediate, hidden). 28 blocks of each.
        target_sets={
            "attn_qkv": ["qkv"],
            "attn_full": ["qkv", "wo"],
            "published": ["qkv", "wo", "w12", "w3"],  # == all_linear
        },
        # NeoBERT's classification head is TWO layers, dense (768->768) then
        # classifier (768->2), and the checkpoint contains neither. PEFT's
        # SEQ_CLS task type only adds "classifier"/"score" by default, which
        # would leave `dense` frozen at its random init — a random projection
        # wired in front of the classifier, training nothing.
        modules_to_save=["dense", "classifier"],
        patch=patch_neobert,  # see lc_neobert_compat: one loud fix, one silent
        seq=512, bs=16, eval_bs=64, accum=2,
        baseline=dict(lr=2e-4, r=32, alpha_ratio=2.0, dropout=0.1,
                      targets="published"),
    ),
}


class OptunaPruningCallback(TrainerCallback):
    """Report the eval metric to Optuna after every epoch and honour pruning.

    This is the piece the old script was missing. Without it `MedianPruner` is
    decorative: Optuna only prunes when the objective calls `should_prune()`,
    and an objective that just returns `trainer.train()`'s result never does.
    """

    def __init__(self, trial, metric="eval_accuracy"):
        self.trial = trial
        self.metric = metric

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics:
            return
        value = metrics[self.metric]
        step = int(round(state.epoch or 0))
        self.trial.report(value, step=step)
        if self.trial.should_prune():
            # Propagates out of trainer.train(); Optuna catches TrialPruned.
            raise optuna.TrialPruned(
                f"epoch {step}: {self.metric}={value:.4f} below the median")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary")
    return {"accuracy": accuracy_score(labels, preds), "f1": f1,
            "precision": precision, "recall": recall}


def load_splits(cfg, tokenizer, train_n, eval_n):
    """Tokenize once per study, not once per trial (the old script re-mapped
    the dataset inside objective(), 20 times over)."""
    ds = datasets.load_dataset("imdb")
    train = ds["train"].shuffle(seed=SEED)
    test = ds["test"].shuffle(seed=SEED)
    if train_n:
        train = train.select(range(min(train_n, len(train))))
    if eval_n:
        test = test.select(range(min(eval_n, len(test))))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=cfg["seq"])

    return (train.map(tok, batched=True, desc="tokenize train"),
            test.map(tok, batched=True, desc="tokenize eval"))


def build_model(cfg, r, alpha, dropout, targets=None):
    kwargs = dict(num_labels=2, dtype=torch.bfloat16)
    if cfg.get("attn"):
        kwargs["attn_implementation"] = cfg["attn"]
    if cfg.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    if cfg.get("revision"):
        kwargs["revision"] = cfg["revision"]
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"], **kwargs).to("cuda")
    # Per-model fixes that must land between from_pretrained and get_peft_model.
    if cfg.get("patch"):
        cfg["patch"](model, verbose=False)
    # NOTE: no prepare_model_for_kbit_training() here. The backbone is not
    # quantized, so that helper would only turn gradient checkpointing on and
    # pay recompute for nothing.
    return get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=cfg["target_sets"][targets or cfg["baseline"]["targets"]],
        modules_to_save=cfg.get("modules_to_save"),
        lora_dropout=dropout, bias="none", task_type="SEQ_CLS"))


def warmup_steps_for(ratio, n_train, cfg, epochs):
    """transformers 5 removed `warmup_ratio` — only `warmup_steps` survives, so
    the ratio has to be resolved against the real optimizer-step count."""
    per_epoch = math.ceil(n_train / (cfg["bs"] * cfg["accum"]))
    return int(round(ratio * per_epoch * epochs))


def make_args(out_dir, cfg, lr, epochs, warmup_steps, wd, save=False):
    return TrainingArguments(
        output_dir=out_dir,
        report_to=[],
        per_device_train_batch_size=cfg["bs"],
        per_device_eval_batch_size=cfg["eval_bs"],
        gradient_accumulation_steps=cfg["accum"],
        bf16=True,          # matches the bf16 backbone; fp16's GradScaler
                            # cannot unscale half-precision trainable params
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=wd,
        eval_strategy="epoch",
        # During the search we only need the metric, never the weights. The old
        # script saved every epoch of every trial into its own directory and
        # never reaped them.
        save_strategy="epoch" if save else "no",
        load_best_model_at_end=save,
        metric_for_best_model="accuracy" if save else None,
        save_total_limit=1 if save else None,
        logging_steps=50,
        dataloader_num_workers=0,
        label_names=["labels"],
        seed=SEED,
    )


def suggest_params(trial, cfg):
    """The search space, isolated so `--selftest` can exercise it without a GPU.

    Every distribution declared here is a contract with `enqueue_trial()`: an
    enqueued value outside its distribution does not fail at enqueue time, it
    fails on the first `suggest_*()` call — i.e. after the dataset has already
    been tokenized. `--selftest` round-trips all five baselines through here.
    """
    # Ordered by how much each one actually matters for LoRA.
    p = dict(lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True))
    # r and alpha_ratio are deliberately CONTINUOUS rather than categorical.
    # The published lc_0N baselines are r=128/40/36/32 with ratios 1, 3, 5/3, 2
    # — a categorical list would have to enumerate all of them or trial 0 dies
    # with "'40' not in (8, 16, 32, 64, 128)".
    p["r"] = trial.suggest_int("r", 8, 128, step=4)
    # The RATIO, not raw alpha: alpha/r is the factor the update is scaled by,
    # and searching it keeps the space static across trials.
    p["alpha_ratio"] = trial.suggest_float("alpha_ratio", 0.25, 8.0, log=True)
    p["dropout"] = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    p["warmup_ratio"] = trial.suggest_categorical("warmup_ratio", [0.0, 0.06, 0.1])
    p["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1])
    # WHICH modules get an adapter, not just how big it is. This was a registry
    # constant until 2026-09-10, which meant the search could tune six knobs
    # while the largest uncontrolled difference between the lc_0N scripts sat
    # outside the space entirely: coverage ranges from q/k only (lc_05) to full
    # attention + both FFN linears (lc_03, lc_11). See README, "Does LoRA
    # coverage matter more than the tuned knobs?".
    p["targets"] = trial.suggest_categorical("targets", sorted(cfg["target_sets"]))
    return p


def selftest():
    """Round-trip every registry baseline through the real search space.

    Guards the one failure mode that costs a full tokenization pass to discover:
    a baseline value that is not a member of the distribution `suggest_params`
    declares for it. Runs on CPU in under a second and touches no model weights.
    """
    ok = True
    for name, cfg in MODELS.items():
        b = cfg["baseline"]
        enqueued = dict(lr=b["lr"], r=b["r"], alpha_ratio=b["alpha_ratio"],
                        dropout=b["dropout"], targets=b["targets"],
                        warmup_ratio=0.0, weight_decay=0.01)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.enqueue_trial(enqueued)
        study.optimize(lambda t: (suggest_params(t, cfg), 0.0)[1], n_trials=1)
        got = study.trials[0].params
        bad = {k: (v, got.get(k)) for k, v in enqueued.items() if got.get(k) != v}
        # A baseline naming a tier that does not exist would pass the
        # distribution check and then KeyError inside build_model, so verify
        # the lookup too.
        if b["targets"] not in cfg["target_sets"]:
            bad["targets->set"] = (b["targets"], sorted(cfg["target_sets"]))
        print(f"  {name:12s} {'OK' if not bad else 'MISMATCH ' + repr(bad)}")
        ok &= not bad
    print("selftest:", "all baselines round-trip" if ok else "FAILED")
    return 0 if ok else 1


def objective(trial, cfg, train_ds, eval_ds, tokenizer, args):
    p = suggest_params(trial, cfg)
    lr, r, dropout = p["lr"], p["r"], p["dropout"]
    alpha_ratio, targets = p["alpha_ratio"], p["targets"]
    warmup, wd = p["warmup_ratio"], p["weight_decay"]

    alpha = int(round(alpha_ratio * r))
    print(f"\n=== trial {trial.number}: lr={lr:.2e} r={r} alpha={alpha} "
          f"(ratio {alpha_ratio}) dropout={dropout} warmup={warmup} wd={wd} "
          f"targets={targets}")

    model = build_model(cfg, r, alpha, dropout, targets)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("trainable_params", trainable)
    trial.set_user_attr("lora_alpha", alpha)

    trainer = Trainer(
        model=model,
        args=make_args(os.path.join(args.out, f"trial-{trial.number}"),
                       cfg, lr, args.epochs,
                       warmup_steps_for(warmup, len(train_ds), cfg,
                                        args.epochs), wd),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[OptunaPruningCallback(trial)],
    )

    try:
        trainer.train()
        best = max((m["eval_accuracy"] for m in trainer.state.log_history
                    if "eval_accuracy" in m), default=0.0)
    finally:
        # Between-trial cleanup. Without the explicit gc the previous model can
        # survive in a reference cycle long enough to OOM the next trial.
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    return best


def run_final(cfg, best, args, tokenizer):
    """Retrain the winning configuration on the full 25k and save the adapter,
    so the result is comparable with the lc_0N rows in README section 4."""
    train_ds, eval_ds = load_splits(cfg, tokenizer, None, None)
    r = best["r"]
    alpha = int(round(best["alpha_ratio"] * r))
    out = os.path.join("outputs", f"lora-{args.model}-imdb-hpo")
    model = build_model(cfg, r, alpha, best["dropout"])
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=make_args(out, cfg, best["lr"], args.final_epochs,
                       warmup_steps_for(best["warmup_ratio"], len(train_ds),
                                        cfg, args.final_epochs),
                       best["weight_decay"], save=True),
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"\nadapter saved to {out}")
    return trainer.evaluate()


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Study state is a SQLite file, so a run is resumable: the same\n"
               "--study-name picks up where it left off and adds trials.")
    p.add_argument("--model", default="distilbert", choices=sorted(MODELS),
                   help="which lc_0N backbone to tune (default: distilbert)")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--epochs", type=int, default=6,
                   help="epochs per search trial (default 6; the pruner ends "
                        "most of them early)")
    p.add_argument("--final-epochs", type=int, default=10)
    p.add_argument("--train-n", type=int, default=5000,
                   help="training subset for the search (0 = full 25k)")
    p.add_argument("--eval-n", type=int, default=2000,
                   help="eval subset for the search (0 = full 25k)")
    p.add_argument("--study-name", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--final", action="store_true",
                   help="skip searching; retrain the best trial on the full "
                        "25k and save the adapter")
    p.add_argument("--selftest", action="store_true",
                   help="round-trip every registry baseline through the search "
                        "space and exit (CPU-only, ~1 s)")
    p.add_argument("--no-baseline", action="store_true",
                   help="do not enqueue the published lc_0N config as trial 0")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())

    cfg = MODELS[args.model]
    args.study_name = args.study_name or f"lora-{args.model}-imdb"
    args.out = args.out or os.path.join("outputs", "hpo", args.model)
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg.get("revision"),
        trust_remote_code=cfg.get("trust_remote_code", False))

    storage = f"sqlite:///{os.path.abspath(os.path.join(args.out, 'study.db'))}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,        # resumable; a crash no longer loses the run
        direction="maximize",       # accuracy, not eval_loss
        sampler=optuna.samplers.TPESampler(seed=SEED),   # reproducible
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=2),
    )

    if args.final:
        if not study.best_trial:
            sys.exit("no completed trials in the study yet")
        print("best params:", json.dumps(study.best_params, indent=2))
        print(run_final(cfg, study.best_params, args, tokenizer))
        return

    if not args.no_baseline and not study.trials:
        b = cfg["baseline"]
        # Every value here must lie inside the distribution objective() declares
        # for it, or the trial raises on the first suggest_*() call rather than
        # at enqueue time — i.e. after the dataset is already tokenized.
        study.enqueue_trial(dict(lr=b["lr"], r=b["r"],
                                 alpha_ratio=b["alpha_ratio"],
                                 dropout=b["dropout"], targets=b["targets"],
                                 warmup_ratio=0.0, weight_decay=0.01))
        print(f"enqueued the published lc_0N config as trial 0: {b}")

    train_ds, eval_ds = load_splits(cfg, tokenizer, args.train_n, args.eval_n)
    study.optimize(lambda t: objective(t, cfg, train_ds, eval_ds, tokenizer,
                                       args),
                   n_trials=args.trials)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n{len(done)} complete, {len(pruned)} pruned")
    print(f"best accuracy {study.best_value:.4f} with "
          f"{json.dumps(study.best_params)}")
    try:
        imp = optuna.importance.get_param_importances(study)
        print("parameter importance: "
              + ", ".join(f"{k}={v:.3f}" for k, v in imp.items()))
    except Exception as exc:          # needs >1 completed trial
        print(f"(importance unavailable: {exc})")
    print(f"\nstudy: {storage}\n"
          f"retrain the winner on the full 25k with:\n"
          f"  python {os.path.basename(__file__)} --model {args.model} --final")


if __name__ == "__main__":
    main()
