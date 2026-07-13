import optuna
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
import torch
import os
from optuna.visualization import plot_optimization_history, plot_param_importances
import plotly.graph_objects as go
import numpy as np

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("TENSORBOARD_LOGGING_DIR", "outputs/logs")

model_name = "distilbert/distilbert-base-uncased"
dir_name = "outputs/optuna-distilbert"
SEED = 42

def objective(trial):
    # Configure all parameters to be optimized
    lora_r = trial.suggest_int('lora_r', 4, 64, step=4)
    # Make alpha proportional to r but allow some variation
    lora_alpha = trial.suggest_float('lora_alpha', lora_r, 4 * lora_r, step=4.0)
    lora_dropout = trial.suggest_float('lora_dropout', 0.1, 0.5, step=0.05)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTrial {trial.number}:")
    print(f"Testing parameters: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Alpha/r ratio: {lora_alpha / lora_r:.2f}")

    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure LoRA with trial parameters
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin", "ffn.lin1", "ffn.lin2"],  # Warstwy docelowe
        lora_dropout=lora_dropout,
        bias="lora_only",
        task_type="SEQ_CLS"
    )

    # Prepare model
    model = prepare_model_for_kbit_training(
        model,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_model = get_peft_model(model, lora_config)

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Load and preprocess data
    dataset = datasets.load_dataset("imdb")
    train_data = dataset["train"].shuffle(seed=SEED).select(range(5000))
    test_data = dataset["test"].shuffle(seed=SEED).select(range(1000))

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=128)

    train_dataset = train_data.map(preprocess_function, batched=True)
    test_dataset = test_data.map(preprocess_function, batched=True)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=f"{dir_name}-{trial.number}",
        report_to=[],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        fp16=True,
        num_train_epochs=20,
        #weight_decay=0.01,
        #max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=1e-4,
        load_best_model_at_end=True,
        save_total_limit=1,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    try:
        trainer.train()
        eval_result = trainer.evaluate()

        # Store additional trial information
        trial.set_user_attr('trainable_params', trainable_params)
        trial.set_user_attr('alpha_r_ratio', lora_alpha / lora_r)

        loss = eval_result["eval_loss"]
    except Exception as e:
        print(f"Trial failed: {e}")
        loss = float('inf')
    finally:
        # Clean up
        del trainer
        del model
        del lora_model
        torch.cuda.empty_cache()

    return loss


def run_optimization(n_trials=20):
    # study = optuna.create_study(
    #     direction="minimize",
    #     study_name="lora_multi_parameter_optimization",
    #     pruner=optuna.pruners.MedianPruner(),
    #     sampler=optuna.samplers.TPESampler(seed=SEED)
    # )
    study = optuna.create_study(direction="minimize", pruner=None)
    study.optimize(objective, n_trials=n_trials)

    # Print results
    print("\nOptimization Results:")
    print("Best trial:")
    print(f"  Value (Loss): {study.best_trial.value:.4f}")
    print(f"  Best parameters:")
    for param, value in study.best_trial.params.items():
        print(f"    {param}: {value}")
    print(f"  Alpha/r ratio: {study.best_trial.params['lora_alpha'] / study.best_trial.params['lora_r']:.2f}")
    print(f"  Trainable parameters: {study.best_trial.user_attrs['trainable_params']:,}")

    # Create results visualizations
    try:
        # Parameter importance plot
        param_importance_fig = plot_param_importances(study)
        param_importance_fig.write_html("lora_param_importance.html")

        # Optimization history plot
        history_fig = plot_optimization_history(study)
        history_fig.write_html("lora_optimization_history.html")

        # Create parallel coordinates plot
        param_names = list(study.best_trial.params.keys())
        fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=[t.value for t in study.trials],
                      colorscale='Viridis'),
            dimensions=[
                dict(range=[min(t.params[param] for t in study.trials),
                            max(t.params[param] for t in study.trials)],
                     label=param,
                     values=[t.params[param] for t in study.trials])
                for param in param_names
            ]
        )
        )
        fig.write_html("lora_parallel_coordinates.html")

    except Exception as e:
        print(f"Warning: Could not generate some visualizations: {e}")

    return study.best_trial.params


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    best_params = run_optimization(20)
    print("\nOptimization complete. Best parameters found:")
    print(f"LoRA rank (r): {best_params['lora_r']}")
    print(f"LoRA alpha: {best_params['lora_alpha']}")
    print(f"LoRA dropout: {best_params['lora_dropout']}")
