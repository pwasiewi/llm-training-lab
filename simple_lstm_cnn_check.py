import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class ModelConfig:
    """
    Configuration for hybrid LSTM-CNN model.
    """

    def __init__(self):
        self.vocab_size = 50000
        self.embedding_dim = 800
        self.hidden_dim = 512
        self.n_filters = 100
        self.filter_sizes = (3, 4, 5)
        self.dropout = 0.3
        self.max_seq_length = 512
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.seed = 42
        self.train_size = 2000
        self.test_size = 500
        self.val_size = 200

class HybridLSTMCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # LSTM component
        self.lstm = nn.LSTM(config.embedding_dim,
                            config.hidden_dim,
                            bidirectional=True,
                            batch_first=True)

        # CNN component
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.n_filters, (fs, config.hidden_dim * 2))
            for fs in config.filter_sizes
        ])

        # Output layers
        self.fc = nn.Linear(len(config.filter_sizes) * config.n_filters, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.dropout(self.embedding(x))

        # LSTM processing
        # lstm_out: (batch_size, seq_length, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)

        # CNN processing on LSTM outputs
        # Add channel dimension for CNN
        lstm_out = lstm_out.unsqueeze(1)

        # Apply each convolution layer
        conved = [F.relu(conv(lstm_out)).squeeze(3) for conv in self.convs]

        # Max pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # Concatenate and apply dropout
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat).squeeze(-1)


class SentimentDataset(Dataset):
    """
    Dataset for sentiment analysis, converts text to tokens and handles labels.
    """

    def __init__(self, texts, labels, tokenizer, max_length, device):
        self.encodings = []
        self.labels = []

        for text, label in zip(texts, labels):
            encoded = tokenizer.encode(text)
            ids = encoded.ids[:max_length] if len(encoded.ids) > max_length else encoded.ids
            padding_length = max_length - len(ids)
            if padding_length > 0:
                ids = ids + [tokenizer.token_to_id("[PAD]")] * padding_length

            self.encodings.append(torch.tensor(ids, device=device))
            self.labels.append(torch.tensor(label, device=device))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

def create_tokenizer(texts):
    """
    Creates and trains a BPE tokenizer on the provided texts.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[UNK]"],
        vocab_size=30000
    )
    tokenizer.pre_tokenizer = Whitespace()
    texts_list = list(texts)
    tokenizer.train_from_iterator(texts_list, trainer)
    return tokenizer

def analyze_sequence_lengths():
    """Analyze sequence lengths in the IMDB dataset"""
    dataset = load_dataset("imdb")
    train_texts = dataset["train"]["text"]

    # Calculate lengths in words
    lengths = [len(text.split()) for text in train_texts]

    stats = {
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "p95": np.percentile(lengths, 95),
        "p99": np.percentile(lengths, 99),
        "max": max(lengths),
        "min": min(lengths),
        "std": np.std(lengths)
    }

    return lengths, stats


def plot_length_distribution(lengths):
    """Plot the distribution of sequence lengths"""
    plt.figure(figsize=(12, 6))

    # Main distribution plot
    sns.histplot(lengths, bins=50)
    plt.title("Distribution of IMDB Review Lengths")
    plt.xlabel("Length (words)")
    plt.ylabel("Count")

    # Add vertical lines for key statistics
    plt.axvline(np.mean(lengths), color='r', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
    plt.axvline(np.median(lengths), color='g', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
    plt.axvline(np.percentile(lengths, 95), color='b', linestyle='--',
                label=f'95th percentile: {np.percentile(lengths, 95):.0f}')

    plt.legend()
    plt.tight_layout()
    plt.savefig('length_distribution.png')
    plt.close()


def run_ablation_experiment(model_class, config, sequence_lengths, device, num_samples=1000):
    """Run ablation study for different sequence lengths"""
    results = defaultdict(list)
    dataset = load_dataset("imdb")

    # Select a subset of data for quick testing
    train_data = dataset["train"].shuffle(seed=42).select(range(num_samples))

    for max_length in tqdm(sequence_lengths, desc="Testing sequence lengths"):
        # Update config
        config.max_seq_length = max_length

        # Initialize model and datasets
        model = model_class(config).to(device)
        tokenizer = create_tokenizer(train_data["text"])
        train_dataset = SentimentDataset(
            train_data["text"],
            train_data["label"],
            tokenizer,
            max_length,
            device
        )
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

        # Measure memory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Training time for one batch
        start_time = time.time()
        batch = next(iter(train_loader))
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        input_ids, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

        batch_time = time.time() - start_time

        # Memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        else:
            memory_usage = 0

        results["sequence_length"].append(max_length)
        results["batch_time"].append(batch_time)
        results["memory_usage"].append(memory_usage)
        results["loss"].append(loss.item())

        # Clean up
        del model
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


def plot_ablation_results(results):
    """Plot the results of the ablation study"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Time plot
    sns.lineplot(data=results, x="sequence_length", y="batch_time", ax=axes[0])
    axes[0].set_title("Training Time per Batch vs Sequence Length")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Time (seconds)")

    # Memory plot
    sns.lineplot(data=results, x="sequence_length", y="memory_usage", ax=axes[1])
    axes[1].set_title("Memory Usage vs Sequence Length")
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Memory Usage (MB)")

    plt.tight_layout()
    plt.savefig('ablation_results.png')
    plt.close()


def main():
    # Analyze dataset
    lengths, stats = analyze_sequence_lengths()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

    # Plot length distribution
    plot_length_distribution(lengths)

    # Setup for ablation study
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()  # Your existing config class

    # Test different sequence lengths
    sequence_lengths = [128, 256, 384, 512, 768, 1024]

    print("\nRunning ablation study...")
    results = run_ablation_experiment(
        HybridLSTMCNN,  # Your model class
        config,
        sequence_lengths,
        device
    )

    # Plot results
    plot_ablation_results(results)

    print("\nAblation Results:")
    print(results.to_string(index=False))

    # Recommendation
    optimal_length = results.loc[
        results["batch_time"] * results["memory_usage"].max() / results["memory_usage"] <
        results["batch_time"].max()
        ]["sequence_length"].iloc[0]

    print(f"\nRecommended sequence length based on time/memory trade-off: {optimal_length}")


if __name__ == "__main__":
    main()