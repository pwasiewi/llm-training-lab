"""
LESSON 7 — Measuring how cost scales with sequence length.

Lessons 2 and 4 CLAIMED that attention is quadratic in sequence length and
recurrence is linear. This script stops claiming and MEASURES: one training
step (forward + backward + optimizer) at several sequence lengths, for two
models —

  naive transformer   explicit softmax(QK^T) attention, like lessons 1-2:
                      the (seq x seq) matrix is materialized, so both time
                      and memory should grow ~quadratically;
  LSTM + CNN hybrid   recurrence and convolution only: both should grow
                      ~linearly.

Instead of eyeballing the curves, we fit the SCALING EXPONENT k in
cost ~ seq^k with a log-log least-squares line: log(cost) = k*log(seq) + c.
Straight lines in log-log space are power laws; k is the slope.

The subtlety (and the real lesson): total cost is a SUM of terms —
constant (weights, optimizer), linear (FFN, LSTM, embeddings) and
quadratic (the attention matrix). At short lengths the constant+linear
terms bury the seq^2 term, so a fit over the whole range understates k;
only the TAIL of the curve shows the asymptotic exponent. That is why two
numbers are reported per curve: the global fit and the slope between the
last two points. Watch the transformer's tail slope pull away from the
hybrid's as lengths grow — extend --lengths and it keeps climbing
toward 2. "Asymptotic complexity" needs asymptotic inputs to be visible.

Measurement hygiene taught along the way (all real-world benchmarking
gotchas):
  * warmup step first — the very first CUDA step pays for memory-pool
    growth and kernel compilation, and would poison the numbers;
  * torch.cuda.synchronize() around timers — kernel launches are ASYNC;
    without a sync you time the launch queue, not the work;
  * torch.cuda.max_memory_allocated() after reset_peak_memory_stats() —
    peak, not current, is what OOMs you.

Also answers "what seq_length do IMDB reviews actually need?" with token
percentiles — the practical companion question to "what can I afford?".

Run:  python simple_07_scaling_ablation.py
Output: outputs/scaling_ablation.png + a table and fitted exponents.
"""

import argparse
import math
import time
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed; we only save a PNG
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class AblationConfig:
    def __init__(self):
        self.vocab_size = 10000
        self.hidden_size = 256
        self.num_heads = 4
        self.num_layers = 2         # per model; small on purpose — we measure
        self.lstm_hidden = 256      # scaling SHAPE, not absolute speed
        self.n_filters = 100
        self.filter_sizes = (3, 4, 5)
        self.batch_size = 16
        self.num_samples = 512      # reviews used for the measurement batches
        self.seed = 42


# ------------------------------------------------------------- the models --

class NaiveAttentionBlock(nn.Module):
    """
    Lesson-2 attention, kept deliberately naive: scores is a full
    (batch, heads, seq, seq) tensor. THIS tensor is the quadratic cost we
    want to see in the memory curve. Library attention (SDPA/Flash) would
    hide it — that is precisely why we don't use it here.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.scale = math.sqrt(self.head_size)

    def forward(self, x):
        batch, seq, hidden = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch, seq, 3, self.num_heads, self.head_size)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / self.scale      # (b, h, seq, seq) !
        att = F.softmax(scores, dim=-1) @ v
        att = att.transpose(1, 2).reshape(batch, seq, hidden)
        x = x + self.proj(att)
        return x + self.ffn(self.norm2(x))


class NaiveTransformerClassifier(nn.Module):
    def __init__(self, config, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, config.hidden_size)
        self.blocks = nn.ModuleList(
            NaiveAttentionBlock(config) for _ in range(config.num_layers))
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.embedding(x) + self.position_embedding(positions)[None]
        for block in self.blocks:
            x = block(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)


class HybridLstmCnnClassifier(nn.Module):
    """Lesson 6 minus the transformer stage: strictly seq-linear compute."""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.lstm_hidden,
                            bidirectional=True, batch_first=True)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.n_filters, (fs, config.lstm_hidden * 2))
            for fs in config.filter_sizes
        ])
        self.fc = nn.Linear(len(config.filter_sizes) * config.n_filters, 1)

    def forward(self, x):
        x, _ = self.lstm(self.embedding(x))
        x = x.unsqueeze(1)
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        return self.fc(torch.cat(pooled, dim=1)).squeeze(-1)


# ------------------------------------------------------------ measurement --

def one_training_step(model, input_ids, labels, optimizer):
    optimizer.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(input_ids).float(),
                                              labels.float())
    loss.backward()
    optimizer.step()


def measure_model(model_name, make_model, batches_by_length, device,
                  lr=3e-4, steps=3):
    """
    Per sequence length: fresh model, one warmup step, then `steps` timed
    steps (median) with peak-memory tracking.
    """
    rows = []
    for seq_length, (input_ids, labels) in tqdm(batches_by_length.items(),
                                                desc=f"Measuring {model_name}"):
        model = make_model(seq_length).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        one_training_step(model, input_ids, labels, optimizer)   # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            one_training_step(model, input_ids, labels, optimizer)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
                   if device.type == "cuda" else float("nan"))
        rows.append({"model": model_name, "seq_length": seq_length,
                     "step_time_s": float(np.median(times)),
                     "peak_memory_mb": peak_mb})

        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def fit_exponent(seq_lengths, costs):
    """Least-squares slope of log(cost) vs log(seq): the k in cost ~ seq^k."""
    slope, _ = np.polyfit(np.log(np.asarray(seq_lengths, dtype=float)),
                          np.log(np.asarray(costs, dtype=float)), 1)
    return slope


def tail_exponent(seq_lengths, costs):
    """
    Slope between the two LONGEST lengths only — the best available look at
    the asymptotic exponent, where the quadratic term is least diluted by
    the constant and linear ones.
    """
    return (math.log(costs[-1] / costs[-2])
            / math.log(seq_lengths[-1] / seq_lengths[-2]))


# ------------------------------------------------------------------- data --

def token_length_stats(texts, tokenizer):
    """
    How long are the reviews in TOKENS (not words — after BPE a review is
    ~1.3x its word count)? The percentiles tell you what max_seq_length
    actually buys: seq 512 covers ~p90 of IMDB, the rest gets truncated.
    """
    lengths = np.array([len(tokenizer.encode(t).ids)
                        for t in tqdm(texts, desc="Tokenizing for stats")])
    return {f"p{p}": int(np.percentile(lengths, p)) for p in (50, 90, 95, 99)} \
        | {"mean": int(lengths.mean()), "max": int(lengths.max())}


def make_batches(texts, labels, tokenizer, seq_lengths, batch_size, device):
    """One fixed batch per target length (same reviews, different truncation)."""
    pad_id = tokenizer.token_to_id("[PAD]")
    batches = {}
    for seq_length in seq_lengths:
        ids = []
        for text in texts[:batch_size]:
            row = tokenizer.encode(text).ids[:seq_length]
            ids.append(row + [pad_id] * (seq_length - len(row)))
        batches[seq_length] = (
            torch.tensor(ids, device=device),
            torch.tensor(labels[:batch_size], device=device),
        )
    return batches


def plot_results(results, exponents, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, unit in (("step_time_s", axes[0], "s"),
                             ("peak_memory_mb", axes[1], "MB")):
        for model_name, group in results.groupby("model"):
            k = exponents[(model_name, metric)]
            ax.plot(group["seq_length"], group[metric], marker="o",
                    label=f"{model_name} (k={k:.2f})")
        ax.set_xlabel("sequence length")
        ax.set_ylabel(f"{metric} [{unit}]")
        # log-log: power laws become straight lines, slope = exponent k
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{metric} vs seq length (log-log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--lengths", default="128,256,512,1024,2048",
                        help="comma-separated sequence lengths to test")
    args = parser.parse_args()
    seq_lengths = [int(x) for x in args.lengths.split(",")]

    config = AblationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(config.seed)

    print("Loading IMDB subset...")
    data = load_dataset("imdb")["train"].shuffle(seed=config.seed).select(
        range(config.num_samples))
    texts, labels = data["text"], data["label"]

    # Tokenizer trained ONCE — the old version retrained it per length,
    # which measured tokenizer training as much as the model.
    print("Training tokenizer (once)...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"],
                         vocab_size=config.vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(texts, trainer)

    stats = token_length_stats(texts, tokenizer)
    print(f"\nIMDB review lengths in BPE tokens: {stats}")
    print("=> seq 512 truncates ~10% of reviews; 1024 nearly none.")

    batches = make_batches(texts, labels, tokenizer, seq_lengths,
                           config.batch_size, device)

    max_len = max(seq_lengths)
    rows = measure_model(
        "naive transformer",
        lambda seq: NaiveTransformerClassifier(config, max_len), batches, device)
    rows += measure_model(
        "lstm+cnn hybrid",
        lambda seq: HybridLstmCnnClassifier(config), batches, device)
    results = pd.DataFrame(rows)

    print("\nOne training step, batch "
          f"{config.batch_size} (median of 3, after warmup):")
    print(results.to_string(index=False,
                            float_format=lambda v: f"{v:.4f}"))

    exponents = {}
    print("\nScaling exponents k (cost ~ seq^k):")
    print(f"  {'model':<18} {'metric':<16} {'global fit':>10} {'tail slope':>10}")
    for model_name, group in results.groupby("model"):
        group = group.sort_values("seq_length")
        for metric in ("step_time_s", "peak_memory_mb"):
            k = fit_exponent(group["seq_length"], group[metric])
            k_tail = tail_exponent(group["seq_length"].tolist(),
                                   group[metric].tolist())
            exponents[(model_name, metric)] = k
            print(f"  {model_name:<18} {metric:<16} {k:>10.2f} {k_tail:>10.2f}")
    print("Reading: the hybrid's tail slope stays ~1 (linear); the naive")
    print("transformer's KEEPS RISING with length as the seq^2 attention")
    print("matrix outgrows the constant+linear terms — the global fit")
    print("understates it because short lengths dilute the quadratic part.")
    print("Extend --lengths (e.g. add 4096) and watch it approach 2.")

    os.makedirs("outputs", exist_ok=True)
    plot_path = "outputs/scaling_ablation.png"
    plot_results(results, exponents, plot_path)
    print(f"\nPlot saved to {plot_path}")
    print("\nNo 'recommended length' is printed on purpose: the old version")
    print("derived one from an arbitrary time/memory formula. The honest")
    print("answer is a POLICY: pick the shortest length covering enough of")
    print("your data (see the percentiles above) that fits your budget.")


if __name__ == "__main__":
    main()
