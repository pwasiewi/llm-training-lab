"""
LESSON 1 — Pretraining a tiny GPT from scratch (decoder-only causal LM).

This is the core "what is an LLM" script. Every large language model —
GPT, Llama, Qwen, the GGUFs served by aillama — is at heart exactly this
pipeline, only scaled up ~10^5 times:

    raw text -> tokenizer -> token ids -> transformer -> next-token logits
             -> cross-entropy loss on the SHIFTED sequence -> AdamW step

What this script teaches:
  1. BPE tokenization: text becomes integers from a learned vocabulary.
  2. The causal mask: position i may only attend to positions <= i.
     This single triangular matrix is what makes a transformer a
     *language model* instead of an encoder.
  3. Next-token prediction: input is tokens[:-1], target is tokens[1:].
     There is no other label — the text supervises itself.
  4. Perplexity = exp(cross-entropy): "how many tokens is the model
     effectively choosing between". Vocab 5000 => untrained PPL ~5000;
     watch it collapse during epoch 1.
  5. Sampling: why greedy argmax degenerates into loops, and how
     temperature + top-k trade coherence against diversity.

Run:  python simple_01_gpt_tinystories.py
Data: TinyStories (10k stories), model ~8M params — trains in minutes on GPU.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class SimplifiedConfig:
    """
    All hyperparameters in one place. For scale: GPT-2 small is
    vocab 50257 / hidden 768 / 12 heads / 12 layers / ctx 1024 (~124M params);
    this config is the same shape, just miniature.
    """

    def __init__(self):
        self.vocab_size = 5000      # BPE merges; tiny corpus -> tiny vocab suffices
        self.hidden_size = 256      # embedding dim = width of every residual stream
        self.num_heads = 4          # 256/4 = 64 dims per head (64 is the classic head size)
        self.num_layers = 4         # depth: each block refines the token representations
        self.max_seq_length = 256   # context window; attention cost grows with its SQUARE
        self.dropout = 0.1
        self.batch_size = 32
        self.learning_rate = 3e-4   # "the best LR for Adam" — Karpathy's constant
        self.num_epochs = 3
        self.tie_weights = True     # share input embedding with output head (see model)


class TextDataset(Dataset):
    """
    Turns each story into a fixed-length id sequence.

    The one idea that matters here is in __getitem__: the *input* is the
    sequence without its last token and the *target* is the same sequence
    shifted left by one. Every position simultaneously predicts its next
    token — one forward pass yields seq_length training signals, not one.

    Note: tensors are pre-built on the GPU (fast for a dataset this small,
    but it holds everything in VRAM — real pipelines stream from disk and
    pin/copy per batch instead).
    """

    def __init__(self, texts, tokenizer, max_length, device):
        pad_id = tokenizer.token_to_id("[PAD]")
        self.encodings = []
        for text in texts:
            # [START]/[END] teach the model where a document begins and ends,
            # which is also how generation knows when to stop.
            encoded = tokenizer.encode("[START] " + text + " [END]")
            ids = encoded.ids[:max_length]
            ids = ids + [pad_id] * (max_length - len(ids))
            self.encodings.append(torch.tensor(ids, device=device))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        return item[:-1], item[1:]  # input, target shifted by one


class MultiHeadAttention(nn.Module):
    """
    Attention from scratch: softmax(Q K^T / sqrt(d)) V, per head.

    Q ("what am I looking for"), K ("what do I contain"), V ("what do I
    give if attended to") are three linear views of the same input.
    Multiple heads let the model look for several relations at once
    (e.g. one head tracks subjects, another recent punctuation).

    Production code calls F.scaled_dot_product_attention (FlashAttention)
    which fuses these steps and never materializes the seq x seq matrix;
    the explicit version below is the reference semantics.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        # One fused projection producing Q, K and V at once (3x width).
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        # Scaling by sqrt(head_size) keeps dot products O(1) so softmax
        # doesn't saturate as dimensions grow.
        self.scale = math.sqrt(self.head_size)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (batch, heads, seq, seq): row i = how much position i attends to each j.
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # The causal mask sets future positions to -inf BEFORE softmax,
        # so they get exactly zero attention weight. Without this the model
        # would "cheat" by reading the token it is supposed to predict.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)

        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch_size, seq_length, self.hidden_size)  # merge heads back
        x = self.output(x)
        return x


class FeedForward(nn.Module):
    """
    Position-wise MLP: expand 4x, non-linearity, project back.
    Attention MOVES information between positions; the FFN TRANSFORMS it
    at each position — this is where most of a transformer's parameters
    (and, per interpretability work, most of its stored facts) live.
    """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm block: x = x + Attn(LN(x)); x = x + FFN(LN(x)).

    The residual "x +" is what lets gradients flow through many layers —
    each block only learns a small CORRECTION to the stream, never has to
    re-encode everything. Pre-norm (normalize before the sublayer, as in
    GPT-2 and everything since) trains stably without warmup tricks;
    the original post-norm formulation did not.
    """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.layer_norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.layer_norm2(x)))
        return x


class SimpleTransformer(nn.Module):
    """
    The full decoder-only LM: embeddings -> N blocks -> LM head.
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # Token embedding: a learned lookup table id -> vector.
        # Position embedding: learned vector per position, ADDED to the token
        # vector — without it, attention is a set operation and "dog bites man"
        # equals "man bites dog". (Lesson 2 shows the modern alternative: RoPE.)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # LM head: hidden vector -> a logit for every vocabulary word.
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

        # Weight tying: the output head reuses the embedding matrix
        # (both map between "token identity" and "meaning vector", just in
        # opposite directions). Saves vocab*hidden params — here 1.3M of ~8M;
        # GPT-2 and most modern LMs do this.
        if config.tie_weights:
            self.output.weight = self.token_embedding.weight

        # Init scale matters, and tying makes it bite: nn.Embedding defaults
        # to N(0,1), so the tied LM head starts with huge weights -> logits
        # with std ~sqrt(hidden) -> first-batch loss ~160 instead of the
        # ln(vocab)=8.5 a uniform guesser would score. GPT-2's std=0.02
        # keeps the untrained model honestly clueless instead of confidently
        # wrong (confident-and-wrong is much harder to descend from).
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, x):
        batch_size, seq_length = x.size()

        # Lower-triangular causal mask, built on the fly (cheap; real code caches it).
        mask = torch.triu(torch.ones((seq_length, seq_length), device=self.device), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)  # True = "may attend"

        positions = torch.arange(seq_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        x = self.layer_norm(x)
        x = self.output(x)  # (batch, seq, vocab) — a next-token distribution at EVERY position
        return x


def create_tokenizer(texts):
    """
    Train a BPE tokenizer on the corpus itself. BPE starts from characters
    and greedily merges the most frequent pairs — frequent words become one
    token, rare words split into pieces, and NOTHING is ever out-of-vocabulary.
    Real models ship a fixed pretrained tokenizer; training our own here
    shows it is just corpus statistics, not magic.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[START]", "[END]", "[UNK]"],
        vocab_size=5000
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def generate_text(model, tokenizer, device, max_length=250, prompt="[START]",
                  temperature=0.8, top_k=50):
    """
    Autoregressive generation: feed the sequence, take the LAST position's
    logits, pick one token, append, repeat. This loop IS "inference" — the
    only difference vs llama.cpp is KV-caching and batching.

    Sampling knobs (the same ones aillama/llama-server expose):
      temperature=0  -> greedy argmax. Deterministic, and on small models it
                        quickly collapses into repetition loops, because the
                        single most likely continuation of a repeated phrase
                        is... the phrase again.
      temperature>0  -> divide logits by T before softmax; T<1 sharpens,
                        T>1 flattens toward uniform (word salad).
      top_k          -> zero out everything but the k best tokens first, so
                        the flat tail of the distribution can't be sampled.
    """
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
    end_id = tokenizer.token_to_id("[END]")

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]  # only the last position predicts

            if temperature <= 0:
                next_token = torch.argmax(next_token_logits).item()
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    kth_best = torch.topk(next_token_logits, top_k).values[-1]
                    next_token_logits[next_token_logits < kth_best] = float('-inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == end_id:
                break

            next_token_tensor = torch.tensor([[next_token]], device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    return tokenizer.decode(input_ids[0].cpu().tolist())


def train_model(config, model, train_loader, optimizer, device, pad_id):
    """
    One epoch of the standard LM loop. Watch two numbers in the progress bar:
      loss — mean cross-entropy per (non-pad) token
      ppl  — exp(loss), "effective branching factor"; starts near vocab size,
             a well-trained tiny model on TinyStories lands in the low tens.
    """
    model.train()
    total_loss = 0
    tokens_seen = 0
    start = time.time()
    progress_bar = tqdm(train_loader, desc="Training")

    for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
        optimizer.zero_grad()
        outputs = model(input_ids)

        # Flatten (batch, seq, vocab) -> (batch*seq, vocab): every position is
        # an independent classification over the vocabulary.
        outputs = outputs.view(-1, outputs.size(-1))
        target_ids = target_ids.view(-1)

        # ignore_index: padding positions contribute NO loss. Without this the
        # model wastes capacity learning "after [END] comes [PAD] forever" and
        # the reported loss is diluted by trivially predictable pad tokens.
        loss = F.cross_entropy(outputs, target_ids, ignore_index=pad_id)
        loss.backward()

        # Clip the global grad norm — cheap insurance against the occasional
        # exploding batch knocking the weights into a bad region.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        tokens_seen += (target_ids != pad_id).sum().item()
        avg = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            "loss": f"{avg:.3f}",
            "ppl": f"{math.exp(min(avg, 20)):.1f}",
            "tok/s": f"{tokens_seen / (time.time() - start):.0f}",
        })

    return total_loss / len(train_loader)


def main():
    config = SimplifiedConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset (TinyStories: short synthetic children's stories)...")
    dataset = load_dataset("roneneldan/TinyStories")
    train_texts = dataset["train"]["text"][:10000]

    print("Training BPE tokenizer on the corpus...")
    tokenizer = create_tokenizer(train_texts)
    pad_id = tokenizer.token_to_id("[PAD]")

    print("Preparing dataset (tokenize + pad, tensors kept on device)...")
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_length, device)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    print("Initializing model...")
    model = SimpleTransformer(config, device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = model.token_embedding.weight.numel()
    print(f"Parameters: {n_params/1e6:.2f}M total "
          f"({n_emb/1e6:.2f}M in token embeddings"
          f"{', tied with LM head' if config.tie_weights else ''})")
    print(f"Untrained perplexity should be ~vocab_size = {config.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("Starting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        train_loss = train_model(config, model, train_loader, optimizer, device, pad_id)
        print(f"Average loss: {train_loss:.4f}  (perplexity {math.exp(train_loss):.1f})")

        # Same model, two decoding strategies — compare the failure modes:
        print("\n--- greedy (temperature=0): deterministic, prone to loops ---")
        print(generate_text(model, tokenizer, device, temperature=0))
        print("\n--- sampled (temperature=0.8, top_k=50): varied, may drift ---")
        print(generate_text(model, tokenizer, device, temperature=0.8, top_k=50))

    print("\nTraining completed!")
    print("Things to try: temperature 1.5 (word salad), top_k=1 (== greedy),")
    print("num_layers=1 (watch coherence drop), tie_weights=False (param count).")


if __name__ == "__main__":
    main()
