import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class SimplifiedConfig:
    """
    Transformer model configuration.
    Contains all hyperparameters needed to define the architecture.
    """

    def __init__(self):
        self.vocab_size = 5000  # Token vocabulary size
        self.hidden_size = 256  # Hidden vector dimension
        self.num_heads = 4  # Number of heads in multi-head attention
        self.num_layers = 4  # Number of transformer layers (blocks)
        self.max_seq_length = 256  # Maximum sequence length
        self.dropout = 0.1  # Dropout rate for regularization
        self.batch_size = 32  # Training batch size
        self.learning_rate = 3e-4  # Learning rate
        self.num_epochs = 3  # Number of training epochs
        self.warmup_steps = 1000  # Number of warmup steps for scheduler


class TextDataset(Dataset):
    """
    Dataset for text processing. Converts texts to tokens and moves them to GPU.
    Immediately transfers data to GPU for performance optimization.
    """

    def __init__(self, texts, tokenizer, max_length, device):
        self.encodings = []
        for text in texts:
            encoded = tokenizer.encode("[START] " + text + " [END]")
            ids = encoded.ids[:max_length] if len(encoded.ids) > max_length else encoded.ids
            padding_length = max_length - len(ids)
            if padding_length > 0:
                ids = ids + [tokenizer.token_to_id("[PAD]")] * padding_length
            # Convert to tensor and immediately move to GPU
            self.encodings.append(torch.tensor(ids, device=device))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        # Return input sequence and target shifted by one token
        return item[:-1], item[1:]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism implementation.
    Allows the model to focus on different parts of the input sequence simultaneously.
    All operations performed on GPU for maximum performance.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads

        # Single linear transformation for query, key and value
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)

        # Output layer combining results from all heads
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

        # Dot-product attention scaling
        self.scale = math.sqrt(self.head_size)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        # Transform input to query, key and value
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores (dot product attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)

        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch_size, seq_length, self.hidden_size)
        x = self.output(x)
        return x


class FeedForward(nn.Module):
    """
    Feed-forward network applied after the attention mechanism.
    Consists of two linear transformations with GELU activation between them.
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
    Basic transformer block consisting of multi-head attention
    and feed-forward network with residual connections.
    """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attention_output = self.attention(self.layer_norm1(x), mask)
        x = x + self.dropout(attention_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + self.dropout(ff_output)
        return x


class SimpleTransformer(nn.Module):
    """
    Main transformer class implementing the language model.
    Optimized for GPU, with all operations performed on device.
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Output layers
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.size()

        # Create causal mask directly on GPU
        mask = torch.triu(torch.ones((seq_length, seq_length), device=self.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = ~mask

        # Generate position indices on GPU
        positions = torch.arange(seq_length, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # Combine embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # Process through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        x = self.layer_norm(x)
        x = self.output(x)
        return x


def create_tokenizer(texts):
    """
    Creates and trains a BPE tokenizer on the provided texts.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[START]", "[END]", "[UNK]"],
        vocab_size=5000
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def generate_text(model, tokenizer, device, max_length=250, prompt="[START]"):
    """
    Generates text using the trained model.
    All operations performed on GPU for improved performance.
    """
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()

            if tokenizer.decode([next_token]) == "[END]":
                break

            # Create next token tensor directly on GPU
            next_token_tensor = torch.tensor([[next_token]], device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    # Move to CPU only for final decoding
    generated_text = tokenizer.decode(input_ids[0].cpu().tolist())
    return generated_text


def train_model(config, model, train_loader, optimizer, device):
    """
    Trains the model for one epoch.
    Optimized for GPU, minimizes transfers between CPU and GPU.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")

    for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
        # Tensors are already on GPU via DataLoader
        optimizer.zero_grad()
        outputs = model(input_ids)

        outputs = outputs.view(-1, outputs.size(-1))
        target_ids = target_ids.view(-1)

        loss = F.cross_entropy(outputs, target_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})

    return total_loss / len(train_loader)


def main():
    """
    Main function that initializes and trains the model.
    Ensures efficient GPU utilization.
    """
    # Configuration and device initialization
    config = SimplifiedConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    train_texts = dataset["train"]["text"][:10000]

    print("Creating tokenizer...")
    tokenizer = create_tokenizer(train_texts)

    print("Preparing dataset...")
    # Dataset moves data to GPU during initialization
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_length, device)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    print("Initializing model...")
    # Model is initialized on GPU
    model = SimpleTransformer(config, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("Starting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        train_loss = train_model(config, model, train_loader, optimizer, device)
        print(f"Average loss: {train_loss:.4f}")

        print("\nGenerating sample text:")
        sample_text = generate_text(model, tokenizer, device)
        print(sample_text)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
