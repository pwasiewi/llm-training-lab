"""
LESSON 2 — Encoder-only transformer for classification, classic vs modern.

Lesson 1 built a *decoder* (causal mask, next-token prediction). This lesson
builds an *encoder*: no causal mask, every token sees every other token in
BOTH directions, and instead of predicting text we pool the token vectors
into one sentence vector and classify it (IMDB sentiment, binary).
This is the BERT family — and its 2024 successor, ModernBERT.

The same script implements BOTH generations, switchable with a flag:

    python simple_02_imdb_encoder.py --arch classic   # BERT (2018) recipe
    python simple_02_imdb_encoder.py --arch modern    # ModernBERT (2024) recipe

What changed in six years (each is a self-contained mini-lesson below):

  1. Positions:  learned absolute embeddings  ->  RoPE (rotary embeddings).
     Classic adds a "position vector" to each token — the model must memorize
     what "position 137" means and can never see position 513. RoPE instead
     ROTATES the query/key vectors by an angle proportional to the position,
     so attention scores depend only on RELATIVE distance (i-j) and the
     model generalizes past its training length.
  2. FFN:  Linear-GELU-Linear  ->  GeGLU (gated linear unit).
     The gate lets the network modulate information flow per-dimension;
     empirically better at equal parameter count (Shazeer 2020,
     "GLU Variants Improve Transformer").
  3. Biases:  everywhere  ->  nowhere.
     Modern models drop bias terms from Linear/LayerNorm layers: at scale
     they add parameters and memory traffic but no measurable quality.
  4. Attention:  all layers global  ->  ALTERNATING global / local sliding
     window. A local layer only attends +-window/2 around each token — cost
     grows LINEARLY with sequence length instead of quadratically. ModernBERT
     makes every 3rd layer global (mixes distant information) and the rest
     local (cheap). This is the single trick that lets it serve 8192-token
     contexts. Lesson 4 does the exact memory math.

Two bugs from the original version of this script are FIXED here, and both
are classic real-world mistakes worth understanding:

  * No padding mask: attention happily attended to [PAD] tokens, so short
    reviews were "diluted" by hundreds of meaningless positions.
  * Unmasked mean-pooling: the sentence vector averaged over pad embeddings
    too — a 20-token review at seq_length 512 was ~96% padding vector.
    Both fixes are marked with "FIX:" comments below.

Data: IMDB (2000 train reviews), model ~10M params — minutes on GPU.
"""

import argparse
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
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, classification_report

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class EncoderConfig:
    """
    One config drives both architectures; `arch` flips the four ingredients.
    vocab_size MUST match the tokenizer's vocab (the original script had a
    50000-row embedding table for a 30000-word tokenizer — 20000 rows of
    dead weight that could never be selected).
    """

    def __init__(self, arch="classic"):
        self.arch = arch            # "classic" (BERT-like) or "modern" (ModernBERT-like)
        self.vocab_size = 30000
        self.hidden_size = 256
        self.num_heads = 4          # head_size = 256/4 = 64
        self.num_layers = 6
        self.max_seq_length = 512
        self.dropout = 0.1
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.num_epochs = 4
        self.seed = 42
        self.train_size = 2000
        self.val_size = 200
        self.test_size = 500
        # modern-only knobs:
        self.local_window = 128     # sliding-window width for local layers
        self.global_every = 3       # every 3rd layer is global (ModernBERT ratio)

    def is_global_layer(self, layer_idx):
        # Layers 0, 3 global; 1, 2, 4, 5 local (for 6 layers, global_every=3).
        # Layer 0 global matters: distant tokens can mix before local layers
        # start chopping the receptive field.
        return self.arch == "classic" or layer_idx % self.global_every == 0


class SentimentDataset(Dataset):
    """
    Text -> fixed-length id sequences + a binary label.
    Unlike lesson 1 there is no input/target shift: the "target" is a human
    label, not the text itself. This is supervised learning; lesson 1 was
    self-supervised.
    """

    def __init__(self, texts, labels, tokenizer, max_length, device):
        pad_id = tokenizer.token_to_id("[PAD]")
        self.encodings = []
        self.labels = []
        for text, label in zip(texts, labels):
            ids = tokenizer.encode(text).ids[:max_length]
            ids = ids + [pad_id] * (max_length - len(ids))
            self.encodings.append(torch.tensor(ids, device=device))
            self.labels.append(torch.tensor(label, device=device))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]


class RotaryEmbedding(nn.Module):
    """
    RoPE — Rotary Position Embedding (Su et al. 2021; used by Llama, Qwen,
    ModernBERT, nearly everything current).

    Idea: treat each consecutive pair of dimensions in q and k as a 2-D
    point and rotate it by angle position * theta_d, where theta_d differs
    per pair (a geometric series of "frequencies", like clock hands moving
    at different speeds). A dot product of two rotated vectors depends only
    on the DIFFERENCE of their angles — i.e. on the relative distance i-j.
    Position information thus lands directly inside the attention score,
    and nothing is added to the token embedding itself.
    """

    def __init__(self, head_size, max_seq_length, base=10000.0):
        super().__init__()
        # One frequency per dimension PAIR: fast for low dims, slow for high.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(max_seq_length).float()
        freqs = torch.outer(positions, inv_freq)            # (seq, head_size/2)
        angles = torch.cat([freqs, freqs], dim=-1)          # (seq, head_size)
        # Buffers (not Parameters): move with .to(device) but never trained.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def rotate_half(x):
        # (x1, x2) -> (-x2, x1): the "multiply by i" half of a 2-D rotation.
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        # q, k: (batch, heads, seq, head_size)
        seq_length = q.size(-2)
        cos = self.cos[:seq_length]
        sin = self.sin[:seq_length]
        # Classic 2-D rotation, vectorized: x' = x*cos + rotate_half(x)*sin
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k


class MultiHeadAttention(nn.Module):
    """
    Same softmax(QK^T/sqrt(d))V core as lesson 1, with two differences:
      - no causal mask (encoder = bidirectional), but a PADDING mask instead;
      - the modern variant applies RoPE to q/k and drops all bias terms.
    """

    def __init__(self, config, rope=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        bias = config.arch == "classic"     # modern: bias-free projections
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=bias)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.scale = math.sqrt(self.head_size)
        self.rope = rope                    # None for the classic arch

    def forward(self, x, attn_mask):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # (3, batch, heads, seq, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            # Position enters HERE, rotating q/k — not at the embedding layer.
            q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # attn_mask: (batch, 1, seq, seq) boolean, True = "may attend".
        # Encodes padding (never attend to [PAD] keys) and, in local layers,
        # the sliding window (never attend beyond +-window/2).
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        attention = F.softmax(scores, dim=-1)

        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).contiguous().reshape(batch_size, seq_length, self.hidden_size)
        return self.output(x)


class FeedForward(nn.Module):
    """
    classic: Linear -> GELU -> Linear, inner dim 4x hidden. (BERT, GPT-2.)
    modern:  GeGLU — the input is projected TWICE (a "gate" and a "value"),
             the gate goes through GELU and multiplies the value elementwise:

                 out = W_out( GELU(W_gate x) * W_up x )

             Parameter parity: a gated FFN has 3 weight matrices instead
             of 2, so modern models shrink the inner dim to ~2/3 * 4x = 8/3x
             to keep the same parameter count (Llama does exactly this).
    """

    def __init__(self, config):
        super().__init__()
        self.arch = config.arch
        if config.arch == "classic":
            inner = 4 * config.hidden_size
            self.fc1 = nn.Linear(config.hidden_size, inner)
            self.fc2 = nn.Linear(inner, config.hidden_size)
        else:
            inner = int(8 * config.hidden_size / 3 / 64) * 64  # multiple of 64
            # One fused projection producing gate and value halves at once.
            self.wi = nn.Linear(config.hidden_size, 2 * inner, bias=False)
            self.wo = nn.Linear(inner, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.arch == "classic":
            return self.fc2(self.dropout(F.gelu(self.fc1(x))))
        gate, up = self.wi(x).chunk(2, dim=-1)
        return self.wo(self.dropout(F.gelu(gate) * up))


class TransformerBlock(nn.Module):
    """
    Pre-norm residual block, identical wiring to lesson 1:
        x = x + Attn(LN(x));  x = x + FFN(LN(x))
    The modern variant uses bias-free LayerNorm — same normalization,
    fewer parameters to store and move.
    """

    def __init__(self, config, rope=None):
        super().__init__()
        bias = config.arch == "classic"
        self.attention = MultiHeadAttention(config, rope=rope)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, bias=bias)
        self.norm2 = nn.LayerNorm(config.hidden_size, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask):
        x = x + self.dropout(self.attention(self.norm1(x), attn_mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class SentimentEncoder(nn.Module):
    """
    Embeddings -> N encoder blocks -> masked mean-pool -> classifier head.
    """

    def __init__(self, config, device, pad_id):
        super().__init__()
        self.config = config
        self.device = device
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.arch == "classic":
            # Learned absolute positions: one trainable vector per slot,
            # ADDED to the token embedding. Hard ceiling at max_seq_length.
            self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
            rope = None
        else:
            # RoPE lives inside attention; nothing is added to embeddings,
            # and all layers share one precomputed cos/sin table.
            self.position_embedding = None
            rope = RotaryEmbedding(config.hidden_size // config.num_heads,
                                   config.max_seq_length)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, rope=rope) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_size,
                                       bias=(config.arch == "classic"))
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
        )

        # Precompute the sliding-window band once: |i-j| <= window/2.
        # Only local (non-global) layers intersect it into their mask.
        idx = torch.arange(config.max_seq_length, device=device)
        self.window_band = ((idx[None, :] - idx[:, None]).abs()
                            <= config.local_window // 2)

    def build_masks(self, input_ids):
        """
        Boolean attention masks, True = "may attend". Two lessons here:

        FIX (padding): keys at [PAD] positions are masked out in EVERY
        layer, so attention only distributes weight over real tokens.

        NaN trap: a local layer's window around a far-out [PAD] query may
        contain ONLY pads -> every score -inf -> softmax = 0/0 = NaN, which
        then poisons the whole batch through the matmul. Re-enabling the
        diagonal (every token may attend to itself) guarantees at least one
        finite score per row. Pad rows still produce garbage vectors, but
        the masked mean-pool below never reads them.
        """
        seq_length = input_ids.size(1)
        key_ok = (input_ids != self.pad_id)                     # (batch, seq)
        pad_mask = key_ok[:, None, None, :]                     # (batch,1,1,seq)
        eye = torch.eye(seq_length, dtype=torch.bool, device=input_ids.device)
        global_mask = (pad_mask | eye).expand(-1, 1, seq_length, -1)
        band = self.window_band[:seq_length, :seq_length]       # (seq, seq)
        local_mask = (pad_mask & band) | eye                    # broadcast -> (batch,1,seq,seq)
        return global_mask, local_mask, key_ok

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        global_mask, local_mask, key_ok = self.build_masks(input_ids)

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_length, device=self.device)
            x = x + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        for layer_idx, block in enumerate(self.blocks):
            mask = global_mask if self.config.is_global_layer(layer_idx) else local_mask
            x = block(x, mask)

        x = self.final_norm(x)

        # FIX (pooling): average ONLY over real tokens. The original
        # x.mean(dim=1) divided by the full seq_length, so a 20-token review
        # padded to 512 contributed mostly [PAD] vectors to its "meaning".
        token_ok = key_ok.unsqueeze(-1).float()                 # (batch, seq, 1)
        x = (x * token_ok).sum(dim=1) / token_ok.sum(dim=1).clamp(min=1.0)

        return self.classifier(x).squeeze(-1)


def create_tokenizer(texts, vocab_size):
    """
    BPE trained on the TRAINING texts only — training it on test data too
    would be a (mild) form of leakage; lesson 6 dissects a real case of it.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(list(texts), trainer)
    return tokenizer


def print_param_breakdown(model, config):
    """
    Where do the parameters live? Compare both archs:
      - modern has NO position table (RoPE is parameter-free) and no biases;
      - its GeGLU FFN has 3 matrices at ~8/3x vs 2 matrices at 4x — nearly
        the same total, which is the point.
    """
    def count(module):
        return sum(p.numel() for p in module.parameters())

    embeddings = count(model.token_embedding)
    positions = count(model.position_embedding) if model.position_embedding is not None else 0
    one_attn = count(model.blocks[0].attention)
    one_ffn = count(model.blocks[0].feed_forward)
    total = count(model)
    print(f"Parameter breakdown ({config.arch}):")
    print(f"  token embeddings   : {embeddings/1e6:6.2f}M")
    print(f"  position embeddings: {positions/1e6:6.2f}M"
          + ("  (RoPE: zero learned params)" if positions == 0 else ""))
    print(f"  attention / layer  : {one_attn/1e6:6.2f}M")
    print(f"  ffn / layer        : {one_ffn/1e6:6.2f}M"
          + ("  (GeGLU 8/3x: 3 matrices ~= classic 2 matrices at 4x)"
             if config.arch == "modern" else "  (classic 4x)"))
    print(f"  TOTAL              : {total/1e6:6.2f}M")


def evaluate_model(model, data_loader, split_name="test"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for input_ids, labels in tqdm(data_loader, desc=f"Evaluating on {split_name}"):
            outputs = model(input_ids)
            total_loss += criterion(outputs.float(), labels.float()).item()
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n{split_name.capitalize()}: loss {total_loss/len(data_loader):.4f}, "
          f"accuracy {accuracy:.4f}")
    if split_name == "test":
        print(classification_report(all_labels, all_preds))
    return accuracy


def train_one_epoch(model, train_loader, optimizer):
    """
    Binary classification loop: one logit per review, BCEWithLogitsLoss
    (sigmoid + binary cross-entropy fused for numerical stability).
    Compare lesson 1: there the loss was over 5000 classes at EVERY
    position; here it is one scalar per WHOLE sequence.
    """
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        optimizer.zero_grad()
        loss = criterion(model(input_ids).float(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{total_loss/(batch_idx+1):.4f}"})
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--arch", choices=["classic", "modern"], default="classic",
                        help="classic = BERT-2018 recipe, modern = ModernBERT-2024 recipe")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = EncoderConfig(arch=args.arch)
    if args.epochs:
        config.num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, architecture: {config.arch}")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    train_split = dataset["train"].shuffle(seed=config.seed).select(
        range(config.train_size + config.val_size))
    test_split = dataset["test"].shuffle(seed=config.seed).select(range(config.test_size))
    train_texts = train_split["text"][:config.train_size]
    train_labels = train_split["label"][:config.train_size]
    val_texts = train_split["text"][config.train_size:]
    val_labels = train_split["label"][config.train_size:]

    print("Training BPE tokenizer on the training split...")
    tokenizer = create_tokenizer(train_texts, config.vocab_size)
    pad_id = tokenizer.token_to_id("[PAD]")

    print("Preparing datasets...")
    train_data = SentimentDataset(train_texts, train_labels, tokenizer,
                                  config.max_seq_length, device)
    val_data = SentimentDataset(val_texts, val_labels, tokenizer,
                                config.max_seq_length, device)
    test_data = SentimentDataset(test_split["text"], test_split["label"], tokenizer,
                                 config.max_seq_length, device)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    print("Initializing model...")
    model = SentimentEncoder(config, device, pad_id).to(device)
    print_param_breakdown(model, config)
    if config.arch == "modern":
        n_global = sum(config.is_global_layer(i) for i in range(config.num_layers))
        print(f"Attention layout: {n_global} global + {config.num_layers - n_global} "
              f"local (window {config.local_window}) of {config.num_layers} layers")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\nSplits: train {len(train_texts)}, val {len(val_texts)}, "
          f"test {len(test_split)}")
    os.makedirs("outputs", exist_ok=True)
    best_model_path = f"outputs/best_encoder_{config.arch}.pt"
    best_val_accuracy = 0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Average training loss: {train_loss:.4f}")
        val_accuracy = evaluate_model(model, val_loader, "validation")
        # Model selection on VALIDATION, final number from TEST — never pick
        # the checkpoint using test accuracy, that overfits the test set.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model!")

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print("Evaluating best checkpoint on the test set...")
    model.load_state_dict(torch.load(best_model_path))
    evaluate_model(model, test_loader, "test")
    print("\nThings to try: --arch modern vs classic wall-clock time per epoch,")
    print("local_window=32 (starves distant context), global_every=6 (only layer 0")
    print("mixes globally), and removing the pooling fix to see accuracy drop.")


if __name__ == "__main__":
    main()
