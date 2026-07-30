"""
LESSON 3 — The same encoder classifier, written with the PyTorch library.

Lesson 2 built every matrix by hand. Real projects almost never do: PyTorch
ships the whole encoder block as `nn.TransformerEncoderLayer`. This lesson
maps each constructor argument back to the handwritten parts, so the library
stops being a black box:

    nn.TransformerEncoderLayer(
        d_model=256,          # hidden_size — width of the residual stream
        nhead=4,              # num_heads in MultiHeadAttention
        dim_feedforward=1024, # the 4x inner dim of the classic FFN
        activation="gelu",    # F.gelu between fc1 and fc2
        norm_first=True,      # PRE-norm: x + Attn(LN(x)), as in lessons 1-2.
                              # The default (False) is the original post-norm
                              # formulation, which trains less stably.
        batch_first=True,     # tensors are (batch, seq, hidden). The legacy
                              # default (False!) expects (seq, batch, hidden)
                              # and forced the old version of this script to
                              # do a permute(1,0,2) dance before and after.
    )

Two more production techniques appear here:

  1. src_key_padding_mask — the library's version of lesson 2's padding
     mask. Convention trap: True means "IGNORE this position", the exact
     OPPOSITE of the mask we built by hand. Off-by-inversion here silently
     trains a model that attends ONLY to padding.
  2. bf16 autocast — mixed precision. Inside the `torch.autocast` block,
     matmuls run in bfloat16 (half the memory traffic, tensor cores) while
     reductions like softmax/layernorm stay in float32. bfloat16 keeps
     float32's exponent range (~1e38) with less mantissa, so unlike fp16 it
     needs NO gradient scaler — this is why modern training defaults to it.

Run:  python simple_03_imdb_library.py
Data: IMDB (2000 train reviews) — a couple of minutes on GPU.
"""

import argparse
import torch
import torch.nn as nn
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


class LibraryConfig:
    def __init__(self):
        self.vocab_size = 30000     # matches the tokenizer (see lesson 2)
        self.hidden_size = 256
        self.num_heads = 4
        self.num_layers = 6
        self.max_seq_length = 512
        self.dropout = 0.1
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.num_epochs = 3
        self.seed = 42
        self.train_size = 2000
        self.val_size = 200
        self.test_size = 500


class SentimentDataset(Dataset):
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


class LibrarySentimentEncoder(nn.Module):
    """
    Embeddings -> nn.TransformerEncoder -> masked mean-pool -> classifier.
    Everything lesson 2 wrote by hand (qkv projection, softmax, residuals,
    layer norms, FFN) hides inside the encoder layer object.
    """

    def __init__(self, config, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Library encoders have no position story at all — you must add one
        # yourself or the model is order-blind (lesson 1's "dog bites man").
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=4 * config.hidden_size,
                dropout=config.dropout,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            ),
            num_layers=config.num_layers,
            # An internal fast path packs variable-length rows into a
            # "nested tensor"; it only kicks in for post-norm models anyway,
            # and disabling it keeps behavior identical to lesson 2.
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
        )

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        x = self.dropout(x)

        # INVERTED convention: True = "this key is padding, IGNORE it".
        # Lesson 2's hand-built mask used True = "may attend".
        padding = input_ids == self.pad_id                     # (batch, seq)
        x = self.encoder(x, src_key_padding_mask=padding)

        x = self.final_norm(x)
        # Masked mean-pool, same fix as lesson 2: average real tokens only.
        token_ok = (~padding).unsqueeze(-1).float()
        x = (x * token_ok).sum(dim=1) / token_ok.sum(dim=1).clamp(min=1.0)
        return self.classifier(x).squeeze(-1)


def create_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(list(texts), trainer)
    return tokenizer


def evaluate_model(model, data_loader, autocast_ctx, split_name="test"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for input_ids, labels in tqdm(data_loader, desc=f"Evaluating on {split_name}"):
            with autocast_ctx():
                outputs = model(input_ids)
                loss = criterion(outputs.float(), labels.float())
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs.float()) > 0.5).long()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n{split_name.capitalize()}: loss {total_loss/len(data_loader):.4f}, "
          f"accuracy {accuracy:.4f}")
    if split_name == "test":
        print(classification_report(all_labels, all_preds))
    return accuracy


def train_one_epoch(model, train_loader, optimizer, autocast_ctx):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        optimizer.zero_grad()
        # Forward + loss under bf16 autocast; backward OUTSIDE the block
        # (gradients are computed in the dtype each op ran in — that part
        # autocast handles automatically). No GradScaler: bf16's exponent
        # range makes gradient underflow a non-issue, unlike fp16.
        with autocast_ctx():
            outputs = model(input_ids)
            loss = criterion(outputs.float(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{total_loss/(batch_idx+1):.4f}"})
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-autocast", action="store_true",
                        help="run pure fp32 to compare speed and memory")
    args = parser.parse_args()

    config = LibraryConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and not args.no_autocast
    print(f"Using device: {device}, bf16 autocast: {use_bf16}")

    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=use_bf16)

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
    model = LibrarySentimentEncoder(config, pad_id).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M "
          f"(compare with lesson 2's breakdown — same shape, same count)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    os.makedirs("outputs", exist_ok=True)
    best_model_path = "outputs/best_encoder_library.pt"
    best_val_accuracy = 0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, autocast_ctx)
        print(f"Average training loss: {train_loss:.4f}")
        val_accuracy = evaluate_model(model, val_loader, autocast_ctx, "validation")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model!")

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print("Evaluating best checkpoint on the test set...")
    model.load_state_dict(torch.load(best_model_path))
    evaluate_model(model, test_loader, autocast_ctx, "test")
    print("\nThings to try: --no-autocast (compare it/s in the progress bar),")
    print("norm_first=False (post-norm: watch early-epoch loss wobble),")
    print("and flipping the padding mask sign to see the silent failure mode.")


if __name__ == "__main__":
    main()
