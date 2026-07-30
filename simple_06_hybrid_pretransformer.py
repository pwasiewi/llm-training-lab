"""
LESSON 6 — Inductive biases: transformer + LSTM + CNN in one model.

Before transformers won, NLP models encoded ASSUMPTIONS about language in
their architecture. This hybrid stacks three generations so you can see
what each one contributes:

  transformer  "any token may relate to any other" — no built-in bias,
               relations are LEARNED from data. Powerful but data-hungry.
  BiLSTM       "text is a sequence read word by word" — order is baked in
               as recurrence, left-to-right and right-to-left states.
               Linear cost in seq_length, but inherently serial (no
               parallelism across positions — the reason training scaled
               poorly and transformers took over).
  CNN + pool   "meaning lives in local n-grams" — each Conv2d filter of
               height fs is literally an fs-gram detector; max-pooling asks
               "did this n-gram pattern appear ANYWHERE in the review?"
               (Kim 2014, the classic CNN text classifier.)

Residual connections (lesson 1) glue the stages: each stage only adds a
correction, so a useless stage can be bypassed by the gradient instead of
destroying the signal.

    --lstm-depth 1|3 controls the recurrent stack (the old res / res4
    variants of this script were exactly these two settings).

THE DATA LEAKAGE LESSON (the res4 variant really had this bug):
its tokenizer was trained on train+test texts:

    tokenizer = create_tokenizer(train_texts + test_texts)      # BUG

The test labels never leaked, so accuracy barely moves here — but the
tokenizer's merge table now "knows" the test vocabulary: a rare word that
appears only in the test set gets its own token instead of being shattered
into subwords, which is information no deployed model would have at
training time. This is the mild end of a spectrum whose severe end
(deduplicated test data inside pretraining corpora) invalidates benchmarks.
Rule: EVERY fitted component — tokenizer, scaler, vocabulary, imputer —
is fitted on training data only. The fix below is one line.

Run:  python simple_06_hybrid_pretransformer.py                # depth 1
      python simple_06_hybrid_pretransformer.py --lstm-depth 3
Data: IMDB (8000 train reviews) — several minutes on GPU.
"""

import argparse
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


class HybridConfig:
    def __init__(self, lstm_depth=1):
        self.vocab_size = 30000
        self.hidden_size = 256      # embedding / transformer width
        self.num_heads = 4
        self.num_transformer_layers = 2
        self.lstm_depth = lstm_depth
        self.lstm_hidden = 512      # per direction; bidirectional -> 1024 out
        self.n_filters = 100        # CNN filters per n-gram size
        self.filter_sizes = (3, 4, 5)   # 3-, 4-, 5-gram detectors
        self.dropout = 0.3
        self.max_seq_length = 256
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.num_epochs = 3
        self.seed = 42
        self.train_size = 8000
        self.val_size = 1000
        self.test_size = 2000


class HybridTrfLstmCnn(nn.Module):
    """
    embedding -> transformer (+res) -> BiLSTM stack (+res) -> CNN -> pool -> fc

    Information flow: the transformer mixes global context into each token,
    the BiLSTM re-reads the result in order, and the CNN scans the LSTM
    states for decisive local patterns ("one of the worst", "a must see").
    """

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=4 * config.hidden_size,
                dropout=config.dropout,
                batch_first=True,       # (batch, seq, hidden) — see lesson 3
                norm_first=True,
            ),
            num_layers=config.num_transformer_layers,
            enable_nested_tensor=False,
        )

        lstm_out_dim = config.lstm_hidden * 2   # bidirectional concat
        # First LSTM changes width (hidden_size -> 2*lstm_hidden), so its
        # residual needs a projection to matching shape (compare lesson 1,
        # where all residuals were same-width and needed none).
        self.lstm_in = nn.LSTM(config.hidden_size, config.lstm_hidden,
                               bidirectional=True, batch_first=True)
        self.residual_proj = nn.Linear(config.hidden_size, lstm_out_dim)
        # Deeper stack (--lstm-depth 3): same-width LSTMs, plain residuals.
        self.lstm_stack = nn.ModuleList([
            nn.LSTM(lstm_out_dim, config.lstm_hidden,
                    bidirectional=True, batch_first=True)
            for _ in range(config.lstm_depth - 1)
        ])

        # Conv2d over (seq, features) with kernel (fs, all-features):
        # slides down the sequence only — an fs-gram detector per filter.
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.n_filters, (fs, lstm_out_dim))
            for fs in config.filter_sizes
        ])
        self.fc = nn.Linear(len(config.filter_sizes) * config.n_filters, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))

        x = self.transformer(embedded) + embedded          # residual

        lstm_out, _ = self.lstm_in(x)
        x = lstm_out + self.residual_proj(x)               # projected residual
        for lstm in self.lstm_stack:
            lstm_out, _ = lstm(x)
            x = lstm_out + x                               # plain residual

        # (batch, seq, feat) -> (batch, 1, seq, feat): the "1" is the image
        # channel dim — we treat the sequence as a 1-channel image.
        x = x.unsqueeze(1)
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # Max-over-time pooling: keep each filter's single strongest match,
        # wherever in the review it fired. Position is discarded on purpose.
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat).squeeze(-1)


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


def create_tokenizer(texts, vocab_size):
    """
    FIX: fitted on the TRAINING texts only. The old res4 variant passed
    train_texts + test_texts here — see the leakage note in the header.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(list(texts), trainer)
    return tokenizer


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
    parser.add_argument("--lstm-depth", type=int, choices=[1, 2, 3], default=1,
                        help="1 = old 'res' variant, 3 = old 'res4' variant")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = HybridConfig(lstm_depth=args.lstm_depth)
    if args.epochs:
        config.num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, LSTM depth: {config.lstm_depth}")

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

    print("Training BPE tokenizer (training split ONLY — see leakage note)...")
    tokenizer = create_tokenizer(train_texts, config.vocab_size)

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

    print("Initializing hybrid model...")
    model = HybridTrfLstmCnn(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_lstm = (sum(p.numel() for p in model.lstm_in.parameters())
              + sum(p.numel() for p in model.lstm_stack.parameters()))
    n_trf = sum(p.numel() for p in model.transformer.parameters())
    n_cnn = sum(p.numel() for p in model.convs.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M total — transformer {n_trf/1e6:.2f}M, "
          f"LSTM {n_lstm/1e6:.2f}M, CNN {n_cnn/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\nSplits: train {len(train_texts)}, val {len(val_texts)}, "
          f"test {len(test_split)}")
    os.makedirs("outputs", exist_ok=True)
    best_model_path = f"outputs/best_hybrid_depth{config.lstm_depth}.pt"
    best_val_accuracy = 0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Average training loss: {train_loss:.4f}")
        val_accuracy = evaluate_model(model, val_loader, "validation")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model!")

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print("Evaluating best checkpoint on the test set...")
    model.load_state_dict(torch.load(best_model_path))
    evaluate_model(model, test_loader, "test")
    print("\nThings to try: --lstm-depth 3 (does depth pay for its cost here?),")
    print("num_transformer_layers=0 vs lstm_depth=0-equivalent ablations,")
    print("filter_sizes=(7,9) (longer n-grams), and re-adding the tokenizer")
    print("leakage to measure how little it changes accuracy — then argue why")
    print("it is still wrong.")


if __name__ == "__main__":
    main()
