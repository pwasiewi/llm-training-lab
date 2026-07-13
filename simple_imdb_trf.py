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


class SentimentConfig:
    """
    Configuration for sentiment analysis transformer model.
    """

    def __init__(self):
        self.vocab_size = 50000
        self.hidden_size = 512
        self.num_heads = 4
        self.num_layers = 4
        self.max_seq_length = 512
        self.dropout = 0.1
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 2
        self.warmup_steps = 1000
        self.seed = 42  # Added seed for reproducibility
        self.train_size = 2000  # Size of training dataset
        self.val_size = 200  # Size of validation dataset
        self.test_size = 500  # Size of test dataset



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

class SentimentTransformer(nn.Module):
    """
    Transformer model adapted for sentiment analysis using PyTorch's built-in TransformerEncoderLayer.
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)

        # Use nn.TransformerEncoderLayer instead of custom TransformerBlock
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=4 * config.hidden_size,
                dropout=config.dropout,
                activation="gelu"
            ),
            num_layers=config.num_layers
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Modified output layer for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )

    def forward(self, x):
        batch_size, seq_length = x.size()

        positions = torch.arange(seq_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # TransformerEncoder expects input of shape (seq_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_length, hidden_size)

        # Global average pooling
        x = x.mean(dim=1)
        x = self.layer_norm(x)
        x = self.classifier(x)
        return x.squeeze(-1)


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


def evaluate_model(model, data_loader, device, split_name="test"):
    """
    Evaluates the model on given dataset split.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for input_ids, labels in tqdm(data_loader, desc=f"Evaluating on {split_name}"):
            outputs = model(input_ids)
            loss = criterion(outputs.float(), labels.float())
            predictions = (torch.sigmoid(outputs) > 0.5).long()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n{split_name.capitalize()} Results:")
    print(f"Average loss: {total_loss / len(data_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return accuracy


def train_model(config, model, train_loader, val_loader, optimizer, device):
    """
    Trains the model and evaluates on validation set.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.float(), labels.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})

    # Validation
    val_accuracy = evaluate_model(model, val_loader, device, "validation")
    return total_loss / len(train_loader), val_accuracy


def main():
    """
    Main function to train and evaluate the sentiment analysis model.
    """
    config = SentimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # Shuffle and select subsets of the data
    train_dataset = dataset["train"].shuffle(seed=config.seed).select(range(config.train_size + config.val_size))
    test_dataset = dataset["test"].shuffle(seed=config.seed).select(range(config.test_size))

    # Split training data into train and validation
    train_texts = train_dataset["text"][:config.train_size]
    train_labels = train_dataset["label"][:config.train_size]
    val_texts = train_dataset["text"][config.train_size:]
    val_labels = train_dataset["label"][config.train_size:]

    # Prepare test data
    test_texts = test_dataset["text"]
    test_labels = test_dataset["label"]

    print("Creating tokenizer...")
    tokenizer = create_tokenizer(train_texts)

    print("Preparing datasets...")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length, device)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length, device)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.max_seq_length, device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    print("Initializing model...")
    model = SentimentTransformer(config, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\nDataset sizes:")
    print(f"Training: {len(train_texts)}")
    print(f"Validation: {len(val_texts)}")
    print(f"Test: {len(test_texts)}")

    os.makedirs("outputs", exist_ok=True)
    print("\nStarting training...")
    best_val_accuracy = 0
    best_model_path = "outputs/best_sentiment_model.pt"

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        train_loss, val_accuracy = train_model(config, model, train_loader, val_loader, optimizer, device)
        print(f"Average training loss: {train_loss:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model!")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_accuracy = evaluate_model(model, test_loader, device, "test")
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()