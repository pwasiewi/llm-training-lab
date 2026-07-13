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
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.makedirs("outputs", exist_ok=True)

best_model_path = "outputs/best_hybrid_double_light_res.pt"

class ModelConfig:
    """
    Configuration for hybrid LSTM-CNN model.
    """

    def __init__(self):
        self.vocab_size = 50000
        self.hidden_size = 512
        self.num_heads = 4
        self.hidden_dim = 1024
        self.n_filters = 200
        self.filter_sizes = (3, 4, 5)
        self.dropout = 0.5
        self.max_seq_length = 256
        self.batch_size = 32 
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.seed = 64
        self.train_size = 23000
        self.val_size = 2000
        self.test_size = 25000
        # self.vocab_size = 50000
        # self.hidden_size = 768
        # self.num_heads = 4
        # self.hidden_dim = 1024
        # self.n_filters = 500
        # self.filter_sizes = (3, 4, 5, 7, 9)
        # self.dropout = 0.3
        # self.max_seq_length = 256
        # self.batch_size = 32
        # self.learning_rate = 1e-4
        # self.num_epochs = 10
        # self.seed = 49
        # self.train_size = 23000
        # self.val_size = 2000
        # self.test_size = 25000

class HybridLSTMCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=4*config.hidden_size,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=2
        )

        self.lstm1 = nn.LSTM(config.hidden_size,
                             config.hidden_dim,
                             bidirectional=True,
                             batch_first=True)

        self.proj1 = nn.Linear(config.hidden_size, config.hidden_dim * 2)  # Projection layer for residuals

        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.n_filters, (fs, config.hidden_dim * 2))
            for fs in config.filter_sizes
        ])

        self.fc = nn.Linear(len(config.filter_sizes) * config.n_filters, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))

        # Transformer with residual connection
        transformed = self.transformer(embedded) + embedded  # Skip connection

        # First LSTM with residual connection
        lstm1_out, _ = self.lstm1(transformed)
        lstm1_out = lstm1_out + self.proj1(transformed)  # Skip connection

        # CNN processing
        lstm1_out = lstm1_out.unsqueeze(1)
        conved = [F.relu(conv(lstm1_out)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # Concatenate and apply dropout
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat).squeeze(-1)

    def _init_weights(self):
        """Initialize weights properly based on tensor dimensions"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # For linear layers and conv layers
                    nn.init.xavier_uniform_(param)
                else:
                    # For 1D tensors (e.g., bias terms or embeddings)
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'embedding' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)


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


def train_model(config, model, train_loader, val_loader, optimizer, device,
                scheduler=None):
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

    # Update learning rate if scheduler is provided
    if scheduler is not None:
        scheduler.step(val_accuracy)  # Use val_accuracy as metric for learning rate reduction
        # Print current learning rate after update
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

    return total_loss / len(train_loader), val_accuracy


def main():
    """
    Main function to train and evaluate the sentiment analysis model.
    """
    config = ModelConfig()
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

    print("Initializing Hybrid LSTM-CNN model...")
    model = HybridLSTMCNN(config).to(device)

    # Check if the best model from previous run exists
    if os.path.exists(best_model_path):
        print("Loading best model from previous run...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("No previous model found, initializing new model...")
        model._init_weights()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.98),  # Slightly different beta values
        eps=1e-8,           # Numerical stability
        amsgrad=True       # AMSGrad variant disabled
    )

    # ReduceLROnPlateau without verbose parameter
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    print(f"\nDataset sizes:")
    print(f"Training: {len(train_texts)}")
    print(f"Validation: {len(val_texts)}")
    print(f"Test: {len(test_texts)}")

    print("\nModel Architecture:")
    print(f"Embedding dim: {config.hidden_size}")
    print(f"LSTM hidden dim: {config.hidden_dim}")
    print(f"CNN filters: {config.n_filters}")
    print(f"Filter sizes: {config.filter_sizes}")

    print("\nStarting training...")
    best_val_accuracy = 0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        train_loss, val_accuracy = train_model(config, model, train_loader, val_loader, optimizer, device, scheduler)
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
