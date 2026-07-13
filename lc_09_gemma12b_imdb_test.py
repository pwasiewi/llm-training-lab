from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
SEED = 42
# Load the fine-tuned model and tokenizer
model_path = "outputs/lora-gemma3-imdb"  # Replace with your model directory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# Move the model to the appropriate device

model.to(device)

def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities and predicted class
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return probabilities, predicted_class


# Load the IMDb dataset for testing
dataset = load_dataset("imdb")

test_data = dataset["test"].shuffle(seed=SEED).select(range(500))
#test_data = dataset["test"].shuffle(seed=SEED)

# Tokenize the test set
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

test_dataset = test_data.map(preprocess_function, batched=True)

# Predict and compute accuracy
predicted_labels = []
true_labels = []

for example in test_dataset:
    text = example["text"]
    true_label = example["label"]
    _, pred_class = predict_sentiment(text)

    predicted_labels.append(pred_class)
    true_labels.append(true_label)

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
