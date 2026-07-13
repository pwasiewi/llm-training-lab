from transformers import AutoModel
import torch
# Load DistilBERT model
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Display model structure
print(model)

# Iterate over layers
for name, module in model.named_children():
    print(f"Layer name: {name}, Layer type: {type(module)}")

# Analyze encoder layer
encoder = model.transformer
print(encoder)

for name, module in model.named_modules():
    if "attention" in name:
        print(name, module)

state_dict = torch.load("/home/guest/PycharmProjects/text-generation-webui/lora-bert-distilbert/checkpoint-6252/training_args.bin")
print(state_dict)