from transformers import pipeline

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = ["summary", "fact", "question"]

# Text to classify
text = "The Eiffel Tower is a world-renowned landmark in Paris."

# Perform zero-shot classification
response = classifier(text, candidate_labels=candidate_labels)

# Output predicted response label
print(response)