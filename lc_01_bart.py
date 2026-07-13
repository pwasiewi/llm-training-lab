from transformers import pipeline

# Load a summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example text
text = "Paris is the capital of France, known for its beautiful landmarks such as the Eiffel Tower and the Louvre Museum."

# Generate a summary
response = summarizer(text, max_length=27, min_length=10, do_sample=False)

# Output response
print(response)