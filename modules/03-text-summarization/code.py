"""
Summarize text using a pre-trained model from Hugging Face.
"""

from transformers import pipeline

# Initialize summarization pipeline
summarizer = pipeline("summarization")

# Example long text
text = """
Generative AI refers to algorithms that can be used to create new content,
including text, images, audio, and more. These models learn from large datasets
and can produce realistic and creative outputs that mimic human work.
"""

# Generate summary
summary = summarizer(text, max_length=30, min_length=10, do_sample=False)

print("Summary:")
print(summary[0]['summary_text'])
