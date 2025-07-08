"""
Experiment with different prompts and observe output differences.
"""

from transformers import pipeline

# Load a text generation pipeline (GPT-2)
generator = pipeline("text-generation", model="gpt2")

prompts = [
    "Write a poem about AI:",
    "Explain AI to a 5-year-old:",
    "List pros and cons of AI:"
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    result = generator(prompt, max_length=30, do_sample=True)
    print("Generated:", result[0]['generated_text'])
