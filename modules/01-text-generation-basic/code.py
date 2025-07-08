"""
Basic text generation using GPT-2.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the prompt text
prompt = "Once upon a time in a world of AI,"

# Encode the prompt text
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text from the model
# max_length controls total output length; do_sample=True enables randomness
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=50
)

# Decode the generated ids back to text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:")
print(generated_text)
