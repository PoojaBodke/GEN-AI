"""
Demo of fine-tuning a small language model on a toy dataset.
"""

from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling

# Load tokenizer and model
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Create a tiny dataset file
with open("tiny_dataset.txt", "w") as f:
    f.write("AI is transforming the world.\nGenerative models are amazing.\n")

# Load dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="tiny_dataset.txt",
    block_size=32
)

# Data collator helps with batching
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Start fine-tuning
trainer.train()

print("Fine-tuning complete!")
