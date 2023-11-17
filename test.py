from transformers import GPT2LMHeadModel
import torch

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
# Extract a single block
block = model.transformer.h[6]  # h is the list of GPT2Block

# Run a forward pass with dummy data
dummy_input = torch.randn(1, 1, 1600)  # Adjust the shape as needed
output = block(dummy_input)

# Inspect the output
print(type(output), len(output))
print("First element shape:", output[0].shape)