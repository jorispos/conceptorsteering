import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

# Dictionary to store activations from all layers
activations = {}

# Function to be called by the hook; stores activations in the dictionary
def get_activations(layer_index):
    def hook(model, inputs, outputs):
        # outputs is a tuple, the actual activations are contained in outputs[0]
        activations[layer_index] = outputs[0].detach()
    return hook

# Register the hook for all transformer layers
for i, layer in enumerate(model.transformer.h):
    layer.register_forward_hook(get_activations(i))

# Pass input through the model
input_text = "Command to"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Forward pass and the hook will store the activations for each layer
model.eval()  # Put the model in evaluation mode
with torch.no_grad():  # Disable gradient calculation
    outputs = model(input_ids)

# Now the activations dictionary will contain the activations from all layers
for layer_index, activation in activations.items():
    print(f"Layer {layer_index} activations: {activation.shape}")