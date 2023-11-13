import os
import numpy as np
import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Settings
LAYER_COUNT = 48  # GPT-2 XL has 48 layers
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'activationsData')
PROMPT_MAP_FILE = os.path.join(OUTPUT_DIR, 'prompt_id_map.json')
PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'wedding_prompts.txt')

# Function to create a hook for a specific layer
def create_hook(layer_index, activations):
    # This inner function will be the actual hook
    def hook(layer, input, output):
        activations[layer_index].append(output[0].detach())
    return hook

# Function to store activations from each layer
def register_hooks(model, activations):
    hooks = []
    for i in range(LAYER_COUNT):
        # Create a hook for the current layer
        hook = create_hook(i, activations)
        # Register the hook
        model_hook = model.transformer.h[i].register_forward_hook(hook)
        hooks.append(model_hook)
    return hooks

# Function to process a single prompt
def process_prompt(model, tokenizer, text, activations, device):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Register hooks to capture activations
    hooks = register_hooks(model, activations)
    
    model.eval()
    with torch.no_grad():
        # Pass input through the model
        model(input_ids=input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

# Function to process and save activations for multiple prompts
def process_and_save_prompts(model, tokenizer, prompts, device):
    prompt_id_map = {}
    for idx, prompt in enumerate(prompts):
        print(f"Processing prompt {idx+1}/{len(prompts)}: '{prompt}'")
        activations = [[] for _ in range(LAYER_COUNT)]
        process_prompt(model, tokenizer, prompt, activations, device)

        # Initialize an empty list to hold flattened activations for each layer
        flattened_activations = []

        # Iterate through each layer's activations
        for layer in activations:
            # Extract the last token's activations, move to CPU, convert to numpy, and flatten
            last_token_activations = layer[0][:, -1, :].cpu().numpy().flatten()
            flattened_activations.append(last_token_activations)

        # Convert the list of flattened activations to a numpy array
        activation_array = np.array(flattened_activations)

        # Save the numpy array to a file
        np.save(os.path.join(OUTPUT_DIR, f'{idx}.npy'), activation_array)

        # Add to prompt ID map
        prompt_id_map[idx] = prompt

    # Save the prompt ID map to a JSON file
    with open(PROMPT_MAP_FILE, 'w') as f:
        json.dump(prompt_id_map, f, indent=4)

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)  # Move model to GPU

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Read prompts from file
with open(PROMPT_FILE, 'r') as file:
    input_prompts = [line.strip() for line in file]

# Process prompts and save activations as numpy arrays
process_and_save_prompts(model, tokenizer, input_prompts, device)