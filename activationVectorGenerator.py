import os
import numpy as np
import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Settings
LAYER_COUNT = 48  # GPT-2 XL has 48 layers
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'activationsData')
PROMPT_MAP_FILE = os.path.join(OUTPUT_DIR, 'prompt_id_map.json')
PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'wedding_prompts.txt')  # Path to the file containing prompts

# Function to store activations from each layer
def register_hooks(model, activations):
    hooks = []
    for i in range(LAYER_COUNT):
        hook = model.transformer.h[i].register_forward_hook(
            lambda layer, _, output, i=i: activations[i].append(output[0].detach())
        )
        hooks.append(hook)
    return hooks

# Function to process a single prompt
def process_prompt(model, tokenizer, text, activations):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
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
def process_and_save_prompts(model, tokenizer, prompts):
    prompt_id_map = {}
    for idx, prompt in enumerate(prompts):
        print(f"Processing prompt {idx+1}/{len(prompts)}: '{prompt}'")
        activations = [[] for _ in range(LAYER_COUNT)]
        process_prompt(model, tokenizer, prompt, activations)

        # Convert activations to a numpy array
        activation_array = np.array([layer[0][:, -1, :].cpu().numpy().flatten() for layer in activations])

        # Save the numpy array to a file
        np.save(os.path.join(OUTPUT_DIR, f'{idx}.npy'), activation_array)

        # Add to prompt ID map
        prompt_id_map[idx] = prompt

    # Save the prompt ID map to a JSON file
    with open(PROMPT_MAP_FILE, 'w') as f:
        json.dump(prompt_id_map, f, indent=4)

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Read prompts from file
with open(PROMPT_FILE, 'r') as file:
    input_prompts = [line.strip() for line in file]

# Process prompts and save activations as numpy arrays
process_and_save_prompts(model, tokenizer, input_prompts)