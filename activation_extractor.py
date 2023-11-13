import os
import numpy as np
import torch
import json
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
LAYER_COUNT = 48  # Number of layers in the GPT-2 XL model
MODEL_NAME = 'gpt2-xl'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'activationsData')
PROMPT_MAP_FILE = os.path.join(OUTPUT_DIR, 'prompt_id_map.json')
PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'wedding_prompts.txt')

def create_hook(layer_index, activations):
    """
    Create a forward hook for a specific layer.
    Args:
        layer_index (int): Index of the layer.
        activations (list): List to store activations from each layer.
    Returns:
        function: A hook function.
    """
    def hook(layer, input, output):
        # Append output activations of the current layer to the activations list.
        activations[layer_index].append(output[0].detach())
    return hook

def register_hooks(model, activations):
    """
    Register forward hooks to capture activations from each layer of the model.
    Args:
        model (torch.nn.Module): The GPT-2 model.
        activations (list): List to store activations from each layer.
    Returns:
        list: List of registered hooks.
    """
    hooks = []
    for i in range(LAYER_COUNT):
        hook = create_hook(i, activations)
        model_hook = model.transformer.h[i].register_forward_hook(hook)
        hooks.append(model_hook)
    return hooks

def process_prompt(model, tokenizer, text, activations, device):
    """
    Process a single prompt to capture and store activations.
    Args:
        model (torch.nn.Module): The GPT-2 model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for GPT-2.
        text (str): The input prompt text.
        activations (list): List to store activations from each layer.
        device (torch.device): The device to run the model on (CPU or GPU).
    """
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    hooks = register_hooks(model, activations)
    model.eval()
    with torch.no_grad():
        model(input_ids=input_ids)
    
    for hook in hooks:
        hook.remove()

def process_and_save_prompts(model, tokenizer, prompts, device):
    """
    Process and save activations for multiple prompts.
    Args:
        model (torch.nn.Module): The GPT-2 model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for GPT-2.
        prompts (list of str): List of input prompts.
        device (torch.device): The device to run the model on (CPU or GPU).
    """
    prompt_id_map = {}
    for idx, prompt in enumerate(prompts):
        logging.info(f"Processing prompt {idx+1}/{len(prompts)}: '{prompt}'")
        activations = [[] for _ in range(LAYER_COUNT)]

        process_prompt(model, tokenizer, prompt, activations, device)

        # Convert and flatten activations to a numpy array
        flattened_activations = [layer[0][:, -1, :].cpu().numpy().flatten() for layer in activations]
        activation_array = np.array(flattened_activations)

        np.save(os.path.join(OUTPUT_DIR, f'{idx}.npy'), activation_array)
        prompt_id_map[idx] = prompt

    with open(PROMPT_MAP_FILE, 'w') as f:
        json.dump(prompt_id_map, f, indent=4)

def read_prompts(file_path):
    """
    Read prompts from a file.
    Args:
        file_path (str): Path to the file containing prompts.
    Returns:
        list of str: List of prompts.
    """
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Read prompts from file
    input_prompts = read_prompts(PROMPT_FILE)

    # Check if input prompts are available
    if input_prompts:
        # Process prompts and save activations as numpy arrays
        process_and_save_prompts(model, tokenizer, input_prompts, device)
    else:
        logging.error("No prompts to process. Exiting.")