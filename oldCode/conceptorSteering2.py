import os
import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Settings
MAX_LENGTH = 10
LAYER_TO_EXTRACT = 6
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'activationsData')

# Function to store activations from the specified layer
def get_activations(activations, layer_num):
    def hook(model, input, output):
        activations.append(output[0].detach())
    return hook

# Function to generate text token by token and record activations
def generate_and_capture_activations(model, tokenizer, text, layer_num=LAYER_TO_EXTRACT-1, max_length=MAX_LENGTH):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    activations = []
    hook = model.transformer.h[layer_num].register_forward_hook(get_activations(activations, layer_num))
    model.eval()
    with torch.no_grad():
        for token_index in range(max_length):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            print(f"Generated token {token_index + 1}/{max_length}")  # Print statement for each generated token
    hook.remove()
    return input_ids, activations

# Function to process multiple prompts and organize data by ID
def process_prompts(model, tokenizer, prompts, layer_num=LAYER_TO_EXTRACT-1, max_length=MAX_LENGTH):
    data = {}
    for idx, prompt in enumerate(prompts):
        print(f"Processing prompt {idx+1}/{len(prompts)}: '{prompt}'")  # Print current prompt being processed
        generated_ids, layer_activations = generate_and_capture_activations(model, tokenizer, prompt, layer_num=layer_num, max_length=max_length)

        activations_dict = {}
        for step_index, step in enumerate(layer_activations):
            # Flatten the activations for each step and store them horizontally
            flattened_activations = step[:, -1, :].cpu().numpy().flatten().tolist()
            activations_dict[step_index] = {'activation': flattened_activations}

        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0][len(tokenizer.encode(prompt)):])

        # Add token information to each activation
        for i, token in enumerate(generated_tokens):
            if i in activations_dict:
                activations_dict[i]['token'] = token

        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        data[idx] = {
            'input_prompt': prompt,
            'generated_text': output_text,
            'activations': activations_dict
        }
    return data

# Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Array of input prompts
input_prompts = ["Steven Abreau is flying", "Joris Postmus is surfing", "Herbert Jaeger is doing Mathematics"]  # Example prompts

# Process prompts and get data
all_data = process_prompts(model, tokenizer, input_prompts, layer_num=LAYER_TO_EXTRACT-1)

# Save the data to a JSON file
with open(os.path.join(OUTPUT_DIR, 'activations.json'), 'w') as f:
    json.dump(all_data, f, indent=4)