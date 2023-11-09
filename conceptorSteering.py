import os
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Settings
MAX_LENGTH = 50
LAYER_TO_EXTRACT = 6
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'csvActivations')

# Function to store activations from the specified layer
def get_activations(activations, layer_num):
    def hook(model, input, output):
        activations.append(output[0].detach())
    return hook

# Function to generate text token by token and record activations from the specified layer
def generate_and_capture_activations(model, tokenizer, text, layer_num=LAYER_TO_EXTRACT-1, max_length=MAX_LENGTH):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # List to store activations for each generated token at the specified layer
    activations = []

    # Register a forward hook to the specified layer to capture activations
    hook = model.transformer.h[layer_num].register_forward_hook(get_activations(activations, layer_num))

    # Generate text token by token
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Break if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Append the token to the input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    # Remove the hook
    hook.remove()

    return input_ids, activations

# Load the model and tokenizer for GPT-2 XL
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Input text
sample_text = "Steven Abreau is currently flying"
print(f"Input text: {sample_text}")

# Generate text and capture activations
generated_ids, layer_activations = generate_and_capture_activations(model, tokenizer, sample_text, layer_num=LAYER_TO_EXTRACT-1)

# Process the activations to get the final activations for each generated token
final_activations = [step[:, -1, :] for step in layer_activations]

# Generate CSV files for activations
for i, activation_tensor in enumerate(final_activations):
    # Convert activation tensor to a pandas DataFrame
    activation_df = pd.DataFrame(activation_tensor.cpu().numpy())
    # Define the filename for the current token's activations
    filename = f"{i}_activations.csv"
    # Save the DataFrame to a CSV file
    activation_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False, header=False)

# Get the list of generated tokens after the input text
generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0][len(tokenizer.encode(sample_text)):])

# Create a dictionary that includes tokens and their activation vectors
activations_dict = {i: (token, final_activations[i]) for i, token in enumerate(generated_tokens)}

# Print the output text
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Output text: {output_text}\n")

# Print the activations dictionary
print("Activations Dictionary:")
for index, (token, activation_vector) in activations_dict.items():
    # Replace 'Ċ' with "<newline>" or remove the special 'Ġ' character
    clean_token = "<newline>" if token == 'Ċ' else token.lstrip('Ġ')
    print(f"Index: {index}, Token: '{clean_token}', Activation Vector Size: {activation_vector.size()}")