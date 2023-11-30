#########################
######### Setup #########
#########################
print(">> Setting up program...")

# Import dependencies
import torch
from transformer_lens import HookedTransformer
from typing import Dict, Union, List
import numpy as np

# Settings from the paper
SEED = 0
sampling_kwargs = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)
extraction_layer = 6
STEERING_PROMPTS_PATH = "./prompts/wedding_tokens.txt"

# Load the model
print(">> Loading model...")
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-xl")
model.eval()
if torch.cuda.is_available():
    model.to('cuda')

#########################
# Activation Extraction #
#########################
print(">> Extracting activations from steering prompts...")

# Open the steering prompts file and read all prompts into a list
steering_prompts = []
with open(STEERING_PROMPTS_PATH, "r") as file:
    for line in file:
        steering_prompts.append(line.strip())

# Add padding to all steering prompts so that all have the same token length
tlen = lambda prompt: model.to_tokens(prompt).shape[1]
pad_right = lambda prompt, length: prompt + " " * (length - tlen(prompt))
l = max([tlen(p) for p in steering_prompts])
steering_prompts = [pad_right(p, l) for p in steering_prompts]

def get_resid_pre(prompt: str, layer: int):
    """
    Retrieves the pre-activation residuals from a specific layer in the model.
    These are the activations from the residual streams before they enter the
    Multi-Head Attention layer/block.

    Args:
        prompt (str): The input prompt for the model.
        layer (int): The layer number from which to retrieve the residuals.

    Returns:
        torch.Tensor: The pre-activation residuals from the specified layer.
    """
    name = f"blocks.{layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]

# Cache activations for all steering prompts into a numpy structure
# The final structure will be of shape (num_steering_prompts, num_tokens, num_activations)
activations = []
for prompt in steering_prompts:
    prompt_activations = get_resid_pre(prompt, extraction_layer)
    prompt_activations = np.squeeze(prompt_activations.detach().cpu().numpy())
    print(str(prompt_activations.shape) + ": Activation vector for prompt: \"" + prompt + "\"")
    activations.append(prompt_activations)
activations = np.array(activations)

print(str(activations.shape) + ": Activations matrix for all steering prompts")

##########################
## Compute Conceptor(s) ##
##########################
print(">> Computing conceptor...")

# TODO: Compute conceptors using activations matrix from above
# If computed per token this would return a numpy array of shape (num_tokens, num_activations, num_activations)

##########################
### Apply Conceptor(s) ###
##########################

# TODO: Apply conceptors to prompts and generate text

# Assume we have a conceptor steering matrix for each token of shape (num_tokens, num_activations, num_activations)
exit()
# The code below is from the original exampleActivationAddition.py file and needs to be modified for conceptor steering
# Generate from modified model
def ave_hook(resid_pre, hook):
    # Makes sure only prompt tokens are modified
    if resid_pre.shape[1] == 1: 
        return

    # We only add to the prompt (first call), not the generated tokens.
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    print(f"Prompt tokens: {ppos}, Activation tokens: {apos}")
    assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

    # add to the beginning (position-wise) of the activations
    resid_pre[:, :apos, :] += coeff * act_diff

def hooked_generate(prompt_batch: List[str], fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        r = model.generate(input=tokenized, max_new_tokens=50, do_sample=True, **kwargs)
    return r

editing_hooks = [(f"blocks.{act_name}.hook_resid_pre", ave_hook)]
res = hooked_generate([prompt] * 4, editing_hooks, seed=SEED, **sampling_kwargs)

# Print results, removing the ugly beginning of sequence token
res_str = model.to_string(res[:, 1:])
print(("\n\n" + "-" * 80 + "\n\n").join(res_str))