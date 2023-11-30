# Import dependencies
import torch
from transformer_lens import HookedTransformer
from typing import Dict, Union, List
import transformer_lens

# Load the model
torch.set_grad_enabled(False)  # save memory
model = HookedTransformer.from_pretrained("gpt2-xl")
model.eval()
if torch.cuda.is_available():
  model.to('cuda')

# Settings from the paper
SEED = 0
sampling_kwargs = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)

# Specific to the love/hate example (For ActAdd)
prompts = [
    "wedding",
    "weddings",
    "wed",
    "marry",
    "married",
    "marriage",
    "bride",
    "groom",
    "honeymoon"
]

coeff = 5
act_name = 6
prompt = "I am going to a "

# Padding
tlen = lambda prompt: model.to_tokens(prompt).shape[1]
pad_right = lambda prompt, length: prompt + " " * (length - tlen(prompt))
l = max([tlen(p) for p in prompts])
prompts = [pad_right(p, l) for p in prompts]

for prompt in prompts:
    print(f"'{prompt}'")

# Get activations
def get_resid_pre(prompt: str, layer: int):
    name = f"blocks.{layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]

for prompt in prompts:
    activations = get_resid_pre(prompt, act_name)
    print(activations.shape)