# Import dependencies
import torch
from transformer_lens import HookedTransformer
from typing import Dict, Union, List

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
prompt_add, prompt_sub = "Love", "Hate"
coeff = 5
act_name = 6
prompt = "I hate you because"

# Padding
tlen = lambda prompt: model.to_tokens(prompt).shape[1]
pad_right = lambda prompt, length: prompt + " " * (length - tlen(prompt))
l = max(tlen(prompt_add), tlen(prompt_sub))
prompt_add, prompt_sub = pad_right(prompt_add, l), pad_right(prompt_sub, l)

print(f"'{prompt_add}'", f"'{prompt_sub}'")

# Get activations
def get_resid_pre(prompt: str, layer: int):
    name = f"blocks.{layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]

act_add = get_resid_pre(prompt_add, act_name)
act_sub = get_resid_pre(prompt_sub, act_name)
act_diff = act_add - act_sub
print(act_diff.shape)

# Generate from modified model
def ave_hook(resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return  # caching in model.generate for new tokens

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