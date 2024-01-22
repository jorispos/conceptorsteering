import argparse
import numpy as np
import torch
from transformer_lens import HookedTransformer
from typing import List
from functools import partial
from utils import compute_conceptor


def load_steering_prompts(path):
    # Open the steering prompts file and read all prompts into a list
    steering_prompts = []
    with open(path, "r") as file:
        for line in file:
            steering_prompts.append(line.strip())

    # Add padding to all steering prompts so that all have the same token length
    tlen = lambda prompt: model.to_tokens(prompt).shape[1]
    pad_right = lambda prompt, length: prompt + " " * (length - tlen(prompt))
    l = max([tlen(p) for p in steering_prompts])
    steering_prompts = [pad_right(p, l) for p in steering_prompts]

    return steering_prompts


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


def hooked_generate(prompt_batch: List[str], fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        r = model.generate(input=tokenized, max_new_tokens=50, do_sample=True, **kwargs)
    return r


def generate_ave_hook(steering_matrices):
    def ave_hook(resid_pre, hook):

            # TODO: implement this

            # Makes sure only prompt tokens are modified
            if resid_pre.shape[1] == 1:
                return

            # resid_pre contains the activations to be modified using the conceptor
            # This is an example where there is a steering conceptor matrix for each token
            prompt_to_steer_length, steering_matrices_length = resid_pre.shape[1], steering_matrices.shape[1]
            print(f"Prompt tokens: {prompt_to_steer_length}, Steering tokens: {steering_matrices_length}")
            assert steering_matrices_length <= prompt_to_steer_length, f"More steering tokens ({steering_matrices_length}) than prompt tokens ({prompt_to_steer_length})!"

            # TODO : modify line below so that dot product is taken with correct conceptor matrix instead of addition
            # resid_pre[:, :apos, :] += coeff * act_diff
    
    return ave_hook


def _parse_args():
    parser = argparse.ArgumentParser(description='Script Configuration')

    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='Name of the model to load')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--steering_prompts_path', type=str, default='./prompts/wedding_tokens.txt', help='Path to steering prompts file')
    parser.add_argument('--prompt_to_steer', type=str, default='I am going to a ', help='Prompt to test steering')

    # conceptor params
    parser.add_argument('--aperture', type=float, default=10.0, help='Aperture for the conceptor')

    # sampling kwargs (settings from paper) -- TODO @joris: which paper?
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.3, help='Top p for sampling')
    parser.add_argument('--freq_penalty', type=float, default=1.0, help='Frequency penalty')
    parser.add_argument('--extraction_layer', type=int, default=6, help='Extraction layer number')

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # Load the model
    print(">> Loading model...")
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    # Activation Extraction
    print(">> Extracting activations from steering prompts...")
    steering_prompts = load_steering_prompts(args.steering_prompts_path)

    activations = []  # (n_prompts, n_tokens, n_activations)
    for prompt in steering_prompts:
        prompt_activations = get_resid_pre(prompt, args.extraction_layer)
        prompt_activations = np.squeeze(prompt_activations.detach().cpu().numpy())
        print(str(prompt_activations.shape) + ": Activation vector for prompt: \"" + prompt + "\"")
        activations.append(prompt_activations)
    activations = np.array(activations)

    print(str(activations.shape) + ": Activations matrix for all steering prompts")

    # Compute Conceptors - one conceptor per token (aggregated over all given prompts)
    print(">> Computing conceptor...")
    steering_matrices = np.array([
        compute_conceptor(activations[:, idx, :], aperture=args.aperture)
        for idx in range(activations.shape[1])
    ])
    # shape: (num_tokens, num_activations, num_activations)

    ######################################
    ### Steer prompts using Conceptors ###
    ######################################
    print(">> Steering prompts using conceptors...")
    ave_hook = generate_ave_hook(steering_matrices=steering_matrices)
    sampling_kwargs = dict(temperature=args.temperature, top_p=args.top_p, freq_penalty=args.freq_penalty)
    editing_hooks = [(f"blocks.{args.extraction_layer}.hook_resid_pre", ave_hook)]
    res = hooked_generate([args.prompt_to_steer] * 4, editing_hooks, seed=args.seed, **sampling_kwargs)

    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))
