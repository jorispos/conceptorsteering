import numpy as np
import torch
from typing import List


def load_steering_prompts(model, path):
    """
    Loads the steering prompts from a file and adds padding to all prompts so
    that all have the same token length.
    """
    steering_prompts = []
    with open(path, "r") as file:
        for line in file:
            steering_prompts.append(line.strip())

    tlen = lambda prompt: model.to_tokens(prompt).shape[1]
    pad_right = lambda prompt, length: prompt + " " * (length - tlen(prompt))
    l = max([tlen(p) for p in steering_prompts])
    steering_prompts = [pad_right(p, l) for p in steering_prompts]

    return steering_prompts


def get_resid_pre(model, prompt: str, layer: int):
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


def extract_activations(model, steering_prompts, extraction_layer, device):
    """
    Extract activations from the given model for the given steering prompts.

    Parameters:
    - model (HookedTransformer): The model to use for generating text.
    - steering_prompts (list): List of steering prompts to extract activations for.
    - extraction_layer (int): The layer to extract activations from.
    - device (str): The device to use for computations.

    Returns:
    - torch.Tensor: The activations for the given steering prompts.
    """
    activations = []  # (n_prompts, n_tokens, n_activations)
    for prompt in steering_prompts:
        prompt_activations = get_resid_pre(model, prompt, extraction_layer)
        prompt_activations = np.squeeze(prompt_activations.detach().cpu().numpy())
        print(str(prompt_activations.shape) + ": Activation vector for prompt: \"" + prompt + "\"")
        activations.append(prompt_activations)
    activations = np.array(activations)
    return torch.Tensor(activations, device=device)


def hooked_generate(model, prompt_batch: List[str], fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        r = model.generate(input=tokenized, max_new_tokens=50, do_sample=True, **kwargs)
    return r


def steer(C, x, beta):
    """
    Steers the given vector x using the conceptor C.

    Args:
        C (torch.Tensor): The conceptor matrix.
        x (torch.Tensor): The vector to be steered.
        beta (float): The steering parameter with 0: no steering, 1: full steering.

    Returns:
        torch.Tensor: The steered vector.
    """
    assert 0 <= beta <= 1, f"beta must be between 0 and 1, but was {beta}"
    return beta * torch.matmul(C, x) + (1 - beta) * x


def generate_ave_hook(steering_matrices, beta=1.0):
    """
    Generates a hook that applies the conceptor to the activations of the
    prompt tokens.

    Args:
        steering_matrices (torch.Tensor): conceptor matrices for each token,
            shape (n_tokens, n_emb, n_emb)
        beta (float): The steering parameter with 0: no steering, 1: full steering.
    """
    def ave_hook(resid_pre, hook):
        """
        Applies the conceptor to the activations of the prompt tokens.

        Args:
            resid_pre (torch.Tensor): The pre-activation residuals from the
                specified layer, shape (n_tokens, n_prompts, n_emb)
            hook (Hook): The hook that called this function.
        """
        # Makes sure only prompt tokens are modified
        if resid_pre.shape[1] == 1:
            return

        # resid_pre contains the activations to be modified using the conceptor
        # This is an example where there is a steering conceptor matrix for each token
        prompt_to_steer_length, steering_matrices_length = resid_pre.shape[1], steering_matrices.shape[0]
        print(f"Prompt tokens: {prompt_to_steer_length}, Steering tokens: {steering_matrices_length}")
        assert steering_matrices_length <= prompt_to_steer_length, f"More steering tokens ({steering_matrices_length}) than prompt tokens ({prompt_to_steer_length})!"

        # Dot product is taken with correct conceptor matrix
        for i in range(steering_matrices.shape[0]):
            for j in range(resid_pre.shape[0]):
                # TODO: batch this properly
                resid_pre[j, i, :] = steer(steering_matrices[i], resid_pre[j, i, :], beta=beta)

    return ave_hook
