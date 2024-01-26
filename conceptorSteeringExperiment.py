import argparse
import numpy as np
import torch
from transformer_lens import HookedTransformer
from typing import List
from functools import partial
from utils import compute_conceptor
import os
from datetime import datetime


def run_experiment(aperture, args, sampling_kwargs, folder_path):
    """
    Run an experiment with the given aperture value.

    Parameters:
    - aperture (float): The aperture value for computing Conceptors.
    - args (Namespace): The command-line arguments.
    - sampling_kwargs (dict): Keyword arguments for generating text.
    - folder_path (str): The path to the folder for saving results.

    Returns:
    None
    """
    print(f"Running experiment with aperture: {aperture}")

    # Compute Conceptors with the given aperture value
    steering_matrices = np.array([
        compute_conceptor(activations[:, idx, :], aperture=aperture)
        for idx in range(activations.shape[1])
    ])
    steering_matrices = torch.Tensor(steering_matrices, device=DEVICE)

    # Generate ave_hook with the steering matrices
    ave_hook = generate_ave_hook(steering_matrices=steering_matrices)

    # Define editing_hooks using ave_hook
    editing_hooks = [(f"blocks.{args.extraction_layer}.hook_resid_pre", ave_hook)]

    # Generate text using the modified model
    steered_res = hooked_generate([args.prompt_to_steer] * args.n_steered_examples, fwd_hooks=editing_hooks, seed=args.seed, **sampling_kwargs)

    # Save results with aperture value in the filename
    save_results(steered_res, aperture, args, folder_path)


def save_results(steered_res, aperture, args, folder_path):
    """
    Save the steered results to a text file.

    Parameters:
    - steered_res (numpy.ndarray): The steered results.
    - aperture (float): The aperture value.
    - args (list): Additional arguments.
    - folder_path (str): The folder path to save the results.

    Returns:
    None
    """
    # Convert to string
    steered_str = model.to_string(steered_res[:, 1:])

    # Generate file path for steered results
    steered_file_path = os.path.join(folder_path, f"steered_results_aperture_{aperture}.txt")

    # Store steered results in the text file
    with open(steered_file_path, 'w') as file:
        file.write(("\n\n" + "-" * 80 + "\n\n").join(steered_str))


def batch_run(apertures, args, folder_path):
    """
    Run batch experiments for each aperture.

    Parameters:
    - apertures (list): List of apertures to run experiments for.
    - args (object): Object containing experiment arguments.
    - folder_path (str): Path to the folder where results will be saved.
    """
    # Generate and save unsteered results once
    sampling_kwargs = dict(temperature=args.temperature, top_p=args.top_p, freq_penalty=args.freq_penalty)
    unsteered_res = hooked_generate([args.prompt_to_steer] * args.n_steered_examples, seed=args.seed, **sampling_kwargs)
    unsteered_str = model.to_string(unsteered_res[:, 1:])
    unsteered_file_path = os.path.join(folder_path, "unsteered_results.txt")
    with open(unsteered_file_path, 'w') as file:
        file.write(("\n\n" + "-" * 80 + "\n\n").join(unsteered_str))

    # Run experiments for each aperture
    for aperture in apertures:
        run_experiment(aperture, args, sampling_kwargs, folder_path)


def load_steering_prompts(path):
    """
    Load steering prompts from a file and return a list of prompts.

    Args:
        path (str): The path to the file containing the steering prompts.

    Returns:
        list: A list of steering prompts.

    """
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
    """
    Generates new tokens based on a given prompt batch using the model.

    Args:
        prompt_batch (List[str]): List of input prompts.
        fwd_hooks (list, optional): List of forward hooks. Defaults to [].
        seed (int, optional): Seed for random number generation. Defaults to None.
        **kwargs: Additional keyword arguments for token generation.

    Returns:
        r: Generated tokens.

    """
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        r = model.generate(input=tokenized, max_new_tokens=50, do_sample=True, **kwargs)
    return r


def generate_ave_hook(steering_matrices):
    """
    Generates an average hook function that modifies prompt tokens based on steering matrices.

    Args:
        steering_matrices (torch.Tensor): The steering matrices used for modifying prompt tokens.

    Returns:
        ave_hook (function): The average hook function that modifies prompt tokens.
    """
    def ave_hook(resid_pre, hook):
        # Makes sure only prompt tokens are modified
        if resid_pre.shape[1] == 1:
            return

        prompt_to_steer_length, steering_matrices_length = resid_pre.shape[1], steering_matrices.shape[0]
        print(f"Prompt tokens: {prompt_to_steer_length}, Steering tokens: {steering_matrices_length}")
        assert steering_matrices_length <= prompt_to_steer_length, f"More steering tokens ({steering_matrices_length}) than prompt tokens ({prompt_to_steer_length})!"

        for i in range(steering_matrices.shape[0]):
            for j in range(resid_pre.shape[0]):
                resid_pre[j, i, :] = torch.matmul(steering_matrices[i], resid_pre[j, i, :])

    return ave_hook


def _parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Script Configuration')

    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='Name of the model to load')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--steering_prompts_path', type=str, default='./prompts/wedding_tokens.txt', help='Path to steering prompts file')
    parser.add_argument('--prompt_to_steer', type=str, default='I am going to a ', help='Prompt to test steering')
    parser.add_argument('--n_steered_examples', type=int, default=4, help='How many times to steer the same prompt')

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

    # mps is not yet supported
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    print(">> Loading model...")
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model_name, device=DEVICE)
    model.eval()

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

    # List of aperture values for experiments
    aperture_values = [1e-2, 1e-1, 1, 10, 50, 100]  # Example values

    # Run batch experiments
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = os.path.join("resultsdata", timestamp)
    os.makedirs(folder_path, exist_ok=True)

    batch_run(aperture_values, args, folder_path)