import numpy as np
import torch
from absl import app, flags
from transformer_lens import HookedTransformer
from typing import List
from utils import compute_conceptor
import os
from datetime import datetime

# Define the flags
FLAGS = flags.FLAGS

# Define the flags
flags.DEFINE_string('model_name', 'gpt2-xl', 'Name of the model to load')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('steering_prompts_path', './prompts/wedding_tokens.txt', 'Path to steering prompts file')
flags.DEFINE_string('prompt_to_steer', 'I am going to a ', 'Prompt to test steering')
flags.DEFINE_integer('n_steered_examples', 4, 'How many times to steer the same prompt')

# Experiment parameters
flags.DEFINE_float('aperture', 10, 'Conceptor aperture to use')

# Sampling kwargs (settings from paper) -- TODO @joris: which paper?
flags.DEFINE_float('temperature', 1.0, 'Temperature for sampling')
flags.DEFINE_float('top_p', 0.3, 'Top p for sampling')
flags.DEFINE_float('freq_penalty', 1.0, 'Frequency penalty')
flags.DEFINE_integer('extraction_layer', 6, 'Extraction layer number')


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


def hooked_generate(model, prompt_batch: List[str], fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        r = model.generate(input=tokenized, max_new_tokens=50, do_sample=True, **kwargs)
    return r


def generate_ave_hook(steering_matrices):
    """
    Generates a hook that applies the conceptor to the activations of the
    prompt tokens.

    Args:
        steering_matrices (torch.Tensor): conceptor matrices for each token,
            shape (n_tokens, n_emb, n_emb)
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
                resid_pre[j, i, :] = torch.matmul(steering_matrices[i], resid_pre[j, i, :])

    return ave_hook


def main(argv):
    # mps is not yet supported
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    print(">> Loading model...")
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(FLAGS.model_name, device=DEVICE)
    model.eval()

    # Activation Extraction
    print(">> Extracting activations from steering prompts...")
    steering_prompts = load_steering_prompts(FLAGS.steering_prompts_path)

    activations = []  # (n_prompts, n_tokens, n_activations)
    for prompt in steering_prompts:
        prompt_activations = get_resid_pre(prompt, FLAGS.extraction_layer)
        prompt_activations = np.squeeze(prompt_activations.detach().cpu().numpy())
        print(str(prompt_activations.shape) + ": Activation vector for prompt: \"" + prompt + "\"")
        activations.append(prompt_activations)
    activations = np.array(activations)

    print(str(activations.shape) + ": Activations matrix for all steering prompts")

    # Compute Conceptors - one conceptor per token (aggregated over all given prompts)
    print(">> Computing conceptor...")
    steering_matrices = np.array([
        compute_conceptor(activations[:, idx, :], aperture=FLAGS.aperture)
        for idx in range(activations.shape[1])
    ])
    steering_matrices = torch.Tensor(steering_matrices, device=DEVICE)
    # shape: (num_tokens, num_activations, num_activations)

    ######################################
    ### Steer prompts using Conceptors ###
    ######################################
    print(">> Steering prompts using conceptors...")
    ave_hook = generate_ave_hook(steering_matrices=steering_matrices)
    sampling_kwargs = dict(temperature=FLAGS.temperature, top_p=FLAGS.top_p, freq_penalty=FLAGS.freq_penalty)
    editing_hooks = [(f"blocks.{FLAGS.extraction_layer}.hook_resid_pre", ave_hook)]
    res = hooked_generate([FLAGS.prompt_to_steer] * FLAGS.n_steered_examples, editing_hooks, seed=FLAGS.seed, **sampling_kwargs)

    ######################################
    ########### Save results #############
    ######################################

    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, 1:])
    
    # Generate file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join("resultsdata", f"results_{timestamp}.txt")

    # Store results in the text file
    with open(file_path, 'w') as file:
        file.write(("\n\n" + "-" * 80 + "\n\n").join(res_str))


if __name__ == '__main__':
    app.run(main)