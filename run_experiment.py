import numpy as np
import torch
from transformer_lens import HookedTransformer
from utils import compute_conceptor
import os
from absl import app, flags
from datetime import datetime
from conceptor_steering import load_steering_prompts, get_resid_pre
from conceptor_steering import hooked_generate, generate_ave_hook

# Define the flags
FLAGS = flags.FLAGS

# Define the flags
flags.DEFINE_string('model_name', 'gpt2-xl', 'Name of the model to load')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('steering_prompts_path', './prompts/wedding_tokens.txt', 'Path to steering prompts file')
flags.DEFINE_string('prompt_to_steer', 'I am going to a ', 'Prompt to test steering')
flags.DEFINE_integer('n_steered_examples', 4, 'How many times to steer the same prompt')

# Experiment parameters
flags.DEFINE_list('apertures', [1e-2, 1e-1, 1, 10, 50, 100], 'List of aperture values to try')

# Sampling kwargs (settings from paper) -- TODO @joris: which paper?
flags.DEFINE_float('temperature', 1.0, 'Temperature for sampling')
flags.DEFINE_float('top_p', 0.3, 'Top p for sampling')
flags.DEFINE_float('freq_penalty', 1.0, 'Frequency penalty')
flags.DEFINE_integer('extraction_layer', 6, 'Extraction layer number')


def run_experiment(model, activations, aperture, sampling_kwargs, folder_path, device):
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
    steering_matrices = torch.Tensor(steering_matrices, device=device)

    # Generate ave_hook with the steering matrices
    ave_hook = generate_ave_hook(steering_matrices=steering_matrices)

    # Define editing_hooks using ave_hook
    editing_hooks = [(f"blocks.{FLAGS.extraction_layer}.hook_resid_pre", ave_hook)]

    # Generate text using the modified model
    steered_res = hooked_generate(model, [FLAGS.prompt_to_steer] * FLAGS.n_steered_examples, fwd_hooks=editing_hooks, seed=FLAGS.seed, **sampling_kwargs)

    # Save results with aperture value in the filename
    save_results(model, steered_res, aperture, folder_path)


def save_results(model, steered_res, aperture, folder_path):
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


def batch_run(model, activations, apertures, folder_path, device):
    """
    Run batch experiments for each aperture.

    Parameters:
    - apertures (list): List of apertures to run experiments for.
    - args (object): Object containing experiment arguments.
    - folder_path (str): Path to the folder where results will be saved.
    """
    # Generate and save unsteered results once
    sampling_kwargs = dict(temperature=FLAGS.temperature, top_p=FLAGS.top_p, freq_penalty=FLAGS.freq_penalty)
    unsteered_res = hooked_generate(model, [FLAGS.prompt_to_steer] * FLAGS.n_steered_examples, seed=FLAGS.seed, **sampling_kwargs)
    unsteered_str = model.to_string(unsteered_res[:, 1:])
    unsteered_file_path = os.path.join(folder_path, "unsteered_results.txt")
    with open(unsteered_file_path, 'w') as file:
        file.write(("\n\n" + "-" * 80 + "\n\n").join(unsteered_str))

    # Run experiments for each aperture
    for aperture in apertures:
        run_experiment(model, activations, aperture, sampling_kwargs, folder_path, device)


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
    steering_prompts = load_steering_prompts(model, FLAGS.steering_prompts_path)

    activations = []  # (n_prompts, n_tokens, n_activations)
    for prompt in steering_prompts:
        prompt_activations = get_resid_pre(model, prompt, FLAGS.extraction_layer)
        prompt_activations = np.squeeze(prompt_activations.detach().cpu().numpy())
        print(str(prompt_activations.shape) + ": Activation vector for prompt: \"" + prompt + "\"")
        activations.append(prompt_activations)
    activations = np.array(activations)

    print(str(activations.shape) + ": Activations matrix for all steering prompts")

    # Run batch experiments
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = os.path.join("resultsdata", timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # store current experiment flags
    with open(os.path.join(folder_path, 'flags.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())

    batch_run(model, activations, FLAGS.apertures, folder_path, device=DEVICE)

if __name__ == '__main__':
    app.run(main)