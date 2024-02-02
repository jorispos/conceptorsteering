import json
import os
import torch
import pandas as pd
from transformer_lens import HookedTransformer
from conceptor import compute_conceptor
from absl import app, flags
from datetime import datetime
from conceptor_steering import load_steering_prompts, extract_activations
from conceptor_steering import hooked_generate, generate_ave_hook

# Define the flags
FLAGS = flags.FLAGS

# Define the flags
flags.DEFINE_string('model_name', 'gpt2-xl', 'Name of the model to load')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('steering_prompts_path', './prompts/wedding_tokens.txt', 'Path to steering prompts file')
flags.DEFINE_string('prompt_to_steer', 'I am going to a ', 'Prompt to test steering')
flags.DEFINE_integer('n_steered_examples', 4, 'How many times to steer the same prompt')

# flag to define if batch (multiple experiments) or single experiment
flags.DEFINE_boolean('batch', False, 'Run multiple experiments in one batch')

# Experiment parameters
flags.DEFINE_float('aperture', 10, 'Conceptor aperture to use')
flags.DEFINE_float('beta', 0.5, 'Conceptor steering parameter (0 no steering, 1 full steering)')

# Batched experiment parameters (list of values that are all run)
flags.DEFINE_list('apertures', [1e-2, 1e-1, 1, 10, 50, 100], 'List of aperture values to try')
flags.DEFINE_list('betas', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 'List of conceptor steering parameters')

# Sampling kwargs (settings from paper) -- TODO @joris: which paper?
flags.DEFINE_float('temperature', 1.0, 'Temperature for sampling')
flags.DEFINE_float('top_p', 0.3, 'Top p for sampling')
flags.DEFINE_float('freq_penalty', 1.0, 'Frequency penalty')
flags.DEFINE_integer('extraction_layer', 6, 'Extraction layer number')


def run_experiment(model, activations, aperture, beta, sampling_kwargs, folder_path, device):
    """
    Run an experiment with the given aperture value.

    Parameters:
    - model (HookedTransformer): The model to use for generating text.
    - activations (torch.Tensor): The activations to use for computing Conceptors.
    - aperture (float): The aperture value for computing Conceptors.
    - sampling_kwargs (dict): Keyword arguments for generating text.
    - folder_path (str): The path to the folder for saving results.
    - device (str): The device to use for computations.

    Returns:
    None
    """
    # Compute Conceptors with the given aperture value
    steering_matrices = torch.stack([
        compute_conceptor(activations[:, idx, :], aperture=aperture)
        for idx in range(activations.shape[1])
    ])

    # Generate hook with the steering matrices
    ave_hook = generate_ave_hook(steering_matrices=steering_matrices, beta=beta)
    editing_hooks = [(f"blocks.{FLAGS.extraction_layer}.hook_resid_pre", ave_hook)]

    # Generate text using the conceptor-steered model
    steered_res = hooked_generate(model, [FLAGS.prompt_to_steer] * FLAGS.n_steered_examples, fwd_hooks=editing_hooks, seed=FLAGS.seed, **sampling_kwargs)

    # Save results with aperture value in the filename
    steered_str = model.to_string(steered_res[:, 1:])
    return steered_str


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
    activations = extract_activations(model, steering_prompts, FLAGS.extraction_layer, DEVICE)
    print(str(activations.shape) + ": Activations matrix for all steering prompts")

    # Run batch experiments
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = os.path.join("resultsdata", timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # store current experiment flags
    with open(os.path.join(folder_path, 'flags.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # Generate and save unsteered results once
    sampling_kwargs = dict(temperature=FLAGS.temperature, top_p=FLAGS.top_p, freq_penalty=FLAGS.freq_penalty)
    unsteered_res = hooked_generate(model, [FLAGS.prompt_to_steer] * FLAGS.n_steered_examples, seed=FLAGS.seed, **sampling_kwargs)
    unsteered_str = model.to_string(unsteered_res[:, 1:])

    # save results to file
    results = [
        {
            "aperture": None,
            "beta": None,
            "steered": False,
            "input": FLAGS.prompt_to_steer,
            "output": unsteered_str[idx],
        }
        for idx in range(FLAGS.n_steered_examples)
    ]
    pd.DataFrame(results).to_csv(os.path.join(folder_path, "results.csv"))

    if FLAGS.batch:
        # Run experiments for each aperture and beta
        for aperture in FLAGS.apertures:
            for beta in FLAGS.betas:
                aperture = float(aperture)
                beta = float(beta)
                print(f"Running experiment with alpha={aperture}, beta={beta}")
                outputs = run_experiment(model, activations, aperture, beta, sampling_kwargs, folder_path, DEVICE)

                # save results to file
                for output in outputs:
                    results.append({
                        "aperture": aperture,
                        "beta": beta,
                        "steered": True,
                        "input": FLAGS.prompt_to_steer,
                        "output": output,
                    })
                pd.DataFrame(results).to_csv(os.path.join(folder_path, "results.csv"))

    else:
        # Run single experiment
        print(">> Running single experiment...")
        outputs = run_experiment(model, activations, FLAGS.aperture, FLAGS.beta, sampling_kwargs, folder_path, DEVICE)

        # save results to file
        for output in outputs:
            results.append({
                "aperture": FLAGS.aperture,
                "beta": FLAGS.beta,
                "steered": True,
                "input": FLAGS.prompt_to_steer,
                "output": output,
            })
        pd.DataFrame(results).to_csv(os.path.join(folder_path, "results.csv"))


if __name__ == '__main__':
    app.run(main)
