import os
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from typing import Callable, Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
LAYER_COUNT = 48  # Number of layers in the GPT-2 XL model
MODEL_NAME = 'gpt2-xl'


_torch_module_call = torch.nn.Module.__call__


class LayerTracker:
    def __init__(self) -> None:
        self.current_layer = 0


def module_forward_wrapper(
        conceptors: Dict[int, torch.Tensor], layer_tracker: LayerTracker
) -> Callable[..., Any]:
    def my_forward(mod: nn.Module, *args, **kwargs) -> Any:
        out = _torch_module_call(mod, *args, **kwargs)

        if isinstance(mod, GPT2Block):
            layer_tracker.current_layer += 1

            # Retrieve the conceptor for the current layer
            c = conceptors.get(layer_tracker.current_layer)

            if c is not None:
                # Apply the conceptor: dot product between the conceptor matrix and the activation vector
                out = torch.matmul(c, out)
                print(f"Conceptor applied at layer {layer_tracker.current_layer}")

        return out

    return my_forward


class ConceptorSteering:
    """Context manager to apply conceptors in the forward pass.

    Example:

    ```python
    with ConceptorSteering(model, conceptor) as tracer, torch.no_grad():
        out = model(data)
    ```
    """

    def __init__(self, mod: nn.Module, c: Dict[int, torch.Tensor]) -> None:
        self.original_torch_call = nn.Module.__call__
        self.conceptors = c
        self.layer_tracker = LayerTracker()

    def __enter__(self) -> "ConceptorSteering":
        # Override the torch call method
        nn.Module.__call__ = module_forward_wrapper(self.conceptors, self.layer_tracker)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Restore normal behavior
        nn.Module.__call__ = self.original_torch_call


def process_prompt(model, tokenizer, text, device):
    """
    Process a single prompt to capture and store activations.
    Args:
        model (torch.nn.Module): The GPT-2 model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for GPT-2.
        text (str): The input prompt text.
        activations (list): List to store activations from each layer.
        device (torch.device): The device to run the model on (CPU or GPU).
    """
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    model.eval()
    with ConceptorSteering(model, {}), torch.no_grad():
        output = model(input_ids=input_ids)

    # Select the most likely token IDs from the logits
    predicted_token_ids = torch.argmax(output.logits, dim=-1)

    # Decode the generated token IDs to text
    generated_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

    print("Generated Text:", generated_text)
    return generated_text


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    prompt = "This research project is about"
    process_prompt(model, tokenizer, prompt, device)