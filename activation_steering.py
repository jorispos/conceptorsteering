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
TARGET_LAYER = 6  # Target layer to apply conceptor

_torch_module_call = torch.nn.Module.__call__

class LayerTracker:
    def __init__(self) -> None:
        self.current_layer = 0  # Initialize to 0

def module_forward_wrapper(conceptors: Dict[int, torch.Tensor], layer_tracker: LayerTracker) -> Callable[..., Any]:
    def my_forward(mod: nn.Module, *args, **kwargs) -> Any:
        out = _torch_module_call(mod, *args, **kwargs)

        if isinstance(mod, GPT2Block):
            if layer_tracker.current_layer == TARGET_LAYER:
                c = conceptors.get(layer_tracker.current_layer, None)
                if c is not None:
                    # Assuming the first element of the tuple is the activations
                    activations = out[0]
                    # Apply the conceptor
                    new_activations = torch.matmul(activations, c)
                    # Reassemble the output tuple
                    out = (new_activations,) + out[1:]

            layer_tracker.current_layer += 1

        return out

    return my_forward


class ConceptorSteering:
    """Context manager to apply conceptors in the forward pass."""

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


def process_prompt(model, tokenizer, text, device, conceptors=None, max_length=50, use_steering=True):
    """
    Process a single prompt to generate text auto-regressively.
    Args:
        model (torch.nn.Module): The GPT-2 model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for GPT-2.
        text (str): The input prompt text.
        device (torch.device): The device to run the model on (CPU or GPU).
        conceptors (Dict[int, torch.Tensor], optional): Dictionary of conceptors. None for unsteered generation.
        max_length (int): Maximum length of generated text in tokens.
        use_steering (bool): Flag to apply conceptor steering.
    """
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    model.eval()

    generated_text = []
    steering_context = ConceptorSteering(model, conceptors) if use_steering and conceptors else None
    with torch.no_grad():
        if steering_context:
            with steering_context:
                generated_text = _generate_text(model, input_ids, max_length, tokenizer)
        else:
            generated_text = _generate_text(model, input_ids, max_length, tokenizer)

    return generated_text


def _generate_text(model, input_ids, max_length, tokenizer):
    """
    Helper function to generate text auto-regressively.
    """
    generated_text = []
    for _ in range(max_length):
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        generated_text.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_text, skip_special_tokens=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    # Initialize the identity matrix conceptor for the target layer
    identity_matrix = torch.eye(1600)  # Adjust as needed
    conceptors = {TARGET_LAYER: identity_matrix}

    prompt = "This research project is about"
    print("Prompt:\n", prompt)

    # Generate steered output
    steered_output = process_prompt(model, tokenizer, prompt, device, conceptors, max_length=50, use_steering=True)
    print("------------------------------\nSteered Completion: (identity matrix for now)\n", steered_output)

    # Generate unsteered output
    unsteered_output = process_prompt(model, tokenizer, prompt, device, max_length=50, use_steering=False)
    print("------------------------------\nUnsteered Completion:\n", unsteered_output)