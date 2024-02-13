from transformer_lens import HookedTransformer
import torch

model_name = "EleutherAI/gpt-j-6B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
model.eval();

print("successfully loaded GPT-J-6B model!")