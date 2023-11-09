import tkinter as tk
from tkinter import scrolledtext
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.eval()  # Put the model in evaluation mode

# Set pad token to eos token
# This is necessary because GPT-2 does not have a native pad token
tokenizer.pad_token = tokenizer.eos_token

# Set the color scheme
dark_background = "#2D2D2D"
light_text = "#D3D3D3"
input_bg = "#1E1E1E"  # Darker background for text input
output_bg = "#252526"  # Slightly different background color for output to distinguish
button_bg = "#3C3C3C"
button_fg = "#FFFFFF"

# Initialize the GUI window with a dark background
window = tk.Tk()
window.title("Conceptor-based Activation Steering")
window.configure(bg=dark_background)

# Dictionary to store activations from all layers
activations = {}

# Function to be called by the hook; stores activations in the dictionary
def get_activations(layer_index):
    def hook(model, inputs, outputs):
        activations[layer_index] = outputs[0].detach()
    return hook

# Register the hook for all transformer layers
for i, layer in enumerate(model.transformer.h):
    layer.register_forward_hook(get_activations(i))

# Function to display activations of the selected layer
def show_layer_activations(event):
    widget = event.widget
    if not widget.curselection():  # If nothing is selected
        return
    selection_index = int(widget.curselection()[0])
    layer_activation = activations[selection_index]
    activation_display.delete('1.0', tk.END)  # Clear the current content
    # Get the shape of the layer activation (batch_size, sequence_length, feature_size)
    activation_shape = layer_activation.shape
    # Create a string representation of the activations tensor
    activation_text = '\n'.join([' '.join(f'{activation:.4f}' for activation in token_activations) for token_activations in layer_activation[0]])
    activation_display.insert(tk.END, activation_text)
    # Update the activation_display_label with the dimensions of the activations
    activation_display_label.config(text=f"Activations (Layer {selection_index} - Seq Length: {activation_shape[1]}, Feature Size: {activation_shape[2]}):")

# Function to run model, store activations, and display model output
def run_model():
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Send the model to the chosen device
    model.to(device)

    # Get input text from the text entry and process it
    input_text = text_entry.get("1.0", tk.END).strip()
    encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Send input tensors to the same device as model
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Clear previous activations
    activations.clear()
    
    # Disable gradient calculation
    with torch.no_grad():
        # Use the generate function with beam search
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=5,  # Number of beams for beam search
            length_penalty=2.0,  # Length penalty to use with beam search
            early_stopping=True  # Stop generation when all beams reach the EOS token
        )

    # Decode the generated ids to text and display
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    model_output_display.delete('1.0', tk.END)
    model_output_display.insert(tk.END, generated_text)
    
    # Populate the listbox with layers
    layers_list.delete(0, tk.END)
    for i in range(len(activations)):
        layers_list.insert(tk.END, f'Layer {i}')

# Function to reset the GUI for a new input
def reset_gui():
    text_entry.delete('1.0', tk.END)  # Clear the text entry
    layers_list.delete(0, tk.END)    # Clear the layers list
    activation_display.delete('1.0', tk.END)  # Clear the activations display
    model_output_display.delete('1.0', tk.END)  # Clear the model output display
    activations.clear()  # Clear the activations dictionary

# GUI elements
text_entry_label = tk.Label(window, text="Enter text:", fg=light_text, bg=dark_background)
text_entry_label.pack()

text_entry = tk.Text(window, height=3, width=50, bg=input_bg, fg=light_text, insertbackground=light_text)
text_entry.pack()

button_frame = tk.Frame(window, bg=dark_background)
button_frame.pack()

run_button = tk.Button(button_frame, text="Run Model", command=run_model, bg=button_bg, fg=button_fg)
run_button.pack(side=tk.LEFT)

reset_button = tk.Button(button_frame, text="Reset", command=reset_gui, bg=button_bg, fg=button_fg)
reset_button.pack(side=tk.LEFT)

layers_label = tk.Label(window, text="Layers:", fg=light_text, bg=dark_background)
layers_label.pack()

layers_list = tk.Listbox(window, bg=output_bg, fg=light_text, selectbackground=button_bg, selectforeground=button_fg)
layers_list.pack()
layers_list.bind('<<ListboxSelect>>', show_layer_activations)

activation_display_label = tk.Label(window, text="Activations:", fg=light_text, bg=dark_background)
activation_display_label.pack()

activation_display = scrolledtext.ScrolledText(window, height=15, width=100, bg=output_bg, fg=light_text)
activation_display.pack()

model_output_label = tk.Label(window, text="Model Output:", fg=light_text, bg=dark_background)
model_output_label.pack()

model_output_display = scrolledtext.ScrolledText(window, height=4, width=50, bg=output_bg, fg=light_text)
model_output_display.pack()

# Run the GUI loop
window.mainloop()