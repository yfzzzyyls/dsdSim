import os
import torch
from transformers import AutoModelForCausalLM

# Define the default max sequence length for compiled models (should match main.py)
DEFAULT_MAX_SEQ_LEN = 256

def load_model(model_id: str):
    """
    Load a model for inference. If a compiled Neuron model exists in the project root, 
    load that (for optimal performance). Otherwise, load the model normally.
    Returns a tuple: (model, compiled_flag).
    """
    base_name = os.path.basename(os.path.normpath(model_id))
    compiled_filename = f"{base_name}_neuron_bf16_{DEFAULT_MAX_SEQ_LEN}.pt"
    compiled_path = os.path.join(os.path.dirname(__file__), "..", compiled_filename)
    compiled_path = os.path.normpath(compiled_path)

    if os.path.isfile(compiled_path):
        # Load compiled TorchScript model
        print(f"[Model Loader] Loading compiled model from {compiled_filename}")
        model = torch.jit.load(compiled_path)
        model.eval()
        compiled_flag = True
    else:
        # Load model normally (not compiled)
        print(f"[Model Loader] Compiled model not found. Loading model '{model_id}' to CPU (bf16)...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.eval()
        compiled_flag = False
    return model, compiled_flag
