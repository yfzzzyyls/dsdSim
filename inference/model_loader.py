import os
import torch
import torch_neuronx
from transformers import AutoModelForCausalLM

# Default maximum sequence length the compiled model supports (should match what was used in compilation)
DEFAULT_MAX_SEQ_LEN = 1024

def _get_compiled_model_path(model_name_or_path: str, max_seq_length: int = DEFAULT_MAX_SEQ_LEN):
    """Generate the expected compiled model file path for a given model name and sequence length."""
    base_name = os.path.basename(os.path.normpath(model_name_or_path))
    return f"{base_name}_neuron_bf16_{max_seq_length}.pt"

def load_model(model_name_or_path: str, use_compiled: bool = True, compile_if_missing: bool = False):
    """
    Load a causal LM model. If use_compiled is True, will load a precompiled Neuron model (TorchScript .pt).
    If the compiled model file is missing and compile_if_missing is True, will compile the model on the fly.
    Otherwise, falls back to loading the model normally (on CPU/GPU).
    """
    if use_compiled:
        # Determine expected compiled model file path
        compiled_path = _get_compiled_model_path(model_name_or_path, max_seq_length=DEFAULT_MAX_SEQ_LEN)
        if os.path.isfile(compiled_path):
            # Load the precompiled model from disk
            print(f"Loading precompiled model from {compiled_path} ...")
            model = torch.jit.load(compiled_path)
            # By default, the Neuron runtime assigns cores in a round-robin manner per process.
            # If multiple models are loaded in one process, they will occupy different NeuronCores automatically.
            # If running in separate processes (e.g., target and draft on different instances), each will use core0 by default.
            # If needed, the model can be moved to a specific NeuronCore with torch_neuronx.move_trace_to_device.
            # Attach metadata for downstream use
            model.max_length = DEFAULT_MAX_SEQ_LEN
            model.is_compiled = True
            return model
        else:
            if compile_if_missing:
                # Compile on the fly using compile_model module
                print(f"Compiled model not found for {model_name_or_path}. Compiling now...")
                from compile_model import compile_model as compile_fn
                output_path = compile_fn(model_name_or_path, compiled_path, DEFAULT_MAX_SEQ_LEN)
                print("Loading newly compiled model...")
                model = torch.jit.load(output_path)
                model.max_length = DEFAULT_MAX_SEQ_LEN
                model.is_compiled = True
                return model
            else:
                raise FileNotFoundError(f"Compiled model not found at {compiled_path}. Please run in compile mode first.")
    # If not using compiled model, load normally (runs on CPU or GPU)
    print(f"Loading model {model_name_or_path} without Neuron compilation (CPU/GPU mode)...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()
    # Attempt to use BF16 on CPU/GPU if supported (for memory savings)
    try:
        model.to(torch.bfloat16)
    except Exception:
        pass
    return model