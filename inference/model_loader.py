import os
from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM

def load_model(model_id, num_cores=1, dtype="bf16"):
    """
    Load a model for inference on Neuron cores. If model_id is a Hugging Face model name, 
    it will be compiled for Neuron. If model_id is a local directory path containing a 
    compiled model, it will load it directly (no recompilation).
    """
    compiler_args = {"auto_cast_type": dtype}
    if num_cores:
        compiler_args["num_cores"] = num_cores

    export_flag = True
    # If model_id is a local path, decide whether to compile or not
    if os.path.isdir(model_id):
        # Heuristic: if compiled Neuron artifacts are present, skip export
        files = os.listdir(model_id)
        # If any file indicates a compiled model (e.g., Neuron artifact or absence of raw weights)
        has_neuron_artifact = any(fname.endswith(".neff") or fname.startswith("neuron") for fname in files)
        if has_neuron_artifact:
            export_flag = False

    model = NeuronModelForCausalLM.from_pretrained(model_id, export=export_flag, **compiler_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id if isinstance(model_id, str) else model_id)
    # Some tokenizers (e.g., Llama) might not have a pad token by default; set pad token to eos to avoid warnings
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer