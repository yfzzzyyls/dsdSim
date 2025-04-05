import torch
import torch_neuronx  # The low-level Neuron tracing API
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_id):
    """
    Load and compile a HF Transformers model on AWS Trainium using torch-neuronx (no optimum-neuron).
    """

    print(f"[ModelLoader] Loading HF model: {model_id}")
    # First load the model & tokenizer in regular CPU memory
    # (We generally do not do .to('cuda') or anything, because we are going to compile for Neuron.)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Some LLM tokenizers do not have a pad token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare a dummy input for tracing (sequence length = 128 is arbitrary; adapt as needed)
    # The shape you choose here must be representative of your typical input sizes
    # for correct compilation.
    example_sequence_len = 32
    dummy_input_ids = torch.randint(
        low=0, 
        high=len(tokenizer), 
        size=(1, example_sequence_len),
        dtype=torch.long
    )

    print("[ModelLoader] Compiling model with torch_neuronx.trace...")
    # Put model in eval mode
    model.eval()
    # Perform the compile step
    # The model must be invoked at least once on a Neuron device to create a compiled graph.
    compiled_model = torch_neuronx.trace(
        model,
        dummy_input_ids,
        compiler_workdir=None,  # or specify a path if you want to inspect artifacts
        # compiler_args={},      # You can pass additional Neuron compiler args if needed
    )
    print("[ModelLoader] Compilation complete. Model is now on Neuron device.")

    # Return the compiled model plus the tokenizer
    return compiled_model, tokenizer