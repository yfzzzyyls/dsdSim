#!/usr/bin/env python3
"""
Utility script to compile a Hugging Face Transformers causal LM model to AWS Neuron (Trainium),
with BF16 precision, and save the compiled model to disk.
"""
import os
import argparse
import torch
import torch_neuronx
from transformers import AutoModelForCausalLM

# Default maximum sequence length for compilation (adjust as needed)
DEFAULT_MAX_SEQ_LEN = 1024

def compile_model(model_name_or_path: str, output_path: str = None, max_seq_length: int = DEFAULT_MAX_SEQ_LEN):
    """
    Compile the given Hugging Face causal LM model to an AWS Neuron-optimized TorchScript and save to disk.
    Args:
        model_name_or_path: HuggingFace model ID or local path to the model to compile.
        output_path: File path to save the compiled model (.pt file).
                     If not provided, it will be generated based on the model name.
        max_seq_length: Maximum sequence length to compile for (affects memory usage and flexibility for prompt length).
    """
    # Determine output file path
    if output_path is None:
        # Derive a safe output filename from model name and max_seq_length
        base_name = os.path.basename(os.path.normpath(model_name_or_path))
        output_path = f"{base_name}_neuron_bf16_{max_seq_length}.pt"
    # Load model with BF16 precision
    print(f"Loading model '{model_name_or_path}' for compilation...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    model.eval()
    # Ensure no caches or unnecessary outputs for efficient tracing
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "return_dict"):
        model.config.return_dict = False
    if hasattr(model.config, "output_hidden_states"):
        model.config.output_hidden_states = False
    if hasattr(model.config, "output_attentions"):
        model.config.output_attentions = False
    # Prepare example inputs for tracing (batch_size=1, seq_len=max_seq_length)
    batch_size = 1
    seq_len = max_seq_length
    # Use int64 input_ids and attention_mask. All 1s in attention_mask (full length) to represent worst-case input.
    example_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    example_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    # Trace (compile) the model using torch_neuronx. Use recommended compiler flags for Transformers.
    compiler_args = ["--target", "trn1", "--fp32-cast", "matmult", "--fast-math", "no-fast-relayout"]
    print(f"Compiling model to Neuron with sequence length {seq_len} (this may take several minutes)...")
    traced_model = torch_neuronx.trace(model, (example_input_ids, example_attention_mask),
                                       compiler_args=compiler_args, timeout=900,  # timeout in seconds (15 minutes)
                                       cpu_backend=True)
    # Save the compiled model
    torch.jit.save(traced_model, output_path)
    print(f"Saved compiled model to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Compile a HuggingFace causal LM model to AWS Neuron")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (HuggingFace Hub ID or local directory) to compile.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the compiled model (.pt file). If not provided, a default will be used.")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help="Maximum sequence length to compile for.")
    args = parser.parse_args()
    try:
        compile_model(args.model, args.output, args.max_seq_len)
    except Exception as e:
        print(f"Error during model compilation: {e}")
        raise

if __name__ == "__main__":
    main()