import os
import torch
from transformers_neuronx import LlamaForSampling
from transformers import AutoTokenizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default compilation settings
DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_TP_DEGREE = 2
DEFAULT_DTYPE = 'bf16'

def load_model(model_name: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH, 
               tp_degree: int = DEFAULT_TP_DEGREE, dtype: str = DEFAULT_DTYPE):
    """
    Load a LLaMA model on AWS Trainium using transformers-neuronx.
    If a compiled model exists, load it; otherwise, compile and save it.
    """
    base_name = os.path.basename(os.path.normpath(model_name))
    compiled_dir = f"{base_name}-neuron-compiled-{sequence_length}"
    # Determine if we have compiled artifacts already
    if os.path.isdir(compiled_dir):
        logger.info(f"Loading compiled model from {compiled_dir}")
        model = LlamaForSampling.from_pretrained(compiled_dir, batch_size=1, tp_degree=tp_degree, 
                                                 n_positions=sequence_length, amp=dtype)
    else:
        logger.info(f"Compiling model {model_name} for sequence length {sequence_length} (tp_degree={tp_degree}, dtype={dtype})")
        model = LlamaForSampling.from_pretrained(model_name, batch_size=1, tp_degree=tp_degree, 
                                                 n_positions=sequence_length, amp=dtype)
        # Compile the model for inference on Neuron cores
        model.to_neuron()
        # Save the compiled model to disk for reuse
        logger.info(f"Saving compiled model to {compiled_dir}")
        model.save(compiled_dir)
    return model

def compile_model(model_name: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH, 
                  tp_degree: int = DEFAULT_TP_DEGREE, dtype: str = DEFAULT_DTYPE):
    """
    Compile the given model and save the compiled artifacts to disk.
    """
    # This function simply uses load_model which will compile if needed
    _ = load_model(model_name, sequence_length=sequence_length, tp_degree=tp_degree, dtype=dtype)
