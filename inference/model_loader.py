import os
import torch
from transformers_neuronx import LlamaForSampling
from transformers import AutoTokenizer, LlamaConfig
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
    # Check if compiled artifacts exist
    if os.path.isdir(compiled_dir):
        logger.info(f"Loading compiled model from {compiled_dir}")
        model = LlamaForSampling.from_pretrained(
            compiled_dir, 
            batch_size=1, 
            tp_degree=tp_degree, 
            n_positions=sequence_length, 
            amp=dtype
        )
    else:
        logger.info(f"Compiled model not found. Compiling model '{model_name}' now...")
        model = compile_model(model_name, sequence_length=sequence_length, tp_degree=tp_degree, dtype=dtype)
    return model

def compile_model(model_name: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH, 
                  tp_degree: int = DEFAULT_TP_DEGREE, dtype: str = DEFAULT_DTYPE):
    """
    Compile the given model using transformers-neuronx and save the compiled artifacts to disk.
    This version overrides NEURON_CC_FLAGS with a minimal set, removing unsupported flags.
    """
    logger.info(f"Compiling model {model_name} with sequence_length={sequence_length}, tp_degree={tp_degree}, dtype={dtype}")
    
    # Override NEURON_CC_FLAGS to a minimal set accepted by your neuronx-cc version.
    # Remove unsupported flags such as --auto-partition and --retry_failed_compilation.
    os.environ["NEURON_CC_FLAGS"] = "--model-type transformer --verbose=1"
    logger.info(f"NEURON_CC_FLAGS set to: {os.environ['NEURON_CC_FLAGS']}")
    
    # Load the model using transformers-neuronx
    model = LlamaForSampling.from_pretrained(
        model_name, 
        batch_size=1, 
        tp_degree=tp_degree, 
        n_positions=sequence_length, 
        amp=dtype
    )
    model.eval()
    # Trigger Neuron compilation
    model.to_neuron()
    
    # Save the compiled model artifacts using save_pretrained so that required files are written.
    base_name = os.path.basename(os.path.normpath(model_name))
    compiled_dir = f"{base_name}-neuron-compiled-{sequence_length}"
    logger.info(f"Saving compiled model to {compiled_dir}")
    model.save(compiled_dir)
    return model

# (Optional) Helper to patch config if needed (e.g., converting LEDConfig to LlamaConfig)
def _patch_config_if_needed(model):
    if model.config.__class__.__name__ == "LEDConfig":
        logger.info("Patching config: converting LEDConfig to LlamaConfig with default intermediate_size=4096")
        config_dict = model.config.to_dict()
        if "intermediate_size" not in config_dict:
            config_dict["intermediate_size"] = 4096  # adjust as needed
        new_config = LlamaConfig(**config_dict)
        model.config = new_config
