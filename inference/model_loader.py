import os
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default compilation settings
DEFAULT_SEQUENCE_LENGTH = 128

def load_model(model_path: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    """
    Load a LLaMA (or other) model for huggingface + Trainium usage in normal HF format.
    If the model_path is already “compiled” (i.e., contains a local directory with
    a ‘pytorch_model.bin’ and possibly a 'config.json'), we load from that folder.
    Otherwise, we do a minimal “compile” (in this example, that just downloads & saves).
    """

    # Check if it’s a local directory with a config & pytorch_model.bin
    config_file = os.path.join(model_path, "config.json")
    weight_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.isdir(model_path) and os.path.isfile(config_file) and os.path.isfile(weight_file):
        # We consider this a “locally available compiled/cached” folder
        logger.info(f"Found existing local model folder at: {model_path}. Loading with AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return model
    else:
        # Perform a minimal “compile” step: load from HF or local, then save
        logger.info(f"No local compiled model found at {model_path}. Attempting to download or load from HF, then save.")
        return compile_model(model_path, sequence_length=sequence_length)

def compile_model(model_path: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    """
    Minimal “compile” flow: load from Hugging Face or local checkpoint, then save to
    a directory for subsequent runs. This is not a real ‘neuron compile’, but it prevents indefinite waits.
    """
    base_name = os.path.basename(os.path.normpath(model_path))
    compiled_dir = f"{base_name}-compiled-{sequence_length}"
    os.makedirs(compiled_dir, exist_ok=True)
    logger.info(f"Loading model {model_path} via AutoModelForCausalLM...")
    # The actual HF model load
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.save_pretrained(compiled_dir)
    # Save the tokenizer too
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(compiled_dir)
    logger.info(f"Model + tokenizer saved locally to {compiled_dir}. Next time we’ll load from it.")
    return model
