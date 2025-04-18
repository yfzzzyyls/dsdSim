import os
import logging
import shutil
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig   # ← add AutoConfig
from transformers_neuronx import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split

logger = logging.getLogger(__name__)

# Default sequence length (can be overridden by function arguments)
DEFAULT_SEQUENCE_LENGTH = 128

class LlamaSamplingWithPast(torch.nn.Module):
    """
    Wrap a Neuron‑compiled LlamaForSampling so .forward returns
    logits and past_key_values (hidden state) in Hugging Face format.
    """
    def __init__(self, neuron_llama):
        super().__init__()
        self.llama = neuron_llama
        self.config = neuron_llama.config  # needed by downstream code

    def forward(self, input_ids, prev_hidden=None):
        extra = {"prev_hidden": prev_hidden} if prev_hidden is not None else {}
        logits, new_hidden = self.llama(input_ids, **extra)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=new_hidden,
        )

def load_model(model_path: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    """
    Load or compile a model for inference.
    """
    logger.info(f"Attempting to download/compile from source.")
    model = compile_model(model_path, sequence_length=sequence_length)
    return model


def compile_model(model_path: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    """
    Compile a model for AWS Neuron. Loads the model (from HF Hub or local checkpoint),
    compiles it to a TorchScript that can run on NeuronCores, and saves the compiled model
    and tokenizer to a local folder for future use.
    """
    base_name = os.path.basename(os.path.normpath(model_path))
    compiled_dir = f"{base_name}-compiled-{sequence_length}"
    logger.info(f"Compiling model '{model_path}' to Neuron (sequence_length={sequence_length})...")

    model_type = ""
    try:
        if os.path.isdir(model_path):
            # If local directory, read model_type from its config if possible
            with open(os.path.join(model_path, "config.json"), "r") as cf:
                cfg = json.load(cf)
            model_type = cfg.get("model_type", "")
        else:
            # If model_path is a HuggingFace model ID, trigger download to get config
            cfg = AutoTokenizer.from_pretrained(model_path).config if hasattr(AutoTokenizer, "from_pretrained") else {}
            model_type = getattr(cfg, "model_type", "")
    except Exception as e:
        logger.warning(f"Could not determine model type for '{model_path}': {e}")

    # Use all available NeuronCores for tensor parallelism
    tp_degree = int(os.environ.get("NEURON_RT_NUM_CORES", "2"))
    if model_type.lower() == "llama" or "llama" in model_path.lower():
        # Compile using optimized LLaMA class for Neuron
        logger.info(f"Compiling model using optimized LLaMA for Neuron ...")
        model = LlamaForSampling.from_pretrained(model_path, batch_size=1, amp='bf16',
                                                 n_positions=sequence_length, tp_degree=tp_degree)
        # Compile the model
        model.config.use_cache = True
        model.to_neuron()
        return LlamaSamplingWithPast(model)
    else:
        # Fallback: load the model weights and save them (no Neuron compilation performed)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.save_pretrained(compiled_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.save_pretrained(compiled_dir)
        logger.info(f"Model and tokenizer saved to '{compiled_dir}' (no Neuron compilation performed).")
        return model