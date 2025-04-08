import os
import logging
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_neuronx import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split

logger = logging.getLogger(__name__)

# Default sequence length (can be overridden by function arguments)
DEFAULT_SEQUENCE_LENGTH = 128

def load_model(model_path: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    """
    Load a model for inference. If a Neuron-compiled model is available in the directory, 
    load it directly. Otherwise, fall back to compiling or loading from Hugging Face.
    """
    # Check for local directory with config
    config_file = os.path.join(model_path, "config.json")
    if os.path.isdir(model_path) and os.path.isfile(config_file):
        # List any Neuron compiled model files in the directory
        compiled_files = [f for f in os.listdir(model_path) 
                          if f.startswith("model_neuron") and (f.endswith(".pt") or f.endswith(".ts"))]
        weight_file = os.path.join(model_path, "pytorch_model.bin")
        if compiled_files:
            # Found Neuron-compiled model artifacts (TorchScript or partitioned files)
            logger.info(f"Found compiled Neuron model artifacts in '{model_path}': {compiled_files}")
            # Identify model architecture from config (to choose appropriate NeuronX class)
            model_type = ""
            try:
                with open(config_file, "r") as cf:
                    cfg = json.load(cf)
                    model_type = cfg.get("model_type", "")
            except Exception as e:
                logger.warning(f"Could not read model config: {e}")
            # Use LLaMA optimized class if applicable
            if model_type.lower() == "llama" or "llama" in model_path.lower():
                tp_degree = len(compiled_files) if len(compiled_files) > 1 else 1
                # Initialize Neuron model (no weights loading needed if artifacts exist)
                model = LlamaForSampling.from_pretrained(model_path, batch_size=1, amp='bf16', 
                                                        n_positions=sequence_length, tp_degree=tp_degree)
                try:
                    model.load(model_path)  # Load compiled model (neuron artifacts) into NeuronCores
                    logger.info("Loaded Neuron-compiled model from TorchScript artifacts.")
                except Exception as e:
                    logger.info(f"No precompiled cache loaded (will compile now): {e}")
                # Ensure model is loaded onto Neuron cores
                model.to_neuron()
                return model
            else:
                logger.info(f"Compiled model artifacts found, but unrecognized model type '{model_type}'. Loading with AutoModelForCausalLM as fallback.")
                return AutoModelForCausalLM.from_pretrained(model_path)
        else:
            # No Neuron artifacts found in directory
            if os.path.isfile(weight_file):
                # If standard Hugging Face weights are present, decide on loading vs compile
                model_type = ""
                try:
                    with open(config_file, "r") as cf:
                        cfg = json.load(cf)
                        model_type = cfg.get("model_type", "")
                except Exception:
                    pass
                if model_type.lower() == "llama" or "llama" in model_path.lower():
                    # For LLaMA models, compile to Neuron rather than loading to CPU
                    logger.info(f"Found HF checkpoint in '{model_path}' – compiling to Neuron for inference.")
                    return compile_model(model_path, sequence_length=sequence_length)
                else:
                    logger.info(f"Loading model from Hugging Face checkpoint at '{model_path}'...")
                    return AutoModelForCausalLM.from_pretrained(model_path)
            else:
                # No weights found at all, attempt to compile from source (HF Hub or other path)
                logger.info(f"No weights found in '{model_path}'. Attempting to download/compile from source.")
                return compile_model(model_path, sequence_length=sequence_length)
    else:
        # model_path is not a local directory (could be a model ID or file path) – attempt to compile/load
        return compile_model(model_path, sequence_length=sequence_length)

def compile_model(model_path: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    """
    Compile a model for AWS Neuron. Loads the model (from HF Hub or local checkpoint), 
    compiles it to a TorchScript that can run on NeuronCores, and saves the compiled model 
    and tokenizer to a local folder for future use.
    """
    base_name = os.path.basename(os.path.normpath(model_path))
    compiled_dir = f"{base_name}-compiled-{sequence_length}"
    os.makedirs(compiled_dir, exist_ok=True)
    logger.info(f"Compiling model '{model_path}' to Neuron (sequence_length={sequence_length})...")
    # Determine model type for optimized loading
    model_type = ""
    try:
        if os.path.isdir(model_path):
            # If local directory, read model_type from its config if possible
            with open(os.path.join(model_path, "config.json"), "r") as cf:
                cfg = json.load(cf)
                model_type = cfg.get("model_type", "")
        else:
            # If model_path is a HuggingFace model ID, try AutoTokenizer/AutoModel to trigger download
            cfg = AutoTokenizer.from_pretrained(model_path).config if hasattr(AutoTokenizer, "from_pretrained") else {}
            model_type = getattr(cfg, "model_type", "")
    except Exception as e:
        logger.warning(f"Could not determine model type for '{model_path}': {e}")
    # Use all available NeuronCores (if set) for tensor parallelism
    tp_degree = int(os.environ.get("NEURON_RT_NUM_CORES", "1"))
    if model_type.lower() == "llama" or "llama" in model_path.lower():
        # Compile using LlamaForSampling for optimized performance on Neuron
        model = LlamaForSampling.from_pretrained(model_path, batch_size=1, amp='bf16',
                                                n_positions=sequence_length, tp_degree=tp_degree)
        model.to_neuron()  # Compile the model (this triggers the Neuron compilation)
        # Save the compiled model artifacts for future runs
        try:
            save_pretrained_split(model, compiled_dir)
        except Exception as e:
            # If the model is single-part, save_pretrained_split may not be needed
            try:
                model.save(compiled_dir)
            except Exception as e2:
                logger.error(f"Failed to save compiled model artifacts: {e2}")
        # Save the tokenizer to the same directory
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.save_pretrained(compiled_dir)
        logger.info(f"Neuron compiled model saved to '{compiled_dir}'.")
        return model
    else:
        # Fallback: load the model weights and save them (no Neuron compilation)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.save_pretrained(compiled_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.save_pretrained(compiled_dir)
        logger.info(f"Model and tokenizer saved to '{compiled_dir}' (no Neuron compilation performed).")
        return model
