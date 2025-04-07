import os
import json
import logging
import torch
from transformers_neuronx import LlamaForSampling
from transformers import AutoTokenizer, LlamaConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default compilation settings
DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_TP_DEGREE = 2
DEFAULT_DTYPE = 'bf16'

def load_model(model_name: str,
               sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
               tp_degree: int = DEFAULT_TP_DEGREE,
               dtype: str = DEFAULT_DTYPE):
    """
    Load a LLaMA model for AWS Neuron.
    If model_name is a pre-compiled folder (contains config.json and a dummy checkpoint),
    load the model using from_pretrained(). Otherwise, compile the model.
    """
    base_name = os.path.basename(os.path.normpath(model_name))
    compiled_dir = f"{base_name}-neuron-compiled-{sequence_length}"
    # If the provided model_name directory contains config.json and a checkpoint, assume it is compiled.
    if os.path.isdir(model_name) and os.path.isfile(os.path.join(model_name, "config.json")) and os.path.isfile(os.path.join(model_name, "pytorch_model.bin")):
        logger.info(f"Detected pre-compiled model folder at '{model_name}'. Loading using from_pretrained().")
        model = LlamaForSampling.from_pretrained(
            model_name,
            batch_size=1,
            tp_degree=tp_degree,
            n_positions=sequence_length,
            amp=dtype
        )
        return model
    # Otherwise, check for a compiled folder using our naming convention.
    if os.path.isdir(compiled_dir) and os.path.isfile(os.path.join(compiled_dir, "config.json")) and os.path.isfile(os.path.join(compiled_dir, "pytorch_model.bin")):
        logger.info(f"Found existing compiled model folder at '{compiled_dir}'. Loading compiled model.")
        model = LlamaForSampling.from_pretrained(
            compiled_dir,
            batch_size=1,
            tp_degree=tp_degree,
            n_positions=sequence_length,
            amp=dtype
        )
        return model

    # If not found, compile the model
    logger.info(f"Compiled model not found for '{model_name}'. Compiling now (seq_len={sequence_length})...")
    model = compile_model(model_name, sequence_length=sequence_length, tp_degree=tp_degree, dtype=dtype)
    return model

def compile_model(model_name: str,
                  sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                  tp_degree: int = DEFAULT_TP_DEGREE,
                  dtype: str = DEFAULT_DTYPE):
    """
    Compile the given LLaMA model (assumed to be a Hugging Face checkpoint) using transformers-neuronx.
    Save the compiled artifacts (compiled model, tokenizer, config, and a dummy checkpoint) into
    a folder named "<base_name>-neuron-compiled-<sequence_length>".
    """
    logger.info(f"Compiling model '{model_name}' with sequence_length={sequence_length}, tp_degree={tp_degree}, dtype={dtype}")
    os.environ["NEURON_CC_FLAGS"] = "--model-type transformer --verbose=1"
    logger.info(f"NEURON_CC_FLAGS set to: {os.environ['NEURON_CC_FLAGS']}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = LlamaForSampling.from_pretrained(
        model_name,
        batch_size=1,
        tp_degree=tp_degree,
        n_positions=sequence_length,
        amp=dtype
    )
    model.eval()
    logger.info("Running Neuron compilation (model.to_neuron())...")
    model.to_neuron()

    base_name = os.path.basename(os.path.normpath(model_name))
    compiled_dir = f"{base_name}-neuron-compiled-{sequence_length}"
    os.makedirs(compiled_dir, exist_ok=True)
    logger.info(f"Saving compiled artifacts into '{compiled_dir}'...")

    tokenizer.save_pretrained(compiled_dir)
    config_dict = {k: v for k, v in vars(model.config).items() if not k.startswith("_")}
    config_path = os.path.join(compiled_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Model config saved to '{config_path}'")
    
    # Save a dummy checkpoint so from_pretrained() can detect it.
    checkpoint_path = os.path.join(compiled_dir, "pytorch_model.bin")
    try:
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved dummy checkpoint to '{checkpoint_path}'")
    except Exception as e:
        logger.error(f"Error saving dummy checkpoint: {e}")

    try:
        model.save_pretrained_split(compiled_dir)
        logger.info("Saved compiled model using save_pretrained_split().")
    except AttributeError:
        compiled_module = getattr(model, "model", None) or getattr(model, "_neuron_module", None)
        if compiled_module is None:
            compiled_module = model
        if isinstance(compiled_module, list):
            logger.info(f"Detected tp_degree={tp_degree}, saving {len(compiled_module)} compiled shards...")
            for i, shard in enumerate(compiled_module):
                shard_path = os.path.join(compiled_dir, f"model_neuron_tp_{i}.pt")
                shard.save(shard_path)
                logger.info(f"Saved shard {i} to '{shard_path}'")
        else:
            shard_path = os.path.join(compiled_dir, "model_neuron.ts")
            compiled_module.save(shard_path)
            logger.info(f"Saved compiled module to '{shard_path}'")
    logger.info("Model compilation complete.")
    return model
