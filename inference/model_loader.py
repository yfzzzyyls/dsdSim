import os
import json
import logging

import torch
from transformers_neuronx import LlamaForSampling
from transformers import AutoTokenizer, LlamaConfig

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
    Load a LLaMA model for AWS Neuron. If a compiled model directory exists for the given 
    model and sequence length, load the compiled artifacts. Otherwise, compile the model, 
    save the artifacts, and return the compiled model.
    """
    # Determine the compiled artifacts directory name
    base_name = os.path.basename(os.path.normpath(model_name))
    compiled_dir = f"{base_name}-neuron-compiled-{sequence_length}"

    # If the given model_name is itself a compiled directory, adjust accordingly
    if os.path.isdir(model_name):
        # Treat model_name as a potential compiled artifact directory
        compiled_files = _get_compiled_files(model_name, tp_degree=tp_degree)
        if compiled_files:
            compiled_dir = model_name
        # If the directory doesn't contain compiled files, we'll proceed to compile into compiled_dir

    # Check for existing compiled model files
    compiled_files = _get_compiled_files(compiled_dir, tp_degree=tp_degree)
    if compiled_files:
        logger.info(f"Loading compiled model from '{compiled_dir}'...")
        model = _load_compiled_model(compiled_dir, tp_degree, sequence_length, dtype)
    else:
        logger.info(f"Compiled model not found for '{model_name}'. Compiling now (seq_len={sequence_length})...")
        model = compile_model(model_name, sequence_length=sequence_length, tp_degree=tp_degree, dtype=dtype)
    return model

def compile_model(model_name: str,
                  sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                  tp_degree: int = DEFAULT_TP_DEGREE,
                  dtype: str = DEFAULT_DTYPE):
    """
    Compile the given LLaMA model for AWS Neuron and save the artifacts:
      1) Compiled TorchScript module(s) for each tensor-parallel shard
      2) The tokenizer (saved via Hugging Face's save_pretrained)
      3) The model configuration (as config.json)
    """
    logger.info(f"Compiling model '{model_name}' with sequence_length={sequence_length}, tp_degree={tp_degree}, dtype={dtype}")
    # Ensure compiler flags are set for transformer model
    os.environ["NEURON_CC_FLAGS"] = "--model-type transformer --verbose=1"
    logger.info(f"NEURON_CC_FLAGS set to: {os.environ['NEURON_CC_FLAGS']}")

    # Load tokenizer (to save later for ease of use)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # Load the model with specified parameters (this loads weights from Hugging Face or local checkpoint)
    model = LlamaForSampling.from_pretrained(
        model_name,
        batch_size=1,
        tp_degree=tp_degree,
        n_positions=sequence_length,
        amp=dtype
    )
    model.eval()
    logger.info("Running Neuron compilation (model.to_neuron())...")
    model.to_neuron()  # Compile the model for Inferentia/Trainium

    # Determine the compiled module(s) attribute. Depending on transformers-neuronx version,
    # the compiled TorchScript modules might be in model.model or model._neuron_module.
    compiled_module = getattr(model, "model", None) or getattr(model, "_neuron_module", None)
    if compiled_module is None:
        logger.info("No specific compiled module attribute found; using the model object itself as compiled module.")
        compiled_module = model

    # Prepare output directory for compiled artifacts
    base_name = os.path.basename(os.path.normpath(model_name))
    compiled_dir = f"{base_name}-neuron-compiled-{sequence_length}"
    os.makedirs(compiled_dir, exist_ok=True)
    logger.info(f"Saving compiled artifacts to '{compiled_dir}'...")

    # Save the tokenizer to the compiled directory
    tokenizer.save_pretrained(compiled_dir)
    # Save the model configuration to config.json for future loads
    config_dict = {k: v for k, v in vars(model.config).items() if not k.startswith("_")}
    config_path = os.path.join(compiled_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Model config saved to '{config_path}'")

    # Save the compiled model modules (TorchScript files)
    if isinstance(compiled_module, list):
        # If model is sharded across tensor-parallel cores
        logger.info(f"Detected tp_degree={tp_degree}, saving {len(compiled_module)} compiled shards...")
        for i, shard in enumerate(compiled_module):
            shard_path = os.path.join(compiled_dir, f"model_neuron_tp_{i}.pt")
            # Use the module's own save method (TorchScript save)
            shard.save(shard_path)
            logger.info(f"Saved compiled shard {i} to '{shard_path}'")
    else:
        shard_path = os.path.join(compiled_dir, "model_neuron.ts")
        compiled_module.save(shard_path)
        logger.info(f"Saved compiled model to '{shard_path}'")

    logger.info("Model compilation complete.")
    return model

def _get_compiled_files(compiled_dir: str, tp_degree: int):
    """
    Check for presence of saved compiled TorchScript files in the compiled_dir.
    Returns a list of file paths if found, otherwise an empty list.
    """
    if not os.path.isdir(compiled_dir):
        return []
    files = []
    if tp_degree > 1:
        # Check for multiple shard files
        for i in range(tp_degree):
            path = os.path.join(compiled_dir, f"model_neuron_tp_{i}.pt")
            if os.path.isfile(path):
                files.append(path)
    else:
        path = os.path.join(compiled_dir, "model_neuron.ts")
        if os.path.isfile(path):
            files.append(path)
    return files

def _load_compiled_model(compiled_dir: str,
                         tp_degree: int,
                         sequence_length: int,
                         dtype: str):
    """
    Load the saved compiled TorchScript module(s) from the compiled_dir, then create 
    a LlamaForSampling model instance and attach the compiled module(s) to it.
    The model's forward method is overridden to use the compiled modules.
    """
    # Load saved config if available
    config_path = os.path.join(compiled_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = LlamaConfig(**config_dict)
    else:
        logger.warning(f"No config.json found in {compiled_dir}; using default LlamaConfig()")
        config = LlamaConfig()

    # Initialize a model instance with the configuration (weights will not be used, since we attach compiled modules)
    model = LlamaForSampling(config, batch_size=1, tp_degree=tp_degree, n_positions=sequence_length, amp=dtype)
    model.eval()

    # Load the compiled TorchScript modules
    if tp_degree > 1:
        # Load multiple shards for tensor-parallel model
        shards = []
        for i in range(tp_degree):
            shard_path = os.path.join(compiled_dir, f"model_neuron_tp_{i}.pt")
            shards.append(torch.jit.load(shard_path))
        compiled_module = shards

        # Define a forward function that distributes inputs across shards and concatenates outputs
        def multi_shard_forward(*args, **kwargs):
            x = args[0]
            # Split the input tensor across tp_degree shards (dim=0 is batch dimension)
            splits = torch.chunk(x, tp_degree, dim=0)
            outputs = []
            for shard, part in zip(compiled_module, splits):
                outputs.append(shard.forward(part, *args[1:], **kwargs))
            return torch.cat(outputs, dim=0)
        model.forward = multi_shard_forward
    else:
        # Single-shard compiled model
        shard_path = os.path.join(compiled_dir, "model_neuron.ts")
        compiled_module = torch.jit.load(shard_path)
        model.forward = lambda *args, **kwargs: compiled_module.forward(*args, **kwargs)

    logger.info("Compiled model loaded and attached to the LlamaForSampling instance.")
    return model

# (Optional) Additional utility functions or patches could be added below
