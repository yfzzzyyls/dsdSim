import os
import json
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

def load_model(model_name: str,
               sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
               tp_degree: int = DEFAULT_TP_DEGREE,
               dtype: str = DEFAULT_DTYPE):
    """
    Load a LLaMA model compiled for AWS Neuron.
    If a compiled model directory exists, load its saved compiled module(s) and attach them.
    Otherwise, compile and save manually.
    """
    base_name = os.path.basename(os.path.normpath(model_name))
    compiled_dir = f"{base_name}-neuron-compiled-{sequence_length}"
    compiled_files = _get_compiled_files(compiled_dir, tp_degree)
    
    if compiled_files:
        logger.info(f"Loading compiled model from '{compiled_dir}'...")
        model = _load_compiled_model(compiled_dir, tp_degree, sequence_length, dtype)
    else:
        logger.info(f"Compiled model not found for '{model_name}'. Compiling now...")
        model = compile_model(model_name, sequence_length=sequence_length, tp_degree=tp_degree, dtype=dtype)
    
    return model


def compile_model(model_name: str,
                  sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                  tp_degree: int = DEFAULT_TP_DEGREE,
                  dtype: str = DEFAULT_DTYPE):
    """
    Compile the given LLaMA model for AWS Neuron and save:
      1) The compiled TorchScript module(s)
      2) The tokenizer (using save_pretrained)
      3) The model configuration as config.json
    """
    logger.info(f"Compiling model '{model_name}' with sequence_length={sequence_length}, tp_degree={tp_degree}, dtype={dtype}")

    os.environ["NEURON_CC_FLAGS"] = "--model-type transformer --verbose=1"
    logger.info(f"NEURON_CC_FLAGS set to: {os.environ['NEURON_CC_FLAGS']}")

    # Load tokenizer so we can save it later.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and compile the model.
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
    
    # Determine the compiled module.
    # Some versions attach it as .model, others as ._neuron_module.
    compiled_module = getattr(model, "model", None) or getattr(model, "_neuron_module", None)
    if compiled_module is None:
        logger.info("No compiled submodule attribute found; using the entire model as the compiled module.")
        compiled_module = model

    # Prepare output directory.
    base_name = os.path.basename(os.path.normpath(model_name))
    compiled_dir = f"{base_name}-neuron-compiled-{sequence_length}"
    os.makedirs(compiled_dir, exist_ok=True)
    logger.info(f"Saving compiled artifacts into '{compiled_dir}'...")

    # Save the tokenizer.
    tokenizer.save_pretrained(compiled_dir)
    
    # Save the configuration manually as JSON.
    config_dict = {k: v for k, v in vars(model.config).items() if not k.startswith("_")}
    config_path = os.path.join(compiled_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to '{config_path}'")
    
    # Save compiled module(s).
    if isinstance(compiled_module, list):
        logger.info(f"Detected tp_degree={tp_degree}, saving {len(compiled_module)} compiled submodules...")
        for i, shard in enumerate(compiled_module):
            shard_path = os.path.join(compiled_dir, f"model_neuron_tp_{i}.pt")
            # Use the module's own save() method instead of torch.jit.save.
            shard.save(shard_path)
            logger.info(f"Saved shard {i} to '{shard_path}'")
    else:
        shard_path = os.path.join(compiled_dir, "model_neuron.ts")
        compiled_module.save(shard_path)
        logger.info(f"Saved compiled module to '{shard_path}'")
    
    logger.info("Model compilation complete.")
    return model


def _get_compiled_files(compiled_dir: str, tp_degree: int):
    """
    Check for the presence of saved compiled TorchScript files in the compiled_dir.
    Returns a list of found files.
    """
    if not os.path.isdir(compiled_dir):
        return []
    
    files = []
    if tp_degree > 1:
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
    Load the configuration and the saved compiled TorchScript module(s),
    then create a new LlamaForSampling instance and attach the compiled module(s).
    The forward method is overridden to use the compiled module(s).
    """
    config_path = os.path.join(compiled_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = LlamaConfig(**config_dict)
    else:
        logger.warning(f"No config.json found in {compiled_dir}, using default LlamaConfig()")
        config = LlamaConfig()
    
    model = LlamaForSampling(config,
                             batch_size=1,
                             tp_degree=tp_degree,
                             n_positions=sequence_length,
                             amp=dtype)
    model.eval()
    
    if tp_degree > 1:
        shards = []
        for i in range(tp_degree):
            shard_path = os.path.join(compiled_dir, f"model_neuron_tp_{i}.pt")
            shard = torch.jit.load(shard_path)
            shards.append(shard)
        compiled_module = shards

        def multi_shard_forward(*args, **kwargs):
            x = args[0]
            splits = torch.chunk(x, tp_degree, dim=0)
            outputs = []
            for shard, part in zip(compiled_module, splits):
                outputs.append(shard.forward(part, *args[1:], **kwargs))
            return torch.cat(outputs, dim=0)
        model.forward = multi_shard_forward
    else:
        shard_path = os.path.join(compiled_dir, "model_neuron.ts")
        compiled_module = torch.jit.load(shard_path)
        model.forward = lambda *args, **kwargs: compiled_module.forward(*args, **kwargs)
    
    logger.info("Compiled module(s) loaded and attached to the model.")
    return model


def _patch_config_if_needed(model):
    """
    (Optional) Patch the configuration if necessary (e.g. converting LEDConfig to LlamaConfig).
    """
    if model.config.__class__.__name__ == "LEDConfig":
        logger.info("Patching config: converting LEDConfig to LlamaConfig with default intermediate_size=4096")
        config_dict = {k: v for k, v in vars(model.config).items() if not k.startswith("_")}
        if "intermediate_size" not in config_dict:
            config_dict["intermediate_size"] = 4096
        new_config = LlamaConfig(**config_dict)
        model.config = new_config
