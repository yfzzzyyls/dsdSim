import os
import logging
import shutil
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers_neuronx import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers_neuronx.fused_speculation import FusedSpeculativeDecoder
# Fused Speculative Decoding is supported.
# fsd = FusedSpeculativeDecoder(draft_model, target_model, spec_length)
# fsd.to_neuron()  # Compile the fused speculative model



logger = logging.getLogger(__name__)

class NeuronHFAdapterWrap(torch.nn.Module):
    """
    Thin wrapper so .forward accepts input_ids, cache_ids=None
    and returns (logits, cache_ids) packaged as CausalLMOutputWithPast.
    """
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.cache_ids = None  # Initialize KV cache pointer storage
        self.config = adapter.config

    def forward(self, input_ids, cache_ids=None, **kwargs):
        """
        Runs a forward pass on the Neuron-compiled draft model with KV cache reuse.
        Accepts a cache_ids pointer to reuse past KV state, and returns logits and new cache_id.
        """
        # Use provided cache_ids or fall back to stored state
        if cache_ids is None:
            cache_ids = self.cache_ids
        logger.info(f"forward() called with cache_ids={cache_ids}")
        # Pass cache_ids to the underlying adapter to reuse KV cache
        if cache_ids is not None:
            out = self.adapter(input_ids=input_ids, cache_ids=cache_ids, return_dict=False)
        else:
            out = self.adapter(input_ids=input_ids, return_dict=False)
        # Unpack logits and new_cache
        if isinstance(out, (tuple, list)):
            logits, new_cache = out[0], (out[1] if len(out) > 1 else None)
        else:
            logits = out.logits if hasattr(out, "logits") else out
            new_cache = getattr(out, "cache_ids", None) or getattr(out, "past_key_values", None)
        # Normalize logits to 1D for sampling
        if hasattr(logits, "dim"):
            if logits.dim() == 3:
                logits = logits[0, -1, :]
            elif logits.dim() == 2:
                logits = logits[0]
        while isinstance(logits, (tuple, list)):
            logits = logits[0]
        # If adapter did not return a new_cache, compute it based on token positions
        if new_cache is None:
            # Determine starting index for new tokens
            if cache_ids is None:
                start_idx = 0
            elif isinstance(cache_ids, torch.Tensor):
                start_idx = int(cache_ids.max().item()) + 1
            else:
                start_idx = int(cache_ids) + 1
            seq_len = input_ids.shape[-1]
            new_cache = torch.arange(start_idx, start_idx + seq_len, dtype=torch.int32)
            if input_ids.dim() == 2:
                new_cache = new_cache.unsqueeze(0)
        # Store the updated cache state
        self.cache_ids = new_cache
        logger.info(f"adapter returned new_cache={new_cache}")
        return logits, new_cache

# Default sequence length (can be overridden by function arguments)
DEFAULT_SEQUENCE_LENGTH = 128

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
            with open(os.path.join(model_path, "config.json"), "r") as cf:
                cfg = json.load(cf)
            model_type = cfg.get("model_type", "")
        else:
            cfg = AutoTokenizer.from_pretrained(model_path).config if hasattr(AutoTokenizer, "from_pretrained") else {}
            model_type = getattr(cfg, "model_type", "")
    except Exception as e:
        logger.warning(f"Could not determine model type for '{model_path}': {e}")

    tp_degree = int(os.environ.get("NEURON_RT_NUM_CORES", "2"))
    if model_type.lower() == "llama" or "llama" in model_path.lower():
        logger.info(f"Compiling model using optimized LLaMA for Neuron ...")
        model = LlamaForSampling.from_pretrained(model_path, batch_size=1, amp='bf16',
                                                 n_positions=sequence_length, tp_degree=tp_degree)
        # Compile so the Neuron graph returns (logits, cache_id)
        model.enable_speculative_decoder(4)
        model.to_neuron()
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        adapter = HuggingFaceGenerationModelAdapter(hf_config, model)
        return NeuronHFAdapterWrap(adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.save_pretrained(compiled_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.save_pretrained(compiled_dir)
        logger.info(f"Model and tokenizer saved to '{compiled_dir}' (no Neuron compilation performed).")
        return model