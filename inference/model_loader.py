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
        self._next_pos = 0  # next position index in the KV cache
        self.config = adapter.config

    # ------------------------------------------------------------------  
    # helper: build a (batch, length) int32 tensor [start, …, start+L‑1]  
    # ------------------------------------------------------------------
    def _build_pos(self, start: int, length: int, batch: int = 1):
        return (torch.arange(start, start + length, dtype=torch.int32)
                       .unsqueeze(0)            # -> (1, L)
                       .repeat(batch, 1))       # -> (B, L)

    def forward(self, input_ids, cache_ids=None, **kwargs):
        """
        Neuron draft forward with explicit per‑call KV‑cache positions.
        We maintain a cursor `_next_pos` so each incremental step passes
        ONLY the positions of the *new* tokens.  This avoids the
        “Tensor with N elements cannot be converted to Scalar” error.
        """
        B, L = input_ids.shape  # batch, new‑token count

        # ------------------------------------------------------------------
        # Decide which position tensor to pass for these L new tokens
        # ------------------------------------------------------------------
        if cache_ids is None:
            if self._next_pos == 0:
                # First (prompt‑priming) call – let Neuron allocate 0…L‑1
                pos_tensor = None                    # Neuron fills it
                next_pos_after = L
            else:
                # Incremental call – build positions [_next_pos, …]
                pos_tensor = self._build_pos(self._next_pos, L, B)
                next_pos_after = self._next_pos + L
        else:
            # Caller supplied explicit positions (e.g. during rollback)
            pos_tensor = cache_ids
            next_pos_after = int(cache_ids.max().item()) + 1

        # If we are generating exactly one token (B==1, L==1), Neuron expects
        # cache_ids to be 1‑D → torch.Size([1]); squeeze the batch dim.
        if pos_tensor is not None and pos_tensor.ndim == 2 and pos_tensor.size(0) == 1 and pos_tensor.size(1) == 1:
            pos_tensor = pos_tensor.squeeze(0)   # shape (1,)  – compatible with Neuron
        # ------------------------------------------------------------------
        # Run Neuron adapter
        # ------------------------------------------------------------------
        out = self.adapter(input_ids=input_ids,
                           cache_ids=pos_tensor,
                           return_dict=False,
                           **kwargs)

        # ------------------------------------------------------------------
        # Update internal cursor & cache pointer
        # ------------------------------------------------------------------
        self._next_pos = next_pos_after
        if pos_tensor is None:
            # Reconstruct positions 0…L‑1 for the prompt stage
            # For prompt (B==1, L>1) keep 2‑D; for B==1, L==1 we can squeeze
            pos_tensor = self._build_pos(0, L, B)
            if pos_tensor.ndim == 2 and pos_tensor.size(0) == 1 and pos_tensor.size(1) == 1:
                pos_tensor = pos_tensor.squeeze(0)
        self.cache_ids = pos_tensor if pos_tensor.ndim == 1 else pos_tensor.squeeze(0)

        # ------------------------------------------------------------------
        # Unpack logits to 1‑D tensor
        # ------------------------------------------------------------------
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out.logits if hasattr(out, "logits") else out

        if logits.dim() == 3:
            logits = logits[0, -1, :]
        elif logits.dim() == 2:
            logits = logits[0]
        while isinstance(logits, (tuple, list)):
            logits = logits[0]

        return logits, pos_tensor

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
                                                 n_positions=sequence_length, 
                                                 context_length_estimate=sequence_length,
                                                 tp_degree=tp_degree)
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