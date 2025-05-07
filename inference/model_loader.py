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
from transformers_neuronx.config import NeuronConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers_neuronx.fused_speculation import FusedSpeculativeDecoder
import types
# Fused Speculative Decoding is supported.
# fsd = FusedSpeculativeDecoder(draft_model, target_model, spec_length)
# fsd.to_neuron()  # Compile the fused speculative model
SPEC_LENGTH_BUCKETS = [1, 2, 3, 4, 5]
logger = logging.getLogger(__name__)

class NeuronHFAdapterWrap(torch.nn.Module):
    """
    Thin wrapper so .forward accepts batched input_ids of shape (B, L)
    and **retains** batch/length dimensions in the returned logits.
    """
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.cache_ids = None  # Initialize KV cache pointer storage
        self.config = adapter.config

    # ------------------------------------------------------------------  
    # helper: build a (batch, length) int32 tensor [start, …, start+L‑1]  
    # ------------------------------------------------------------------
    def _build_pos(self, start: int, length: int, batch: int = 1):
        return (torch.arange(start, start + length, dtype=torch.int32)
                       .unsqueeze(0)            # -> (1, L)
                       .repeat(batch, 1))       # -> (B, L)

    # ------------------------------------------------------------------
    # Speculative helpers – expose Neuron-native multi-token kernels
    # ------------------------------------------------------------------
    def speculative_forward(self, input_ids, *, cache_ids=None,
                            spec_length=None, **kwargs):
        """
        Pass-through to the Neuron graph that returns logits for **all**
        tokens in `input_ids` (shape = (B, N, V)).
        """

        # Ensure speculation length default mirrors input_ids length
        if spec_length is None:
            raise RuntimeError("spec_length must be provided for speculative_forward")
            # spec_length = input_ids.shape[1]

        return self.adapter.model.speculative_forward(
            input_ids=input_ids,
            cache_ids=cache_ids,
            speculation_length=spec_length, # or input_ids.shape[1],
            **kwargs
        )

    def tree_speculative_forward(self, input_ids, *, cache_ids=None,
                                 spec_length=None, **kwargs):

        # Ensure speculation length default mirrors input_ids length
        if spec_length is None:
            raise RuntimeError("spec_length must be provided for tree_speculative_forward")
            # spec_length = input_ids.shape[1]
        logger.info(f"Calling tree_speculative_forward with spec_length={spec_length}")
        return self.adapter.model.tree_speculative_forward(
            input_ids=input_ids,
            cache_ids=cache_ids,
            speculation_length=spec_length,
            **kwargs,
        )

    def forward(self, input_ids, cache_ids=None, **kwargs):
        """
        Thin pass‑through to the underlying Neuron adapter.

        Parameters
        ----------
        input_ids : LongTensor
            Shape (B, L) where B is batch and L is token‑count.
        cache_ids : IntTensor or None
            KV‑cache pointer vector of shape (B,) indicating current
            sequence lengths on device.

        Returns
        -------
        logits : FloatTensor
            • Shape (B, V) if L == 1  (one‑token forward)  
            • Shape (B, L, V) for multi‑token calls  
            where V is vocab size.
        cache_ids : same object that was passed in (for compatibility)
        """
        out = self.adapter(
            input_ids=input_ids,
            cache_ids=cache_ids,
            return_dict=False,
            **kwargs,
        )

        # ----------------------------------------------
        # Extract logits tensor from various return types
        # ----------------------------------------------
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out.logits if hasattr(out, "logits") else out

        # ------------------------------------------------------------------
        # IMPORTANT CHANGE: **do NOT squeeze batch / length dims**.
        # Leave logits as‑is so downstream code can handle batched tensors:
        #   (B, L, V)  or  (B, V)  depending on input shape.
        # ------------------------------------------------------------------
        return logits, cache_ids

    # ------------------------------------------------------------------
    # Convenience: greedy sampling so verify.py can run target model alone
    # ------------------------------------------------------------------
    def sample(
        self,
        input_ids,
        sequence_length: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        Generate `sequence_length - input_ids.shape[1]` new tokens using the
        same high‑level sampling implementation that HuggingFace provides.
        This avoids subtle bugs in the bespoke nucleus loop.
        """
        num_new = max(0, sequence_length - input_ids.shape[1])
        if num_new == 0:
            return input_ids

        out = self.adapter.generate(
            input_ids,
            max_new_tokens=num_new,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.config.eos_token_id,
        )
        return out

# Default sequence length (can be overridden by function arguments)
DEFAULT_SEQUENCE_LENGTH = 128

def load_model(model_path: str,
               sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
               spec_length: int | None = None,
               batch_size: int = 1):
    """
    Load or compile a model for inference.
    """
    logger.info(f"Attempting to download/compile from source.")
    model = compile_model(model_path, sequence_length=sequence_length, spec_length=spec_length, batch_size=batch_size)
    return model

def compile_model(model_path: str,
                  sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                  spec_length: int | None = None,
                  batch_size: int = 1):
    """
    Compile a model for AWS Neuron. Loads the model (from HF Hub or local checkpoint),
    compiles it to a TorchScript that can run on NeuronCores, and saves the compiled model
    and tokenizer to a local folder for future use.
    """
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
        # Enable 2‑D cache‑id layout so batched single‑token calls can pass
        # a vector of cache pointers without hitting `.item()` errors in
        # decoder.forward_single().
        neuron_cfg = NeuronConfig(padding_side='right')
        model = LlamaForSampling.from_pretrained(
            model_path,
            batch_size=batch_size,
            amp='bf16',
            n_positions=sequence_length,
            context_length_estimate=sequence_length,
            spec_length = spec_length,
            neuron_config = neuron_cfg,
            tp_degree=tp_degree,
            on_device_generation=False,
            return_all_logits=True,
            return_dict=True,
            torchscript=True,
            use_cache=True,
            trust_remote_code=True,
            fuse_qkv=True,
            attention_layout="BSH",
            use_2d_cache_ids=True
        )
        model.to_neuron()
        # ------------------------------------------------------------------
        # Ensure tokenizer & configs have an explicit PAD token.
        # This silences HF warnings and lets Neuron skip attention‑mask logic.
        # ------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  padding_side="right",
                                                  use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token           # reuse </s>
        # Make the id available everywhere
        tokenizer.pad_token_id = tokenizer.eos_token_id
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hf_config.pad_token_id is None:
            hf_config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = hf_config.pad_token_id
        adapter = HuggingFaceGenerationModelAdapter(hf_config, model)

        # --------------------------------------------------------------
        # Sanity‑check: both Neuron config *and* tokenizer must use
        # right‑padding now that use_2d_cache_ids is deprecated.
        # --------------------------------------------------------------
        assert model.neuron_config.padding_side == "right", \
            "NeuronConfig.padding_side must be 'right' for batched cache IDs"
        assert tokenizer.padding_side == "right", \
            "Tokenizer.padding_side must be 'right' for batched cache IDs"
        # --------------------------------------------------------------
        # DEBUG: Inspect raw Neuron model output shape *before* wrapping
        # --------------------------------------------------------------
        # try:
        #     # create a small random batch (1, 4) just to probe the output
        #     debug_batch = torch.randint(
        #         low=0,
        #         high=hf_config.vocab_size,
        #         size=(1, 4),
        #         dtype=torch.int64
        #     )
        #     raw_out = adapter(input_ids=debug_batch, cache_ids=None, return_dict=True)
        #     logger.info(f"[DEBUG] raw adapter logits shape: {raw_out.logits.shape}")
        # except Exception as e:
        #     logger.warning(f"[DEBUG] raw adapter call failed: {e}")


        # --------------------------------------------------------------
        # Quick sanity‑check: call the underlying model.forward so we can pass spec_length.
        # --------------------------------------------------------------
        # debug_batch = torch.randint(0, hf_config.vocab_size, (1, 4))
        # # Call the underlying LlamaForSampling forward so we can pass spec_length.
        # raw_out = model.forward(
        #     input_ids=debug_batch,
        #     cache_ids=None,
        #     spec_length=debug_batch.shape[1],   # ask for all-token logits
        #     return_dict=True,
        # )
        # logger.info(
        #     "[DEBUG] LlamaForSampling raw logits shape %s",
        #     raw_out.shape,
        # )

        # Wrap the adapter so downstream code keeps the KV‑pointer logic
        return NeuronHFAdapterWrap(adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Non-Neuron path: loaded model & tokenizer; skipping on‑disk save because we re‑compile on every run.")
        return model
    

def compile_target_model(model_path: str,
                         sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                         spec_buckets: list[int] | None = None,
                         batch_size: int = 1):
    """
    Compile the *target* model.  Differs from `compile_model` (used for
    the draft model) in that we always expose **all** logits for the
    tokens supplied in each forward pass so the verification step can
    access the full distribution.  No other behavioural changes are
    introduced.
    """
    # Compile γ + 1 buckets so verify pass includes the bonus‑token row
    spec_buckets = SPEC_LENGTH_BUCKETS

    logger.info(f"[Target-compile] Compiling '{model_path}' -> Neuron "
                f"(sequence_length={sequence_length})")

    tp_degree = int(os.environ.get("NEURON_RT_NUM_CORES", "2"))
    neuron_cfg = NeuronConfig(
        is_eagle_target=False,
        cast_logits_dtype="bfloat16",
        padding_side="right",
    )
    model = LlamaForSampling.from_pretrained(
        model_path,
        batch_size            = batch_size,
        amp                   = "bf16",
        n_positions           = sequence_length,
        context_length_estimate = sequence_length,
        # Provide the *largest* bucket to the base model so positional buffers
        # are sized correctly; per‑length graphs are added by
        # enable_speculative_decoder() below.
        spec_length           = max(spec_buckets),
        neuron_config         = neuron_cfg,
        tp_degree             = tp_degree,
        on_device_generation  = False,        # we need raw logits on host
        return_all_logits     = False,         # **key line – full distributions**
        return_all_outputs    = True,
        return_dict           = True,
        torchscript           = True,
        use_cache             = True,
        trust_remote_code     = True,
        fuse_qkv              = True,
        attention_layout      = "BSH",
        use_2d_cache_ids      = True,
    )
    model.enable_speculative_decoder(spec_buckets)
    model.to_neuron()

    logger.info("Requested spec-length buckets: %s", spec_buckets)
    logger.info("Compiled speculation buckets: %s", list(model.decoder_lm_head_for_speculation.keys()))

    # Ensure PAD token is defined so we can always right‑pad inputs.
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              padding_side="right",
                                              use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hf_config.pad_token_id is None:
        hf_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = hf_config.pad_token_id

    adapter = HuggingFaceGenerationModelAdapter(hf_config, model)
    return NeuronHFAdapterWrap(adapter)


def load_target_model(model_path: str,
                      sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                      batch_size: int = 1):
    """
    Convenience wrapper the *target* side should call instead of `load_model`.
    """
    return compile_target_model(model_path,
                                sequence_length=sequence_length,
                                batch_size=batch_size)