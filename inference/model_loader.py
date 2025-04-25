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
        # self.cache_ids = pos_tensor if pos_tensor.ndim == 1 else pos_tensor.squeeze(0)
        self.cache_ids = torch.tensor([self._next_pos], dtype=torch.int32)

        # ------------------------------------------------------------------
        # Unpack logits to 1‑D tensor
        # ------------------------------------------------------------------
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out.logits if hasattr(out, "logits") else out

        # KEEP THE FULL (B, L, V) TENSOR – verification code will
        # select the rows it needs.
        if logits.dim() == 3:
            logits = logits[0, -1, :]
        if logits.dim() == 2:
            logits = logits[0]

        return logits, pos_tensor

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

def load_model(model_path: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH, spec_length: int = None):
    """
    Load or compile a model for inference.
    """
    logger.info(f"Attempting to download/compile from source.")
    model = compile_model(model_path, sequence_length=sequence_length, spec_length=spec_length)
    return model

def compile_model(model_path: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH, spec_length: int = None):
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
        model = LlamaForSampling.from_pretrained(
            model_path,
            batch_size=1,
            amp='bf16',
            n_positions=sequence_length,
            context_length_estimate=sequence_length,
            spec_length = spec_length,
            tp_degree=tp_degree,
            on_device_generation=False,
            return_all_logits=True,
            return_dict=True,
            torchscript=True,
            use_cache=True,
            trust_remote_code=True,
        )
        model.to_neuron()
        # ------------------------------------------------------------------
        # Ensure tokenizer & configs have an explicit PAD token.
        # This silences HF warnings and lets Neuron skip attention‑mask logic.
        # ------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True,
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
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Non‑Neuron path: loaded model & tokenizer; skipping on‑disk save because we re‑compile on every run.")
        return model
    

def compile_target_model(model_path: str,
                         sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                         spec_buckets: list[int] | None = None):
    """
    Compile the *target* model.  Differs from `compile_model` (used for
    the draft model) in that we always expose **all** logits for the
    tokens supplied in each forward pass so the verification step can
    access the full distribution.  No other behavioural changes are
    introduced.
    """
    # Compile γ + 1 buckets so verify pass includes the bonus‑token row
    max_gamma = 8
    spec_buckets = list(range(1, max_gamma + 2))   # [1,2,3,4,5,6,7,8,9]

    if spec_buckets is None:
        raise RuntimeError("spec_buckets must be provided for compile_target_model")

    logger.info(f"[Target‑compile] Compiling '{model_path}' → Neuron "
                f"(sequence_length={sequence_length},")

    tp_degree = int(os.environ.get("NEURON_RT_NUM_CORES", "2"))
    neuron_cfg = NeuronConfig(is_eagle_target=False,
                          cast_logits_dtype="bfloat16")   # <- enables safe cast
    model = LlamaForSampling.from_pretrained(
        model_path,
        batch_size            = 1,
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
        return_all_outputs    = True,   #  ← add this
        return_dict           = True,
        torchscript           = True,
        use_cache             = True,
        trust_remote_code     = True,
    )
    model.enable_speculative_decoder(spec_buckets)
    model.to_neuron()

    logger.info("Requested spec-length buckets: %s", spec_buckets)
    logger.info("Compiled speculation buckets: %s", list(model.decoder_lm_head_for_speculation.keys()))

    # Ensure PAD token is defined so we can always right‑pad inputs.
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
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
                      sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    """
    Convenience wrapper the *target* side should call instead of `load_model`.
    """
    return compile_target_model(model_path,
                                sequence_length=sequence_length,
                                spec_buckets=[1, 2, 4, 8])