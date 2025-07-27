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
from transformers_neuronx.config import NeuronConfig, ContinuousBatchingConfig
import types
# Fused Speculative Decoding is supported.
# fsd = FusedSpeculativeDecoder(draft_model, target_model, spec_length)
# fsd.to_neuron()  # Compile the fused speculative model
 # Speculation buckets are specified by *length*; the compiler will build
 # both batch‑1 and batch‑2 heads automatically when `dynamic_batch_size=True`.
SPEC_LENGTH_BUCKETS = [3, 5]  # supports gamma=2 and gamma=4
BATCH_BUCKETS = [1, 2]
logger = logging.getLogger(__name__)

def get_spec_bucket_for_gamma(gamma: int, available_buckets=None) -> int:
    """
    Map a requested gamma to the smallest available spec bucket that can accommodate it.
    
    Args:
        gamma: Requested speculation length
        available_buckets: List of compiled bucket sizes (e.g., [3, 5])
    
    Returns:
        The bucket size to use (e.g., if gamma=3 and buckets=[3,5], return 5 for the +1 bonus token)
    """
    if available_buckets is None:
        available_buckets = SPEC_LENGTH_BUCKETS
    
    # We need gamma + 1 tokens (gamma draft + 1 bonus), so find the smallest bucket >= gamma + 1
    required_size = gamma + 1
    
    valid_buckets = [b for b in available_buckets if b >= required_size]
    if not valid_buckets:
        raise ValueError(f"No compiled bucket can handle gamma={gamma}. "
                        f"Available buckets: {available_buckets}, required size: {required_size}")
    
    return min(valid_buckets)

def pad_tokens_to_bucket(tokens: list, target_bucket_size: int, pad_token_id: int = 0) -> tuple[list, int]:
    """
    Pad token list to match the target bucket size.
    
    Args:
        tokens: List of token IDs
        target_bucket_size: The bucket size to pad to
        pad_token_id: Token ID to use for padding
    
    Returns:
        (padded_tokens, original_length) where original_length is the number of real tokens
    """
    original_length = len(tokens)
    if original_length >= target_bucket_size:
        # Truncate if too long (shouldn't happen with proper bucket selection)
        return tokens[:target_bucket_size], min(original_length, target_bucket_size)
    
    # Pad to target size
    padded_tokens = tokens + [pad_token_id] * (target_bucket_size - original_length)
    return padded_tokens, original_length

class NeuronHFAdapterWrap(torch.nn.Module):
    """
    Thin wrapper mainly used to manage the KV cache pointer for the
    underlying Neuron model.
    """
    def __init__(self, adapter, batch_size: int = 1):
        super().__init__()
        self.adapter = adapter
        # Initialise KV‑cache pointer storage
        self.cache_id = torch.tensor([0], dtype=torch.int32)               # shape (1,)
        self.batched_cache_ids = torch.zeros(batch_size, dtype=torch.int32) # shape (B,)
        self.config = adapter.config

    def update_cache(self, delta):
        """
        Increment the *single‑row* KV‑cache pointer by `delta`.
        `delta` may be an int, float, or 0‑D tensor; it is cast to int.
        """
        # Normalise delta to Python int
        if isinstance(delta, torch.Tensor):
            delta = int(delta.item())
        else:
            delta = int(delta)

        # Lazily initialise cache_ids on first call
        assert self.cache_id is not None, "cache_ids should not be None"

        self.cache_id = self.cache_id + delta

    def update_batched_cache(self, delta, row_idx):
        """
        Increment specific rows of the batched `(B,)` KV‑cache pointer by `delta`.
        `delta` may be an int, float, or 0‑D tensor; it is cast to int.
        The rows to update are specified by `row_idx`, which may be an int, list, tuple, or tensor.
        Only manipulates self.batched_cache_ids.
        """
        # --------------------------------------------------------------
        # Normalize delta to an int
        # --------------------------------------------------------------
        if isinstance(delta, torch.Tensor):
            delta = int(delta.item())
        else:
            delta = int(delta)

        # --------------------------------------------------------------
        # Ensure batched pointer tensor exists
        # --------------------------------------------------------------
        assert self.batched_cache_ids is not None, "batched_cache_ids should not be None"

        # --------------------------------------------------------------
        # Convert row_idx into a flat list of integer indices
        # --------------------------------------------------------------
        if isinstance(row_idx, torch.Tensor):
            indices = row_idx.flatten().tolist()
        elif isinstance(row_idx, (list, tuple)):
            indices = list(row_idx)
        else:
            indices = [int(row_idx)]

        # Validate all indices
        B = self.batched_cache_ids.shape[0]
        assert all(0 <= idx < B for idx in indices), (
            f"row_idx values {indices} out of range for batched_cache_ids with length {B}"
        )
        # --------------------------------------------------------------
        # Increment only the requested rows
        # --------------------------------------------------------------
        for idx in indices:
            self.batched_cache_ids[idx] += delta

    # ------------------------------------------------------------------
    # Accessors for external components
    # ------------------------------------------------------------------
    def get_cache_id_vec(self):
        """
        Return a **clone** of the single‑row KV‑cache pointer vector so
        callers cannot mutate internal state accidentally.
        """
        assert self.cache_id is not None, "cache_id should not be None"
        return self.cache_id.clone()

    def get_batched_cache_id_vec(self):
        """
        Return a **clone** of the batched `(B,)` KV‑cache pointer vector.
        """
        assert self.batched_cache_ids is not None, "batched_cache_ids should not be None"
        return self.batched_cache_ids.clone()


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

        All KV‑cache management (cache_ids and next_pos) is handled by the
        caller.  This wrapper simply forwards the tensors and returns the
        logits for the last token plus the same `cache_ids` object so
        existing call‑sites that expect `(logits, pos_tensor)` keep working.
        """
        out = self.adapter(input_ids=input_ids,
                           cache_ids=cache_ids,
                           return_dict=False,
                           **kwargs)

        # Extract logits tensor
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out.logits if hasattr(out, "logits") else out

        # Reduce shape to (V)
        if logits.dim() == 3:       # (B, L, V)
            logits = logits[0, -1, :]
        elif logits.dim() == 2:     # (B, V)
            logits = logits[0]

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
        neuron_cfg = NeuronConfig(
            padding_side="right",
        )
        model = LlamaForSampling.from_pretrained(
            model_path,
            batch_size=batch_size,
            amp='bf16',
            n_positions=sequence_length,
            context_length_estimate=sequence_length,
            spec_length=spec_length,
            neuron_config=neuron_cfg,
            tp_degree=tp_degree,
            on_device_generation=False,
            return_all_logits=True,
            return_dict=True,
            torchscript=True,
            use_cache=True,
            trust_remote_code=True,
            fuse_qkv=True,
            attention_layout="BSH",
            enable_chunked_prefill=False,
            use_2d_cache_ids=True,
        )
        model.to_neuron()
        # ------------------------------------------------------------------
        # Ensure tokenizer & configs have an explicit PAD token.
        # This silences HF warnings and lets Neuron skip attention‑mask logic.
        # ------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        # ----------------------------------------------------------
        # Standardise: always use ID 0 as the PAD token so every
        # component agrees (draft, target, scheduler).
        # ----------------------------------------------------------
        tokenizer.pad_token_id = 0
        tokenizer.pad_token    = tokenizer.convert_ids_to_tokens(0)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_config.pad_token_id = 0
        model.config.pad_token_id = 0
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
        return NeuronHFAdapterWrap(adapter, batch_size=batch_size)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.pad_token_id = 0
        tokenizer.pad_token    = tokenizer.convert_ids_to_tokens(0)
        # Optionally set config pad_token_id if available
        if hasattr(model, "config"):
            model.config.pad_token_id = 0
        logger.info("Non‑Neuron path: loaded model & tokenizer; skipping on‑disk save because we re‑compile on every run.")
        return NeuronHFAdapterWrap(HuggingFaceGenerationModelAdapter(model.config, model), batch_size=batch_size)
    

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
    # --------------------------------------------------------------
    # Naïve Continuous Batching: minimal parameters for current SDK
    # --------------------------------------------------------------
    CB = ContinuousBatchingConfig(
        max_num_seqs   = batch_size,       # logical seq‑ids / physical rows
        max_model_len  = sequence_length,  # bucket size (128)
    )

    neuron_cfg = NeuronConfig(
        continuous_batching = CB,
        padding_side        = "right",
        is_eagle_target     = False,
        cast_logits_dtype   = "bfloat16",
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
        return_all_logits     = True,          # need full vocab probs for verification
        return_all_outputs    = True,
        return_dict           = True,
        torchscript           = True,
        use_cache             = True,
        trust_remote_code     = True,
        fuse_qkv              = True,
        attention_layout      = "BSH",
        use_2d_cache_ids      = True,
        dynamic_batch_size    = True,      # build (len,1) and (len,2) heads
        enable_chunked_prefill= False,
    )
    model.enable_speculative_decoder(SPEC_LENGTH_BUCKETS, BATCH_BUCKETS)
    model.to_neuron()

    logger.info("Requested spec-length buckets: %s", spec_buckets)
    logger.info("Compiled speculation buckets: %s", list(model.decoder_lm_head_for_speculation.keys()))

    # Ensure PAD token is defined so we can always right‑pad inputs.
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              use_fast=False)
    if tokenizer.pad_token is None:
        pad_tok_string = tokenizer.convert_ids_to_tokens(0)
        tokenizer.pad_token = pad_tok_string
    tokenizer.pad_token_id = 0

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hf_config.pad_token_id is None:
        hf_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = hf_config.pad_token_id

    adapter = HuggingFaceGenerationModelAdapter(hf_config, model)
    return NeuronHFAdapterWrap(adapter, batch_size=batch_size)


def load_target_model(model_path: str,
                      sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                      batch_size: int = 1):
    """
    Convenience wrapper the *target* side should call instead of `load_model`.
    """
    return compile_target_model(model_path,
                                sequence_length=sequence_length,
                                batch_size=batch_size)


def load_fused_speculative_model(draft_model_path: str,
                                 target_model_path: str,
                                 sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                                 speculation_length: int = 5,
                                 batch_size: int = 1,
                                 tp_degree: int = 4):
    """
    Load and compile a fused speculative decoding model.
    
    Args:
        draft_model_path: Path to the draft model
        target_model_path: Path to the target model
        sequence_length: Maximum sequence length
        speculation_length: Number of tokens to speculate at once
        batch_size: Batch size (ignored - fused models always use batch_size=1)
        tp_degree: Tensor parallelism degree
        
    Returns:
        (fused_model, tokenizer) tuple
    """
    logger.info(f"Loading fused speculative model with draft={draft_model_path}, target={target_model_path}")
    
    # Fused models always use batch_size=1
    batch_size = 1
    
    # Create NeuronConfig for fused model
    from transformers_neuronx.config import GenerationConfig
    
    # Create a GenerationConfig for on-device generation
    # This is required for fused speculative decoding
    generation_config = GenerationConfig(
        max_length=sequence_length,
        do_sample=True,
        top_k=128,      # Match distributed TOP_K
        top_p=0.9,      # Match distributed default
        temperature=1.0, # Match distributed default
        eos_token_id=2,  # Common EOS token ID for LLaMA models
        dynamic=False,   # Match distributed (no dynamic changes)
    )
    
    # Create NeuronConfig with both on_device_embedding AND on_device_generation
    neuron_config = NeuronConfig(
        on_device_embedding=True,           # Required for fused speculation
        on_device_generation=generation_config,  # Required for fused speculation
        padding_side="right",
    )
    
    # Load draft model
    logger.info("Loading draft model...")
    draft_model = LlamaForSampling.from_pretrained(
        draft_model_path,
        batch_size=1,  # Fused models use batch_size=1
        amp='bf16',
        n_positions=sequence_length,
        context_length_estimate=sequence_length,  # Single value, not list
        neuron_config=neuron_config,
        tp_degree=tp_degree,  # Must match target model for fused speculation
        return_dict=True,
        use_cache=True,
        fuse_qkv=True,
        attention_layout="BSH",
    )
    
    # Compile draft model to Neuron BEFORE creating fused decoder
    logger.info("Compiling draft model to Neuron...")
    draft_model.to_neuron()
    
    # Load target model
    logger.info("Loading target model...")
    target_model = LlamaForSampling.from_pretrained(
        target_model_path,
        batch_size=1,  # Fused models use batch_size=1
        amp='bf16',
        n_positions=sequence_length,
        context_length_estimate=sequence_length,  # Single value, not list
        neuron_config=neuron_config,
        tp_degree=tp_degree,
        return_dict=True,
        use_cache=True,
        fuse_qkv=True,
        attention_layout="BSH",
    )
    
    # Compile target model to Neuron BEFORE creating fused decoder
    logger.info("Compiling target model to Neuron...")
    target_model.to_neuron()
    
    # Create the fused speculative decoder
    logger.info("Creating fused speculative decoder...")
    fused_model = FusedSpeculativeDecoder(
        draft_model,
        target_model,
        k=speculation_length  # k is the number of draft tokens to speculate
    )
    
    # Compile fused model to Neuron
    logger.info("Compiling fused model to Neuron...")
    fused_model.to_neuron()
    
    # Load tokenizer (use target model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Fused speculative model loaded and compiled successfully")
    return fused_model, tokenizer