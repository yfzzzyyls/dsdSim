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
# fused speculative decoding utilities
# generic Neuron interfaces (present in transformers‑neuronx >= 0.13)
from transformers_neuronx import NeuronAutoModelForCausalLM, NeuronConfig, GenerationConfig

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
    # def _build_pos(self, start: int, length: int, batch: int = 1):
    #     return (torch.arange(start, start + length, dtype=torch.int32)
    #                    .unsqueeze(0)            # -> (1, L)
    #                    .repeat(batch, 1))       # -> (B, L)
    # inference/model_loader.py  – inside class NeuronHFAdapterWrap

    # helper: build a 1‑D Int32 vector  [start, …, start+L‑1]
    def _build_pos(self, start: int, length: int):
        return torch.arange(start, start + length, dtype=torch.int32)   # <‑‑ 1‑D

    def forward(self, input_ids, cache_ids=None, return_all_logits=False, **kwargs):
        """
        Run one incremental forward on the Neuron adapter, fabricating an explicit 2‑D `cache_ids`
        tensor for models that require it, but passing `cache_ids=None` for draft models
        that expose a speculative‑decoder interface (i.e., have attribute `spec_length`).

        Parameters
        ----------
        input_ids : torch.LongTensor  shape (B, L_new)
        cache_ids : torch.IntTensor   shape (B, L_new) OR None
        """
        B, L = input_ids.shape
        # ------------------------------------------------------------------
        # Decide whether to fabricate an explicit position tensor.
        # Draft models compiled with `.enable_speculative_decoder()`
        # expect `cache_ids=None`; giving them a (B,L) tensor triggers
        # the RuntimeError you saw ("Tensor with 12 elements cannot be
        # converted to scalar").  We therefore *skip* building cache_ids
        # when the wrapped adapter exposes attribute `spec_length`.
        # ------------------------------------------------------------------
        if cache_ids is None:
            needs_explicit_pos = not hasattr(self.adapter, "spec_length")
            if needs_explicit_pos:
                # Target‑side or non‑fused model → build [start…start+L‑1]
                # cache_ids = self._build_pos(self._next_pos, L, B)
                cache_ids = self._build_pos(self._next_pos, L)   # now 1‑D
            else:
                # Draft model with fused speculative decoder – let Neuron
                # generate positions internally.
                cache_ids = None
        # else: caller supplied explicit tensor – use it as‑is

        # ------------------------------------------------------------------
        # Forward through the wrapped HuggingFaceGenerationModelAdapter
        # ------------------------------------------------------------------
        out = self.adapter(input_ids=input_ids,
                           cache_ids=cache_ids,
                           return_dict=False,
                           **kwargs)

        # Advance internal pointer
        self._next_pos = int(cache_ids.max().item()) + 1
        self.cache_ids = torch.tensor([self._next_pos], dtype=torch.int32)

        # ------------------------------------------------------------------
        # Extract logits
        #   • if return_all_logits==False  → keep only the *last* step
        #   • if return_all_logits==True   → keep every step
        # ------------------------------------------------------------------
        logits = out[0] if isinstance(out, (tuple, list)) else (
                 out.logits if hasattr(out, "logits") else out)

        if logits.ndim == 3:                           # (B, L, V)
            if return_all_logits:
                # keep full (L, V) slice, squeeze batch if B == 1
                logits = logits.squeeze(0) if logits.size(0) == 1 else logits
            else:
                logits = logits[:, -1, :]              # last step only
        if logits.ndim == 2 and logits.size(0) == 1 and not return_all_logits:
            logits = logits[0]                         # → (V,)

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
    
    # inference/model_loader.py  – inside class NeuronHFAdapterWrap


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
    # ------------------------------------------------------------------
    # Ensure the compiled graph supports “prompt + max γ + safety”
    # ------------------------------------------------------------------
    ctx_len = sequence_length
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
        model = LlamaForSampling.from_pretrained(
            model_path,
            batch_size=1,
            amp='bf16',
            n_positions=ctx_len,
            tp_degree=tp_degree
        )

        # If a speculative length is provided, enable that many-token speculative decoder
        model.to_neuron()
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        adapter = HuggingFaceGenerationModelAdapter(hf_config, model)
        return NeuronHFAdapterWrap(adapter)
    else:
        RuntimeError(f"Model type '{model_type}' not supported for compilation.")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.save_pretrained(compiled_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.save_pretrained(compiled_dir)
        logger.info(f"Model and tokenizer saved to '{compiled_dir}' (no Neuron compilation performed).")
        return model
    
# ---------------------------------------------------------------------------
# Target‑only helpers: compile a fused speculative decoder (γ draft + 1 bonus)
# ---------------------------------------------------------------------------

def compile_target_model(
    model_path: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    spec_length: int = 4,
    top_k: int = 512,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    """
    Compile ONE Neuron target model (no fused graph) so we can
    capture FULL logits for every draft token.

    Returns
    -------
    NeuronHFAdapterWrap  – exposes .forward returning logits
    """
    logger.info(
        "[compile_target_model] building Neuron target‑only model %s  γ=%d  seq_len=%d",
        model_path, spec_length, sequence_length,
    )

    ctx_len = sequence_length
    tp_deg  = int(os.environ.get("NEURON_RT_NUM_CORES", "2"))

    # ——— Neuron config, only used when on_device_generation = True ———
    gen_cfg = GenerationConfig(top_k=top_k, top_p=top_p,
                               do_sample=True, temperature=temperature)
    
    neuron_cfg = NeuronConfig(
        padding_side="right",
        attention_layout="BSH",
        collectives_layout="BSH",
        on_device_embedding=True,       # keep fast embeds
        on_device_generation=None,      # <-- DISABLE device sampling ⇒ we get logits back
    )

    target = NeuronAutoModelForCausalLM.from_pretrained(
        model_path,
        batch_size=1,
        n_positions=ctx_len,
        tp_degree=tp_deg,
        amp="bf16",
        neuron_config=neuron_cfg,
    )
    target.to_neuron()     # compile

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    adapter = HuggingFaceGenerationModelAdapter(cfg, target)
    logger.info("[compile_target_model] Neuron graph compiled (ctx_len=%d)", ctx_len)
    return NeuronHFAdapterWrap(adapter)


def load_target_model(
    model_path: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    spec_length: int = 4,
    top_k: int = 512,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    """
    Always (re)compile the fused target graph.  No on‑disk caching to keep the
    code path simple and stateless.
    """
    return compile_target_model(
        model_path,
        sequence_length=sequence_length,
        spec_length=spec_length,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )