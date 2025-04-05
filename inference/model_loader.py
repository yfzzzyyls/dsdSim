import torch
import torch_neuronx
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class LlamaDecoderWrapper(torch.nn.Module):
    """Wrapper to adjust LLaMA model's head_dim and key/value heads for AWS Trainium (Neuron) compilation."""
    def __init__(self, model, head_dim, num_key_value_heads):
        super().__init__()
        # Patch model configuration for head_dim and grouped key/value heads (GQA)
        model.config.head_dim = head_dim
        model.config.num_key_value_heads = num_key_value_heads
        # Update derived attributes in attention layers (num_key_value_groups) to avoid mismatch
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for layer in model.model.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.num_key_value_groups = model.config.num_attention_heads // num_key_value_heads
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_model(model_name_or_path, compile_to_neuron: bool = True):
    """
    Load a HuggingFace model and tokenizer. If on AWS Trainium (torch-neuronx available),
    compile the model for Neuron. Applies LlamaDecoderWrapper for LLaMA models to fix
    head_dim mismatch issues (using grouped query attention) without modifying Transformers.
    """
    # 1) Load model configuration with torch_dtype = bfloat16
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.torch_dtype = torch.bfloat16

    # 2) Load model with the updated config
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # 3) If the model is a LLaMA variant, apply wrapper for specific sizes
    if model.config.model_type == "llama":
        # Example case: LLaMA 1B (hidden_size=1024)
        if model.config.hidden_size == 1024:
            model = LlamaDecoderWrapper(model, head_dim=64, num_key_value_heads=8)
        # Example case: LLaMA 3B (hidden_size=2048)
        elif model.config.hidden_size == 2048:
            model = LlamaDecoderWrapper(model, head_dim=128, num_key_value_heads=8)

    # 4) Optionally compile the model to TorchScript on Neuron
    if compile_to_neuron and torch_neuronx is not None:
        model.eval()  # set to inference mode

        # Create dummy inputs in bfloat16; seq length = 32
        dummy_input_ids = torch.ones((1, 32), dtype=torch.bfloat16)
        dummy_attn_mask = torch.ones_like(dummy_input_ids)  # also bfloat16

        # Compile (trace) the model for Neuron
        model = torch_neuronx.trace(
            model,
            example_inputs=(dummy_input_ids, dummy_attn_mask)
        )

    return model, tokenizer