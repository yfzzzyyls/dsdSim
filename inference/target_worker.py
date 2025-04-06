import model_loader
import torch

class TargetWorker:
    def __init__(self, model_name):
        # Load the target model (attempt to use precompiled model on Neuron if available)
        self.model = model_loader.load_model(model_name)
        # Flag to indicate if model is a compiled Neuron model
        self.compiled = hasattr(self.model, "is_compiled") and self.model.is_compiled

    def generate_token(self, input_ids):
        """Generate a single token using the target model given the current input sequence."""
        # Ensure attention mask matches input length
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        orig_len = input_ids.shape[1]
        if self.compiled:
            # Pad input and mask to max length for compiled model (to match traced shape)
            max_len = model_loader.DEFAULT_MAX_SEQ_LEN
            if orig_len > max_len:
                raise RuntimeError(f"Input length {orig_len} exceeds compiled max length {max_len}.")
            if orig_len < max_len:
                pad_len = max_len - orig_len
                pad_ids = torch.zeros((1, pad_len), dtype=torch.long)
                pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
                input_ids = torch.cat([input_ids, pad_ids], dim=1)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
        # Run the model (Neuron or HF)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # Extract logits from output
        if self.compiled:
            # Compiled model likely returns a tuple or tensor
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # get the logits tensor
            logits = outputs
            # Only consider the first orig_len tokens (ignore padded positions)
            logits = logits[:, :orig_len, :]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        # Return the logits of the last real token in the sequence
        return logits[:, orig_len-1, :]

    def verify_sequence(self, input_ids):
        """
        Verify a sequence of tokens using the target model.
        Returns logits for each position in the sequence.
        """
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        orig_len = input_ids.shape[1]
        if self.compiled:
            max_len = model_loader.DEFAULT_MAX_SEQ_LEN
            if orig_len > max_len:
                raise RuntimeError(f"Input length {orig_len} exceeds compiled max length {max_len}.")
            if orig_len < max_len:
                pad_len = max_len - orig_len
                pad_ids = torch.zeros((1, pad_len), dtype=torch.long)
                pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
                input_ids = torch.cat([input_ids, pad_ids], dim=1)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        if self.compiled:
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            logits = outputs[:, :orig_len, :]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return logits