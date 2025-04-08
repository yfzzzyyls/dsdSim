import time
import logging
import torch
from transformers import AutoTokenizer
from inference.model_loader import load_model
import json
from datetime import datetime
from inference.performance_profile import PerformanceProfiler

logger = logging.getLogger(__name__)

def run_model(model_name: str, prompt: str, max_tokens: int = 50, sequence_length: int = 128, role: str = "target", profile: bool = False):
    """
    Load a model (target or draft) and generate text from a prompt, printing tokens one-by-one.
    Performance metrics are recorded with the PerformanceProfiler if `profile` is True.
    """
    logger.info(f"Loading {role} model '{model_name}' for standalone generation (sequence_length={sequence_length})...")
    model = load_model(model_name, sequence_length=sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if not prompt:
        logger.error("No prompt provided for generation.")
        return

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    generated_text = ""
    profiler = PerformanceProfiler() if profile else None
    if profiler:
        profiler.start()

    logger.info(f"Starting generation for prompt: {prompt!r}")
    for i in range(max_tokens):
        iter_start = time.time() if profiler else None
        output = model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        # Extract last token
        if isinstance(output, (list, tuple)):
            token_id = int(output[0][-1])
        else:
            token_id = int(output[0, -1])
        # Use improved spacing
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        print(f"Token {i+1}: {repr(token_text)}", flush=True)
        generated_text += token_text
        new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        if profiler:
            iter_end = time.time()
            profiler.record_token(iter_end - iter_start)
        # Early stop if EOS
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            logger.info("EOS token encountered, stopping generation.")
            break

    if profiler:
        profiler.finish()
        profiler.export_metrics(role=role, output_dir=".")

    full_output = prompt + generated_text
    print("\n=== Final Output ===\n" + full_output)
    return full_output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verification for standalone model generation")
    parser.add_argument("--model", type=str, required=True, help="Path to the model (target or draft) for verification")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model inference")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    parser.add_argument("--role", type=str, default="target", choices=["target", "draft"],
                        help="Specify the model role for logging purposes")
    args = parser.parse_args()
    run_model(args.model, prompt=args.prompt, max_tokens=args.max_tokens,
              sequence_length=args.sequence_length, role=args.role, profile=args.profile)
