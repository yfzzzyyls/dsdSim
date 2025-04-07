import time
import logging
import torch
from transformers import AutoTokenizer
from inference.model_loader import load_model
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def run_model(model_name: str, prompt: str, max_tokens: int = 50, sequence_length: int = 128, role: str = "target"):
    """
    Load a single model (target or draft) and generate text from a prompt, printing tokens one-by-one.
    Also collects performance metrics: total latency, token throughput, per-token generation time.
    The 'role' parameter is used for logging purposes and should be 'target' or 'draft'.
    Metrics are saved to CSV and JSON files.
    """
    logger.info(f"Loading {role} model '{model_name}' for standalone generation (sequence_length={sequence_length})...")
    model = load_model(model_name, sequence_length=sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if not prompt:
        logger.error("No prompt provided for generation.")
        return
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    generated_text = ""
    token_times = []
    tokens_generated = 0
    logger.info(f"Starting generation for prompt: {prompt!r}")
    start_time = time.time()
    for i in range(max_tokens):
        iter_start = time.time()
        output = model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        # Extract the last token generated
        if isinstance(output, (list, tuple)):
            token_id = int(output[0][-1])
        else:
            token_id = int(output[0, -1])
        token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        # Print the token for step-by-step verification
        print(f"Token {i+1}: {repr(token_text)}", flush=True)
        generated_text += token_text
        # Update input_ids with the new token for the next iteration
        new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1
        iter_end = time.time()
        token_times.append(iter_end - iter_start)
        # If an EOS token is generated, stop early
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            logger.info("EOS token encountered, stopping generation.")
            break
    total_time = time.time() - start_time
    avg_time = total_time / tokens_generated if tokens_generated > 0 else 0.0
    throughput = tokens_generated / total_time if total_time > 0 else float('inf')
    logger.info(f"{role.capitalize()} model generation completed in {total_time:.2f} seconds.")
    logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} tokens/sec, Average token time: {avg_time:.4f} sec")
    # Save metrics to CSV and JSON files with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"performance_{role}_only_{timestamp}.csv"
    json_file = f"performance_{role}_only_{timestamp}.json"
    try:
        with open(csv_file, 'w') as f:
            f.write("total_latency,tokens_generated,throughput,avg_token_time\n")
            f.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f}\n")
        metrics = {
            "total_latency": total_time,
            "tokens_generated": tokens_generated,
            "throughput": throughput,
            "per_token_times": token_times
        }
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
    except Exception as e:
        logger.error(f"Failed to save profiling data: {e}")
    full_output = prompt + generated_text
    print("\n=== Final Output ===\n" + full_output)
    return full_output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verification for standalone model generation")
    parser.add_argument("--model", type=str, required=True, help="Path to the model (target or draft) for verification")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model compilation/inference")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    parser.add_argument("--role", type=str, default="target", choices=["target", "draft"],
                        help="Specify the model role for logging purposes")
    args = parser.parse_args()
    run_model(args.model, prompt=args.prompt, max_tokens=args.max_tokens, sequence_length=args.sequence_length, role=args.role)
