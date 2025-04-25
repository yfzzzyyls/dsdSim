import time
import logging
import torch
from transformers import AutoTokenizer
from inference.model_loader import load_model
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def run_model(
    model_name: str,
    prompt: str,
    max_tokens: int = 50,
    sequence_length: int = 128,
    role: str = "target",
    profile: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    logger.info(f"Loading {role} model '{model_name}' for standalone generation (sequence_length={sequence_length})...")
    model = load_model(model_name, sequence_length=sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if model is None:
        logger.error("Failed to load the model for verification.")
        return  # Exit early if model could not be loaded
    if not prompt:
        logger.error("No prompt provided for generation.")
        return

    logger.info(f"Starting generation for prompt: {prompt!r}")
    start_time = time.time() if profile else None
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    output_text = ""
    tokens_generated = 0

    for i in range(max_tokens):
        # output = model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        output = model.sample(
            input_ids,
            sequence_length=input_ids.shape[1] + 1,
            temperature=temperature,
            top_p=top_p,
        )
        if isinstance(output, (list, tuple)):
            token_id = int(output[0][-1])
        else:
            token_id = int(output[0, -1])

        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        # print(f"Token {i+1}: {repr(token_text)}", flush=True)
        output_text += token_text
        new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            logger.info("EOS token encountered, stopping generation.")
            break

    end_time = time.time()
    total_time = 0.0
    if profile:
        total_time = end_time - start_time
        throughput = tokens_generated / total_time if total_time > 0 else float('inf')
        logger.info(f"{role.capitalize()} model generation completed in {total_time:.2f} seconds.")
        logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} t/s")
        csv_file = f"performance_{role}_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        json_file = csv_file.replace(".csv", ".json")
        try:
            with open(csv_file, 'w', newline='') as f:
                f.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                avg_time = (total_time / tokens_generated) if tokens_generated > 0 else 0.0
                f.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},N/A\n")
            metrics = {
                "total_latency": total_time,
                "tokens_generated": tokens_generated,
                "throughput": throughput,
                "token_match_rate": None
            }
            with open(json_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
        except Exception as e:
            logger.error(f"Failed to save profiling data: {e}")

    full_output = prompt + output_text
    print("\n=== Final Output ===\n" + full_output)
    return full_output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verification for standalone model generation")
    parser.add_argument("--model", type=str, required=True, help="Path to the model (target or draft) for verification")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top‑p nucleus cutoff")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model inference")
    parser.add_argument("--profile", action="store_true", help="Enable total-time performance profiling")
    parser.add_argument("--role", type=str, default="target", choices=["target", "draft"],
                        help="Model role for logging (e.g. 'target' or 'draft')")
    args = parser.parse_args()
    run_model(
        args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        sequence_length=args.sequence_length,
        role=args.role,
        profile=args.profile,
        temperature=args.temperature,
        top_p=args.top_p,
    )
