import logging
import grpc
from grpc_comm import inference_pb2_grpc
from grpc_comm import inference_pb2
from inference.model_loader import load_model
from inference.speculative import speculative_decode
from transformers import AutoTokenizer
import torch
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def save_perf_stats(perf_stats: dict, file_prefix: str):
    """
    Writes performance stats to a CSV and a JSON file.

    :param perf_stats: Dictionary with keys like 
                       ["total_time", "tokens_generated", "throughput", "avg_token_time", "token_match_rate"].
    :param file_prefix: Prefix for the CSV/JSON filenames (e.g. "performance_draft_only").
    """
    # Extract stats with defaults to avoid KeyErrors
    total_time = perf_stats.get("total_time", 0.0)
    tokens_generated = perf_stats.get("tokens_generated", 0)
    throughput = perf_stats.get("throughput", 0.0)
    avg_time = perf_stats.get("avg_token_time", 0.0)
    token_match_rate = perf_stats.get("token_match_rate", None)

    # Build file names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = f"{file_prefix}_{timestamp}.csv"
    json_file = csv_file.replace(".csv", ".json")

    try:
        with open(csv_file, 'w') as cf:
            cf.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
            # Convert None match_rate to a string (e.g. "N/A")
            match_str = f"{token_match_rate:.6f}" if (token_match_rate is not None) else "N/A"
            cf.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},{match_str}\n")

        # Write JSON
        metrics = {
            "total_latency": total_time,
            "tokens_generated": tokens_generated,
            "throughput": throughput,
            "avg_token_time": avg_time,
            "token_match_rate": token_match_rate,
        }
        with open(json_file, 'w') as jf:
            json.dump(metrics, jf, indent=2)

        logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
    except Exception as e:
        logger.error(f"Failed to write performance metrics: {e}")


def run_client(draft_model_name: str,
               target_host: str = "localhost",
               port: int = 50051,
               prompt: str = "",
               target_tokenizer: str = None,
               max_new_tokens: int = 50,
               sequence_length: int = 128,
               gamma: int = 4,
               profile: bool = False,
               no_target: bool = False):
    """
    Run the draft client process. If no_target is False, connects to target server via gRPC for speculative decoding.
    If no_target is True, runs the draft model independently (for verification or standalone generation).
    """
    # Load or compile the draft model
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length})...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)

    # Use the target model's tokenizer if provided, otherwise the draft's
    tokenizer_source = target_tokenizer or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    if not prompt:
        logger.error("No prompt provided for draft client.")
        return

    if no_target or target_host is None:
        # Run DRAFT MODEL in standalone mode
        output_text = ""
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        tokens_generated = 0
        start_time = time.time() if profile else None

        for i in range(max_new_tokens):
            try:
                output = draft_model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
            except Exception as e:
                logger.error(f"Draft model generation failed: {e}")
                break

            # Get the newly generated token ID
            token_id = int(output[0, -1]) if not isinstance(output, (list, tuple)) else int(output[0][-1])
            token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
            print(f"Token {i+1}: {repr(token_text)}", flush=True)
            output_text += token_text

            # Append new token to input_ids for next iteration
            new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
            tokens_generated += 1

            # Check for EOS token
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                logger.info("EOS token encountered, stopping generation.")
                break
        end_time = time.time()
        # Log & save profiling info if needed
        if profile and start_time is not None:
            total_time = end_time - start_time
            throughput = tokens_generated / total_time if total_time > 0 else float('inf')
            logger.info(f"Draft model generation completed in {total_time:.2f} seconds.")
            logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} tokens/sec")

            # Build perf_stats dict for standalone
            perf_stats = {
                "total_time": total_time,
                "tokens_generated": tokens_generated,
                "throughput": throughput,
                "avg_token_time": (total_time / tokens_generated) if tokens_generated > 0 else 0.0,
                "token_match_rate": None,  # Not applicable in standalone mode
            }
            # Save to CSV/JSON
            save_perf_stats(perf_stats, file_prefix="performance_draft_only")

        full_output = prompt + output_text
        print("\n=== Final Output ===\n" + full_output)
        return full_output

    # Otherwise, we do SPECULATIVE DECODING with the target server
    address = f"{target_host}:{port}"
    logger.info(f"Connecting to target server at {address}...")
    channel = grpc.insecure_channel(address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)

    # CRITICAL FIX: Prime the target server with prompt & max_new_tokens
    stub.StartGeneration(
        inference_pb2.StartRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma
        )
    )

    logger.info(f"Starting speculative decoding for prompt: \"{prompt}\"")
    # Speculative_decode to return (generated_text, perf_stats)
    generated_text, perf_stats = speculative_decode(
        draft_model,
        tokenizer,
        stub,
        prompt,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        profile=profile
    )

    logger.info("Speculative decoding completed.")
    full_output = prompt + generated_text
    print("\n=== Final Output ===\n" + full_output)

    # If profiling is enabled, save performance stats
    if profile and perf_stats:
        save_perf_stats(perf_stats, file_prefix="performance_speculative")

    return full_output
