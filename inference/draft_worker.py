import logging
import grpc
import time
import json
from datetime import datetime

from transformers import AutoTokenizer
import torch

from grpc_comm import inference_pb2
from grpc_comm import inference_pb2_grpc

from inference.model_loader import load_model
from inference.speculative import speculative_decode

logger = logging.getLogger(__name__)

def save_perf_stats(perf_stats: dict, file_prefix: str):
    """
    Writes performance stats to a CSV and a JSON file.
    """
    total_time = perf_stats.get("total_time", 0.0)
    tokens_generated = perf_stats.get("tokens_generated", 0)
    throughput = perf_stats.get("throughput", 0.0)
    avg_time = perf_stats.get("avg_token_time", 0.0)
    token_match_rate = perf_stats.get("token_match_rate", None)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = f"{file_prefix}_{timestamp}.csv"
    json_file = csv_file.replace(".csv", ".json")

    try:
        with open(csv_file, 'w') as cf:
            cf.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
            match_str = f"{token_match_rate:.6f}" if (token_match_rate is not None) else "N/A"
            cf.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},{match_str}\n")

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
    Run the draft client process. If no_target=False, we do speculative decoding with the target server via gRPC.
    If no_target=True, run the draft model standalone.
    """
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length})...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)

    tokenizer_source = target_tokenizer or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    if not prompt:
        logger.error("No prompt provided for draft client.")
        return

    if no_target or target_host is None:
        # Standalone mode
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

            token_id = int(output[0, -1]) if not isinstance(output, (list, tuple)) else int(output[0][-1])
            token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
            print(f"Token {i+1}: {repr(token_text)}", flush=True)
            output_text += token_text

            new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
            tokens_generated += 1

            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                logger.info("EOS token encountered, stopping generation.")
                break
        end_time = time.time()
        if profile and start_time is not None:
            total_time = end_time - start_time
            throughput = tokens_generated / total_time if total_time > 0 else float('inf')
            logger.info(f"Draft model generation completed in {total_time:.2f} seconds.")
            logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} tokens/sec")

            perf_stats = {
                "total_time": total_time,
                "tokens_generated": tokens_generated,
                "throughput": throughput,
                "avg_token_time": (total_time / tokens_generated) if tokens_generated > 0 else 0.0,
                "token_match_rate": None
            }
            save_perf_stats(perf_stats, file_prefix="performance_draft_only")

        full_output = prompt + output_text
        print("\n=== Final Output ===\n" + full_output)
        return full_output

    # Otherwise, do SPECULATIVE DECODING with target server
    address = f"{target_host}:{port}"
    logger.info(f"Connecting to target server at {address}...")
    stub = create_stub(address)

    # Start Generation
    stub.StartGeneration(
        inference_pb2.StartRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma
        )
    )

    logger.info(f"Starting speculative decoding for prompt: \"{prompt}\"")
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

    if profile and perf_stats:
        save_perf_stats(perf_stats, file_prefix="performance_speculative")

    return full_output

# -------------------------------------------------------------------------
# MERGED FROM grpc_client.py
# -------------------------------------------------------------------------

def create_stub(target_address):
    """
    Create and return a gRPC client stub for the SpeculativeService at the given address.
    """
    channel = grpc.insecure_channel(target_address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    return stub

def verify_draft_tokens(stub, draft_tokens):
    """
    Call the VerifyDraftTokens RPC with the given list of draft token IDs.
    Returns a tuple (target_probs, finished).
    """
    request = inference_pb2.VerifyRequest(draft_tokens=draft_tokens)
    response = stub.VerifyDraftTokens(request)
    target_probs = list(response.target_probs)
    finished = response.finished
    return target_probs, finished

def finalize_tokens(stub, accepted_count, draft_chunk_size):
    """
    Call FinalizeTokens RPC with:
      - accepted_count
      - draft_chunk_size
    Returns (final_token_id, finished).
    """
    request = inference_pb2.FinalizeRequest(
        accepted_count=accepted_count,
        draft_chunk_size=draft_chunk_size
    )
    response = stub.FinalizeTokens(request)
    final_token_id = response.final_token
    finished = response.finished
    return final_token_id, finished
