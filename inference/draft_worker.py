import logging
import grpc
from grpc_comm import inference_pb2_grpc
from inference.model_loader import load_model
from inference.speculative import speculative_decode
from transformers import AutoTokenizer
import torch
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def run_client(draft_model_name: str,
               target_host: str = "localhost",
               port: int = 50051,
               prompt: str = "",
               target_tokenizer: str = None,
               max_new_tokens: int = 50,
               sequence_length: int = 128,
               draft_chunk_size: int = 4,
               profile: bool = False,
               no_target: bool = False):
    """Run the draft client process. If no_target is False, connects to target server via gRPC for speculative decoding.
       If no_target is True, runs the draft model independently (for verification or standalone generation)."""
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
        # Run draft model standalone (without target verification)
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
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                logger.info("EOS token encountered, stopping generation.")
                break
        # Log profiling info if enabled
        if profile and start_time is not None:
            total_time = time.time() - start_time
            throughput = tokens_generated / total_time if total_time > 0 else float('inf')
            logger.info(f"Draft model generation completed in {total_time:.2f} seconds.")
            logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} tokens/sec")
            # Save standalone draft performance metrics
            csv_file = f"performance_draft_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            json_file = csv_file.replace(".csv", ".json")
            try:
                with open(csv_file, 'w') as cf:
                    cf.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                    avg_time = (total_time / tokens_generated) if tokens_generated > 0 else 0.0
                    cf.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},N/A\n")
                metrics = {
                    "total_latency": total_time,
                    "tokens_generated": tokens_generated,
                    "throughput": throughput,
                    "token_match_rate": None
                }
                with open(json_file, 'w') as jf:
                    json.dump(metrics, jf, indent=2)
                logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
            except Exception as e:
                logger.error(f"Failed to write performance metrics: {e}")
        full_output = prompt + output_text
        print("\n=== Final Output ===\n" + full_output)
        return full_output

    # If a target host is specified, perform speculative decoding with the target server
    address = f"{target_host}:{port}"
    logger.info(f"Connecting to target server at {address}...")
    channel = grpc.insecure_channel(address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    logger.info(f"Starting speculative decoding for prompt: \"{prompt}\"")
    generated_text = speculative_decode(draft_model, tokenizer, stub, prompt,
                                        max_new_tokens=max_new_tokens,
                                        chunk_size=draft_chunk_size,
                                        profile=profile)
    logger.info("Speculative decoding completed.")
    full_output = prompt + generated_text
    print("\n=== Final Output ===\n" + full_output)
    return full_output
