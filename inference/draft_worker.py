import logging
import grpc
from grpc_comm import inference_pb2_grpc
from grpc_comm import inference_pb2
from inference.model_loader import load_model
# import your speculative_decode
from inference.speculative import speculative_decode
from transformers import AutoTokenizer
import torch
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def save_perf_stats(perf_stats: dict, file_prefix: str):
    ...
    # (unchanged code)

def run_client(draft_model_name: str,
               target_host: str = "localhost",
               port: int = 50051,
               prompt: str = "",
               target_tokenizer: str = None,
               max_new_tokens: int = 50,
               sequence_length: int = 128,
               gamma: int = 4,
               profile: bool = False,
               no_target: bool = False,
               # NEW: add these two arguments
               top_p: float = 0.9,
               temperature: float = 1.0):
    """
    Run the draft client process. If no_target is False, connects to target server via gRPC for speculative decoding.
    If no_target is True, runs the draft model independently (standalone).

    :param top_p: top-p for draft sampling
    :param temperature: temperature for draft sampling
    """
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length})...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)

    tokenizer_source = target_tokenizer or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    if not prompt:
        logger.error("No prompt provided for draft client.")
        return

    if no_target or target_host is None:
        # STANDALONE mode (no speculative decoding)
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
                "token_match_rate": None,
            }
            save_perf_stats(perf_stats, file_prefix="performance_draft_only")

        full_output = prompt + output_text
        print("\n=== Final Output ===\n" + full_output)
        return full_output

    # Otherwise, connect to the target server for speculative decoding
    address = f"{target_host}:{port}"
    logger.info(f"Connecting to target server at {address}...")
    channel = grpc.insecure_channel(address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)

    # Prime the target server
    stub.StartGeneration(
        inference_pb2.StartRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma
        )
    )

    logger.info(f"Starting speculative decoding for prompt: \"{prompt}\"")
    # Now pass top_p and temperature to speculative_decode
    generated_text, perf_stats = speculative_decode(
        draft_model,
        tokenizer,
        stub,
        prompt,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        profile=profile,
        top_p=top_p,            # <-- pass top_p
        temperature=temperature # <-- pass temperature
    )

    logger.info("Speculative decoding completed.")
    full_output = prompt + generated_text
    print("\n=== Final Output ===\n" + full_output)

    # If profiling ...
    if profile and perf_stats:
        save_perf_stats(perf_stats, file_prefix="performance_speculative")

    return full_output
