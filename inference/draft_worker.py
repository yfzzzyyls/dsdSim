import logging
import grpc
import os
import time
import json
import threading
import uuid
from datetime import datetime

from grpc_comm import inference_pb2_grpc
from grpc_comm import inference_pb2
from inference.model_loader import load_model
from inference.speculative import speculative_decode
from transformers import AutoTokenizer
import torch

logger = logging.getLogger(__name__)

def save_perf_stats(perf_stats: dict, file_prefix: str):
    """Save performance stats to CSV and JSON as before."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{file_prefix}_{timestamp}.csv"
    json_path = f"{file_prefix}_{timestamp}.json"
    try:
        # CSV
        with open(csv_path, "w", newline='') as cf:
            cf.write("total_time,tokens_generated,throughput,avg_token_time,token_match_rate\n")
            total_time = perf_stats.get("total_time", 0.0)
            tokens_generated = perf_stats.get("tokens_generated", 0)
            throughput = perf_stats.get("throughput", 0.0)
            avg_token_time = perf_stats.get("avg_token_time", 0.0)
            token_match_rate = perf_stats.get("token_match_rate", None)
            line = f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_token_time:.6f},{token_match_rate}\n"
            cf.write(line)

        # JSON
        with open(json_path, "w") as jf:
            json.dump(perf_stats, jf, indent=2)
        logger.info(f"Performance metrics saved to {csv_path} and {json_path}")
    except Exception as e:
        logger.error(f"Failed to save performance data: {e}")

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
               top_p: float = 0.9,
               temperature: float = 1.0):
    """
    Single-prompt mode (preserved).
    """
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length})...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)

    tokenizer_source = target_tokenizer or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    if not prompt:
        logger.error("No prompt provided for draft client.")
        return

    if no_target:
        # STANDALONE mode (no target)
        return _run_standalone_draft(draft_model, tokenizer, prompt, max_new_tokens, profile)
    else:
        # Use gRPC to connect to the target
        address = f"{target_host}:{port}"
        logger.info(f"Connecting to target server at {address}...")
        channel = grpc.insecure_channel(address)
        stub = inference_pb2_grpc.SpeculativeServiceStub(channel)

        # StartGeneration
        session_id = _gen_session_id()
        stub.StartGeneration(
            inference_pb2.StartRequest(
                session_id=session_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                gamma=gamma
            )
        )

        logger.info(f"Starting speculative decoding (single) for prompt: \"{prompt}\"")
        generated_text, perf_stats = speculative_decode(
            draft_model, tokenizer, stub, prompt, max_new_tokens, gamma,
            profile=profile, top_p=top_p, temperature=temperature, session_id=session_id
        )

        logger.info("Speculative decoding completed.")
        full_output = prompt + generated_text
        print("\n=== Final Output ===\n" + full_output)

        # If profiling, save performance stats
        if profile and perf_stats:
            save_perf_stats(perf_stats, file_prefix="performance_speculative")

        return full_output

def _run_standalone_draft(draft_model, tokenizer, prompt, max_new_tokens, profile):
    """Helper: single prompt standalone (no target)."""
    output_text = ""
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    tokens_generated = 0
    start_time = time.time() if profile else None

    # Sample new tokens from the draft model alone
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

    end_time = time.time() if profile else None
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

def run_concurrent_clients(draft_model_name: str,
                           target_host: str = "localhost",
                           port: int = 50051,
                           prompt_text_file: str = "",
                           target_tokenizer: str = None,
                           max_new_tokens: int = 50,
                           sequence_length: int = 128,
                           gamma: int = 4,
                           profile: bool = False,
                           no_target: bool = False,
                           top_p: float = 0.9,
                           temperature: float = 1.0):
    """
    Batch/concurrent mode:
    - Reads multiple prompts from 'prompt_text_file' (one per line).
    - Spawns a thread per prompt, each connecting to the target and running the same speculative decode steps.
    - Each thread uses its own session_id so the target can maintain separate generation states.
    - Results are printed in the original prompt order.
    """
    if not os.path.exists(prompt_text_file):
        logger.error(f"Prompt text file not found: {prompt_text_file}")
        return

    with open(prompt_text_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        logger.error("No valid (non-empty) lines in the prompt file.")
        return

    # Load the draft model once
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length}) for concurrency...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)

    tokenizer_source = target_tokenizer or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    # If no_target == True, each prompt is run in standalone mode concurrently. 
    # But typically concurrency is only relevant with a target server. We'll allow it for consistency though.
    address = f"{target_host}:{port}"
    channel = None
    stub = None
    if not no_target:
        logger.info(f"Connecting to target server at {address} for concurrency...")
        channel = grpc.insecure_channel(address)
        stub = inference_pb2_grpc.SpeculativeServiceStub(channel)

    results = [None]*len(prompts)
    threads = []

    def worker(idx, prompt_text):
        # Each prompt is processed in a separate thread
        if no_target:
            # Standalone
            out = _run_standalone_draft(draft_model, tokenizer, prompt_text, max_new_tokens, profile)
            results[idx] = out
        else:
            # Create a unique session_id for each prompt so target can track separate states
            session_id = _gen_session_id()
            stub.StartGeneration(
                inference_pb2.StartRequest(
                    session_id=session_id,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    gamma=gamma
                )
            )
            logger.info(f"[Thread-{idx}] Starting speculative decoding with session_id={session_id}")

            gen_text, perf_stats = speculative_decode(
                draft_model, tokenizer, stub,
                prompt_text, max_new_tokens, gamma,
                profile=profile, top_p=top_p, temperature=temperature,
                session_id=session_id
            )
            # Combine prompt + generated
            full_output = prompt_text + gen_text
            results[idx] = full_output

            if profile and perf_stats:
                prefix = f"performance_speculative_prompt{idx}"
                save_perf_stats(perf_stats, file_prefix=prefix)

    # Spawn a thread for each prompt
    for i, prompt_text in enumerate(prompts):
        t = threading.Thread(target=worker, args=(i, prompt_text), daemon=True)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Print results in the original order
    print("\n=== Final Batched Outputs ===")
    for i, text in enumerate(results):
        print(f"\n[Prompt {i} Output]:\n{text}")

def _gen_session_id():
    """Generate a unique session ID. Could also use an int counter, or uuid."""
    # Example: integer from last 8 hex chars of uuid4
    return int(uuid.uuid4()) & 0xFFFFFFFF