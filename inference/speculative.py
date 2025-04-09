import time
import json
import os
import logging
import torch
from grpc_comm.inference_pb2 import StartRequest, VerifyChunkRequest  # import the correct request class
logger = logging.getLogger(__name__)

def speculative_decode(draft_model, tokenizer, target_stub, prompt_text: str,
                       max_new_tokens: int = 50, chunk_size: int = 4, profile: bool = False) -> str:
    """
    Perform distributed speculative decoding using a draft model (local) and a target model (via gRPC).
    Generates up to max_new_tokens tokens continuing from prompt_text.
    Returns the generated continuation text (excluding the prompt).
    """
    # Encode the prompt for the draft model
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids
    output_text = ""

    # Initialize target with the prompt and max token count
    start_req = StartRequest(prompt=prompt_text, max_new_tokens=max_new_tokens)
    try:
        start_resp = target_stub.StartGeneration(start_req)
        if not start_resp.acknowledged:
            logger.error("Target server failed to acknowledge StartGeneration.")
            return ""
    except Exception as e:
        logger.error(f"Failed to start generation on target server: {e}")
        return ""

    # (Optional profiling start)
    start_time = time.perf_counter() if profile else None

    tokens_generated = 0
    matched_tokens = 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

    # Generate tokens until max_new_tokens reached or generation finishes
    while tokens_generated < max_new_tokens:
        # Determine how many tokens to speculate in this iteration
        tokens_remaining = max_new_tokens - tokens_generated
        current_chunk_size = min(chunk_size, tokens_remaining)
        seq_len_before = input_ids.shape[1]

        # Draft model generates the next chunk of tokens
        try:
            draft_output = draft_model.sample(input_ids, sequence_length=seq_len_before + current_chunk_size)
        except Exception as e:
            logger.error(f"Draft model generation failed: {e}")
            break

        # Extract draft's newly generated tokens
        draft_seq = draft_output[0] if isinstance(draft_output, (list, tuple)) else draft_output[0]
        draft_new_ids = draft_seq[seq_len_before:]
        draft_new_ids = [int(t) for t in draft_new_ids]

        # Check for EOS token in draft output
        draft_finished = False
        if eos_token_id is not None and eos_token_id in draft_new_ids:
            idx = draft_new_ids.index(eos_token_id)
            draft_new_ids = draft_new_ids[:idx+1]  # include EOS in the chunk
            draft_finished = True
            current_chunk_size = len(draft_new_ids)

        # Send draft tokens to target for verification (batch verification)
        verify_req = VerifyChunkRequest(draft_tokens=draft_new_ids)
        try:
            verify_resp = target_stub.VerifyDraftChunk(verify_req)
        except Exception as e:
            logger.error(f"VerifyDraftChunk RPC failed: {e}")
            break

        # Retrieve verification results
        all_matched = verify_resp.all_matched
        match_count = verify_resp.match_count
        correct_token = verify_resp.correct_token  # target's token at divergence (0 if none)
        target_finished = verify_resp.finished

        # Append tokens to output_text based on verification result
        if all_matched:
            # All draft tokens are accepted by target
            text_to_add = tokenizer.decode(draft_new_ids, skip_special_tokens=False)
            output_text += text_to_add
            tokens_generated += len(draft_new_ids)
            matched_tokens += len(draft_new_ids)
            # Append these tokens to the draft model context for next iteration
            if draft_new_ids:
                new_tokens_tensor = torch.tensor([draft_new_ids], dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)
            logger.info(f"Draft chunk of {len(draft_new_ids)} tokens accepted (all matched).")
        else:
            # Partial match: accept the matching prefix and use target’s token for the first mismatch
            prefix_ids = draft_new_ids[:match_count]
            correct_id = correct_token  # token from target at the divergence point
            text_to_add = ""
            if prefix_ids:
                text_to_add += tokenizer.decode(prefix_ids, skip_special_tokens=False)
            if correct_id != 0:
                text_to_add += tokenizer.decode([correct_id], skip_special_tokens=False)
            output_text += text_to_add
            tokens_generated += (match_count + (1 if correct_id != 0 else 0))
            matched_tokens += match_count
            # Update draft context with accepted prefix plus the target’s correct token
            accepted_ids = prefix_ids + ([correct_id] if correct_id != 0 else [])
            if accepted_ids:
                new_tokens_tensor = torch.tensor([accepted_ids], dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)
            logger.info(f"Draft predicted {match_count} tokens correctly, then diverged. Replaced mismatch token with target's token.")

        # Determine if generation should stop
        if target_finished or draft_finished or tokens_generated >= max_new_tokens:
            logger.info(f"Generation finished (target_finished={target_finished}, draft_finished={draft_finished}, tokens_generated={tokens_generated}).")
            break
    end_time = time.perf_counter()
    # (Optional profiling end)
    if profile and start_time is not None:
        # end_time = time.perf_counter()
        total_time = end_time - start_time
        total_tokens = tokens_generated
        avg_time_per_token = total_time / total_tokens if total_tokens > 0 else 0.0
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        match_rate = matched_tokens / total_tokens if total_tokens > 0 else 0.0
        logger.info(f"Speculative decoding completed in {total_time:.2f} seconds, avg {avg_time_per_token:.4f}s per token.")
        logger.info(f"Tokens generated: {total_tokens}, Throughput: {throughput:.2f} tokens/sec, Match rate: {match_rate:.2f}")

        # Store these in a dict so the caller can use them
        stats = {
            "total_time": total_time,
            "tokens_generated": total_tokens,
            "avg_time_per_token": avg_time_per_token,
            "throughput": throughput,
            "match_rate": match_rate
        }

    return output_text, stats