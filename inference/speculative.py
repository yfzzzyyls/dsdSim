import logging
import time
import json
import os
import torch

from grpc_comm.inference_pb2 import StartRequest, VerifyRequest

logger = logging.getLogger(__name__)

def speculative_decode(draft_model, tokenizer, target_stub, prompt_text: str,
                       max_new_tokens: int = 50, chunk_size: int = 4, profile: bool = False) -> str:
    """
    Perform distributed speculative decoding using a draft model (local) and a target model (via gRPC).
    Generates up to max_new_tokens from the prompt_text.
    Returns the generated text (excluding the prompt).
    """
    # Encode the prompt into input IDs for the draft model
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids
    output_text = ""

    # Initialize the target server with the prompt and maximum token count
    start_req = StartRequest(prompt=prompt_text, max_new_tokens=max_new_tokens)
    try:
        start_resp = target_stub.StartGeneration(start_req)
        if not start_resp.acknowledged:
            logger.error("Target server failed to acknowledge StartGeneration.")
            return ""
    except Exception as e:
        logger.error(f"Failed to start generation on target server: {e}")
        return ""

    # Profiling setup
    start_time = time.perf_counter() if profile else None
    tokens_generated = 0
    matched_tokens = 0

    # ID of the EOS token (if any) for detection of termination
    eos_token_id = None
    if tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id

    # Loop until we've generated the desired number of tokens or target signals finish
    while tokens_generated < max_new_tokens:
        # Determine how many tokens to draft in this iteration
        tokens_remaining = max_new_tokens - tokens_generated
        current_chunk_size = min(chunk_size, tokens_remaining)

        # Draft model generates a chunk of tokens
        # Use the model's sample method to extend the sequence by current_chunk_size tokens
        seq_len_before = input_ids.shape[1]
        try:
            draft_output = draft_model.sample(input_ids, sequence_length=seq_len_before + current_chunk_size)
        except Exception as e:
            logger.error(f"Draft model generation failed: {e}")
            break

        # Extract the newly generated draft tokens
        if isinstance(draft_output, (list, tuple)):
            draft_seq = draft_output[0]  # assume first element is sequence output
        else:
            draft_seq = draft_output[0]
        # draft_seq length should be seq_len_before + current_chunk_size (including prompt/context + new tokens)
        draft_new_ids = draft_seq[seq_len_before:]
        draft_new_ids = [int(t) for t in draft_new_ids]

        # Check if draft model predicted an EOS token in this chunk
        draft_finished = False
        if eos_token_id is not None and eos_token_id in draft_new_ids:
            idx = draft_new_ids.index(eos_token_id)
            # Truncate any tokens after EOS, include EOS itself as draft's prediction
            draft_new_ids = draft_new_ids[:idx+1]
            draft_finished = True
            current_chunk_size = len(draft_new_ids)  # adjust chunk size to actual generated before EOS

        # Send the draft tokens to the target for verification
        verify_req = VerifyRequest(draft_tokens=draft_new_ids)
        try:
            verify_resp = target_stub.VerifyDraftTokens(verify_req)
        except Exception as e:
            logger.error(f"VerifyDraftTokens RPC failed: {e}")
            break

        all_matched = verify_resp.all_matched
        match_count = verify_resp.match_count
        correct_token = verify_resp.correct_token  # 0 if not applicable
        target_finished = verify_resp.finished

        # Append tokens to output_text based on verification result
        if all_matched:
            # All draft tokens are correct
            text_to_add = tokenizer.decode(draft_new_ids, skip_special_tokens=False)
            output_text += text_to_add
            tokens_generated += len(draft_new_ids)
            matched_tokens += len(draft_new_ids)
            # Update input_ids context for draft model (append these tokens)
            if draft_new_ids:
                new_tokens_tensor = torch.tensor([draft_new_ids], dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)
            logger.info(f"Draft chunk of {len(draft_new_ids)} tokens accepted (all matched).")
        else:
            # Partial match - accept prefix and the target's correct token at mismatch
            prefix_ids = draft_new_ids[:match_count]
            correct_id = correct_token  # target's token at first mismatch position
            # Decode and append matched prefix (if any) and the correct token from target
            text_to_add = ""
            if prefix_ids:
                text_to_add += tokenizer.decode(prefix_ids, skip_special_tokens=False)
            if correct_id != 0:
                text_to_add += tokenizer.decode([correct_id], skip_special_tokens=False)
            output_text += text_to_add
            # Update counts: prefix tokens were correctly predicted by draft
            tokens_generated += (match_count + 1)  # accepted tokens = matched prefix + one correct token
            matched_tokens += match_count  # only the prefix were draft matches
            # Update draft model context: use target's correct token at mismatch
            accepted_ids = prefix_ids + ([correct_id] if correct_id != 0 else [])
            if accepted_ids:
                new_tokens_tensor = torch.tensor([accepted_ids], dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)
            logger.info(f"Draft predicted {match_count} tokens correctly, then diverged. Replaced token at mismatch with target's token.")
        # If target or draft signaled completion, or we've reached max tokens, stop
        if target_finished or draft_finished or tokens_generated >= max_new_tokens:
            logger.info("Generation finished (target_finished=%s, draft_finished=%s, tokens_generated=%d).", 
                        target_finished, draft_finished, tokens_generated)
            break
        # Continue loop for next chunk if not finished
    # End of generation loop

    # Profiling: calculate and log metrics if enabled
    if profile and start_time is not None:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_tokens = tokens_generated
        # Avoid division by zero
        avg_time_per_token = total_time / total_tokens if total_tokens > 0 else 0.0
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        match_rate = matched_tokens / total_tokens if total_tokens > 0 else 0.0

        # Log the metrics
        logger.info(f"End-to-end latency: {total_time:.3f} seconds for {total_tokens} tokens")
        logger.info(f"Per-token generation time: {avg_time_per_token*1000:.1f} ms/token")
        logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"Token match rate: {match_rate*100:.1f}% ({matched_tokens}/{total_tokens} tokens from draft model)")

        # Save metrics to CSV and JSON files
        try:
            # Prepare data
            metrics = {
                "prompt": prompt_text,
                "total_tokens": total_tokens,
                "chunk_size": chunk_size,
                "total_time_sec": round(total_time, 4),
                "avg_time_per_token_sec": round(avg_time_per_token, 6),
                "throughput_tokens_per_sec": round(throughput, 4),
                "match_rate": round(match_rate, 4)
            }
            # Write/update CSV
            csv_header = "prompt,total_tokens,chunk_size,total_time_sec,avg_time_per_token_sec,throughput_tokens_per_sec,match_rate"
            csv_line = f"\"{prompt_text}\",{total_tokens},{chunk_size},{metrics['total_time_sec']},{metrics['avg_time_per_token_sec']},{metrics['throughput_tokens_per_sec']},{metrics['match_rate']}\n"
            csv_file = "profile.csv"
            write_header = not os.path.isfile(csv_file) or os.path.getsize(csv_file) == 0
            with open(csv_file, "a") as cf:
                if write_header:
                    cf.write(csv_header + "\n")
                cf.write(csv_line)
            # Write JSON (overwrite or create new)
            with open("profile.json", "w") as jf:
                json.dump(metrics, jf, indent=4)
        except Exception as e:
            logger.error(f"Failed to write profile data to file: {e}")
    return output_text
