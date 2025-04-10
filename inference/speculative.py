import random
import torch
import logging
from grpc_comm import grpc_client

logger = logging.getLogger(__name__)

def speculative_decode(draft_model, tokenizer, stub, max_new_tokens, gamma, profile=False, top_p=0.9):
    """
    Perform probability-based speculative decoding using a draft model and a target model via gRPC.
    draft_model: the smaller draft model (with a HuggingFace-like interface for generation)
    tokenizer: tokenizer used (shared by draft and target models)
    stub: gRPC stub for SpeculativeService connecting to the target model server
    max_new_tokens: maximum number of tokens to generate
    gamma: number of draft tokens to sample per speculative iteration (gamma)
    top_p: top-p sampling cutoff for draft model token generation
    Returns the generated text (continuation) as a string.
    """
    # Initialize output token list and tracking
    output_tokens = []
    tokens_generated = 0
    finished = False

    # Prepare initial context: the target server should have been primed with prompt via StartGeneration,
    # and the draft model can be primed by encoding the prompt if not empty.
    # Here we assume the draft_model's context is already set to the prompt (e.g., by passing input_ids).
    past = None  # for storing draft model's past state if available (for incremental generation)
    accepted_tokens_total = 0
    target_tokens_total = 0

    # Main decoding loop
    while not finished and tokens_generated < max_new_tokens:
        # 1. Draft model generates a chunk of tokens with top-p sampling
        draft_tokens = []
        draft_probs = []
        for i in range(gamma):
            # Get next token logits from draft model
            if past is None:
                # Feed the current context (output_tokens so far) to get initial logits
                if output_tokens:
                    # Use the already generated tokens as additional context
                    input_ids = torch.tensor([output_tokens], dtype=torch.long)
                else:
                    # If no prior output, just feed the last token of prompt context (handled externally)
                    input_ids = torch.tensor([[]], dtype=torch.long)  # empty tensor as placeholder
                outputs = draft_model(input_ids=input_ids, use_cache=True)
            else:
                # Use cached past state for faster generation of next token
                last_token_id = torch.tensor([[output_tokens[-1]]], dtype=torch.long) if output_tokens else None
                outputs = draft_model(input_ids=last_token_id, use_cache=True, past_key_values=past)
            logits = outputs.logits  # shape: [batch=1, seq_len=1 (for new token), vocab_size]
            past = getattr(outputs, "past_key_values", None)  # update past state if available

            # Apply softmax to get probabilities
            logits = logits[0, -1, :]  # vocab distribution for the next token
            probs = torch.softmax(logits, dim=-1)

            # Top-p filtering: select the smallest set of tokens whose cumulative probability >= top_p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            # Find cutoff index for top_p
            cutoff_index = torch.where(cumulative_probs >= top_p)[0][0].item() if torch.any(cumulative_probs >= top_p) else (len(sorted_probs) - 1)
            top_probs = sorted_probs[:cutoff_index+1]
            top_indices = sorted_indices[:cutoff_index+1]
            # Renormalize the probabilities of the top-p set
            top_probs = top_probs / torch.sum(top_probs)
            # Sample a token from the top-p filtered distribution
            choice_index = torch.multinomial(top_probs, num_samples=1).item()
            token_id = int(top_indices[choice_index].item())
            # Probability of the chosen token under the *truncated* draft distribution (Q_draft)
            token_prob = float(top_probs[choice_index].item())

            # Append draft token and its probability
            draft_tokens.append(token_id)
            draft_probs.append(token_prob)
            # Add this token to context for further generation
            output_tokens.append(token_id)
            tokens_generated += 1

            # Check for end-of-sequence token
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                # We include the EOS token in output and stop extending the draft chunk further
                break

            # If reached max_new_tokens, stop drafting further tokens
            if tokens_generated >= max_new_tokens:
                break

        # Remove any extra tokens in draft_tokens beyond max_new_tokens (if loop exited due to limit)
        if len(draft_tokens) > 0 and tokens_generated > max_new_tokens:
            # Adjust for overshoot
            overshoot = tokens_generated - max_new_tokens
            # Trim the last 'overshoot' tokens from draft_tokens and output_tokens
            draft_tokens = draft_tokens[:-overshoot]
            draft_probs = draft_probs[:-overshoot]
            output_tokens = output_tokens[:-overshoot]
            tokens_generated = max_new_tokens
            finished = True

        if not draft_tokens:
            # No draft tokens generated (should not happen unless max_new_tokens is 0 or immediate finish)
            break

        # 2. Verify draft tokens with target model to get acceptance probabilities
        target_probs, target_finished = grpc_client.verify_draft_tokens(stub, draft_tokens)
        # Ensure lengths match or truncate if target finished early
        if target_finished and len(target_probs) < len(draft_tokens):
            # Target model reached EOS before consuming all draft tokens
            # Treat remaining draft tokens as effectively rejected (target cannot continue)
            draft_tokens = draft_tokens[:len(target_probs)]
            draft_probs = draft_probs[:len(target_probs)]
            finished = True  # Target finished implies overall generation finished

        # 3. Accept or reject each draft token based on acceptance ratio
        accept_count = 0
        break_point = False
        for idx, token_id in enumerate(draft_tokens):
            # Calculate acceptance ratio = P_target(token) / P_draft(token), capped at 1
            p_target = float(target_probs[idx]) if idx < len(target_probs) else 0.0
            p_draft = float(draft_probs[idx]) if idx < len(draft_probs) else 1e-9
            ratio = p_target / p_draft if p_draft > 0 else 0.0
            if ratio > 1.0:
                ratio = 1.0  # cap the acceptance probability at 1

            # Accept or reject this token
            if random.random() < ratio:
                # Accept the draft token
                accept_count += 1
                accepted_tokens_total += 1
                # If this token is EOS, we accept and finish generation
                if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                    finished = True
                    # Accept EOS and stop further processing
                    break_point = True
                    break
                # Otherwise, continue to next token in draft_tokens
            else:
                # Reject this draft token - break at this point
                break_point = True
                break

        # If we broke out early (token idx was rejected or EOS reached), we may not use the rest of draft_tokens
        if break_point and accept_count < len(draft_tokens):
            # There are draft tokens that were not accepted (from index accept_count onward)
            # Remove unaccepted draft tokens from output (they were tentatively added)
            while len(output_tokens) > 0 and len(output_tokens) > accepted_tokens_total + target_tokens_total:
                # Remove tokens beyond the accepted ones from the output context
                output_tokens.pop()
                tokens_generated -= 1

        # 4. Finalize: update target model state and possibly get a token from target if rejection occurred
        final_token_id, finalize_finished = grpc_client.finalize_tokens(stub, accept_count)
        if final_token_id != 0:
            # A token was generated by target to replace a rejected draft token
            output_tokens.append(final_token_id)
            tokens_generated += 1
            target_tokens_total += 1
            # Check if the final token is EOS
            if tokenizer.eos_token_id is not None and final_token_id == tokenizer.eos_token_id:
                finished = True

        # If target model signaled it finished, or we've reached max tokens, end the loop
        if finalize_finished or tokens_generated >= max_new_tokens:
            finished = True

    # Convert the generated tokens (beyond the prompt) back to text
    generated_text = tokenizer.decode(output_tokens[len(output_tokens) - tokens_generated:]) if output_tokens else ""
    # Log the match rate (percentage of tokens from draft model accepted)
    total_output_tokens = accepted_tokens_total + target_tokens_total
    if total_output_tokens > 0:
        match_rate = accepted_tokens_total / total_output_tokens
        logger.info(f"Speculative decoding match rate: {match_rate:.2%} "
                    f"(Draft accepted: {accepted_tokens_total}, Target generated: {target_tokens_total})")
    return generated_text