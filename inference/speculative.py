import random
import torch
import logging

from grpc_comm import grpc_client

logger = logging.getLogger(__name__)

def speculative_decode(
    draft_model,
    tokenizer,
    stub,
    prompt,
    max_new_tokens,
    gamma,
    profile=False,
    top_p=0.9,
    temperature=1.0,
    session_id=0
):
    """
    Perform probability-based speculative decoding using a draft model and a target model via gRPC,
    with full rollback of the draft model's past states.
    Extended to handle a session_id so multiple prompts can run concurrently on the server.
    """
    # Initial setup: process prompt through draft model to initialize cache
    output_tokens = []
    draft_model.cache_ids = None
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
    if prompt_ids.shape[-1] > 0:
        _ = draft_model.forward(input_ids=prompt_ids)[0]
    prev_token = prompt_ids[0, -1] if prompt_ids.shape[-1] > 0 else None

    tokens_generated = 0
    finished = False
    accepted_tokens_total = 0
    target_tokens_total = 0

    import time
    start_t = time.time()

    while not finished and tokens_generated < max_new_tokens:
        # The draft model proposes up to 'gamma' tokens
        speculative_tokens = []
        speculative_probs = []
        past_states = [draft_model.cache_ids]
        for _ in range(gamma):
            # Generate next token using KV cache
            logits, new_cache = draft_model.forward(input_ids=prev_token.unsqueeze(0))
            # ---- Our improved numeric stability start ----
            logits = logits / temperature
            logits = torch.clamp(logits, min=-1e10, max=1e10)
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all():
                probs = torch.ones_like(probs)
                probs /= probs.sum()
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            if torch.any(cumulative_probs >= top_p):
                cutoff_index = torch.where(cumulative_probs >= top_p)[0][0].item()
            else:
                cutoff_index = len(sorted_probs) - 1
            top_probs = sorted_probs[:cutoff_index+1]
            top_indices = sorted_indices[:cutoff_index+1]
            top_sum = top_probs.sum()
            if not torch.isfinite(top_sum) or top_sum <= 1e-9:
                top_probs = torch.ones_like(top_probs)
                top_sum = top_probs.sum()
            top_probs = top_probs / top_sum
            top_probs = torch.clamp(top_probs, min=0.0, max=1.0)
            choice_index = torch.multinomial(top_probs, 1).item()
            next_token = torch.tensor([top_indices[choice_index]])
            token_id = int(next_token.item())
            token_prob = float(top_probs[choice_index].item())
            # ---- End numeric stability patch ----
            speculative_tokens.append(token_id)
            speculative_probs.append(token_prob)
            prev_token = next_token
            past_states.append(new_cache)
            # Stop if end-of-sequence or max_new_tokens reached
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                break
            if tokens_generated + len(speculative_tokens) >= max_new_tokens:
                break
        tokens_generated += len(speculative_tokens)

        # If overshoot
        if len(speculative_tokens) > 0 and tokens_generated > max_new_tokens:
            overshoot = tokens_generated - max_new_tokens
            speculative_tokens = speculative_tokens[:-overshoot]
            speculative_probs = speculative_probs[:-overshoot]
            output_tokens = output_tokens[:-overshoot]
            tokens_generated = max_new_tokens
            finished = True

        if not speculative_tokens:
            break

        # 8) Verify tokens with target model
        target_probs, target_finished = grpc_client.verify_draft_tokens(
            stub, speculative_tokens, session_id=session_id
        )
        if target_finished and len(target_probs) < len(speculative_tokens):
            # partial consumption => treat rest as rejected
            speculative_tokens = speculative_tokens[:len(target_probs)]
            speculative_probs = speculative_probs[:len(target_probs)]
            finished = True

        # 9) Accept or reject
        accept_count = 0
        break_point = False
        for idx, token_id in enumerate(speculative_tokens):
            p_target = float(target_probs[idx]) if idx < len(target_probs) else 0.0
            p_draft = float(speculative_probs[idx]) if idx < len(speculative_probs) else 1e-9
            ratio = p_target / p_draft if p_draft > 0 else 0.0
            if ratio > 1.0:
                ratio = 1.0

            if random.random() < ratio:
                accept_count += 1
                accepted_tokens_total += 1
                if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                    finished = True
                    break_point = True
                    break
            else:
                break_point = True
                break

        if break_point and accept_count < len(speculative_tokens):
            # rollback token count
            unaccepted = len(speculative_tokens) - accept_count
            while unaccepted > 0:
                output_tokens.pop()
                tokens_generated -= 1
                unaccepted -= 1
            # restore cache pointer to the last accepted state
            # Roll back draft model to last accepted token's cache state
            draft_model.cache_ids = past_states[accept_count]
            logger.info(f"[rollback] restored cache_ids = {draft_model.cache_ids}")
            # trim any saved pointers beyond this point
            past_states = past_states[:accept_count+1]

        final_token_id, finalize_finished = grpc_client.finalize_tokens(
            stub, accept_count, len(speculative_tokens), session_id=session_id
        )
        if final_token_id != 0:
            _ = draft_model.forward(input_ids=torch.tensor([[final_token_id]], dtype=torch.int32))
            output_tokens.append(final_token_id)
            tokens_generated += 1
            target_tokens_total += 1
            if tokenizer.eos_token_id is not None and final_token_id == tokenizer.eos_token_id:
                finished = True

        if finalize_finished or tokens_generated >= max_new_tokens:
            finished = True

    # Build final text
    generated_text = tokenizer.decode(output_tokens[-tokens_generated:]) if output_tokens else ""

    # Performance stats
    end_t = time.time()
    total_time = end_t - start_t
    perf_stats = {}
    if profile:
        tokens_generated_total = accepted_tokens_total + target_tokens_total
        throughput = tokens_generated_total / total_time if total_time>0 else 0.0
        perf_stats["total_time"] = total_time
        perf_stats["tokens_generated"] = tokens_generated_total
        perf_stats["throughput"] = throughput
        perf_stats["avg_token_time"] = total_time / tokens_generated_total if tokens_generated_total>0 else 0.0

    total_output_tokens = accepted_tokens_total + target_tokens_total
    if total_output_tokens > 0:
        match_rate = accepted_tokens_total / total_output_tokens
        logger.info(f"Latency: {total_time:.2f} seconds")
        logger.info(f"Speculative decoding match rate: {match_rate:.2%} (Draft accepted: {accepted_tokens_total}, Target generated: {target_tokens_total})")
        perf_stats["token_match_rate"] = match_rate

    return generated_text, perf_stats