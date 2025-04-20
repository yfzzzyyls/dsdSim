import random
import torch
import logging

from grpc_comm import grpc_client

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

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
    logger.debug(
        f"[session={session_id}] Starting speculative_decode: "
        f"prompt='{prompt[:60]}...' max_new_tokens={max_new_tokens} gamma={gamma}"
    )
    # Initial setup: process prompt through draft model to initialize cache
    output_tokens = []
    draft_model.cache_ids = None
    draft_model._next_pos = 0  # next position index in the KV cache
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
    prev_token_id = int(prompt_ids[0, -1].item()) if prompt_ids.shape[-1] > 0 else tokenizer.bos_token_id
    # Feed the entire prompt once so the draft model builds its KV cache
    # Feed the prompt so Neuron caches 0…L‑1, then set pointer to NEXT index (=L)
    if prompt_ids.shape[-1] > 0:
       _ = draft_model.forward(input_ids=prompt_ids)          # fills 0…L‑1
       prompt_len = prompt_ids.shape[-1]
       # Overwrite cache pointer with a single‑index tensor [L]
       draft_model.cache_ids = torch.tensor([prompt_len], dtype=torch.int32)
       draft_model._next_pos = prompt_len
    else:
       prompt_len = 0
       draft_model.cache_ids = torch.tensor([0], dtype=torch.int32)
       draft_model._next_pos = 0

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
        logger.debug("[session=%s] Entering inner loop, tokens_generated=%d", session_id, tokens_generated)
        past_states = [draft_model.cache_ids]
        for _ in range(gamma):
            # Prepare a 2‑D input_ids tensor [[token_id]]
            input_ids = torch.tensor([[prev_token_id]], dtype=torch.int64)
            logits, new_cache = draft_model.forward(input_ids=input_ids)
            # ---- Our improved numeric stability start ----
            # Temperature‑scale logits then apply classic nucleus (top‑p) filter
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
 
                # ---------- nucleus filter (fast top‑k) ----------
                k = min(512, probs.shape[-1])          # limit sort to top‑512
                top_vals, top_idx = torch.topk(probs, k)         # O(V) → O(k log k)
                cum_p = torch.cumsum(top_vals, dim=0)
                cut = torch.searchsorted(cum_p, top_p, right=True).item()
                nucleus_idx   = top_idx[:cut + 1]
                nucleus_probs = top_vals[:cut + 1]
                nucleus_probs = nucleus_probs / nucleus_probs.sum()

                # ---------- vectorised repetition penalty ----------
                recent = output_tokens[-50:] + speculative_tokens
                if recent:
                    recent_t = torch.tensor(recent, device=nucleus_idx.device)
                    mask = (nucleus_idx.unsqueeze(1) == recent_t).any(dim=1)
                    if mask.any():
                        nucleus_probs = torch.where(mask, nucleus_probs * 0.4, nucleus_probs)
                        nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            # sample a token from the renormalised nucleus
            sample_idx = torch.multinomial(nucleus_probs, 1).item()
            token_id = int(nucleus_idx[sample_idx].item())
 
            token_prob = float(nucleus_probs[sample_idx].item())  # probability under q_draft
            # ---- End numeric stability patch ----
            speculative_tokens.append(token_id)
            speculative_probs.append(token_prob)
            
            prev_token_id = token_id
            # past_states.append(new_cache)
            past_states.append(draft_model.cache_ids.clone())   # pointer to next slot
            # Stop if end-of-sequence or max_new_tokens reached
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                break
            if tokens_generated + len(speculative_tokens) >= max_new_tokens:
                break
 
        logger.debug("[session=%s] Proposed tokens: %s", session_id, speculative_tokens)

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
        logger.debug(
            f"[session={session_id}] Target probs len={len(target_probs)}, finished={target_finished}"
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
        logger.debug(
            f"[session={session_id}] accept_count={accept_count} break_point={break_point}"
        )

        # ----------------------------------------------------------
        # After accept/reject: commit accepted draft tokens
        # ----------------------------------------------------------
        if accept_count > 0:
            output_tokens.extend(speculative_tokens[:accept_count])
            tokens_generated += accept_count
            # advance prev_token_id so the next chunk starts correctly
            prev_token_id = speculative_tokens[accept_count - 1]
            logger.debug(
                f"[session={session_id}] Committed tokens → "
                f"tokens_generated={tokens_generated} prev_token_id={prev_token_id}"
            )

        if break_point and accept_count < len(speculative_tokens):
            # The unaccepted tokens were *not* appended to output_tokens,
            # so we should NOT pop anything here.  Simply rewind the draft
            # model’s KV pointer.
            draft_model.cache_ids = past_states[accept_count].clone()
            if hasattr(draft_model, "_next_pos"):
                draft_model._next_pos = int(draft_model.cache_ids.item())
            # sanity: cursor and cache_ids agree after rollback
            assert int(draft_model.cache_ids.item()) == draft_model._next_pos, \
                "Draft KV pointer mismatch after rollback"
            prev_token_id = output_tokens[-1] if output_tokens else prompt_ids[0, -1].item()
            logger.debug(f"[session={session_id}] Rollback: unaccepted={len(speculative_tokens) - accept_count}, cache_ids_restored={draft_model.cache_ids.tolist()}")
            past_states = past_states[:accept_count+1]

        final_token_id, finalize_finished = grpc_client.finalize_tokens(
            stub, accept_count, len(speculative_tokens), session_id=session_id
        )
        if final_token_id != 0:
            # Always commit the fallback / extra target token
            _, _ = draft_model.forward(
                input_ids=torch.tensor([[final_token_id]], dtype=torch.int64)
            )
            draft_model.cache_ids = torch.tensor([draft_model._next_pos], dtype=torch.int32)
            # sanity: cursor and cache_ids agree
            assert int(draft_model.cache_ids.item()) == draft_model._next_pos, \
                "Draft KV pointer mismatch after committing target token"
            prev_token_id = final_token_id
            output_tokens.append(final_token_id)
            tokens_generated += 1
            target_tokens_total += 1
            logger.debug(
                f"[session={session_id}] Target token committed: {final_token_id}"
            )
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

    logger.debug(
        f"[session={session_id}] Finished: generated_text='{generated_text[:120]}...'"
    )
    # Make these counters available to callers even when profiling is off
    perf_stats["accepted_tokens_total"] = accepted_tokens_total
    perf_stats["target_tokens_total"] = target_tokens_total
    return generated_text, perf_stats