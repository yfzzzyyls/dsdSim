import random
import torch
import logging
import collections

from grpc_comm import grpc_client

logger = logging.getLogger(__name__)

# Repetition‑penalty strength (0 < α ≤ 1).  Smaller → stronger penalty
REP_PENALTY = 0.4
NGRAM_WINDOW = 3    # penalise 1‑ to 3‑gram repeats
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
    valid_gammas = (1, 2, 4, 8)   # must match compiled buckets on target
    # snap initial γ to the largest compiled bucket ≤ user request
    current_gamma = max(g for g in valid_gammas if g <= max(1, gamma))
    gamma_max     = 8                      # hard ceiling
    current_temp  = temperature            # draft temperature we can tweak
    target_accept = 0.7                    # desired per‑loop acceptance rate

    logger.debug(
        f"[session={session_id}] Starting speculative_decode: "
        f"prompt='{prompt[:60]}...' max_new_tokens={max_new_tokens} gamma={gamma}"
    )
    # Initial setup: process prompt through draft model to initialize cache
    output_tokens = []
    draft_model.cache_ids = None
    draft_model._next_pos = 0  # next position index in the KV cache

    # pre-filling: Feed the entire prompt once so the draft model builds its KV cache
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
    prev_token_id = int(prompt_ids[0, -1].item()) if prompt_ids.shape[-1] > 0 else tokenizer.bos_token_id
    
    # Feed the prompt so Neuron caches 0…L‑1, then set pointer to NEXT index (=L)
    if prompt_ids.shape[-1] > 0:
       # build the KV cache for the prompt
       _ = draft_model.forward(input_ids=prompt_ids)          # fills 0…L‑1
       prompt_len = prompt_ids.shape[-1]
       # Overwrite cache pointer with a single‑index tensor [L]
       draft_model.cache_ids = torch.tensor([prompt_len], dtype=torch.int32)
       draft_model._next_pos = prompt_len
    else:
       # no prompt given
       prompt_len = 0
       draft_model.cache_ids = torch.tensor([0], dtype=torch.int32)
       draft_model._next_pos = 0

    tokens_generated = 0
    # reusable scratch tensor (1,1) for single-token forwards
    scratch_token = torch.empty((1, 1), dtype=torch.int64)
    # fixed-size deque for fast repetition penalty history
    recent_deque  = collections.deque(maxlen=50)
    finished = False
    accepted_tokens_total = 0
    target_tokens_total = 0

    import time
    # detailed timing metrics
    timing = {
        "draft_forward_time":       0.0,   # local draft forwards
        "grpc_server_time":         0.0,   # Verify + Finalize wait
        "target_verification_time": 0.0,   # placeholder
        "rollback_time":            0.0,
    }
    start_t = time.time()

    while not finished and tokens_generated < max_new_tokens:
        # The draft model proposes up to 'gamma' tokens
        speculative_tokens = []
        speculative_probs = []
        logger.debug("[session=%s] Entering inner loop, tokens_generated=%d", session_id, tokens_generated)
        past_states = [draft_model.cache_ids]
        for _ in range(current_gamma):
            scratch_token[0, 0] = prev_token_id
            if profile:
                _t0 = time.perf_counter()
            logits, _ = draft_model.forward(input_ids=scratch_token)
            if profile:
                timing["draft_forward_time"] += time.perf_counter() - _t0
            logits = logits.float()
            # ---- Our improved numeric stability start ----
            # Temperature‑scale logits then apply classic nucleus (top‑p) filter
            logits = logits / current_temp
            probs = torch.softmax(logits, dim=-1)
 
            # ---------- nucleus filter (fast top‑k) ----------
            k = min(512, probs.shape[-1])          # limit sort to top‑512
            top_vals, top_idx = torch.topk(probs, k)         # O(V) → O(k log k)
            cum_p = torch.cumsum(top_vals, dim=0)
            cut = torch.searchsorted(cum_p, top_p, right=True).item()
            nucleus_idx   = top_idx[:cut + 1]
            nucleus_probs = top_vals[:cut + 1]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()


            # ---------- repetition penalty (1‑ to NGRAM_WINDOW‑gram) ----------
            # if recent_deque:
            #     # Build a flat list of recent output + current draft chunk
            #     recent_ids = list(recent_deque) + speculative_tokens
            #     # Penalise 1‑gram repeats first
            #     recent1 = torch.tensor(recent_ids, device=nucleus_idx.device)
            #     mask1   = (nucleus_idx.unsqueeze(1) == recent1).any(dim=1)

            #     # Optional higher‑order n‑gram penalty (up to NGRAM_WINDOW)
            #     mask_ngram = mask1.clone()
            #     if len(recent_ids) >= 2 and NGRAM_WINDOW >= 2:
            #         for n in range(2, NGRAM_WINDOW + 1):
            #             if len(recent_ids) < n:
            #                 break
            #             tail = recent_ids[-(n-1):]                # last n‑1 tokens
            #             # create tensor [tail + candidate] for each nucleus candidate
            #             cand = torch.cat([
            #                 torch.tensor(tail, device=nucleus_idx.device).repeat(nucleus_idx.size(0), 1),
            #                 nucleus_idx.unsqueeze(1)
            #             ], dim=1)
            #             # search n‑gram occurrences in recent_ids as sliding window
            #             recent_ng = torch.tensor(recent_ids, device=nucleus_idx.device)
            #             windows = recent_ng.unfold(0, n, 1)       # shape (L-n+1, n)
            #             match = (cand.unsqueeze(1) == windows).all(dim=2).any(dim=1)
            #             mask_ngram |= match

            #     if mask_ngram.any():
            #         nucleus_probs = torch.where(mask_ngram, nucleus_probs * REP_PENALTY, nucleus_probs)
            #         nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            # sample a token from the renormalised nucleus
            sample_idx = torch.multinomial(nucleus_probs, 1).item()
            token_id = int(nucleus_idx[sample_idx].item())
 
            token_prob = float(nucleus_probs[sample_idx].item())  # probability under q_draft
            # ---- End numeric stability patch ----

            # store the token and its probability for later verification
            speculative_tokens.append(token_id)
            speculative_probs.append(token_prob)
            
            prev_token_id = token_id
            past_states.append(draft_model.cache_ids.clone())   # pointer to next slot
            # Stop if end-of-sequence or max_new_tokens reached
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                break
            if tokens_generated + len(speculative_tokens) >= max_new_tokens:
                break
 
        logger.debug("[session=%s] Proposed tokens: %s", session_id, speculative_tokens)

        # # If overshoot
        # if len(speculative_tokens) > 0 and tokens_generated > max_new_tokens:
        #     overshoot = tokens_generated - max_new_tokens
        #     speculative_tokens = speculative_tokens[:-overshoot]
        #     speculative_probs = speculative_probs[:-overshoot]
        #     output_tokens = output_tokens[:-overshoot]
        #     tokens_generated = max_new_tokens
        #     finished = True

        if not speculative_tokens:
            break

        # 8) Verify tokens with target model
        # --------------------------------------------------------------
        # Ensure the speculative chunk length is one of the compiled
        # buckets {1, 2, 4, 8}.  If we broke early (e.g. hit EOS or
        # max_new_tokens) we may have a length like 3 or 5, which the
        # target model cannot verify in a single pass.  Truncate to the
        # largest supported γ ≤ current length.
        # --------------------------------------------------------------
        if len(speculative_tokens) not in valid_gammas:
            allowed_len = max(g for g in valid_gammas if g < len(speculative_tokens))
            speculative_tokens = speculative_tokens[:allowed_len]
            speculative_probs  = speculative_probs[:allowed_len]
            past_states        = past_states[:allowed_len + 1]  # keep matching ptrs
        # --- Verify + commit in one RPC ---
        commit_ids, accepted_count, target_finished = grpc_client.verify_draft_tokens(
            stub, speculative_tokens, speculative_probs, session_id=session_id
        )
        accepted_tokens_total += accepted_count
        target_tokens_total   += len(commit_ids) - accepted_count

        # --------------------------------------------------------------
        # ROLLBACK draft KV pointer to the state *before* the first
        # rejected token so we don't "leak" unused cache slots.
        # past_states[0] is the pointer *before* emitting any speculative
        # token; past_states[k] is after accepting k tokens.
        # --------------------------------------------------------------
        rollback_ptr = past_states[accepted_count].clone()
        draft_model.cache_ids = rollback_ptr
        draft_model._next_pos = int(rollback_ptr.item())

        # Feed committed tokens back into the draft model (we have already
        # rolled back the KV pointer to the correct slot). 
        for tok in commit_ids:
            scratch_token[0, 0] = tok
            if profile:
                _t0 = time.perf_counter()
            _, _ = draft_model.forward(input_ids=scratch_token)
            if profile:
                timing["draft_forward_time"] += time.perf_counter() - _t0
            draft_model.cache_ids = torch.tensor([draft_model._next_pos], dtype=torch.int32)
            prev_token_id = tok
            output_tokens.append(tok)
            recent_deque.append(tok)
            tokens_generated += 1
            if tokenizer.eos_token_id is not None and tok == tokenizer.eos_token_id:
                finished = True

        # Propagate server‑side finished flag
        finished = finished or target_finished

        # ---------- adaptive γ and temperature (P‑controller) ----------
        if current_gamma > 0:
            loop_accept_rate = accepted_count / current_gamma
            error = target_accept - loop_accept_rate

            # PID suggestion
            desired_gamma = int(max(1, min(gamma_max,
                                           current_gamma + 0.5 * error * current_gamma)))
            # snap to nearest compiled bucket _not exceeding_ desired_gamma
            new_gamma = max(g for g in valid_gammas if g <= desired_gamma)
            if new_gamma != current_gamma:
                logger.debug("[session=%s] Adjust γ %d → %d (acc_rate=%.2f, desired=%d)",
                             session_id, current_gamma, new_gamma, loop_accept_rate, desired_gamma)
            current_gamma = new_gamma

            new_temp = max(0.3, min(2.0, current_temp * (1 + 0.2 * error)))
            if abs(new_temp - current_temp) > 1e-3:
                logger.debug("[session=%s] Adjust draft temperature %.3f → %.3f",
                             session_id, current_temp, new_temp)
            current_temp = new_temp

        if target_finished or tokens_generated >= max_new_tokens:
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
        perf_stats.update({
            "draft_forward_time":       timing["draft_forward_time"],
            "grpc_server_time":         timing["grpc_server_time"],
            "target_verification_time": timing["target_verification_time"],
            "rollback_time":            timing["rollback_time"],
        })

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