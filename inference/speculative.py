import random
import torch
import logging
import collections

from transformers_neuronx import sampling
from grpc_comm import grpc_client

logger = logging.getLogger(__name__)

# Repetition‑penalty strength (0 < α ≤ 1).  Smaller → stronger penalty
REP_PENALTY = 0.4
NGRAM_WINDOW = 3    # penalise 1‑ to 3‑gram repeats
TOP_K = 512

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
    valid_gammas = (1, 2, 3, 4, 5)   # must match compiled buckets on target
    # snap initial γ to the largest compiled bucket ≤ user request
    current_gamma = max(g for g in valid_gammas if g <= max(1, gamma))
    gamma_max     = 4                      # hard ceiling
    current_temp  = temperature            # draft temperature we can tweak
    target_accept = 0.5                    # desired per‑loop acceptance rate

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
        # ---------- just before the debug call, print he actual word generated ----------
        token_texts = [tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                    for tid in speculative_tokens]

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

            # apply ngram filter
            masked = sampling.filter_ngrams(
                NGRAM_WINDOW,
                torch.tensor(output_tokens + speculative_tokens).unsqueeze(0),
                logits.unsqueeze(0),    # (1, V) expected
                draft_model._next_pos
            )

            # apply temperature scaling
            logits = logits / current_temp

            # apply top‑k and top‑p filtering
            masked, candidate_idx = sampling.top_k_top_p_filtering(
                logits.unsqueeze(0),             # (1, V) expected
                top_k=TOP_K,
                top_p=top_p
            )

            probs = torch.softmax(masked, dim=-1).squeeze(0)
            sample_in_topk = torch.multinomial(probs, 1).item()
            token_id   = int(candidate_idx[0, sample_in_topk])
            token_prob = float(probs[sample_in_topk])
 
            # store the token and its probability for later verification
            speculative_tokens.append(token_id)
            speculative_probs.append(token_prob)
            
            prev_token_id = token_id
            past_states.append(draft_model.cache_ids.clone())   # save pointer to next slot
            # Stop if end-of-sequence or max_new_tokens reached
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                break
            if tokens_generated + len(speculative_tokens) >= max_new_tokens:
                break
 
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
        # Fast‑fail if the speculative chunk length is NOT one of the
        # compiled γ buckets.  We no longer truncate; instead we raise to
        # surface a configuration / logic error immediately.
        # --------------------------------------------------------------
        if len(speculative_tokens) not in valid_gammas:
            assert len(speculative_tokens) in valid_gammas, (
                f"Speculative chunk length {len(speculative_tokens)} is not "
                f"supported by the compiled target model; valid γ buckets = "
                f"{valid_gammas}"
            )
        # ------------------------------------------------------------------
        # Ensure that speculative_probs and past_states are consistent with
        # speculative_tokens length.  This can become mismatched when we
        # break early from the inner‑token loop (e.g. EOS or max_new_tokens).
        # ------------------------------------------------------------------
        if len(speculative_tokens) != len(speculative_probs):
            speculative_probs  = speculative_probs[:len(speculative_tokens)]
            past_states        = past_states[:len(speculative_tokens) + 1]
        
        # --- Verify + commit in one RPC ---
                # logger.debug("[session=%s] Proposed tokens: %s", session_id, speculative_tokens)
        # --- show draft chunk as words instead of IDs -----------------
        token_texts_dbg = [
            tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            for tid in speculative_tokens
        ]
        logger.debug("[session=%s] draft model proposed: chunk len=%d, proposed tokens (text)=%s, ids=%s, probs=%s", session_id, len(speculative_tokens), token_texts_dbg, speculative_tokens, speculative_probs)
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
            # write at the true position
            scratch_token[0, 0] = tok             # ← add this
            cache_id_tensor = torch.tensor([draft_model._next_pos], dtype=torch.int32)
            _ = draft_model.forward(input_ids=scratch_token, cache_ids=cache_id_tensor)
            draft_model.cache_ids = torch.tensor([draft_model._next_pos], dtype=torch.int32)
            prev_token_id = tok
            output_tokens.append(tok)
            recent_deque.append(tok)
            tokens_generated += 1
            if tokenizer.eos_token_id is not None and tok == tokenizer.eos_token_id:
                finished = True

        logger.debug("ACCEPT cnt=%d  committed=%s",
             accepted_count,
             speculative_tokens[:accepted_count])
        # Propagate server‑side finished flag
        finished = finished or target_finished

        # ---------- adaptive γ and temperature (P‑controller) ----------
        # if current_gamma > 0:
        #     loop_accept_rate = accepted_count / current_gamma
        #     error = target_accept - loop_accept_rate

        #     # PID suggestion
        #     desired_gamma = int(max(1, min(gamma_max,
        #                                    current_gamma + 0.5 * error * current_gamma)))
        #     # snap to nearest compiled bucket _not exceeding_ desired_gamma
        #     new_gamma = max(g for g in valid_gammas if g <= desired_gamma)
        #     if new_gamma != current_gamma:
        #         logger.debug("[session=%s] Adjust γ %d → %d (acc_rate=%.2f, desired=%d)",
        #                      session_id, current_gamma, new_gamma, loop_accept_rate, desired_gamma)
        #     current_gamma = new_gamma

        #     new_temp = max(0.3, min(2.0, current_temp * (1 + 0.2 * error)))
        #     if abs(new_temp - current_temp) > 1e-3:
        #         logger.debug("[session=%s] Adjust draft temperature %.3f → %.3f",
        #                      session_id, current_temp, new_temp)
        #     current_temp = new_temp

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
        logger.info(f"Speculative decoding match rate: {match_rate:.2%} (Draft accepted: {accepted_tokens_total}, Target generated: {target_tokens_total})\n")
        perf_stats["token_match_rate"] = match_rate

    logger.debug(
        f"[session={session_id}] Finished: generated_text='{generated_text[:120]}...'"
    )
    # Make these counters available to callers even when profiling is off
    perf_stats["accepted_tokens_total"] = accepted_tokens_total
    perf_stats["target_tokens_total"] = target_tokens_total
    return generated_text, perf_stats