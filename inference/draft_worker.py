import logging
import grpc
import os
import time
import json
import threading
import uuid
from datetime import datetime
from grpc_comm import inference_pb2_grpc, inference_pb2, grpc_client
from inference.model_loader import load_model
from transformers import AutoTokenizer
import torch
import random
import collections
from transformers_neuronx import sampling
from inference.model_loader import SPEC_LENGTH_BUCKETS

logger = logging.getLogger(__name__)

# Repetition‑penalty strength (0 < α ≤ 1).  Smaller → stronger penalty
REP_PENALTY = 0.4
NGRAM_WINDOW = 3    # penalise 1‑ to 3‑gram repeats
TOP_K = 128

if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helper: flatten tensors for gRPC payloads
# ---------------------------------------------------------------------------
def _tensor_to_flat(t: torch.Tensor):
    return {"data": t.flatten().tolist(), "shape": list(t.shape)}

# ---------------------------------------------------------------------------
# Batched speculative decoding (B prompts at once)
# ---------------------------------------------------------------------------
def batched_speculative_decode(
    draft_model, tokenizer, stub, prompts,
    batch_input_ids, max_new_tokens, gamma,
    *, profile=False, top_p=0.9, temperature=1.0, session_id=None,
    true_lengths: torch.Tensor | None = None,   # (B,) vector of real prompt lengths
):
    B, L = batch_input_ids.shape

    # 1) pre-fill
    # --------------------------------------------------------------
    # Pre-fill exactly like the target: call .forward() once with
    # cache_ids=None, then set the per-row KV pointer to true_lengths.
    # --------------------------------------------------------------
    assert true_lengths is not None, "true_lengths must be provided"

    # ------------------------------------------------------------------
    # Provide a (B, L) cache‑id matrix so vectorize_last_token_id=True
    # works exactly as on the target side.  Slots ≥ true_len[b] get ‑1.
    # ------------------------------------------------------------------
    B, L = batch_input_ids.shape
    row_pos   = torch.arange(L, dtype=torch.int32).unsqueeze(0).repeat(B, 1)
    over_len  = row_pos >= true_lengths.unsqueeze(1)
    cache_mat = row_pos.masked_fill(over_len, -1)          # (B, L)

    _ = draft_model.forward(batch_input_ids, cache_ids=cache_mat)
    draft_model.cache_ids = true_lengths.clone()           # (B,) next-free slot per row
    assert torch.equal(draft_model.cache_ids, true_lengths), (
        "Draft KV cache pointer desynchronised: cache_ids != true_lengths"
    )

    eos_tok = tokenizer.eos_token_id
    finished = torch.zeros(B, dtype=torch.bool)
    out_bufs = [[] for _ in range(B)]
    tokens_generated = torch.zeros(B, dtype=torch.int32)
    # the prev_token used to generate the next token
    last_token = batch_input_ids[:, -1].clone()

    scratch = torch.empty((1,1), dtype=torch.int64)
    valid_g = [b - 1 for b in SPEC_LENGTH_BUCKETS if b > 1]   # γ values that were compiled
    requested_gamma = max(1, gamma)                            # never allow γ < 1

    if requested_gamma not in valid_g:
        snapped_gamma = max(g for g in valid_g if g <= requested_gamma)
        logger.warning(
            "Requested gamma=%d is not supported by the compiled buckets %s; "
            "using gamma=%d instead.",
            requested_gamma, valid_g, snapped_gamma
        )
        gamma = snapped_gamma
    else:
        gamma = requested_gamma

    while not finished.all():
        # ------------------------------------------------------------------
        # Fully‑batched draft generation: do *γ* batched forward passes total.
        # We maintain an `active` mask so we only update rows that
        # still need tokens and skip those that are finished.
        # ------------------------------------------------------------------
        tok_rows  = [torch.empty((0,), dtype=torch.int64)  for _ in range(B)]
        prob_rows = [torch.empty((0,), dtype=torch.float32) for _ in range(B)]

        active = ~finished.clone()          # (B,) bool
        prev_tokens = last_token.clone()    # (B,)

        for step in range(gamma):
            if not active.any():
                break

            # (1) build batched (B_active, 1) input tensor of prev tokens
            active_idx = active.nonzero(as_tuple=False).squeeze(-1)    # indices
            inputs = prev_tokens[active_idx].unsqueeze(1)              # (B_a, 1)

            # (2) build cache_vec: Neuron single-token kernel needs (B_a, 1)
            cache_vec = draft_model.cache_ids[active_idx].unsqueeze(1) # (B_a, 1)

            # (3) single batched forward pass
            logits, _ = draft_model.forward(inputs, cache_ids=cache_vec)
            logits = logits / temperature                              # (B_a, V)

            # logits  : (B_active, V)  after temperature scaling
            # probs   : (B_active, V)  is no longer used directly

            # (4) sampling: apply n-gram filter + top-k / top-p filtering
            # (A) n‑gram repeat penalty
            # Build a right‑padded (B_a, C) context tensor so filter_ngrams
            # can run in one call even when rows have different history length.
            ctx_rows = [
                torch.tensor(out_bufs[b], dtype=torch.int64)
                for b in active_idx.tolist()
            ]
            if ctx_rows:
                C = max(r.numel() for r in ctx_rows)
                ctx_pad = torch.full((len(ctx_rows), C), 0, dtype=torch.int64)
                for i, row_t in enumerate(ctx_rows):
                    k = row_t.numel()
                    if k:
                        ctx_pad[i, :k] = row_t
            else:
                ctx_pad = torch.zeros((len(active_idx), 0), dtype=torch.int64)

            # --------------------------------------------------------------
            # transformers_neuronx.sampling.filter_ngrams expects `cur_len`
            # to be **scalar per call**.  It cannot accept a tensor of
            # different lengths (one per row).  We therefore call it in a
            # tiny per‑row loop here; γ is small, so the overhead is minimal.
            # --------------------------------------------------------------
            cache_vec_a = draft_model.cache_ids[active_idx]
            next_ids = []
            probs_row = []
            for j, row_logits in enumerate(logits):          # iterate B_active
                cur_len  = int(cache_vec_a[j].item())        # scalar length
                row_ctx  = ctx_pad[j : j+1, :]               # (1, C)
                row_mask = sampling.filter_ngrams(
                    NGRAM_WINDOW,
                    row_ctx,
                    row_logits.unsqueeze(0),   # (1, V)
                    cur_len
                )
                # Optional: apply nucleus/top‑k filtering.
                # The combined n‑gram mask is already quite restrictive and
                # the previous batch‑level call caused shape‑mismatch errors
                # inside `transformers_neuronx.top_k_top_p_filtering`.
                # For stability we skip that function here and simply
                # soft‑max over the n‑gram‑masked logits.
                row_probs = torch.softmax(row_mask, dim=-1).squeeze(0)  # (V,)
                # If numerical underflow masks all probs, fall back to uniform
                if row_probs.sum() <= 0:
                    row_probs = torch.ones_like(row_probs) / row_probs.numel()
                nid = int(torch.multinomial(row_probs, 1).item())
                next_ids.append(nid)
                prob_val = float(row_probs[nid]) if nid < row_probs.shape[0] else 1.0
                probs_row.append(prob_val)
            next_ids = torch.tensor(next_ids, dtype=torch.int64)
            probs    = torch.tensor(probs_row, dtype=torch.float32)

            # (5) write results back into row‑lists
            for idx, row in enumerate(active_idx):
                tok_rows[row]  = torch.cat([tok_rows[row], next_ids[idx:idx+1]])
                prob_rows[row] = torch.cat([prob_rows[row], probs[idx:idx+1]])

            # (6) update prev_tokens
            prev_tokens[active_idx] = next_ids

            # (7) check stopping conditions per row
            reached_eos = (eos_tok is not None) & (next_ids == eos_tok)
            reach_budget = (tokens_generated[active_idx] +
                            torch.tensor([len(tok_rows[r]) for r in active_idx])) >= max_new_tokens
            should_stop = reached_eos | reach_budget
            # mark finished rows
            active[active_idx[should_stop]] = False
            finished[active_idx[should_stop]] = True

        # ---- exit if nobody produced a token --------------------------------
        if all(row.numel() == 0 for row in tok_rows):
            break

        # ------------------------------------------------------------------
        # 2) Right-pad rows so they stack into (B, γ′) tensors
        # ------------------------------------------------------------------
        max_len = max(r.numel() for r in tok_rows)
        pad_val = 0
        tok_batch  = torch.full((B, max_len), pad_val, dtype=torch.int64)
        prob_batch = torch.zeros((B, max_len), dtype=torch.float32)
        for b in range(B):
            k = tok_rows[b].numel()
            if k:
                tok_batch[b, :k]  = tok_rows[b]
                prob_batch[b, :k] = prob_rows[b]

        # ------------------------------------------------------------------
        # 3) Verify whole batch in *one* RPC
        # ------------------------------------------------------------------
        result = grpc_client.verify_batch_tokens(
            stub,
            session_id,
            _tensor_to_flat(tok_batch),      # Int32Tensor  (B, γ′)
            _tensor_to_flat(prob_batch),     # FloatTensor (B, γ′)
        )
        # committed_ids is Int32Tensor → reshape to (B, K)
        commit_t = torch.tensor(result.committed_ids.data,
                                dtype=torch.int64).view(B, -1)    # (B, K)

        # ------------------------------------------------------------------
        # 4) Vectorised application of committed tokens
        # ------------------------------------------------------------------
        K = commit_t.shape[1]

        # (B,) length of real (non-PAD) commits per row
        commit_len = commit_t.ne(0).sum(dim=1, dtype=torch.int64)          # (B,)

        # Respect remaining budget per row
        remain     = (max_new_tokens - tokens_generated).clamp(min=0)      # (B,)
        commit_len = torch.minimum(commit_len, remain)

        # Mask of rows that actually receive new tokens *and* aren’t finished
        valid_mask = (commit_len > 0) & (~finished)
        if valid_mask.any():
            idx = valid_mask.nonzero(as_tuple=False).squeeze(1)            # (N,)

            # ---- tensorised state updates ----
            tokens_generated[idx] += commit_len[idx]
            draft_model.cache_ids[idx] += commit_len[idx].to(torch.int32)

            # last_token becomes the final committed ID on each row
            last_pos              = commit_len[idx] - 1                    # (N,)
            last_token[idx]       = commit_t[idx, last_pos]

            # rows that hit EOS or token budget are marked finished
            eos_row     = torch.zeros_like(idx, dtype=torch.bool)
            if eos_tok is not None:
                eos_row = (commit_t[idx] == eos_tok).any(dim=1)
            budget_row  = tokens_generated[idx] >= max_new_tokens
            finished[idx] = eos_row | budget_row

            # Python-list buffer extension (still per row)
            for b in idx.tolist():
                k = int(commit_len[b])
                if k:
                    out_bufs[b].extend(commit_t[b, :k].tolist())

    return [
        tokenizer.decode(buf, skip_special_tokens=True,
                         clean_up_tokenization_spaces=False)
        for buf in out_bufs
    ]


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
    # snap initial γ to the largest compiled bucket ≤ user request
    # Derive valid gammas and gamma_max from the bucket list
    valid_gammas = tuple(b - 1 for b in SPEC_LENGTH_BUCKETS if b > 1)
    gamma_max    = max(valid_gammas)
    current_gamma = max(g for g in valid_gammas if g <= max(1, gamma))
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

    # --------------------------------------------------------------
    # Per‑stage timing buckets (all values in seconds)
    # --------------------------------------------------------------
    timing = {
        "draft_prefill_time":       0.0,
        "draft_generation_time":    0.0,
        "grpc_roundtrip_time":      0.0,   # pure network + (de)serialisation latency
        "target_verification_time": 0.0,   # server‑side compute only
        "sampling_filter_time":     0.0,   # time spent on n‑gram mask + top‑k/p filter
    }
    
    # Feed the prompt so Neuron caches 0…L‑1, then set pointer to NEXT index (=L)
    if prompt_ids.shape[-1] > 0:
       # build the KV cache for the prompt
       time_draftprefill = time.perf_counter()
       _ = draft_model.forward(input_ids=prompt_ids)          # fills 0…L‑1
       timing["draft_prefill_time"] += time.perf_counter() - time_draftprefill
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

    while not finished and tokens_generated < max_new_tokens:
        # The draft model proposes up to 'gamma' tokens
        speculative_tokens = []
        speculative_probs = []
        # # ---------- just before the debug call, print he actual word generated ----------
        # token_texts = [tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        #             for tid in speculative_tokens]

        # past_states = [draft_model.cache_ids]
        for i in range(current_gamma):
            scratch_token[0, 0] = prev_token_id
            next_pos = draft_model._next_pos + i
            cache_vec = torch.tensor([next_pos], dtype=torch.int32)
            time_draftgen = time.perf_counter()            
            logits, _ = draft_model.forward(input_ids=scratch_token, cache_ids=cache_vec)
            timing["draft_generation_time"] += time.perf_counter() - time_draftgen
            # logits = logits.float()

            # ---- Our improved numeric stability start ----
            # Temperature‑scale logits then apply classic nucleus (top‑p) filter
            time_sample = time.perf_counter()
            # apply ngram filter
            masked = sampling.filter_ngrams(
                NGRAM_WINDOW,
                torch.tensor(output_tokens + speculative_tokens).unsqueeze(0),
                logits.unsqueeze(0),    # (1, V) expected
                next_pos
            )

            # apply temperature scaling
            logits = logits / current_temp

            # apply top‑k and top‑p filtering
            masked, candidate_idx = sampling.top_k_top_p_filtering(
                logits.unsqueeze(0),             # (1, V) expected
                top_k=TOP_K,
                top_p=top_p
            )
            timing["sampling_filter_time"] += time.perf_counter() - time_sample

            probs = torch.softmax(masked, dim=-1).squeeze(0)
            sample_in_topk = torch.multinomial(probs, 1).item()
            token_id   = int(candidate_idx[0, sample_in_topk])
            token_prob = float(probs[sample_in_topk])
 
            # store the token and its probability for later verification
            speculative_tokens.append(token_id)
            speculative_probs.append(token_prob)
            
            prev_token_id = token_id
            # past_states.append(draft_model.cache_ids + i)   # save pointer to next slot
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

        # ==============================================================
        # if len(speculative_tokens) not in valid_gammas:
        #     assert len(speculative_tokens) in valid_gammas, (
        #         f"Speculative chunk length {len(speculative_tokens)} is not "
        #         f"supported by the compiled target model; valid γ buckets = "
        #         f"{valid_gammas}"
        #     )
        # ============================================================

        # ============================================================
        # Ensure speculative_probs is the same length as speculative_tokens.
        # This can become mismatched when we break early from the inner loop
        # (e.g. EOS or token‑budget exhaustion).  Truncate to the shorter
        # length so we never advance _next_pos past the compiled context.
        # ============================================================
        
        # # ====================================================================
        # # --- Verify + commit in one RPC ---
        # # logger.debug("[session=%s] Proposed tokens: %s", session_id, speculative_tokens)
        # # --- show draft chunk as words instead of IDs -----------------
        # token_texts_dbg = [
        #     tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        #     for tid in speculative_tokens
        # ]
        # logger.debug("[session=%s] draft model proposed: chunk len=%d, proposed tokens (text)=%s, ids=%s, probs=%s", session_id, len(speculative_tokens), token_texts_dbg, speculative_tokens, speculative_probs)
        # # ====================================================================


        # ----- measure RPC round‑trip and split into network vs. verify compute -----
        time_roundtrip = time.perf_counter()
        commit_ids, accepted_count, verify_time_ms, target_finished = grpc_client.verify_draft_tokens(
            stub, speculative_tokens, speculative_probs, session_id=session_id
        )
        rpc_roundtrip = time.perf_counter() - time_roundtrip

        verify_sec   = verify_time_ms / 1000.0
        # Network + (de)serialisation + client/server scheduling
        network_sec  = max(0.0, rpc_roundtrip - verify_sec)

        timing["grpc_roundtrip_time"]      += network_sec
        timing["target_verification_time"] += verify_sec

        # ------------------------------------------------------------------
        # Respect the remaining token budget so we never exceed max_new_tokens.
        # If the target returned more tokens than we can still emit, truncate
        # the commit list *before* we touch any state‑tracking counters.
        # ------------------------------------------------------------------
        tokens_generated += len(commit_ids)
        remaining_budget = max_new_tokens - tokens_generated
        if remaining_budget <= 0:
            finished = True
            commit_ids = []
            break
        elif len(commit_ids) > remaining_budget:
            commit_ids = commit_ids[:remaining_budget]
            # clamp accepted_count as well
            if accepted_count > len(commit_ids):
                accepted_count = len(commit_ids)

        # Now that commit_ids is final, update global counters
        accepted_tokens_total += accepted_count
        # Track how many tokens were generated by the target model this loop
        target_tokens_total += max(0, len(commit_ids) - accepted_count)

        # --------------------------------------------------------------
        # ROLLBACK draft KV pointer to the state *before* the first
        # rejected token so we don't "leak" unused cache slots.
        # past_states[0] is the pointer *before* emitting any speculative
        # token; past_states[k] is after accepting k tokens.
        # --------------------------------------------------------------
        # rollback_ptr = past_states[accepted_count].clone()
        # draft_model.cache_ids = rollback_ptr
        # draft_model._next_pos = int(rollback_ptr.item())

        # ---------------- Hierarchical KV cache update -----------------
        # 1) Roll back the cache pointer so the *rejected* draft tokens
        #    disappear from the device KV.  They occupy the range
        #       [_next_pos - len(speculative_tokens), _next_pos)
        #    so we rewind by (len(speculative_tokens) - accepted_count).



        # 2) Forward **one** bonus token; accepted tokens already occupy 0…A‑1.
        bonus_id = commit_ids[-1]           # always present
        scratch_token[0, 0] = bonus_id

        # ============================================================
        # set bonus id to the next beginning of generating new draft tokens
        prev_token_id = bonus_id
        tok = bonus_id
        # ==============================================================
        # # 3) Advance pointer past the newly‑written bonus token.
        # this is used to update the KV cache ptr
        draft_model._next_pos += (accepted_count + 1)
        draft_model.cache_ids[0] = draft_model._next_pos
        # ==============================================================

        # Record every token that will appear in the final text
        output_tokens.extend(commit_ids)
        recent_deque.extend(commit_ids)
        if tokens_generated >= max_new_tokens:
            finished = True
            break
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

        # =======================================================================
        if target_finished or tokens_generated >= max_new_tokens:
            finished = True

    # Build final text
    generated_text = tokenizer.decode(
            output_tokens[-tokens_generated:],
            skip_special_tokens=True,               # ← strip EOS / PAD / BOS
            clean_up_tokenization_spaces=False,
        ) if output_tokens else ""

    # Performance stats
    perf_stats = {}
    # Compute total tokens produced by both draft (accepted) and target
    total_output_tokens = accepted_tokens_total + target_tokens_total
    if profile:
        tokens_generated_total = total_output_tokens
        perf_stats["tokens_generated"] = tokens_generated_total
        perf_stats.update({
            "draft_prefill_time":       timing["draft_prefill_time"],
            "draft_generation_time":    timing["draft_generation_time"],
            "grpc_roundtrip_time":      timing["grpc_roundtrip_time"],
            "target_verification_time": timing["target_verification_time"],
            "sampling_filter_time":     timing["sampling_filter_time"],
        })

    if total_output_tokens > 0:
        match_rate = accepted_tokens_total / total_output_tokens
        perf_stats["token_match_rate"] = match_rate

    if total_output_tokens > 0:
        logger.debug(
            f"Speculative decoding match rate: {match_rate:.2%} "
            f"(Draft accepted: {accepted_tokens_total}, Target generated: {target_tokens_total})"
        )

    logger.debug(
        f"[session={session_id}] Finished: generated_text='{generated_text[:120]}...'"
    )
    # Make these counters available to callers even when profiling is off
    perf_stats["accepted_tokens_total"] = accepted_tokens_total
    perf_stats["target_tokens_total"] = target_tokens_total
    return generated_text, perf_stats

def save_perf_stats(perf_stats: dict, file_prefix: str):
    """
    Save perf_stats to <file_prefix>.csv (append a row) and
    <file_prefix>.json (overwrite latest snapshot).
    """
    csv_path  = f"{file_prefix}.csv"
    json_path = f"{file_prefix}.json"
    try:
        total_time_val = perf_stats.get("total_time", 0.0)
        
        def fmt(t):
            if total_time_val > 0:
                pct = (t / total_time_val) * 100.0
                return f"{pct:.1f}%({t:.3f})"
            else:
                return f"0.0%({t:.3f})"
        
        # Append CSV row; write header if file does not exist
        header = ["total_time","tokens_generated","tokens_per_second",
                "avg_token_time","token_match_rate",
                "target_prefill_time","draft_prefill_time",
                "draft_generation_time",
                "grpc_roundtrip_time","target_verification_time",
                "sampling_filter_time"]

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline='') as cf:
            if write_header:
                cf.write(",".join(header) + "\n")
            row = [
                f"{total_time_val:.3f}",
                perf_stats.get("tokens_generated", ""),
                perf_stats.get("throughput", ""),
                perf_stats.get("avg_token_time", ""),
                perf_stats.get("token_match_rate", ""),
                fmt(perf_stats.get("target_prefill_time", 0.0)),
                fmt(perf_stats.get("draft_prefill_time", 0.0)),
                fmt(perf_stats.get("draft_generation_time", 0.0)),
                fmt(perf_stats.get("grpc_roundtrip_time", 0.0)),
                fmt(perf_stats.get("target_verification_time", 0.0)),
                fmt(perf_stats.get("sampling_filter_time", 0.0)),
            ]
            cf.write(",".join(str(x) for x in row) + "\n")

        # Always dump latest JSON snapshot
        with open(json_path, "w") as jf:
            json.dump(perf_stats, jf, indent=2)
        logger.info(f"Perf metrics appended to {csv_path} (snapshot {json_path})")
    except Exception as e:
        logger.error(f"Failed to save performance data: {e}")


def run_client(
    draft_model_name: str,
    target_host: str = "localhost",
    port: int = 50051,
    prompt_text_file: str = "",
    target_tokenizer: str = None,
    max_new_tokens: int = 50,
    sequence_length: int = 128,
    gamma: int = 4,
    profile: bool = False,
    top_p: float = 0.9,
    temperature: float = 1.0,
    batch_size: int = 1,
):
    if not os.path.exists(prompt_text_file):
        logger.error(f"Prompt text file not found: {prompt_text_file}")
        return
    with open(prompt_text_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    # ------------------------------------------------------------------
    # Batch‑size guardrail: ensure #prompts ≤ compiled batch_size
    # ------------------------------------------------------------------
    B = len(prompts)
    if B > batch_size:
        raise ValueError(
            f"Prompt file contains {B} lines, which exceeds the compiled "
            f"batch_size={batch_size}.  Either trim the prompt file or re-compile the "
            f"model with a larger --batch value."
        )
        # logger.error(
        #     "Prompt file contains %d lines, which exceeds the compiled "
        #     "batch_size=%d.  Either trim the prompt file or re‑compile the "
        #     "model with a larger --batch value.",
        #     B, batch_size,
        # )
        # return

    # ------------------------------------------------------------------
    # Tokenise all prompts together → (B, L) tensor  (right‑padded)
    # ------------------------------------------------------------------
    batch_tokenizer = AutoTokenizer.from_pretrained(
        target_tokenizer or draft_model_name,
        use_fast=False,
        padding_side="right",
    )
    # ------------------------------------------------------------------
    # Ensure the tokenizer has a PAD token and remap padding in the *tensor*
    # (not the tokenizer vocab) to ID 0.  We store the original pad ID so
    # we can swap it out after tokenisation.
    # ------------------------------------------------------------------
    if batch_tokenizer.pad_token_id is None:
        batch_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    original_pad_id = batch_tokenizer.pad_token_id
    # We won’t change pad_token_id yet; we first need to tokenise so we can
    # find every occurrence of the old pad ID in the tensor.

    # ============================================================
    # Tokenise prompts twice:
    #   (1) WITHOUT padding  → get true length of each prompt
    #   (2)  WITH  padding   → get right‑padded (B, L) tensor for the model
    # ============================================================

    # (1) true per‑row lengths BEFORE padding
    true_len_list = [
        len(batch_tokenizer.encode(p, add_special_tokens=True))
        for p in prompts
    ]
    true_lengths = torch.tensor(true_len_list, dtype=torch.int32)   # (B,)

    # (2) build the padded batch‑input tensor
    batch_tok = batch_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,      # right‑pad to max length across prompts
        truncation=False,  # never truncate – force caller to shorten
    )
    batch_input_ids = batch_tok["input_ids"]      # (B, L)
    # Replace old pad ID with 0 in the tensor, then make 0 the official PAD ID
    batch_input_ids[batch_input_ids == original_pad_id] = 0
    batch_tokenizer.pad_token_id = 0
    logger.info(
        "Prepared batch_input_ids tensor with shape %s (B=%d, L=%d)",
        tuple(batch_input_ids.shape),
        batch_input_ids.shape[0],
        batch_input_ids.shape[1],
    )

    # ============================================================
    # Check if the prompt tensor is empty
    assert prompts, "Prompt list is empty. Please provide valid prompts."
    # ============================================================

    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length}) for speculative decoding...")

    # ============================================================
    # Load the draft model
    assert isinstance(draft_model_name, str), (
        f"draft_model_name must be a string (path), not {type(draft_model_name)}"
    )
    draft_model = load_model(
        draft_model_name,
        sequence_length=sequence_length,
        spec_length=gamma,
        batch_size=batch_size
    )
    model_path_str = draft_model_name

    # Decide which tokenizer to load
    if target_tokenizer:
        tokenizer_source = target_tokenizer
    elif isinstance(model_path_str, str):
        tokenizer_source = model_path_str
    else:
        raise ValueError(
            "Cannot determine tokenizer_source: provide --target_tokenizer when "
            "passing a pre-loaded draft model."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False, padding_side="right")

    # start the loop
    start_time = time.time()

    address = f"{target_host}:{port}"
    logger.info(f"Connecting to target server at {address} (host={target_host}, port={port})...")
    channel = grpc.insecure_channel(address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)

    # ============================================================
    # ---------- create batch sessions ----------
    prompt_tensor = _tensor_to_flat(batch_input_ids)
    length_tensor  = _tensor_to_flat(true_lengths)
    sid_server = grpc_client.start_generation_batch(stub, prompt_tensor, length_tensor)
    # The server MUST return exactly one session‑id for the whole batch.
    if not sid_server or len(sid_server) == 0:
        raise RuntimeError("Target server returned no session_id for StartGenerationBatch()")

    # for a single-session batch the server sends exactly one ID, so the list always has length 1.
    batch_session_sid = sid_server[0]   # single authoritative session_id
    # ============================================================

    # ---------- run batched speculative decode ----------
    generated_texts = batched_speculative_decode(
        draft_model, tokenizer, stub,
        prompts, batch_input_ids,
        max_new_tokens, gamma,
        profile=profile, top_p=top_p, temperature=temperature,
        session_id=batch_session_sid,
        true_lengths = true_lengths,   # (B,) vector of real prompt lengths
    )

    for i, txt in enumerate(generated_texts):
        print(f"\n=== Output [{i}] ===\n{prompts[i]}{txt}\n")

    total_time = time.time() - start_time
    logger.info(f"Distributed speculative decode completed in {total_time:.2f}s.")

    print("\n=== Final Outputs ===")
    for i, text in enumerate(generated_texts):
        print(f"[Prompt {i} Output]:\n{text}\n")

def _gen_session_id():
    return int(uuid.uuid4()) & 0xFFFFFFFF