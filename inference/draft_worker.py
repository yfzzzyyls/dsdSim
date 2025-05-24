import logging
import grpc
import os
import time
import json
import threading
import uuid
from datetime import datetime
from grpc_comm import inference_pb2_grpc, inference_pb2, grpc_client
from inference.model_loader import load_model, get_spec_bucket_for_gamma, pad_tokens_to_bucket, SPEC_LENGTH_BUCKETS
from transformers import AutoTokenizer
import torch
import random
import collections
import concurrent.futures
from transformers_neuronx import sampling

logger = logging.getLogger(__name__)

# Repetition penalty strength (0 < α ≤ 1). Smaller → stronger penalty
REP_PENALTY = 0.4
NGRAM_WINDOW = 3    # penalise 1 to 3 gram repeats
TOP_K = 128

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
    # Derive valid gammas and gamma_max from the bucket list
    valid_gammas = tuple(b - 1 for b in SPEC_LENGTH_BUCKETS if b > 1)

    # Fail fast if the global bucket list is misconfigured
    if not valid_gammas:
        raise ValueError(
            "SPEC_LENGTH_BUCKETS must contain integers > 1; "
            f"current value = {SPEC_LENGTH_BUCKETS}"
        )
    gamma_max = max(valid_gammas)
    
    # Keep gamma stable at requested value (no dynamic adjustment for debugging)
    current_gamma = min(gamma, gamma_max)  # Start with requested gamma, clamped to max
    current_temp = temperature

    logger.debug(
        f"[session={session_id}] Starting speculative_decode with stable gamma: "
        f"gamma={current_gamma}, max_gamma={gamma_max}, buckets={SPEC_LENGTH_BUCKETS}"
    )

    # Initial setup: process prompt through draft model to initialize cache
    output_tokens = []
    
    # pre-filling: Feed the entire prompt once so the draft model builds its KV cache
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
    assert prompt_ids is not None, "Prompt tokenization failed due to empty input."
    prev_token_id = int(prompt_ids[0, -1].item())

    # Per stage timing buckets (all values in seconds)
    timing = {
        "draft_prefill_time":       0.0,
        "draft_generation_time":    0.0,
        "grpc_roundtrip_time":      0.0,   # pure network + (de)serialisation latency
        "target_verification_time": 0.0,   # server side compute only
        "target_prefill_time":      0.0,   # server side prefill time
        "sampling_filter_time":     0.0,   # time spent on n gram mask + top k/p filter
    }
    
    # Feed the prompt so Neuron caches 0…L-1, then set pointer to NEXT index (=L)
    # build the KV cache for the prompt
    time_draftprefill = time.perf_counter()
    L = prompt_ids.shape[1]
    cache_vec = torch.arange(L, dtype=torch.int32).unsqueeze(0)   # (1, L)
    _ = draft_model.forward(
        input_ids=prompt_ids,
        cache_ids=cache_vec,
    )                                 # fills 0…L-1
    timing["draft_prefill_time"] += time.perf_counter() - time_draftprefill
    # Overwrite cache pointer with a single index tensor [L]
    draft_model.update_cache(L)

    tokens_generated = 0
    # reusable scratch tensor (1,1) for single-token forwards
    scratch_token = torch.empty((1, 1), dtype=torch.int64)
    # fixed-size deque for fast repetition penalty history
    recent_deque  = collections.deque(maxlen=50)
    finished = False
    accepted_tokens_total = 0
    target_tokens_total = 0

    while not finished and tokens_generated < max_new_tokens:
        # Determine the bucket size needed for current gamma
        spec_bucket = get_spec_bucket_for_gamma(current_gamma, SPEC_LENGTH_BUCKETS)
        
        # The draft model proposes up to 'current_gamma' tokens
        speculative_tokens = []
        speculative_probs = []

        for i in range(current_gamma):
            scratch_token[0, 0] = prev_token_id
            # Compute the absolute KV-cache position for this token
            current_ptr = int(draft_model.get_cache_id_vec().item())
            next_pos = current_ptr + i
            # Neuron decoder expects (B, 1) – wrap pos in an extra bracket
            cache_vec = torch.tensor([[next_pos]],
                                     dtype=torch.int32,
                                     device=scratch_token.device)   # shape = (1, 1)
            time_draftgen = time.perf_counter()            
            logits, _ = draft_model.forward(input_ids=scratch_token, cache_ids=cache_vec)
            timing["draft_generation_time"] += time.perf_counter() - time_draftgen

            # Temperature scale logits then apply classic nucleus (top p) filter
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

            # apply top k and top p filtering
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
            
            # Stop if end-of-sequence or max_new_tokens reached
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                break
            if tokens_generated + len(speculative_tokens) >= max_new_tokens:
                break

        if not speculative_tokens:
            break

        # Pad draft tokens to the spec bucket size if needed (for target model compatibility)
        # The bucket size includes the bonus token, so for draft tokens we need bucket_size - 1
        max_draft_slots = spec_bucket - 1  # Reserve 1 slot for bonus token
        padded_tokens, original_length = pad_tokens_to_bucket(
            speculative_tokens, max_draft_slots, pad_token_id=0
        )
        # Ensure we pad the probabilities to match
        padded_probs = speculative_probs + [0.0] * (len(padded_tokens) - len(speculative_probs))

        # measure RPC round trip and split into network vs. verify compute
        time_roundtrip = time.perf_counter()
        commit_ids, accepted_count, verify_time_ms, target_finished = grpc_client.verify_draft_tokens(
            stub, padded_tokens, padded_probs, session_id=session_id
        )
        rpc_roundtrip = time.perf_counter() - time_roundtrip

        verify_sec   = verify_time_ms / 1000.0
        # Network + (de)serialisation + client/server scheduling
        network_sec  = max(0.0, rpc_roundtrip - verify_sec)

        timing["grpc_roundtrip_time"]      += network_sec
        timing["target_verification_time"] += verify_sec
        
        # Only consider acceptances up to the original draft length (ignore padded tokens)
        actual_accepted = min(accepted_count, original_length)

        # Respect the remaining token budget so we never exceed max_new_tokens.
        tokens_generated += len(commit_ids)
        remaining_budget = max_new_tokens - tokens_generated
        if remaining_budget <= 0:
            finished = True
            commit_ids = []
            break
        elif len(commit_ids) > remaining_budget:
            commit_ids = commit_ids[:remaining_budget]
            # clamp accepted_count as well
            if actual_accepted > len(commit_ids):
                actual_accepted = len(commit_ids)

        # Now that commit_ids is final, update global counters
        accepted_tokens_total += actual_accepted
        # Track how many tokens were generated by the target model this loop
        target_tokens_total += max(0, len(commit_ids) - actual_accepted)

        # Forward **one** bonus token only if we actually committed tokens
        if not commit_ids:
            # Log empty commit but don't change gamma (stable gamma for debugging)
            logger.debug("[session=%s] empty commit, keeping gamma=%d", session_id, current_gamma)
            continue

        bonus_id = commit_ids[-1]
        scratch_token[0, 0] = bonus_id

        # Set the last committed token as the new context token for draft model
        prev_token_id = bonus_id
        
        # Advance the draft model's KV pointer by (accepted_count + bonus)
        delta = actual_accepted + 1          # bonus token counts as +1
        draft_model.update_cache(delta)     # unified, increment-by-delta API

        # Record every token that will appear in the final text
        output_tokens.extend(commit_ids)
        recent_deque.extend(commit_ids)
        if tokens_generated >= max_new_tokens:
            finished = True
            break
        if tokenizer.eos_token_id is not None and bonus_id == tokenizer.eos_token_id:
            finished = True

        logger.debug("ACCEPT cnt=%d  committed=%s",
             actual_accepted,
             speculative_tokens[:actual_accepted])
        # Propagate server side finished flag
        finished = finished or target_finished

        if target_finished or tokens_generated >= max_new_tokens:
            finished = True

    # Build final text
    generated_text = tokenizer.decode(
            output_tokens[-tokens_generated:],
            skip_special_tokens=True,               # strip EOS / PAD / BOS
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
            "target_prefill_time":      timing["target_prefill_time"],
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
        header = [
            'total_time',
            'tokens_generated',
            'tokens_per_second',
            'token_match_rate',
            'target_prefill_time',
            'draft_prefill_time',
            'draft_generation_time',
            'grpc_roundtrip_time',
            'target_verification_time',
            'sampling_filter_time',
        ]

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline='') as cf:
            if write_header:
                cf.write(",".join(header) + "\n")
            row = [
                f"{total_time_val:.3f}",
                perf_stats.get("tokens_generated", ""),
                perf_stats.get("tokens_per_second", ""),
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
    target_tokenizer: str = "",
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
    if not prompts:
        logger.error("No valid lines in the prompt file.")
        return
    
    assert (gamma <= max(SPEC_LENGTH_BUCKETS) - 1), (
        f"Gamma {gamma} exceeds the maximum supported length "
        f"({max(SPEC_LENGTH_BUCKETS) - 1}). Please choose a smaller value."
    )
    assert (gamma >= 1), (
        f"Gamma {gamma} is less than the minimum supported length "
        f"(1). Please choose a larger value."
    )
    
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length}) for speculative decoding...")
    if isinstance(draft_model_name, str):
        # draft_model_name is a path → load the model
        draft_model = load_model(
            draft_model_name,
            sequence_length=sequence_length,
            spec_length=gamma,
            batch_size=batch_size
        )
        model_path_str = draft_model_name
    else:
        raise TypeError("draft_model_name must be a string (path).")

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

    _tokenizer_local = threading.local()
    def _get_tokenizer():
        """Return a tokenizer unique to the current thread."""
        if not hasattr(_tokenizer_local, "tok"):
            _tokenizer_local.tok = AutoTokenizer.from_pretrained(
                tokenizer_source, use_fast=False
            )
        return _tokenizer_local.tok

    def _worker(prompt_idx, prompt_text):
        tokenizer = _get_tokenizer()        # thread-specific copy
        # Ask the target to assign a canonical session-id
        # Create a simple request object that mimics the protobuf message
        request = type('StartGenerationRequest', (), {
            'session_id': 0,
            'prompt': prompt_text,
            'max_new_tokens': max_new_tokens,
            'gamma': gamma,
        })()
        
        start_resp = stub.StartGeneration(request)
        sid = start_resp.session_id
        t0 = time.time()
        gen_text, perf_stats = speculative_decode(
            draft_model, tokenizer, stub,
            prompt_text, max_new_tokens, gamma,
            profile=profile, top_p=top_p, temperature=temperature,
            session_id=sid
        )
        latency = time.time() - t0
        final_text = prompt_text + gen_text

        # Compute per thread throughput
        tokens_out = perf_stats.get("tokens_generated")
        if tokens_out is None:
            # Derive from accepted + target counts when profiling is off
            tokens_out = (
                perf_stats.get("accepted_tokens_total", 0) +
                perf_stats.get("target_tokens_total", 0)
            )
            perf_stats["tokens_generated"] = tokens_out

        # Record total wall-clock time and throughput
        perf_stats["total_time"] = latency
        throughput = tokens_out / latency if latency > 0 else 0.0
        perf_stats["tokens_per_second"] = throughput

        logger.info(
            "[Thread %d] completed in %.2fs, tokens=%d, throughput=%.2f t/s",
            prompt_idx, latency,
            tokens_out,
            throughput,
        )
        return {
            "prompt_idx": prompt_idx,
            "text": final_text,
            "latency": latency,
            "perf": perf_stats,
        }

    start_time = time.time()
    address = f"{target_host}:{port}"
    logger.info(f"Connecting to target server at {address} (host={target_host}, port={port})...")
    channel = grpc.insecure_channel(address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), 8)) as pool:
        fut2idx = {pool.submit(_worker, i, p): i for i, p in enumerate(prompts)}
        for fut in concurrent.futures.as_completed(fut2idx):
            results.append(fut.result())

    # Sort results by original prompt order
    results.sort(key=lambda r: r["prompt_idx"])

    # Pretty print outputs
    print("\n=== Final Outputs (CONCURRENT) ===")
    for r in results:
        print(f"[Prompt {r['prompt_idx']} Output]:\n{r['text']}\n")

    # Aggregate CSV if profiling
    if profile:
        for r in results:
            save_perf_stats(r["perf"], file_prefix="performance_speculative")

    total_time = time.time() - start_time
    logger.info(f"Distributed speculative decode completed in {total_time:.2f}s.")
