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
        top_p=0.9):

    output_tokens = []
    tokens_generated = 0
    finished = False
    past = None
    accepted_tokens_total = 0
    target_tokens_total = 0

    # ====== NEW: tokenized prompt for the first call when output_tokens is empty ====== 
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids

    while not finished and tokens_generated < max_new_tokens:
        draft_tokens = []
        draft_probs = []

        for i in range(gamma):
            # 1) Get next-token logits
            if past is None:
                # If we already have some generated tokens
                if output_tokens:
                    input_ids = torch.tensor([output_tokens], dtype=torch.long)
                else:
                    # <-- FIX: use the actual prompt tokens, not an empty tensor
                    input_ids = prompt_ids
            else:
                # We have past state, so pass just the last token
                if output_tokens:
                    last_token_id = torch.tensor([[output_tokens[-1]]], dtype=torch.long)
                else:
                    # In theory this shouldn't happen if we have a `past`, but just in case:
                    last_token_id = prompt_ids
                input_ids = last_token_id

            outputs = draft_model(input_ids=input_ids, use_cache=True, past_key_values=past)

            # Distinguish standard HF vs. compiled model shape
            try:
                logits = outputs.logits            # shape [batch=1, seq_len=1, vocab_size]
                logits = logits[0, -1, :]          # final step => shape [vocab_size]
            except AttributeError:
                logits = outputs[0]               # compiled => shape [vocab_size]

            past = getattr(outputs, "past_key_values", None)

            # 2) Top-p sample from the logits
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            if torch.any(cumulative_probs >= top_p):
                cutoff_index = torch.where(cumulative_probs >= top_p)[0][0].item()
            else:
                cutoff_index = len(sorted_probs) - 1
            top_probs = sorted_probs[:cutoff_index+1]
            top_indices = sorted_indices[:cutoff_index+1]
            top_probs = top_probs / torch.sum(top_probs)
            choice_index = torch.multinomial(top_probs, num_samples=1).item()
            token_id = int(top_indices[choice_index].item())
            token_prob = float(top_probs[choice_index].item())

            draft_tokens.append(token_id)
            draft_probs.append(token_prob)
            output_tokens.append(token_id)
            tokens_generated += 1

            # EOS / length checks
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                break
            if tokens_generated >= max_new_tokens:
                break

        # 3) Handle overshoot
        if len(draft_tokens) > 0 and tokens_generated > max_new_tokens:
            overshoot = tokens_generated - max_new_tokens
            draft_tokens = draft_tokens[:-overshoot]
            draft_probs = draft_probs[:-overshoot]
            output_tokens = output_tokens[:-overshoot]
            tokens_generated = max_new_tokens
            finished = True

        if not draft_tokens:
            break

        # 4) Verify with target
        target_probs, target_finished = grpc_client.verify_draft_tokens(stub, draft_tokens)
        if target_finished and len(target_probs) < len(draft_tokens):
            draft_tokens = draft_tokens[:len(target_probs)]
            draft_probs = draft_probs[:len(target_probs)]
            finished = True

        # 5) Accept or reject
        accept_count = 0
        break_point = False
        for idx, token_id in enumerate(draft_tokens):
            p_target = float(target_probs[idx]) if idx < len(target_probs) else 0.0
            p_draft = float(draft_probs[idx]) if idx < len(draft_probs) else 1e-9
            ratio = p_target / p_draft if p_draft > 0 else 0.0
            ratio = min(ratio, 1.0)  # cap at 1

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

        # Remove unaccepted tokens
        if break_point and accept_count < len(draft_tokens):
            while len(output_tokens) > (accepted_tokens_total + target_tokens_total):
                output_tokens.pop()
                tokens_generated -= 1

        # 6) Finalize => possibly get a fallback token
        final_token_id, finalize_finished = grpc_client.finalize_tokens(stub, accept_count)
        if final_token_id != 0:
            output_tokens.append(final_token_id)
            tokens_generated += 1
            target_tokens_total += 1
            if tokenizer.eos_token_id is not None and final_token_id == tokenizer.eos_token_id:
                finished = True

        if finalize_finished or tokens_generated >= max_new_tokens:
            finished = True

    # 7) Decode final
    generated_text = tokenizer.decode(output_tokens[-tokens_generated:]) if output_tokens else ""

    total_output_tokens = accepted_tokens_total + target_tokens_total
    perf_stats = {}
    if total_output_tokens > 0:
        match_rate = accepted_tokens_total / total_output_tokens
        logger.info(f"Speculative decoding match rate: {match_rate:.2%} "
                    f"(Draft accepted: {accepted_tokens_total}, Target generated: {target_tokens_total})")
        perf_stats["token_match_rate"] = match_rate

    return generated_text, perf_stats
