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
    top_p=0.9
):
    """
    Perform probability-based speculative decoding using a draft model and a target model via gRPC,
    with full rollback of the draft model's past states, just like lucidrains.
    Returns (generated_text, perf_stats).
    """
    output_tokens = []
    tokens_generated = 0
    finished = False

    accepted_tokens_total = 0
    target_tokens_total = 0

    # Tokenize the user prompt for the first pass
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids

    # We'll keep 'past' for the draft model so it can incrementally generate
    past = None

    # Main decoding loop
    while not finished and tokens_generated < max_new_tokens:
        # The draft model proposes up to 'gamma' tokens
        draft_tokens = []
        draft_probs = []

        # We also store a list of 'past' states, so we can rollback
        # past_states[i] = the 'past' right BEFORE generating draft_tokens[i].
        # Start with the 'past' from the last iteration (accepted state).
        past_states = [past]

        for i in range(gamma):
            # 1) Prepare input_ids
            if past is None:
                # If no 'past' yet
                if output_tokens:
                    input_ids = torch.tensor([output_tokens], dtype=torch.long)
                else:
                    # Use actual prompt tokens
                    input_ids = prompt_ids
            else:
                # We have a valid 'past'; feed only the last accepted token
                if output_tokens:
                    last_token_id = torch.tensor([[output_tokens[-1]]], dtype=torch.long)
                else:
                    last_token_id = prompt_ids
                input_ids = last_token_id

            # 2) Forward pass on the draft model
            outputs = draft_model(input_ids=input_ids, use_cache=True, past_key_values=past)

            # 3) Extract logits from standard HF or compiled model
            try:
                # HF => shape [batch=1, seq_len=1, vocab_size]
                logits = outputs.logits
                logits = logits[0, -1, :]
            except AttributeError:
                # Compiled => shape [1, vocab_size]
                logits = outputs[0]

            # 4) Prepare next 'past'
            new_past = getattr(outputs, "past_key_values", None)

            # 5) Sample next token from top-p
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
            choice_index = torch.multinomial(top_probs, 1).item()
            token_id = int(top_indices[choice_index].item())
            token_prob = float(top_probs[choice_index].item())

            # 6) Save draft token + prob
            draft_tokens.append(token_id)
            draft_probs.append(token_prob)

            # 7) Append the draft token tentatively to output + update 'past'
            output_tokens.append(token_id)
            tokens_generated += 1

            # Also store the new_past in past_states for rollback
            past_states.append(new_past)

            # 8) Check EOS or token limit
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                break
            if tokens_generated >= max_new_tokens:
                break

            # Update 'past' to new_past so next token generation uses it
            past = new_past

        # If we overshoot max_new_tokens
        if len(draft_tokens) > 0 and tokens_generated > max_new_tokens:
            overshoot = tokens_generated - max_new_tokens
            draft_tokens = draft_tokens[:-overshoot]
            draft_probs = draft_probs[:-overshoot]
            output_tokens = output_tokens[:-overshoot]
            tokens_generated = max_new_tokens
            finished = True

        if not draft_tokens:
            break

        # 9) Verify tokens with target model
        target_probs, target_finished = grpc_client.verify_draft_tokens(stub, draft_tokens)
        if target_finished and len(target_probs) < len(draft_tokens):
            # partial consumption => treat the rest as rejected
            draft_tokens = draft_tokens[:len(target_probs)]
            draft_probs = draft_probs[:len(target_probs)]
            finished = True

        # 10) Accept or reject
        accept_count = 0
        break_point = False

        for idx, token_id in enumerate(draft_tokens):
            p_target = float(target_probs[idx]) if idx < len(target_probs) else 0.0
            p_draft = float(draft_probs[idx]) if idx < len(draft_probs) else 1e-9
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
                # reject at this token
                break_point = True
                break

        # 11) If partial rejection, remove unaccepted tokens from output_tokens
        #     and revert the small model's past to the correct state
        if break_point and accept_count < len(draft_tokens):
            # unaccepted_count = len(draft_tokens) - accept_count
            # pop from output_tokens
            unaccepted = len(draft_tokens) - accept_count
            while unaccepted > 0:
                output_tokens.pop()  # remove last token
                tokens_generated -= 1
                unaccepted -= 1

            # rollback the draft model's 'past'
            # the state at 'past_states[accept_count]' is the last accepted token's past
            past = past_states[accept_count]

        # 12) Finalize => if partial acceptance, the target might give us a fallback token
        final_token_id, finalize_finished = grpc_client.finalize_tokens(stub, accept_count)
        if final_token_id != 0:
            output_tokens.append(final_token_id)
            tokens_generated += 1
            target_tokens_total += 1
            if tokenizer.eos_token_id is not None and final_token_id == tokenizer.eos_token_id:
                finished = True

        if finalize_finished or tokens_generated >= max_new_tokens:
            finished = True

    # Build final text
    generated_text = tokenizer.decode(output_tokens[-tokens_generated:]) if output_tokens else ""

    # Build perf stats
    total_output_tokens = accepted_tokens_total + target_tokens_total
    perf_stats = {}
    if total_output_tokens > 0:
        match_rate = accepted_tokens_total / total_output_tokens
        logger.info(
            f"Speculative decoding match rate: {match_rate:.2%} "
            f"(Draft accepted: {accepted_tokens_total}, Target generated: {target_tokens_total})"
        )
        perf_stats["token_match_rate"] = match_rate

    return generated_text, perf_stats
