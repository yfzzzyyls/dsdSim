import inference_pb2

def run_speculative_decoding(draft_model, tokenizer, stub, prompt, max_tokens, num_speculative):
    """
    Coordinates speculative decoding between the draft model and the target via gRPC.
    Returns the final output token list (including prompt and generated continuation).
    """
    # Tokenize prompt for the draft model
    prompt_enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_enc["input_ids"][0].tolist()
    # Start the session on target
    start_req = inference_pb2.StartRequest(prompt=prompt, max_new_tokens=max_tokens)
    stub.StartGeneration(start_req)
    output_tokens = prompt_ids[:]  # initialize output with prompt tokens
    generated_count = 0  # number of new tokens generated so far

    eos_token_id = tokenizer.eos_token_id
    finished = False

    # Generation loop
    while not finished and generated_count < max_tokens:
        # Draft model generates up to `num_speculative` new tokens greedily (or until EOS if it comes sooner).
        draft_context = tokenizer.decode(output_tokens, skip_special_tokens=False)
        # We will generate one token at a time in a loop to have control, or use model.generate for speed.
        draft_tokens = []
        for _ in range(num_speculative):
            # Stop if we reached max_tokens
            if generated_count >= max_tokens:
                break
            input_ids = tokenizer(output_tokens, return_tensors="pt").input_ids
            outputs = draft_model(input_ids)
            logits = outputs.logits[0, -1, :]
            next_id = int(logits.argmax())
            # Append draft predicted token
            draft_tokens.append(next_id)
            # Append to output_tokens as provisional (they'll be removed if mismatch later)
            output_tokens.append(next_id)
            generated_count += 1
            # Break if draft predicts EOS
            if eos_token_id is not None and next_id == eos_token_id:
                break

        if len(draft_tokens) == 0:
            # No tokens generated (max_tokens might have been reached)
            break

        # Send draft tokens to target for verification
        verify_req = inference_pb2.VerifyRequest(draft_tokens=draft_tokens)
        response = stub.VerifyDraftTokens(verify_req)
        # Process response
        if response.all_matched:
            # All speculative tokens were correct
            # They are already appended in output_tokens, and target model has accepted them.
            finished = bool(response.finished)
            # If finished is True, it means target encountered EOS (matching draft) or length limit
        else:
            # There was a mismatch at response.match_count (number of tokens matched)
            # Truncate any extra draft tokens beyond the match
            match_count = response.match_count
            # The output_tokens list currently includes all draft_tokens; keep only those that matched
            output_tokens = output_tokens[:len(output_tokens) - (len(draft_tokens) - match_count)]
            # Append the correct token from target
            correct_id = int(response.correct_token)
            output_tokens.append(correct_id)
            generated_count = len(output_tokens) - len(prompt_ids)  # recompute new tokens count
            # If the correct token was EOS or target finished, mark finished
            finished = bool(response.finished or (eos_token_id is not None and correct_id == eos_token_id))
        # If we've generated max_tokens new tokens, end as well
        if generated_count >= max_tokens:
            finished = True

    return output_tokens