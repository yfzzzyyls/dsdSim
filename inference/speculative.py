import logging
logger = logging.getLogger(__name__)

def speculative_decode(draft_model, tokenizer, target_stub, prompt_text: str, max_new_tokens: int = 50, window_size: int = 4):
    """
    Perform distributed speculative decoding: uses a draft model locally and a target model via gRPC.
    Returns the generated text (excluding the prompt).
    """
    # Encode the prompt text to input IDs
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids
    output_text = ""
    # Initialize conversation with target: send prompt to target and get first token
    logger.info("Sending initial prompt to target for first token...")
    from inference_pb2 import NextTokenRequest  # import here to avoid dependency issues
    request = NextTokenRequest(prompt=prompt_text)
    response = target_stub.NextToken(request)
    target_token_id = response.token_id
    target_token_text = response.token_text
    # Append to output
    output_text += target_token_text
    # Extend the input_ids with the new token for both draft and tracking target context
    import torch
    new_token_tensor = torch.tensor([[target_token_id]], dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
    logger.info(f"Received first token from target: {repr(target_token_text)} (id {target_token_id})")
    # Generate remaining tokens
    tokens_generated = 1
    while tokens_generated < max_new_tokens:
        # Use draft model to predict the next token
        # (We generate one token at a time for simplicity; window_size could be used for future optimization)
        draft_seq = draft_model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        # draft_seq may be a PyTorch tensor or list; handle accordingly
        if isinstance(draft_seq, (list, tuple)):
            draft_token_id = int(draft_seq[0][-1]) if isinstance(draft_seq[0], (list, tuple)) else int(draft_seq[0])
        else:
            # Assume it's a tensor
            draft_token_id = int(draft_seq[0, -1])
        draft_token_text = tokenizer.decode([draft_token_id])
        # Ask target for the next token (without sending prompt again)
        request = NextTokenRequest(prompt="")  # empty prompt indicates continuation
        response = target_stub.NextToken(request)
        target_token_id = response.token_id
        target_token_text = response.token_text
        # Compare draft's prediction with target's actual token
        if target_token_id == draft_token_id:
            # Speculative token accepted
            logger.info(f"Draft predicted token correctly: {repr(draft_token_text)}")
            output_text += target_token_text
        else:
            # Draft was wrong, use target's token
            logger.info(f"Draft prediction {repr(draft_token_text)} diverged, accepting target token {repr(target_token_text)} instead")
            output_text += target_token_text
            # Discard the draft token (if wrong, we reset the draft's assumed context to match target)
            draft_token_id = target_token_id
            draft_token_text = target_token_text
        # Append the accepted token (target_token_id) to input_ids for draft model context
        new_token_tensor = torch.tensor([[target_token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1
        # (Optional: break if EOS token is generated)
    return output_text
