import logging
import torch
import time

logger = logging.getLogger(__name__)

def speculative_decode(draft_model, tokenizer, target_stub, prompt_text: str,
                       max_new_tokens: int = 50, window_size: int = 4, profile_data=None):
    """
    Perform distributed speculative decoding using the draft model locally and the target model via gRPC.
    Returns the generated text (excluding the prompt). If profile_data (a dict) is provided, it will be updated
    with per-token generation times, token count, and token match count.
    """
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids
    output_text = ""
    from grpc_comm import inference_pb2  # Updated relative import
    logger.info("Sending initial prompt to target for first token...")
    start_time = time.time() if profile_data is not None else None
    request = inference_pb2.GenerateRequest(prompt=prompt_text, max_new_tokens=1)
    response = target_stub.GenerateFull(request)
    target_token_text = response.output_text.strip()
    target_token_ids = tokenizer.encode(target_token_text, add_special_tokens=False)
    if not target_token_ids:
        logger.error("Target server returned empty token text.")
        return output_text
    target_token_id = target_token_ids[0]
    output_text += target_token_text
    new_token_tensor = torch.tensor([[target_token_id]], dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
    logger.info(f"Received first token from target: '{target_token_text}' (id {target_token_id})")
    tokens_generated = 1
    token_times = [] if profile_data is not None else None
    match_count = 0
    if profile_data is not None:
        end_time = time.time()
        token_times.append(end_time - start_time)
    while tokens_generated < max_new_tokens:
        iter_start = time.time() if profile_data is not None else None
        draft_seq = draft_model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        if isinstance(draft_seq, (list, tuple)):
            draft_token_id = int(draft_seq[0][-1])
        else:
            draft_token_id = int(draft_seq[0, -1])
        draft_token_text = tokenizer.decode([draft_token_id]).strip()
        request = inference_pb2.GenerateRequest(prompt="", max_new_tokens=1)
        response = target_stub.GenerateFull(request)
        target_token_text = response.output_text.strip()
        target_token_ids = tokenizer.encode(target_token_text, add_special_tokens=False)
        if not target_token_ids:
            logger.error("Target server returned empty token text.")
            break
        target_token_id = target_token_ids[0]
        if target_token_id == draft_token_id:
            match_count += 1
            logger.info(f"Draft predicted correctly: '{draft_token_text}'")
            output_text += target_token_text
        else:
            logger.info(f"Draft predicted '{draft_token_text}' but target returned '{target_token_text}'. Accepting target token.")
            output_text += target_token_text
            draft_token_id = target_token_id
            draft_token_text = target_token_text
        new_token_tensor = torch.tensor([[target_token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1
        if profile_data is not None:
            iter_end = time.time()
            token_times.append(iter_end - iter_start)
    if profile_data is not None:
        profile_data['tokens_generated'] = tokens_generated
        profile_data['match_count'] = match_count
        profile_data['token_times'] = token_times
    return output_text
