import logging
import torch

logger = logging.getLogger(__name__)

def speculative_decode(draft_model, tokenizer, target_stub, prompt_text: str, max_new_tokens: int = 50, window_size: int = 4):
    """
    Perform distributed speculative decoding using the draft model locally and the target model via gRPC.
    Returns the generated text (excluding the prompt).
    """
    # Encode the prompt text into input IDs
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids
    output_text = ""
    
    # Get the first token from the target model via GenerateFull RPC
    logger.info("Sending initial prompt to target for first token...")
    from inference_pb2 import GenerateRequest
    request = GenerateRequest(prompt=prompt_text, max_new_tokens=1)
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
    logger.info(f"Received first token from target: {target_token_text} (id {target_token_id})")
    
    tokens_generated = 1
    while tokens_generated < max_new_tokens:
        draft_seq = draft_model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        if isinstance(draft_seq, (list, tuple)):
            draft_token_id = int(draft_seq[0][-1])
        else:
            draft_token_id = int(draft_seq[0, -1])
        draft_token_text = tokenizer.decode([draft_token_id]).strip()
        
        # Request the next token from target
        request = GenerateRequest(prompt="", max_new_tokens=1)
        response = target_stub.GenerateFull(request)
        target_token_text = response.output_text.strip()
        target_token_ids = tokenizer.encode(target_token_text, add_special_tokens=False)
        if not target_token_ids:
            logger.error("Target server returned empty token text.")
            break
        target_token_id = target_token_ids[0]
        
        if target_token_id == draft_token_id:
            logger.info(f"Draft predicted correctly: {draft_token_text}")
            output_text += target_token_text
        else:
            logger.info(f"Draft predicted {draft_token_text} but target returned {target_token_text}. Accepting target token.")
            output_text += target_token_text
            draft_token_id = target_token_id
            draft_token_text = target_token_text
        
        new_token_tensor = torch.tensor([[target_token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1
        
    return output_text
