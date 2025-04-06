from . import model_loader
from . import target_worker
from . import draft_worker
import torch

def speculative_decode(prompt, target_worker, draft_worker, max_steps=100, k=4):
    """
    Perform speculative decoding given a prompt, a target model worker, and a draft model worker.
    - prompt: list of input token IDs (already tokenized input sequence)
    - target_worker: an instance or interface to the target model (must support generate_token and verify_sequence)
    - draft_worker: an instance or interface to the draft model (must support generate_token)
    - max_steps: maximum number of decoding steps to generate
    - k: speculative decoding window (target provides 1 token, draft speculates the next k-1 tokens)
    Returns the full generated token sequence (including the prompt tokens and newly generated tokens).
    """
    # Ensure prompt is a list of token IDs (integers)
    if isinstance(prompt, str):
        raise ValueError("Prompt must be tokenized into token IDs before decoding.")
    output_tokens = list(prompt)
    end_of_text_token = None  # Define this based on the tokenizer (e.g., tokenizer.eos_token_id)

    for step in range(max_steps):
        # Step 1: Target model generates one guaranteed token
        input_ids = torch.tensor([output_tokens], dtype=torch.long)
        target_logits = target_worker.generate_token(input_ids)
        # Determine target_token (for simplicity, take argmax; could sample according to distribution if needed)
        target_token = int(torch.argmax(target_logits, dim=-1)[0])
        output_tokens.append(target_token)
        if end_of_text_token is not None and target_token == end_of_text_token:
            break  # Stop if end-of-sequence token generated

        # Step 2: Draft model speculates the next k-1 tokens
        draft_tokens = []
        for i in range(k - 1):
            input_ids = torch.tensor([output_tokens], dtype=torch.long)
            draft_logits = draft_worker.generate_token(input_ids)
            # Sample a token from draft model's distribution (using argmax here for simplicity)
            draft_token = int(torch.argmax(draft_logits, dim=-1)[0])
            draft_tokens.append(draft_token)
            output_tokens.append(draft_token)
            if end_of_text_token is not None and draft_token == end_of_text_token:
                break
        # If no speculative tokens (e.g., if draft immediately predicted EOS), continue to next loop iteration
        if not draft_tokens:
            continue

        # Step 3: Verify the draft tokens with the target model in one pass
        input_ids = torch.tensor([output_tokens], dtype=torch.long)
        target_logits_seq = target_worker.verify_sequence(input_ids)
        # Check each draft token against target's output distribution at that position
        accept_all = True
        for j, draft_token in enumerate(draft_tokens, start=1):
            # Compare target model's predicted token at the position of this draft token
            # (Using argmax for target's prediction; in practice, acceptance criteria could be probabilistic)
            target_index = - (len(draft_tokens) - j + 1)  # index from the end for the j-th draft token
            target_pred_id = int(torch.argmax(target_logits_seq[0, target_index, :]))
            if target_pred_id != draft_token:
                # Mismatch: target would have chosen a different token at this position
                # Roll back the sequence to the point of divergence
                output_tokens = output_tokens[: -(len(draft_tokens) - j + 1)]
                output_tokens.append(target_pred_id)
                accept_all = False
                break
        if not accept_all:
            # If a mismatch occurred, we accepted target's token at the divergence and discard remaining draft tokens
            continue
        # If all draft tokens are accepted, they remain in output_tokens and we continue generating further tokens.
    return output_tokens