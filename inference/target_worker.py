import logging
import grpc
from concurrent import futures

from transformers import AutoTokenizer
from inference import model_loader
from grpc_comm import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        # Load the target model (compiled or compile on the fly) and tokenizer
        logger.info(f"Loading target model from '{model_path}' (sequence_length={sequence_length})...")
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        logger.info("Target model and tokenizer loaded.")
        # Initialize generation state
        self.current_ids = None
        self.max_tokens = None
        self.tokens_generated = 0
        # Determine EOS token id for this model (if any)
        self.eos_token_id = self.tokenizer.eos_token_id

    def StartGeneration(self, request, context):
        """Initialize generation with the given prompt and optional max token limit."""
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens if request.max_new_tokens > 0 else None
        logger.info("StartGeneration called with prompt: \"%s\", max_new_tokens: %s", prompt_text, str(max_tokens))
        # Encode prompt into input IDs and reset state
        self.current_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        self.max_tokens = max_tokens
        self.tokens_generated = 0
        # Acknowledge the start of generation
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        """Verify a chunk of draft-predicted tokens against the target model's next tokens."""
        draft_tokens = list(request.draft_tokens)
        logger.info("VerifyDraftTokens called with draft_tokens (IDs): %s", draft_tokens)
        if self.current_ids is None:
            # If StartGeneration was not called, we cannot proceed
            logger.error("No active generation context. Call StartGeneration first.")
            return inference_pb2.VerifyResponse(all_matched=False, match_count=0, correct_token=0, finished=True)

        # Determine how many tokens to generate with target (length of draft_tokens)
        num_tokens = len(draft_tokens)
        # Get current sequence length (prompt + tokens already accepted)
        current_len = self.current_ids.shape[1]
        try:
            # Generate next `num_tokens` tokens from the current context
            output = self.model.sample(self.current_ids, sequence_length=current_len + num_tokens)
        except Exception as e:
            logger.error(f"Target model generation failed: {e}")
            return inference_pb2.VerifyResponse(all_matched=False, match_count=0, correct_token=0, finished=True)
        # Extract the newly generated target tokens
        if isinstance(output, (list, tuple)):
            target_seq = output[0]
        else:
            target_seq = output[0]
        target_new_ids = target_seq[current_len:]  # new tokens predicted by target
        target_new_ids = [int(t) for t in target_new_ids]
        logger.info("Target model predicted tokens (IDs): %s", target_new_ids)

        # Compare draft tokens with target tokens
        match_count = 0
        all_matched = True
        correct_token_id = 0
        finished = False

        # Determine if target hit EOS in its predictions
        if self.eos_token_id is not None and self.eos_token_id in target_new_ids:
            eos_index = target_new_ids.index(self.eos_token_id)
            # Truncate target_new_ids at EOS (including EOS token)
            target_new_ids = target_new_ids[:eos_index+1]
            # If EOS occurred before generating all requested tokens, mark finished
            finished = True
            logger.info("Target model generated EOS token at position %d of chunk.", eos_index)
            # We will compare only up to eos_index (inclusive) with draft tokens (draft may not have EOS)
            # If draft_tokens length is longer than target_new_ids now, treat those extra draft tokens as mismatch.
            # (We'll handle in comparison loop below.)

        # Compare token by token
        for i in range(min(len(draft_tokens), len(target_new_ids))):
            if draft_tokens[i] == target_new_ids[i]:
                match_count += 1
            else:
                all_matched = False
                correct_token_id = target_new_ids[i]
                break
        else:
            # If loop did not break (no mismatch in overlapped length)
            if len(draft_tokens) == len(target_new_ids):
                # All tokens compared are equal and lengths are the same
                all_matched = True
            else:
                # Lengths differ (target ended early or draft had fewer tokens)
                all_matched = False
                match_count = min(len(draft_tokens), len(target_new_ids))
                # If target produced fewer tokens (ended) and draft had more, then first "mismatch" is at target end
                if len(draft_tokens) > len(target_new_ids):
                    # target finished early (e.g., EOS) while draft predicted additional tokens
                    correct_token_id = 0  # no next token from target (it ended)
                    finished = True
                else:
                    # draft ended earlier (unlikely in normal operation since target controls finish)
                    correct_token_id = target_new_ids[len(draft_tokens)]
        # Update internal state with accepted tokens
        if all_matched:
            # All tokens in this chunk are correct (or target ended exactly at this chunk length)
            accepted_ids = target_new_ids  # same as draft_tokens in content for matched case
            # If target ended with EOS in this chunk, we mark finished
            if self.eos_token_id is not None and accepted_ids and accepted_ids[-1] == self.eos_token_id:
                finished = True
        else:
            # Mismatch occurred at index match_count
            # Accept the matching prefix and the target's correct token at mismatch (if any)
            accepted_ids = []
            if match_count > 0:
                accepted_ids += target_new_ids[:match_count]  # (same as draft_tokens[:match_count])
            if correct_token_id != 0:
                accepted_ids.append(correct_token_id)
        # Append accepted_ids to current context
        if accepted_ids:
            new_tokens_tensor = torch.tensor([accepted_ids], dtype=self.current_ids.dtype)
            self.current_ids = torch.cat([self.current_ids, new_tokens_tensor], dim=1)
        # Update count of tokens generated
        self.tokens_generated += len(accepted_ids)
        # If max_tokens limit is set, check if we've reached or exceeded it
        if self.max_tokens is not None and self.tokens_generated >= self.max_tokens:
            finished = True

        # Prepare response
        response = inference_pb2.VerifyResponse(
            all_matched=all_matched,
            match_count=match_count,
            correct_token=correct_token_id,
            finished=finished
        )
        logger.info("VerifyDraftTokens result: all_matched=%s, match_count=%d, correct_token_id=%s, finished=%s",
                    all_matched, match_count, str(correct_token_id), finished)
        return response

    def GenerateFull(self, request, context):
        """Generate a continuation for the given prompt using the target model (one-shot full generation)."""
        prompt = request.prompt
        max_new_tokens = request.max_new_tokens
        logger.info("GenerateFull called with prompt: \"%s\", max_new_tokens=%d", prompt, max_new_tokens)
        # Generate max_new_tokens tokens after the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.sample(input_ids, sequence_length=input_ids.shape[1] + max_new_tokens)
        # Extract all generated tokens (excluding the prompt)
        if isinstance(output, (list, tuple)):
            seq = output[0]
        else:
            seq = output[0]
        gen_ids = seq[input_ids.shape[1]:]
        output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        logger.info("GenerateFull returning text: \"%s\"", output_text)
        return inference_pb2.GenerateResponse(output_text=output_text)

def run_server(model_path, port=50051, sequence_length=128):
    """Launch the gRPC server hosting the target model for speculative decoding."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Target server starting on port %d (sequence_length=%d)", port, sequence_length)
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Target worker for speculative decoding")
    parser.add_argument("--model", type=str, required=True, help="Path to the target model (compiled model directory or pre-trained model path)")
    parser.add_argument("--port", type=int, default=50051, help="Port for the gRPC server")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model inference")
    args = parser.parse_args()
    run_server(args.model, port=args.port, sequence_length=args.sequence_length)
