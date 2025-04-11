import logging
import time
import torch
from concurrent import futures
import grpc

from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        # Load or compile the target model for inference
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.current_ids = None
        self.max_tokens = 0
        self.tokens_generated = 0

        # Identify EOS token if present
        try:
            self.eos_token_id = self.tokenizer.eos_token_id
        except:
            self.eos_token_id = None

    # 1) StartGeneration (already matches proto)
    def StartGeneration(self, request, context):
        """Initialize generation with the given prompt and optional max token limit."""
        prompt_text = request.prompt or ""
        max_tokens = request.max_new_tokens
        logger.info(f"StartGeneration called with prompt: \"{prompt_text}\", max_new_tokens: {max_tokens}")

        # Encode prompt into input IDs and reset generation state
        self.current_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        self.max_tokens = max_tokens
        self.tokens_generated = 0
        return inference_pb2.StartResponse(acknowledged=True)

    # 2) ADD THIS: VerifyDraftTokens (exactly matches your .proto)
    def VerifyDraftTokens(self, request, context):
        """
        Merged from grpc_server.py:
        Called by the client with `VerifyRequest` (draft_tokens).
        Must return `VerifyResponse` with (target_probs, finished).
        """
        draft_tokens = list(request.draft_tokens)
        logger.info(f"VerifyDraftTokens called with draft_tokens={draft_tokens}")

        # You need a function that actually computes probabilities for each draft token
        # Example: we'll define a helper method self.verify_tokens(...) below.
        result = self.verify_tokens(draft_tokens)
        target_probs = result["target_probs"]  # list of float probabilities for each token
        finished = result.get("finished", False)

        return inference_pb2.VerifyResponse(
            target_probs=target_probs,
            finished=finished
        )

    # 3) ADD THIS: FinalizeTokens (exactly matches your .proto)
    def FinalizeTokens(self, request, context):
        """
        Merged from grpc_server.py:
        Called by the client with `FinalizeRequest(accepted_count)`.
        Must return `FinalizeResponse(final_token, finished)`.
        """
        accepted_count = request.accepted_count
        logger.info(f"FinalizeTokens called with accepted_count={accepted_count}")

        # We'll define a helper method self.finalize_tokens(...) below
        result = self.finalize_tokens(accepted_count)
        final_token = result.get("final_token", 0)
        finished = result.get("finished", False)

        return inference_pb2.FinalizeResponse(
            final_token=final_token,
            finished=finished
        )

    # 4) GenerateFull (already matches proto)
    def GenerateFull(self, request, context):
        """Generate a full continuation for the given prompt using the target model (one-shot)."""
        prompt_text = request.prompt or ""
        max_tokens = request.max_new_tokens
        logger.info(f"GenerateFull called with prompt: \"{prompt_text}\", max_new_tokens: {max_tokens}")

        # Reset and encode prompt
        self.current_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        self.tokens_generated = 0
        output = self.model.sample(self.current_ids, sequence_length=self.current_ids.shape[1] + max_tokens)
        generated_ids = output[0][self.current_ids.shape[1]:] if isinstance(output, (list, tuple)) else output[0][self.current_ids.shape[1]:]
        generated_ids = [int(t) for t in generated_ids]
        logger.info(f"Target model one-shot generated tokens: {generated_ids}")

        # Return a textual result in the output_text field
        gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return inference_pb2.GenerateResponse(output_text=gen_text)

    # ------------------------------------------------------------------------
    # HELPER METHODS below (not part of proto) so your code can fill the logic
    # ------------------------------------------------------------------------

    def verify_tokens(self, draft_tokens):
        """
        Example: run the big model forward pass to get float logits, for each draft token,
        so we can compute probabilities. Instead of 'self.model.sample()', we do a forward pass
        that returns a float logits tensor.
        """
        if self.current_ids is None:
            logger.warning("verify_tokens called but no StartGeneration context found.")
            return {"target_probs": [0.0]*len(draft_tokens), "finished": True}

        target_probs = []
        temp_ids = self.current_ids.clone()  # do not mutate self.current_ids

        for token_id in draft_tokens:
            seq_len = temp_ids.shape[1]
            
            # 1) forward pass => raw logits over next token
            #    If your model is a Neuron-compiled *CausalLM*, you might call it like:
            #    outputs = self.model(input_ids=temp_ids) 
            #    then outputs.logits => shape [batch=1, seq_len, vocab_size].
            #    We'll do a single pass for the last position, but to keep it simple:
            
            outputs = self.model(temp_ids)  
            # If using TorchScript / neuron compiled, it might return just a Tensor 
            # shaped [batch=1, seq_len, vocab_size] or [seq_len, vocab_size].
            
            # 2) extract the logits of the *last position*
            if isinstance(outputs, torch.Tensor):
                # e.g. shape [seq_len, vocab_size] or [1, seq_len, vocab_size]
                if outputs.dim() == 3:
                    # [batch=1, seq_len, vocab_size]
                    logits_for_lastpos = outputs[0, -1, :]  # shape [vocab_size]
                else:
                    # [seq_len, vocab_size]
                    logits_for_lastpos = outputs[-1, :]     # shape [vocab_size]
            else:
                # If HF model => 'outputs.logits'
                logits_for_lastpos = outputs.logits[0, -1, :]  # shape [vocab_size]

            # cast to float just to be safe
            logits_for_lastpos = logits_for_lastpos.float()
            
            # 3) get probability of this 'token_id'
            probs = torch.softmax(logits_for_lastpos, dim=-1)
            p = float(probs[token_id].item())
            target_probs.append(p)

            # 4) append the draft token to temp_ids so the next iteration 
            #    sees that token in context
            token_tensor = torch.tensor([[token_id]], dtype=temp_ids.dtype)
            temp_ids = torch.cat([temp_ids, token_tensor], dim=1)

            # 5) check if we generated EOS
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                return {"target_probs": target_probs, "finished": True}

        # If never hit EOS, finished=False
        return {"target_probs": target_probs, "finished": False}
    
    def finalize_tokens(self, accepted_count):
        """
        After verifying a chunk, the client tells us how many draft tokens were accepted.
        We commit those to self.current_ids, and if there's a rejection, we generate a fallback token.
        Must return dict: {"final_token": <int>, "finished": bool}
        """
        if self.current_ids is None:
            logger.warning("finalize_tokens called but no StartGeneration context found.")
            return {"final_token": 0, "finished": True}

        # We'll assume the client has appended the 'accepted_count' tokens to self.current_ids 
        # or we can do it ourselves. For now, let's do a simple approach:
        # If accepted_count < #draft_tokens => that means one token was rejected and replaced.
        # We generate 1 big-model token to replace it.

        # For demonstration, let's just always generate 0 if everything was accepted (no replacement).
        # or 1 new token if there's a rejection. In practice, you'd keep track of how many tokens 
        # were originally proposed vs how many were accepted, etc.

        if accepted_count == 0:
            # Full rejection => generate 1 fallback token
            seq_len = self.current_ids.shape[1]
            output = self.model.sample(self.current_ids, sequence_length=seq_len + 1)
            new_token = int(output[0, -1].item())
            # Append it to self.current_ids
            self.current_ids = torch.cat([self.current_ids, torch.tensor([[new_token]])], dim=1)
            self.tokens_generated += 1
            # Check EOS
            finished = (new_token == self.eos_token_id) if (self.eos_token_id is not None) else False
            return {"final_token": new_token, "finished": finished}

        else:
            # Some tokens were accepted => client has appended them in its context
            # Possibly no new token needed if everything is accepted
            # But if partial acceptance => first 'accepted_count' were appended, one was rejected => 
            # we still do a fallback for the rejection. We'll assume the client always calls 
            # finalize_tokens exactly once, so let's do a new token if accepted_count < total draft?
            # We'll do a simplified version:

            # If partial acceptance => generate 1 fallback:
            # (This logic can be improved depending on your algorithm.)
            # For brevity, let's do it only if accepted_count < gamma. 
            # The client is the one that knows the original chunk size, though.

            # We'll just assume if accepted_count < gamma => we do 1 fallback:
            gamma = 4  # or wherever you store it
            if accepted_count < gamma:
                seq_len = self.current_ids.shape[1]
                output = self.model.sample(self.current_ids, sequence_length=seq_len + 1)
                new_token = int(output[0, -1].item())
                self.current_ids = torch.cat([self.current_ids, torch.tensor([[new_token]])], dim=1)
                self.tokens_generated += 1
                finished = (new_token == self.eos_token_id) if (self.eos_token_id is not None) else False
                return {"final_token": new_token, "finished": finished}
            else:
                # all tokens accepted => no fallback
                # accepted_count == gamma => fully accepted => generate “+1” big token
                seq_len = self.current_ids.shape[1]
                output = self.model.sample(self.current_ids, sequence_length=seq_len + 1)
                new_token = int(output[0, -1].item())
                self.current_ids = torch.cat([self.current_ids, torch.tensor([[new_token]])], dim=1)
                self.tokens_generated += 1
                finished = bool(self.eos_token_id and new_token == self.eos_token_id)
                return {"final_token": new_token, "finished": finished}

    # (You can refine the above logic however you prefer to manage accepted vs. rejected tokens.)

def run_server(model_path, port=50051, sequence_length=128, profile=False):
    """Launch the gRPC server hosting the target model for speculative decoding."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Target server starting on port {port} (sequence_length={sequence_length})")
    server.start()
    server.wait_for_termination()

# (Optional) A local run function for target-only generation, used by main.py for profiling single-model performance
def run_local(model_path, prompt="", max_new_tokens=50, sequence_length=128, profile=False):
    """Run the target model locally (without gRPC) to generate text for a prompt."""
    logger.info("Running target model locally for output verification/profiling.")
    model = model_loader.load_model(model_path, sequence_length=sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids if prompt else torch.zeros((1,0), dtype=torch.long)
    output_text = ""
    tokens_generated = 0
    start_time = time.time() if profile else None
    for i in range(max_new_tokens):
        try:
            output = model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        except Exception as e:
            logger.error(f"Target model generation failed: {e}")
            break
        token_id = int(output[0, -1]) if not isinstance(output, (list, tuple)) else int(output[0][-1])
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        print(f"Token {i+1}: {repr(token_text)}", flush=True)
        output_text += token_text
        # Append new token to input_ids for next iteration
        new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            logger.info("EOS token encountered, stopping generation.")
            break
    # Profiling logs
    end_time = time.time()
    if profile and start_time is not None:
        total_time = end_time - start_time
        throughput = tokens_generated / total_time if total_time > 0 else float('inf')
        logger.info(f"Target model generation completed in {total_time:.2f} seconds.")
        logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} tokens/sec")
        # Save performance metrics to CSV/JSON
        csv_file = f"performance_target_only_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        json_file = csv_file.replace(".csv", ".json")
        try:
            with open(csv_file, "w") as cf:
                cf.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                avg_time = (total_time / tokens_generated) if tokens_generated > 0 else 0.0
                cf.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},N/A\n")
            metrics = {
                "total_latency": total_time,
                "tokens_generated": tokens_generated,
                "throughput": throughput,
                "token_match_rate": None
            }
            with open(json_file, "w") as jf:
                import json
                json.dump(metrics, jf, indent=2)
            logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
        except Exception as e:
            logger.error(f"Failed to write performance metrics: {e}")
    print("\n=== Final Output ===\n" + (prompt + output_text))
    return output_text
