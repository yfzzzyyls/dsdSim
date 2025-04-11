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
        try:
            self.eos_token_id = self.tokenizer.eos_token_id
        except:
            self.eos_token_id = None

    def StartGeneration(self, request, context):
        prompt_text = request.prompt or ""
        max_tokens = request.max_new_tokens
        self.gamma = request.gamma  # store from the client

        logger.info(f"StartGeneration called with prompt=\"{prompt_text}\", max_new_tokens={max_tokens}, gamma={self.gamma}")
        self.current_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        self.max_tokens = max_tokens
        self.tokens_generated = 0
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        draft_tokens = list(request.draft_tokens)
        logger.info(f"VerifyDraftTokens called with draft_tokens={draft_tokens}")
        result = self.verify_tokens(draft_tokens)
        target_probs = result["target_probs"]
        finished = result.get("finished", False)
        return inference_pb2.VerifyResponse(target_probs=target_probs, finished=finished)

    def FinalizeTokens(self, request, context):
        accepted_count = request.accepted_count
        draft_chunk_size = request.draft_chunk_size
        logger.info(f"FinalizeTokens called with accepted_count={accepted_count}, chunk_size={draft_chunk_size}")

        # FIX: Now we call the method that is inside this class:
        result = self.finalize_tokens(accepted_count, draft_chunk_size)
        final_token = result.get("final_token", 0)
        finished = result.get("finished", False)
        return inference_pb2.FinalizeResponse(final_token=final_token, finished=finished)

    def GenerateFull(self, request, context):
        prompt_text = request.prompt or ""
        max_tokens = request.max_new_tokens
        logger.info(f"GenerateFull called with prompt: \"{prompt_text}\", max_new_tokens: {max_tokens}")
        self.current_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        self.tokens_generated = 0
        output = self.model.sample(self.current_ids, sequence_length=self.current_ids.shape[1] + max_tokens)
        generated_ids = output[0][self.current_ids.shape[1]:] if isinstance(output, (list, tuple)) else output[0][self.current_ids.shape[1]:]
        generated_ids = [int(t) for t in generated_ids]
        logger.info(f"Target model one-shot generated tokens: {generated_ids}")

        gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return inference_pb2.GenerateResponse(output_text=gen_text)

    def verify_tokens(self, draft_tokens):
        if self.current_ids is None:
            logger.warning("verify_tokens called but no StartGeneration context found.")
            return {"target_probs": [0.0]*len(draft_tokens), "finished": True}

        target_probs = []
        temp_ids = self.current_ids.clone()
        for token_id in draft_tokens:
            seq_len = temp_ids.shape[1]
            outputs = self.model(temp_ids)  
            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 3:
                    logits_for_lastpos = outputs[0, -1, :]
                else:
                    logits_for_lastpos = outputs[-1, :]
            else:
                logits_for_lastpos = outputs.logits[0, -1, :]
            logits_for_lastpos = logits_for_lastpos.float()
            probs = torch.softmax(logits_for_lastpos, dim=-1)
            p = float(probs[token_id].item())
            target_probs.append(p)
            token_tensor = torch.tensor([[token_id]], dtype=temp_ids.dtype)
            temp_ids = torch.cat([temp_ids, token_tensor], dim=1)
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                return {"target_probs": target_probs, "finished": True}
        return {"target_probs": target_probs, "finished": False}

    # FIX: Put finalize_tokens INSIDE the class:
    def finalize_tokens(self, accepted_count, draft_chunk_size):
        """
        If accepted_count < draft_chunk_size => partial acceptance => fallback token
        If accepted_count == 0 => full reject => fallback
        If accepted_count == draft_chunk_size => fully accepted => +1 big token
        """
        if self.current_ids is None:
            logger.warning("No active generation context.")
            return {"final_token": 0, "finished": True}

        if accepted_count == 0:
            # full rejection => fallback
            seq_len = self.current_ids.shape[1]
            output = self.model.sample(self.current_ids, sequence_length=seq_len + 1)
            new_token = int(output[0, -1].item())
            self.current_ids = torch.cat([self.current_ids, torch.tensor([[new_token]])], dim=1)
            self.tokens_generated += 1
            finished = bool(self.eos_token_id and new_token == self.eos_token_id)
            return {"final_token": new_token, "finished": finished}

        if accepted_count < draft_chunk_size:
            # partial => fallback
            seq_len = self.current_ids.shape[1]
            output = self.model.sample(self.current_ids, sequence_length=seq_len + 1)
            new_token = int(output[0, -1].item())
            self.current_ids = torch.cat([self.current_ids, torch.tensor([[new_token]])], dim=1)
            self.tokens_generated += 1
            finished = bool(self.eos_token_id and new_token == self.eos_token_id)
            return {"final_token": new_token, "finished": finished}
        else:
            # fully accepted => +1 big token
            seq_len = self.current_ids.shape[1]
            output = self.model.sample(self.current_ids, sequence_length=seq_len + 1)
            new_token = int(output[0, -1].item())
            self.current_ids = torch.cat([self.current_ids, torch.tensor([[new_token]])], dim=1)
            self.tokens_generated += 1
            finished = bool(self.eos_token_id and new_token == self.eos_token_id)
            return {"final_token": new_token, "finished": finished}

def run_server(model_path, port=50051, sequence_length=128, profile=False):
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
