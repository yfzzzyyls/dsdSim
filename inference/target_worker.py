import logging
import os
import grpc
from concurrent import futures
from transformers import AutoTokenizer
from inference import model_loader
from grpc_comm import inference_pb2, inference_pb2_grpc
import time
import json
from datetime import datetime
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128, profile=False):
        logger.info(f"Loading target model from '{model_path}' (sequence_length={sequence_length})...")
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.profile = profile
        logger.info("Target model and tokenizer loaded.")

    def StartGeneration(self, request, context):
        logger.info("StartGeneration called with prompt: %s", request.prompt)
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        logger.info("VerifyDraftTokens called with draft_tokens: %s", request.draft_tokens)
        return inference_pb2.VerifyResponse(all_matched=True,
                                            match_count=len(request.draft_tokens),
                                            correct_token=0,
                                            finished=False)

    def GenerateFull(self, request, context):
        logger.info("GenerateFull called with prompt: %s", request.prompt)
        if self.profile:
            start_time = time.time()
        input_ids = self.tokenizer(request.prompt, return_tensors="pt").input_ids
        output = self.model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        if isinstance(output, (list, tuple)):
            token_id = int(output[0][-1])
        else:
            token_id = int(output[0, -1])
        token_text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        if self.profile:
            end_time = time.time()
            logger.info(f"GenerateFull returning token: '{token_text}' (took {end_time - start_time:.4f} sec)")
        else:
            logger.info(f"GenerateFull returning token: '{token_text}'")
        return inference_pb2.GenerateResponse(output_text=token_text)

def run_server(model_path, port=50051, sequence_length=128, profile=False):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length, profile=profile)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Target server starting on port %d", port)
    server.start()
    server.wait_for_termination()

def run_local(model_path, prompt: str, max_new_tokens=50, sequence_length=128, profile=False):
    logger.info(f"Running standalone target generation for prompt: {prompt!r}")
    model = model_loader.load_model(model_path, sequence_length=sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    token_times = [] if profile else None
    tokens_generated = 0
    if profile:
        start_time = time.time()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    output_text = ""
    for i in range(max_new_tokens):
        iter_start = time.time() if profile else None
        output = model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        token_id = int(output[0, -1]) if not isinstance(output, (list, tuple)) else int(output[0][-1])
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        print(f"Token {i+1}: {repr(token_text)}", flush=True)
        output_text += token_text
        new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1
        if profile:
            iter_end = time.time()
            token_times.append(iter_end - iter_start)
    if profile and tokens_generated > 0:
        total_time = time.time() - start_time
        avg_time = sum(token_times) / tokens_generated
        throughput = tokens_generated / total_time if total_time > 0 else float('inf')
        logger.info(f"Target-only generation completed in {total_time:.2f} seconds.")
        logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} tokens/sec, Avg token time: {avg_time:.4f} sec")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"performance_target_only_{timestamp}.csv"
        json_file = f"performance_target_only_{timestamp}.json"
        try:
            with open(csv_file, 'w', newline='') as f:
                f.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                f.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},N/A\n")
            metrics = {
                "total_latency": total_time,
                "tokens_generated": tokens_generated,
                "throughput": throughput,
                "per_token_times": token_times,
                "token_match_rate": None
            }
            with open(json_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
        except Exception as e:
            logger.error(f"Failed to save profiling data: {e}")
    full_output = prompt + output_text
    print("\n=== Final Output ===\n" + full_output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Target worker for speculative decoding")
    parser.add_argument("--model", type=str, required=True, help="Path to the target model (original or compiled folder)")
    parser.add_argument("--port", type=int, default=50051, help="Port for the gRPC server")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model inference")
    parser.add_argument("--prompt", type=str, help="Prompt text for standalone generation (use with --profile)")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum new tokens for standalone generation")
    parser.add_argument("--profile", action="store_true", help="Enable profiling of performance metrics")
    args = parser.parse_args()
    if args.prompt:
        run_local(args.model, prompt=args.prompt, max_new_tokens=args.max_new_tokens,
                  sequence_length=args.sequence_length, profile=args.profile)
    else:
        run_server(args.model, port=args.port, sequence_length=args.sequence_length, profile=args.profile)
