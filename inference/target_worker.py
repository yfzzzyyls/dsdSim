import logging
import os
import grpc
from concurrent import futures
from transformers import AutoTokenizer
from inference import model_loader
from grpc_comm import inference_pb2
from grpc_comm import inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        # Load the pre-compiled (or compile on the fly) target model
        # model_path should be the path to the model or compiled model directory.
        logger.info(f"Loading target model from '{model_path}' (sequence_length={sequence_length})...")
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        logger.info("Target model and tokenizer loaded.")

    def StartGeneration(self, request, context):
        logger.info("StartGeneration called with prompt: %s", request.prompt)
        # (Optionally initialize a generation session here - not used in this implementation)
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        logger.info("VerifyDraftTokens called with draft_tokens: %s", request.draft_tokens)
        # Placeholder implementation: always acknowledge all tokens as matched
        return inference_pb2.VerifyResponse(
            all_matched=True,
            match_count=len(request.draft_tokens),
            correct_token=0,
            finished=False
        )

    def GenerateFull(self, request, context):
        logger.info("GenerateFull called with prompt: %s", request.prompt)
        # Use the target model to generate one token continuation for the given prompt
        input_ids = self.tokenizer(request.prompt, return_tensors="pt").input_ids
        output = self.model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        # Extract the generated token ID
        if isinstance(output, (list, tuple)):
            token_id = int(output[0][-1])
        else:
            token_id = int(output[0, -1])
        token_text = self.tokenizer.decode([token_id]).strip()
        logger.info("GenerateFull returning token: %s", token_text)
        return inference_pb2.GenerateResponse(output_text=token_text)

def run_server(model_path, port=50051, sequence_length=128):
    # Start a gRPC server for the target model
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
    parser.add_argument("--model", type=str, required=True, help="Path to the target model (original or compiled directory)")
    parser.add_argument("--port", type=int, default=50051, help="Port for the gRPC server")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model inference")
    args = parser.parse_args()
    run_server(args.model, port=args.port, sequence_length=args.sequence_length)
