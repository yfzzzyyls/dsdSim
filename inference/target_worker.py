import grpc
from concurrent import futures
import logging
from transformers_neuronx import LlamaForSampling
from transformers import AutoTokenizer, LlamaConfig
import inference_pb2
import inference_pb2_grpc
import os
from inference import model_loader  # Ensure this loads your precompiled model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        # Load the pre-compiled target model (compiled folder)
        # model_path should be the path to the pre-compiled target model folder,
        # for example: '/home/ubuntu/Choral-Spec/llama-3.2-3b-neuron-compiled-128'
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        logger.info("Target model and tokenizer loaded.")

    def StartGeneration(self, request, context):
        logger.info("StartGeneration called with prompt: %s", request.prompt)
        # (Optionally, you might initialize a generation session here)
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        logger.info("VerifyDraftTokens called with draft_tokens: %s", request.draft_tokens)
        # Placeholder: for now, assume all tokens match.
        return inference_pb2.VerifyResponse(
            all_matched=True,
            match_count=len(request.draft_tokens),
            correct_token=0,
            finished=False
        )

    def GenerateFull(self, request, context):
        logger.info("GenerateFull called with prompt: %s", request.prompt)
        # Use the target model to generate one token.
        input_ids = self.tokenizer(request.prompt, return_tensors="pt").input_ids
        # Generate one additional token.
        output = self.model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        # Handle both tensor and list outputs.
        if isinstance(output, (list, tuple)):
            token_id = int(output[0][-1])
        else:
            token_id = int(output[0, -1])
        token_text = self.tokenizer.decode([token_id]).strip()
        logger.info("GenerateFull returning token: %s", token_text)
        return inference_pb2.GenerateResponse(output_text=token_text)

def run_server(model_path, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SpeculativeServiceServicer(model_path)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Target server starting on port %d", port)
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Target worker for speculative decoding")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the pre-compiled target model (e.g., /home/ubuntu/Choral-Spec/llama-3.2-3b-neuron-compiled-128)")
    parser.add_argument("--port", type=int, default=50051, help="Port for gRPC server")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model inference")
    args = parser.parse_args()
    run_server(args.model, port=args.port)
