import logging
from concurrent import futures

import grpc

import inference_pb2, inference_pb2_grpc  # gRPC generated classes
from inference import model_loader
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        # Load the target model (compile if necessary). model_path can be a base model path or a compiled model directory.
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        # Load tokenizer from the same path (compiled dir contains tokenizer files if model was compiled)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        logger.info("Target model and tokenizer loaded.")

    def StartGeneration(self, request, context):
        logger.info("StartGeneration called with prompt: %s", request.prompt)
        # (Optional) Initialize generation state if needed
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        logger.info("VerifyDraftTokens called with draft_tokens: %s", request.draft_tokens)
        # **Note**: This is a placeholder. In a full implementation, the target model 
        # would verify which of the draft tokens are correct. Here we assume all match.
        return inference_pb2.VerifyResponse(
            all_matched=True,
            match_count=len(request.draft_tokens),
            correct_token=0,
            finished=False
        )

    def GenerateFull(self, request, context):
        logger.info("GenerateFull called with prompt: %s", request.prompt)
        # Generate one token from the target model given the prompt
        input_ids = self.tokenizer(request.prompt, return_tensors="pt").input_ids
        # Use the model's sampling method to get the next token (sequence_length = current length + 1)
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
    # Create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Instantiate the service servicer, which will load/compile the model
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Target server starting on port %d", port)
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Target worker for speculative decoding")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the target model (base model path or compiled model directory)")
    parser.add_argument("--port", type=int, default=50051, help="Port for the target gRPC server")
    parser.add_argument("--sequence_length", type=int, default=128, 
                        help="Sequence length for model compilation (if the model is not yet compiled)")
    args = parser.parse_args()
    run_server(args.model, port=args.port, sequence_length=args.sequence_length)
