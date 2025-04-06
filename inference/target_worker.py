import grpc
from concurrent import futures
import logging
import torch
from transformers import AutoTokenizer
import inference_pb2
import inference_pb2_grpc
from . import model_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeculativeService(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_name: str, sequence_length: int = 128):
        # Load the target model (compiling if necessary)
        logger.info(f"Loading target model '{model_name}'...")
        self.model = model_loader.load_model(model_name, sequence_length=sequence_length)
        # Initialize tokenizer for the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Track the current context tokens for generation
        self.input_ids = None

    def NextToken(self, request, context):
        # If a prompt is provided in the request, treat this as a new generation or reset
        prompt = request.prompt
        if prompt:
            # Encode prompt and set as current context
            enc = self.tokenizer(prompt, return_tensors='pt')
            self.input_ids = enc.input_ids
            # First call: generate one token after the prompt
            seq = self.model.sample(self.input_ids, sequence_length=self.input_ids.shape[1] + 1)
        else:
            # Continue from existing context (generate next token)
            if self.input_ids is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No prompt provided and no existing context.")
                return inference_pb2.NextTokenResponse()
            seq = self.model.sample(self.input_ids, sequence_length=self.input_ids.shape[1] + 1)
        # Extract the newly generated token
        if isinstance(seq, (list, tuple)):
            token_id = int(seq[0][-1]) if isinstance(seq[0], (list, tuple)) else int(seq[0])
        else:
            token_id = int(seq[0, -1])
        token_text = self.tokenizer.decode([token_id])
        # Append the new token to context for future generation
        new_token_tensor = torch.tensor([[token_id]])
        self.input_ids = torch.cat([self.input_ids, new_token_tensor], dim=1)
        logger.info(f"Target model generated token: {repr(token_text)} (id {token_id})")
        return inference_pb2.NextTokenResponse(token_id=token_id, token_text=token_text)

def run_server(model_name: str, port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    speculative_service = SpeculativeService(model_name)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(speculative_service, server)
    server_address = f'[::]:{port}'
    server.add_insecure_port(server_address)
    server.start()
    logger.info(f"Target server is running on port {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Target worker (server) for speculative decoding")
    parser.add_argument("--model", type=str, required=True, help="Target model name or path")
    parser.add_argument("--port", type=int, default=50051, help="Port to run the gRPC server on")
    args = parser.parse_args()
    run_server(args.model, port=args.port)
