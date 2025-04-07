import logging
import grpc

import inference_pb2_grpc
from inference.model_loader import load_model
from inference.speculative import speculative_decode
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_client(draft_model_name: str, target_host: str = "localhost", port: int = 50051, 
               prompt: str = "", target_model_name: str = None, max_new_tokens: int = 50,
               sequence_length: int = 128):
    # Load the draft model (compile if necessary)
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length})...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)
    # Determine which tokenizer to use (use target model's tokenizer if provided for vocabulary consistency)
    tokenizer_source = target_model_name or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    # Connect to the target server via gRPC
    address = f"{target_host}:{port}"
    logger.info(f"Connecting to target server at {address}...")
    channel = grpc.insecure_channel(address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)

    if not prompt:
        logger.error("No prompt provided for draft client.")
        return

    logger.info(f"Starting speculative decoding for prompt: {prompt!r}")
    # Perform speculative decoding using the loaded draft model and the target stub
    generated_text = speculative_decode(draft_model, tokenizer, stub, prompt, max_new_tokens=max_new_tokens)
    logger.info("Speculative decoding completed.")

    full_output = prompt + generated_text
    print("\n=== Final Output ===\n" + full_output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Draft worker (client) for speculative decoding")
    parser.add_argument("--model", type=str, required=True, help="Draft model name or path")
    parser.add_argument("--target_host", type=str, default="localhost", help="Address of the target server host")
    parser.add_argument("--port", type=int, default=50051, help="Port of the target server")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to start generation")
    parser.add_argument("--target_model", type=str, 
                        help="Path or name of target model (for tokenizer compatibility; optional)")
    parser.add_argument("--max_new_tokens", type=int, default=50, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--sequence_length", type=int, default=128, 
                        help="Sequence length for model compilation (if the draft model is not yet compiled)")
    args = parser.parse_args()
    run_client(args.model, target_host=args.target_host, port=args.port, prompt=args.prompt, 
               target_model_name=args.target_model, max_new_tokens=args.max_new_tokens,
               sequence_length=args.sequence_length)
