import argparse
import logging
import os

# Ensure logging is configured for info level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Choral-Spec main launcher")
    parser.add_argument(
        "--role", 
        choices=["target", "draft", "compile", "verify_target", "verify_draft"], 
        required=True, 
        help=("Role to run: 'target' for target server, 'draft' for draft client, "
              "'compile' to compile a model, 'verify_target' to run the target model standalone, "
              "'verify_draft' to run the draft model standalone")
    )
    parser.add_argument("--model", type=str, help="Model name or path (for compile or single-model verification roles, or as default model path for target/draft roles)")
    parser.add_argument("--target_model", type=str, help="Target model name or path (for draft role, if different from --model)")
    parser.add_argument("--draft_model", type=str, help="Draft model name or path (for draft role, if different from --model)")
    parser.add_argument("--prompt", type=str, help="Initial prompt text for generation (for draft or verification roles)")
    parser.add_argument("--port", type=int, default=50051, help="Port for gRPC server (target role) or for client to connect (draft role)")
    parser.add_argument("--target_host", type=str, default="localhost", help="Target host address (draft role connects to target at this host)")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model compilation or inference")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of new tokens to generate (for draft and verification roles)")
    args = parser.parse_args()

    if args.role == "compile":
        # Compile the specified model
        model_name = args.model
        seq_length = args.sequence_length
        if model_name is None:
            logger.error("Please specify --model for compile role")
            return
        from inference import model_loader
        logger.info(f"Compiling model '{model_name}' with sequence length {seq_length}...")
        model_loader.compile_model(model_name, sequence_length=seq_length)
        logger.info("Model compilation completed.")
    elif args.role == "target":
        model_name = args.model or args.target_model
        if model_name is None:
            logger.error("Please specify --model (target model path) for target role")
            return
        from inference import target_worker
        # Start target model gRPC server (will load/compile model if needed)
        target_worker.run_server(model_name, port=args.port, sequence_length=args.sequence_length)
    elif args.role == "draft":
        # Draft client requires draft model and target model (target model mainly for tokenizer vocabulary)
        draft_model = args.model or args.draft_model
        target_model = args.target_model
        if draft_model is None:
            logger.error("Please specify --model (draft model path) for draft role")
            return
        prompt = args.prompt or ""
        from inference import draft_worker
        # Run speculative decoding draft client (connects to target server)
        draft_worker.run_client(
            draft_model, 
            target_host=args.target_host, 
            port=args.port, 
            prompt=prompt, 
            target_model_name=target_model, 
            max_new_tokens=args.max_tokens, 
            sequence_length=args.sequence_length
        )
    elif args.role == "verify_target":
        model_name = args.model
        if model_name is None:
            logger.error("Please specify --model for verify_target role")
            return
        prompt = args.prompt or ""
        from inference import verify
        # Run the target model locally to generate text token-by-token
        verify.run_model(model_name, prompt=prompt, max_tokens=args.max_tokens, sequence_length=args.sequence_length)
    elif args.role == "verify_draft":
        model_name = args.model
        if model_name is None:
            logger.error("Please specify --model for verify_draft role")
            return
        prompt = args.prompt or ""
        from inference import verify
        # Run the draft model locally to generate text token-by-token
        verify.run_model(model_name, prompt=prompt, max_tokens=args.max_tokens, sequence_length=args.sequence_length)
    else:
        logger.error("Unknown role. Use --role target|draft|compile|verify_target|verify_draft.")

if __name__ == "__main__":
    main()
