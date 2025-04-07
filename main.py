import argparse
import logging
import os

# Ensure logging is configured for info level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Choral-Spec main launcher")
    parser.add_argument("--role", choices=["target", "draft"], required=True,
                        help="Role to run: 'target' for target server, 'draft' for draft client")
    parser.add_argument("--model", type=str, 
                        help="Model name or path (for target or draft role's primary model)")
    parser.add_argument("--target_model", type=str, 
                        help="Target model name or path (for draft role, used for tokenizer or loading compiled target model)")
    parser.add_argument("--draft_model", type=str, 
                        help="Draft model name or path (alternative to --model for draft role)")
    parser.add_argument("--prompt", type=str, 
                        help="Initial prompt text for generation (for draft role)")
    parser.add_argument("--port", type=int, default=50051, 
                        help="Port for gRPC server (target role) or client connection (draft role)")
    parser.add_argument("--target_host", type=str, default="localhost", 
                        help="Target host address (for draft role to connect to target server)")
    parser.add_argument("--sequence_length", type=int, default=128, 
                        help="Sequence length for model compilation (if not already compiled)")
    args = parser.parse_args()

    if args.role == "target":
        model_name = args.model or args.target_model
        if model_name is None:
            logger.error("Please specify --model (target model path) for target role")
            return
        # Run target server (will load or compile model as needed)
        from inference import target_worker
        target_worker.run_server(model_name, port=args.port, sequence_length=args.sequence_length)

    elif args.role == "draft":
        draft_model = args.model or args.draft_model
        target_model = args.target_model
        if draft_model is None:
            logger.error("Please specify --model (draft model path) for draft role")
            return
        prompt = args.prompt or ""
        # Run draft client (will load or compile draft model as needed, and connect to target)
        from inference import draft_worker
        draft_worker.run_client(draft_model, target_host=args.target_host, port=args.port, 
                                prompt=prompt, target_model_name=target_model, 
                                sequence_length=args.sequence_length)
    else:
        logger.error("Unknown role. Use --role target|draft.")

if __name__ == "__main__":
    main()
