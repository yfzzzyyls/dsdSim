import argparse
import logging
import os

# Ensure logging is configured for info level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Choral-Spec main launcher")
    parser.add_argument("--role", choices=["target", "draft", "compile"], required=True,
                        help="Role to run: 'target' for target server, 'draft' for draft client, 'compile' to compile a model")
    parser.add_argument("--model", type=str, help="Model name or path (for compile or single-role use)")
    parser.add_argument("--target_model", type=str, help="Target model name or path (for draft role, if different)")
    parser.add_argument("--draft_model", type=str, help="Draft model name or path (for draft role, if different)")
    parser.add_argument("--prompt", type=str, help="Initial prompt text for generation (for draft role)")
    parser.add_argument("--port", type=int, default=50051, help="Port for gRPC server (target role) or client to connect (draft role)")
    parser.add_argument("--target_host", type=str, default="localhost",
                        help="Target host address (draft role connects to target at this host)")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model compilation")
    args = parser.parse_args()

    if args.role == "compile":
        # Compile the specified model
        model_name = args.model
        seq_length = args.sequence_length
        if model_name is None:
            logger.error("Please specify --model for compile role")
            return
        # Import model_loader and compile the model
        import model_loader
        logger.info(f"Compiling model '{model_name}' with sequence length {seq_length}...")
        model_loader.compile_model(model_name, seq_length)
        logger.info("Model compilation completed.")
    elif args.role == "target":
        model_name = args.model or args.target_model
        if model_name is None:
            logger.error("Please specify --model (target model) for target role")
            return
        # Run target server
        import target_worker
        target_worker.run_server(model_name, port=args.port)
    elif args.role == "draft":
        # For draft, need both draft and target model names if provided separately
        draft_model = args.model or args.draft_model
        target_model = args.target_model
        if draft_model is None:
            logger.error("Please specify --model (draft model) for draft role")
            return
        prompt = args.prompt or ""
        # Run draft client
        import draft_worker
        draft_worker.run_client(draft_model, target_host=args.target_host, port=args.port, prompt=prompt, target_model_name=target_model)
    else:
        logger.error("Unknown role. Use --role target|draft|compile.")

if __name__ == "__main__":
    main()
