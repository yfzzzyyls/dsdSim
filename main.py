import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Distributed Speculative Decoding CLI")
    parser.add_argument("--role", type=str, required=True, choices=["compile", "target", "draft", "verify_target", "verify_draft"],
                        help="Role of this process: compile models, run target server, run draft client, or verify outputs.")
    parser.add_argument("--model", type=str, help="Path or name of the model to use (for compile/target roles, or for verify roles).")
    parser.add_argument("--draft_model", type=str, help="Path or name of the draft model (for draft role, if different from --model).")
    parser.add_argument("--target_model", type=str, help="Path or name of the target model (for draft role, used for tokenizer consistency).")
    parser.add_argument("--prompt", type=str, help="Initial prompt text for generation (for draft or verify roles).")
    parser.add_argument("--port", type=int, default=50051, help="Port for gRPC communication (target server listens, draft connects).")
    parser.add_argument("--target_host", type=str, default="localhost", help="Target host address (for draft role to connect).")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for model compilation or inference.")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of new tokens to generate (for draft and verify roles).")
    parser.add_argument("--draft_chunk_size", type=int, default=4, help="Number of tokens the draft model generates per speculative chunk.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling to log latency, throughput, and match rate to CSV/JSON.")
    args = parser.parse_args()

    if args.role == "compile":
        # Compile the specified model for AWS Trainium
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
        # Launch the target model gRPC server
        model_name = args.model or args.target_model
        if model_name is None:
            logger.error("Please specify --model (target model path) for target role")
            return
        from inference import target_worker
        logger.info(f"Starting target server with model '{model_name}' on port {args.port}...")
        target_worker.run_server(model_name, port=args.port, sequence_length=args.sequence_length)

    elif args.role == "draft":
        # Run the speculative decoding draft client, which connects to the target server
        draft_model = args.model or args.draft_model
        target_model = args.target_model  # used for tokenizer vocabulary if provided
        if draft_model is None:
            logger.error("Please specify --model (draft model path) for draft role")
            return
        prompt = args.prompt or ""
        if prompt == "":
            logger.error("Please specify --prompt for draft role (the prompt cannot be empty).")
            return
        from inference import draft_worker
        logger.info(f"Running draft client with draft model '{draft_model}', connecting to {args.target_host}:{args.port}")
        draft_worker.run_client(
            draft_model,
            target_host=args.target_host,
            port=args.port,
            prompt=prompt,
            target_model_name=target_model,
            max_new_tokens=args.max_tokens,
            sequence_length=args.sequence_length,
            draft_chunk_size=args.draft_chunk_size,
            profile=args.profile
        )

    elif args.role == "verify_target":
        # Run the target model locally (single-process) to generate text token-by-token for verification
        model_name = args.model
        if model_name is None:
            logger.error("Please specify --model for verify_target role")
            return
        prompt = args.prompt or ""
        from inference import verify
        logger.info(f"Verifying output using target model '{model_name}' locally...")
        verify.run_model(model_name, prompt=prompt, max_tokens=args.max_tokens, sequence_length=args.sequence_length)

    elif args.role == "verify_draft":
        # Run the draft model locally to generate text token-by-token for verification
        model_name = args.model
        if model_name is None:
            logger.error("Please specify --model for verify_draft role")
            return
        prompt = args.prompt or ""
        from inference import verify
        logger.info(f"Verifying output using draft model '{model_name}' locally...")
        verify.run_model(model_name, prompt=prompt, max_tokens=args.max_tokens, sequence_length=args.sequence_length)

    else:
        logger.error("Unknown role. Use --role compile|target|draft|verify_target|verify_draft.")

if __name__ == "__main__":
    main()
