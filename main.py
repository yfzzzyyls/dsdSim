import argparse
import logging
import os

# Configure logging globally
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Choral-Spec main launcher")
    parser.add_argument("--role", choices=["target", "draft", "compile", "verify_target", "verify_draft"], required=True,
                        help=("Role to run: 'target' for target server, 'draft' for draft client, "
                              "'compile' to compile a model, 'verify_target' to run the target model standalone, "
                              "'verify_draft' to run the draft model standalone"))
    parser.add_argument("--model", type=str,
                        help="Model path for the primary model (for target, draft, or verification roles)")
    parser.add_argument("--target_model", type=str,
                        help="Target model path (for draft role, used for tokenizer or loading compiled target model)")
    parser.add_argument("--draft_model", type=str,
                        help="Draft model path (alternative to --model for draft role)")
    parser.add_argument("--prompt", type=str,
                        help="Initial prompt text for generation (for draft or verification roles)")
    parser.add_argument("--port", type=int, default=50051,
                        help="Port for gRPC server (target role) or client connection (draft role)")
    parser.add_argument("--target_host", type=str, default="localhost",
                        help="Target server host address (for draft role to connect to target server)")
    parser.add_argument("--sequence_length", type=int, default=128,
                        help="Sequence length for model compilation (if not already compiled)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance profiling (latency/throughput metrics)")
    parser.add_argument("--no_target", action="store_true",
                        help="(Draft role only) Run draft model without target (standalone draft mode)")
    parser.add_argument("--draft_chunk_size", type=int, default=4,
                        help="Number of tokens per speculative draft step (chunk size for draft model)")
    args = parser.parse_args()

    if args.role == "compile":
        # Compile the specified model to Neuron
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
        # If a prompt is provided with --profile, run a local generation for profiling (target-only mode)
        if args.profile and args.prompt:
            from inference import target_worker
            logger.info("Profiling enabled for standalone target generation.")
            target_worker.run_local(model_name, prompt=args.prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    sequence_length=args.sequence_length,
                                    profile=True)
        else:
            # Launch the target model gRPC server
            from inference import target_worker
            target_worker.run_server(model_name, port=args.port,
                                     sequence_length=args.sequence_length,
                                     profile=args.profile)

    elif args.role == "draft":
        draft_model = args.model or args.draft_model
        target_model = args.target_model  # Path to target model (for tokenizer consistency)
        if draft_model is None:
            logger.error("Please specify --model (draft model path) for draft role")
            return
        prompt_text = args.prompt or ""
        from inference import draft_worker
        if args.no_target:
            # Run draft model standalone (no target server)
            if args.profile:
                logger.info("Profiling enabled for standalone draft generation.")
            else:
                logger.info("Standalone draft generation (no target server).")
            draft_worker.run_client(draft_model, target_host=None, port=args.port,
                                     prompt=prompt_text, target_model_name=target_model,
                                     max_new_tokens=args.max_new_tokens, sequence_length=args.sequence_length,
                                     draft_chunk_size=args.draft_chunk_size,
                                     profile=args.profile, no_target=True)
        else:
            # Run speculative decoding with a target server
            if args.profile:
                logger.info("Profiling enabled for speculative decoding (draft client).")
            draft_worker.run_client(draft_model, target_host=args.target_host, port=args.port,
                                     prompt=prompt_text, target_model_name=target_model,
                                     max_new_tokens=args.max_new_tokens, sequence_length=args.sequence_length,
                                     draft_chunk_size=args.draft_chunk_size,
                                     profile=args.profile, no_target=False)

    elif args.role == "verify_target":
        model_name = args.model
        if model_name is None:
            logger.error("Please specify --model for verify_target role")
            return
        prompt_text = args.prompt or ""
        from inference import verify
        verify.run_model(model_name, prompt=prompt_text,
                         max_tokens=args.max_new_tokens,
                         sequence_length=args.sequence_length,
                         role="target", profile=args.profile)

    elif args.role == "verify_draft":
        model_name = args.model
        if model_name is None:
            logger.error("Please specify --model for verify_draft role")
            return
        prompt_text = args.prompt or ""
        from inference import verify
        verify.run_model(model_name, prompt=prompt_text,
                         max_tokens=args.max_new_tokens,
                         sequence_length=args.sequence_length,
                         role="draft", profile=args.profile)
    else:
        logger.error("Unknown role. Use --role target|draft|compile|verify_target|verify_draft.")

if __name__ == "__main__":
    main()
