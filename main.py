import argparse
import logging
import os
# Enable Transformer optimizations *and* expose past_key_values to Python
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer"
os.environ["NEURON_RT_NUM_CORES"] = "2"

# Configure logging globally
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Choral-Spec main launcher")
    parser.add_argument("--role", choices=["target", "draft", "verify_target", "verify_draft"], required=True,
                        help=("Role to run: 'target' for target server, 'draft' for draft client, "
                              "'verify_target' to run the target model standalone, "
                              "'verify_draft' to run the draft model standalone"))
    parser.add_argument("--model", type=str,
                        help="Model path for the primary model (for target, draft, or verification roles)")
    parser.add_argument("--target_model", type=str,
                        help="Target model path (for draft role, used for tokenizer or loading compiled target model)")
    parser.add_argument("--draft_model", type=str,
                        help="Draft model path (alternative to --model for draft role)")
    parser.add_argument("--prompt", type=str,
                        help="Single prompt text for generation (used if --prompt_text is not provided)")
    parser.add_argument("--prompt_text", type=str,
                        help="Path to a text file containing multiple prompts (one per line) for concurrent/batch decoding.")
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
    parser.add_argument("--gamma", type=int, default=4,
                        help="Number of draft tokens to generate per verification step (speculative decoding chunk size).")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for draft sampling (default 0.9)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for draft sampling (default 1.0)")
    args = parser.parse_args()

    if args.role == "target":
        model_name = args.model or args.target_model
        if model_name is None:
            logger.error("Please specify --model (target model path) for target role")
            return
        # Launch the target model gRPC server
        from inference import target_worker
        target_worker.run_server(
            model_name,
            port=args.port,
            sequence_length=args.sequence_length,
            spec_length=args.gamma,
            profile=args.profile
        )

    elif args.role == "draft":
        # Running the draft side
        draft_model = args.model or args.draft_model
        if draft_model is None:
            logger.error("Please specify --model (draft model path) for draft role")
            return

        # If the user provided --prompt_text, we'll run concurrency with multiple prompts
        if args.prompt_text:
            # Batch mode: multiple prompts from file, each in a separate gRPC session
            from inference import draft_worker
            draft_model = load_model(
                draft_model_name,
                sequence_length=args.sequence_length,
                spec_length=args.gamma
            )
            draft_worker.run_batched_prompt_file(
                draft_model_name=draft_model,
                target_host=args.target_host,
                port=args.port,
                prompt_text_file=args.prompt_text,
                target_tokenizer=args.target_model,
                max_new_tokens=args.max_new_tokens,
                sequence_length=args.sequence_length,
                gamma=args.gamma,
                profile=args.profile,
                top_p=args.top_p,
                temperature=args.temperature
            )
        else:
            logger.error("No prompt text .txt file provided. Use --verify_target to run draft model independently.")
            return

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
        logger.error("Unknown role. Use --role target|draft|verify_target|verify_draft.")

if __name__ == "__main__":
    main()