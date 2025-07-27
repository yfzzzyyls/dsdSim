import argparse
import logging
import os
from inference.model_loader import load_model
from inference.draft_worker import save_perf_stats
import time
import torch
from transformers import AutoTokenizer
import json
from datetime import datetime
# Enable Transformer optimizations *and* expose past_key_values to Python
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer"
os.environ["NEURON_RT_NUM_CORES"] = "2"

# -----------------------------------------------------------------------------
# Configure root logging BEFORE importing heavy libraries that may configure
# logging first.  Use `force=True` so our settings override any prior config
# that third‑party packages may have installed (e.g. transformers, neuronx).
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,         # override any existing handlers
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="choral-spec main launcher")
    parser.add_argument("--role", choices=["target", "draft", "verify_target", "verify_draft", "fused_target", "fused_client"], required=True,
                        help=("Role to run: 'target' for target server, 'draft' for draft client, "
                              "'verify_target' to run the target model standalone, "
                              "'verify_draft' to run the draft model standalone, "
                              "'fused_target' for fused speculative server, "
                              "'fused_client' for fused speculative client"))
    parser.add_argument("--model", type=str,
                        help="Model path for the primary model (for target, draft, or verification roles)")
    parser.add_argument("--target_model", type=str,
                        help="Target model path (for draft role, used for tokenizer or loading compiled target model)")
    parser.add_argument("--draft_model", type=str,
                        help="Draft model path (alternative to --model for draft role)")
    parser.add_argument("--prompt_text", type=str,
                        help="Path to a text file containing multiple prompts (one per line) for concurrent/batch decoding.")
    parser.add_argument("--port", type=int, default=50051,
                        help="Port for gRPC server (target role) or client connection (draft role)")
    parser.add_argument("--target_host", type=str, default="localhost",
                        help="Target server host address (for draft role to connect to target server)")
    parser.add_argument("--sequence_length", type=int, default=128,
                        help="Sequence length for model compilation (if not already compiled)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum number of new tokens to generate (None or unspecified: generate up to sequence length)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance profiling (latency/throughput metrics)")
    parser.add_argument("--no_target", action="store_true",
                        help="(Draft role only) Run draft model without target (standalone draft mode)")
    parser.add_argument("--gamma", type=int, default=4,
                        help="Number of draft tokens to generate per verification step (speculative decoding chunk size).")
    parser.add_argument("--speculation_length", type=int, default=5,
                        help="Number of tokens to speculate at once for fused speculative decoding.")
    parser.add_argument("--tp_degree", type=int, default=4,
                        help="Tensor parallelism degree for model compilation.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for draft sampling (default 0.9)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for draft sampling (default 1.0)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size used when compiling models (default: 1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose DEBUG logging (prints logger.debug lines)")
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Bump verbosity if --debug was requested.  This promotes *all*
    # loggers (root + module children) from INFO → DEBUG at runtime.
    # -----------------------------------------------------------------
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)   # root logger
        # Ensure the module‑level logger defined above follows suit
        logger.setLevel(logging.DEBUG)
        logger.debug("Global DEBUG logging enabled via --debug flag")


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
            profile=args.profile,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch,
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
            draft_worker.run_client(
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
                temperature=args.temperature,
                batch_size=args.batch,
            )
        else:
            logger.error("No prompt text .txt file provided. Use --verify_target to run draft model independently.")
            return

    elif args.role == "verify_target":
        model_name = args.model
        if model_name is None:
            logger.error("Please specify --model for verify_target role")
            return
        if not args.prompt_text:
            logger.error("--prompt_text must be provided for verify_target role (file with prompt).")
            return
        try:
            with open(args.prompt_text, 'r', encoding='utf-8') as fp:
                prompt_text = fp.readline().rstrip('\n')
        except Exception as e:
            logger.error(f"Failed to read prompt_text file: {e}")
            return
        res = run_model(
            model_name,
            prompt=prompt_text,
            max_tokens=args.max_new_tokens,
            sequence_length=args.sequence_length,
            role="target",
            profile=args.profile,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch,
        )
        if isinstance(res, tuple) and len(res) == 2:
            output_text, perf_stats = res
        else:
            output_text, perf_stats = res, None
        if args.profile and perf_stats:
            save_perf_stats(perf_stats, file_prefix="performance_verify_target")

    elif args.role == "verify_draft":
        model_name = args.model
        if model_name is None:
            logger.error("Please specify --model for verify_draft role")
            return
        if not args.prompt_text:
            logger.error("--prompt_text must be provided for verify_draft role (file with prompt).")
            return
        try:
            with open(args.prompt_text, 'r', encoding='utf-8') as fp:
                prompt_text = fp.readline().rstrip('\n')
        except Exception as e:
            logger.error(f"Failed to read prompt_text file: {e}")
            return
        res = run_model(
            model_name,
            prompt=prompt_text,
            max_tokens=args.max_new_tokens,
            sequence_length=args.sequence_length,
            role="draft",
            profile=args.profile,
            batch_size=args.batch,
        )
        if isinstance(res, tuple) and len(res) == 2:
            output_text, perf_stats = res
        else:
            output_text, perf_stats = res, None
        if args.profile and perf_stats:
            save_perf_stats(perf_stats, file_prefix="performance_verify_draft")
            
    elif args.role == "fused_target":
        # Running the fused speculative server
        draft_model = args.draft_model or args.model
        target_model = args.target_model
        if draft_model is None or target_model is None:
            logger.error("Please specify both --draft_model and --target_model for fused_target role")
            return
        
        from inference import fused_target_worker
        fused_target_worker.run_fused_server(
            draft_model_path=draft_model,
            target_model_path=target_model,
            port=args.port,
            sequence_length=args.sequence_length,
            speculation_length=args.speculation_length,
            batch_size=args.batch,
            profile=args.profile,
            tp_degree=args.tp_degree
        )
        
    elif args.role == "fused_client":
        # Running the fused speculative client
        if not args.prompt_text:
            logger.error("Please specify --prompt_text for fused_client role")
            return
            
        from inference import fused_draft_worker
        fused_draft_worker.run_fused_client(
            target_host=args.target_host,
            port=args.port,
            prompt_text_file=args.prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            speculation_length=args.speculation_length,
            profile=args.profile,
            sequence_length=args.sequence_length
        )
    else:
        logger.error("Unknown role. Use --role target|draft|verify_target|verify_draft|fused_target|fused_client.")


def run_model(
    model_name: str,
    prompt: str,
    max_tokens: int = 50,
    sequence_length: int = 128,
    role: str = "target",
    profile: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    batch_size: int = 1,
):
    logger.info(f"Loading {role} model '{model_name}' for standalone generation (sequence_length={sequence_length})...")
    HFmodel = load_model(model_name, sequence_length=sequence_length, batch_size=batch_size)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if HFmodel is None:
        logger.error("Failed to load the model for verification.")
        return  # Exit early if model could not be loaded
    if not prompt:
        logger.error("No prompt provided for generation.")
        return

    logger.info(f"Starting generation for prompt: {prompt!r}")

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    start = time.time()
    outs = HFmodel.adapter.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    latency = time.time() - start

    # --- post‑process generation ---
    tokens_generated = outs.shape[1] - input_ids.shape[1]
    throughput = tokens_generated / latency if latency > 0 else 0.0
    output_text = tokenizer.decode(
        outs[0, input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    logger.info(f"{role.capitalize()} model generation completed in {latency:.2f} seconds.")
    logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} t/s")
    if profile:
        csv_file = f"performance_{role}_only.csv"
        header_needed = not os.path.exists(csv_file)
        try:
            with open(csv_file, 'a', newline='') as f:
                if header_needed:
                    f.write("total_latency,tokens_generated,throughput,avg_token_time,prompt_text\n")
                avg_time = (latency / tokens_generated) if tokens_generated > 0 else 0.0
                # Escape any internal commas in the prompt to keep CSV shape
                safe_prompt = prompt.replace(',', ';')
                f.write(f"{latency:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},{safe_prompt}\n")
            logger.info(f"Performance metrics saved to {csv_file}")
        except Exception as e:
            logger.error(f"Failed to save profiling data: {e}")

    full_output = prompt + output_text
    print("\n=== Final Output ===\n" + full_output)
    return full_output


if __name__ == "__main__":
    main()