import argparse
import time

# Local imports
from grpc_comm import grpc_server, grpc_client
from inference import model_loader, speculative, verify
from inference import target_worker

def main():
    parser = argparse.ArgumentParser(description="Launch a speculative decoding node (draft or target).")
    parser.add_argument("--role", choices=["draft", "target"], required=True,
                        help="Node role: 'draft' or 'target'.")
    parser.add_argument("--peer-ip", type=str,
                        help="IP address of peer node (required for draft role).")
    parser.add_argument("--port", type=int, default=50051,
                        help="gRPC server port (both nodes must use same).")
    parser.add_argument("--model-id", type=str, default=None,
                        help="HuggingFace model ID or local path for this node's model.")
    parser.add_argument("--prompt", type=str, default="Once upon a time,",
                        help="Prompt text for generation (draft only).")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max new tokens to generate for completion.")
    parser.add_argument("--num-speculative", type=int, default=3,
                        help="Number of tokens the draft model speculates ahead per batch.")
    parser.add_argument("--verify", action="store_true",
                        help="Enable functional correctness verification.")
    parser.add_argument("--perf-test", action="store_true",
                        help="Enable performance timing comparison.")
    args = parser.parse_args()

    if args.role == "target":
        # Target node: load the large model and start gRPC server
        model_id = args.model_id or "meta-llama/Meta-Llama-3-8B"
        print(f"[Target] Loading model '{model_id}' for validation...")
        model, tokenizer = model_loader.load_model(model_id)
        target_logic = target_worker.TargetWorker(model, tokenizer)
        server = grpc_server.start_server(target_logic, port=args.port)
        print(f"[Target] gRPC server is running on port {args.port}, awaiting draft connection...")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            print("[Target] Server shutting down.")
    else:
        # Draft node: load the small model, connect to target, run speculative decoding
        if not args.peer_ip:
            raise ValueError("Draft node requires --peer-ip of the target node.")

        model_id = args.model_id or "meta-llama/Meta-Llama-3-1B"
        print(f"[Draft] Loading model '{model_id}' for drafting...")
        model, tokenizer = model_loader.load_model(model_id)
        target_address = f"{args.peer_ip}:{args.port}"
        print(f"[Draft] Connecting to target at {target_address}...")
        stub = grpc_client.create_stub(target_address)

        print("[Draft] Starting speculative decoding...")
        start_time = time.time()
        output_tokens = speculative.run_speculative_decoding(
            draft_model=model,
            tokenizer=tokenizer,
            stub=stub,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            num_speculative=args.num_speculative
        )
        spec_time = time.time() - start_time
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"[Draft] Speculative decoding output: {output_text}")

        # Verification/perf
        if args.verify or args.perf_test:
            verify.check_and_compare(
                stub=stub,
                tokenizer=tokenizer,
                prompt=args.prompt,
                spec_output_tokens=output_tokens,
                max_tokens=args.max_tokens,
                measure_perf=args.perf_test
            )

        print(f"[Draft] Finished. Total speculative decode time: {spec_time:.2f} s.")

if __name__ == "__main__":
    main()