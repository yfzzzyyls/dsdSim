#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torch_neuronx  # AWS Neuron SDK (required for model compilation & execution)
import grpc

# Import Hugging Face Transformers (for model and tokenizer)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the project modules for model loading and gRPC communication
from inference import model_loader
import inference_pb2
import inference_pb2_grpc

# Default maximum sequence length for Neuron compilation (context length)
DEFAULT_MAX_SEQ_LEN = 256

def compile_model_to_neuron(model_id: str, role: str, output_path: str = None, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
    """
    Compile a Hugging Face Transformers causal LM model to an AWS Neuron optimized TorchScript.
    The compiled model is saved to disk in the project root.
    """
    # Determine output file path in project root
    base_name = os.path.basename(os.path.normpath(model_id))
    if output_path is None:
        # Include role in filename to distinguish target vs draft models if needed
        suffix = f"_{role}" if role is not None else ""
        output_file = f"{base_name}{suffix}_neuron_bf16_{max_seq_len}.pt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ensure directory exists

    print(f"[{role.capitalize()}] Compiling model '{model_id}' to Neuron (bf16, seq_len={max_seq_len})...")
    # Load the model in bf16 precision
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.eval()
    # Disable features not needed for tracing (to simplify model graph)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "return_dict"):
        model.config.return_dict = False
    if hasattr(model.config, "output_hidden_states"):
        model.config.output_hidden_states = False
    if hasattr(model.config, "output_attentions"):
        model.config.output_attentions = False

    # Prepare dummy inputs for tracing (batch 1, seq_len = max_seq_len)
    batch_size = 1
    seq_len = max_seq_len
    example_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    example_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    # Set up Neuron compiler arguments (targeting Trainium)
    compiler_args = ["--target", "trn1", "--fp32-cast", "matmult", "--fast-math", "no-fast-relayout"]

    # Perform the trace/compile
    print(f"[{role.capitalize()}] Starting compilation... this may take several minutes.")
    traced_model = torch_neuronx.trace(model, (example_input_ids, example_attention_mask),
                                       compiler_args=compiler_args, timeout=900,  # up to 15 minutes
                                       cpu_backend=True)
    # Save the compiled model
    torch.jit.save(traced_model, output_path)
    print(f"[{role.capitalize()}] Model compiled and saved to: {output_path}")
    return output_path

def run_target(model_id: str, max_tokens: int, port: int):
    """
    Launch the target node: load the compiled model and start the gRPC server.
    """
    # Load the (compiled) model using model_loader
    model, compiled_flag = model_loader.load_model(model_id)
    if not compiled_flag:
        print(f"[Target] Warning: running uncompiled model for target (performance will be degraded).")

    # Create gRPC server and add the SpeculativeService servicer
    server = grpc.server(thread_pool=None)  # use default thread pool
    # Initialize the servicer with the loaded model
    class TargetServicer(inference_pb2_grpc.SpeculativeServiceServicer):
        def __init__(self, model):
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Ensure pad token exists for encoding prompt
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.current_ids = []  # will hold token ids for current sequence (prompt + verified tokens)

        def StartGeneration(self, request, context):
            # Receive prompt text, tokenize and store initial context
            prompt_text = request.prompt
            # Encode prompt to token IDs
            inputs = self.tokenizer(prompt_text, return_tensors='pt')
            prompt_ids = inputs["input_ids"][0].tolist()
            self.current_ids = prompt_ids[:]  # copy to current sequence state
            # No explicit output needed, just acknowledge
            return inference_pb2.StartResponse()

        def VerifyDraftTokens(self, request, context):
            draft_tokens = list(request.draft_tokens)
            all_matched = True
            match_count = 0
            correct_token = None
            finished = False

            for i, draft_token in enumerate(draft_tokens):
                # Prepare input tensor for target model: current_ids (context so far)
                input_ids = torch.tensor([self.current_ids], dtype=torch.long)
                attn_mask = torch.ones_like(input_ids)
                # If model is compiled, pad input to max sequence length
                orig_len = input_ids.shape[1]
                if compiled_flag:
                    max_len = model_loader.DEFAULT_MAX_SEQ_LEN
                    if orig_len > max_len:
                        # This should not happen if inputs are kept <= compiled context length
                        raise RuntimeError(f"Input length {orig_len} exceeds compiled max length {max_len}.")
                    if orig_len < max_len:
                        # Pad to max_len
                        pad_len = max_len - orig_len
                        pad_ids = torch.zeros((1, pad_len), dtype=torch.long)
                        pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
                        input_ids = torch.cat([input_ids, pad_ids], dim=1)
                        attn_mask = torch.cat([attn_mask, pad_mask], dim=1)
                # Run the model to get next-token logits
                outputs = self.model(input_ids, attention_mask=attn_mask)
                # Handle model outputs (compiled model returns tuple, non-compiled returns ModelOutput)
                if compiled_flag:
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # tuple -> tensor
                    logits = outputs[:, :orig_len, :]
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                # Get predicted token from target model (greedy)
                target_token_id = int(torch.argmax(logits[:, -1, :], dim=1).item())
                # Compare with draft token
                if target_token_id == draft_token:
                    # Token matches
                    match_count += 1
                    self.current_ids.append(draft_token)  # accept draft token
                    # If token is EOS, end generation
                    if target_token_id == self.tokenizer.eos_token_id:
                        finished = True
                        all_matched = True
                        # No need to continue verifying remaining tokens in this batch
                        break
                    # Continue to next token in draft batch
                    continue
                else:
                    # Mismatch: use target's token
                    all_matched = False
                    correct_token = target_token_id
                    self.current_ids.append(target_token_id)  # append the correct token from target
                    # If the correct token is EOS, mark finished
                    if target_token_id == self.tokenizer.eos_token_id:
                        finished = True
                    # Stop at the first mismatch
                    break

            # If all draft tokens were matched and none produced EOS, then all_matched remains True.
            # Build response
            response = inference_pb2.VerifyResponse(
                all_matched=all_matched,
                match_count=match_count,
                correct_token=(correct_token if correct_token is not None else 0),
                finished=finished
            )
            return response

        def GenerateFull(self, request, context):
            # (This method might be used for evaluation: generate completion with target-only.)
            prompt_text = request.prompt
            max_new = request.max_new_tokens
            tokenizer = self.tokenizer
            inputs = tokenizer(prompt_text, return_tensors='pt')
            input_ids = inputs["input_ids"][0].tolist()
            output_ids = input_ids[:]  # start with prompt tokens
            # Greedily generate max_new tokens
            for _ in range(max_new):
                ids_tensor = torch.tensor([output_ids], dtype=torch.long)
                attn_mask = torch.ones_like(ids_tensor)
                if compiled_flag:
                    # Pad to max length for compiled model
                    orig_len = ids_tensor.shape[1]
                    max_len = model_loader.DEFAULT_MAX_SEQ_LEN
                    if orig_len < max_len:
                        pad_len = max_len - orig_len
                        pad_ids = torch.zeros((1, pad_len), dtype=torch.long)
                        pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
                        ids_tensor = torch.cat([ids_tensor, pad_ids], dim=1)
                        attn_mask = torch.cat([attn_mask, pad_mask], dim=1)
                outputs = self.model(ids_tensor, attention_mask=attn_mask)
                if compiled_flag:
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    logits = outputs[:, :ids_tensor.shape[1], :]
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                next_id = int(torch.argmax(logits[:, -1, :], dim=1).item())
                output_ids.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    break
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            return inference_pb2.GenerateResponse(output_text=output_text)

    # Register the servicer and start the server
    servicer = TargetServicer(model)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"[Target] gRPC server is running on port {port}. Waiting for requests...")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Target] Server stopping...")
        server.stop(0)

def run_draft(model_id: str, peer_ip: str, port: int, max_tokens: int, num_speculative: int, prompt: str):
    """
    Launch the draft node: connect to target gRPC server, perform speculative decoding.
    """
    # Load draft model (use compiled if available for speed)
    model, compiled_flag = model_loader.load_model(model_id)
    if not compiled_flag:
        print(f"[Draft] Warning: running uncompiled model for draft (performance will be degraded).")

    # Load tokenizer for encoding prompt and decoding output
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    prompt_ids = inputs["input_ids"][0].tolist()
    curr_ids = prompt_ids[:]  # list of token IDs representing current sequence

    # Set up gRPC channel and stub for target communication
    target_address = f"{peer_ip}:{port}"
    channel = grpc.insecure_channel(target_address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    # Initiate generation on target (send prompt)
    try:
        stub.StartGeneration(inference_pb2.StartRequest(prompt=prompt), timeout=5.0)
    except Exception as e:
        print(f"[Draft] ERROR: Could not connect to target at {target_address}: {e}")
        sys.exit(1)

    print(f"[Draft] Connected to target at {target_address}. Starting speculative decoding...")
    start_time = time.time()
    generated_tokens = 0
    output_complete = False

    # Main speculative decoding loop
    while generated_tokens < max_tokens and not output_complete:
        # Determine how many tokens to speculate in this batch
        tokens_needed = max_tokens - generated_tokens
        batch_size = num_speculative if num_speculative < tokens_needed else tokens_needed
        draft_batch = []
        eos_proposed = False

        # Draft model generates `batch_size` tokens speculatively
        for j in range(batch_size):
            # Prepare input tensor for draft model (similar to target logic)
            input_ids = torch.tensor([curr_ids], dtype=torch.long)
            attn_mask = torch.ones_like(input_ids)
            orig_len = input_ids.shape[1]
            if compiled_flag:
                max_len = model_loader.DEFAULT_MAX_SEQ_LEN
                if orig_len > max_len:
                    raise RuntimeError(f"Input length {orig_len} exceeds compiled max length {max_len}.")
                if orig_len < max_len:
                    pad_len = max_len - orig_len
                    pad_ids = torch.zeros((1, pad_len), dtype=torch.long)
                    pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
                    input_ids = torch.cat([input_ids, pad_ids], dim=1)
                    attn_mask = torch.cat([attn_mask, pad_mask], dim=1)
            outputs = model(input_ids, attention_mask=attn_mask)
            if compiled_flag:
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                logits = outputs[:, :orig_len, :]
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            # Select next token (greedy)
            next_token_id = int(torch.argmax(logits[:, -1, :], dim=1).item())
            draft_batch.append(next_token_id)
            curr_ids.append(next_token_id)
            # If draft predicts EOS, stop batch generation early
            if next_token_id == tokenizer.eos_token_id:
                eos_proposed = True
                break

        # Send the batch of draft tokens to target for verification
        verify_request = inference_pb2.VerifyRequest(draft_tokens=draft_batch)
        verify_response = stub.VerifyDraftTokens(verify_request)
        # Process the response
        if verify_response.all_matched:
            # All tokens in draft_batch were accepted by target
            generated_tokens += len(draft_batch)
            if verify_response.finished:
                # Target indicated generation should finish (EOS reached)
                output_complete = True
        else:
            # Mismatch occurred at position match_count (0-indexed in batch)
            match_count = verify_response.match_count
            correct_token = verify_response.correct_token
            # Remove any draft-predicted tokens beyond the match_count from current sequence
            # (They were not accepted)
            while len(curr_ids) > len(prompt_ids) + generated_tokens + match_count:
                curr_ids.pop()  # remove the unaccepted draft tokens
            # Accept the tokens that matched
            generated_tokens += match_count
            # Append the target's correct token for the mismatch position
            curr_ids.append(correct_token)
            generated_tokens += 1
            if correct_token == tokenizer.eos_token_id or verify_response.finished:
                # If target's correct token is EOS or signaled finish, end generation
                output_complete = True
        # If we proposed fewer tokens than batch_size due to early EOS, we should stop as well
        if eos_proposed:
            # (Even if target didn't finish, if draft hit EOS, we consider generation done)
            output_complete = True

    # Decode the full output text (prompt + generated tokens)
    output_ids = curr_ids
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    elapsed = time.time() - start_time
    speed = generated_tokens / elapsed if elapsed > 0 else float('inf')
    print(f"[Draft] Final output: {output_text}")
    print(f"[Draft] Time: {elapsed:.2f} s, tokens generated: {generated_tokens}, speed: {speed:.2f} tokens/s")
    # Close the gRPC channel
    channel.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choral-Spec unified script: compile and run draft/target nodes.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Subparser for compile mode
    parser_compile = subparsers.add_parser("compile", help="Compile a model to an AWS Neuron-optimized .pt")
    parser_compile.add_argument("--model-id", type=str, required=True,
                                help="HuggingFace model name or local path to model directory")
    parser_compile.add_argument("--role", type=str, choices=["target", "draft"], required=True,
                                help="Role for which to compile the model (informs output file naming)")
    parser_compile.add_argument("--output", type=str, default=None,
                                help="Optional output path for the compiled model (.pt). If not specified, it will be saved as <model>_<role>_neuron_bf16_<max_seq_len>.pt in the project root.")
    parser_compile.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                                help="Maximum sequence length (context length) to compile for (default 1024)")

    # Subparser for run mode
    parser_run = subparsers.add_parser("run", help="Run a draft or target node for speculative decoding")
    parser_run.add_argument("--model-id", type=str, required=True,
                             help="HuggingFace model name or local path to model directory (should match compiled model if already compiled)")
    parser_run.add_argument("--role", type=str, choices=["target", "draft"], required=True,
                             help="Node role to run ('target' starts the server, 'draft' starts the client)")
    parser_run.add_argument("--peer-ip", type=str, default="127.0.0.1",
                             help="Peer IP address of the other node. For draft, this is the target server's IP. (Ignored for target role)")
    parser_run.add_argument("--port", type=int, default=50051,
                             help="Port for gRPC communication (default 50051). Target listens on this port; draft connects to this port.")
    parser_run.add_argument("--max-tokens", type=int, default=32,
                             help="Maximum number of new tokens to generate in speculative decoding")
    parser_run.add_argument("--num-speculative", type=int, default=4,
                             help="Number of tokens to speculate ahead before verification (speculative batch size)")
    parser_run.add_argument("--prompt", type=str, default="",
                             help="Initial prompt text for generation (applicable to draft role only). If not provided, an empty prompt is used.")

    # Subparser for compile_and_run mode
    parser_both = subparsers.add_parser("compile_and_run", help="Compile the model and then run the node (combines 'compile' and 'run').")
    parser_both.add_argument("--model-id", type=str, required=True,
                              help="HuggingFace model name or local path to model directory")
    parser_both.add_argument("--role", type=str, choices=["target", "draft"], required=True,
                              help="Node role to run after compilation")
    parser_both.add_argument("--peer-ip", type=str, default="127.0.0.1",
                              help="Peer IP address of the other node (if role=draft). Ignored for target.")
    parser_both.add_argument("--port", type=int, default=50051,
                              help="Port for gRPC communication (default 50051)")
    parser_both.add_argument("--max-tokens", type=int, default=32,
                              help="Maximum new tokens to generate (if role=draft)")
    parser_both.add_argument("--num-speculative", type=int, default=4,
                              help="Number of speculative tokens (if role=draft)")
    parser_both.add_argument("--prompt", type=str, default="",
                              help="Initial prompt text for generation (if role=draft).")

    args = parser.parse_args()

    if args.mode == "compile":
        # Compile the specified model for the given role
        compile_model_to_neuron(args["model_id"] if isinstance(args, dict) else args.model_id,
                                args["role"] if isinstance(args, dict) else args.role,
                                output_path=(args["output"] if isinstance(args, dict) else args.output),
                                max_seq_len=(args["max_seq_len"] if isinstance(args, dict) else args.max_seq_len))
    elif args.mode == "run":
        if args.role == "target":
            run_target(args.model_id, max_tokens=args.max_tokens, port=args.port)
        else:  # draft
            if args.prompt is None:
                parser.error("The --prompt argument is required when running in draft mode.")
            run_draft(args.model_id, peer_ip=args.peer_ip, port=args.port,
                      max_tokens=args.max_tokens, num_speculative=args.num_speculative, prompt=args.prompt)
    elif args.mode == "compile_and_run":
        # First, compile the model
        compiled_path = compile_model_to_neuron(args.model_id, args.role, output_path=None, max_seq_len=DEFAULT_MAX_SEQ_LEN)
        # Then run the appropriate node
        if args.role == "target":
            run_target(args.model_id, max_tokens=args.max_tokens, port=args.port)
        else:  # draft
            if args.prompt is None:
                parser.error("The --prompt argument is required when running in draft mode.")
            run_draft(args.model_id, peer_ip=args.peer_ip, port=args.port,
                      max_tokens=args.max_tokens, num_speculative=args.num_speculative, prompt=args.prompt)
