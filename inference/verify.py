import time
import logging
import torch
from transformers import AutoTokenizer
from inference.model_loader import load_model
from grpc_comm import inference_pb2

logger = logging.getLogger(__name__)

def check_and_compare(stub, tokenizer, prompt, spec_output_tokens, max_tokens, measure_perf=False):
    """
    If verify or perf-test is enabled, generate output from target-only and compare with speculative output.
    Prints differences and timing information if requested.
    """
    # Decode speculative output for comparison
    spec_output_text = tokenizer.decode(spec_output_tokens, skip_special_tokens=True)
    # Request full generation from target for the same prompt
    gen_req = inference_pb2.GenerateRequest(prompt=prompt, max_new_tokens=max_tokens)
    start_time = time.time()
    gen_response = stub.GenerateFull(gen_req)
    target_time = time.time() - start_time
    target_output_text = gen_response.output_text
    # Print outputs for comparison
    print(f"[Draft] Target-only output: {target_output_text}")
    if spec_output_text == target_output_text:
        print("[Draft] Outputs match ✓")
    else:
        print("[Draft] Outputs differ! (Speculative vs Target)")
    # Optionally, find the first difference
    min_len = min(len(spec_output_text), len(target_output_text))
    diff_index = None
    for i in range(min_len):
        if spec_output_text[i] != target_output_text[i]:
            diff_index = i
            break
    if diff_index is not None:
        print(f"[Draft] First difference at position {diff_index}:")
        print(f" Speculative: ...{spec_output_text[diff_index:diff_index+20]}")
        print(f" Target: ...{target_output_text[diff_index:diff_index+20]}")
    # If performance test is requested, print timing info
    if measure_perf:
        # Note: spec_time should be measured outside and passed in for more precise comparison.
        # Here we only measured target time. We'll indicate that spec time is not measured here.
        print(f"[Draft] Target-only generation time: {target_time:.3f}s (speculative time was measured separately).")

def run_model(model_name: str, prompt: str, max_tokens: int = 50, sequence_length: int = 128):
    """
    Load a single model and generate text from a prompt, printing tokens one by one.
    This is used for standalone verification of draft or target models.
    """
    logger.info(f"Loading model '{model_name}' for standalone generation (sequence_length={sequence_length})...")
    model = load_model(model_name, sequence_length=sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if not prompt:
        logger.error("No prompt provided for generation.")
        return
    # Encode the prompt and prepare for generation
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_text = ""
    logger.info(f"Starting generation for prompt: {prompt!r}")
    for i in range(max_tokens):
        # Generate one token using the model
        output = model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        # Extract the last token ID from the output
        if isinstance(output, (list, tuple)):
            last_token_id = int(output[0][-1]) if isinstance(output[0], (list, tuple)) else int(output[0])
        else:
            last_token_id = int(output[0, -1])
        # Decode the token to text
        token_text = tokenizer.decode([last_token_id], skip_special_tokens=False)
        # Print the token (use repr to make spaces/newlines visible)
        print(f"Token {i+1}: {repr(token_text)}", flush=True)
        # Append the token to the generated text and update input_ids
        generated_text += token_text
        new_token_tensor = torch.tensor([[last_token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        # If an EOS token is generated, we stop early
        if tokenizer.eos_token_id is not None and last_token_id == tokenizer.eos_token_id:
            logger.info("EOS token encountered, stopping generation.")
            break
    logger.info("Generation completed.")
    full_output = prompt + generated_text
    print("\n=== Final Output ===\n" + full_output)
    return full_output
