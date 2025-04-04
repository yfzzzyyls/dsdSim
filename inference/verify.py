import time
from grpc_comm.proto import inference_pb2

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
            print(f"    Speculative: ...{spec_output_text[diff_index:diff_index+20]}")
            print(f"    Target:      ...{target_output_text[diff_index:diff_index+20]}")
    # If performance test is requested, print timing info
    if measure_perf:
        # Note: spec_time should be measured outside and passed in for more precise comparison.
        # Here we only measured target time. We'll indicate that spec time is not measured here.
        print(f"[Draft] Target-only generation time: {target_time:.3f}s (speculative time was measured separately).")