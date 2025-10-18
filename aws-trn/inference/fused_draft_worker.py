import logging
import grpc
import time
from grpc_comm import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


class FusedSpeculativeClient:
    def __init__(self, server_address: str = "localhost:50051"):
        """
        Initialize the client for fused speculative decoding.
        
        Args:
            server_address: Address of the target server with fused model
        """
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = inference_pb2_grpc.SpeculativeServiceStub(self.channel)
        logger.info(f"Connected to fused speculative server at {server_address}")
    
    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 temperature: float = 1.0, top_p: float = 0.9,
                 speculation_length: int = 5):
        """
        Send a generation request to the fused speculative server.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            speculation_length: Number of tokens to speculate at once
            
        Returns:
            Generated text and statistics
        """
        try:
            # Create request
            request = inference_pb2.GenerateRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                speculation_length=speculation_length
            )
            
            logger.info(f"Sending generation request: prompt='{prompt[:50]}...', "
                       f"max_new_tokens={max_new_tokens}")
            
            # Send request and get response
            # Use perf_counter for high-precision timing (matches distributed)
            start_time = time.perf_counter()
            response = self.stub.Generate(request)
            end_time = time.perf_counter()
            
            client_latency = (end_time - start_time) * 1000  # ms
            
            logger.info(f"Received response: {response.total_tokens} tokens generated")
            logger.info(f"Server generation time: {response.generation_time_ms:.2f}ms")
            logger.info(f"Client round-trip latency: {client_latency:.2f}ms")
            logger.info(f"Tokens per second: {response.tokens_per_second:.2f}")
            logger.info(f"Acceptance rate: {response.acceptance_rate:.2%}")
            
            return {
                'generated_text': response.generated_text,
                'total_tokens': response.total_tokens,
                'generation_time_ms': response.generation_time_ms,
                'client_latency_ms': client_latency,
                'tokens_per_second': response.tokens_per_second,
                'total_speculations': response.total_speculations,
                'accepted_tokens': response.accepted_tokens,
                'acceptance_rate': response.acceptance_rate
            }
            
        except grpc.RpcError as e:
            logger.error(f"RPC failed: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


def run_fused_client(
    target_host: str = "localhost",
    port: int = 50051,
    prompt_text_file: str = "",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    speculation_length: int = 5,
    profile: bool = False
):
    """Run the fused speculative client with file-based prompts."""
    import os
    
    if not os.path.exists(prompt_text_file):
        logger.error(f"Prompt text file not found: {prompt_text_file}")
        return
    
    with open(prompt_text_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        logger.error("No valid lines in the prompt file.")
        return
    
    server_address = f"{target_host}:{port}"
    client = FusedSpeculativeClient(server_address=server_address)
    
    try:
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"\n{'='*60}\nProcessing prompt {i+1}/{len(prompts)}\n{'='*60}")
            
            result = client.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                speculation_length=speculation_length
            )
            
            # Add prompt info to result
            result['prompt_idx'] = i
            result['prompt'] = prompt
            result['full_text'] = prompt + result['generated_text']
            results.append(result)
            
            # Print output
            print(f"\n[Prompt {i} Output]:\n{result['full_text']}\n")
        
        # Summary statistics
        if results:
            avg_latency = sum(r['client_latency_ms'] for r in results) / len(results)
            avg_tps = sum(r['tokens_per_second'] for r in results) / len(results)
            avg_acceptance = sum(r['acceptance_rate'] for r in results) / len(results)
            
            print(f"\n{'='*60}")
            print("SUMMARY STATISTICS")
            print(f"{'='*60}")
            print(f"Total prompts processed: {len(results)}")
            print(f"Average latency: {avg_latency:.2f}ms")
            print(f"Average tokens/second: {avg_tps:.2f}")
            print(f"Average acceptance rate: {avg_acceptance:.2%}")
            
            # Save performance data if profiling
            if profile:
                import json
                perf_file = "performance_fused_speculative.json"
                with open(perf_file, 'w') as f:
                    json.dump({
                        'results': results,
                        'summary': {
                            'avg_latency_ms': avg_latency,
                            'avg_tokens_per_second': avg_tps,
                            'avg_acceptance_rate': avg_acceptance
                        }
                    }, f, indent=2)
                logger.info(f"Performance data saved to {perf_file}")
    
    finally:
        client.close()