import grpc
from concurrent import futures

# Import the generated classes from the compiled proto module
import inference_pb2
import inference_pb2_grpc

class SpeculativeServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, target_worker):
        """
        target_worker: an instance of TargetWorker (from inference/target_worker.py)
        that holds the target model, tokenizer, and state.
        """
        self.target = target_worker

    def StartGeneration(self, request, context):
        # Initialize the target model context with the prompt and max token limit
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens if request.max_new_tokens > 0 else None
        self.target.start_generation(prompt_text, max_tokens=max_tokens)
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        # Verify a sequence of draft tokens with the target model
        draft_tokens = list(request.draft_tokens)
        # Obtain target probabilities for each draft token (and finished flag) without updating state
        result = self.target.verify_tokens(draft_tokens)
        target_probs = result["target_probs"]  # list of float probabilities for each token
        finished = result.get("finished", False)
        return inference_pb2.VerifyResponse(target_probs=target_probs, finished=finished)

    def FinalizeTokens(self, request, context):
        # Finalize after verification: commit accepted tokens and possibly generate target token
        accepted_count = request.accepted_count
        result = self.target.finalize_tokens(accepted_count)
        final_token = result.get("final_token", 0)
        finished = result.get("finished", False)
        return inference_pb2.FinalizeResponse(final_token=final_token, finished=finished)

    def GenerateFull(self, request, context):
        # Baseline full generation using target model only
        prompt = request.prompt
        max_new_tokens = request.max_new_tokens
        output_text = self.target.generate_full(prompt, max_new_tokens=max_new_tokens)
        return inference_pb2.GenerateResponse(output_text=output_text)

def start_server(target_worker, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = SpeculativeServicer(target_worker)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server