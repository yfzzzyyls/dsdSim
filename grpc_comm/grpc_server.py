import grpc
from concurrent import futures

# Import the generated classes from the compiled proto module
import inference_pb2, inference_pb2_grpc

class SpeculativeServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, target_worker):
        """
        target_worker: an instance of TargetWorker (from inference/target_worker.py)
        that holds the target model, tokenizer, and state.
        """
        self.target = target_worker

    def StartGeneration(self, request, context):
        # Set up the prompt context in the target worker
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens if request.max_new_tokens > 0 else None
        self.target.start_generation(prompt_text, max_tokens=max_tokens)
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        draft_tokens = list(request.draft_tokens)
        result = self.target.verify_tokens(draft_tokens)
        # result is a dict or tuple containing match_count, all_matched, correct_token, finished
        all_matched = result["all_matched"]
        match_count = result["match_count"]
        correct_token = result.get("correct_token", 0)
        finished = result.get("finished", False)
        return inference_pb2.VerifyResponse(all_matched=all_matched,
                                            match_count=match_count,
                                            correct_token=correct_token,
                                            finished=finished)

    def GenerateFull(self, request, context):
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