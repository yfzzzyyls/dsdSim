import grpc
from . import inference_pb2
from . import inference_pb2_grpc

def create_stub(target_address):
    channel = grpc.insecure_channel(target_address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    return stub

def verify_draft_tokens(stub, draft_tokens, session_id=0):
    request = inference_pb2.VerifyRequest(
        session_id=session_id,
        draft_tokens=draft_tokens
    )
    response = stub.VerifyDraftTokens(request)
    target_probs = list(response.target_probs)
    finished = response.finished
    return target_probs, finished

def finalize_tokens(stub, accepted_count, draft_chunk_size, session_id=0):
    request = inference_pb2.FinalizeRequest(
        session_id=session_id,
        accepted_count=accepted_count,
        draft_chunk_size=draft_chunk_size
    )
    response = stub.FinalizeTokens(request)
    final_token_id = response.final_token
    finished = response.finished
    return final_token_id, finished