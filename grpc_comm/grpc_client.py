# ==========================
# 2) grpc_client.py
# ==========================
import grpc
from . import inference_pb2
from . import inference_pb2_grpc


def create_stub(target_address):
    # Enable compression to reduce message size
    channel_options = [
        ('grpc.max_message_length', 64 * 1024 * 1024),  # 64MB max message
        ('grpc.max_receive_message_length', 64 * 1024 * 1024),
        ('grpc.default_compression_algorithm', 'gzip'),  # Enable compression
        ('grpc.default_compression_level', 'high'),     # High compression
    ]
    channel = grpc.insecure_channel(target_address, options=channel_options)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    return stub

# -----------------------------------------
# BATCH-ORIENTED CLIENT CALLS
# -----------------------------------------

def verify_batch_tokens(stub, sequences):
    # sequences is a list of (session_id, [draft_tokens])
    # build the request
    seq_msgs = []
    for s in sequences:
        session_id, draft_toks = s
        seq_msgs.append(
            inference_pb2.DraftSequence(
                session_id=session_id,
                draft_tokens=draft_toks
            )
        )
    request = inference_pb2.VerifyBatchRequest(sequences=seq_msgs)
    # Use compression for batch requests (typically larger)
    response = stub.VerifyBatchTokens(request, compression='gzip')
    # returns a list of results
    results = []
    for r in response.results:
        results.append({
            'session_id': r.session_id,
            'tokens_accepted': r.tokens_accepted,
            'target_token': r.target_token,
            'finished': r.finished,
            'committed_ids': [],  # Add this field for compatibility
            'verify_time_ms': 0.0,  # Add this field for compatibility
        })
    return results


def finalize_batch_tokens(stub, sequences):
    # sequences is a list of (session_id, [accepted_tokens])
    seq_msgs = []
    for s in sequences:
        session_id, tok_list = s
        seq_msgs.append(
            inference_pb2.FinalizeSequence(
                session_id=session_id,
                tokens=tok_list
            )
        )
    request = inference_pb2.FinalizeBatchRequest(sequences=seq_msgs)
    response = stub.FinalizeBatchTokens(request)
    results = []
    for r in response.results:
        results.append({
            'session_id': r.session_id,
            'finished': r.finished,
        })
    return results

# -----------------------------------------
# SINGLE-SEQUENCE CLIENT CALLS (existing)
# -----------------------------------------

def verify_draft_tokens(stub, draft_tokens, draft_probs, session_id=0):
    request = inference_pb2.VerifyRequest(
        session_id   = session_id,
        draft_tokens = draft_tokens,
        draft_probs  = draft_probs,   # <<<
    )
    # Use compression for large messages
    resp = stub.VerifyDraftTokens(request, compression='gzip')
    return (
        list(resp.committed_ids),
        resp.accepted_count,
        resp.verify_time_ms,
        resp.finished,
    )