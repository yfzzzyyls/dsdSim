# ==========================
# 2) grpc_client.py
# ==========================
import grpc
from . import inference_pb2
from . import inference_pb2_grpc

# ------- helper builders for Int32Tensor / FloatTensor -------------
def _make_i32(data, shape):
    return inference_pb2.Int32Tensor(data=data, shape=shape)

def _make_f32(data, shape):
    return inference_pb2.FloatTensor(data=data, shape=shape)


def create_stub(target_address):
    channel = grpc.insecure_channel(target_address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    return stub

# -----------------------------------------
# BATCH-ORIENTED CLIENT CALLS
# -----------------------------------------

def start_generation_batch(stub, prompt_id_tensor):
    """
    prompt_id_tensor : torch/int list flattened (data) and its shape [B, L]
    """
    B, L = prompt_id_tensor['shape']
    req = inference_pb2.StartBatchRequest(
        prompt_ids=_make_i32(prompt_id_tensor['data'], [B, L])
    )
    resp = stub.StartGenerationBatch(req)
    return list(resp.session_ids)


# sid_tok_prob_triplets : list[(sid, tok_tensor, prob_tensor)]
def verify_batch_tokens(stub, sid_tok_prob_triplets):
    # sid_tok_prob_triplets : list[(sid, tok_tensor, prob_tensor)]
    seq_msgs = []
    for sid, tok_t, prob_t in sid_tok_prob_triplets:
        # tok_t / prob_t are dicts: {'data': flat_list, 'shape':[B, gamma]}
        seq_msgs.append(
            inference_pb2.DraftSequence(
                session_id=sid,
                draft_tokens=_make_i32(tok_t['data'], tok_t['shape']),
                draft_probs=_make_f32(prob_t['data'], prob_t['shape']),
            )
        )
    request  = inference_pb2.VerifyBatchRequest(sequences=seq_msgs)
    response = stub.VerifyBatchTokens(request)
    results  = []
    for r in response.results:
        results.append({
            "session_id":      r.session_id,
            "tokens_accepted": r.tokens_accepted,
            "committed_ids":   list(r.committed_ids.data),
            "finished":        r.finished,
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
    resp = stub.VerifyDraftTokens(request)
    return (
        list(resp.committed_ids),
        resp.accepted_count,
        resp.verify_time_ms,
        resp.finished,
    )