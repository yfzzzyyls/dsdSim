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

def start_generation_batch(stub, prompt_id_tensor, prompt_length_tensor):
    """
    prompt_id_tensor : dict {"data": flat_ids, "shape": [B, L]}
    lens_tensor      : dict {"data": [true lengths], "shape": [B]}, or None to auto-derive
    """
    assert prompt_length_tensor is not None, "prompt_len_tensor must be provided"
    req = inference_pb2.StartBatchRequest(
        prompt_ids=_make_i32(prompt_id_tensor['data'], prompt_id_tensor['shape']),
        prompt_lens=_make_i32(prompt_length_tensor["data"], prompt_length_tensor["shape"])
    )
    resp = stub.StartGenerationBatch(req)
    return list(resp.session_ids)


def verify_batch_tokens(stub, session_id, tok_tensor, prob_tensor):
    """
    Send one DraftSequence that already contains the whole (B, γ) tensors.

    Parameters
    ----------
    session_id : int
        The single batched session-id returned by StartGenerationBatch.
    tok_tensor / prob_tensor : dict
        {'data': flat_list, 'shape': [B, γ]}  — output of _tensor_to_flat().
    Returns
    -------
    VerifyResult protobuf message (first entry of VerifyBatchResponse).
    """
    seq = inference_pb2.DraftSequence(
        session_id=session_id,
        draft_tokens=_make_i32(tok_tensor["data"], tok_tensor["shape"]),
        draft_probs=_make_f32(prob_tensor["data"], prob_tensor["shape"]),
    )
    resp = stub.VerifyBatchTokens(
        inference_pb2.VerifyBatchRequest(sequences=[seq])
    )
    return resp.results[0] if resp.results else None


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