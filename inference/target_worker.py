
"""
Target‑side service for distributed speculative decoding **using the
official vLLM + Transformers‑NeuronX continuous‑batch engine**.

Design assumptions
------------------
* One Trn1 host loads the *large* verify model inside vLLM (`LLMEngine`)
  compiled with:
      max_num_seqs = 2
      max_model_len = block_size = 128
* Two Inf2 draft hosts connect over gRPC.
* γ (draft chunk) is the same for all clients (e.g. γ = 4 → 5‑token blocks).
* We expose only two RPCs:
      StartGeneration(prompt, γ)  –> session_id
      VerifyDraftTokens(session_id, draft_tokens, draft_probs)
  The accept/reject rule is executed server‑side for simplicity.

This file no longer contains any hand‑rolled row pools, KV‑pointer tensors,
or speculative_forward helpers—the vLLM engine manages slots & cache.
"""
import logging
import time
import uuid
from concurrent import futures

import grpc
import torch
from transformers import AutoTokenizer
from vllm.engine.llm_engine import LLMEngine, EngineArgs

from grpc_comm import inference_pb2, inference_pb2_grpc

# ──────────────── logging ────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────
# TargetSession: *only* stores prompt text & engine request‑id
# ──────────────────────────────────────────
class TargetSession:
    def __init__(self, req_id: str, prompt_ids):
        self.req_id = req_id                 # vLLM request handle
        self.prompt_ids = prompt_ids         # list[int]
        self.finished = False
        self.tokens_generated = 0


# ──────────────────────────────────────────
# gRPC service
# ──────────────────────────────────────────
class SpeculativeService(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self,
                 model_path: str,
                 max_num_seqs: int = 2,
                 max_model_len: int = 128,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 gamma: int = 4):
        self.gamma = gamma
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        # ---- Launch vLLM engine (continuous batching on Neuron) ----
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(
            model=model_path,
            device="neuron",
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            block_size=max_model_len,
        )
        self.engine = LLMEngine.from_engine_args(args)

        self.sessions = {}                # session_id → TargetSession
        self.lock = torch.multiprocessing.Lock()

    # ──────────────────────────────────────
    # RPC 1: StartGeneration
    # ──────────────────────────────────────
    def StartGeneration(self, request, context):
        prompt = request.prompt
        if not prompt:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "prompt must be non‑empty")

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0).tolist()
        session_id = int(uuid.uuid4()) & 0xFFFFFFFF
        req_id     = f"{session_id}"

        # add slot & run prefill once
        self.engine.add_request(req_id, prompt_ids)
        self.engine.prefill([req_id])          # fills KV cache, no logits needed

        self.sessions[session_id] = TargetSession(req_id, prompt_ids)
        logger.info("StartGeneration sid=%s  prompt_len=%d  slots=%d/%d",
                    session_id, len(prompt_ids),
                    len(self.sessions), self.engine.scheduler.max_num_seqs)

        return inference_pb2.StartResponse(acknowledged=True, session_id=session_id)

    # ──────────────────────────────────────
    # RPC 2: VerifyDraftTokens
    # ──────────────────────────────────────
    def VerifyDraftTokens(self, request, context):
        sid = request.session_id
        sess = self.sessions.get(sid)
        if sess is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "unknown session_id")

        draft = list(request.draft_tokens)
        if not draft:
            return inference_pb2.VerifyResponse(committed_ids=[], accepted_count=0,
                                                verify_time_ms=0.0, finished=sess.finished)

        prev_tok = sess.prompt_ids[-1]
        tok_block = [prev_tok] + draft                   # γ+1
        start_t = time.perf_counter()
        out = self.engine.decode([sess.req_id],
                                 prompt_ids=tok_block,
                                 num_tokens=len(tok_block))
        logits = out.logits[0]                           # (γ+1, vocab)
        verify_ms = (time.perf_counter() - start_t) * 1000

        # ---- accept / reject rule (server‑side) ----
        probs = torch.softmax(logits[:-1], dim=-1)       # γ rows
        draft_probs = torch.tensor(list(request.draft_probs),
                                   dtype=torch.float32, device=probs.device)
        tgt_p = probs[torch.arange(len(draft)), draft]
        ratio = tgt_p / draft_probs
        randv = torch.rand_like(ratio)
        accept_mask = (tgt_p >= draft_probs) | (randv < ratio)
        first_rej = torch.nonzero(~accept_mask)
        accepted_cnt = int(first_rej[0]) if first_rej.numel() else len(draft)

        committed = draft[:accepted_cnt]
        if accepted_cnt < len(draft):
            bonus_logits = logits[accepted_cnt]
        else:
            bonus_logits = logits[-1]

        bonus_id = int(torch.multinomial(torch.softmax(bonus_logits, dim=-1), 1).item())
        committed.append(bonus_id)

        # ---- bump KV pointer by delta ----
        delta = len(committed)
        self.engine.update_kv(sess.req_id, delta)
        sess.prompt_ids.extend(committed)
        sess.tokens_generated += delta

        # EOS?
        eos = self.tokenizer.eos_token_id
        if committed[-1] == eos or sess.tokens_generated >= request.max_new_tokens:
            sess.finished = True
            self.engine.remove_request(sess.req_id)

        logger.info("Verify sid=%s  accepted=%d/%d  bonus=%d  finished=%s",
                    sid, accepted_cnt, len(draft), bonus_id, sess.finished)

        return inference_pb2.VerifyResponse(
            committed_ids=committed,
            accepted_count=accepted_cnt,
            verify_time_ms=verify_ms,
            finished=sess.finished
        )


# ──────────────────────────────────────────
# gRPC server launcher
# ──────────────────────────────────────────
def run_server(model_path: str,
               port: int = 50051,
               max_num_seqs: int = 2,
               max_model_len: int = 128,
               gamma: int = 4):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(
        SpeculativeService(model_path,
                           max_num_seqs=max_num_seqs,
                           max_model_len=max_model_len,
                           gamma=gamma),
        server
    )
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Target server (vLLM continuous batching) listening on :%d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--slots", type=int, default=2, help="max_num_seqs")
    parser.add_argument("--ctx", type=int, default=128, help="max_model_len")
    parser.add_argument("--gamma", type=int, default=4)
    args = parser.parse_args()
    run_server(args.model, args.port, args.slots, args.ctx, args.gamma)
