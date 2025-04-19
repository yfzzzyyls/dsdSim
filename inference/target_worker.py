import logging
import torch
from concurrent import futures
import grpc

from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

class TargetSession:
    def __init__(self, input_ids):
        self.current_ids = input_ids  # Torch tensor [1, seq_len]
        self.finished = False
        self.tokens_generated = 0
        self.last_draft_chunk = None

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.sessions = {}  # session_id -> TargetSession
        self.lock = torch.multiprocessing.Lock()

    def StartGeneration(self, request, context):
        session_id = request.session_id
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens
        gamma = request.gamma
        logger.info(f"[session={session_id}] StartGeneration: prompt='{prompt_text}', max_new_tokens={max_tokens}, gamma={gamma}")
        with self.lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, overwriting.")
            if prompt_text:
                enc = self.tokenizer(prompt_text, return_tensors='pt')
                current_ids = enc["input_ids"]
            else:
                current_ids = torch.zeros((1,0), dtype=torch.long)
            self.sessions[session_id] = TargetSession(current_ids)
        return inference_pb2.StartResponse(acknowledged=True)

    # =============================
    # BATCH calls for multi-seq
    # =============================
    def VerifyBatchTokens(self, request, context):
        results = []
        with self.lock:
            for seq in request.sequences:
                sid = seq.session_id
                draft_tokens = list(seq.draft_tokens)
                if sid not in self.sessions:
                    logger.warning(f"Session {sid} not found in VerifyBatchTokens.")
                    # Return something: accepted=0, finished?
                    results.append(inference_pb2.VerifyResult(session_id=sid, tokens_accepted=0, target_token=0, finished=True))
                    continue

                sess = self.sessions[sid]
                if sess.finished:
                    logger.info(f"Session {sid} is already finished.")
                    results.append(inference_pb2.VerifyResult(session_id=sid, tokens_accepted=0, target_token=0, finished=True))
                    continue
                if not draft_tokens:
                    # no tokens => accept none
                    results.append(inference_pb2.VerifyResult(session_id=sid, tokens_accepted=0, target_token=0, finished=False))
                    continue

                # Expand input_ids
                expanded_ids = torch.cat([sess.current_ids, torch.tensor([draft_tokens], dtype=sess.current_ids.dtype)], dim=1)
                outputs = self.model(expanded_ids)
                all_logits = _extract_logits_all(outputs)  # shape [1, expanded_len, vocab]
                expanded_len = expanded_ids.size(1)
                actual_time_dim = all_logits.shape[1]

                if actual_time_dim >= expanded_len:
                    num_new = len(draft_tokens)
                    logits_slice = all_logits[:, -num_new:, :]
                    target_probs = []
                    for i, token_id in enumerate(draft_tokens):
                        row_logits = logits_slice[0, i, :]
                        row_probs = torch.softmax(row_logits, dim=-1)
                        p = float(row_probs[token_id].item())
                        target_probs.append(p)

                    # store chunk in the session for finalizing
                    sess.last_draft_chunk = draft_tokens
                    # we do not set finished here. We'll set finished if we see EOS in finalize.
                    # or if we wanted to check right now.
                    # we can do that if we want:
                    finished_flag = False
                    results.append(inference_pb2.VerifyResult(
                        session_id=sid,
                        tokens_accepted=0,  # we'll finalize acceptance in the finalize call
                        target_token=0,
                        finished=finished_flag
                    ))
                else:
                    # fallback single-step loop
                    fallback_probs = self._verify_single_step(sess, draft_tokens)
                    sess.last_draft_chunk = draft_tokens
                    results.append(
                        inference_pb2.VerifyResult(session_id=sid, tokens_accepted=0, target_token=0, finished=False)
                    )
        return inference_pb2.VerifyBatchResponse(results=results)

    def FinalizeBatchTokens(self, request, context):
        results = []
        with self.lock:
            for seq in request.sequences:
                sid = seq.session_id
                tokens = list(seq.tokens)
                if sid not in self.sessions:
                    logger.warning(f"Session {sid} not found in FinalizeBatchTokens.")
                    results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=True))
                    continue
                sess = self.sessions[sid]
                if sess.finished:
                    results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=True))
                    continue

                # Accept these tokens into sess.current_ids
                for t in tokens:
                    new_tok = torch.tensor([[t]], dtype=sess.current_ids.dtype)
                    sess.current_ids = torch.cat([sess.current_ids, new_tok], dim=1)
                    if self.eos_token_id is not None and t == self.eos_token_id:
                        sess.finished = True
                results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=sess.finished))
        return inference_pb2.FinalizeBatchResponse(results=results)

    def _verify_single_step(self, sess, draft_tokens):
        # fallback approach, calls model per token
        probs = []
        temp_ids = sess.current_ids.clone()
        for t in draft_tokens:
            out = self.model(temp_ids)
            logits = _extract_logits(out)
            row_probs = torch.softmax(logits, dim=-1)
            p = float(row_probs[0, t].item())
            probs.append(p)
            appended_tok = torch.tensor([[t]], dtype=temp_ids.dtype)
            temp_ids = torch.cat([temp_ids, appended_tok], dim=1)
        return probs

    # =============================
    # SINGLE-SEQUENCE calls
    # =============================

    def VerifyDraftTokens(self, request, context):
        sid = request.session_id
        draft_tokens = list(request.draft_tokens)
        # logger.info(f"[session={sid}] VerifyDraftTokens: {draft_tokens}")
        with self.lock:
            if sid not in self.sessions:
                logger.warning(f"Session {sid} not found.")
                return inference_pb2.VerifyResponse(target_probs=[0.0]*len(draft_tokens), finished=True)
            sess = self.sessions[sid]
            if sess.finished:
                logger.info(f"Session {sid} is finished.")
                return inference_pb2.VerifyResponse(target_probs=[], finished=True)
            if not draft_tokens:
                return inference_pb2.VerifyResponse(target_probs=[], finished=False)
            expanded_ids = torch.cat([sess.current_ids, torch.tensor([draft_tokens], dtype=sess.current_ids.dtype)], dim=1)
            outputs = self.model(expanded_ids)
            all_logits = _extract_logits_all(outputs)
            expanded_len = expanded_ids.size(1)
            actual_time_dim = all_logits.shape[1]
            target_probs = []
            finished_flag = False
            if actual_time_dim >= expanded_len:
                num_new = len(draft_tokens)
                logits_slice = all_logits[:, -num_new:, :]
                for i, token_id in enumerate(draft_tokens):
                    row_logits = logits_slice[0, i, :]
                    row_probs = torch.softmax(row_logits, dim=-1)
                    p = float(row_probs[token_id].item())
                    target_probs.append(p)
            else:
                # fallback single-step loop
                fallback = self._verify_single_step(sess, draft_tokens)
                target_probs = fallback

            sess.last_draft_chunk = draft_tokens
            return inference_pb2.VerifyResponse(target_probs=target_probs, finished=finished_flag)

    def FinalizeTokens(self, request, context):
        sid = request.session_id
        accepted_count = request.accepted_count
        draft_chunk_size = request.draft_chunk_size
        # logger.info(f"[session={sid}] FinalizeTokens: accepted_count={accepted_count}, chunk_size={draft_chunk_size}")
        with self.lock:
            if sid not in self.sessions:
                logger.warning(f"Session {sid} not found.")
                return inference_pb2.FinalizeResponse(final_token=0, finished=True)
            sess = self.sessions[sid]
            if sess.finished:
                logger.info(f"Session {sid} is already finished.")
                return inference_pb2.FinalizeResponse(final_token=0, finished=True)

            if sess.last_draft_chunk:
                chunk = sess.last_draft_chunk
                accepted = chunk[:accepted_count]
                # accept them
                for t in accepted:
                    sess.current_ids = torch.cat([sess.current_ids, torch.tensor([[t]], dtype=sess.current_ids.dtype)], dim=1)
                    if self.eos_token_id is not None and t == self.eos_token_id:
                        sess.finished = True
                # if partial acceptance:
                fallback_token = 0
                if accepted_count < draft_chunk_size:
                    fallback_token = self._generate_one_token(sess)
                sess.last_draft_chunk = None
                if fallback_token != 0 and self.eos_token_id is not None and fallback_token == self.eos_token_id:
                    sess.finished = True
                return inference_pb2.FinalizeResponse(final_token=fallback_token, finished=sess.finished)
            else:
                # no chunk stored => nothing accepted => just generate one token
                fallback_token = self._generate_one_token(sess)
                return inference_pb2.FinalizeResponse(final_token=fallback_token, finished=sess.finished)

    def GenerateFull(self, request, context):
        # baseline target-only decoding, optional
        return super().GenerateFull(request, context)

    def _generate_one_token(self, sess: TargetSession):
        outputs = self.model(sess.current_ids)
        logits = _extract_logits(outputs)
        token_id = int(torch.argmax(logits, dim=-1)[0].item())
        appended_tok = torch.tensor([[token_id]], dtype=sess.current_ids.dtype)
        sess.current_ids = torch.cat([sess.current_ids, appended_tok], dim=1)
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            sess.finished = True
        sess.tokens_generated += 1
        return token_id


def _extract_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        out_t = outputs[0]
    elif hasattr(outputs, "logits"):
        out_t = outputs.logits[:, -1, :]
    else:
        out_t = outputs
    if len(out_t.shape) == 3:
        return out_t[:, -1, :].float()
    elif len(out_t.shape) == 2:
        return out_t.float()
    elif len(out_t.shape) == 1:
        return out_t.unsqueeze(0).float()
    else:
        raise ValueError(f"Unknown shape for outputs: {out_t.shape}")


def _extract_logits_all(outputs):
    if isinstance(outputs, (tuple, list)):
        out_t = outputs[0]
    elif hasattr(outputs, "logits"):
        return outputs.logits.float()
    else:
        out_t = outputs
    if len(out_t.shape) == 3:
        return out_t.float()
    elif len(out_t.shape) == 2:
        return out_t.unsqueeze(1).float()
    elif len(out_t.shape) == 1:
        return out_t.unsqueeze(0).unsqueeze(0).float()
    else:
        raise ValueError(f"Unhandled shape for model output: {out_t.shape}")


def run_server(model_path, port=50051, sequence_length=128, profile=False):
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading target model from {model_path} seq_len={sequence_length}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server_address = f"[::]:{port}"
    logger.info(f"Target server starting on {server_address}")
    server.add_insecure_port(server_address)
    server.start()
    server.wait_for_termination()


def run_local(model_path, prompt="", max_new_tokens=50, sequence_length=128, profile=False):
    # same as before
    pass