import logging
import torch
from concurrent import futures
import grpc

from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

class TargetSession:
    def __init__(self, input_ids):
        self.current_ids = input_ids  # Torch tensor [1, seq_len]
        self.finished = False
        self.tokens_generated = 0
        self.last_draft_chunk = None
        # pointer to the *next* KV slot
        self.cache_ids = torch.tensor([input_ids.shape[1]], dtype=torch.int32)
        self.pending_logits = None

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128, spec_length=None, temperature: float = 1.0, top_p: float = 0.9):
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length, spec_length=spec_length)
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        self._ctx_estimate = sequence_length
        self.sessions = {}  # session_id -> TargetSession
        self.lock = torch.multiprocessing.Lock()

    
    def _sync_kv_pointer(self, sess: TargetSession):
        self.model.cache_ids = sess.cache_ids.clone()
        if hasattr(self.model, "_next_pos"):
            self.model._next_pos = int(sess.cache_ids.item())
        # ---- sanity check ----
        assert int(self.model.cache_ids.item()) == int(sess.cache_ids.item()), \
            "Target KV cache_ids desynchronised after sync"


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
            # --- prime Neuron KV cache on the prompt ---
            self.model.cache_ids = None
            self.model._next_pos = 0
            if current_ids.shape[1] > 0:
                _ = self.model.forward(current_ids)
            # store pointer (next index) inside the session
            self.sessions[session_id].cache_ids = torch.tensor(
                [current_ids.shape[1]], dtype=torch.int32
            )
        return inference_pb2.StartResponse(acknowledged=True)

    # =============================
    # BATCH calls for multi‑seq
    # =============================
    def VerifyBatchTokens(self, request, context):
        """
        Verify several session‑specific draft token chunks in one RPC.
        Each element of request.sequences carries:
            • session_id   - int
            • draft_tokens - repeated int32
        For every sequence we compute P_target(draft_token | context) **incrementally**
        using the target KV cache (one forward per token).  No concat / pad.
        """
        results = []
        with self.lock:
            for seq in request.sequences:
                sid = seq.session_id
                draft_tokens = list(seq.draft_tokens)

                # 1) Session validation
                if sid not in self.sessions:
                    logger.warning(f"[VerifyBatchTokens] Session {sid} not found.")
                    results.append(
                        inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=True,            # treat as finished / invalid
                        )
                    )
                    continue

                sess = self.sessions[sid]
                if sess.finished:
                    results.append(
                        inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=True,
                        )
                    )
                    continue

                if not draft_tokens:
                    # Empty chunk – nothing to verify
                    results.append(
                        inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=False,
                        )
                    )
                    continue

                # 2) Incremental verify using the session’s KV cache
                target_probs = self._verify_single_step(sess, draft_tokens)

                # 3) Remember this chunk so FinalizeTokens can accept/rollback
                sess.last_draft_chunk = draft_tokens

                # 4) Return a VerifyResult (no tokens accepted yet;
                #    acceptance happens in FinalizeTokens)
                results.append(
                    inference_pb2.VerifyResult(
                        session_id=sid,
                        tokens_accepted=0,
                        target_token=0,
                        finished=False,
                    )
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

    # def _verify_single_step(self, sess, draft_tokens):
    #     # fallback approach, calls model per token
    #     probs = []
    #     # temp_ids = sess.current_ids.clone()
    #     temp_ids = self._pad_ids(sess.current_ids.clone())
    #     for t in draft_tokens:
    #         out = self.model(temp_ids)
    #         logits = _extract_logits(out)
    #         row_probs = torch.softmax(logits, dim=-1)
    #         p = float(row_probs[0, t].item())
    #         probs.append(p)
    #         # appended_tok = torch.tensor([[t]], dtype=temp_ids.dtype)
    #         # temp_ids = torch.cat([temp_ids, appended_tok], dim=1)
    #         appended_tok = torch.tensor([[t]], dtype=temp_ids.dtype)
    #         temp_ids = torch.cat([temp_ids, appended_tok], dim=1)
    #         temp_ids = self._pad_ids(temp_ids)
    #     return probs
    
    def _verify_single_step(self, sess: TargetSession, draft_tokens):
        """
         Compute p_target(d₁…dₖ | context) **without mutating** permanent state.
 
         We assume:
           • sess.cache_ids holds the *next‑free* KV slot index.
           • sess.pending_logits (if not None) already contains logits for the
             next position (temperature‑scaled).
 
         Returns
         -------
         probs : List[float] ‑ probability for each draft token under the
                 target model’s *temperature‑scaled* soft‑max.
         """
        # Early exit: no draft tokens – keep pending_logits intact
        if not draft_tokens:
            return []
        # ---------- snapshot current pointer & logits ----------
        orig_cache   = sess.cache_ids.clone()
        orig_nextpos = int(orig_cache.item())
        logits_next  = sess.pending_logits            # may be None
        sess.pending_logits = None                    # consume

        probs = []
        prev_tok  = int(sess.current_ids[0, -1])
        prev_pos  = orig_nextpos - 1                  # position of prev_tok
        next_pos  = orig_nextpos                      # where d₁ will go

        self._sync_kv_pointer(sess)                   # align model→session

        for tok in draft_tokens:
            # --- P_target(tok | context) ---
            if logits_next is None:
                lgts, _ = self.model.forward(
                    input_ids=torch.tensor([[prev_tok]], dtype=sess.current_ids.dtype),
                    cache_ids=torch.tensor([prev_pos], dtype=torch.int32),
                )
                logits_row = lgts[0] if lgts.dim() == 2 else lgts
            else:
                logits_row = logits_next[0] if logits_next.dim() == 2 else logits_next

            logits_row = logits_row / max(self.temperature, 1e-6)
            p = float(torch.softmax(logits_row, dim=-1)[tok].item())
            probs.append(p)

            # --- advance cache by *consuming* tok, capture logits for next ---
            lgts2, _ = self.model.forward(
                input_ids=torch.tensor([[tok]], dtype=sess.current_ids.dtype),
                cache_ids=torch.tensor([next_pos], dtype=torch.int32),
            )
            logits_next = lgts2[0] if lgts2.dim() == 2 else lgts2
            prev_tok = tok
            prev_pos = next_pos
            next_pos += 1

        sess.pending_logits = logits_next.clone()     # reuse next time

        # ---------- restore snapshot ----------
        self.model.cache_ids = orig_cache.clone()
        if hasattr(self.model, "_next_pos"):
            self.model._next_pos = orig_nextpos
        sess.cache_ids = orig_cache
        # final sanity: model and session pointers must match
        assert int(self.model.cache_ids.item()) == int(sess.cache_ids.item()), \
            "KV desync detected on verify exit"
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
            self._sync_kv_pointer(sess)
            if not draft_tokens:
                target_probs = self._verify_single_step(sess, draft_tokens)
                sess.last_draft_chunk = draft_tokens
                return inference_pb2.VerifyResponse(target_probs=target_probs, finished=False)
            # expanded_ids = torch.cat([sess.current_ids, torch.tensor([draft_tokens], dtype=sess.current_ids.dtype)], dim=1)
            # expanded_ids = torch.cat(
            #     [sess.current_ids, torch.tensor([draft_tokens], dtype=sess.current_ids.dtype)],
            #     dim=1,
            # )
            target_probs = self._verify_single_step(sess, draft_tokens)
            sess.last_draft_chunk = draft_tokens
            return inference_pb2.VerifyResponse(target_probs=target_probs, finished=False)

    def FinalizeTokens(self, request, context):
        sid              = request.session_id
        accepted_count   = request.accepted_count
        draft_chunk_size = request.draft_chunk_size

        with self.lock:
            # ---------- session checks ----------
            if sid not in self.sessions:
                logger.warning(f"Session {sid} not found.")
                return inference_pb2.FinalizeResponse(final_token=0, finished=True)

            sess = self.sessions[sid]
            if sess.finished:
                return inference_pb2.FinalizeResponse(final_token=0, finished=True)

            # ---------- retrieve last draft chunk ----------
            chunk = sess.last_draft_chunk or []
            accepted = chunk[:accepted_count]

            # ---------- 1) commit accepted tokens ----------
            for t in accepted:
                sess.current_ids = torch.cat(
                    [sess.current_ids,
                     torch.tensor([[t]], dtype=sess.current_ids.dtype)],
                    dim=1)
                self._sync_kv_pointer(sess)
                lgts, _ = self.model.forward(
                    input_ids=torch.tensor([[t]], dtype=sess.current_ids.dtype),
                    cache_ids=torch.tensor([self.model._next_pos], dtype=torch.int32),
                )
                sess.pending_logits = lgts[0] if lgts.dim()==2 else lgts
                sess.cache_ids = torch.tensor([self.model._next_pos], dtype=torch.int32)
                if self.eos_token_id is not None and t == self.eos_token_id:
                    sess.finished = True

            # ---------- 2) always generate ONE token from target ----------
            # fallback_token = self._generate_one_token(sess)
            fallback_token = self._generate_one_token(
                sess,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # clear chunk for next round
            sess.last_draft_chunk = None

            # ---------- EOS handling ----------
            if (
                fallback_token != 0
                and self.eos_token_id is not None
                and fallback_token == self.eos_token_id
            ):
                sess.finished = True

            token_text = self.tokenizer.decode([fallback_token]).strip() if fallback_token != 0 else "<none>"
            logger.info(f"[Finalize] returning token_id={fallback_token} ‹{token_text}› to draft model")
            return inference_pb2.FinalizeResponse(
                final_token=fallback_token,
                finished=sess.finished,
            )

    def GenerateFull(self, request, context):
        # baseline target-only decoding, optional
        return super().GenerateFull(request, context)

    def _generate_one_token(self, sess: TargetSession, temperature: float = 1.0, top_p: float = 0.9):
        """
        Sample one token from the target model’s distribution (temperature +
        nucleus/top‑p).  This replaces the old greedy argmax, which caused the
        same fallback tokens (e.g. “and”, token‑ID 323) to repeat and poison
        the context.

        Parameters
        ----------
        sess        : TargetSession
        temperature : float  (default = 1.0)
        top_p       : float  (default = 0.9)
        """
        self._sync_kv_pointer(sess)
        input_ids = sess.current_ids  # shape (1, L)
        out_ids = self.model.sample(
            input_ids,
            sequence_length=input_ids.shape[1] + 1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        token_id = int(out_ids[0, -1].item())

        # Advance KV cache inside the Neuron model to reflect the new token
        _, _ = self.model.forward(
            input_ids=torch.tensor([[token_id]], dtype=sess.current_ids.dtype),
            cache_ids=torch.tensor([self.model._next_pos], dtype=torch.int32)
        )
        sess.cache_ids = torch.tensor([self.model._next_pos], dtype=torch.int32)

        # Append token to context
        sess.current_ids = out_ids
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


def run_server(model_path, port=50051, sequence_length=128,
               spec_length=None, profile=False,
               temperature: float = 1.0, top_p: float = 0.9):
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading target model from {model_path} seq_len={sequence_length}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    servicer = SpeculativeServiceServicer(
        model_path,
        sequence_length=sequence_length,
        spec_length=spec_length,
        temperature=temperature,
        top_p=top_p,
    )
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server_address = f"[::]:{port}"
    logger.info(f"Target server starting on {server_address}")
    server.add_insecure_port(server_address)
    server.start()
    server.wait_for_termination()


def run_local(model_path, prompt="", max_new_tokens=50, sequence_length=128, spec_length=None, profile=False):
    # same as before
    pass