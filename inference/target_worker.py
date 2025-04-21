import logging
import torch
from concurrent import futures
import grpc
import time
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
        self.verification_time = 0.0   # cumulative time spent verifying draft tokens (seconds)
        self.finalize_calls    = 0     # count of FinalizeTokens invocations
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

    # ------------------------------------------------------------------
    # Utility: right‑pad an (1, L) tensor with zeros to ctx_estimate
    # ------------------------------------------------------------------
    def _pad_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Neuron‑compiled forward graphs expect the input length to be
        >= the compile‑time estimate (self._ctx_estimate, defaults to the
        --sequence_length used at compile time).  If the supplied tensor
        is shorter we right‑pad with zeros so its shape is (1, ctx_estimate).
 
        Parameters
        ----------
        input_ids : torch.Tensor   shape (1, L), dtype = same as model input
 
        Returns
        -------
        torch.Tensor  shape (1, max(L, ctx_estimate))
        """
        seq_len = input_ids.shape[1]
        if seq_len >= self._ctx_estimate:            # already long enough
            return input_ids
        pad_len = self._ctx_estimate - seq_len
        pad = torch.zeros((1, pad_len), dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, pad], dim=1)

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
        ONE forward pass:
          – returns P_target for every draft token, and
          – provides logits for the next position (row N).
        Assumes Neuron graph emits len(draft_tokens)+1 rows of logits.
        """
        if not draft_tokens:
            return [], []

        n_new = len(draft_tokens)
        orig_cache = sess.cache_ids.clone()
        start_pos  = int(orig_cache.item())

        ids_tensor = torch.tensor([draft_tokens], dtype=sess.current_ids.dtype)
        pos_tensor = torch.arange(start_pos, start_pos + n_new,
                                  dtype=torch.int32).unsqueeze(0)

        padded_ids = self._pad_ids(ids_tensor)
        pad_cache  = torch.full_like(padded_ids, -1, dtype=torch.int32)
        pad_cache[0, : n_new] = pos_tensor[0]
        if pad_cache.ndim == 2:
            pad_cache = pad_cache.squeeze(0)

        logits_all, _ = self.model.forward(
            input_ids=padded_ids,
            cache_ids=pad_cache,
        )                           # shape (N+1, vocab)
        logits_all = logits_all[: n_new + 1].float()

        # probs for each draft token
        row_slice = logits_all[:n_new]            # could be 1‑D when n_new == 1
        row_probs = torch.softmax(row_slice, dim=-1)

        if row_probs.dim() == 1:                  # only one draft token
            probs = [float(row_probs[tok].item())]
        else:
            probs = [
                float(row_probs[i, tok].item())
                for i, tok in enumerate(draft_tokens)
            ]

        logits_rows = [logits_all[i].clone() for i in range(n_new + 1)]
        sess.pending_logits = logits_rows[-1]

        # restore pointer
        sess.cache_ids = orig_cache.clone()
        self._sync_kv_pointer(sess)
        return probs, logits_rows

    def VerifyDraftTokens(self, request, context):
        sid = request.session_id
        draft_tokens = list(request.draft_tokens)

        with self.lock:
            if sid not in self.sessions:
                return inference_pb2.VerifyResponse(committed_ids=[],
                                                    accepted_count=0,
                                                    finished=True)

            sess = self.sessions[sid]
            if sess.finished or not draft_tokens:
                return inference_pb2.VerifyResponse(committed_ids=[],
                                                    accepted_count=0,
                                                    finished=sess.finished)

            committed    = []
            accepted_cnt = 0

            # ----- single forward pass (scores + per‑step logits) -----
            probs_all, logits_rows = self._verify_single_step(sess, draft_tokens)

            # Walk until first rejection
            for idx, tok in enumerate(draft_tokens):
                current_logits = logits_rows[idx]
                p_target = probs_all[idx]
                if p_target >= 1e-3:
                    accepted_cnt += 1
                    self._commit_token(sess, tok)
                    committed.append(tok)
                    if tok == self.eos_token_id:
                        break
                else:
                    # first rejection → use logits at this position
                    fallback = self._sample_from_logits(
                        current_logits,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    self._commit_token(sess, fallback)
                    committed.append(fallback)
                    break
            else:
                # all accepted → use logits_rows[-1] as bonus token
                bonus = self._sample_from_logits(
                    logits_rows[-1],
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                self._commit_token(sess, bonus)
                committed.append(bonus)

            return inference_pb2.VerifyResponse(
                committed_ids=committed,
                accepted_count=accepted_cnt,
                finished=sess.finished,
            )

    # helper used above
    def _commit_token(self, sess, tok_id):
        tok = torch.tensor([[tok_id]], dtype=sess.current_ids.dtype)
        sess.current_ids = torch.cat([sess.current_ids, tok], dim=1)
        self._sync_kv_pointer(sess)
        _, _ = self.model.forward(input_ids=tok,
                                  cache_ids=torch.tensor([self.model._next_pos],
                                                         dtype=torch.int32))
        sess.cache_ids = torch.tensor([self.model._next_pos], dtype=torch.int32)
        if self.eos_token_id == tok_id:
            sess.finished = True

    def _sample_from_logits(self, logits, temperature=1.0, top_p=0.9):
        logits = logits / max(1e-8, temperature)
        probs  = torch.softmax(logits, dim=-1)

        # nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = torch.searchsorted(cumsum, top_p, right=True).item()
        keep_idx   = sorted_idx[: cutoff + 1]
        keep_probs = sorted_probs[: cutoff + 1]
        keep_probs = keep_probs / keep_probs.sum()

        choice = torch.multinomial(keep_probs, 1).item()
        return int(keep_idx[choice].item())

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
            start_t = time.perf_counter()
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
            # Log cumulative verification latency **once** when the session ends
            if sess.finished:
                logger.info("[session=%s] total verification latency: %.3f s",
                            sid, sess.verification_time)

            # ---------- periodic verification‑time log ----------
            sess.finalize_calls += 1
            should_log = (
                sess.finished or
                sess.finalize_calls % 10 == 0 or
                (accepted_count == 0 and draft_chunk_size == 0)   # client flush / end
            )
            if should_log:
                logger.info(
                    "[session=%s] cumulative verification latency: %.3f s  calls=%d",
                    sid, sess.verification_time, sess.finalize_calls
                )

            token_text = self.tokenizer.decode([fallback_token]).strip() if fallback_token != 0 else "<none>"
            logger.debug(f"[Finalize] returning token_id={fallback_token} ‹{token_text}› to draft model")
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