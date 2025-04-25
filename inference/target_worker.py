import logging
import torch
from concurrent import futures
import grpc
import time
from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc
import random      # ← NEW

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
        self.model = model_loader.load_target_model(model_path,
                                            sequence_length=sequence_length)
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        self._ctx_estimate = sequence_length
        self.sessions = {}  # session_id -> TargetSession
        self.lock = torch.multiprocessing.Lock()
        self._vocab_size = len(self.tokenizer)

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
                sid          = request.session_id
                draft_tokens = list(request.draft_tokens)
                draft_probs  = list(request.draft_probs) if hasattr(request, "draft_probs") else []

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

    def _commit_tokens_bulk(self, sess, tok_ids):
        """
        Commit a list of tokens (accepted draft + bonus) in ONE Neuron
        speculative_forward call so the KV cache advances in bulk.
        We use `speculative_forward` (not plain forward) to bypass the
        context‑bucket check that requires input length ≥ ctx_estimate.
        """
        logger.info("Commit raw tok_ids=%s", tok_ids)
        tok_ids = [t for t in tok_ids if t != self.tokenizer.bos_token_id]
        if not tok_ids:
            return
        
        # ===============================================================
        # Discover the compiled speculation bucket sizes ONCE per call.
        # ===============================================================
        inner = self.model.adapter.model
        bucket_lengths = {k[0] if isinstance(k, tuple) else int(k)
                            for k in inner.decoder_lm_head_for_speculation.keys()}
        # ---------------------------------------------------------------
        # Strip any placeholder IDs before sanity checks
        # ---------------------------------------------------------------

        # Sanity checks
        assert len(tok_ids) in bucket_lengths, \
            f"Commit length {len(tok_ids)} not compiled; buckets={sorted(bucket_lengths)}"
        assert all(0 < t < self._vocab_size for t in tok_ids), \
            f"OOV or placeholder token in commit_ids: {tok_ids} (vocav_size={self._vocab_size})"

        self._sync_kv_pointer(sess)

        # (1, K) tensor of the new tokens
        in_tensor = torch.tensor([tok_ids], dtype=sess.current_ids.dtype)

        # Positions of these K new tokens start at current _next_pos
        cache_vec = torch.arange(len(tok_ids), dtype=torch.int32) + self.model._next_pos

        # ------------------------------------------------------------------
        # Figure out which speculation buckets were actually compiled.
        # self.model is a NeuronHFAdapterWrap → .adapter → HFAdapter →
        # .model (the underlying LlamaForSampling).
        # ------------------------------------------------------------------
        
        inner = self.model.adapter.model
        raw_keys = inner.decoder_lm_head_for_speculation.keys()
        # Accept both (k, batch_size) and k-only keys
        def _extract_k(k):
            if isinstance(k, tuple):
                return k[0]
            return k
        spec_ok = len(tok_ids) in {_extract_k(k) for k in raw_keys}
        assert spec_ok, f"speculative_forward not compiled for {len(tok_ids)} tokens"

        # ONE speculative_forward advances the cache and avoids the
        # `context_length (…) shouldn't be smaller than estimate` error
        _ = self.model.speculative_forward(
            input_ids=in_tensor,
            cache_ids=cache_vec,
            spec_length=len(tok_ids),
        )

        # ----- Update KV pointer and session history -----
        new_next_pos = int(self.model._next_pos) + len(tok_ids)
        self.model._next_pos = new_next_pos          # advance global pointer
        sess.cache_ids = torch.tensor([new_next_pos], dtype=torch.int32)

        # Append committed ids to session's token history
        new_tok_tensor = torch.tensor([tok_ids], dtype=sess.current_ids.dtype)
        sess.current_ids = torch.cat([sess.current_ids, new_tok_tensor], dim=1)

        if self.eos_token_id is not None and any(t == self.eos_token_id for t in tok_ids):
            sess.finished = True

        logger.info("Committed %d tokens; _next_pos -> %d", len(tok_ids), new_next_pos)

    def _verify_single_step(self, sess: TargetSession, draft_tokens):
        """
        Fast path: score all draft_tokens and bonus in ONE forward pass.
        Returns
        -------
        probs : List[float]   – P_target(d_i | prefix + d_<i)   for each i
        bonus_probs : tensor  – P_target(vocab) for bonus token
        """
        # ---------- short‑circuit ----------
        if not draft_tokens:
            return [], None

        orig_cache   = sess.cache_ids.clone()
        orig_nextpos = int(orig_cache.item())
        logits_next  = sess.pending_logits          # may be None
        sess.pending_logits = None                  # consume
        self._sync_kv_pointer(sess)
        pad_id = self.eos_token_id if self.eos_token_id is not None else 0
        n_new = len(draft_tokens) + 1          # γ + 1 rows
        bonus_placeholder = self.eos_token_id
        assert self.eos_token_id is not None, \
            "Bonus placeholder must be set to a valid token ID"
        input_ids = torch.tensor([draft_tokens + [bonus_placeholder]],
                                dtype=sess.current_ids.dtype)
        spec_len  = n_new
        cache_vec = torch.arange(spec_len, dtype=torch.int32) + orig_nextpos
        logger.info("TARGET verify call K=%d input_ids=%s", input_ids.shape[1], input_ids.tolist())
        logits_all = self.model.speculative_forward(
            input_ids=input_ids,
            cache_ids=cache_vec,
            spec_length=n_new,
        )
        if logits_all.dim() == 3:
            logits_all = logits_all.squeeze(-1)   # shape -> (N, V)
        # The last row (index -1) corresponds to the bonus placeholder.
        bonus_logits = logits_all[-1]        # row γ  → bonus/fallback
        draft_logits = logits_all[:-1]       # first γ rows (one per draft token)
        with torch.no_grad():
            row_probs   = torch.softmax(draft_logits.float(), dim=-1)
            bonus_probs = torch.softmax(bonus_logits.float(), dim=-1)
        probs = [float(row_probs[i, tok].item()) for i, tok in enumerate(draft_tokens)]
        # ---------- restore snapshot ----------
        self.model.cache_ids = orig_cache.clone()
        if hasattr(self.model, "_next_pos"):
            self.model._next_pos = orig_nextpos
        sess.cache_ids = orig_cache
        assert int(self.model.cache_ids.item()) == int(sess.cache_ids.item()), \
            "KV desync detected on verify exit"
        return probs, bonus_probs

    def VerifyDraftTokens(self, request, context):
        sid          = request.session_id
        draft_tokens = list(request.draft_tokens)
        draft_probs  = list(request.draft_probs) if hasattr(request, "draft_probs") else []

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

            committed     = []
            accepted_cnt  = 0

            # ---- ONE verification pass for the entire chunk + bonus ----
            probs, bonus_probs = self._verify_single_step(sess, draft_tokens)
            # Probabilistic acceptance:
            for i, (tok, p_tgt) in enumerate(zip(draft_tokens, probs)):
                q_draft = draft_probs[i] if i < len(draft_probs) else 0.0
                if q_draft <= 0.0:
                    accept = (p_tgt >= 1e-3)
                elif p_tgt >= q_draft:
                    accept = True
                else:
                    accept = random.random() < (p_tgt / q_draft)
                if accept:
                    accepted_cnt += 1
                    committed.append(tok)
                    if self.eos_token_id == tok:
                        break
                else:
                    # first rejection → sample bonus token from precomputed bonus_probs
                    bonus_id = int(torch.multinomial(bonus_probs, 1).item())
                    committed.append(bonus_id)
                    break
            else:
                # all accepted → sample bonus token from bonus_probs
                bonus_id = int(torch.multinomial(bonus_probs, 1).item())
                committed.append(bonus_id)

            # Bulk commit all tokens at once
            logger.debug("FINAL commit_ids=%s", committed)
            self._commit_tokens_bulk(sess, committed)
            logger.info("After commit, _next_pos=%d", int(self.model._next_pos))


            # ------------------------------------------------------------------
            # Filter out BOS (<|begin_of_text|>) and any placeholder / OOV ids
            # **before** sending them back to the draft client.
            # ------------------------------------------------------------------
            safe_ids = [t for t in committed
                        if 0 < t < self._vocab_size
                        and t != self.tokenizer.bos_token_id]

            return inference_pb2.VerifyResponse(committed_ids=safe_ids,
                                                accepted_count=accepted_cnt,
                                                finished=sess.finished)

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
        # Fast‑path: if pending_logits is set, sample directly without a forward
        if sess.pending_logits is not None:
            probs = torch.softmax(sess.pending_logits.float() / max(1e-5, temperature), dim=-1)
            token_id = int(torch.multinomial(probs, 1).item())
            sess.pending_logits = None   # consume
            # Advance KV in a lightweight call
            self._commit_tokens_bulk(sess, [token_id])
            return token_id
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