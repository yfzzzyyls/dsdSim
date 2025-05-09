import logging
import torch
from concurrent import futures
import grpc
import time
from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc
import random
from transformers.generation import LogitsProcessorList, SuppressTokensLogitsProcessor
import queue
import threading
import uuid


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
    def __init__(self, model_path, sequence_length=128, spec_length=None,
                 batch_size: int = 1, temperature: float = 1.0, top_p: float = 0.9):
        self.model = model_loader.load_target_model(
            model_path,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        self._ctx_estimate = sequence_length
        self.sessions = {}  # session_id -> TargetSession
        # ------------------------------------------------------------------
        # Scheduler state (Stage‑2 incremental batching)
        # ------------------------------------------------------------------
        self.verify_queue   = queue.Queue()               # (req_dict) items
        self.batch_timeout  = 0.03                             # seconds to wait for more peers
        self.max_batch      = batch_size                  # honour compile batch
        # map session_id -> Queue for the blocking VerifyDraftTokens call
        self.result_queues  = {}
        self._sched_thread  = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self._sched_thread.start()
        self.lock = torch.multiprocessing.Lock()

    # ------------------------------------------------------------------
    # Utility: right‑pad an (1, L) tensor with zeros to ctx_estimate
    # ------------------------------------------------------------------
    def _pad_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        if seq_len >= self._ctx_estimate:
            return input_ids                   # long enough
        pad_len = self._ctx_estimate - seq_len

        # ① choose a *legal* pad-id
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:                    # just in case
            pad_id = self.tokenizer.eos_token_id

        # ② fill token pad with pad_id  (keep -1 only for cache_ids)
        pad_tokens = torch.full(
            (1, pad_len),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, pad_tokens], dim=1)

    def _sync_kv_pointer(self, sess: TargetSession):
        # ---- sanity check ----
        assert int(self.model.cache_ids.item()) == int(sess.cache_ids.item()), \
            "Target KV cache_ids desynchronised after sync"


    def StartGeneration(self, request, context):
        session_id = request.session_id
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens
        gamma = request.gamma
        logger.debug(f"[session={session_id}] StartGeneration: prompt='{prompt_text}', max_new_tokens={max_tokens}, gamma={gamma}")
        with self.lock:
            assert session_id == 0, \
                f"Session id {session_id} must be 0 for StartGeneration."
            assert not session_id in self.sessions, \
                f"Session {session_id} should not exists."
            assert prompt_text is not None, \
                f"Prompt text is required for session."
            # Generate a fresh, 32-bit random id
            session_id = int(uuid.uuid4()) & 0xFFFFFFFF
            enc = self.tokenizer(prompt_text, return_tensors='pt')
            current_ids = enc["input_ids"]
            self.sessions[session_id] = TargetSession(current_ids)
            # ------------------------------------------------------------------
            # Priming Neuron KV cache for the prompt
            # ------------------------------------------------------------------
            self.model.cache_ids = None
            self.model._next_pos = 0

            # propmt shape: (1, L)  where L = prompt length
            L = current_ids.shape[1]
            # ----------------------------------------------------------
            # Build a (2, L) batched prompt so it matches the compiled
            # batch_size=2 graphs.  Row‑0 = real prompt, Row‑1 = PAD.
            # TODO: Should be (N, L) in the future
            # ----------------------------------------------------------
            pad_id = self.tokenizer.pad_token_id or 0
            pad_row = torch.full_like(current_ids, pad_id)     # (1, L)
            batched_ids = torch.cat([current_ids, pad_row], dim=0)   # (2, L)

            # 2‑D cache_ids tensor (2, L): each row [0 … L‑1]
            cache_vec = torch.arange(L, dtype=torch.int32).unsqueeze(0).repeat(2, 1)

            _ = self.model.forward(
                input_ids=batched_ids,
                cache_ids=cache_vec,
            )

            # --- MANUALLY align wrapper pointer with the prompt length ---
            next_pos = current_ids.shape[1]                       # L
            self.model.update_cache(torch.tensor([next_pos], dtype=torch.int32),
                                    next_pos)
            
            # self.model._next_pos = next_pos
            # self.model.cache_ids = torch.tensor([next_pos], dtype=torch.int32)

            # record in session
            self.sessions[session_id].cache_ids = self.model.cache_ids.clone()
        return inference_pb2.StartResponse(acknowledged=True, session_id=session_id)
    
    def _commit_tokens_bulk(self, sess, tok_ids, row_idx: int):
        """
        Commit a list of tokens (accepted draft + bonus) in ONE Neuron
        speculative_forward call so the KV cache advances in bulk.
        We call `forward` (not speculative_forward) here so that the Neuron
        wrapper's _postprocess() updates the Python‑side KV pointer.
        """
        # Skip dummy padding rows (should never be called with None)
        if sess is None:
            return

        if not tok_ids:
            return

        # Manually bump ONLY the cache‑ids row that belongs to this session
        orig_next_pos = int(sess.cache_ids.item())
        new_next_pos  = orig_next_pos + len(tok_ids)

        # update session's cache_ids, and the model's cache_ids
        sess.cache_ids = torch.tensor([new_next_pos], dtype=torch.int32)
        self.model.update_cache(sess.cache_ids, new_next_pos)

        self._sync_kv_pointer(sess)

        # print(f"commit tokens: _next_pos={self.model._next_pos} -> current_shape{sess.current_ids.shape[1]} + {len(tok_ids)}")

        # Append committed ids to session's token history
        new_tok_tensor = torch.tensor([tok_ids], dtype=sess.current_ids.dtype)
        sess.current_ids = torch.cat([sess.current_ids, new_tok_tensor], dim=1)

        # Optionally assert pointer is correct
        if self.eos_token_id is not None and any(t == self.eos_token_id for t in tok_ids):
            sess.finished = True

        # # --------------------------------------------------------------
        # # 9) Detailed commit log: committed tokens *and* full context words
        # # --------------------------------------------------------------
        # current_words = [
        #     self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        #     for tid in sess.current_ids.squeeze(0).tolist()
        # ]
        # logger.debug(
        #     "Committed tokens (text)=%s  ids=%s; _next_pos -> %d | current_ids (text)=%s",
        #     token_texts,
        #     tok_ids,
        #     self.model._next_pos,
        #     current_words,
        # )

    def verify(self, sess: TargetSession, draft_tokens):
        """
        Fast path: score all draft_tokens and bonus in ONE forward pass.
        Returns
        -------
        probs : List[float]   - P_target(d_i | prefix + d_<i)   for each i
        bonus_probs : tensor  - P_target(vocab) for bonus token
        """
        # ---------- short‑circuit ----------
        if not draft_tokens:
            return [], None

        # ==========================================
        # get the last commit KV cache position
        # ==========================================
        orig_cache   = sess.cache_ids.clone()
        orig_nextpos = int(orig_cache.item())

        # ==========================================
        # set indices for speculative forward
        # ==========================================
        prev_token_id = int(sess.current_ids[0, -1])
        spec_tokens   = [prev_token_id] + draft_tokens          # γ + 1 tokens
        spec_len      = len(spec_tokens)

        input_ids = torch.tensor([spec_tokens], dtype=sess.current_ids.dtype)
        cache_vec = torch.arange(spec_len, dtype=torch.int32) + orig_nextpos - 1

        # ------------------------------------
        # ------------------------------------
        assert cache_vec.numel() == spec_len, (
            f"VERIFY cache_vec length {cache_vec.numel()} must equal spec token length {spec_len}"
        )
        logger.debug(
            f"VERIFY cache_vec length {cache_vec.numel()} must equal spec token length {spec_len}"
        )
        # # ------------------------------------
        # # ------------------------------------
        # token_texts = [self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        #        for tid in spec_tokens]
        # logger.debug("verify call K(gamma[%d] + 1)=%d tokens(text)=%s ids=%s",
        #             len(draft_tokens), input_ids.shape[1], token_texts, input_ids.tolist())
        # #------------------------------------
        # # ------------------------------------
        
        logits_all = self.model.speculative_forward(
            input_ids=input_ids,
            cache_ids=cache_vec,
            # start_ids = torch.tensor([0], dtype=torch.int32), # can pass multiple batches in the future
            spec_length=spec_len,
        )
        
        # (B, N, V)  → after squeeze  (N, V) where N = γ + 1
        if logits_all.dim() == 3:
            logger.debug(f"speculative_forward logits_all shape={logits_all.shape}")
            logits_all = logits_all.squeeze(-1)          # (N, V)

        #-----------------------------------
        # print the shape of the logits
        #-----------------------------------
        # logger.debug("verify logits_all shape=%s", logits_all.shape)

        # ------------------------------------------------------------------
        # Library-style masking of BOS / PAD with SuppressTokensLogitsProcessor
        # ------------------------------------------------------------------
        # special_ids = []
        # for attr in ("bos_token_id", "pad_token_id"):
        #     tid = getattr(self.tokenizer, attr, None)
        #     if tid is not None:
        #         special_ids.append(tid)

        # if special_ids:
        #     processors = LogitsProcessorList(
        #         [SuppressTokensLogitsProcessor(special_ids)]
        #     )
        #     # dummy_input_ids shape (N,1) – only the seq-len matters
        #     dummy_input_ids = torch.zeros(
        #         (logits_all.size(0), 1), dtype=torch.long, device=logits_all.device
        #     )
        #     logits_all = processors(dummy_input_ids, logits_all)
        # ===========================================================

        # # ---------- restore snapshot ----------
        # self.model.cache_ids = orig_cache.clone()
        # self.model._next_pos = orig_nextpos
        # sess.cache_ids = orig_cache
        # # self.model.adapter.model.reset_cache(orig_cache)   # hypothetical helper
        # assert int(self.model.cache_ids.item()) == int(sess.cache_ids.item()), \
        #     "KV desync detected on verify exit"
        
        return logits_all

    def VerifyDraftTokens(self, request, context):
        """
        Non-blocking enqueue: place request in scheduler queue, wait for result.
        """
        start_verify_t = time.perf_counter()
        sid           = request.session_id
        draft_tokens  = list(request.draft_tokens)
        draft_probs   = list(request.draft_probs)

        if not draft_tokens:
            return inference_pb2.VerifyResponse(
                committed_ids=[], accepted_count=0,
                verify_time_ms=0.0, finished=True
            )

        # Prepare per‑call rendez‑vous Queue
        resp_q = queue.Queue(maxsize=1)
        self.result_queues[sid] = resp_q

        # Enqueue work for the scheduler thread
        self.verify_queue.put({
            "session_id":    sid,
            "draft_tokens":  draft_tokens,
            "draft_probs":   draft_probs,
            "enqueue_time":  start_verify_t,
        })

        # Block until scheduler puts result
        committed_ids, accepted_cnt, verify_ms, finished = resp_q.get()

        return inference_pb2.VerifyResponse(
            committed_ids=committed_ids,
            accepted_count=accepted_cnt,
            verify_time_ms=verify_ms,
            finished=finished,
        )

    # ------------------------------------------------------------------
    # STAGE‑2: scheduler thread that batches requests
    # ------------------------------------------------------------------
    def _scheduler_loop(self):
        """
        Drain self.verify_queue; batch together all requests that (a) arrive
        within self.batch_timeouts *or* (b) fill up to self.max_batch items.
        For the first implementation we assume *all* requests use the same
        γ (chunk length) so they share one compiled Neuron graph.
        """
        pending = []
        last_pop = time.monotonic()
        while True:
            try:
                # Non‑blocking get with small timeout so we can flush
                req = self.verify_queue.get(timeout=self.batch_timeout)
                pending.append(req)
                # Flush if we filled the batch
                flush = len(pending) >= self.max_batch
            except queue.Empty:
                flush = len(pending) > 0

            now = time.monotonic()
            if flush or (pending and (now - last_pop) >= self.batch_timeout):
                self._process_batch(pending)
                pending.clear()
                last_pop = now

    def _process_batch(self, batch_reqs):
        """
        batch_reqs: List[dict] with keys session_id, draft_tokens, draft_probs.
        Performs one batched speculative_forward and writes result to each
        caller's result_queue.
        """
        if not batch_reqs:
            return

        # Group tensors
        sess_list      = []
        draft_tok_lens = {len(r["draft_tokens"]) for r in batch_reqs}
        # Simple: require identical γ; otherwise process sequentially.
        # TODO: Changed to "padding-to-max" and adjust accept/reject rule to ignore padded slots. 
        assert len(draft_tok_lens) == 1, \
            f"Only support same γ for multiple draft models: {draft_tok_lens} (expected 1)"

        gamma = draft_tok_lens.pop()
        B     = len(batch_reqs)
        real_B = B            # remember how many *real* rows we have

        # Log real_B, gamma, session ids
        logger.info(
            "[Batch] real_B=%d  gamma=%d  session_ids=%s",
            real_B, gamma, [r['session_id'] for r in batch_reqs]
        )

        # ------------------------------------------------------------------
        # Build input tensors (B, γ+1) and cache_vec (B, γ+1)
        # ------------------------------------------------------------------
        input_ids  = []
        cache_vecs = []
        for r in batch_reqs:
            sid  = r["session_id"]
            sess = self.sessions[sid]
            self._sync_kv_pointer(sess)

            prev_token = int(sess.current_ids[0, -1].item())
            toks = [prev_token] + r["draft_tokens"]          # γ+1

            start_pos = int(sess.cache_ids.item())
            vec = torch.arange(len(toks), dtype=torch.int32) + start_pos

            # Decode the first five token-ids to human‑readable strings
            token_words = [
                self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                for tid in toks[:5]
            ]
            logger.info(
                "[BatchRow %d] sid=%s  ids=%s  words=%s  cache_vec=%s",
                len(sess_list), sid, toks[:5], token_words, vec[:5].tolist()
            )

            input_ids.append(torch.tensor(toks, dtype=torch.int32))
            cache_vecs.append(vec)
            sess_list.append(sess)

        # --------------------------------------------------------------
        # Pad *lists* before stacking so the final stacked tensor keeps
        # the original (B, N, V) layout coming out of Neuron.  This avoids
        # collapsing the vocab axis down to 2 columns.
        # Also pad sess_list with None for dummy rows.
        # --------------------------------------------------------------
        assert real_B <= self.max_batch, \
            f"Batch size {real_B} compiled now should be the less than max_batch {self.max_batch}"
        if real_B < self.max_batch:
            pad_n   = self.max_batch - real_B
            pad_ids = input_ids[0].clone()
            pad_vec = cache_vecs[0].clone()
            for _ in range(pad_n):
                input_ids.append(pad_ids)
                cache_vecs.append(pad_vec)
                sess_list.append(None)          # placeholder for dummy row

        input_ids  = torch.stack(input_ids, 0)        # (B, γ+1)
        cache_vecs = torch.stack(cache_vecs, 0)       # (B, γ+1)

        # ------------------------------------------------------------------
        # Batched speculative_forward
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        logits = self.model.speculative_forward(
            input_ids   = input_ids,
            cache_ids   = cache_vecs,
            spec_length = gamma + 1,
        )
        # logger.info("[scheduler - process_batch] speculative_forward raw shape = %s", tuple(logits.shape))
        # Raw Neuron layout is (N, V, B)  where:
        #   N = γ + 1, V = vocab shards (≈ vocab_size × TP), B = batch
        # We keep this layout to avoid the transpose overhead.
        verify_ms = (time.perf_counter() - t0) * 1000.0

        # logits: (N, V, B)  → split per session
        for b, req in enumerate(batch_reqs):
            # Skip the padded dummy rows – they have no corresponding request
            sess = sess_list[b]
            if sess is None:
                continue
            sid          = req["session_id"]
            draft_tokens = req["draft_tokens"]
            draft_probs  = req["draft_probs"]

            # logits[:, :, b] → (N, V) for this session
            all_row_probs = torch.softmax(
                logits[:, :, b].float(), dim=-1
            )                               # (γ+1, V)
            tgt_row_probs = all_row_probs[:-1]

            device = tgt_row_probs.device
            dr_tokens = torch.tensor(draft_tokens, device=device)
            row_idx   = torch.arange(len(draft_tokens), device=device)
            p_tgt   = tgt_row_probs[row_idx, dr_tokens]
            q_draft = torch.tensor(draft_probs, device=device)

            ratio  = p_tgt / q_draft
            rand_v = torch.rand_like(ratio)
            accept = (p_tgt >= q_draft) | (rand_v < ratio)
            rej    = (~accept).nonzero(as_tuple=False)
            first_rej = int(rej[0].item()) if rej.numel() > 0 else len(draft_tokens)

            accepted_cnt = first_rej
            committed    = draft_tokens[:accepted_cnt]

            if accepted_cnt < len(draft_tokens):
                bonus_id = int(torch.multinomial(all_row_probs[first_rej], 1).item())
            else:
                bonus_id = int(torch.multinomial(all_row_probs[-1], 1).item())
            committed.append(bonus_id)

            self._commit_tokens_bulk(sess, committed, b)
            finished = sess.finished

            # Respond to the waiting gRPC handler
            self.result_queues[sid].put(
                (committed, accepted_cnt, verify_ms, finished)
            )

def run_server(model_path, port=50051, sequence_length=128,
               spec_length=None, profile=False,
               temperature: float = 1.0, top_p: float = 0.9,
               batch_size: int = 1):
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Initializing target server with model: {model_path}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    servicer = SpeculativeServiceServicer(
        model_path,
        sequence_length=sequence_length,
        spec_length=spec_length,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
    )
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server_address = f"[::]:{port}"
    logger.info(f"Target server starting on {server_address}")
    server.add_insecure_port(server_address)
    server.start()
    server.wait_for_termination()
