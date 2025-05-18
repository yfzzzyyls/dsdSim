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
from inference.model_loader import SPEC_LENGTH_BUCKETS   # available bucket lengths

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ───────────────────────────────────────────────────────────
# Thread-local tokenizer pool – one instance per server thread
# ───────────────────────────────────────────────────────────
_TOKENIZER_LOCAL = threading.local()
def get_thread_tokenizer(model_path: str):
    """Return a tokenizer instance unique to the current thread."""
    if not hasattr(_TOKENIZER_LOCAL, "tok"):
        _TOKENIZER_LOCAL.tok = AutoTokenizer.from_pretrained(
            model_path, use_fast=False
        )
    return _TOKENIZER_LOCAL.tok


def _gen_session_id():
    return int(uuid.uuid4()) & 0xFFFFFFFF

class TargetSession:
    def __init__(self, input_ids, row_idx: int, max_tokens: int):
        self.current_ids = input_ids  # Torch tensor [1, seq_len]
        self.finished = False
        self.tokens_generated = 0
        self.verification_time = 0.0   # cumulative time spent verifying draft tokens (seconds)
        self.finalize_calls    = 0     # count of FinalizeTokens invocations
        self.last_draft_chunk = None
        # pointer to the *next* KV slot (scalar tensor, starts at 0)
        self.cache_id = torch.tensor([0], dtype=torch.int32)
        self.pending_logits = None
        self.row_idx = row_idx
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Accessors / mutators for session‑local KV pointer
    # ------------------------------------------------------------------
    def get_session_cache_id(self):
        """
        Return the scalar tensor that holds this session's next KV slot.
        """
        return self.cache_id

    def update_session_cache_id(self, delta):
        """
        Increment the session's KV pointer by `delta`.
        `delta` may be an int, float, or 0‑D tensor.
        """
        if isinstance(delta, torch.Tensor):
            delta = int(delta.item())
        else:
            delta = int(delta)

        # Increment in place to preserve tensor identity where callers keep a reference
        self.cache_id += delta

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
        self.model_path = model_path
        tok = get_thread_tokenizer(model_path)
        self.eos_token_id = tok.eos_token_id
        self._ctx_estimate = sequence_length
        self.sessions = {}  # session_id -> TargetSession
        # ------------------------------------------------------------------
        # Scheduler state (Stage‑2 incremental batching)
        # ------------------------------------------------------------------
        self.verify_queue   = queue.Queue()               # (req_dict) items
        self.batch_timeout  = 0.03                             # seconds to wait for more peers
        self.max_batch      = batch_size                  # honour compile batch
        # ----------------------------------------------------------
        # Row‑index pool for static‑batch Neuron graph
        # ----------------------------------------------------------
        # TODO: make this dynamic, now hardcoded to 2 rows
        self._row_pool = list(range(2))          # free rows 0…B‑1

        def _allocate_row():
            assert self._row_pool, "No free Neuron batch rows left"
            return self._row_pool.pop(0)

        def _release_row(idx: int):
            self._row_pool.append(idx)

        self._allocate_row = _allocate_row
        self._release_row  = _release_row
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
    # def _pad_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    #     seq_len = input_ids.shape[1]
    #     if seq_len >= self._ctx_estimate:
    #         return input_ids                   # long enough
    #     pad_len = self._ctx_estimate - seq_len

    #     # ① choose a *legal* pad-id
    #     pad_id = self.tokenizer.pad_token_id
    #     if pad_id is None:                    # just in case
    #         pad_id = self.tokenizer.eos_token_id

    #     # ② fill token pad with pad_id  (keep -1 only for cache_id)
    #     pad_tokens = torch.full(
    #         (1, pad_len),
    #         pad_id,
    #         dtype=input_ids.dtype,
    #         device=input_ids.device,
    #     )
    #     return torch.cat([input_ids, pad_tokens], dim=1)

    def _sync_kv_pointer(self, sess: TargetSession):
        """
        Verify that the Python‑side pointer stored in `sess.cache_id`
        is identical to the Neuron model’s batched KV‑cache pointer for
        this session’s row.  Any divergence means the session and model
        are out of sync (likely a race or logic bug) and continuing
        would corrupt text generation, so we raise immediately.
        """
        # The adapter exposes the current per‑row KV pointer vector.
        model_vec = self.model.get_batched_cache_id_vec()

        row_idx = sess.row_idx
        if not (0 <= row_idx < model_vec.shape[0]):
            raise AssertionError(
                f"Row index {row_idx} is out of range for KV pointer "
                f"vector of length {model_vec.shape[0]}"
            )

        model_ptr = int(model_vec[row_idx].item())
        sess_ptr  = int(sess.get_session_cache_id().item())

        if model_ptr != sess_ptr:
            # Log first, then raise for immediate visibility.
            logger.error(
                "[KV‑DESYNC] session row=%d  model_ptr=%d  sess_ptr=%d",
                row_idx, model_ptr, sess_ptr
            )
            raise AssertionError(
                f"KV cache pointer mismatch on row {row_idx}: "
                f"model={model_ptr} vs session={sess_ptr}"
            )
        # Optional noisy trace:
        # logger.debug("[KV‑SYNC] row=%d ptr=%d", row_idx, model_ptr)


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
            session_id = _gen_session_id()
            tok = get_thread_tokenizer(self.model_path)
            enc = tok(prompt_text, return_tensors='pt')
            current_ids = enc["input_ids"]
            # Allocate a free Neuron‑batch row for this session
            row_idx = self._allocate_row()
            self.sessions[session_id] = TargetSession(current_ids, row_idx, max_tokens)
            # ------------------------------------------------------------------
            # Priming Neuron KV cache for *only* the newly‑allocated row
            # (avoid replaying prompts for other rows)
            # ------------------------------------------------------------------
            ctx_len = self._ctx_estimate          # compile‑time context length (128)
            row_idx = self.sessions[session_id].row_idx
            L_new   = current_ids.shape[1]

            # In continuous batching each sequence writes inside its own 0‑based slice,
            # so cache_ids should start at 0 for a brand‑new row.
            cache_vec = torch.arange(L_new, dtype=torch.int32,
                                     device=current_ids.device).unsqueeze(0)  # (1, L_new)

            # ---- Prefill only this row (batch size 1) ----
            seq_ids = torch.tensor([row_idx], dtype=torch.int32, device=current_ids.device)
            _ = self.model.forward(
                input_ids=current_ids,
                cache_ids=cache_vec,
                start_ids=seq_ids           # pass logical sequence‑id
            )

            # ---- Advance KV pointer for *this* row only ----
            delta = L_new
            self.model.update_batched_cache(delta, row_idx)          # target‑side vector
            self.sessions[session_id].update_session_cache_id(delta) # session‑side scalar

            # Sanity‑check KV pointer alignment after pre‑fill
            self._sync_kv_pointer(self.sessions[session_id])
        return inference_pb2.StartResponse(acknowledged=True, session_id=session_id)
    
    def _commit_tokens_bulk(self, sess, tok_ids):
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

        row_idx = sess.row_idx

        # --------------------------------------------------------------
        # Bump KV‑cache pointers by the number of tokens we just committed
        # --------------------------------------------------------------
        delta = len(tok_ids)                     # how many new tokens
        self.model.update_batched_cache(delta, row_idx)   # target side (B,)
        sess.update_session_cache_id(delta)               # draft/session side

        self._sync_kv_pointer(sess)

        # print(f"commit tokens: _next_pos={self.model._next_pos} -> current_shape{sess.current_ids.shape[1]} + {len(tok_ids)}")

        # Append committed ids to session's token history
        new_tok_tensor = torch.tensor([tok_ids], dtype=sess.current_ids.dtype)
        sess.current_ids = torch.cat([sess.current_ids, new_tok_tensor], dim=1)
        sess.tokens_generated += len(tok_ids)
        # Optionally assert pointer is correct
        if self.eos_token_id is not None and any(t == self.eos_token_id for t in tok_ids):
            sess.finished = True
            self._release_row(sess.row_idx)

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
        # If this session has now hit its max_tokens budget, force cleanup
        sess = self.sessions.get(sid)
        if sess is not None and sess.tokens_generated >= sess.max_tokens:
            finished = True
        if finished:
            # Release Neuron batch row and clean up session
            sess = self.sessions.pop(sid, None)
            if sess is not None:
                self._release_row(sess.row_idx)
            # Remove rendezvous queue entry
            self.result_queues.pop(sid, None)


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

        tokenizer = get_thread_tokenizer(self.model_path)

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

        # # Log real_B, gamma, session ids
        # logger.info(
        #     "[Batch] real_B=%d  gamma=%d  session_ids=%s",
        #     real_B, gamma, [r['session_id'] for r in batch_reqs]
        # )

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
            toks = [prev_token] + r["draft_tokens"]          # γ+1 tokens

            # ---------- Dynamic padding to the closest spec bucket ----------
            spec_buckets = sorted(SPEC_LENGTH_BUCKETS)       # e.g. [5, 8, 128]
            # choose the smallest bucket ≥ current γ+1, else fall back to largest
            desired_len = next((b for b in spec_buckets if b >= len(toks)),
                               spec_buckets[-1])
            pad_id = 0
            assert len(toks) <= desired_len, \
                f"Batch size {real_B} compiled now should be the less than max_batch {self.max_batch}"
            if len(toks) < desired_len:                      # right-pad
                toks += [pad_id] * (desired_len - len(toks))
            # ------------------------------------------------------------------

            start_pos = int(sess.get_session_cache_id().item())
            # start_pos is already the local pointer for this row (0‑based within its slice)
            vec = torch.arange(len(toks), dtype=torch.int32) + start_pos
            # (no row‑offset added because continuous batching routes by seq_id)

            # Decode the first five token-ids to human‑readable strings
            # token_words = [
            #     self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            #     for tid in toks[:5]
            # ]
            # logger.info(
            #     "[BatchRow %d] sid=%s  ids=%s  words=%s  cache_vec=%s",
            #     len(sess_list), sid, toks[:5], token_words, vec[:5].tolist()
            # )

            input_ids.append(torch.tensor(toks, dtype=torch.int32))
            cache_vecs.append(vec)
            sess_list.append(sess)

        # --------------------------------------------------------------
        # Pad *lists* before stacking so the final stacked tensor keeps
        # the original (B, N, V) layout coming out of Neuron.  This avoids
        # collapsing the vocab axis down to 2 columns.
        # --------------------------------------------------------------
        assert real_B <= self.max_batch, \
            f"Batch size {real_B} compiled now should be the less than max_batch {self.max_batch}"
        # REMOVE dummy-row padding: do not pad input_ids, cache_vecs, or sess_list
        input_ids  = torch.stack(input_ids, 0)        # (B, γ+1)
        cache_vecs = torch.stack(cache_vecs, 0)       # (B, γ+1)

        # Add start_ids tensor for batch
        start_ids = torch.tensor(
            [s.row_idx for s in sess_list],
            dtype=torch.int32,
            device=input_ids.device
        )

        # ------------------------------------------------------------------
        # Batched speculative_forward
        # ------------------------------------------------------------------
        # --------------------------------------------------------------
        # DEBUG: show the raw tensors that will be fed to speculative_forward
        # --------------------------------------------------------------
        # logger.info(
        #     "[SpecForward] γ=%d  input_ids=%s  cache_vecs(row0…)=%s",
        #     gamma,
        #     input_ids.tolist(),
        #     cache_vecs.tolist()[:2]   # print first 2 rows to avoid log spam
        # )
        t0 = time.perf_counter()
        logits = self.model.speculative_forward(
            input_ids = input_ids,
            cache_ids = cache_vecs,
            start_ids = start_ids,          # continuous batching
            spec_length = input_ids.size(1),   # use the padded/truncated length
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
            # ----------------------------------------------------------
            # DEBUG: sample *one* token from the target distribution for
            #        each position i (0…γ) so we can see what the large
            #        model "wants" to emit.  Decode to words.
            # ----------------------------------------------------------
            # sampled_ids = []
            # sampled_words = []
            # for i in range(all_row_probs.size(0)):
            #     samp_id = int(torch.multinomial(all_row_probs[i], 1).item())
            #     sampled_ids.append(samp_id)
            #     sampled_words.append(
            #         self.tokenizer.decode([samp_id], clean_up_tokenization_spaces=False)
            #     )
            # logger.info(
            #     "[TargetSample] sid=%s  sampled_ids=%s  words=%s",
            #     sid, sampled_ids, sampled_words
            # )
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
            # logger.info(
            #     f"[ACCEPTANCE DEBUG] "
            #     f"accept={accept.cpu().tolist()} "
            #     f"reject_indices={(rej.squeeze(-1).cpu().tolist() if rej.numel() else [])} "
            #     f"first_rej={first_rej}"
            # )


            accepted_cnt = first_rej
            committed    = draft_tokens[:accepted_cnt]

            if accepted_cnt < len(draft_tokens):
                bonus_id = int(torch.multinomial(all_row_probs[first_rej], 1).item())
            else:
                bonus_id = int(torch.multinomial(all_row_probs[-1], 1).item())
            committed.append(bonus_id)

            # ----------------------------------------------------------
            # DEBUG: Show draft proposals, target predictions, accept/reject,
            #        and final committed chunk in **human‑readable words**
            # ----------------------------------------------------------
            draft_words = [
                tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                for tid in draft_tokens
            ]
            tgt_preds = torch.argmax(tgt_row_probs, dim=-1).tolist()
            tgt_words = [
                tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                for tid in tgt_preds
            ]
            committed_words = [
                tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                for tid in committed
            ]
            # status = (
            #     "all_accepted"
            #     if accepted_cnt == len(draft_tokens)
            #     else f"rejected_from_pos_{accepted_cnt}"
            # )
            # logger.info(
            #     "[Verify] sid=%s  status=%s\n"
            #     "  draft  = %s\n"
            #     "  target = %s\n"
            #     "  commit = %s",
            #     sid, status,
            #     draft_words, tgt_words, committed_words,
            # )

            self._commit_tokens_bulk(sess, committed)
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
