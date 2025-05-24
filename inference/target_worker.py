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
from inference.model_loader import SPEC_LENGTH_BUCKETS, BATCH_BUCKETS   # available bucket lengths
import collections


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)





# ───────────────────────────────────────────────
# Simple thread‑safe scheduler with peek + batch
# ───────────────────────────────────────────────
class Scheduler:
    """
    Internal queue for the target server.

    • enqueue(job_dict) – append a job (dict with at least "kind").
    • pop_batch(max_batch) – block up to batch_timeout and return
      (kind, [jobs]) where all jobs share the same "kind" and the list
      length ≤ max_batch.  FIFO order across kinds is preserved.
    """
    def __init__(self, batch_timeout: float = 0.03):
        self._q   = collections.deque()
        self._cv  = threading.Condition()
        self._tmo = batch_timeout
        self._seq = 0  # global FIFO sequence number

    # ---------- producer ----------
    def enqueue(self, job: dict):
        with self._cv:
            job["_seq"] = self._seq; self._seq += 1
            self._q.append(job)
            self._cv.notify()

    # ---------- consumer ----------
    def pop_batch(self, max_batch: int):
        """Return (kind, batch_list) or (None, []) on timeout."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._q, timeout=self._tmo):
                return None, []

            # Oldest job decides the batch kind
            batch_kind = self._q[0]["kind"]
            batch = []

            # Scan the deque and collect up-to max_batch jobs of that kind,
            # even if other kinds are interleaved.  Non-matching jobs stay
            # in place so their FIFO order is preserved.
            i = 0
            while i < len(self._q) and len(batch) < max_batch:
                if self._q[i]["kind"] == batch_kind:
                    batch.append(self._q[i])
                    del self._q[i]            # remove selected job; next element shifts into i
                    # Do NOT increment i so we examine the new element now at index i
                else:
                    i += 1                    # skip different-kind job
            return batch_kind, batch


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

# ----------------------------------------------------------------------
# Utility: count how many *real* (non-pad) tokens are in each row.
# ----------------------------------------------------------------------
def count_real_tokens(input_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    Return a 1-D tensor where each element is the number of tokens in the
    corresponding row of `input_ids` that are NOT equal to `pad_id`.
    Shape (batch_size,)
    """
    return (input_ids != pad_id).sum(dim=1)

def _gen_session_id():
    return int(uuid.uuid4()) & 0xFFFFFFFF

class TargetSession:
    def __init__(self, session_id: int, input_ids, row_idx: int, max_tokens: int):
        self.session_id = session_id
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
        self.batch_timeout  = 0.03
        self.max_batch      = batch_size
        self.scheduler      = Scheduler(batch_timeout=self.batch_timeout)        
        # ----------------------------------------------------------
        # Row‑index pool for static‑batch Neuron graph
        # ----------------------------------------------------------
        # Use the maximum bucket size for the row pool.
        self._row_pool = list(range(max(BATCH_BUCKETS)))  # free rows 0…max bucket‑1

        # map session_id -> Queue for the blocking VerifyDraftTokens call
        self.result_queues  = {}
        self._sched_thread  = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self._sched_thread.start()
        self.model_mutex = torch.multiprocessing.Lock()
        self.lock = torch.multiprocessing.Lock()

    def _allocate_row(self):
        assert self._row_pool, "No free Neuron batch rows left"
        row_idx = self._row_pool.pop(0)
        # Fully rewind Neuron runtime pointer for this row.
        old_ptr = int(self.model.get_batched_cache_id_vec()[row_idx].item())
        if old_ptr:
            self.model.update_batched_cache(-old_ptr, row_idx)
        return row_idx

    def _release_row(self, idx: int):
        self._row_pool.append(idx)
        # Rewind Neuron runtime pointer before recycling this row.
        current_ptr = int(self.model.get_batched_cache_id_vec()[idx].item())
        if current_ptr:
            self.model.update_batched_cache(-current_ptr, idx)

    # -----------------------------  FINALIZE  -----------------------------
    def _finalize_session(self, sid: int):
        """
        Idempotent cleanup: free the Neuron batch row and remove all
        session-specific state. Safe to call multiple times.
        """
        sess = self.sessions.pop(sid, None)
        if sess is not None:
            try:
                self._release_row(sess.row_idx)
            except Exception as e:
                logger.warning("Ignoring error while releasing row %s: %s",
                               sess.row_idx, e)

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
                "[KV-DESYNC] session row=%d  model_ptr=%d  sess_ptr=%d",
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
            self.sessions[session_id] = TargetSession(session_id, current_ids, row_idx, max_tokens)
            # Enqueue a prefill job for the scheduler
            self.scheduler.enqueue({
                "kind":        "prefill",
                "session_id":  session_id,
                "input_ids":   current_ids,
            })
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
            self._finalize_session(sess.session_id)

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
        self.scheduler.enqueue({
            "kind":         "decode",
            "session_id":   sid,
            "draft_tokens": draft_tokens,
            "draft_probs":  draft_probs,
            "enqueue_time": start_verify_t,
        })

        # Block until scheduler puts result
        committed_ids, accepted_cnt, verify_ms, finished = resp_q.get()
        # If this session has now hit its max_tokens budget, force cleanup
        sess = self.sessions.get(sid)
        if sess is not None and sess.tokens_generated >= sess.max_tokens:
            finished = True
        if finished:
            self._finalize_session(sid)

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
        Consumer loop: pop homogeneous batches from Scheduler and
        dispatch them to the appropriate processing routine.
        """

        while True:
            kind, batch = self.scheduler.pop_batch(self.max_batch)
            if not batch:
                continue  # timeout – retry

            try:
                if kind == "decode":
                    self._process_decode_batch(batch)
                elif kind == "prefill":
                    self._process_prefill_batch(batch)
                else:
                    raise RuntimeError(f"Scheduler returned unknown kind '{kind}'")
            except Exception as e:
                logger.error("Scheduler batch error: %s", e, exc_info=True)
                for j in batch:
                    sid = j.get("session_id")
                    q = self.result_queues.get(sid)
                    if q is not None:
                        try:
                            q.put_nowait(([], 0, 0.0, True))  # sentinel
                        except queue.Full:
                            pass
                    self._finalize_session(sid)
                continue

    def _process_decode_batch(self, batch_reqs):
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
            # sess = self.sessions[sid]
            sess = self.sessions.get(sid)
            if sess is None:
                # Session was already finalized – drop the request and notify caller
                resp_q = self.result_queues.pop(sid, None)
                if resp_q is not None:
                    resp_q.put(([], 0, 0.0, True))
                continue
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
        
        if not input_ids:          # all requests were skipped
            return
        
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
            # draft_words = [
            #     tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            #     for tid in draft_tokens
            # ]
            # tgt_preds = torch.argmax(tgt_row_probs, dim=-1).tolist()
            # tgt_words = [
            #     tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            #     for tid in tgt_preds
            # ]
            # committed_words = [
            #     tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            #     for tid in committed
            # ]
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

            resp_q = self.result_queues.get(sid)
            if resp_q is not None:
                resp_q.put((committed, accepted_cnt, verify_ms, finished))
                if finished:
                    self._finalize_session(sess.session_id)
                    # After delivering the final response, remove the queue
                    self.result_queues.pop(sid, None)

    def _process_prefill_batch(self, jobs):
        """
        Batched prompt prefill for brand-new sessions.
        Each job dict must contain:
            kind:       "prefill"
            session_id: int
            input_ids:  (1, L) tensor
        """
        if not jobs:
            return
        assert len(jobs) <= self.max_batch, "prefill batch too large"
        
        pad_id = 0

        # Build batched tensors with right‑padding
        max_len = max(j["input_ids"].shape[1] for j in jobs)
        input_ids_lst = []
        cache_vecs_lst = []
        start_ids_lst  = []
        row_lengths    = []

        for j in jobs:
            sid   = j["session_id"]
            sess  = self.sessions[sid]
            ids   = j["input_ids"]
            # Count *non-pad* tokens for this row
            L_real = int(count_real_tokens(ids, pad_id=pad_id).item())
            row_lengths.append(L_real)

            if L_real < max_len:
                pad = torch.full((1, max_len - L_real), pad_id, dtype=ids.dtype)
                ids = torch.cat([ids, pad], dim=1)

            # ----------------------------------------------------------
            # Build a cache-position vector that matches **real** tokens:
            # real tokens → start_pos … start_pos+L-1
            # pad tokens  → continue that arithmetic sequence.
            # ----------------------------------------------------------
            start_pos = int(sess.get_session_cache_id().item())  # 0 for brand-new row
            vec_real  = torch.arange(start_pos, start_pos + L_real,
                                     dtype=torch.int32).unsqueeze(0)   # (1, L)

            if L_real < max_len:
                pad_vec = torch.arange(start_pos + L_real,
                                       start_pos + max_len,
                                       dtype=torch.int32).unsqueeze(0)
                vec = torch.cat([vec_real, pad_vec], dim=1)           
            else:
                vec = vec_real

            input_ids_lst.append(ids)
            cache_vecs_lst.append(vec)
            start_ids_lst.append(sess.row_idx)   # ← ensure start_ids tensor is built

        input_ids  = torch.cat(input_ids_lst, 0)
        cache_vecs = torch.cat(cache_vecs_lst, 0)
        start_ids  = torch.tensor(start_ids_lst, dtype=torch.int32, device=input_ids.device)

        # One Neuron forward under mutex
        with self.model_mutex:
            _ = self.model.forward(
                input_ids=input_ids,
                cache_ids=cache_vecs,
                start_ids=start_ids,
            )

        # Neuron only counts non-pad tokens, so advance each KV pointer
        # by the true prompt length stored in row_lengths[idx].
        # ------------------------------------------------------------------
        # Advance each session’s KV pointer by its real prompt length, not by
        # the padded bucket length.
        for idx, j in enumerate(jobs):
            sid  = j["session_id"]
            sess = self.sessions[sid]
            delta = row_lengths[idx]              # real (unpadded) length
            self.model.update_batched_cache(delta, sess.row_idx)
            sess.update_session_cache_id(delta)
            self._sync_kv_pointer(sess)

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
