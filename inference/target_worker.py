#
# NOTE: If you update grpc_comm/inference.proto, re‑run:
#   python -m grpc_tools.protoc -Igrpc_comm --python_out=grpc_comm --grpc_python_out=grpc_comm grpc_comm/inference.proto

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


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Tensor helpers  Int32Tensor / FloatTensor ⇄ torch.Tensor
# ---------------------------------------------------------------------------
from grpc_comm.inference_pb2 import Int32Tensor as _I32, FloatTensor as _F32

def _tensor_i32(tmsg: _I32) -> torch.Tensor:
    if not tmsg.shape:
        return torch.tensor(tmsg.data, dtype=torch.int32)
    return torch.tensor(tmsg.data, dtype=torch.int32).view(*tmsg.shape)

def _tensor_f32(tmsg: _F32) -> torch.Tensor:
    if not tmsg.shape:
        return torch.tensor(tmsg.data, dtype=torch.float32)
    return torch.tensor(tmsg.data, dtype=torch.float32).view(*tmsg.shape)

def _make_i32(data, shape):
    return _I32(data=data, shape=shape)

class TargetSession:
    def __init__(self, input_ids):
        self.current_ids = input_ids              # (B, L)
        self.finished = False
        self.tokens_generated = 0
        self.verification_time = 0.0
        self.finalize_calls = 0
        self.last_draft_chunk = None
        self.cache_ids = torch.tensor(
            [input_ids.shape[1]] * input_ids.shape[0], dtype=torch.int32
        )  # (B,)
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
        self.model.cache_ids = sess.cache_ids.clone()
        assert torch.equal(self.model.cache_ids, sess.cache_ids), \
            "Target KV cache_ids desynchronised after sync"

    # ------------------------------------------------------------------
    # New batched start (one session for B prompts)
    # ------------------------------------------------------------------
    def StartGenerationBatch(self, request, context):
        prompt_ids = _tensor_i32(request.prompt_ids)   # (B, L)
        B, L = prompt_ids.shape
        session_id = random.getrandbits(64)

        logger.info("StartGenerationBatch: B=%d, L=%d, session=%d", B, L, session_id)
        with self.lock:
            # Build TargetSession and prime KV cache
            sess = TargetSession(prompt_ids)
            self.sessions[session_id] = sess

            # Prime Neuron KV
            self.model.cache_ids = None

            # ==========================
            # L must be > 0 for forward() to work
            # Do not support L = 0 for now
            # ==========================
            assert (prompt_ids.shape[1] > 0, "Prompt length must be > 0")
            _ = self.model.forward(prompt_ids)
            # --------------------------------------------------------------
            # Set `cache_ids` to the TRUE (unpadded) length of each row so
            # the pointer always indicates the *next* free KV slot.
            # --------------------------------------------------------------
            
            # The client supplied true lengths, respect them
            assert request.HasField("prompt_lens"), "prompt_lens must be provided by the client"

            true_len = _tensor_i32(request.prompt_lens)  # (B,)

            self.model.cache_ids = true_len.clone()   # authoritative per-row pointer
            sess.cache_ids       = true_len.clone()

        # Return the single session-id that represents the whole batch
        return inference_pb2.StartBatchResponse(session_ids=[session_id])


    def StartGeneration(self, request, context):
        session_id = request.session_id
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens
        gamma = request.gamma
        logger.debug(f"[session={session_id}] StartGeneration: prompt='{prompt_text}', max_new_tokens={max_tokens}, gamma={gamma}")
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

            # --- MANUALLY align wrapper pointer with the prompt length ---
            next_pos = current_ids.shape[1]                       # L
            self.model.cache_ids = torch.tensor([next_pos], dtype=torch.int32)
            # _next_pos line removed

            # record in session
            self.sessions[session_id].cache_ids = self.model.cache_ids.clone()
        return inference_pb2.StartResponse(acknowledged=True)
    
    def _commit_tokens_bulk(self, sess, tok_batch):
        """
        Commit a **batch** of token lists.  
        tok_batch : List[List[int]] length B, each inner list may be empty.
        """

        if all(len(row) == 0 for row in tok_batch):
            return

        B = len(tok_batch)
        max_k = max(len(r) for r in tok_batch)
        if max_k == 0:
            return

        # Shape (B, max_k) padded with eos_token_id
        pad_val = self.eos_token_id if self.eos_token_id is not None else 0
        padded = torch.full((B, max_k), pad_val, dtype=torch.int64)
        lengths = torch.zeros(B, dtype=torch.int32)
        for b, row in enumerate(tok_batch):
            if row:
                padded[b, :len(row)] = torch.tensor(row, dtype=torch.int64)
                lengths[b] = len(row)

        # Synchronise cache pointer
        self._sync_kv_pointer(sess)

        # Advance pointers per row
        sess.cache_ids += lengths
        self.model.cache_ids = sess.cache_ids.clone()

        # Append to current_ids
        sess.current_ids = torch.cat([sess.current_ids, padded], dim=1)

        if self.eos_token_id is not None:
            eos_mask = (padded == self.eos_token_id) & (torch.arange(max_k)[None, :] < lengths[:, None])
            if eos_mask.any():
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

    def verify(self, sess: TargetSession, draft_tokens_t: torch.Tensor):
        """
        Vectorised speculative_forward for a (B, γ) draft tensor.
        """
        if draft_tokens_t.numel() == 0:
            return torch.empty((0,0,0))

        B, gamma = draft_tokens_t.shape
        prev = sess.current_ids[:, -1:].clone()           # (B,1)
        spec_in = torch.cat([prev, draft_tokens_t], dim=1)   # (B, γ+1)

        steps = torch.arange(gamma+1).view(1, -1).repeat(B, 1)
        cache_mat = sess.cache_ids.unsqueeze(1) + steps     # (B, γ+1)

        logits = self.model.speculative_forward(
            input_ids=spec_in,             # (B, γ+1)
            cache_ids=cache_mat.int(),
            spec_length=gamma+1,
        )                                   # (B, γ+1, V)
        return logits  # keep full shape

    # ------------------------------------------------------------------
    # Verify B×γ draft tokens in one RPC (single-session batch)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Verify B×γ draft tokens in one RPC (single-session batch)
    # ------------------------------------------------------------------
    def VerifyBatchTokens(self, request, context):
        """
        Each DraftSequence in `request.sequences` holds one row of the batch.
        This is now a fast-path implementation: vectorized verify logic is inlined here.
        """
        if len(request.sequences) == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("VerifyBatchTokens received zero sequences.")
            return inference_pb2.VerifyBatchResponse(results=[])

        # All rows share the same session-id for the batched session
        session_id = request.sequences[0].session_id

        # --- stack rows into (B, γ) tensors ---------------------------------
        tok_rows, prob_rows = [], []
        for seq in request.sequences:
            tok_rows.append(_tensor_i32(seq.draft_tokens))   # (1, γ’)
            prob_rows.append(_tensor_f32(seq.draft_probs))   # (1, γ’)

        # --------------------------------------------------------------
        # Pad each row so all tensors have equal width before torch.cat().
        # Finished rows may send shape (1, 0); we right-pad with EOS (or 0).
        # --------------------------------------------------------------
        pad_val_i = self.eos_token_id if self.eos_token_id is not None else 0
        max_len   = max(t.shape[1] for t in tok_rows)     # global γ (may vary)

        padded_tok_rows  = []
        padded_prob_rows = []
        for t_row, p_row in zip(tok_rows, prob_rows):
            cur_len = t_row.shape[1]
            if cur_len < max_len:
                pad_cols_i = torch.full((1, max_len - cur_len), pad_val_i, dtype=t_row.dtype)
                pad_cols_f = torch.zeros((1, max_len - cur_len),      dtype=p_row.dtype)
                t_row = torch.cat([t_row, pad_cols_i], dim=1)
                p_row = torch.cat([p_row, pad_cols_f], dim=1)
            padded_tok_rows.append(t_row)
            padded_prob_rows.append(p_row)

        draft_tok_batch  = torch.cat(padded_tok_rows,  dim=0)   # (B, max_len)
        draft_prob_batch = torch.cat(padded_prob_rows, dim=0)   # (B, max_len)
        B = draft_tok_batch.shape[0]

        # ------------------------------------------------------------------
        # Inline fast‑path verification
        # ------------------------------------------------------------------
        with self.lock:
            if session_id not in self.sessions:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Session {session_id} not found")
                empty_result = inference_pb2.VerifyResult(
                    session_id=session_id,
                    tokens_accepted=0,
                    committed_ids=_make_i32([], [B, 0]),
                    finished=True,
                )
                return inference_pb2.VerifyBatchResponse(results=[empty_result])

            sess = self.sessions[session_id]
            if sess.finished or draft_tok_batch.numel() == 0:
                empty_result = inference_pb2.VerifyResult(
                    session_id=session_id,
                    tokens_accepted=0,
                    committed_ids=_make_i32([], [B, 0]),
                    finished=sess.finished,
                )
                return inference_pb2.VerifyBatchResponse(results=[empty_result])

            # ------------- vectorised verify -----------------------
            logits_all = self.verify(sess, draft_tok_batch)    # (B, γ+1, V)
            tgt_probs  = torch.softmax(logits_all.float(), dim=-1)

            tgt_row  = tgt_probs[:, :-1, :]      # (B, γ, V)
            B, gamma, V = tgt_row.shape
            device = tgt_row.device

            q_draft  = draft_prob_batch.to(device)             # (B, γ)
            idx_b    = torch.arange(B, device=device).unsqueeze(1).repeat(1, gamma)
            idx_g    = torch.arange(gamma, device=device)
            p_tgt    = tgt_row[idx_b, idx_g, draft_tok_batch.to(device)]

            ratio    = p_tgt / q_draft
            rand     = torch.rand_like(ratio)
            accept   = (p_tgt >= q_draft) | (rand < ratio)      # (B, γ)

            first_rej   = (~accept).float().argmax(dim=1)
            all_accept  = accept.all(dim=1)

            committed_batch = []
            accepted_total  = 0
            # ------------------------------------------------------------------
            # Vectorised build of committed tokens per row
            # ------------------------------------------------------------------
            acc_len = torch.where(
                all_accept,
                torch.full_like(first_rej, gamma),   # accept full γ
                first_rej,                           # accept up to first reject
            )                                         # (B,)

            # ----- accepted draft tokens mask (B, γ) --------------------------
            idx_tok     = torch.arange(gamma, device=device).view(1, -1)
            commit_mask = idx_tok < acc_len.unsqueeze(1)          # True where token accepted

            pad_val_i   = self.eos_token_id if self.eos_token_id is not None else 0
            accepted_tok = torch.full((B, gamma), pad_val_i, dtype=torch.int64, device=device)
            accepted_tok[commit_mask] = draft_tok_batch[commit_mask]   # copy only accepted IDs
            accepted_tok = accepted_tok.cpu()                          # move to host for list()

            # ----- bonus token selection per row -----------------------------
            bonus_row  = torch.where(acc_len < gamma, acc_len, torch.full_like(acc_len, gamma))
            bonus_logits = tgt_probs[torch.arange(B, device=device), bonus_row]   # (B,V)
            bonus_id   = torch.multinomial(bonus_logits, 1).squeeze(1).cpu()      # (B,)

            # ----- compose committed_batch as list-of-lists -------------------
            committed_batch = [
                accepted_tok[b, : int(acc_len[b])].tolist() + [int(bonus_id[b])]
                for b in range(B)
            ]
            accepted_total = int(acc_len.sum().item())

            # commit to KV + session state
            self._commit_tokens_bulk(sess, committed_batch)

            # pack committed tensor
            Kc       = max(len(r) for r in committed_batch)
            pad_val  = self.eos_token_id if self.eos_token_id is not None else 0
            commit_pad = []
            for row in committed_batch:
                commit_pad.extend(row + [pad_val]*(Kc - len(row)))

            committed_tensor = _make_i32(commit_pad, [B, Kc])

            result = inference_pb2.VerifyResult(
                session_id      = session_id,
                tokens_accepted = accepted_total,
                committed_ids   = committed_tensor,
                finished        = sess.finished,
            )

            return inference_pb2.VerifyBatchResponse(results=[result])

    def VerifyDraftTokens(self, request, context):
        start_verify_t = time.perf_counter()
        sid          = request.session_id
        draft_tokens_t = _tensor_i32(request.draft_tokens)   # (B, γ)
        # # ============================
        # # Decode IDs → words for easier debugging
        # draft_texts = [self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        #             for tid in draft_tokens]
        # logger.debug("[session=%s] received draft tokens (text)=%s  ids=%s",
        #             sid, draft_texts, draft_tokens)
        # # ============================
        draft_probs_t  = _tensor_f32(request.draft_probs)    # (B, γ)
        B, gamma = draft_tokens_t.shape

        with self.lock:
            if sid not in self.sessions:
                logger.error(f"[VerifyDraftTokens] Session {sid} not found.")
                verify_time_ms = (time.perf_counter() - start_verify_t) * 1000.0
                return inference_pb2.VerifyResponse(committed_ids=[],
                                                    accepted_count=0,
                                                    verify_time_ms=verify_time_ms,
                                                    finished=True)
            sess = self.sessions[sid]
            if sess.finished or draft_tokens_t.numel() == 0:
                verify_time_ms = (time.perf_counter() - start_verify_t) * 1000.0
                return inference_pb2.VerifyResponse(committed_ids=[],
                                                    accepted_count=0,
                                                    verify_time_ms=verify_time_ms,
                                                    finished=sess.finished)

            committed_batch = [[] for _ in range(B)]
            accepted_total  = 0

            logits_all = self.verify(sess, draft_tokens_t)   # (B, γ+1, V)

            # Soft-max after masking
            all_row_probs = torch.softmax(logits_all.float(), dim=-1)
            logger.debug("all_row_probs shape=%s", all_row_probs.shape)

            target_row_probs = all_row_probs[:, :-1, :]    # (B, γ, V)

            device = target_row_probs.device
            draft_token_tensor = draft_tokens_t.to(device)        # (B, γ)
            q_draft_t = draft_probs_t.to(device)                  # (B, γ)

            row_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, gamma)
            p_tgt_t = target_row_probs[row_idx, torch.arange(gamma, device=device), draft_token_tensor]

            ratio = p_tgt_t / q_draft_t
            rand  = torch.rand_like(ratio)
            accept = (p_tgt_t >= q_draft_t) | (rand < ratio)     # (B, γ) bool

            first_rej = (~accept).float().argmax(dim=1)  # index of first False per row
            all_acc_mask = accept.all(dim=1)             # rows fully accepted

            for b in range(B):
                row_committed = []
                acc_len = int(first_rej[b].item()) if not all_acc_mask[b] else gamma
                if acc_len > 0:
                    row_committed.extend(draft_tokens_t[b, :acc_len].tolist())
                # choose bonus
                bonus_row = acc_len if acc_len < gamma else gamma
                bonus_logits = all_row_probs[b, bonus_row]
                bonus_id = int(torch.multinomial(bonus_logits, 1).item())
                row_committed.append(bonus_id)

                committed_batch[b] = row_committed
                accepted_total += acc_len

            self._commit_tokens_bulk(sess, committed_batch)

            max_len = max(len(r) for r in committed_batch)
            pad_val = self.eos_token_id if self.eos_token_id is not None else 0
            flat = []
            for row in committed_batch:
                row += [pad_val] * (max_len - len(row))
                flat.extend(row)
            verify_time_ms = (time.perf_counter() - start_verify_t) * 1000.0
            return inference_pb2.VerifyResponse(
                committed_ids=_make_i32(flat, [B, max_len]),
                accepted_count=accepted_total,
                verify_time_ms=verify_time_ms,
                finished=sess.finished,
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