import logging
import torch
from concurrent import futures
import grpc
import time
import random
TOP_K_TARGET = 512      # keep in sync with speculative.py
from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc

# Small helper: Softmax once per row in float32 for numeric stability
def _row_softmax(t: torch.Tensor) -> torch.Tensor:
    return torch.softmax(t.float(), dim=-1)

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
        self.model = model_loader.load_target_model(
            model_path,
            sequence_length=sequence_length,
            spec_length=spec_length or 4,
            top_k=TOP_K_TARGET,
            top_p=top_p,
            temperature=temperature,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        # Start with zero; will be set per‑prompt in StartGeneration
        self._ctx_estimate = 0
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

    # ------------------------------------------------------------
    # Utility: sample from <scores, indices> row returned by top‑k
    # ------------------------------------------------------------
    def _sample_from_topk(self, scores_row, idx_row, temperature: float = 1.0, top_p: float = 0.9):
        """
        scores_row : 1‑D tensor, un‑normalised logits for top‑k tokens
        idx_row    : 1‑D tensor, token‑ids that correspond to scores_row
        """
        logits = scores_row.float() / max(1e-8, temperature)
        probs  = torch.softmax(logits, dim=-1)

        cum_p = torch.cumsum(torch.sort(probs, descending=True)[0], dim=0)
        cut   = torch.searchsorted(cum_p, top_p, right=True).item()
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        keep_probs = sorted_probs[: cut + 1]
        keep_idx   = idx_row[sorted_idx[: cut + 1]]
        keep_probs = keep_probs / keep_probs.sum()

        choice = torch.multinomial(keep_probs, 1).item()
        return int(keep_idx[choice].item())


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
            # --- prime Neuron KV cache on the prompt (with padding) ---
            self.model.cache_ids = None
            self.model._next_pos = 0

            prompt_len = current_ids.shape[1]
            # Use the real prompt length as the compile‑time estimate for this session
            self._ctx_estimate = prompt_len

            if current_ids.shape[1] > 0:
                # Build an explicit position tensor 0 … L‑1 so Neuron
                # sees cache_ids and does not hit the .max(None) bug.
                pos_tensor = torch.arange(
                    current_ids.shape[1], dtype=torch.int32
                ).unsqueeze(0)        # shape (1, L)
                _ = self.model.forward(
                    input_ids=current_ids,
                    cache_ids=pos_tensor,
                )
            # store pointer (next index) inside the session
            self.sessions[session_id].cache_ids = torch.tensor(
                [prompt_len], dtype=torch.int32
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
        Forward the whole draft chunk through the Neuron target, capture logits.

        Returns
        -------
        probs_all   : List[float]       – P_target(d_i) for each draft token
        logits_rows : List[Tuple[scores, idx]]  – len = n_new
        """
        if not draft_tokens:
            return [], []

        self._sync_kv_pointer(sess)
        ids = torch.tensor([draft_tokens], dtype=sess.current_ids.dtype)   # (1, n)
        logits = self.model.forward(input_ids=ids)[0]      # (1, n, V)
        logits = logits.squeeze(0)                         #  → (n, V)

        # ---- DEBUG: dump one row so we can inspect distributions ----
        logger.info("[verify] logits chunk shape=%s  row0_sum=%.3f",
                    tuple(logits.shape),
                    _row_softmax(logits[0]).sum())

        scores_all, idx_all = torch.topk(logits, TOP_K_TARGET, dim=-1)     # (n, 512)

        probs_all = []
        for step, tok in enumerate(draft_tokens):
            idx_row  = idx_all[step]
            scr_row  = scores_all[step]
            hit = (idx_row == tok).nonzero(as_tuple=True)[0]
            if hit.numel():
                j = int(hit[0].item())
                probs_all.append(float(_row_softmax(scr_row)[j].item()))
            else:
                probs_all.append(0.0)

        logits_rows = [(scores_all[i], idx_all[i]) for i in range(len(draft_tokens))]
        return probs_all, logits_rows

    def VerifyDraftTokens(self, request, context):
        sid = request.session_id
        draft_tokens = list(request.draft_tokens)
        draft_probs  = list(request.draft_probs)
        if len(draft_probs) != len(draft_tokens):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          "draft_probs length must equal draft_tokens length")

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

            # Walk until first rejection (Metropolis–Hastings acceptance)
            for idx, tok in enumerate(draft_tokens):
                current_scores, current_idx = logits_rows[idx]
                p_target = probs_all[idx]
                q_draft  = max(draft_probs[idx], 1e-8)
                ratio    = min(1.0, p_target / q_draft)
                if random.random() < ratio:
                    accepted_cnt += 1
                    self._commit_token(sess, tok)
                    committed.append(tok)
                    if tok == self.eos_token_id:
                        break
                else:
                    fallback = self._sample_from_topk(
                        current_scores, current_idx,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    self._commit_token(sess, fallback)
                    committed.append(fallback)
                    break
            else:
                bonus_scores, bonus_idx = logits_rows[-1]
                bonus = self._sample_from_topk(
                    bonus_scores, bonus_idx,
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
        """
        Sample a token from 'logits' using:
 
            1. Optional temperature scaling
            2. Hard top‑k cutoff (k = TOP_K_TARGET)
            3. Nucleus (top‑p) sampling inside that slice
 
        Keeping the same top‑k (512) as the draft side ensures both
        models see the same candidate set size, which reduces divergence.
        """
        logits = logits.float() / max(1e-8, temperature)      # temp‑scale (float32)
        probs  = torch.softmax(logits, dim=-1)
 
        # ---------- hard top‑k (512) ----------
        k = min(TOP_K_TARGET, probs.shape[-1])
        top_vals, top_idx = torch.topk(probs, k)              # (k,) each
 
        # ---------- nucleus filter ----------
        cum_p = torch.cumsum(top_vals, dim=0)
        cutoff = torch.searchsorted(cum_p, top_p, right=True).item()
        nucleus_idx   = top_idx[: cutoff + 1]
        nucleus_probs = top_vals[: cutoff + 1]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
 
        choice = torch.multinomial(nucleus_probs, 1).item()
        return int(nucleus_idx[choice].item())

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