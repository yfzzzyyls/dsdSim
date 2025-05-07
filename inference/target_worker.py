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
        self.model._next_pos = int(sess.cache_ids.item())
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
            self.model._next_pos = next_pos
            self.model.cache_ids = torch.tensor([next_pos], dtype=torch.int32)

            # record in session
            self.sessions[session_id].cache_ids = self.model.cache_ids.clone()
        return inference_pb2.StartResponse(acknowledged=True)
    
    def _commit_tokens_bulk(self, sess, tok_ids):
        """
        Commit a list of tokens (accepted draft + bonus) in ONE Neuron
        speculative_forward call so the KV cache advances in bulk.
        We call `forward` (not speculative_forward) here so that the Neuron
        wrapper's _postprocess() updates the Python‑side KV pointer.
        """

        if not tok_ids:
            return

        # Ensure model & session pointers match before we bump
        self._sync_kv_pointer(sess)

        # ===============================================================
        # Discover the compiled speculation bucket sizes ONCE per call.
        # ===============================================================
        # LlamaForSamplingModel = self.model.adapter.model
        # bucket_lengths = {k[0] if isinstance(k, tuple) else int(k)
        #                     for k in LlamaForSamplingModel.decoder_lm_head_for_speculation.keys()}

        # #=================================================
        # token_texts = [self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        #         for tid in tok_ids]
        # logger.debug("Commit tokens (text)=%s  ids=%s", token_texts, tok_ids)
        # #==================================================

        # # Sanity checks
        # assert len(tok_ids) in bucket_lengths, \
        #     f"Commit length {len(tok_ids)} not compiled; buckets={sorted(bucket_lengths)}"
        # # --- allow EOS even when tokenizer.vocab_size is stale ---
        # valid = [
        #     t for t in tok_ids
        #     if t == self.eos_token_id or t < self.tokenizer.vocab_size
        # ]
        # assert len(valid) == len(tok_ids), \
        #     f"OOV token(s) in commit_ids: {tok_ids}"

        # self._sync_kv_pointer(sess)

        # # (1, K) tensor of the new (real) tokens
        # in_tensor = torch.tensor([tok_ids], dtype=sess.current_ids.dtype)
        # cache_vec = torch.arange(len(tok_ids), dtype=torch.int32) + self.model._next_pos

        # ------------------------------------------------------------------
        # Figure out which speculation buckets were actually compiled.
        # self.model is a NeuronHFAdapterWrap → .adapter → HFAdapter →
        # .model (the underlying LlamaForSampling).
        # ------------------------------------------------------------------
        # raw_keys = LlamaForSamplingModel.decoder_lm_head_for_speculation.keys()
        # # Accept both (k, batch_size) and k-only keys
        # def _extract_k(k):
        #     if isinstance(k, tuple):
        #         return k[0]
        #     return k
        # spec_ok = len(tok_ids) in {_extract_k(k) for k in raw_keys}
        # assert spec_ok, f"speculative_forward not compiled for {len(tok_ids)} tokens"

        # # Use speculative_forward to process all tokens and update the KV cache.
        # _ = self.model.speculative_forward(
        #     input_ids=in_tensor,
        #     cache_ids=cache_vec,
        #     spec_length=len(tok_ids),
        # )

        # Manually bump wrapper pointer so it matches the device
        orig_next_pos = int(sess.cache_ids.item())    # authoritative
        new_next_pos  = orig_next_pos + len(tok_ids)
        self.model._next_pos = new_next_pos
        self.model.cache_ids = torch.tensor([new_next_pos], dtype=torch.int32)

        # Keep the session pointer consistent with the model.
        sess.cache_ids = self.model.cache_ids.clone()

        # print(f"commit tokens: _next_pos={self.model._next_pos} -> current_shape{sess.current_ids.shape[1]} + {len(tok_ids)}")
        assert int(self.model._next_pos) == sess.current_ids.shape[1] + len(tok_ids), \
            "KV pointer mismatch after manual bump"

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
        start_verify_t = time.perf_counter()
        sid          = request.session_id
        draft_tokens = list(request.draft_tokens)
        # # ============================
        # # Decode IDs → words for easier debugging
        # draft_texts = [self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        #             for tid in draft_tokens]
        # logger.debug("[session=%s] received draft tokens (text)=%s  ids=%s",
        #             sid, draft_texts, draft_tokens)
        # # ============================
        
        draft_probs  = list(request.draft_probs)
        logger.debug("[session=%s] draft_probs=%s", sid, draft_probs)

        assert draft_probs, (
            f"[session={sid}] VerifyDraftTokens received empty draft_probs for "
            f"{len(draft_tokens)} draft_tokens"
        )

        with self.lock:
            if sid not in self.sessions:
                logger.error(f"[VerifyDraftTokens] Session {sid} not found.")
                verify_time_ms = (time.perf_counter() - start_verify_t) * 1000.0
                return inference_pb2.VerifyResponse(committed_ids=[],
                                                    accepted_count=0,
                                                    verify_time_ms=verify_time_ms,
                                                    finished=True)
            sess = self.sessions[sid]
            if sess.finished or not draft_tokens:
                verify_time_ms = (time.perf_counter() - start_verify_t) * 1000.0
                return inference_pb2.VerifyResponse(committed_ids=[],
                                                    accepted_count=0,
                                                    verify_time_ms=verify_time_ms,
                                                    finished=sess.finished)

            committed     = []
            accepted_cnt  = 0

            # ---- ONE verification pass for the entire chunk + bonus ----
            logits_all = self.verify(sess, draft_tokens)

            # # ===========================================
            # # ---- DEBUG: show draft chunk size, words, and token IDs ----
            # draft_size  = len(draft_tokens)
            # draft_words = [
            #     self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            #     for tid in draft_tokens
            # ]
            # logger.debug(
            #     "[DEBUG draft chunk] size=%d  words=%s  ids=%s",
            #     draft_size,
            #     draft_words,
            #     draft_tokens,
            # )
            # # ===========================================

            # Soft-max after masking
            all_row_probs = torch.softmax(logits_all.float(), dim=-1)
            logger.debug("all_row_probs shape=%s", all_row_probs.shape)
            
            # ================================================
            # multinomial probe: sample one token from each row, then log once ---
            # ================================================
            # sampled_tokens = []
            # sampled_ids    = []
            # sampled_ps     = []

            # for r in range(all_row_probs.size(0)):
            #     row_probs   = all_row_probs[r]
            #     sampled_id  = int(torch.multinomial(row_probs, 1).item())
            #     sampled_p   = float(row_probs[sampled_id].item())
            #     sampled_tok = self.tokenizer.decode(
            #         [sampled_id], clean_up_tokenization_spaces=False
            #     )
            #     sampled_tokens.append(sampled_tok)
            #     sampled_ids.append(sampled_id)
            #     sampled_ps.append(sampled_p)

            # logger.debug(
            #     "[DEBUG multinomial] sampled_tokens=%s  ids=%s  p_rows=%s",
            #     sampled_tokens,
            #     sampled_ids,
            #     ["{:.6f}".format(p) for p in sampled_ps],
            # )
            # =============================================


            target_row_probs = all_row_probs[:-1]      # γ rows

            probs = [float(target_row_probs[i, tok].item())
                    for i, tok in enumerate(draft_tokens)]
                        
            # --- diagnostic shapes ------------------------------------------------
            logger.debug(
                "[session=%s] verify returned shapes: "
                "probs=%d  bonus_row_probs=%s  target_row_probs=%s",
                sid,
                len(probs),
                tuple(all_row_probs[-1].shape),
                tuple(target_row_probs.shape),
            )
            # ----------------------------------------------------------------------

            # Sanity‑check: we must have one q_draft per draft token
            assert len(draft_probs) == len(draft_tokens), (
                f"[session={sid}] Length mismatch: "
                f"{len(draft_probs)} draft_probs vs {len(draft_tokens)} draft_tokens"
            )
            # ------------------------------------------------------------------
            # Vectorised probabilistic acceptance (device‑resident, no Python loops)
            # ------------------------------------------------------------------
            device = target_row_probs.device            # stay on same device
            draft_token_tensor = torch.tensor(draft_tokens, device=device)

            # p_tgt_t: gather target P(draft_token | prefix) directly from tensor
            row_idx = torch.arange(len(draft_tokens), device=device)
            p_tgt_t = target_row_probs[row_idx, draft_token_tensor]     # (γ,)

            # q_draft comes from RPC as Python list – move to same device
            q_draft_t = torch.tensor(draft_probs, device=device)

            # Guard against zero probabilities to avoid NaNs
            torch._assert((q_draft_t > 0).all(), "q_draft contains zeros")

            ratio    = p_tgt_t / q_draft_t
            rand_vec = torch.rand_like(ratio)
            accept_vec = (p_tgt_t >= q_draft_t) | (rand_vec < ratio)

            # index of first rejection; if none, len(draft_tokens)
            reject_idx = (~accept_vec).nonzero(as_tuple=False)
            first_rej  = int(reject_idx[0].item()) if reject_idx.numel() > 0 else len(draft_tokens)

            accepted_cnt = first_rej
            committed    = draft_tokens[:accepted_cnt]

            if accepted_cnt < len(draft_tokens):
                # one rejection → sample bonus from that row
                row_idx    = accepted_cnt        # 0‑based
                bonus_id   = int(torch.multinomial(all_row_probs[row_idx], 1).item())
                committed.append(bonus_id)
                # ============================
                # DEBUG logging (single call)
                # if logger.isEnabledFor(logging.DEBUG):
                #     d_tok  = self.tokenizer.decode([draft_tokens[row_idx]], clean_up_tokenization_spaces=False)
                #     b_tok  = self.tokenizer.decode([bonus_id], clean_up_tokenization_spaces=False)
                #     logger.debug(
                #         "[DEBUG token rejected] i=%d draft token='%s' bonus token='%s' "
                #         "id=%d  p_tgt=%.6f  q_draft=%.6f",
                #         row_idx, d_tok, b_tok, draft_tokens[row_idx],
                #         p_tgt_t[row_idx], q_draft_t[row_idx]
                #     )
                # ============================
            else:
                # all accepted – sample bonus from last row (γ)
                bonus_id = int(torch.multinomial(all_row_probs[-1], 1).item())
                committed.append(bonus_id)
                # # ============================
                # if logger.isEnabledFor(logging.DEBUG):
                #     b_tok = self.tokenizer.decode([bonus_id], clean_up_tokenization_spaces=False)
                #     logger.debug(
                #         "[DEBUG all token accepted] bonus token='%s' id=%d",
                #         b_tok, bonus_id
                #     )
                # # ============================

            # stop condition: if EOS was accepted
            if self.eos_token_id in committed:
                sess.finished = True
                
            # # ---------------------------------------------------------------
            # # Bulk commit all tokens at once
            # commit_texts = [self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            #                 for tid in committed]
            # logger.debug("target generate/verified tokens (text)=%s  ids=%s", commit_texts, committed)
            # ---------------------------------------------------------------


            # ==========================================
            # Commit all accepted tokens in one call
            self._commit_tokens_bulk(sess, committed)
            # ==========================================
            
            logger.debug("commmit response to draft: _next_pos=%d", int(self.model._next_pos))

            verify_time_ms = (time.perf_counter() - start_verify_t) * 1000.0
            return inference_pb2.VerifyResponse(committed_ids=committed,
                                                accepted_count=accepted_cnt,
                                                verify_time_ms=verify_time_ms,
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


    def GenerateFull(self, request, context):
        # baseline target-only decoding, optional
        return super().GenerateFull(request, context)


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
