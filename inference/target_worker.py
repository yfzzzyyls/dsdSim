import logging
import time
import torch
from concurrent import futures
import grpc

from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

class TargetSession:
    def __init__(self, input_ids):
        self.current_ids = input_ids  # Torch tensor with shape [1, seq_len]
        self.tokens_generated = 0
        self.finished = False
        self.last_draft_chunk = None  # for storing the last draft tokens we verified

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        # Load or compile the target model
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.sessions = {}  # dict: session_id -> TargetSession
        self.lock = torch.multiprocessing.Lock()  # or threading.Lock()

    def StartGeneration(self, request, context):
        session_id = request.session_id
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens
        gamma = request.gamma
        logger.info(f"[session={session_id}] StartGeneration: prompt='{prompt_text}', max_new_tokens={max_tokens}, gamma={gamma}")

        with self.lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, overwriting.")
            # Build input_ids
            if prompt_text:
                enc = self.tokenizer(prompt_text, return_tensors='pt')
                current_ids = enc["input_ids"]
            else:
                # empty prompt => just bos?
                current_ids = torch.zeros((1,0), dtype=torch.long)

            self.sessions[session_id] = TargetSession(current_ids)
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftTokens(self, request, context):
        session_id = request.session_id
        draft_tokens = list(request.draft_tokens)
        logger.info(f"[session={session_id}] VerifyDraftTokens: {draft_tokens}")

        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found in VerifyDraftTokens.")
                return inference_pb2.VerifyResponse(target_probs=[0.0]*len(draft_tokens), finished=True)

            sess = self.sessions[session_id]
            if sess.finished:
                logger.info(f"Session {session_id} is already finished. Returning empty verification.")
                return inference_pb2.VerifyResponse(target_probs=[], finished=True)

            if not draft_tokens:
                # no tokens => no probabilities
                return inference_pb2.VerifyResponse(target_probs=[], finished=False)

            # Convert draft_tokens to a tensor of shape [1, len(draft_tokens)]
            draft_tokens_tensor = torch.tensor([draft_tokens], dtype=sess.current_ids.dtype)

            # Build a temp input by concatenating the current_ids with the entire draft chunk
            # shape = [1, seq_len + len(draft_tokens)]
            expanded_ids = torch.cat([sess.current_ids, draft_tokens_tensor], dim=1)

            # Single forward pass over the entire new sequence
            outputs = self.model(expanded_ids)
            logits = _extract_logits_all(outputs)  # We'll define a helper to get all logits, not just the last.
            # Suppose logits now has shape [1, expanded_len, vocab_size]

            # The portion of interest is the last len(draft_tokens) positions
            num_new = len(draft_tokens)
            # slice shape: [1, num_new, vocab_size]
            logits_slice = logits[:, -num_new:, :]

            # Now compute probabilities for each position
            # We'll gather each token's probability from the corresponding row
            # i.e. draft_tokens[i] from logits_slice[0, i, :]
            target_probs = []
            for i, token_id in enumerate(draft_tokens):
                row_logits = logits_slice[0, i, :]
                row_probs = torch.softmax(row_logits, dim=-1)
                p = float(row_probs[token_id].item())
                target_probs.append(p)

            # Store draft tokens in last_draft_chunk for finalize
            sess.last_draft_chunk = draft_tokens

            return inference_pb2.VerifyResponse(
                target_probs=target_probs,
                finished=False
            )

    def FinalizeTokens(self, request, context):
        session_id = request.session_id
        accepted_count = request.accepted_count
        draft_chunk_size = request.draft_chunk_size

        logger.info(f"[session={session_id}] FinalizeTokens: accepted_count={accepted_count}, chunk_size={draft_chunk_size}")

        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found in FinalizeTokens.")
                return inference_pb2.FinalizeResponse(final_token=0, finished=True)
            sess = self.sessions[session_id]
            if sess.finished:
                logger.info(f"Session {session_id} is already finished. Returning no-op.")
                return inference_pb2.FinalizeResponse(final_token=0, finished=True)

            # If we have stored draft tokens, incorporate the accepted portion
            if sess.last_draft_chunk:
                # accept_count tokens are accepted
                accepted_tokens = sess.last_draft_chunk[:accepted_count]
                for t in accepted_tokens:
                    token_t = torch.tensor([[t]], dtype=sess.current_ids.dtype)
                    sess.current_ids = torch.cat([sess.current_ids, token_t], dim=1)
                # Clear last_draft_chunk or set it to None
                sess.last_draft_chunk = None

            if accepted_count < draft_chunk_size:
                # fallback => generate 1 token from the target model
                fallback_token = self._generate_one_token(sess)
                if self.eos_token_id is not None and fallback_token == self.eos_token_id:
                    sess.finished = True
                return inference_pb2.FinalizeResponse(final_token=fallback_token, finished=sess.finished)
            else:
                # fully accepted => we generate 1 big token from the target model
                big_token = self._generate_one_token(sess)
                if self.eos_token_id is not None and big_token == self.eos_token_id:
                    sess.finished = True
                return inference_pb2.FinalizeResponse(final_token=big_token, finished=sess.finished)

    def GenerateFull(self, request, context):
        # Baseline target-only decoding, ignoring concurrency. 
        # We can preserve your existing single-sequence code or skip.
        logger.info(f"GenerateFull not heavily used in concurrency scenario.")
        return super().GenerateFull(request, context)

    def _generate_one_token(self, sess: TargetSession):
        """Helper: runs the model on sess.current_ids, greedily picks next token, appends it."""
        outputs = self.model(sess.current_ids)
        logits = _extract_logits(outputs)
        token_id = int(torch.argmax(logits, dim=-1)[0].item())
        new_tok = torch.tensor([[token_id]], dtype=sess.current_ids.dtype)
        sess.current_ids = torch.cat([sess.current_ids, new_tok], dim=1)
        sess.tokens_generated += 1
        return token_id

def _extract_logits(outputs):
    """Helper to extract final-logits from various possible shapes (HF or compiled)."""
    if hasattr(outputs, "logits"):
        # HF style => [batch, seq_len, vocab_size]
        return outputs.logits[:, -1, :].float()
    # else we assume it's a raw tensor
    out_t = outputs
    # check shape
    if len(out_t.shape) == 3:
        # e.g. [batch, seq_len, vocab]
        out_t = out_t[:, -1, :]  # shape: [batch, vocab]
    elif len(out_t.shape) == 2:
        pass
        # e.g. [batch, vocab]
        # do nothing, out_t is already [batch, vocab]
    elif len(out_t.shape) == 1:
        # e.g. [vocab], make it [1, vocab]
        out_t = out_t.unsqueeze(0)
    else:
        raise ValueError(...)

    return out_t.float()

def _extract_logits_all(outputs):
    """Return the full logits tensor for [batch, seq_len, vocab]. If HF style, it's outputs.logits."""
    if hasattr(outputs, "logits"):
        return outputs.logits.float()  # shape [B, seq_len, vocab]
    # Otherwise assume it's a raw tensor
    # shape could be [B, seq_len, vocab] or something else
    out_t = outputs
    if len(out_t.shape) == 3:
        return out_t.float()  # already [B, seq_len, vocab]
    elif len(out_t.shape) == 2:
        # [B, vocab], unsqueeze a seq_len=1 dimension
        return out_t.unsqueeze(1).float()
    elif len(out_t.shape) == 1:
        # [vocab], unsqueeze batch=1 and seq=1
        return out_t.unsqueeze(0).unsqueeze(0).float()
    else:
        raise ValueError(f"Unhandled shape for model output: {out_t.shape}")

def run_server(model_path, port=50051, sequence_length=128, profile=False):
    import sys

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading target model from {model_path} with sequence_length={sequence_length} ...")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server_address = f"[::]:{port}"
    server.add_insecure_port(server_address)
    logger.info(f"Target server starting on {server_address}")
    server.start()
    server.wait_for_termination()

def run_local(model_path, prompt="", max_new_tokens=50, sequence_length=128, profile=False):
    """Run local generation for profiling single-model performance (unchanged)."""
    logger.info("Running target model locally for output verification/profiling.")
    from inference import model_loader
    from transformers import AutoTokenizer
    import time

    model = model_loader.load_model(model_path, sequence_length=sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if not prompt:
        logger.info("No prompt. Using empty input.")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids if prompt else torch.zeros((1,0), dtype=torch.long)
    output_text = ""
    tokens_generated = 0
    start_time = time.time() if profile else None

    for i in range(max_new_tokens):
        try:
            output = model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        except Exception as e:
            logger.error(f"Target model generation failed: {e}")
            break
        token_id = int(output[0, -1]) if not isinstance(output, (list, tuple)) else int(output[0][-1])
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        print(f"Token {i+1}: {repr(token_text)}", flush=True)
        output_text += token_text

        new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            logger.info("EOS token encountered, stopping generation.")
            break

    if profile and start_time is not None:
        total_time = time.time() - start_time
        throughput = tokens_generated / total_time if total_time > 0 else float('inf')
        logger.info(f"Target model generation completed in {total_time:.2f} seconds.")
        logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} tokens/sec")
        # Save performance metrics ...
        csv_file = f"performance_target_only_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        json_file = csv_file.replace(".csv", ".json")
        try:
            with open(csv_file, "w") as cf:
                cf.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                avg_time = total_time / tokens_generated if tokens_generated>0 else 0
                cf.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},N/A\n")
            import json
            with open(json_file, "w") as jf:
                json.dump({
                    "total_latency": total_time,
                    "tokens_generated": tokens_generated,
                    "throughput": throughput,
                    "token_match_rate": None
                }, jf, indent=2)
            logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
        except Exception as e:
            logger.error(f"Failed to write performance metrics: {e}")
    print("\n=== Final Output ===\n" + (prompt + output_text))
    return output_text