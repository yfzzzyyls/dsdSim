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
    """Stores per-session context: the current input_ids and stats."""
    def __init__(self, input_ids):
        self.current_ids = input_ids  # Torch tensor with shape [1, seq_len]
        self.tokens_generated = 0
        self.finished = False

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

            # We'll compute the probability of each draft token under the target model, stepping one token at a time
            target_probs = []
            temp_ids = sess.current_ids.clone().detach()  # shape [1, seq_len]
            for token in draft_tokens:
                if self.eos_token_id is not None and token == self.eos_token_id and sess.finished is False:
                    # Even if the draft says EOS, we still want to compute p(EOS).
                    pass

                # forward pass
                outputs = self.model(temp_ids)
                logits_for_last = _extract_logits(outputs)
                probs = torch.softmax(logits_for_last, dim=-1)
                p = float(probs[0, token].item())
                target_probs.append(p)

                # append the draft token to the temp context
                next_token_t = torch.tensor([[token]], dtype=temp_ids.dtype)
                temp_ids = torch.cat([temp_ids, next_token_t], dim=1)

                if self.eos_token_id is not None and token == self.eos_token_id:
                    logger.info(f"[session={session_id}] Draft proposed EOS => might finish after acceptance.")
                    # We do not break here because we want to compute probabilities for the entire chunk
                    # But finishing logic occurs after acceptance.

            # Return them. finished stays false unless user forcibly ended.
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

            # If accepted_count < draft_chunk_size, partial reject => fallback
            # If accepted_count == draft_chunk_size, fully accepted => big token
            # If accepted_count == 0, full reject => fallback
            if accepted_count < draft_chunk_size:
                # fallback => generate 1 token from the target model on the current session context
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
        return out_t[:, -1, :].float()
    elif len(out_t.shape) == 2:
        # e.g. [1, vocab], just return out_t[0]
        if out_t.size(0) == 1:
            return out_t[0, :].float()
        else:
            # if out_t is e.g. [B, vocab], pick the last row?
            # in typical usage B=1 anyway, but let's be safe.
            return out_t[-1, :].float()
    elif len(out_t.shape) == 1:
        # e.g. [vocab]
        return out_t.float()
    else:
        raise ValueError(f"Unknown shape for outputs: {out_t.shape}")

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