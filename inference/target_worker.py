import torch

class TargetWorker:
    def __init__(self, model, tokenizer):
        """
        Wraps the target model and tokenizer and maintains generation state.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context_tokens = []    # tokens confirmed so far (including prompt and any accepted tokens)
        self.max_new_tokens = None  # optional limit for generation
        self.finished = False

        # If the model device requires special handling, ensure model is in inference mode
        self.model.eval()

    def start_generation(self, prompt_text, max_tokens=None):
        """
        Initialize a new generation with the given prompt text.
        Tokenize the prompt and reset state.
        """
        # Tokenize prompt (get list of token IDs)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
        self.context_tokens = prompt_ids[:]  # copy prompt tokens as starting context
        self.finished = False
        self.max_new_tokens = max_tokens
        # (No actual model invocation here; will generate on Verify or GenerateFull calls.)

    def verify_tokens(self, draft_tokens):
        """
        Given a batch of draft-predicted tokens, have the target model verify them in order.
        Returns a dictionary with keys: all_matched, match_count, correct_token, finished.
        """
        result = {
            "all_matched": False,
            "match_count": 0,
            "correct_token": 0,
            "finished": False
        }
        if self.finished:
            # If already finished, no tokens should be processed
            result.update({"all_matched": False, "match_count": 0, "correct_token": 0, "finished": True})
            return result

        match_count = 0
        correct_token = None

        for i, draft_token in enumerate(draft_tokens):
            # Generate the target model's next token for the current context
            # Prepare input tensor for the model (context)
            input_ids = torch.tensor([self.context_tokens], dtype=torch.long)
            # Run one step of the model to get next-token logits
            outputs = self.model(input_ids)
            # outputs.logits shape: [1, seq_len, vocab_size]
            next_logits = outputs.logits[0, -1, :]  # logits for the next token
            # Determine target model's top token (greedy decoding)
            target_token_id = int(torch.argmax(next_logits))
            # Compare with draft token
            if target_token_id == draft_token:
                # Tokens match
                match_count += 1
                # Append the token to context
                self.context_tokens.append(target_token_id)
                # Check for end-of-sequence token
                if target_token_id == (self.tokenizer.eos_token_id or -1):
                    # If EOS token encountered, mark finished
                    self.finished = True
                    correct_token = target_token_id
                    break
                # If we have a max_new_tokens limit, check if we reached it
                if self.max_new_tokens is not None and len(self.context_tokens) - len(draft_tokens) >= self.max_new_tokens:
                    # Reached token limit after adding this token
                    self.finished = True
                    break
                # Continue to next token in draft_tokens
            else:
                # Mismatch: target model's token differs
                correct_token = target_token_id
                # Append the target's correct token to context (since that is now confirmed output)
                self.context_tokens.append(target_token_id)
                # If that token is EOS, mark finished
                if target_token_id == (self.tokenizer.eos_token_id or -1):
                    self.finished = True
                # Mismatch found, break out
                break

        all_matched = (match_count == len(draft_tokens))
        result["all_matched"] = all_matched
        result["match_count"] = match_count
        if correct_token is not None:
            result["correct_token"] = int(correct_token)
        result["finished"] = self.finished

        return result

    def generate_full(self, prompt_text, max_new_tokens=50):
        """
        Generate a complete sequence using only the target model (no speculative draft assistance).
        Returns the output text (including the prompt and continuation).
        """
        # Use the model's generate method for convenience
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        output_ids = self.model.generate(input_ids, **gen_kwargs)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text