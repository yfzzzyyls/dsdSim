import torch

class TargetWorker:
    def __init__(self, model, tokenizer):
        """
        target_worker holds the compiled model, tokenizer, and state for verifying tokens.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context_tokens = []    # tokens confirmed so far (prompt + accepted tokens)
        self.max_new_tokens = None
        self.finished = False

        self.model.eval()

    def start_generation(self, prompt_text, max_tokens=None):
        """
        Initialize a new generation with the given prompt text.
        """
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
        self.context_tokens = prompt_ids[:]
        self.finished = False
        self.max_new_tokens = max_tokens

    def verify_tokens(self, draft_tokens):
        """
        For each draft token, do a single forward pass on the model to get the next token
        from the target perspective. If they match, accept; if mismatch, use target's token.
        """
        result = {
            "all_matched": False,
            "match_count": 0,
            "correct_token": 0,
            "finished": False
        }
        if self.finished:
            # Already done
            result["finished"] = True
            return result

        match_count = 0
        correct_token = None

        for draft_token in draft_tokens:
            # Prepare the input (context) for the next token
            input_ids = torch.tensor([self.context_tokens], dtype=torch.long)
            # Forward pass on the compiled model
            # Expecting a logits shape [1, seq_len, vocab_size], so we look at the last position
            with torch.no_grad():
                outputs = self.model(input_ids)
            next_logits = outputs.logits[0, -1, :]
            target_token_id = int(torch.argmax(next_logits))

            if target_token_id == draft_token:
                # Tokens match
                match_count += 1
                self.context_tokens.append(target_token_id)
                # Check for EOS
                if target_token_id == (self.tokenizer.eos_token_id or -1):
                    self.finished = True
                    correct_token = target_token_id
                    break
                # Check max tokens
                if (self.max_new_tokens is not None
                        and (len(self.context_tokens) - len(input_ids[0])) >= self.max_new_tokens):
                    self.finished = True
                    break
            else:
                # Mismatch
                correct_token = target_token_id
                self.context_tokens.append(target_token_id)
                if target_token_id == (self.tokenizer.eos_token_id or -1):
                    self.finished = True
                break

        all_matched = (match_count == len(draft_tokens))
        result["all_matched"] = all_matched
        result["match_count"] = match_count
        if correct_token is not None:
            result["correct_token"] = correct_token
        result["finished"] = self.finished

        return result

    def generate_full(self, prompt_text, max_new_tokens=50):
        """
        (Naive) single-token greedy generation with the compiled model.
        """
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
        output_ids = input_ids[:]
        tokens_generated = 0
        eos_token_id = self.tokenizer.eos_token_id

        while tokens_generated < max_new_tokens:
            model_input = torch.tensor([output_ids], dtype=torch.long)
            with torch.no_grad():
                outputs = self.model(model_input)
            next_logits = outputs.logits[0, -1, :]
            next_token_id = int(torch.argmax(next_logits))
            output_ids.append(next_token_id)
            tokens_generated += 1
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)