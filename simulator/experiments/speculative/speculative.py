#!/usr/bin/env python3
# Copyright 2025
# SPDX-License-Identifier: Apache-2.0
"""
Offline speculative decoding profiler that measures acceptance behaviour for
arbitrary drafter / verifier model pairs using Hugging Face Transformers.

The script runs a manual speculative loop:
  1. Drafter proposes up to `k` tokens per iteration.
  2. Verifier greedily validates the proposal token-by-token.
  3. Tokens accepted by the verifier are committed; the first mismatch token is
     replaced by the verifier's output and the remaining draft tokens are
     discarded.

Per-iteration outcomes are aggregated into acceptance statistics that can be
used to populate the simulator's acceptance-rate lookup tables.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Avoid importing TensorFlow / Flax when pulling Transformers.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from vllm.v1.sample.sampler import Sampler as VllmSampler
from vllm.v1.worker.gpu_input_batch import (
    LogitsProcessors as VllmLogitsProcessors,
    SamplingMetadata as VllmSamplingMetadata,
)

VLLM_SAMPLING_EPS = 1e-5

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None
    _HAS_TQDM = False

DEFAULT_PROMPTS: List[str] = [
    "Summarize the core idea behind speculative decoding in large language models.",
    "Provide three bullet points comparing Llama-2-7B and Llama-2-70B.",
    "Explain why acceptance-rate calibration matters when replaying traces without tokens.",
]


@dataclass
class IterationLog:
    context_length_before: int
    draft_token_ids: List[int]
    accepted_flags: List[bool]
    accepted_count: int
    mismatch_token_id: Optional[int]
    context_length_after: int


@dataclass
class PromptLog:
    prompt_index: int
    prompt: str
    generated_text: str
    total_generated_tokens: int
    iterations: List[IterationLog]
    reached_eos: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile speculative decoding acceptance rates "
        "between a drafter and verifier model pair.",
    )
    parser.add_argument(
        "--drafter-model",
        required=True,
        help="Name or path of the drafter (speculative) model.",
    )
    parser.add_argument(
        "--verifier-model",
        required=True,
        help="Name or path of the verifier (target) model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to both models (default: auto).",
    )
    parser.add_argument(
        "--drafter-device-map",
        help="Optional device map override for the drafter model.",
    )
    parser.add_argument(
        "--verifier-device-map",
        help="Optional device map override for the verifier model.",
    )
    parser.add_argument(
        "--drafter-dtype",
        default="bfloat16",
        help="Torch dtype for drafter weights (default: bfloat16).",
    )
    parser.add_argument(
        "--verifier-dtype",
        default="bfloat16",
        help="Torch dtype for verifier weights (default: bfloat16).",
    )
    parser.add_argument(
        "--drafter-load-in-8bit",
        action="store_true",
        help="Load drafter model in 8-bit quantized mode (bitsandbytes).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generation tokens per prompt (default: 128).",
    )
    parser.add_argument(
        "--spec-tokens",
        type=int,
        default=4,
        help="Number of speculative tokens proposed per iteration (default: 4).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Drafter sampling temperature (0 for greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Drafter nucleus sampling top-p value (default: 1.0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Drafter top-k sampling (0 disables).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for drafter sampling (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility (default: 1337).",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Optional JSONL file containing prompts with a 'prompt' field.",
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        help="Append aggregated metrics as JSONL records to this file.",
    )
    parser.add_argument(
        "--details-jsonl",
        type=Path,
        help="Optional JSONL file capturing per-prompt iteration logs.",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print generated text for each prompt.",
    )
    parser.add_argument(
        "--stop-on-pad",
        action="store_true",
        help="Stop iterations when drafter emits pad tokens.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="Optional truncation length for prompt tokens before speculation.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        help="Limit the number of prompts processed.",
    )
    parser.add_argument(
        "--debug-progress",
        action="store_true",
        help="Print detailed progress/debug information for each prompt.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    lookup = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in lookup:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(lookup)}.")
    return lookup[key]


def load_model(
    model_name: str,
    *,
    dtype: torch.dtype,
    device_map: str,
    load_in_8bit: bool = False,
) -> AutoModelForCausalLM:
    kwargs: Dict[str, object] = {
        "device_map": device_map,
    }
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    else:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_cache=True,
        trust_remote_code=False,
        **kwargs,
    )
    model.eval()
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_prompts(prompts_file: Optional[Path]) -> List[str]:
    if prompts_file is None:
        return DEFAULT_PROMPTS
    prompts: List[str] = []
    with prompts_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "prompt" not in payload:
                raise ValueError(f"Prompt record missing 'prompt': {payload}")
            prompts.append(payload["prompt"])
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    return prompts


def build_sampling_metadata(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    prompt_token_ids: Optional[torch.Tensor],
    generated_tokens: List[int],
    generator: Optional[torch.Generator],
    device: torch.device,
) -> VllmSamplingMetadata:
    all_greedy = (
        temperature <= VLLM_SAMPLING_EPS and top_p >= 1.0 and top_k <= 0
    )
    all_random = not all_greedy
    temp_value = float(temperature)
    if not all_random:
        temp_value = max(temp_value, VLLM_SAMPLING_EPS)
    temperature_tensor = torch.tensor(
        [temp_value], dtype=torch.float32, device=device
    )
    top_p_tensor = (
        None
        if top_p >= 1.0
        else torch.tensor([float(top_p)], dtype=torch.float32, device=device)
    )
    top_k_tensor = (
        None
        if top_k <= 0
        else torch.tensor([int(top_k)], dtype=torch.int32, device=device)
    )
    generators: Dict[int, torch.Generator] = {}
    if generator is not None and not all_greedy:
        generators[0] = generator

    no_penalties = repetition_penalty == 1.0
    frequency_penalties = torch.zeros(1, dtype=torch.float32, device=device)
    presence_penalties = torch.zeros(1, dtype=torch.float32, device=device)
    repetition_penalties = torch.full(
        (1,), float(repetition_penalty), dtype=torch.float32, device=device
    )

    return VllmSamplingMetadata(
        temperature=temperature_tensor,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=top_p_tensor,
        top_k=top_k_tensor,
        generators=generators,
        max_num_logprobs=None,
        no_penalties=no_penalties,
        prompt_token_ids=None if no_penalties else prompt_token_ids,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
        output_token_ids=[generated_tokens.copy()],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=VllmLogitsProcessors(),
    )


def greedy_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def forward_with_past(
    model: AutoModelForCausalLM,
    *,
    input_ids: torch.Tensor,
    past_key_values,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=past_key_values,
    )
    return outputs.past_key_values, outputs.logits[:, -1, :]


def infer_primary_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "device"):
        device = getattr(model, "device")
        if isinstance(device, torch.device):
            return device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def speculative_loop(
    prompt: str,
    drafter_model: AutoModelForCausalLM,
    verifier_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    spec_tokens: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    stop_on_pad: bool,
    max_prompt_tokens: Optional[int],
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[PromptLog, dict]:
    encode_kwargs = dict(return_tensors="pt", add_special_tokens=True)
    if max_prompt_tokens is not None:
        encode_kwargs.update(dict(truncation=True, max_length=max_prompt_tokens))
    encoded = tokenizer(prompt, **encode_kwargs)
    context_ids = encoded["input_ids"]
    if logger:
        logger(
            f"    initial context tokens: {context_ids.shape[1]}"
        )
    prompt_len = context_ids.shape[1]
    total_generated = 0
    reached_eos = False
    iterations: List[IterationLog] = []

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    position_attempts = [0] * spec_tokens
    position_accepts = [0] * spec_tokens

    pad_token = tokenizer.pad_token_id
    eos_token = tokenizer.eos_token_id
    drafter_device = infer_primary_device(drafter_model)
    verifier_device = infer_primary_device(verifier_model)

    context_tokens = context_ids[0].tolist()

    with torch.no_grad():
        drafter_outputs = drafter_model(
            input_ids=context_ids.to(drafter_device),
            use_cache=True,
        )
        drafter_past = drafter_outputs.past_key_values
        drafter_next_logits = drafter_outputs.logits[:, -1, :]
        verifier_outputs = verifier_model(
            input_ids=context_ids.to(verifier_device),
            use_cache=True,
        )
        verifier_past = verifier_outputs.past_key_values
        verifier_next_logits = verifier_outputs.logits[:, -1, :]

    vllm_sampler = VllmSampler().to(drafter_device)
    vllm_sampler.eval()
    requires_random = not (
        temperature <= VLLM_SAMPLING_EPS and top_p >= 1.0 and top_k <= 0
    )
    sampler_generator: Optional[torch.Generator] = None
    if requires_random:
        generator_device = (
            drafter_device.type if drafter_device.type in ("cuda", "cpu") else "cpu"
        )
        sampler_generator = torch.Generator(device=generator_device)
        sampler_generator.manual_seed(seed)

    initial_prompt_tokens = context_tokens.copy()
    prompt_token_tensor = torch.tensor(
        [initial_prompt_tokens], dtype=torch.int64, device=drafter_device
    )
    generated_history: List[int] = []

    while total_generated < max_tokens and not reached_eos:
        remaining = max_tokens - total_generated
        spec_len = min(spec_tokens, remaining)
        if spec_len <= 0:
            break
        context_len_before = len(context_tokens)
        if logger:
            logger(
                f"    remaining budget {remaining}, requesting up to {spec_len} draft tokens"
            )

        # Drafter speculative proposal with cached past.
        draft_tokens: List[int] = []
        drafter_stack: List[Tuple[Tuple[torch.Tensor, ...], torch.Tensor]] = [
            (drafter_past, drafter_next_logits)
        ]

        for _ in range(spec_len):
            current_past, current_logits = drafter_stack[-1]
            sampling_metadata = build_sampling_metadata(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                prompt_token_ids=prompt_token_tensor,
                generated_tokens=generated_history,
                generator=sampler_generator,
                device=drafter_device,
            )
            sampler_output = vllm_sampler(
                current_logits,
                sampling_metadata,
            )
            next_token = int(sampler_output.sampled_token_ids[0, 0].item())
            draft_tokens.append(next_token)
            if logger:
                logger(f"      drafter proposed token {next_token}")
            with torch.no_grad():
                new_past, new_logits = forward_with_past(
                    drafter_model,
                    input_ids=torch.tensor([[next_token]], device=drafter_device),
                    past_key_values=current_past,
                )
            drafter_stack.append((new_past, new_logits))
            if next_token == eos_token or (stop_on_pad and next_token == pad_token):
                break

        effective_len = len(draft_tokens)
        if eos_token in draft_tokens:
            effective_len = min(effective_len, draft_tokens.index(eos_token) + 1)
        if stop_on_pad and pad_token in draft_tokens:
            pad_idx = draft_tokens.index(pad_token)
            effective_len = min(effective_len, pad_idx)
        if effective_len < len(draft_tokens):
            draft_tokens = draft_tokens[:effective_len]
            drafter_stack = drafter_stack[: effective_len + 1]
        num_draft_tokens += len(draft_tokens)
        accepted_flags: List[bool] = []
        mismatch_token: Optional[int] = None
        committed_count = 0

        for idx, draft_token in enumerate(draft_tokens):
            if idx < spec_tokens:
                position_attempts[idx] += 1
            verifier_choice = greedy_token(verifier_next_logits)
            is_accepted = verifier_choice == draft_token
            accepted_flags.append(is_accepted)
            if logger:
                logger(
                    f"      position {idx}: draft={draft_token}, verifier={verifier_choice}, accepted={is_accepted}"
                )
            if is_accepted:
                position_accepts[idx] += 1
                context_tokens.append(draft_token)
                if len(context_tokens) > prompt_len:
                    generated_history.append(draft_token)
                total_generated += 1
                committed_count += 1
                if draft_token == eos_token:
                    reached_eos = True
                with torch.no_grad():
                    verifier_past, verifier_next_logits = forward_with_past(
                        verifier_model,
                        input_ids=torch.tensor(
                            [[draft_token]], device=verifier_device
                        ),
                        past_key_values=verifier_past,
                    )
                if reached_eos or total_generated >= max_tokens:
                    break
                continue

            # First mismatch
            mismatch_token = verifier_choice
            context_tokens.append(mismatch_token)
            if len(context_tokens) > prompt_len:
                generated_history.append(mismatch_token)
            total_generated += 1
            if mismatch_token == eos_token:
                reached_eos = True
            with torch.no_grad():
                verifier_past, verifier_next_logits = forward_with_past(
                    verifier_model,
                    input_ids=torch.tensor([[mismatch_token]], device=verifier_device),
                    past_key_values=verifier_past,
                )
            # Reset drafter state to last accepted prefix and include mismatch.
            drafter_past, drafter_next_logits = drafter_stack[committed_count]
            with torch.no_grad():
                drafter_past, drafter_next_logits = forward_with_past(
                    drafter_model,
                    input_ids=torch.tensor([[mismatch_token]], device=drafter_device),
                    past_key_values=drafter_past,
                )
            break

        else:
            # All speculative tokens accepted; consume verifier bonus token.
            verifier_bonus = greedy_token(verifier_next_logits)
            context_tokens.append(verifier_bonus)
            if len(context_tokens) > prompt_len:
                generated_history.append(verifier_bonus)
            total_generated += 1
            if logger:
                logger(
                    f"      all draft tokens accepted; verifier produced bonus token {verifier_bonus}"
                )
            if verifier_bonus == eos_token:
                reached_eos = True
            with torch.no_grad():
                verifier_past, verifier_next_logits = forward_with_past(
                    verifier_model,
                    input_ids=torch.tensor([[verifier_bonus]], device=verifier_device),
                    past_key_values=verifier_past,
                )
            drafter_past, drafter_next_logits = drafter_stack[len(draft_tokens)]
            with torch.no_grad():
                drafter_past, drafter_next_logits = forward_with_past(
                    drafter_model,
                    input_ids=torch.tensor([[verifier_bonus]], device=drafter_device),
                    past_key_values=drafter_past,
                )

        num_drafts += 1
        num_accepted_tokens += sum(accepted_flags)

        iterations.append(
            IterationLog(
                context_length_before=context_len_before,
                draft_token_ids=draft_tokens,
                accepted_flags=accepted_flags,
                accepted_count=sum(accepted_flags),
                mismatch_token_id=mismatch_token,
                context_length_after=len(context_tokens),
            )
        )

        if reached_eos or total_generated >= max_tokens:
            break

    generated_tensor = torch.tensor([context_tokens], dtype=torch.long)
    generated_text = tokenizer.decode(
        generated_tensor[0, prompt_len:], skip_special_tokens=True
    )

    prompt_log = PromptLog(
        prompt_index=-1,
        prompt=prompt,
        generated_text=generated_text,
        total_generated_tokens=total_generated,
        iterations=iterations,
        reached_eos=reached_eos,
    )

    summary = {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "position_attempts": position_attempts,
        "position_accepts": position_accepts,
        "total_generated": total_generated,
    }
    return prompt_log, summary


def aggregate_metrics(
    prompt_logs: List[PromptLog],
    per_prompt_summaries: List[dict],
    max_positions: int,
) -> dict:
    total_drafts = sum(summary.get("num_drafts", 0) for summary in per_prompt_summaries)
    total_draft_tokens = sum(summary.get("num_draft_tokens", 0) for summary in per_prompt_summaries)
    total_accepted_tokens = sum(summary.get("num_accepted_tokens", 0) for summary in per_prompt_summaries)
    total_tokens = sum(summary.get("total_generated", 0) for summary in per_prompt_summaries)

    position_attempts = [0] * max_positions
    position_accepts = [0] * max_positions
    for summary in per_prompt_summaries:
        for idx in range(max_positions):
            position_attempts[idx] += summary.get("position_attempts", [0] * max_positions)[idx]
            position_accepts[idx] += summary.get("position_accepts", [0] * max_positions)[idx]

    acceptance_per_position = []
    for attempts, accepts in zip(position_attempts, position_accepts):
        if attempts == 0:
            acceptance_per_position.append(0.0)
        else:
            acceptance_per_position.append(accepts / attempts)

    mean_acceptance_length = (
        1.0 + (total_accepted_tokens / total_drafts) if total_drafts > 0 else 1.0
    )

    return {
        "num_prompts": len(prompt_logs),
        "num_drafts": total_drafts,
        "num_draft_tokens": total_draft_tokens,
        "num_accepted_tokens": total_accepted_tokens,
        "total_output_tokens": total_tokens,
        "mean_acceptance_length": mean_acceptance_length,
        "acceptance_per_position": acceptance_per_position,
        "position_attempts": position_attempts,
        "position_accepts": position_accepts,
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.spec_tokens <= 0:
        raise ValueError("--spec-tokens must be a positive integer.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be a positive integer.")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")
    if not (0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in the interval (0, 1].")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0.")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be > 0.")

    set_seed(args.seed)
    random.seed(args.seed)

    prompts = load_prompts(args.prompts_file)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    print(
        f"[startup] Loading tokenizer for drafter model '{args.drafter_model}'...",
        flush=True,
    )
    drafter_tokenizer = load_tokenizer(args.drafter_model)
    print("[startup] Drafter tokenizer ready.", flush=True)
    print(
        f"[startup] Loading tokenizer for verifier model '{args.verifier_model}'...",
        flush=True,
    )
    verifier_tokenizer = load_tokenizer(args.verifier_model)
    print("[startup] Verifier tokenizer ready.", flush=True)
    if drafter_tokenizer.get_vocab() != verifier_tokenizer.get_vocab():
        raise ValueError(
            "Drafter and verifier tokenizers differ. Please ensure both "
            "models share the same tokenizer."
        )
    tokenizer = drafter_tokenizer

    print(
        f"[startup] Loading drafter model '{args.drafter_model}' "
        f"(dtype={args.drafter_dtype}, device_map={args.drafter_device_map or args.device_map}, "
        f"load_in_8bit={args.drafter_load_in_8bit})... this may take a few minutes.",
        flush=True,
    )
    drafter_model = load_model(
        args.drafter_model,
        dtype=resolve_dtype(args.drafter_dtype),
        device_map=args.drafter_device_map or args.device_map,
        load_in_8bit=args.drafter_load_in_8bit,
    )
    print("[startup] Drafter model ready.", flush=True)
    print(
        f"[startup] Loading verifier model '{args.verifier_model}' "
        f"(dtype={args.verifier_dtype}, device_map={args.verifier_device_map or args.device_map})... "
        "this may take a few minutes.",
        flush=True,
    )
    verifier_model = load_model(
        args.verifier_model,
        dtype=resolve_dtype(args.verifier_dtype),
        device_map=args.verifier_device_map or args.device_map,
        load_in_8bit=False,
    )
    print("[startup] Verifier model ready.", flush=True)

    prompt_logs: List[PromptLog] = []
    summaries: List[dict] = []

    total_prompts = len(prompts)
    use_progress_bar = _HAS_TQDM and total_prompts > 1

    if use_progress_bar:
        progress_iter = tqdm(range(total_prompts), desc="Profiling prompts", unit="prompt")
        write_fn = progress_iter.write
    else:
        progress_iter = range(total_prompts)
        write_fn = print

    for idx in progress_iter:
        prompt = prompts[idx]
        prompt_log, summary = speculative_loop(
            prompt,
            drafter_model,
            verifier_model,
            tokenizer,
            spec_tokens=args.spec_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed + idx,
            stop_on_pad=args.stop_on_pad,
            max_prompt_tokens=args.max_prompt_tokens,
            logger=write_fn if args.debug_progress else None,
        )
        prompt_log.prompt_index = idx
        prompt_logs.append(prompt_log)
        summaries.append(summary)

        if args.print_output:
            write_fn("-" * 60)
            write_fn(f"[Prompt {idx}] {prompt}")
            write_fn(f"[Generated] {prompt_log.generated_text}")
            write_fn(
                f"[Iterations] {len(prompt_log.iterations)} "
                f"(accepted tokens: {summary['num_accepted_tokens']}, "
                f"draft tokens: {summary['num_draft_tokens']})"
            )

        if args.details_jsonl:
            record = {
                "prompt_index": idx,
                "prompt": prompt,
                "generated_text": prompt_log.generated_text,
                "metadata": {
                    "drafter_model": args.drafter_model,
                    "verifier_model": args.verifier_model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "spec_tokens": args.spec_tokens,
                },
                "iterations": [
                    {
                        "context_length_before": log.context_length_before,
                        "draft_token_ids": log.draft_token_ids,
                        "draft_tokens": tokenizer.convert_ids_to_tokens(log.draft_token_ids),
                        "accepted_flags": log.accepted_flags,
                        "accepted_count": log.accepted_count,
                        "mismatch_token_id": log.mismatch_token_id,
                        "context_length_after": log.context_length_after,
                    }
                    for log in prompt_log.iterations
                ],
                "reached_eos": prompt_log.reached_eos,
                "total_generated_tokens": prompt_log.total_generated_tokens,
            }
            append_jsonl(args.details_jsonl, record)

        if args.debug_progress:
            write_fn(
                f"    prompt {idx + 1}/{total_prompts} summary: accepted"
                f" {summary['num_accepted_tokens']} of {summary['num_draft_tokens']} draft tokens"
            )

        if (not args.print_output) and (not use_progress_bar):
            write_fn(f"[{idx + 1}/{total_prompts}] prompts profiled")

        if use_progress_bar:
            progress_iter.update(0)  # ensure immediate refresh

    if _HAS_TQDM and hasattr(progress_iter, "close"):
        progress_iter.close()

    summary = aggregate_metrics(prompt_logs, summaries, args.spec_tokens)

    print("Speculative decoding summary")
    print(f"  Prompts processed  : {summary['num_prompts']}")
    print(f"  Draft iterations   : {summary['num_drafts']}")
    print(f"  Draft tokens       : {summary['num_draft_tokens']}")
    print(f"  Accepted tokens    : {summary['num_accepted_tokens']}")
    print(f"  Output tokens      : {summary['total_output_tokens']}")
    print(
        f"  Mean acceptance len: {summary['mean_acceptance_length']:.3f} tokens per draft"
    )
    for pos, rate in enumerate(summary["acceptance_per_position"]):
        print(f"    Acceptance@{pos:<2d}: {rate:.3f}")

    if args.metrics_jsonl:
        metrics_record = {
            "drafter_model": args.drafter_model,
            "verifier_model": args.verifier_model,
            "spec_tokens": args.spec_tokens,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "num_prompts": summary["num_prompts"],
            "num_drafts": summary["num_drafts"],
            "num_draft_tokens": summary["num_draft_tokens"],
            "num_accepted_tokens": summary["num_accepted_tokens"],
            "total_output_tokens": summary["total_output_tokens"],
            "mean_acceptance_length": summary["mean_acceptance_length"],
            "acceptance_per_position": summary["acceptance_per_position"],
            "position_attempts": summary["position_attempts"],
            "position_accepts": summary["position_accepts"],
        }
        append_jsonl(args.metrics_jsonl, metrics_record)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    main()
