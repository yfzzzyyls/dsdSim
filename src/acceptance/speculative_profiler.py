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
        help="Number of speculative tokens the drafter proposes each iteration.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit number of prompts processed from --prompts-file (default: all).",
    )
    parser.add_argument(
        "--prompts-file",
        help="Optional JSONL file containing prompts. If omitted, uses built-ins.",
    )
    parser.add_argument(
        "--metrics-jsonl",
        help="Path to write summary metrics (JSONL appended per run).",
    )
    parser.add_argument(
        "--details-jsonl",
        help="Path to append per-prompt detailed iteration logs.",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print generated text for each prompt (for debugging).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for drafter (default: 1.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p for drafter (default: 1.0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Sampling top-k for drafter (default: 0 = disabled).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for drafter sampling (default: 1.0).",
    )
    parser.add_argument(
        "--stop-on-pad",
        action="store_true",
        help="Stop generation when the verifier emits PAD token.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=2048,
        help="Truncate prompts longer than this many tokens (default: 2048).",
    )
    parser.add_argument(
        "--debug-progress",
        action="store_true",
        help="Print intermediate stats per prompt (slower).",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Display tqdm progress bar.",
    )
    return parser.parse_args()


def load_model(
    model_name: str,
    *,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    load_in_8bit: bool = False,
) -> AutoModelForCausalLM:
    kwargs = {
        "device_map": device_map,
        "torch_dtype": getattr(torch, torch_dtype),
    }
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_prompts(path: Optional[str], limit: Optional[int]) -> List[str]:
    if path is None:
        prompts = DEFAULT_PROMPTS.copy()
    else:
        prompts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    prompts.append(obj.get("prompt", line))
                except json.JSONDecodeError:
                    prompts.append(line)
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def _build_sampling_metadata(
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Tuple[VllmLogitsProcessors, VllmSamplingMetadata]:
    processors = VllmLogitsProcessors(
        temperature=max(temperature, VLLM_SAMPLING_EPS),
        min_p=0.0,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=max(repetition_penalty, VLLM_SAMPLING_EPS),
    )
    metadata = VllmSamplingMetadata(
        prompt_adapter_request_ids=None,
        position_ids=None,
        span_ids=None,
    )
    return processors, metadata


def _sample_next_token(
    logits: torch.Tensor,
    sampler: VllmSampler,
    processors: VllmLogitsProcessors,
    metadata: VllmSamplingMetadata,
) -> int:
    token = sampler(logits, processors, metadata)
    return token.item()


def speculative_loop(
    prompt: str,
    drafter: AutoModelForCausalLM,
    verifier: AutoModelForCausalLM,
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
    max_prompt_tokens: int,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[PromptLog, Dict[str, int]]:
    set_seed(seed)
    processors, metadata = _build_sampling_metadata(
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    )
    sampler = VllmSampler()

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded["input_ids"].to(drafter.device)
    attention_mask = encoded["attention_mask"].to(drafter.device)

    total_generated = 0
    iterations: List[IterationLog] = []
    context_ids = input_ids
    context_mask = attention_mask
    reached_eos = False
    generated_tokens: List[int] = []

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0

    while total_generated < max_tokens and not reached_eos:
        num_drafts += 1
        context_len = context_ids.shape[-1]

        draft_logits = drafter(
            input_ids=context_ids,
            attention_mask=context_mask,
        ).logits[:, -1, :]

        draft_tokens: List[int] = []
        for _ in range(spec_tokens):
            token_id = _sample_next_token(draft_logits, sampler, processors, metadata)
            draft_tokens.append(token_id)
            # Append token and continue sampling autoregressively
            next_input = torch.tensor([[token_id]], device=context_ids.device)
            context_ids = torch.cat([context_ids, next_input], dim=1)
            context_mask = torch.cat(
                [context_mask, torch.ones_like(next_input, device=context_ids.device)],
                dim=1,
            )
            draft_logits = drafter(
                input_ids=context_ids,
                attention_mask=context_mask,
            ).logits[:, -1, :]
        num_draft_tokens += len(draft_tokens)

        verify_ids = torch.cat([context_ids[:, :-len(draft_tokens)], torch.tensor([draft_tokens], device=context_ids.device)], dim=1)

        verifier_out = verifier(
            input_ids=verify_ids[:, :-1],
            attention_mask=torch.ones_like(verify_ids[:, :-1], device=verify_ids.device),
        )
        verify_logits = verifier_out.logits[:, -len(draft_tokens) :, :]

        accepted = []
        mismatch_token = None
        for idx, draft_token in enumerate(draft_tokens):
            verifier_token = torch.argmax(verify_logits[:, idx, :], dim=-1).item()
            if verifier_token == draft_token:
                accepted.append(True)
                num_accepted_tokens += 1
                generated_tokens.append(draft_token)
                total_generated += 1
                if draft_token == tokenizer.eos_token_id or (
                    stop_on_pad and draft_token == tokenizer.pad_token_id
                ):
                    reached_eos = True
                    break
            else:
                accepted.append(False)
                mismatch_token = verifier_token
                generated_tokens.append(verifier_token)
                total_generated += 1
                context_ids = torch.cat(
                    [context_ids[:, :- (len(draft_tokens) - idx)], torch.tensor([[verifier_token]], device=context_ids.device)],
                    dim=1,
                )
                context_mask = torch.ones_like(context_ids, device=context_ids.device)
                if verifier_token == tokenizer.eos_token_id or (
                    stop_on_pad and verifier_token == tokenizer.pad_token_id
                ):
                    reached_eos = True
                break
        else:
            context_ids = torch.cat(
                [context_ids, torch.tensor([[draft_tokens[-1]]], device=context_ids.device)],
                dim=1,
            )
            context_mask = torch.ones_like(context_ids, device=context_ids.device)

        iterations.append(
            IterationLog(
                context_length_before=context_len,
                draft_token_ids=draft_tokens,
                accepted_flags=accepted,
                accepted_count=sum(accepted),
                mismatch_token_id=mismatch_token,
                context_length_after=context_ids.shape[-1],
            )
        )

        if logger:
            logger(
                f"Draft iteration {num_drafts}: accepted {sum(accepted)} / {len(draft_tokens)} tokens "
                f"(context {context_len} → {context_ids.shape[-1]})"
            )

        if total_generated >= max_tokens:
            break

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
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
        "total_output_tokens": total_generated,
    }
    return prompt_log, summary


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record))
        f.write("\n")


def aggregate_metrics(
    prompt_logs: Sequence[PromptLog],
    summaries: Sequence[Dict[str, int]],
    spec_tokens: int,
) -> Dict[str, float]:
    total_prompts = len(prompt_logs)
    total_drafts = sum(summary["num_drafts"] for summary in summaries)
    total_draft_tokens = sum(summary["num_draft_tokens"] for summary in summaries)
    total_accepted_tokens = sum(summary["num_accepted_tokens"] for summary in summaries)
    total_output_tokens = sum(summary["total_output_tokens"] for summary in summaries)

    acceptance_per_position = [0.0] * spec_tokens
    position_attempts = [0] * spec_tokens
    position_accepts = [0] * spec_tokens

    for prompt in prompt_logs:
        for iteration in prompt.iterations:
            for pos, accepted in enumerate(iteration.accepted_flags):
                position_attempts[pos] += 1
                if accepted:
                    position_accepts[pos] += 1

    for pos in range(spec_tokens):
        attempts = position_attempts[pos]
        acceptance_per_position[pos] = (
            position_accepts[pos] / attempts if attempts else 0.0
        )

    return {
        "num_prompts": total_prompts,
        "num_drafts": total_drafts,
        "num_draft_tokens": total_draft_tokens,
        "num_accepted_tokens": total_accepted_tokens,
        "total_output_tokens": total_output_tokens,
        "mean_acceptance_length": (
            total_accepted_tokens / total_drafts if total_drafts else 0.0
        ),
        "acceptance_per_position": acceptance_per_position,
        "position_attempts": position_attempts,
        "position_accepts": position_accepts,
    }


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file, args.max_prompts)
    if not prompts:
        raise ValueError("No prompts provided or loaded.")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.drafter_model)
    print("Tokenizer loaded.")

    print(f"Loading drafter model: {args.drafter_model}")
    drafter_model = load_model(
        args.drafter_model,
        device_map=args.drafter_device_map or args.device_map,
        torch_dtype=args.drafter_dtype,
        load_in_8bit=args.drafter_load_in_8bit,
    )
    print("Drafter ready.")

    print(f"Loading verifier model: {args.verifier_model}")
    verifier_model = load_model(
        args.verifier_model,
        device_map=args.verifier_device_map or args.device_map,
        torch_dtype=args.verifier_dtype,
    )
    print("Verifier ready.")

    prompt_logs: List[PromptLog] = []
    summaries: List[Dict[str, int]] = []

    total_prompts = len(prompts)
    use_progress_bar = args.progress_bar and _HAS_TQDM and total_prompts > 1

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
            append_jsonl(Path(args.details_jsonl), record)

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
        append_jsonl(Path(args.metrics_jsonl), metrics_record)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    main()
