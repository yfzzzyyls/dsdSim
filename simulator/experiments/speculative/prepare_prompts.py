#!/usr/bin/env python3
# Copyright 2025
# SPDX-License-Identifier: Apache-2.0
"""
Utility to sample prompts from a Hugging Face dataset (saved via load_from_disk)
and export train/test JSONL files for speculative decoding profiling.

Example:
    python prepare_prompts.py \
        --dataset-path simulator/thirdparty/benchmarks/gsm8k \
        --split train \
        --text-column question \
        --train-size 640 \
        --test-size 160 \
        --train-output prompts/gsm8k_train.jsonl \
        --test-output prompts/gsm8k_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample prompts from a dataset and export JSONL train/test files.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to dataset directory (load_from_disk).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="Column containing the prompt text.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        required=True,
        help="Number of prompts to export for training.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        required=True,
        help="Number of prompts to export for testing.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Destination JSONL file for training prompts.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        required=True,
        help="Destination JSONL file for testing prompts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for shuffling (default: 1337).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional string prefixed to each prompt.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional string appended to each prompt.",
    )
    return parser.parse_args()


def select_examples(
    dataset_path: Path,
    split: str,
    text_column: str,
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    dataset = load_from_disk(str(dataset_path))
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset {dataset_path}.")
    table = dataset[split]
    if text_column not in table.column_names:
        raise ValueError(
            f"Column '{text_column}' not found in dataset. "
            f"Available columns: {table.column_names}"
        )

    total_required = train_size + test_size
    if len(table) < total_required:
        raise ValueError(
            f"Dataset split '{split}' contains {len(table)} rows, "
            f"fewer than requested {total_required}."
        )

    indices = list(range(len(table)))
    random.Random(seed).shuffle(indices)
    selected_indices = indices[:total_required]
    train_indices = selected_indices[:train_size]
    test_indices = selected_indices[train_size:]

    train_prompts = [table[i][text_column] for i in train_indices]
    test_prompts = [table[i][text_column] for i in test_indices]

    return train_prompts, test_prompts


def write_jsonl(path: Path, prompts: list[str], prefix: str, suffix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            payload = {
                "prompt": f"{prefix}{prompt}{suffix}",
            }
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")


def main() -> None:
    args = parse_args()
    train_prompts, test_prompts = select_examples(
        dataset_path=args.dataset_path,
        split=args.split,
        text_column=args.text_column,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    write_jsonl(args.train_output, train_prompts, args.prefix, args.suffix)
    write_jsonl(args.test_output, test_prompts, args.prefix, args.suffix)
    print(
        f"Wrote {len(train_prompts)} training prompts to {args.train_output} "
        f"and {len(test_prompts)} test prompts to {args.test_output}."
    )


if __name__ == "__main__":
    main()
