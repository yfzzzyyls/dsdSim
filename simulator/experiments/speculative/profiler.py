#!/usr/bin/env python3
# Copyright 2025
# SPDX-License-Identifier: Apache-2.0
"""
Aggregate speculative-decoding iteration logs into acceptance statistics
suitable for the simulator's acceptance-rate interpreter.

Usage
-----
python profiler.py \
  --details-jsonl runs/simple_details.jsonl \
  --output-json acceptance/llama2_7b_vs_70b.json \
  --output-yaml acceptance/llama2_7b_vs_70b.yaml \
  --context-bins 256 512 1024 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate speculative decoding logs into acceptance metrics.",
    )
    parser.add_argument(
        "--details-jsonl",
        type=Path,
        required=True,
        help="JSONL file produced by speculative.py with per-iteration logs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write aggregated metrics to this JSON file.",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        help="Write aggregated metrics to this YAML file.",
    )
    parser.add_argument(
        "--context-bins",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192],
        help="Upper bounds (inclusive) for context length buckets.",
    )
    parser.add_argument(
        "--spec-tokens",
        type=int,
        help="Number of speculative positions (k). If omitted, inferred from data.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        default=[],
        help="Optional key=value metadata to include in the output.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print aggregated summary to stdout.",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            records.append(payload)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def infer_spec_tokens(records: Iterable[dict]) -> int:
    max_positions = 0
    for record in records:
        for iteration in record.get("iterations", []):
            draft_len = len(iteration.get("draft_token_ids", []))
            flags_len = len(iteration.get("accepted_flags", []))
            max_positions = max(max_positions, draft_len, flags_len)
    if max_positions == 0:
        raise ValueError("Unable to infer spec tokens: no iteration data found.")
    return max_positions


def parse_metadata(pairs: Iterable[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Metadata entry '{pair}' must be key=value")
        key, value = pair.split("=", 1)
        metadata[key] = value
    return metadata


def build_buckets(bounds: List[int]) -> List[Tuple[int, Optional[int], str]]:
    """
    Return list of (lower, upper, label) tuples where lower/upper are inclusive.
    Upper may be None to denote open-ended final bucket.
    """
    sorted_bounds = sorted(bounds)
    buckets: List[Tuple[int, Optional[int], str]] = []
    lower = 0
    for upper in sorted_bounds:
        label = f"{lower}-{upper}"
        buckets.append((lower, upper, label))
        lower = upper + 1
    buckets.append((lower, None, f">{sorted_bounds[-1]}"))
    return buckets


def assign_bucket(length: int, buckets: List[Tuple[int, Optional[int], str]]) -> str:
    for lower, upper, label in buckets:
        if upper is None:
            return label
        if lower <= length <= upper:
            return label
    return buckets[-1][2]


def aggregate(
    records: List[dict],
    *,
    spec_tokens: int,
    context_buckets: List[Tuple[int, Optional[int], str]],
) -> dict:
    bucket_stats = {
        label: {
            "iterations": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "per_position": [
                {"attempts": 0, "accepts": 0} for _ in range(spec_tokens)
            ],
        }
        for _, _, label in context_buckets
    }

    overall = {
        "iterations": 0,
        "draft_tokens": 0,
        "accepted_tokens": 0,
        "per_position": [
            {"attempts": 0, "accepts": 0} for _ in range(spec_tokens)
        ],
    }

    for record in records:
        iterations = record.get("iterations", [])
        for iteration in iterations:
            context_len = int(iteration.get("context_length_before", 0))
            flags: List[bool] = list(iteration.get("accepted_flags", []))
            draft_ids: List[int] = list(iteration.get("draft_token_ids", []))
            attempts = len(flags)
            if attempts == 0:
                continue
            accepted = sum(1 for flag in flags if flag)
            label = assign_bucket(context_len, context_buckets)

            bucket = bucket_stats[label]
            bucket["iterations"] += 1
            bucket["draft_tokens"] += len(draft_ids)
            bucket["accepted_tokens"] += accepted

            overall["iterations"] += 1
            overall["draft_tokens"] += len(draft_ids)
            overall["accepted_tokens"] += accepted

            for pos in range(attempts):
                bucket["per_position"][pos]["attempts"] += 1
                overall["per_position"][pos]["attempts"] += 1
                if flags[pos]:
                    bucket["per_position"][pos]["accepts"] += 1
                    overall["per_position"][pos]["accepts"] += 1

    def finalize(stats: dict) -> None:
        attempts_total = sum(pos["attempts"] for pos in stats["per_position"])
        accepts_total = sum(pos["accepts"] for pos in stats["per_position"])
        stats["acceptance_rate"] = (
            accepts_total / attempts_total if attempts_total else 0.0
        )
        stats["mean_acceptance_length"] = (
            1.0 + (stats["accepted_tokens"] / stats["iterations"])
            if stats["iterations"] > 0
            else 0.0
        )
        for pos, entry in enumerate(stats["per_position"]):
            attempts = entry["attempts"]
            entry["position"] = pos
            entry["acceptance"] = entry["accepts"] / attempts if attempts else 0.0

    for stats in bucket_stats.values():
        finalize(stats)
    finalize(overall)

    return {
        "spec_tokens": spec_tokens,
        "context_buckets": context_buckets,
        "buckets": bucket_stats,
        "overall": overall,
    }


def serialize_context_buckets(
    buckets: List[Tuple[int, Optional[int], str]]
) -> List[dict]:
    result: List[dict] = []
    for lower, upper, label in buckets:
        result.append(
            {
                "label": label,
                "lower": lower,
                "upper": upper,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    records = load_records(args.details_jsonl)
    spec_tokens = args.spec_tokens or infer_spec_tokens(records)
    metadata = parse_metadata(args.metadata)
    context_buckets = build_buckets(args.context_bins)

    aggregate_data = aggregate(
        records,
        spec_tokens=spec_tokens,
        context_buckets=context_buckets,
    )

    output_payload = {
        "source": str(args.details_jsonl),
        "metadata": metadata,
        "spec_tokens": aggregate_data["spec_tokens"],
        "context_buckets": serialize_context_buckets(context_buckets),
        "buckets": aggregate_data["buckets"],
        "overall": aggregate_data["overall"],
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2)

    if args.output_yaml:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for --output-yaml but is not installed."
            )
        args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
        with args.output_yaml.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(output_payload, handle, sort_keys=False)

    if args.print_summary:
        overall = output_payload["overall"]
        print("Speculative acceptance summary")
        print(f"  Iterations        : {overall['iterations']}")
        print(f"  Draft tokens      : {overall['draft_tokens']}")
        print(f"  Accepted tokens   : {overall['accepted_tokens']}")
        print(
            f"  Acceptance rate   : {overall['acceptance_rate']:.3f}"
        )
        print(
            f"  Mean accept length: {overall['mean_acceptance_length']:.3f}"
        )
        for bucket_label, stats in output_payload["buckets"].items():
            print(f"Bucket {bucket_label}: acceptance_rate={stats['acceptance_rate']:.3f}, "
                  f"iterations={stats['iterations']}")


if __name__ == "__main__":
    main()
