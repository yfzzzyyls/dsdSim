#!/usr/bin/env python3
# Copyright 2025
# SPDX-License-Identifier: Apache-2.0
"""
Evaluate a trained acceptance-rate regressor bundle on held-out detail logs.

Example:
    python evaluate_regressor.py \
        --model acceptance/llama2_7b_vs_70b.joblib \
        --details-jsonl results/gsm8k_test_details.jsonl \
        --print-report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from simulator.experiments.speculative.regressor import (
    build_datasets,
    load_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained acceptance regressor on held-out logs.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to joblib bundle produced by regressor.py.",
    )
    parser.add_argument(
        "--details-jsonl",
        type=Path,
        required=True,
        help="Held-out details JSONL file (from speculative.py --details-jsonl).",
    )
    parser.add_argument(
        "--spec-tokens",
        type=int,
        help="Override number of speculative tokens (defaults to bundle value).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print metrics to stdout.",
    )
    return parser.parse_args()


def evaluate(
    model_bundle: dict,
    records: list[dict],
    spec_tokens: int,
) -> dict:
    reg = model_bundle["regressor"]
    clf = model_bundle["classifier"]

    X_count, y_count, X_accept, y_accept = build_datasets(records, spec_tokens)

    count_pred = reg.predict(X_count)
    mse = mean_squared_error(y_count, count_pred)
    mae = mean_absolute_error(y_count, count_pred)

    accept_pred = clf.predict(X_accept)
    accept_acc = accuracy_score(y_accept, accept_pred)

    metrics = {
        "count_mse": float(mse),
        "count_mae": float(mae),
        "accept_accuracy": float(accept_acc),
        "num_iterations": int(len(X_count)),
        "num_positions": int(len(X_accept)),
    }
    return metrics


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    spec_tokens = args.spec_tokens or bundle.get("spec_tokens")
    if spec_tokens is None:
        raise ValueError(
            "spec_tokens must be provided either via --spec-tokens or stored in the bundle."
        )

    records = load_records(args.details_jsonl)
    metrics = evaluate(bundle, records, spec_tokens)

    if args.print_report:
        print("Acceptance regressor evaluation")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_json.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
