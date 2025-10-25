#!/usr/bin/env python3
"""
Evaluate a trained acceptance regressor bundle on held-out detail logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.acceptance.metrics import (
    bucket_classification_summary,
    bucket_error_summary,
    classification_metrics,
    position_accuracy_summary,
    positive_class_probabilities,
    regression_metrics,
)
from src.acceptance.train_regressor import build_datasets, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained acceptance regressor on held-out logs.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to joblib bundle produced by train_regressor.py.",
    )
    parser.add_argument(
        "--details-jsonl",
        type=Path,
        required=True,
        help="Held-out details JSONL file (from speculative_profiler.py --details-jsonl).",
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
        "--bucket-size",
        type=int,
        default=256,
        help="Context-length bucket size for summary diagnostics (set 0 to disable).",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins for calibration metrics (default 10).",
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
    *,
    bucket_size: int,
    ece_bins: int,
) -> dict:
    reg = model_bundle["regressor"]
    clf = model_bundle["classifier"]

    X_count, y_count, X_accept, y_accept, _ = build_datasets(records, spec_tokens=spec_tokens)

    count_pred = reg.predict(X_count)

    prob_matrix = clf.predict_proba(X_accept)
    accept_prob = positive_class_probabilities(clf, prob_matrix)
    if len(getattr(clf, "classes_", [])) == 1:
        accept_pred = np.full(len(X_accept), clf.classes_[0], dtype=np.int32)
    else:
        accept_pred = (accept_prob >= 0.5).astype(np.int32)

    reg_metrics = regression_metrics(y_count, count_pred)
    clf_metrics = classification_metrics(
        y_accept,
        accept_pred,
        accept_prob,
        ece_bins=ece_bins if ece_bins > 0 else None,
    )

    abs_err = np.abs(count_pred - y_count)
    bucketed = {}
    if bucket_size > 0:
        bucketed["count_regression"] = bucket_error_summary(
            X_count[:, 0],
            abs_err,
            bucket_size=bucket_size,
        )
        bucketed["accept_classification"] = bucket_classification_summary(
            X_accept[:, 0],
            y_accept,
            accept_pred,
            accept_prob,
            bucket_size=bucket_size,
        )

    pos_accuracy = position_accuracy_summary(X_accept[:, 1], y_accept, accept_pred)

    metrics = {
        "count_regression": reg_metrics,
        "accept_classification": clf_metrics,
        "count_mse": float(reg_metrics["mse"]),
        "count_mae": float(reg_metrics["mae"]),
        "accept_accuracy": float(clf_metrics["accuracy"]),
        "num_iterations": int(len(X_count)),
        "num_positions": int(len(X_accept)),
        "bucket_size": int(bucket_size) if bucket_size > 0 else None,
        "bucketed": bucketed,
        "position_accuracy": pos_accuracy,
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
    metrics = evaluate(
        bundle,
        records,
        spec_tokens,
        bucket_size=args.bucket_size,
        ece_bins=args.ece_bins,
    )

    if args.print_report:
        print("Acceptance regressor evaluation")
        print(
            f"  iterations={metrics['num_iterations']} positions={metrics['num_positions']}"
        )
        count = metrics["count_regression"]
        print(
            "  Count regression: "
            f"MSE={count['mse']:.4f} MAE={count['mae']:.4f} "
            f"bias={count['bias']:.4f} p95_abs={count['p95_abs_error']:.4f}"
        )
        accept = metrics["accept_classification"]
        calib_str = ""
        calibration = accept.get("calibration")
        if calibration:
            calib_str = f" ECE={calibration['ece']:.4f}"
        print(
            "  Accept classifier: "
            f"accuracy={accept['accuracy']:.4f} precision={accept['precision']:.4f} "
            f"recall={accept['recall']:.4f} f1={accept['f1']:.4f} "
            f"brier={accept['brier']:.4f}{calib_str}"
        )
        if metrics.get("bucketed"):
            bucket_keys = list(metrics["bucketed"].keys())
            print(f"  Bucketed diagnostics for: {', '.join(bucket_keys)} (size={metrics['bucket_size']})")

    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_json.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
