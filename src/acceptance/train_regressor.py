#!/usr/bin/env python3
# Copyright 2025
# SPDX-License-Identifier: Apache-2.0
"""
Train acceptance-rate regressors (count + classifier) from speculative logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from src.acceptance.metrics import (
    classification_metrics,
    positive_class_probabilities,
    regression_metrics,
)

# Feature layout mirrors VIDUR's baseline
COUNT_FEATURES = [
    "context_length",
    "pending_tokens",
    "queue_depth",
]
ACCEPT_FEATURES = [
    "context_length",
    "position",
    "pending_tokens",
    "queue_depth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train acceptance-rate regressors from speculative logs.",
    )
    parser.add_argument(
        "--details-jsonl",
        type=Path,
        required=True,
        help="JSONL file produced by speculative_profiler.py (--details-jsonl).",
    )
    parser.add_argument(
        "--spec-tokens",
        type=int,
        required=True,
        help="Number of speculative positions (k).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        required=True,
        help="Destination joblib bundle for the trained models.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        default=[],
        help="Optional key=value metadata stored alongside the bundle.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation (default 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=1337,
        help="Random seed for splits and model initialisation.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees per RandomForest (default 200).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (default None => unrestricted).",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print metrics after training.",
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


def parse_metadata(entries: Iterable[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Metadata entry '{entry}' must be key=value")
        key, value = entry.split("=", 1)
        metadata[key] = value
    return metadata


def _default_feature_row(
    context_length: float,
    position: Optional[int],
    pending_tokens: float,
    queue_depth: float,
) -> List[float]:
    row = [float(context_length)]
    if position is not None:
        row.append(float(position))
    row.append(float(pending_tokens))
    row.append(float(queue_depth))
    return row


def build_datasets(
    records: Iterable[dict],
    *,
    spec_tokens: int,
    metadata: Mapping[str, str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
    count_rows: List[List[float]] = []
    count_targets: List[float] = []
    accept_rows: List[List[float]] = []
    accept_targets: List[int] = []

    categorical_maps: Dict[str, Dict[str, int]] = defaultdict(dict)

    for record in records:
        metadata_row = record.get("metadata") or {}
        iterations = record.get("iterations", [])
        for iteration in iterations:
            context_len = float(iteration.get("context_length_before", 0.0))
            pending = float(metadata_row.get("pending_tokens", 0.0))
            queue_depth = float(metadata_row.get("queue_depth", 0.0))
            flags: List[bool] = list(iteration.get("accepted_flags", []))
            accepted_count = sum(1 for flag in flags if flag)

            count_rows.append(
                _default_feature_row(context_len, None, pending, queue_depth)
            )
            count_targets.append(float(accepted_count))

            for pos, accepted in enumerate(flags):
                if pos >= spec_tokens:
                    break
                accept_rows.append(
                    _default_feature_row(context_len, pos, pending, queue_depth)
                )
                accept_targets.append(1 if accepted else 0)

    X_count = np.array(count_rows, dtype=np.float32)
    y_count = np.array(count_targets, dtype=np.float32)
    X_accept = np.array(accept_rows, dtype=np.float32)
    y_accept = np.array(accept_targets, dtype=np.int32)

    if metadata:
        # Track categorical maps (useful for future experts)
        for key, value in metadata.items():
            cat_map = categorical_maps.setdefault(key, {})
            if value not in cat_map:
                cat_map[value] = len(cat_map)

    return X_count, y_count, X_accept, y_accept, categorical_maps


def train_models(
    X_count: np.ndarray,
    y_count: np.ndarray,
    X_accept: np.ndarray,
    y_accept: np.ndarray,
    *,
    n_estimators: int,
    max_depth: Optional[int],
    random_state: int,
    test_size: float,
) -> Tuple[
    RandomForestRegressor,
    RandomForestClassifier,
    Dict[str, float],
]:
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_count,
        y_count,
        test_size=test_size,
        random_state=random_state,
    )
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    reg.fit(Xc_train, yc_train)
    yc_pred = reg.predict(Xc_test)
    reg_diagnostics = regression_metrics(yc_test, yc_pred)

    Xa_train, Xa_test, ya_train, ya_test = train_test_split(
        X_accept,
        y_accept,
        test_size=test_size,
        random_state=random_state,
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(Xa_train, ya_train)
    ya_proba = clf.predict_proba(Xa_test)
    ya_prob = positive_class_probabilities(clf, ya_proba)
    if len(getattr(clf, "classes_", [])) == 1:
        ya_pred = np.full(len(Xa_test), clf.classes_[0], dtype=np.int32)
    else:
        ya_pred = (ya_prob >= 0.5).astype(np.int32)
    clf_diagnostics = classification_metrics(ya_test, ya_pred, ya_prob, ece_bins=10)

    metrics = {
        "count_regression": reg_diagnostics,
        "accept_classification": clf_diagnostics,
        "count_mse": float(mean_squared_error(yc_test, yc_pred)),
        "accept_accuracy": float(accuracy_score(ya_test, ya_pred)),
        "train_iterations": int(len(X_count)),
        "train_positions": int(len(X_accept)),
    }
    return reg, clf, metrics


def main() -> None:
    args = parse_args()
    records = load_records(args.details_jsonl)
    metadata = parse_metadata(args.metadata)

    X_count, y_count, X_accept, y_accept, categorical_maps = build_datasets(
        records,
        spec_tokens=args.spec_tokens,
        metadata=metadata,
    )

    reg_model, clf_model, metrics = train_models(
        X_count,
        y_count,
        X_accept,
        y_accept,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    bundle = {
        "metadata": metadata,
        "spec_tokens": args.spec_tokens,
        "feature_columns": {
            "count": COUNT_FEATURES,
            "accept": ACCEPT_FEATURES,
        },
        "categorical_maps": categorical_maps,
        "regressor": reg_model,
        "classifier": clf_model,
        "metrics": metrics,
        "details_source": str(args.details_jsonl),
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model)

    if args.print_report:
        print("Acceptance regressor training report")
        print(
            f"  iterations={metrics['train_iterations']} positions={metrics['train_positions']}"
        )
        count = metrics["count_regression"]
        print(
            "  Count regression: "
            f"MSE={count['mse']:.4f} MAE={count['mae']:.4f} "
            f"bias={count['bias']:.4f} p95_abs={count['p95_abs_error']:.4f}"
        )
        accept = metrics["accept_classification"]
        calib = accept.get("calibration")
        calib_str = f" ECE={calib['ece']:.4f}" if calib else ""
        print(
            "  Accept classifier: "
            f"accuracy={accept['accuracy']:.4f} precision={accept['precision']:.4f} "
            f"recall={accept['recall']:.4f} f1={accept['f1']:.4f} "
            f"brier={accept['brier']:.4f}{calib_str}"
        )


if __name__ == "__main__":
    main()
