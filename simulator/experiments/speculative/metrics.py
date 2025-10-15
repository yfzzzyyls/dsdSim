"""Shared helpers for speculative acceptance regressor diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def positive_class_probabilities(classifier: Any, proba_matrix: np.ndarray) -> np.ndarray:
    """Extract probability mass for the positive (1) class, handling degenerate fits."""
    classes = getattr(classifier, "classes_", None)
    if classes is None or len(classes) == 0:
        return np.zeros(proba_matrix.shape[0], dtype=np.float64)
    classes_arr = np.asarray(classes)
    if classes_arr.shape[0] == 1:
        if classes_arr[0] == 1:
            return proba_matrix[:, 0]
        return np.zeros(proba_matrix.shape[0], dtype=np.float64)
    matches = np.where(classes_arr == 1)[0]
    if matches.size == 0:
        return np.zeros(proba_matrix.shape[0], dtype=np.float64)
    index = int(matches[0])
    return proba_matrix[:, index]


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Compute standard regression diagnostics for accepted-count predictions."""
    y_true_arr = np.asarray(list(y_true), dtype=np.float64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.float64)
    residuals = y_pred_arr - y_true_arr
    abs_err = np.abs(residuals)

    metrics = {
        "mse": float(mean_squared_error(y_true_arr, y_pred_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "bias": float(residuals.mean()),
        "mean_abs_error": float(abs_err.mean()),
        "median_abs_error": float(np.median(abs_err)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "max_abs_error": float(abs_err.max()),
    }
    return metrics


def expected_calibration_error(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    *,
    n_bins: int = 10,
) -> Dict[str, object]:
    """Compute Expected Calibration Error (ECE) and per-bin stats."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    y_true_arr = np.asarray(list(y_true), dtype=np.float64)
    y_prob_arr = np.asarray(list(y_prob), dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true_arr)
    ece = 0.0
    per_bin: List[Dict[str, float]] = []

    for idx in range(n_bins):
        low = bins[idx]
        high = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob_arr >= low) & (y_prob_arr <= high)
        else:
            mask = (y_prob_arr >= low) & (y_prob_arr < high)
        count = int(mask.sum())
        if count == 0:
            per_bin.append(
                {
                    "lower": float(low),
                    "upper": float(high),
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                }
            )
            continue
        accuracy = float(y_true_arr[mask].mean())
        confidence = float(y_prob_arr[mask].mean())
        weight = count / max(1, total)
        ece += weight * abs(accuracy - confidence)
        per_bin.append(
            {
                "lower": float(low),
                "upper": float(high),
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
            }
        )

    return {"ece": float(ece), "bins": per_bin}


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Iterable[float],
    *,
    ece_bins: int | None = None,
) -> Dict[str, object]:
    """Compute classification diagnostics for per-position acceptance."""
    y_true_arr = np.asarray(list(y_true), dtype=np.int32)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int32)
    y_prob_arr = np.asarray(list(y_prob), dtype=np.float64)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    brier = float(np.mean((y_prob_arr - y_true_arr) ** 2))
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    metrics: Dict[str, object] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier": brier,
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    if ece_bins:
        metrics["calibration"] = expected_calibration_error(
            y_true_arr,
            y_prob_arr,
            n_bins=ece_bins,
        )
    return metrics


def bucket_error_summary(
    contexts: Iterable[float],
    abs_errors: Iterable[float],
    *,
    bucket_size: int,
) -> Dict[str, Dict[str, float]]:
    """Aggregate absolute error statistics by context-length bucket."""
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    buckets: Dict[str, List[float]] = {}
    for ctx, err in zip(contexts, abs_errors):
        bucket_idx = int(math.floor(ctx / bucket_size))
        lower = bucket_idx * bucket_size
        upper = lower + bucket_size - 1
        label = f"{lower}-{upper}"
        buckets.setdefault(label, []).append(float(err))

    summary: Dict[str, Dict[str, float]] = {}
    for label, values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary[label] = {
            "count": int(len(arr)),
            "mean_abs_error": float(arr.mean()),
            "median_abs_error": float(np.median(arr)),
            "p95_abs_error": float(np.percentile(arr, 95)),
            "max_abs_error": float(arr.max()),
        }
    return summary


def bucket_classification_summary(
    contexts: Iterable[float],
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Iterable[float],
    *,
    bucket_size: int,
) -> Dict[str, Dict[str, float]]:
    """Aggregate classification quality by context-length bucket."""
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    grouped: Dict[str, List[Tuple[int, int, float]]] = {}
    for ctx, actual, pred, prob in zip(contexts, y_true, y_pred, y_prob):
        bucket_idx = int(math.floor(ctx / bucket_size))
        lower = bucket_idx * bucket_size
        upper = lower + bucket_size - 1
        label = f"{lower}-{upper}"
        grouped.setdefault(label, []).append((int(actual), int(pred), float(prob)))

    summary: Dict[str, Dict[str, float]] = {}
    for label, rows in sorted(grouped.items()):
        actuals = np.array([row[0] for row in rows], dtype=np.int32)
        preds = np.array([row[1] for row in rows], dtype=np.int32)
        probs = np.array([row[2] for row in rows], dtype=np.float64)
        support = len(actuals)
        if support == 0:
            continue
        accuracy = float((actuals == preds).mean())
        positive_rate = float(actuals.mean())
        predicted_rate = float(preds.mean())
        confidence = float(probs.mean())

        summary[label] = {
            "count": int(support),
            "accuracy": accuracy,
            "true_positive_rate": positive_rate,
            "predicted_positive_rate": predicted_rate,
            "mean_confidence": confidence,
        }
    return summary


def position_accuracy_summary(
    positions: Iterable[float],
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """Compute acceptance accuracy per speculative position."""
    grouped: Dict[int, List[Tuple[int, int]]] = {}
    for pos, actual, pred in zip(positions, y_true, y_pred):
        key = int(pos)
        grouped.setdefault(key, []).append((int(actual), int(pred)))

    summary: Dict[str, Dict[str, float]] = {}
    for pos in sorted(grouped.keys()):
        rows = grouped[pos]
        actuals = np.array([row[0] for row in rows], dtype=np.int32)
        preds = np.array([row[1] for row in rows], dtype=np.int32)
        support = len(actuals)
        summary[str(pos)] = {
            "count": int(support),
            "accuracy": float((actuals == preds).mean()) if support else 0.0,
            "accept_rate": float(actuals.mean()) if support else 0.0,
            "predicted_accept_rate": float(preds.mean()) if support else 0.0,
        }
    return summary
