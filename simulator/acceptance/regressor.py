"""Acceptance regressor loader and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import joblib
import numpy as np


class AcceptanceRegressor:
    """Wrapper around the acceptance count/position models."""

    def __init__(
        self,
        *,
        spec_tokens: int,
        regressor,
        classifier,
        feature_columns: Mapping[str, Sequence[str]],
        metadata: Mapping[str, object],
    ) -> None:
        if spec_tokens <= 0:
            spec_tokens = 1
        self.spec_tokens = int(spec_tokens)
        self._regressor = regressor
        self._classifier = classifier
        self._feature_columns = {
            key: list(cols) for key, cols in (feature_columns or {}).items()
        }
        self.metadata = dict(metadata)

        classes = getattr(classifier, "classes_", None)
        positive_index: Optional[int] = None
        positive_value = 1
        if classes is not None:
            classes = list(classes)
            if len(classes) == 1:
                positive_value = classes[0]
                positive_index = 0
            else:
                if 1 in classes:
                    positive_index = classes.index(1)
                    positive_value = 1
                else:
                    positive_index = classes.index(max(classes))
                    positive_value = classes[positive_index]
        self._positive_class_index = positive_index
        self._positive_class_value = positive_value

        self._count_features = self._feature_columns.get("count") or ["context_length"]
        self._accept_features = self._feature_columns.get("accept") or ["context_length", "position"]

    @classmethod
    def from_file(cls, path: str | Path) -> "AcceptanceRegressor":
        bundle = joblib.load(Path(path))
        spec_tokens = int(bundle.get("spec_tokens", 1))
        regressor = bundle.get("regressor")
        classifier = bundle.get("classifier")
        if regressor is None or classifier is None:
            raise ValueError("Acceptance bundle missing regressor or classifier")
        feature_columns = bundle.get("feature_columns") or {}
        metadata = bundle.get("metadata") or {}
        return cls(
            spec_tokens=spec_tokens,
            regressor=regressor,
            classifier=classifier,
            feature_columns=feature_columns,
            metadata=metadata,
        )

    def expected_accepts(self, context_length: float) -> float:
        features = np.array([[self._feature_value(col, context_length, 0) for col in self._count_features]], dtype=np.float32)
        prediction = self._regressor.predict(features)
        return float(max(0.0, prediction[0]))

    def position_probabilities(
        self,
        *,
        context_length: float,
        depth: int,
        default: Optional[float] = None,
    ) -> List[float]:
        depth = int(max(0, depth))
        if depth == 0:
            return []
        limit = min(depth, self.spec_tokens)
        rows: List[List[float]] = []
        for pos in range(limit):
            row = [self._feature_value(col, context_length, pos) for col in self._accept_features]
            rows.append(row)
        probs: List[float] = []
        if rows:
            arr = np.array(rows, dtype=np.float32)
            if hasattr(self._classifier, "predict_proba"):
                proba = self._classifier.predict_proba(arr)
                if proba.ndim == 2:
                    if proba.shape[1] == 1:
                        if self._positive_class_value == 1:
                            probs.extend(float(p[0]) for p in proba)
                        else:
                            probs.extend(0.0 for _ in proba)
                    else:
                        idx = self._positive_class_index if self._positive_class_index is not None else (proba.shape[1] - 1)
                        probs.extend(float(p[idx]) for p in proba)
                else:
                    probs.extend(0.0 for _ in rows)
            else:
                predictions = self._classifier.predict(arr)
                probs.extend(1.0 if pred == self._positive_class_value else 0.0 for pred in predictions)
        if len(probs) < depth:
            tail = probs[-1] if probs else (default if default is not None else 0.0)
            probs.extend([float(tail)] * (depth - len(probs)))
        clipped = [max(0.0, min(1.0, float(p))) for p in probs]
        return clipped

    def _feature_value(self, column: str, context_length: float, position: int) -> float:
        if column == "context_length":
            return float(context_length)
        if column == "position":
            return float(position)
        if column == "bias":
            return 1.0
        return 0.0
