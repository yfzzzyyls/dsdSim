"""Acceptance regressor loader and helpers."""

from __future__ import annotations

from collections import OrderedDict
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
        categorical_maps: Optional[Mapping[str, Mapping[str, int]]] = None,
        algorithm: str = "random_forest",
        experts: Optional[Mapping[str, Mapping[str, Any]]] = None,
        group_features: Optional[Sequence[str]] = None,
        group_info: Optional[Mapping[str, Mapping[str, object]]] = None,
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
        self._categorical_maps: Dict[str, Dict[str, int]] = {
            key: dict(value) for key, value in (categorical_maps or {}).items()
        }
        self.algorithm = str(algorithm or "random_forest").lower()
        self._experts: Dict[str, Dict[str, Any]] = {}
        for key, models in (experts or {}).items():
            if isinstance(models, Mapping):
                reg = models.get("regressor")
                clf = models.get("classifier")
                if reg is not None and clf is not None:
                    self._experts[str(key)] = {"regressor": reg, "classifier": clf}
        self._group_features = list(group_features or [])
        self._group_info = {str(k): dict(v) for k, v in (group_info or {}).items()}

        # Memoisation cache for expensive probability lookups
        self._prob_cache: "OrderedDict[Tuple[Any, ...], List[float]]" = OrderedDict()
        self._prob_cache_limit = 2048  # configurable cap

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
        self._accept_features = self._feature_columns.get("accept") or [
            "context_length",
            "position",
        ]
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
        categorical_maps = bundle.get("categorical_maps") or {}
        algorithm = bundle.get("algorithm", "random_forest")
        experts = bundle.get("experts") or {}
        group_features = bundle.get("group_features") or []
        group_info = bundle.get("group_info") or {}
        return cls(
            spec_tokens=spec_tokens,
            regressor=regressor,
            classifier=classifier,
            feature_columns=feature_columns,
            metadata=metadata,
            categorical_maps=categorical_maps,
            algorithm=algorithm,
            experts=experts,
            group_features=group_features,
            group_info=group_info,
        )

    def expected_accepts(
        self,
        context_length: float,
        *,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> float:
        row = np.array(
            [self._make_feature_row(self._count_features, context_length, None, feature_context)],
            dtype=np.float32,
        )
        regressor = self._select_regressor(feature_context)
        prediction = regressor.predict(row)
        return float(max(0.0, prediction[0]))

    def position_probabilities(
        self,
        *,
        context_length: float,
        depth: int,
        default: Optional[float] = None,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> List[float]:
        from src import sim as _sim_module
        profiler = getattr(_sim_module, "_GLOBAL_PROFILER", None)
        start_total = time.perf_counter()
        depth = int(max(0, depth))
        if depth == 0:
            if profiler is not None:
                profiler["acceptance_calls"] = profiler.get("acceptance_calls", 0) + 1
                profiler["acceptance_proba_ms"] += (time.perf_counter() - start_total) * 1000.0
            return []
        key = self._make_cache_key(context_length, depth, feature_context)
        cached = self._prob_cache.get(key)
        if cached is not None:
            self._prob_cache.move_to_end(key)
            if profiler is not None:
                profiler["acceptance_calls"] = profiler.get("acceptance_calls", 0) + 1
                profiler["acceptance_cached"] = profiler.get("acceptance_cached", 0) + 1
                profiler["acceptance_proba_ms"] += (time.perf_counter() - start_total) * 1000.0
            return cached[:depth]

        limit = min(depth, self.spec_tokens)
        rows: List[List[float]] = []
        for pos in range(limit):
            rows.append(
                self._make_feature_row(self._accept_features, context_length, pos, feature_context)
            )
        probs: List[float] = []
        classifier_time = 0.0
        regressor_time = 0.0
        if rows:
            arr = np.array(rows, dtype=np.float32)
            classifier = self._select_classifier(feature_context)
            if hasattr(classifier, "predict_proba"):
                start_classifier = time.perf_counter()
                proba = classifier.predict_proba(arr)
                classifier_time += (time.perf_counter() - start_classifier) * 1000.0
                if proba.ndim == 2:
                    if proba.shape[1] == 1:
                        if self._positive_class_value == 1:
                            probs.extend(float(p[0]) for p in proba)
                        else:
                            probs.extend(0.0 for _ in proba)
                    else:
                        idx = (
                            self._positive_class_index
                            if self._positive_class_index is not None
                            else (proba.shape[1] - 1)
                        )
                        probs.extend(float(p[idx]) for p in proba)
                else:
                    probs.extend(0.0 for _ in rows)
            else:
                predictions = classifier.predict(arr)
                probs.extend(1.0 if pred == self._positive_class_value else 0.0 for pred in predictions)
        if len(probs) < depth:
            tail = probs[-1] if probs else (default if default is not None else 0.0)
            probs.extend([float(tail)] * (depth - len(probs)))
        clipped = [max(0.0, min(1.0, float(p))) for p in probs]
        self._store_cache_entry(key, clipped)
        if profiler is not None:
            profiler["acceptance_calls"] = profiler.get("acceptance_calls", 0) + 1
            profiler["acceptance_proba_ms"] += (time.perf_counter() - start_total) * 1000.0
            profiler["acceptance_classifier_ms"] += classifier_time
            profiler["acceptance_regressor_ms"] += regressor_time
        return clipped

    def position_probabilities_batch(
        self,
        requests: Sequence[Tuple[float, int, Optional[Mapping[str, Any]]]],
        *,
        default: Optional[float] = None,
    ) -> List[List[float]]:
        results: List[List[float]] = []
        missing: Dict[Tuple[Any, ...], Tuple[int, float, int, Optional[Mapping[str, Any]]]] = {}

        for idx, (context_length, depth, feature_context) in enumerate(requests):
            key = self._make_cache_key(context_length, depth, feature_context)
            cached = self._prob_cache.get(key)
            if cached is not None:
                self._prob_cache.move_to_end(key)
                results.append(cached[:depth])
            else:
                missing[key] = (idx, context_length, depth, feature_context)
                results.append([])

        # Compute missing entries one by one (still amortised through cache)
        for key, (_, context_length, depth, feature_context) in missing.items():
            self.position_probabilities(
                context_length=context_length,
                depth=depth,
                default=default,
                feature_context=feature_context,
            )

        # Populate results now that cache is filled
        for i, (context_length, depth, feature_context) in enumerate(requests):
            if not results[i]:
                key = self._make_cache_key(context_length, depth, feature_context)
                results[i] = self._prob_cache[key][:depth]
        return results

    def _make_feature_row(
        self,
        feature_names: Sequence[str],
        context_length: float,
        position: Optional[int],
        feature_context: Optional[Mapping[str, Any]],
    ) -> List[float]:
        ctx = dict(feature_context or {})
        row: List[float] = []
        for name in feature_names:
            if name == "context_length":
                row.append(float(context_length))
            elif name == "position":
                row.append(float(position if position is not None else 0))
            elif name == "spec_tokens":
                value = ctx.get("spec_tokens", self.spec_tokens)
                try:
                    row.append(float(value))
                except (TypeError, ValueError):
                    row.append(float(self.spec_tokens))
            elif name == "bias":
                row.append(1.0)
            elif name in self._categorical_maps:
                row.append(float(self._encode_category(name, ctx.get(name))))
            else:
                value = ctx.get(name)
                if value is None:
                    row.append(0.0)
                else:
                    try:
                        row.append(float(value))
                    except (TypeError, ValueError):
                        row.append(0.0)
        return row

    def _make_cache_key(
        self,
        context_length: float,
        depth: int,
        feature_context: Optional[Mapping[str, Any]],
    ) -> Tuple[Any, ...]:
        ctx_items: Tuple[Any, ...] = ()
        if feature_context:
            ctx_items = tuple(
                (str(k), self._normalise_value(v))
                for k, v in sorted(feature_context.items())
            )
        return (round(float(context_length), 3), int(depth), ctx_items)

    def _normalise_value(self, value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, (list, tuple)):
            return tuple(self._normalise_value(v) for v in value)
        if isinstance(value, dict):
            return tuple((str(k), self._normalise_value(v)) for k, v in sorted(value.items()))
        return value

    def _store_cache_entry(self, key: Tuple[Any, ...], value: List[float]) -> None:
        self._prob_cache[key] = list(value)
        self._prob_cache.move_to_end(key)
        if len(self._prob_cache) > self._prob_cache_limit:
            self._prob_cache.popitem(last=False)

    def _encode_category(self, feature: str, value: Any) -> int:
        mapping = self._categorical_maps.setdefault(feature, {})
        key = str(value) if value is not None else UNKNOWN_CATEGORY
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    def _resolve_group_key(self, feature_context: Mapping[str, Any]) -> str:
        if not self._group_features:
            return ""
        values: List[str] = []
        for name in self._group_features:
            value = feature_context.get(name)
            if name == "spec_tokens":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = self.spec_tokens
            if value is None:
                value = UNKNOWN_CATEGORY
            values.append(str(value))
        return "||".join(values)

    def _select_regressor(self, feature_context: Optional[Mapping[str, Any]]) -> Any:
        if self.algorithm != "moe_gbdt":
            return self._regressor
        context = feature_context or {}
        key = self._resolve_group_key(context)
        expert = self._experts.get(key)
        if expert and expert.get("regressor") is not None:
            return expert["regressor"]
        return self._regressor

    def _select_classifier(self, feature_context: Optional[Mapping[str, Any]]) -> Any:
        if self.algorithm != "moe_gbdt":
            return self._classifier
        context = feature_context or {}
        key = self._resolve_group_key(context)
        expert = self._experts.get(key)
        if expert and expert.get("classifier") is not None:
            return expert["classifier"]
        return self._classifier
