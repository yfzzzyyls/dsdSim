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
        surrogate_config: Optional[Mapping[str, Any]] = None,
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

        # Feature lists used by the surrogate / predictors
        self._count_features = self._feature_columns.get("count") or ["context_length"]
        self._accept_features = self._feature_columns.get("accept") or [
            "context_length",
            "position",
        ]

        # Lightweight surrogate (pre-baked acceptance surface)
        self._surrogate: Optional[Dict[str, Any]] = None
        if surrogate_config and surrogate_config.get("enabled"):
            self._build_surrogate(surrogate_config)

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
    @classmethod
    def from_file(cls, path: str | Path, *, surrogate_config: Optional[Mapping[str, Any]] = None) -> "AcceptanceRegressor":
        bundle = joblib.load(Path(path))
        spec_tokens = int(bundle.get("spec_tokens", 1))
        regressor = bundle.get("regressor")
        classifier = bundle.get("classifier")
        if classifier is None:
            raise ValueError("Acceptance bundle missing classifier")
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
            surrogate_config=surrogate_config,
        )

    def expected_accepts(
        self,
        context_length: float,
        *,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> float:
        value, _ = self._predict_expected_accepts(
            context_length,
            feature_context=feature_context,
        )
        return value

    def predict_expected_accepts(
        self,
        context_length: float,
        *,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, bool]:
        return self._predict_expected_accepts(
            context_length,
            feature_context=feature_context,
        )

    def _predict_expected_accepts(
        self,
        context_length: float,
        *,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, bool]:
        from src import sim as _sim_module

        profiler = getattr(_sim_module, "_GLOBAL_PROFILER", None)
        start_time = time.perf_counter()
        surrogate_rate = self._surrogate_rate(context_length, feature_context)
        if surrogate_rate is not None:
            depth = self._extract_depth(feature_context)
            value = max(0.0, surrogate_rate * depth)
            if profiler is not None:
                profiler["acceptance_surrogate_ms"] += (time.perf_counter() - start_time) * 1000.0
                profiler["acceptance_surrogate_hits"] = profiler.get("acceptance_surrogate_hits", 0) + 1
            return value, True

        row = np.array(
            [self._make_feature_row(self._count_features, context_length, None, feature_context)],
            dtype=np.float32,
        )
        regressor = self._select_regressor(feature_context)
        prediction = regressor.predict(row)
        return float(max(0.0, prediction[0])), False

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

    def _build_surrogate(self, config: Mapping[str, Any]) -> None:
        """Precompute acceptance rates over a coarse feature grid."""
        from src import sim as _sim_module

        profiler = getattr(_sim_module, "_GLOBAL_PROFILER", None)
        start_time = time.perf_counter()

        def _step(value: float, fallback: float) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = float(fallback)
            return fallback if numeric <= 0 else numeric

        context_step = _step(config.get("context_step", config.get("context_bucket", 32.0)), 32.0)
        context_min = float(config.get("context_min", 0.0))
        context_max = float(config.get("context_max", 4096.0))
        if context_max < context_min:
            context_max = context_min
        contexts = np.arange(context_min, context_max + context_step, context_step, dtype=np.float32)
        if contexts.size == 0:
            contexts = np.array([context_min], dtype=np.float32)

        depth_step = max(1, int(config.get("depth_step", config.get("depth_bucket", 8)) or 8))
        depth_min = int(max(1, config.get("depth_min", depth_step)))
        depth_default_max = int(max(depth_min, self.spec_tokens))
        depth_max = int(config.get("depth_max", depth_default_max))
        if depth_max < depth_min:
            depth_max = depth_min
        depths = np.arange(depth_min, depth_max + depth_step, depth_step, dtype=np.int32)
        if depths.size == 0:
            depths = np.array([depth_min], dtype=np.int32)

        pending_step = max(1, int(config.get("pending_step", config.get("pending_bucket", 64)) or 64))
        pending_min = int(max(0, config.get("pending_min", 0)))
        pending_max = int(max(pending_min, config.get("pending_max", 1024)))
        pending_vals = np.arange(pending_min, pending_max + pending_step, pending_step, dtype=np.int32)
        if pending_vals.size == 0:
            pending_vals = np.array([pending_min], dtype=np.int32)

        queue_step = max(1, int(config.get("queue_step", config.get("queue_bucket", 4)) or 4))
        queue_min = int(max(0, config.get("queue_min", 0)))
        queue_max = int(max(queue_min, config.get("queue_max", 64)))
        queue_vals = np.arange(queue_min, queue_max + queue_step, queue_step, dtype=np.int32)
        if queue_vals.size == 0:
            queue_vals = np.array([queue_min], dtype=np.int32)

        drafter_models = list(config.get("drafter_models") or [])
        verifier_models = list(config.get("verifier_models") or [])
        if not drafter_models:
            drafter_models = ["surrogate_draft"]
        if not verifier_models:
            verifier_models = ["surrogate_target"]
        base_context_overrides = dict(config.get("feature_context", {}) or {})

        total_combos = (
            len(contexts)
            * len(depths)
            * len(pending_vals)
            * len(queue_vals)
            * len(drafter_models)
            * len(verifier_models)
        )
        max_combos = int(config.get("max_combos", 2_000_000) or 2_000_000)
        if total_combos > max_combos:
            if profiler is not None:
                profiler["acceptance_surrogate_build_ms"] += (time.perf_counter() - start_time) * 1000.0
            return

        feature_rows: List[List[float]] = []
        feature_contexts: List[Dict[str, Any]] = []
        keys: List[Tuple[int, int, int, int, int, int]] = []
        lookup: Dict[Tuple[int, int, int, int, int, int], float] = {}

        # Precompute rows grouped by regressor instance (to support MoE)
        reg_groups: Dict[int, List[int]] = {}
        reg_map: Dict[int, Any] = {}

        for drafter_idx, drafter_model in enumerate(drafter_models):
            for verifier_idx, verifier_model in enumerate(verifier_models):
                for ctx_idx, context in enumerate(contexts):
                    for depth_idx, depth in enumerate(depths):
                        spec_tokens = int(max(1, depth))
                        for pend_idx, pending in enumerate(pending_vals):
                            pending_tokens = int(max(0, pending))
                            for queue_idx, queue in enumerate(queue_vals):
                                queue_depth = int(max(0, queue))
                                fc = dict(base_context_overrides)
                                fc.update(
                                    {
                                        "spec_tokens": spec_tokens,
                                        "pending_decode_tokens": pending_tokens,
                                        "target_queue_depth": queue_depth,
                                        "drafter_model": drafter_model,
                                        "verifier_model": verifier_model,
                                    }
                                )
                                feature_contexts.append(fc)
                                feature_rows.append(
                                    self._make_feature_row(self._count_features, float(context), None, fc)
                                )
                                global_idx = len(feature_rows) - 1
                                keys.append(
                                    (
                                        drafter_idx,
                                        verifier_idx,
                                        ctx_idx,
                                        depth_idx,
                                        pend_idx,
                                        queue_idx,
                                    )
                                )
                                reg = self._select_regressor(fc)
                                reg_key = id(reg)
                                reg_map[reg_key] = reg
                                reg_groups.setdefault(reg_key, []).append(global_idx)

        if not feature_rows:
            return

        rows_array = np.asarray(feature_rows, dtype=np.float32)
        rates_flat = np.zeros(len(feature_rows), dtype=np.float32)

        for reg_key, indices in reg_groups.items():
            reg = reg_map[reg_key]
            batch_rows = rows_array[indices]
            preds = reg.predict(batch_rows)
            for row_idx, prediction in zip(indices, preds):
                fc = feature_contexts[row_idx]
                depth_val = float(max(1, fc.get("spec_tokens", self.spec_tokens)))
                rate = float(prediction) / depth_val
                rates_flat[row_idx] = max(0.0, min(1.0, rate))

        for key, rate in zip(keys, rates_flat):
            lookup[key] = float(rate)

        self._surrogate = {
            "contexts": contexts,
            "depths": depths,
            "pending": pending_vals,
            "queues": queue_vals,
            "context_step": float(context_step),
            "depth_step": int(depth_step),
            "pending_step": int(pending_step),
            "queue_step": int(queue_step),
            "context_min": float(contexts[0]),
            "depth_min": int(depths[0]),
            "pending_min": int(pending_vals[0]),
            "queue_min": int(queue_vals[0]),
            "drafter_models": list(drafter_models),
            "verifier_models": list(verifier_models),
            "lookup": lookup,
        }

        if profiler is not None:
            profiler["acceptance_surrogate_build_ms"] += (time.perf_counter() - start_time) * 1000.0
        if self._surrogate is not None:
            # Track that the surrogate is active in metrics cache
            self._surrogate["enabled"] = True

    def _surrogate_rate(
        self,
        context_length: float,
        feature_context: Optional[Mapping[str, Any]],
    ) -> Optional[float]:
        surrogate = self._surrogate
        if not surrogate:
            return None

        fc = dict(feature_context or {})
        drafter_models: List[str] = surrogate.get("drafter_models", [])
        verifier_models: List[str] = surrogate.get("verifier_models", [])
        if not drafter_models or not verifier_models:
            return None

        drafter = str(fc.get("drafter_model", drafter_models[0]))
        verifier = str(fc.get("verifier_model", verifier_models[0]))
        try:
            drafter_idx = drafter_models.index(drafter)
        except ValueError:
            drafter_idx = 0
        try:
            verifier_idx = verifier_models.index(verifier)
        except ValueError:
            verifier_idx = 0

        depth_value = self._extract_depth(fc)
        pending_value = fc.get("pending_decode_tokens", 0)
        queue_value = fc.get("target_queue_depth", 0)

        ctx_idx = self._surrogate_axis_index(
            float(context_length),
            surrogate["context_min"],
            surrogate["context_step"],
            len(surrogate["contexts"]),
        )
        depth_idx = self._surrogate_axis_index(
            depth_value,
            float(surrogate["depth_min"]),
            float(surrogate["depth_step"]),
            len(surrogate["depths"]),
        )
        pending_idx = self._surrogate_axis_index(
            pending_value,
            float(surrogate["pending_min"]),
            float(surrogate["pending_step"]),
            len(surrogate["pending"]),
        )
        queue_idx = self._surrogate_axis_index(
            queue_value,
            float(surrogate["queue_min"]),
            float(surrogate["queue_step"]),
            len(surrogate["queues"]),
        )

        key = (drafter_idx, verifier_idx, ctx_idx, depth_idx, pending_idx, queue_idx)
        rate = surrogate["lookup"].get(key)
        if rate is None:
            return None
        return float(rate)

    def _surrogate_axis_index(self, value: float, min_value: float, step: float, size: int) -> int:
        if size <= 1:
            return 0
        if step <= 0:
            step = 1.0
        offset = (float(value) - float(min_value)) / float(step)
        idx = int(round(offset))
        if idx < 0:
            idx = 0
        elif idx >= size:
            idx = size - 1
        return idx

    def _extract_depth(self, feature_context: Optional[Mapping[str, Any]]) -> float:
        if feature_context is None:
            return float(self.spec_tokens)
        value = feature_context.get("spec_tokens", self.spec_tokens)
        try:
            depth = float(value)
        except (TypeError, ValueError):
            depth = float(self.spec_tokens)
        return max(1.0, depth)

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
