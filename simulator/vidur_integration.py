"""VIDUR latency oracle integration helpers."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

try:  # Optional dependency for Parquet support
    import pyarrow.parquet as _pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _pq = None

# ---------------------------------------------------------------------------
# Binning thresholds (aligned with design doc recommendations)
# ---------------------------------------------------------------------------
PROMPT_BINS: Tuple[int, ...] = (1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096)
CONTEXT_BINS: Tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
SPEC_BINS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
BATCH_BINS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"  # Alias for decode in some traces


@dataclass(frozen=True)
class ModelConfig:
    """Static model configuration."""

    model_type: str
    gpu_sku: str
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    max_batch_size: int = 32
    max_seq_length: int = 4096
    is_draft: bool = False
    speculation_length: int = 4


@dataclass(frozen=True)
class DeviceConfig:
    """Physical device configuration (subset used by oracle)."""

    device_id: str
    model_config: ModelConfig
    network_rtt_ms: float = 0.0
    network_bandwidth_gbps: float = 0.0


@dataclass(frozen=True)
class Request:
    """Represents a single speculative decoding request within a batch."""

    phase: Phase
    prompt_tokens: int = 0
    speculation_tokens: int = 0
    context_tokens: int = 0
    tokens_to_generate: int = 0
    fanout: int = 1
    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.speculation_tokens < 0 or self.context_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.tokens_to_generate < 0:
            raise ValueError("tokens_to_generate must be non-negative")


@dataclass
class BatchFeatures:
    """Aggregated batch statistics passed to Vidur oracle."""

    requests: Sequence[Request]
    phase: Optional[Phase] = None

    # Populated via compute_statistics()
    batch_size: int = 0
    prompt_tokens: Tuple[int, ...] = field(default_factory=tuple)
    context_tokens: Tuple[int, ...] = field(default_factory=tuple)
    speculation_tokens: Tuple[int, ...] = field(default_factory=tuple)
    avg_prompt_length: float = 0.0
    avg_context_length: float = 0.0
    avg_speculation_length: float = 0.0
    max_prompt_length: int = 0
    max_context_length: int = 0
    max_speculation_length: int = 0
    total_prompt_tokens: int = 0
    total_speculation_tokens: int = 0
    prompt_token_bin: int = 0
    context_token_bin: int = 0
    speculation_token_bin: int = 0
    batch_bin: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("BatchFeatures requires at least one request")
        self.compute_statistics()

    def compute_statistics(self) -> None:
        phases = {req.phase for req in self.requests}
        if self.phase is None:
            if len(phases) != 1:
                raise ValueError("BatchFeatures cannot infer phase from mixed requests")
            self.phase = phases.pop()
        elif len(phases) and self.phase not in phases:
            raise ValueError("BatchFeatures phase mismatch with requests")

        self.batch_size = len(self.requests)
        prompt = tuple(int(req.prompt_tokens) for req in self.requests)
        context = tuple(int(req.context_tokens) for req in self.requests)
        speculation = tuple(int(req.speculation_tokens or req.tokens_to_generate) for req in self.requests)

        self.prompt_tokens = prompt
        self.context_tokens = context
        self.speculation_tokens = speculation

        self.total_prompt_tokens = sum(prompt)
        self.total_speculation_tokens = sum(speculation)

        self.avg_prompt_length = (self.total_prompt_tokens / self.batch_size) if self.batch_size else 0.0
        self.avg_context_length = (sum(context) / self.batch_size) if self.batch_size else 0.0
        self.avg_speculation_length = (self.total_speculation_tokens / self.batch_size) if self.batch_size else 0.0

        self.max_prompt_length = max(prompt) if prompt else 0
        self.max_context_length = max(context) if context else 0
        self.max_speculation_length = max(speculation) if speculation else 0

        phase = self.phase or Phase.PREFILL
        prompt_metric = self.max_prompt_length if phase == Phase.PREFILL else 0
        spec_metric = self.max_speculation_length if phase != Phase.PREFILL else 0
        context_metric = self.max_context_length

        self.prompt_token_bin = _bin_value(prompt_metric, PROMPT_BINS)
        self.context_token_bin = _bin_value(max(1, context_metric), CONTEXT_BINS)
        self.speculation_token_bin = _bin_value(max(1, spec_metric), SPEC_BINS)
        self.batch_bin = _bin_value(self.batch_size, BATCH_BINS)

        if phase == Phase.PREFILL:
            self.total_tokens = self.total_prompt_tokens
        else:
            self.total_tokens = self.total_speculation_tokens or (self.batch_size * max(1, self.max_speculation_length))


@dataclass
class IterationMetrics:
    """Result returned by Vidur oracle."""

    total_latency_ms: float
    tokens_processed: int = 0
    tokens_per_second_iter: float = 0.0
    latency_p95_ms: Optional[float] = None
    energy_mj: Optional[float] = None
    gpu_utilization: Optional[float] = None
    bottleneck: Optional[str] = None
    source: str = "table"
    metadata: Dict[str, float] = field(default_factory=dict)


class ILatencyOracle(Protocol):
    """Protocol definition for Vidur latency oracle."""

    def predict_iteration(
        self,
        model_config: ModelConfig,
        batch_features: BatchFeatures,
        device_config: Optional[DeviceConfig] = None,
    ) -> IterationMetrics:
        ...

    def predict_sequence(
        self,
        model_config: ModelConfig,
        requests: Sequence[Request],
        scheduling_policy: str = "fcfs",
    ) -> Sequence[IterationMetrics]:
        ...


@dataclass
class LatencyEntry:
    """Single latency observation from Vidur traces or LUT."""

    phase: Phase
    model_type: str
    gpu_sku: str
    tensor_parallel: int
    pipeline_parallel: int
    batch_size: int
    prompt_bin: int
    context_bin: int
    spec_bin: int
    latency_ms: float
    latency_p95_ms: Optional[float] = None
    energy_mj: Optional[float] = None
    source: str = "table"

    def matches(self, model: ModelConfig, phase: Phase) -> bool:
        return (
            self.phase == phase
            and self.model_type == model.model_type
            and self.gpu_sku == model.gpu_sku
            and self.tensor_parallel == model.tensor_parallel
            and self.pipeline_parallel == model.pipeline_parallel
        )

    def distance(self, features: BatchFeatures) -> float:
        return (
            abs(self.batch_size - features.batch_size)
            + abs(self.prompt_bin - features.prompt_token_bin)
            + abs(self.context_bin - features.context_token_bin)
            + abs(self.spec_bin - features.speculation_token_bin)
        )

    def key(self) -> Tuple:
        return (
            self.phase,
            self.model_type,
            self.gpu_sku,
            self.tensor_parallel,
            self.pipeline_parallel,
            self.batch_size,
            self.prompt_bin,
            self.context_bin,
            self.spec_bin,
        )

    @classmethod
    def interpolated(cls, neighbors: Sequence[LatencyEntry], features: BatchFeatures) -> Optional[LatencyEntry]:
        if not neighbors:
            return None
        weight_sum = 0.0
        latency_sum = 0.0
        latency_p95_sum = 0.0
        latency_p95_weight = 0.0
        energy_sum = 0.0
        energy_weight = 0.0
        for entry in neighbors:
            dist = entry.distance(features)
            weight = 1.0 / (1.0 + dist)
            weight_sum += weight
            latency_sum += weight * entry.latency_ms
            if entry.latency_p95_ms is not None:
                latency_p95_sum += weight * entry.latency_p95_ms
                latency_p95_weight += weight
            if entry.energy_mj is not None:
                energy_sum += weight * entry.energy_mj
                energy_weight += weight
        if weight_sum == 0.0:
            return None
        reference = neighbors[0]
        return cls(
            phase=features.phase or reference.phase,
            model_type=reference.model_type,
            gpu_sku=reference.gpu_sku,
            tensor_parallel=reference.tensor_parallel,
            pipeline_parallel=reference.pipeline_parallel,
            batch_size=features.batch_size,
            prompt_bin=features.prompt_token_bin,
            context_bin=features.context_token_bin,
            spec_bin=features.speculation_token_bin,
            latency_ms=latency_sum / weight_sum,
            latency_p95_ms=(latency_p95_sum / latency_p95_weight) if latency_p95_weight else None,
            energy_mj=(energy_sum / energy_weight) if energy_weight else None,
            source="interpolated",
        )


class VidurLatencyTable:
    """In-memory lookup table backed by Vidur profiling data."""

    def __init__(self, entries: Optional[Iterable[LatencyEntry]] = None) -> None:
        self._entries: Dict[Tuple, LatencyEntry] = {}
        if entries:
            for entry in entries:
                self.add_entry(entry)

    def add_entry(self, entry: LatencyEntry) -> None:
        self._entries[entry.key()] = entry

    def load_file(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vidur latency table not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text())
            records = data.get("entries") if isinstance(data, dict) else data
            for item in _iter_records(records):
                self.add_entry(_entry_from_mapping(item))
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    self.add_entry(_entry_from_mapping(json.loads(line)))
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    self.add_entry(_entry_from_mapping(row))
        elif suffix in {".parquet", ".pq"}:
            if _pq is None:
                raise RuntimeError("pyarrow is required to load Parquet latency tables")
            table = _pq.read_table(path)  # type: ignore[operator]
            for record in table.to_pylist():
                self.add_entry(_entry_from_mapping(record))
        else:
            raise ValueError(f"Unsupported latency table format: {suffix}")

    def lookup(
        self,
        model_config: ModelConfig,
        features: BatchFeatures,
        *,
        neighbors: int = 3,
        prefer_exact: bool = True,
    ) -> Optional[LatencyEntry]:
        key = (
            features.phase,
            model_config.model_type,
            model_config.gpu_sku,
            model_config.tensor_parallel,
            model_config.pipeline_parallel,
            features.batch_size,
            features.prompt_token_bin,
            features.context_token_bin,
            features.speculation_token_bin,
        )
        entry = self._entries.get(key)
        if entry is not None:
            return entry
        if prefer_exact:
            return None
        candidates = [
            candidate
            for candidate in self._entries.values()
            if candidate.matches(model_config, features.phase or Phase.PREFILL)
        ]
        if not candidates:
            return None
        sorted_candidates = sorted(candidates, key=lambda e: e.distance(features))
        chosen = sorted_candidates[: max(1, neighbors)]
        return LatencyEntry.interpolated(chosen, features)


# ---------------------------------------------------------------------------
# Oracle implementations
# ---------------------------------------------------------------------------


DEFAULT_LATENCY_ENTRIES: Tuple[LatencyEntry, ...] = (
    LatencyEntry(Phase.PREFILL, "llama-70b", "H100", 1, 1, 1, 512, 512, 1, 97.2),
    LatencyEntry(Phase.PREFILL, "llama-70b", "H100", 1, 1, 4, 512, 512, 1, 287.3),
    LatencyEntry(Phase.DECODE, "llama-70b", "H100", 1, 1, 1, 1, 1024, 4, 6.2),
    LatencyEntry(Phase.DECODE, "llama-70b", "H100", 1, 1, 8, 1, 1024, 4, 31.5),
    LatencyEntry(Phase.PREFILL, "llama-13b", "A100", 1, 1, 1, 256, 256, 1, 45.3),
    LatencyEntry(Phase.DECODE, "llama-13b", "A100", 1, 1, 8, 1, 512, 4, 18.9),
)


class VidurOracle(ILatencyOracle):
    """Lookup/interpolation oracle backed by Vidur latency tables."""

    def __init__(
        self,
        *,
        table_path: Optional[str] = None,
        default_latency_ms: float = 50.0,
        neighbors: int = 3,
        prefer_exact: bool = True,
        bootstrap_defaults: bool = True,
    ) -> None:
        self._table = VidurLatencyTable(DEFAULT_LATENCY_ENTRIES if bootstrap_defaults else None)
        if table_path:
            self._table.load_file(Path(table_path))
        self._default_latency_ms = default_latency_ms
        self._neighbors = max(1, neighbors)
        self._prefer_exact = prefer_exact
        self._cache: Dict[Tuple, IterationMetrics] = {}

    def predict_iteration(
        self,
        model_config: ModelConfig,
        batch_features: BatchFeatures,
        device_config: Optional[DeviceConfig] = None,
    ) -> IterationMetrics:
        cache_key = self._cache_key(model_config, batch_features)
        if cache_key in self._cache:
            return self._cache[cache_key]

        entry = self._table.lookup(
            model_config,
            batch_features,
            neighbors=self._neighbors,
            prefer_exact=self._prefer_exact,
        )
        if entry is None:
            latency = self._fallback_latency(model_config, batch_features, device_config)
            metrics = IterationMetrics(
                total_latency_ms=latency,
                tokens_processed=batch_features.total_tokens,
                tokens_per_second_iter=_safe_throughput(batch_features.total_tokens, latency),
                source="fallback",
            )
        else:
            latency = entry.latency_ms
            metrics = IterationMetrics(
                total_latency_ms=latency,
                tokens_processed=batch_features.total_tokens,
                tokens_per_second_iter=_safe_throughput(batch_features.total_tokens, latency),
                latency_p95_ms=entry.latency_p95_ms,
                energy_mj=entry.energy_mj,
                source=entry.source,
            )
        self._cache[cache_key] = metrics
        return metrics

    def predict_sequence(
        self,
        model_config: ModelConfig,
        requests: Sequence[Request],
        scheduling_policy: str = "fcfs",
    ) -> Sequence[IterationMetrics]:
        if scheduling_policy not in {"fcfs", "fifo"}:
            raise NotImplementedError(f"Unsupported scheduling policy: {scheduling_policy}")
        if not requests:
            return []
        batches: List[IterationMetrics] = []
        current: List[Request] = []
        max_batch = max(1, model_config.max_batch_size)
        for request in requests:
            current.append(request)
            if len(current) >= max_batch:
                features = BatchFeatures(tuple(current))
                batches.append(self.predict_iteration(model_config, features))
                current.clear()
        if current:
            features = BatchFeatures(tuple(current))
            batches.append(self.predict_iteration(model_config, features))
        return batches

    def _cache_key(self, model_config: ModelConfig, batch: BatchFeatures) -> Tuple:
        return (
            batch.phase,
            model_config.model_type,
            model_config.gpu_sku,
            model_config.tensor_parallel,
            model_config.pipeline_parallel,
            batch.batch_size,
            batch.prompt_token_bin,
            batch.context_token_bin,
            batch.speculation_token_bin,
        )

    def _fallback_latency(
        self,
        model_config: ModelConfig,
        batch: BatchFeatures,
        device_config: Optional[DeviceConfig],
    ) -> float:
        base = self._default_latency_ms
        batch_scale = math.log2(batch.batch_size + 1)
        context_scale = math.log1p(batch.context_token_bin)
        spec_scale = math.log1p(batch.speculation_token_bin if batch.phase != Phase.PREFILL else batch.prompt_token_bin)
        parallel_factor = max(model_config.tensor_parallel * model_config.pipeline_parallel, 1)
        latency = base * (1 + 0.12 * batch_scale) * (1 + 0.05 * spec_scale) * (1 + 0.03 * context_scale)
        latency /= math.sqrt(parallel_factor)
        if device_config and device_config.network_rtt_ms:
            latency += device_config.network_rtt_ms * 0.1
        return max(latency, 1.0)


class MockVidurOracle(ILatencyOracle):
    """Deterministic oracle useful for tests/debug."""

    def __init__(self, prefill_base: float = 80.0, decode_base: float = 5.0) -> None:
        self.prefill_base = prefill_base
        self.decode_base = decode_base

    def predict_iteration(
        self,
        model_config: ModelConfig,
        batch_features: BatchFeatures,
        device_config: Optional[DeviceConfig] = None,
    ) -> IterationMetrics:
        if batch_features.phase == Phase.PREFILL:
            latency = self.prefill_base * math.log1p(batch_features.prompt_token_bin) * (1 + 0.02 * batch_features.batch_size)
        else:
            latency = self.decode_base * (1 + 0.05 * batch_features.context_token_bin) * (1 + 0.1 * batch_features.batch_size)
        latency = max(latency, 1.0)
        tokens = batch_features.total_tokens
        return IterationMetrics(
            total_latency_ms=latency,
            tokens_processed=tokens,
            tokens_per_second_iter=_safe_throughput(tokens, latency),
            source="mock",
        )

    def predict_sequence(
        self,
        model_config: ModelConfig,
        requests: Sequence[Request],
        scheduling_policy: str = "fcfs",
    ) -> Sequence[IterationMetrics]:
        features = BatchFeatures(tuple(requests))
        return [self.predict_iteration(model_config, features)]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _iter_records(container: object) -> Iterable[Mapping[str, object]]:
    if container is None:
        return []
    if isinstance(container, Mapping):
        return container.values()  # type: ignore[return-value]
    if isinstance(container, (list, tuple)):
        return container  # type: ignore[return-value]
    raise TypeError("Unsupported record container type")


def _entry_from_mapping(raw: Mapping[str, object]) -> LatencyEntry:
    phase = Phase(str(raw.get("phase") or raw.get("phase_name") or raw.get("phase_type") or "prefill").lower())
    model = str(raw.get("model") or raw.get("model_name") or raw.get("model_type") or "llama-70b").strip()
    gpu = str(raw.get("gpu") or raw.get("gpu_sku") or raw.get("hardware") or "H100").strip()
    tp = int(raw.get("tensor_parallel") or raw.get("tp") or raw.get("tp_degree") or 1)
    pp = int(raw.get("pipeline_parallel") or raw.get("pp") or raw.get("pp_degree") or 1)
    batch = int(raw.get("batch_size") or raw.get("batch") or 1)
    prompt_bin = _bin_value(int(raw.get("prompt_bin") or raw.get("tokens_per_seq_bin") or raw.get("prompt_tokens") or 1), PROMPT_BINS)
    context_bin = _bin_value(int(raw.get("context_bin") or raw.get("context_len_bin") or raw.get("context_tokens") or raw.get("kv_cache_length") or 128), CONTEXT_BINS)
    spec_bin = _bin_value(int(raw.get("spec_bin") or raw.get("spec_tokens_bin") or raw.get("spec_tokens") or raw.get("tokens_per_decode") or 1), SPEC_BINS)
    latency = float(raw.get("latency_ms") or raw.get("latency") or 0.0)
    latency_p95 = raw.get("latency_p95_ms")
    energy = raw.get("energy_mj") or raw.get("energy")
    source = str(raw.get("source") or "table")
    return LatencyEntry(
        phase=phase,
        model_type=model,
        gpu_sku=gpu,
        tensor_parallel=tp,
        pipeline_parallel=pp,
        batch_size=batch,
        prompt_bin=prompt_bin,
        context_bin=context_bin,
        spec_bin=spec_bin,
        latency_ms=latency,
        latency_p95_ms=float(latency_p95) if latency_p95 is not None else None,
        energy_mj=float(energy) if energy is not None else None,
        source=source,
    )


def _safe_throughput(tokens: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return tokens / (latency_ms / 1000.0)


def _bin_value(value: int, bins: Sequence[int]) -> int:
    value = max(1, int(value))
    for threshold in bins:
        if value <= threshold:
            return int(threshold)
    return int(bins[-1])


__all__ = [
    "Phase",
    "ModelConfig",
    "DeviceConfig",
    "Request",
    "BatchFeatures",
    "IterationMetrics",
    "ILatencyOracle",
    "LatencyEntry",
    "VidurLatencyTable",
    "VidurOracle",
    "MockVidurOracle",
]
