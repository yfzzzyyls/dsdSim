"""Synthetic trace generation utilities."""

from __future__ import annotations

import gzip
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

from .types import TraceRecord

try:
    from typing import Literal
except ImportError:  # pragma: no cover - python<3.8 fallback
    from typing_extensions import Literal  # type: ignore


@dataclass(frozen=True)
class LengthDistribution:
    kind: Literal["fixed", "uniform", "normal"] = "fixed"
    fixed: Optional[int] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            value = self.fixed if self.fixed is not None else self.minimum
            if value is None:
                raise ValueError("fixed distribution requires 'fixed' or 'minimum'")
            return max(1, int(round(value)))
        if self.kind == "uniform":
            if self.minimum is None or self.maximum is None:
                raise ValueError("uniform distribution requires 'minimum' and 'maximum'")
            low = int(self.minimum)
            high = int(self.maximum)
            if low > high:
                raise ValueError("uniform distribution minimum must be <= maximum")
            return max(1, rng.randint(low, high))
        if self.kind == "normal":
            sample = rng.gauss(self.mean, self.stddev)  # type: ignore[arg-type]
            if self.minimum is not None:
                sample = max(sample, float(self.minimum))
            if self.maximum is not None:
                sample = min(sample, float(self.maximum))
            return max(1, int(round(sample)))
        raise ValueError(f"unsupported distribution kind: {self.kind}")


@dataclass(frozen=True)
class DeviceClassWeight:
    weight: float
    draft_id: Optional[str] = None
    device_tier: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("device weight must be positive")
        if self.draft_id is None and self.device_tier is None:
            raise ValueError("device weight requires draft_id or device_tier")


@dataclass(frozen=True)
class SyntheticTraceConfig:
    duration_ms: float = 60_000.0
    start_ms: float = 0.0
    arrival_process: Literal["poisson", "deterministic"] = "poisson"
    rate_rps: float = 10.0
    interarrival_ms: float = 100.0
    burst_factor: float = 1.0
    prompt: LengthDistribution = field(default_factory=lambda: LengthDistribution(kind="uniform", minimum=32, maximum=256))
    target: LengthDistribution = field(default_factory=lambda: LengthDistribution(kind="normal", mean=320.0, stddev=80.0, minimum=16, maximum=1024))
    device_mix: Sequence[DeviceClassWeight] = field(default_factory=list)
    default_slo_class: Optional[str] = None
    default_mode_hint: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    max_requests: Optional[int] = None
    assign_request_ids: bool = True
    request_id_prefix: str = "req"
    assign_request_seeds: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.device_mix:
            raise ValueError("device_mix must contain at least one entry")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.arrival_process == "poisson" and self.rate_rps <= 0:
            raise ValueError("rate_rps must be positive for poisson arrivals")
        if self.arrival_process == "deterministic" and self.interarrival_ms <= 0:
            raise ValueError("interarrival_ms must be positive for deterministic arrivals")
        if self.burst_factor <= 0:
            raise ValueError("burst_factor must be positive")


class SyntheticTraceGenerator:
    """Generate trace records based on ``SyntheticTraceConfig``."""

    def __init__(self, config: SyntheticTraceConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._weights = [dc.weight for dc in config.device_mix]
        self._population = list(config.device_mix)

    def iter_records(self) -> Iterator[TraceRecord]:
        cfg = self.config
        end_time = cfg.start_ms + cfg.duration_ms
        now = cfg.start_ms
        count = 0
        while True:
            if cfg.max_requests is not None and count >= cfg.max_requests:
                break
            if count == 0:
                arrival = now
            else:
                arrival = now + max(0.0, self._next_interarrival())
            if arrival > end_time + 1e-9:
                break
            now = arrival

            device = self._sample_device()
            prompt_tokens = cfg.prompt.sample(self._rng)
            target_tokens = cfg.target.sample(self._rng)
            request_seed = self._rng.getrandbits(32) if cfg.assign_request_seeds else None

            metadata: Dict[str, object] = dict(cfg.metadata)
            metadata.setdefault("source", "synthetic")
            metadata.setdefault("generator", "SyntheticTraceGenerator")
            metadata.setdefault("arrival_process", cfg.arrival_process)
            metadata.setdefault("burst_factor", cfg.burst_factor)
            metadata["sequence_id"] = count

            request_id = None
            if cfg.assign_request_ids:
                request_id = f"{cfg.request_id_prefix}_{count:06d}"

            record = TraceRecord(
                arrival_ms=arrival,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                draft_id=device.draft_id,
                device_tier=device.device_tier,
                slo_class=cfg.default_slo_class,
                mode_hint=cfg.default_mode_hint,
                seed=request_seed,
                metadata=metadata,
                request_id=request_id,
            )
            yield record
            count += 1

    def write_jsonl(self, path: Union[str, Path]) -> None:
        records = (record.to_dict() for record in self.iter_records())
        output_path = Path(path)
        if output_path.suffix == ".gz":
            opener = lambda p: gzip.open(p, "wt", encoding="utf-8")
        else:
            opener = lambda p: p.open("w", encoding="utf-8")
        with opener(output_path) as fh:  # type: ignore[misc]
            for record in records:
                fh.write(json.dumps(record, separators=(",", ":")))
                fh.write("\n")

    def _next_interarrival(self) -> float:
        cfg = self.config
        if cfg.arrival_process == "poisson":
            rate_per_ms = cfg.rate_rps / 1000.0
            base = self._rng.expovariate(rate_per_ms)
        else:
            base = cfg.interarrival_ms
        if cfg.burst_factor > 1:
            scale = self._rng.uniform(1.0 / cfg.burst_factor, 1.0)
            base *= scale
        return base

    def _sample_device(self) -> DeviceClassWeight:
        return self._rng.choices(self._population, weights=self._weights, k=1)[0]


def build_device_mix_from_specs(
    draft_specs: Iterable[Mapping[str, object]],
    *,
    weight_key: Optional[str] = "capability",
    default_weight: float = 1.0,
) -> list[DeviceClassWeight]:
    mix: list[DeviceClassWeight] = []
    for spec in draft_specs:
        if spec.get("role") not in ("draft", "drafter"):
            continue
        draft_id = spec.get("id")
        if not isinstance(draft_id, str):
            raise ValueError("draft device spec must include string 'id'")

        weight = default_weight
        if weight_key:
            raw = spec.get(weight_key, default_weight)
            try:
                weight = float(raw) if raw is not None else default_weight
            except (TypeError, ValueError):
                weight = default_weight
        if not math.isfinite(weight) or weight <= 0:
            weight = default_weight

        mix.append(DeviceClassWeight(weight=weight, draft_id=draft_id))

    if not mix:
        raise ValueError("no draft devices found while building device mix")
    return mix


__all__ = [
    "LengthDistribution",
    "DeviceClassWeight",
    "SyntheticTraceConfig",
    "SyntheticTraceGenerator",
    "build_device_mix_from_specs",
]
