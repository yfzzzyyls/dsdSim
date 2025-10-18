"""Fallback provider using static per-token estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Mapping

from .base import PhaseMetrics, PhaseRequest, PerformanceProvider


@dataclass
class _TargetProfile:
    model: Optional[str]
    hardware: Optional[str]
    metadata: Mapping[str, object]


class DefaultPerformanceProvider(PerformanceProvider):
    """Uses configuration-supplied per-token latencies."""

    def __init__(self) -> None:
        self._targets: Dict[str, _TargetProfile] = {}

    def register_target(
        self,
        *,
        target_id: str,
        model: str,
        hardware: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._targets[target_id] = _TargetProfile(
            model=model or None,
            hardware=hardware or None,
            metadata=dict(metadata or {}),
        )

    def get_metrics(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        profile = None
        if request.target_id and request.target_id in self._targets:
            profile = self._targets[request.target_id]
        elif request.hardware:
            profile = next((p for p in self._targets.values() if p.hardware == request.hardware), None)
        if profile is None:
            return None

        prefill_ms = 0.0
        decode_ms = 0.0
        meta = profile.metadata or {}
        if isinstance(meta, Mapping):
            prefill_ms = float(meta.get("prefill_latency_per_token", 0.0) or 0.0)
            decode_ms = float(meta.get("decode_latency_per_token", 0.0) or 0.0)
        if request.phase == "prefill":
            if prefill_ms <= 0.0:
                return None
            latency = prefill_ms * max(1, request.sequence_length)
        else:
            if decode_ms <= 0.0:
                return None
            tokens = request.tokens_to_generate or request.sequence_length or 1
            latency = decode_ms * max(1, tokens)
        return PhaseMetrics(latency_ms=max(0.0, latency))

    def flush(self) -> None:  # pragma: no cover - nothing to persist
        return
