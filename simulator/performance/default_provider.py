"""Fallback provider using static per-token estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Mapping

from .base import PhaseMetrics, PhaseRequest, PerformanceProvider


@dataclass
class _TargetProfile:
    model: Optional[str]
    hardware: Optional[str]
    prefill_per_token_ms: float
    decode_per_token_ms: float


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
        prefill_per_token_ms: float,
        decode_per_token_ms: float,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._targets[target_id] = _TargetProfile(
            model=model or None,
            hardware=hardware or None,
            prefill_per_token_ms=prefill_per_token_ms,
            decode_per_token_ms=decode_per_token_ms,
        )

    def get_metrics(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        profile = None
        if request.target_id and request.target_id in self._targets:
            profile = self._targets[request.target_id]
        elif request.hardware:
            profile = next((p for p in self._targets.values() if p.hardware == request.hardware), None)
        if profile is None:
            return None

        if request.phase == "prefill":
            latency = profile.prefill_per_token_ms * max(1, request.sequence_length)
        else:
            tokens = request.tokens_to_generate or request.sequence_length or 1
            latency = profile.decode_per_token_ms * max(1, tokens)
        return PhaseMetrics(latency_ms=max(0.0, latency))

    def flush(self) -> None:  # pragma: no cover - nothing to persist
        return

