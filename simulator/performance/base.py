"""Base protocol and data containers for performance providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class PhaseRequest:
    """Describes a model execution request for profiling."""

    phase: str  # prefill | draft_decode | target_decode | verify | decode
    model: Optional[str]
    hardware: Optional[str]
    batch_size: int = 1
    microbatch_size: int = 1
    fanout: int = 1
    sequence_length: int = 0
    tokens_to_generate: int = 0
    context_length: int = 0
    request_id: Optional[str] = None
    target_id: Optional[str] = None
    draft_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    context_tokens: Optional[int] = None
    tokens_per_request: Optional[Sequence[int]] = None
    context_per_request: Optional[Sequence[int]] = None
    extra_metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class PhaseMetrics:
    latency_ms: float
    latency_p95_ms: Optional[float] = None
    energy_mj: Optional[float] = None
    kv_delta_mb: Optional[float] = None


class PerformanceProvider(Protocol):
    """Abstract provider interface for iterative performance modelling."""

    def register_target(
        self,
        *,
        target_id: str,
        model: str,
        hardware: str,
        prefill_per_token_ms: float,
        decode_per_token_ms: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Optional hook allowing providers to learn about targets."""

    def get_metrics(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        """Return metrics for the given request or ``None`` if unavailable."""

    def flush(self) -> None:
        """Persist any on-disk caches if the provider maintains them."""

