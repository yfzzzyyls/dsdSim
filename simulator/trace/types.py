"""Shared trace record definitions and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, TypedDict


class TraceParseError(ValueError):
    """Raised when a trace entry fails validation."""


class TraceRecordDict(TypedDict, total=False):
    request_id: str
    arrival_ms: float
    draft_id: str
    device_tier: str
    prompt_tokens: int
    target_tokens: int
    slo_class: str
    mode_hint: str
    seed: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class TraceRecord:
    arrival_ms: float
    prompt_tokens: int
    target_tokens: int
    draft_id: Optional[str] = None
    device_tier: Optional[str] = None
    slo_class: Optional[str] = None
    mode_hint: Optional[str] = None
    seed: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None

    def to_dict(self) -> TraceRecordDict:
        data: TraceRecordDict = {
            "arrival_ms": self.arrival_ms,
            "prompt_tokens": self.prompt_tokens,
            "target_tokens": self.target_tokens,
        }
        if self.request_id is not None:
            data["request_id"] = self.request_id
        if self.draft_id is not None:
            data["draft_id"] = self.draft_id
        if self.device_tier is not None:
            data["device_tier"] = self.device_tier
        if self.slo_class is not None:
            data["slo_class"] = self.slo_class
        if self.mode_hint is not None:
            data["mode_hint"] = self.mode_hint
        if self.seed is not None:
            data["seed"] = self.seed
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceRecord":
        arrival_ms = _coerce_float(payload.get("arrival_ms"), "arrival_ms")
        prompt_tokens = _coerce_positive_int(payload.get("prompt_tokens"), "prompt_tokens")
        target_tokens = _coerce_positive_int(payload.get("target_tokens"), "target_tokens")

        draft_id = _coerce_optional_str(payload.get("draft_id"), "draft_id")
        device_tier = _coerce_optional_str(payload.get("device_tier"), "device_tier")
        slo_class = _coerce_optional_str(payload.get("slo_class"), "slo_class")
        mode_hint = _coerce_optional_str(payload.get("mode_hint"), "mode_hint")
        request_id = _coerce_optional_str(payload.get("request_id"), "request_id")

        seed_field = payload.get("seed")
        seed = None if seed_field is None else _coerce_int(seed_field, "seed", allow_zero=True)

        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TraceParseError("metadata must be a mapping")

        if draft_id is None and device_tier is None:
            raise TraceParseError("trace record must include draft_id or device_tier")
        if arrival_ms < 0:
            raise TraceParseError("arrival_ms must be non-negative")

        return cls(
            arrival_ms=arrival_ms,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            draft_id=draft_id,
            device_tier=device_tier,
            slo_class=slo_class,
            mode_hint=mode_hint,
            seed=seed,
            metadata=dict(metadata),
            request_id=request_id,
        )

    def with_defaults(self, defaults: Mapping[str, Any]) -> "TraceRecord":
        merged = self.to_dict()
        for key, value in defaults.items():
            merged.setdefault(key, value)
        return TraceRecord.from_dict(merged)

    def with_draft(self, draft_id: str) -> "TraceRecord":
        payload = self.to_dict()
        payload["draft_id"] = draft_id
        return TraceRecord.from_dict(payload)


def _coerce_float(value: Any, field_name: str) -> float:
    if value is None:
        raise TraceParseError(f"{field_name} is required")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TraceParseError(f"{field_name} must be a float") from exc


def _coerce_positive_int(value: Any, field_name: str) -> int:
    result = _coerce_int(value, field_name)
    if result <= 0:
        raise TraceParseError(f"{field_name} must be positive")
    return result


def _coerce_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    if value is None:
        raise TraceParseError(f"{field_name} is required")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TraceParseError(f"{field_name} must be an integer") from exc
    if not allow_zero and result == 0:
        raise TraceParseError(f"{field_name} must be non-zero")
    return result


def _coerce_optional_str(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    raise TraceParseError(f"{field_name} must be a string if provided")

