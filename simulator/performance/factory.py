"""Factory for creating performance providers from config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .base import PerformanceProvider
from .default_provider import DefaultPerformanceProvider
from .vidur_provider import VidurPerformanceProvider, VidurProviderConfig


@dataclass
class PerformanceModelConfig:
    type: str = "default"
    vidur: Dict[str, object] = field(default_factory=dict)


def create_performance_provider(config: PerformanceModelConfig) -> PerformanceProvider:
    provider_type = (config.type or "default").lower()
    if provider_type == "vidur":
        vidur_cfg = VidurProviderConfig(
            binary=_optional_str(config.vidur.get("binary")),
            cache_path=_optional_path(config.vidur.get("cache_path")),
            default_dtype=str(config.vidur.get("dtype", "fp16")),
            table_path=_optional_path(config.vidur.get("table_path")),
            default_latency_ms=_optional_float(config.vidur.get("default_latency_ms"), 50.0),
            neighbors=_optional_int(config.vidur.get("neighbors"), 3),
            prefer_exact=_optional_bool(config.vidur.get("prefer_exact"), True),
            bootstrap_defaults=_optional_bool(config.vidur.get("bootstrap_defaults"), True),
        )
        return VidurPerformanceProvider(vidur_cfg)
    return DefaultPerformanceProvider()


def _optional_str(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_path(value):
    if not value:
        return None
    from pathlib import Path

    return Path(str(value))


def _optional_float(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_int(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _optional_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)

