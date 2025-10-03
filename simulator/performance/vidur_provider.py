"""VIDUR-backed performance provider."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .base import PhaseMetrics, PhaseRequest, PerformanceProvider

try:  # pragma: no cover - optional dependency
    import vidur  # type: ignore
except Exception:  # pragma: no cover
    vidur = None


@dataclass
class VidurProviderConfig:
    binary: Optional[str] = None
    cache_path: Optional[Path] = None
    default_dtype: str = "fp16"


class VidurPerformanceProvider(PerformanceProvider):
    def __init__(self, config: VidurProviderConfig) -> None:
        self._config = config
        self._cache: Dict[Tuple, PhaseMetrics] = {}
        if config.cache_path:
            config.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def register_target(self, *, target_id: str, model: str, hardware: str,
                        prefill_per_token_ms: float, decode_per_token_ms: float) -> None:
        # No-op; VIDUR measures on demand.
        return

    def get_metrics(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        key = (
            request.phase,
            request.model,
            request.hardware,
            request.batch_size,
            request.microbatch_size,
            request.sequence_length,
            request.tokens_to_generate,
            request.context_length,
            request.fanout,
        )
        if key in self._cache:
            return self._cache[key]

        metrics = self._invoke_vidur(request)
        if metrics is not None:
            self._cache[key] = metrics
        return metrics

    def flush(self) -> None:
        if not self._config.cache_path or not self._cache:
            return
        serialisable = {
            "config": {
                "binary": self._config.binary,
                "default_dtype": self._config.default_dtype,
            },
            "entries": [
                {
                    "key": list(key),
                    "latency_ms": metrics.latency_ms,
                    "latency_p95_ms": metrics.latency_p95_ms,
                    "energy_mj": metrics.energy_mj,
                    "kv_delta_mb": metrics.kv_delta_mb,
                }
                for key, metrics in self._cache.items()
            ],
        }
        self._config.cache_path.write_text(json.dumps(serialisable, indent=2))

    # ------------------------------------------------------------------

    def _invoke_vidur(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        if vidur is not None:
            return self._call_vidur_python(request)
        if self._config.binary:
            return self._call_vidur_cli(request)
        raise RuntimeError(
            "VIDUR provider requires either the `vidur` Python package or a CLI binary."
        )

    def _call_vidur_python(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        if request.phase == "prefill":
            result = vidur.prefill_latency(
                model=request.model,
                gpu=request.hardware,
                batch_size=request.batch_size,
                sequence_length=request.sequence_length,
                dtype=self._config.default_dtype,
            )
        else:
            result = vidur.decode_latency(
                model=request.model,
                gpu=request.hardware,
                batch_size=request.batch_size,
                tokens_to_generate=request.tokens_to_generate,
                kv_cache_length=request.context_length,
                mode=request.phase,
                dtype=self._config.default_dtype,
            )
        return self._normalise_result(result)

    def _call_vidur_cli(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        payload = {
            "phase": request.phase,
            "model": request.model,
            "gpu": request.hardware,
            "batch_size": request.batch_size,
            "sequence_length": request.sequence_length,
            "tokens_to_generate": request.tokens_to_generate,
            "kv_cache_length": request.context_length,
            "fanout": request.fanout,
            "dtype": self._config.default_dtype,
        }
        cmd = [self._config.binary, "--json"]  # type: ignore[list-item]
        proc = subprocess.run(cmd, input=json.dumps(payload), text=True, capture_output=True, check=True)
        data = json.loads(proc.stdout)
        return self._normalise_result(data)

    @staticmethod
    def _normalise_result(data) -> Optional[PhaseMetrics]:
        if data is None:
            return None
        latency_ms = float(data.get("latency_ms", 0.0))
        latency_p95 = data.get("latency_p95_ms")
        energy = data.get("energy_mj")
        kv_delta = data.get("kv_delta_mb")
        return PhaseMetrics(
            latency_ms=latency_ms,
            latency_p95_ms=float(latency_p95) if latency_p95 is not None else None,
            energy_mj=float(energy) if energy is not None else None,
            kv_delta_mb=float(kv_delta) if kv_delta is not None else None,
        )

