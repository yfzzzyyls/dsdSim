"""VIDUR-backed performance provider."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Mapping

from .base import PhaseMetrics, PhaseRequest, PerformanceProvider
from .vidur_realtime import VidurRealtimeRunner, get_shared_vidur_runner

def _append_vidur_repo_to_path() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    candidate = base_dir / "thirdparty" / "vidur"
    if candidate.exists() and candidate.is_dir():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

_append_vidur_repo_to_path()

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import vidur  # type: ignore
except Exception:  # pragma: no cover
    vidur = None

try:  # pragma: no cover - optional dependency
    from vidur_integration import (
        BatchFeatures as _VidurBatchFeatures,
        ModelConfig as _VidurModelConfig,
        Phase as _VidurPhase,
        Request as _VidurRequest,
        VidurOracle as _VidurOracle,
    )
except Exception:  # pragma: no cover
    _VidurBatchFeatures = None
    _VidurModelConfig = None
    _VidurPhase = None
    _VidurRequest = None
    _VidurOracle = None


@dataclass
class VidurProviderConfig:
    binary: Optional[str] = None
    cache_path: Optional[Path] = None
    default_dtype: str = "fp16"
    table_path: Optional[Path] = None
    default_latency_ms: float = 50.0
    neighbors: int = 3
    prefer_exact: bool = True
    bootstrap_defaults: bool = True
    oracle: Optional[object] = None  # Expected to implement ILatencyOracle
    realtime_enabled: bool = False
    realtime_cache_dir: Optional[Path] = None


class SyntheticVidurOracle:
    """Heuristic fallback when no VIDUR assets are available."""

    def __init__(self, default_latency_ms: float = 50.0) -> None:
        self._default_latency_ms = max(1.0, float(default_latency_ms))

    def predict(self, request: PhaseRequest, target_info: Optional[Dict[str, Any]] = None) -> PhaseMetrics:
        model = _infer_model_name(request, target_info)
        hardware = _infer_hardware_name(request, target_info)
        model_scale = _infer_model_scale(model)
        hw_factor = _hardware_speed_factor(hardware)
        batch_factor = 1.0 + max(0, request.batch_size - 1) * 0.12
        microbatch_factor = 1.0 + max(0, request.microbatch_size - 1) * 0.05
        fanout_factor = 1.0 + max(0, request.fanout - 1) * 0.08

        if request.phase == "prefill":
            tokens = max(
                request.sequence_length,
                request.context_length,
                request.prompt_tokens or 0,
                1,
            )
            per_token_ms = 0.32 * (model_scale / 70.0)
            context_factor = 1.0 + (request.sequence_length / 4096.0) * 0.18
            latency_ms = 8.0 + tokens * per_token_ms * hw_factor * batch_factor * context_factor
            kv_delta = tokens * model_scale * 0.004
        else:
            tokens = max(request.tokens_to_generate, 1)
            per_token_ms = 0.95 * (model_scale / 70.0)
            context_len = max(request.context_length, request.context_tokens or 0, 1)
            context_factor = 1.0 + (context_len / 4096.0) * 0.12
            latency_ms = (
                4.5
                + tokens * per_token_ms * hw_factor * batch_factor * microbatch_factor * fanout_factor * context_factor
            )
            kv_delta = tokens * model_scale * 0.0025

        latency_ms = max(latency_ms, self._default_latency_ms * 0.25)
        jitter = _stable_jitter(
            request.phase,
            model,
            hardware,
            request.batch_size,
            request.sequence_length,
            request.tokens_to_generate,
            request.context_length,
        )
        latency_ms *= jitter
        latency_p95 = latency_ms * 1.18
        energy_mj = tokens * model_scale * hw_factor * 0.0009
        return PhaseMetrics(
            latency_ms=latency_ms,
            latency_p95_ms=latency_p95,
            energy_mj=energy_mj,
            kv_delta_mb=max(kv_delta, 0.0),
        )


def _infer_model_name(request: PhaseRequest, target_info: Optional[Dict[str, Any]]) -> str:
    if request.model:
        return str(request.model)
    if target_info and target_info.get("model"):
        return str(target_info["model"])
    return "generic-13b"


def _infer_hardware_name(request: PhaseRequest, target_info: Optional[Dict[str, Any]]) -> str:
    if request.hardware:
        return str(request.hardware)
    if target_info and target_info.get("hardware"):
        return str(target_info["hardware"])
    return "A100"


def _infer_model_scale(model_name: str) -> float:
    lowered = model_name.lower()
    if "70" in lowered:
        return 70.0
    if "65" in lowered:
        return 65.0
    if "40" in lowered:
        return 40.0
    if "34" in lowered or "33" in lowered:
        return 34.0
    if "30" in lowered:
        return 30.0
    if "13" in lowered or "12" in lowered:
        return 13.0
    if "7" in lowered or "8" in lowered:
        return 7.0
    if "3" in lowered:
        return 3.0
    return 13.0


def _hardware_speed_factor(hardware: str) -> float:
    table = {
        "H100": 0.55,
        "H200": 0.5,
        "A100": 0.9,
        "A800": 1.05,
        "V100": 1.2,
        "L40": 1.1,
        "L4": 1.45,
        "A10": 1.35,
        "T4": 1.8,
        "RTX4090": 0.85,
    }
    return table.get(hardware.upper(), table.get(hardware, 1.0))


def _stable_jitter(*values: Any) -> float:
    data = "|".join(str(v) for v in values).encode("utf-8")
    digest = hashlib.blake2b(data, digest_size=8).digest()
    jitter_raw = int.from_bytes(digest, "big") / float(2**64 - 1)
    return 0.9 + jitter_raw * 0.2


class VidurPerformanceProvider(PerformanceProvider):
    def __init__(self, config: VidurProviderConfig) -> None:
        self._config = config
        self._cache: Dict[Tuple, PhaseMetrics] = {}
        self._targets: Dict[str, Dict[str, object]] = {}
        self._realtime_runner: Optional[VidurRealtimeRunner] = (
            get_shared_vidur_runner(dtype=config.default_dtype, cache_root=config.realtime_cache_dir)
            if config.realtime_enabled
            else None
        )
        if self._realtime_runner is None:
            raise RuntimeError("VIDUR realtime runner is required but was not initialised")
        if config.cache_path:
            config.cache_path.parent.mkdir(parents=True, exist_ok=True)

        oracle = config.oracle
        self._oracle = oracle

    def register_target(
        self,
        *,
        target_id: str,
        model: str,
        hardware: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "model": model,
            "hardware": hardware,
        }
        if not metadata:
            raise ValueError(f"Target {target_id} metadata must include VIDUR profiling information")
        record["metadata"] = dict(metadata)
        vidur_profile = metadata.get("vidur") or metadata.get("vidur_profile")
        if not isinstance(vidur_profile, Mapping):
            raise ValueError(f"Target {target_id} metadata must include a `vidur` or `vidur_profile` mapping")
        record["vidur_profile"] = dict(vidur_profile)
        self._realtime_runner.validate_profile(record["vidur_profile"])
        draft_profile = metadata.get("fused_draft_profile")
        if draft_profile:
            record["fused_draft_profile"] = dict(draft_profile)
        self._targets[target_id] = record

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
            tuple(request.tokens_per_request) if request.tokens_per_request is not None else None,
            tuple(request.context_per_request) if request.context_per_request is not None else None,
            request.prompt_tokens,
            request.context_tokens,
            request.target_id,
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
        if self._realtime_runner is None:
            raise RuntimeError("VIDUR realtime runner is required but unavailable")
        target_info = self._targets.get(request.target_id or "")
        if target_info is None:
            raise RuntimeError("VIDUR target metadata missing for realtime prediction; ensure targets register VIDUR profiles")
        try:
            metrics = self._realtime_runner.predict(request, target_info)
        except Exception as exc:  # pragma: no cover - escalate for visibility
            raise RuntimeError("VIDUR realtime prediction failed") from exc
        if metrics is None:
            raise RuntimeError("VIDUR realtime prediction returned no metrics")
        return metrics

    def _call_vidur_python(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        if request.phase == "prefill":
            result = vidur.prefill_latency(  # type: ignore[call-arg]
                model=request.model,
                gpu=request.hardware,
                batch_size=request.batch_size,
                sequence_length=request.sequence_length,
                dtype=self._config.default_dtype,
            )
        else:
            result = vidur.decode_latency(  # type: ignore[call-arg]
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

    def _call_vidur_oracle(self, request: PhaseRequest) -> Optional[PhaseMetrics]:
        if _VidurBatchFeatures is None or _VidurModelConfig is None or _VidurRequest is None or _VidurPhase is None:
            return None
        phase = _normalise_phase(request.phase)
        if phase is None:
            return None
        requests = _build_vidur_requests(request, phase)
        if not requests:
            return None

        model_name = request.model or self._targets.get(request.target_id or "", {}).get("model") or "unknown"
        gpu_sku = request.hardware or self._targets.get(request.target_id or "", {}).get("hardware") or "unknown"
        max_batch = request.batch_size or len(requests)

        model_config = _VidurModelConfig(  # type: ignore[call-arg]
            model_type=str(model_name),
            gpu_sku=str(gpu_sku),
            max_batch_size=max(1, max_batch),
        )
        batch_features = _VidurBatchFeatures(tuple(requests), phase=phase)  # type: ignore[call-arg]

        oracle = self._oracle
        if oracle is None:
            return None
        metrics = oracle.predict_iteration(model_config, batch_features, None)  # type: ignore[arg-type]
        if metrics is None:
            return None
        latency_p95 = getattr(metrics, "latency_p95_ms", None)
        energy = getattr(metrics, "energy_mj", None)
        return PhaseMetrics(
            latency_ms=float(metrics.total_latency_ms),
            latency_p95_ms=float(latency_p95) if latency_p95 is not None else None,
            energy_mj=float(energy) if energy is not None else None,
        )

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


def _normalise_phase(value: Optional[str]):
    if _VidurPhase is None:
        return None
    if value is None:
        return None
    text = value.lower()
    if text in {"prefill", "prompt"}:
        return _VidurPhase.PREFILL
    if text in {"decode", "draft_decode", "target_decode", "verify"}:
        return _VidurPhase.DECODE
    return None


def _build_vidur_requests(request: PhaseRequest, phase) -> Sequence[object]:  # pragma: no cover - thin wrapper
    if _VidurRequest is None:
        return ()
    batch_size = int(request.batch_size or 0)
    if batch_size <= 0:
        tokens_seq = request.tokens_per_request or ()
        batch_size = len(tokens_seq) if tokens_seq else 1

    prompt_default = max(1, request.prompt_tokens or request.sequence_length or request.context_length or 1)
    context_default = max(0, request.context_length or request.sequence_length or prompt_default)
    if phase == _VidurPhase.PREFILL:
        spec_default = 0
    else:
        spec_default = max(1, request.tokens_to_generate or request.fanout or 1)

    prompt_values = _expand_sequence(
        request.tokens_per_request if phase == _VidurPhase.PREFILL else None,
        batch_size,
        prompt_default,
    )
    spec_values = _expand_sequence(
        request.tokens_per_request if phase != _VidurPhase.PREFILL else None,
        batch_size,
        spec_default,
    )
    context_values = _expand_sequence(
        request.context_per_request,
        batch_size,
        context_default,
    )

    requests = []
    for idx in range(batch_size):
        prompt_tokens = prompt_values[idx] if phase == _VidurPhase.PREFILL else 0
        spec_tokens = spec_values[idx] if phase != _VidurPhase.PREFILL else 0
        context_tokens = context_values[idx]
        request_obj = _VidurRequest(  # type: ignore[call-arg]
            phase=phase,
            prompt_tokens=prompt_tokens,
            speculation_tokens=spec_tokens,
            context_tokens=context_tokens,
            tokens_to_generate=spec_tokens if phase != _VidurPhase.PREFILL else prompt_tokens,
            fanout=max(1, request.fanout),
        )
        requests.append(request_obj)
    return tuple(requests)


def _expand_sequence(values: Optional[Sequence[int]], size: int, default: int) -> Sequence[int]:
    if size <= 0:
        return ()
    if values:
        seq = [max(0, int(v)) for v in values]
        if not seq:
            seq = [max(0, int(default))]
    else:
        seq = [max(0, int(default))]
    if len(seq) < size:
        seq.extend([seq[-1]] * (size - len(seq)))
    return tuple(seq[:size])
