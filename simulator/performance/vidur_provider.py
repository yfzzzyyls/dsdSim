"""VIDUR-backed performance provider."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from .base import PhaseMetrics, PhaseRequest, PerformanceProvider

def _append_vidur_repo_to_path() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    candidate = base_dir / "thirdparty" / "vidur"
    if candidate.exists() and candidate.is_dir():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

_append_vidur_repo_to_path()

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


class VidurPerformanceProvider(PerformanceProvider):
    def __init__(self, config: VidurProviderConfig) -> None:
        self._config = config
        self._cache: Dict[Tuple, PhaseMetrics] = {}
        self._targets: Dict[str, Dict[str, object]] = {}
        if config.cache_path:
            config.cache_path.parent.mkdir(parents=True, exist_ok=True)

        oracle = config.oracle
        if oracle is None and _VidurOracle is not None and (config.table_path or config.bootstrap_defaults):
            oracle = _VidurOracle(
                table_path=str(config.table_path) if config.table_path else None,
                default_latency_ms=config.default_latency_ms,
                neighbors=max(1, config.neighbors),
                prefer_exact=config.prefer_exact,
                bootstrap_defaults=config.bootstrap_defaults,
            )
        self._oracle = oracle

    def register_target(self, *, target_id: str, model: str, hardware: str,
                        prefill_per_token_ms: float, decode_per_token_ms: float) -> None:
        self._targets[target_id] = {
            "model": model,
            "hardware": hardware,
            "prefill_per_token_ms": prefill_per_token_ms,
            "decode_per_token_ms": decode_per_token_ms,
        }

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
        if self._oracle is not None:
            metrics = self._call_vidur_oracle(request)
            if metrics is not None:
                return metrics
        if vidur is not None:
            return self._call_vidur_python(request)
        if self._config.binary:
            return self._call_vidur_cli(request)
        raise RuntimeError(
            "VIDUR provider requires either the `vidur` Python package, vidur_integration oracle, or a CLI binary."
        )

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

