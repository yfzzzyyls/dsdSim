"""Runtime VIDUR latency predictions using the vendored simulator."""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .base import PhaseMetrics, PhaseRequest


logger = logging.getLogger(__name__)


def _append_vidur_repo_to_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    vidur_root = base_dir / "thirdparty" / "vidur"
    if str(vidur_root) not in sys.path:
        sys.path.insert(0, str(vidur_root))
    return vidur_root


_VIDUR_ROOT = _append_vidur_repo_to_path()

try:  # pragma: no cover - import side effects are heavy but optional
    from vidur.config import (
        MetricsConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        SarathiSchedulerConfig,
    )
    from vidur.entities import Batch, Request
    from vidur.execution_time_predictor.random_forrest_execution_time_predictor import (
        RandomForrestExecutionTimePredictor,
    )
except Exception as exc:  # pragma: no cover
    logger.debug("Failed to import vidur runtime modules: %s", exc)
    throw_on_use = exc
else:
    throw_on_use = None


_DEVICE_ALIAS = {
    "a40": "DeviceSKUType.A40",
    "a100": "DeviceSKUType.A100",
    "h100": "DeviceSKUType.H100",
}

_NODE_ALIAS = {
    "a40_pairwise_nvlink": "NodeSKUType.A40_PAIRWISE_NVLINK",
    "a100_pairwise_nvlink": "NodeSKUType.A100_PAIRWISE_NVLINK",
    "h100_pairwise_nvlink": "NodeSKUType.H100_PAIRWISE_NVLINK",
    "a100_dgx": "NodeSKUType.A100_DGX",
    "h100_dgx": "NodeSKUType.H100_DGX",
}

_DEFAULT_NODE = {
    "DeviceSKUType.A40": "NodeSKUType.A40_PAIRWISE_NVLINK",
    "DeviceSKUType.A100": "NodeSKUType.A100_PAIRWISE_NVLINK",
    "DeviceSKUType.H100": "NodeSKUType.H100_PAIRWISE_NVLINK",
}


@dataclass(frozen=True)
class _PredictorKey:
    model: str
    device: str
    node: str
    tensor_parallel: int
    pipeline_parallel: int
    scheduler: str
    chunk_size: int


class VidurRealtimeRunner:
    """Evaluate latency using VIDUR's learned execution-time predictors."""

    def __init__(self, *, dtype: str = "fp16", cache_root: Optional[Path] = None) -> None:
        if throw_on_use is not None:
            raise RuntimeError(
                "VIDUR runtime modules are unavailable; ensure VIDUR dependencies are installed"
            ) from throw_on_use

        self._dtype = dtype
        self._vidur_root = _VIDUR_ROOT
        self._cache_root = Path(cache_root or (Path(tempfile.gettempdir()) / "vidur_realtime_cache"))
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._model_cache = self._cache_root / "model_cache"
        self._model_cache.mkdir(parents=True, exist_ok=True)
        self._metrics_root = self._cache_root / "metrics"
        self._metrics_root.mkdir(parents=True, exist_ok=True)
        self._predictors: Dict[_PredictorKey, RandomForrestExecutionTimePredictor] = {}

    def predict(self, request: PhaseRequest, target_info: Mapping[str, Any]) -> Optional[PhaseMetrics]:
        profile = self._extract_profile(target_info)
        if profile is None:
            raise ValueError(
                "Target metadata is missing a `vidur` profile required for realtime predictions."
            )

        predictor = self._get_predictor(profile)
        batch = self._build_batch(request, profile)
        if batch is None:
            return None

        execution_time = predictor.get_execution_time(batch, pipeline_stage=0)
        latency_ms = execution_time.total_time * 1000.0
        if not math.isfinite(latency_ms):
            raise ValueError("VIDUR realtime predictor returned a non-finite latency")
        return PhaseMetrics(latency_ms=latency_ms)

    # ------------------------------------------------------------------
    # Predictor initialisation helpers
    # ------------------------------------------------------------------

    def _extract_profile(self, target_info: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        if isinstance(target_info, dict):
            if "vidur_profile" in target_info and isinstance(target_info["vidur_profile"], Mapping):
                return target_info["vidur_profile"]
            metadata = target_info.get("metadata")
            if isinstance(metadata, Mapping):
                nested = metadata.get("vidur") or metadata.get("vidur_profile")
                if isinstance(nested, Mapping):
                    return nested
        return None

    def _get_predictor(self, profile: Mapping[str, Any]) -> RandomForrestExecutionTimePredictor:
        key = _PredictorKey(
            model=str(profile.get("model_name")),
            device=self._resolve_device(profile.get("device")),
            node=self._resolve_node(profile.get("network_device"), profile.get("device")),
            tensor_parallel=int(profile.get("tensor_parallel", 1)),
            pipeline_parallel=int(profile.get("pipeline_parallel", 1)),
            scheduler=str(profile.get("scheduler", "sarathi")).lower(),
            chunk_size=int(profile.get("chunk_size", 512)),
        )

        if key.model == "None":
            raise ValueError("vidur profile must include `model_name`")

        predictor = self._predictors.get(key)
        if predictor is not None:
            return predictor

        metrics_config = MetricsConfig(
            write_metrics=False,
            write_json_trace=False,
            enable_chrome_trace=False,
            save_table_to_wandb=False,
            store_plots=False,
            store_operation_metrics=False,
            store_token_completion_metrics=False,
            store_request_metrics=False,
            store_batch_metrics=False,
            store_utilization_metrics=False,
            output_dir=str(self._metrics_root),
            cache_dir=str(self._model_cache),
        )

        scheduler_config = SarathiSchedulerConfig(chunk_size=key.chunk_size)

        predictor_config = RandomForrestExecutionTimePredictorConfig(
            compute_input_file=self._format_path("data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv"),
            attention_input_file=self._format_path(
                "data/profiling/compute/{DEVICE}/{MODEL}/attention.csv"
            ),
            all_reduce_input_file=self._format_path(
                "data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv"
            ),
            send_recv_input_file=self._format_path(
                "data/profiling/network/{NETWORK_DEVICE}/send_recv.csv"
            ),
            cpu_overhead_input_file=self._format_path(
                "data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv"
            ),
            no_cache=False,
        )

        replica_config = ReplicaConfig(
            model_name=key.model,
            tensor_parallel_size=key.tensor_parallel,
            num_pipeline_stages=key.pipeline_parallel,
            device=key.device,
            network_device=key.node,
        )

        try:
            predictor = RandomForrestExecutionTimePredictor(
                predictor_config=predictor_config,
                replica_config=replica_config,
                replica_scheduler_config=scheduler_config,
                metrics_config=metrics_config,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"VIDUR profiling assets missing for model={key.model}, device={key.device}, node={key.node}"
            ) from exc
        self._predictors[key] = predictor
        return predictor

    def _format_path(self, relative: str) -> str:
        return str(self._vidur_root / relative)

    def _resolve_device(self, value: Any) -> str:
        if value is None:
            raise ValueError("vidur profile must include a `device`")
        text = str(value).strip()
        if text.startswith("DeviceSKUType."):
            return text
        alias = _DEVICE_ALIAS.get(text.lower())
        if alias:
            return alias
        raise ValueError(f"Unsupported VIDUR device alias: {value}")

    def _resolve_node(self, value: Any, device_hint: Any) -> str:
        if value is None:
            device_key = self._resolve_device(device_hint) if device_hint else None
            if device_key and device_key in _DEFAULT_NODE:
                return _DEFAULT_NODE[device_key]
            raise ValueError("Unable to infer VIDUR network device; specify `network_device` explicitly")
        text = str(value).strip()
        if text.startswith("NodeSKUType."):
            return text
        alias = _NODE_ALIAS.get(text.lower())
        if alias:
            return alias
        raise ValueError(f"Unsupported VIDUR network alias: {value}")

    # ------------------------------------------------------------------
    # Batch construction helpers
    # ------------------------------------------------------------------

    def _build_batch(self, request: PhaseRequest, profile: Mapping[str, Any]) -> Optional[Batch]:
        batch_size = max(1, int(request.batch_size or 1))
        if request.phase.lower() in {"prefill", "prompt"}:
            tokens = self._expand_sequence(request.tokens_per_request, batch_size, self._default_prefill_tokens(request))
            contexts = self._expand_sequence(request.context_per_request, batch_size, request.context_length)
            batch_requests: List[Request] = []
            token_counts: List[int] = []
            for prompt_tokens, context_tokens in zip(tokens, contexts):
                prompt_tokens = max(1, int(prompt_tokens))
                context_tokens = max(0, int(context_tokens or 0))
                req = Request(arrived_at=0.0, num_prefill_tokens=prompt_tokens, num_decode_tokens=0)
                req._num_processed_tokens = context_tokens
                req._scheduled = True
                batch_requests.append(req)
                token_counts.append(prompt_tokens)
            return Batch(replica_id=0, requests=batch_requests, num_tokens=token_counts)

        # decode / verify phases
        decode_tokens = self._expand_sequence(
            request.tokens_per_request,
            batch_size,
            max(1, request.tokens_to_generate or 1),
        )
        contexts = self._expand_sequence(
            request.context_per_request,
            batch_size,
            max(request.context_length, request.sequence_length, 1),
        )
        batch_requests = []
        token_counts = []
        for tokens_to_generate, context_tokens in zip(decode_tokens, contexts):
            decode = max(1, int(tokens_to_generate))
            context = max(1, int(context_tokens))
            req = Request(arrived_at=0.0, num_prefill_tokens=context, num_decode_tokens=decode)
            req._num_processed_tokens = context
            req._is_prefill_complete = True
            req._scheduled = True
            batch_requests.append(req)
            token_counts.append(decode)
        return Batch(replica_id=0, requests=batch_requests, num_tokens=token_counts)

    def _default_prefill_tokens(self, request: PhaseRequest) -> int:
        for candidate in (request.prompt_tokens, request.sequence_length, request.context_length, 1):
            if candidate and candidate > 0:
                return int(candidate)
        return 1

    def _expand_sequence(self, values: Optional[Sequence[int]], size: int, default: int) -> List[int]:
        if values is None:
            return [int(default)] * size
        seq = list(int(max(0, v)) for v in values)
        if not seq:
            return [int(default)] * size
        if len(seq) < size:
            seq.extend([seq[-1]] * (size - len(seq)))
        return seq[:size]
