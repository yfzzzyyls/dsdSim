from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src import sim


@dataclass
class RequestSample:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    ttft_ms: Optional[float] = None
    draft_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def finish_ms(self) -> float:
        return self.arrival_ms + self.total_latency_ms


@dataclass
class ActualRun:
    samples: List[RequestSample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def span_ms(self) -> float:
        if not self.samples:
            return 0.0
        start = min(r.arrival_ms for r in self.samples)
        end = max(r.finish_ms for r in self.samples)
        return max(0.0, end - start)


@dataclass
class CosimReport:
    trace_path: Path
    actual_summary: Dict[str, float]
    simulated_summary: Dict[str, float]
    simulated_metrics: Dict[str, Any]
    differences: List[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]]

    def to_dict(self) -> Dict[str, Any]:
        diff_rows = [
            {
                "metric": name,
                "actual": actual,
                "simulated": simulated,
                "delta": delta,
                "relative_pct": pct,
            }
            for name, actual, simulated, delta, pct in self.differences
        ]
        return {
            "trace_path": str(self.trace_path),
            "actual_summary": self.actual_summary,
            "simulated_summary": self.simulated_summary,
            "simulated_metrics": self.simulated_metrics,
            "differences": diff_rows,
        }


_EXPECTED_KEYS = {
    "request_id",
    "arrival_ms",
    "prompt_tokens",
    "completion_tokens",
    "completion_tokens",
    "total_latency_ms",
}


def load_actual_run(path: Path) -> ActualRun:
    """Load request-level metrics from a JSON or JSONL file.

    The loader expects each record to contain at least:
      - ``request_id`` (str)
      - ``arrival_ms`` (float)
      - ``prompt_tokens`` (int)
      - ``completion_tokens`` (int)
      - ``total_latency_ms`` (float)

    Optional fields:
      - ``ttft_ms``
      - ``draft_id``
      - arbitrary metadata (captured verbatim)
    """

    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text().strip()
    records: Iterable[Mapping[str, Any]]
    if not text:
        records = []
    elif text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of request entries")
        records = data
    else:
        entries: List[Mapping[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(json.loads(line))
        records = entries

    samples: List[RequestSample] = []
    for raw in records:
        sample = _parse_request_entry(raw)
        samples.append(sample)

    samples.sort(key=lambda s: (s.arrival_ms, s.request_id))
    return ActualRun(samples=samples)


def _parse_request_entry(payload: Mapping[str, Any]) -> RequestSample:
    missing = [key for key in ("request_id", "arrival_ms", "prompt_tokens", "completion_tokens") if key not in payload]
    if missing:
        raise ValueError(f"request entry missing required fields: {missing}")

    request_id = str(payload["request_id"])
    arrival_ms = float(payload["arrival_ms"])
    prompt_tokens = int(payload["prompt_tokens"])
    completion_tokens = int(payload["completion_tokens"])

    total_latency = payload.get("total_latency_ms")
    finish_ms = payload.get("finish_ms")
    if total_latency is None and finish_ms is not None:
        total_latency = float(finish_ms) - arrival_ms
    if total_latency is None and "ttft_ms" in payload and "decode_latency_ms" in payload:
        total_latency = float(payload["ttft_ms"]) + float(payload["decode_latency_ms"])
    if total_latency is None:
        raise ValueError(f"request {request_id} missing total latency information")

    total_latency = float(total_latency)
    if total_latency <= 0:
        raise ValueError(f"request {request_id} total_latency_ms must be positive")

    ttft_ms = payload.get("ttft_ms")
    draft_id = payload.get("draft_id")
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {}

    return RequestSample(
        request_id=request_id,
        arrival_ms=arrival_ms,
        prompt_tokens=max(1, prompt_tokens),
        completion_tokens=max(1, completion_tokens),
        total_latency_ms=total_latency,
        ttft_ms=None if ttft_ms is None else float(ttft_ms),
        draft_id=None if draft_id is None else str(draft_id),
        metadata=metadata,
    )


def generate_trace(run: ActualRun, trace_path: Path, *, default_draft: str = "trace_draft") -> Path:
    """Write a simulator trace JSONL derived from ``run``."""

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as fh:
        for idx, sample in enumerate(run.samples):
            draft_id = sample.draft_id or f"{default_draft}_{idx % 32}"
            record = {
                "request_id": sample.request_id,
                "arrival_ms": sample.arrival_ms,
                "draft_id": draft_id,
                "prompt_tokens": sample.prompt_tokens,
                "target_tokens": sample.completion_tokens,
            }
            if sample.metadata:
                record["metadata"] = sample.metadata
            fh.write(json.dumps(record))
            fh.write("\n")
    return trace_path


def _summarise_requests(run: ActualRun) -> Dict[str, float]:
    if not run.samples:
        return {}

    latencies = [sample.total_latency_ms for sample in run.samples]
    latencies.sort()
    start = min(sample.arrival_ms for sample in run.samples)
    end = max(sample.finish_ms for sample in run.samples)
    span = max(end - start, 1e-9)

    def _pct(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        idx = int(round(percentile * (len(values) - 1)))
        idx = max(0, min(idx, len(values) - 1))
        return values[idx]

    summary: Dict[str, float] = {
        "count": float(len(run.samples)),
        "throughput_jobs_s": 1000.0 * len(run.samples) / span,
        "avg_ms": statistics.mean(latencies),
        "p50_ms": _pct(latencies, 0.50),
        "p95_ms": _pct(latencies, 0.95),
        "p99_ms": _pct(latencies, 0.99),
    }

    ttft_values = [s.ttft_ms for s in run.samples if s.ttft_ms is not None]
    if ttft_values:
        ttft_values.sort()
        summary["ttft_avg_ms"] = statistics.mean(ttft_values)
        summary["ttft_p50_ms"] = _pct(ttft_values, 0.50)
        summary["ttft_p95_ms"] = _pct(ttft_values, 0.95)

    total_tokens = sum(s.completion_tokens for s in run.samples)
    summary["completion_tokens"] = float(total_tokens)
    summary["tokens_per_s"] = 1000.0 * total_tokens / span
    return summary


def _run_simulation(config_path: Path, trace_path: Path, run: ActualRun, *, emit_output: bool = False) -> Tuple[Dict[str, float], Dict[str, Any]]:
    cfg = sim.load_config(str(config_path))
    cfg.trace_path = str(trace_path)
    max_finish = max((sample.finish_ms for sample in run.samples), default=0.0)
    cfg.sim_time_ms = max(cfg.sim_time_ms, math.ceil(max_finish + 10.0))
    result = sim.run(cfg)
    metrics, targets, drafts = sim._unpack_run_result(result)  # type: ignore[attr-defined]
    summary = metrics.summary() if hasattr(metrics, "summary") else {}
    metrics_json = sim._collect_metrics_json(cfg, metrics, summary, targets)  # type: ignore[attr-defined]
    if emit_output:
        sim._print_report(cfg, metrics, summary, targets, drafts, metrics_json)  # type: ignore[attr-defined]
    return summary, metrics_json


def compare_runs(actual: ActualRun, config_path: Path, *, workdir: Optional[Path] = None, emit_output: bool = False) -> CosimReport:
    if workdir is None:
        workdir = Path("experiments") / "results" / "cosim"
    workdir.mkdir(parents=True, exist_ok=True)
    trace_path = workdir / "trace_actual.jsonl"
    generate_trace(actual, trace_path)

    actual_summary = _summarise_requests(actual)
    simulated_summary, metrics_json = _run_simulation(config_path, trace_path, actual, emit_output=emit_output)

    differences: List[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    keys = set(actual_summary.keys()) | set(simulated_summary.keys())
    for key in sorted(keys):
        actual_value = actual_summary.get(key)
        simulated_value = simulated_summary.get(key)
        delta = None
        pct = None
        if actual_value is not None and simulated_value is not None:
            delta = simulated_value - actual_value
            if actual_value != 0:
                pct = 100.0 * delta / actual_value
        differences.append((key, actual_value, simulated_value, delta, pct))

    return CosimReport(
        trace_path=trace_path,
        actual_summary=actual_summary,
        simulated_summary=simulated_summary,
        simulated_metrics=metrics_json,
        differences=differences,
    )
