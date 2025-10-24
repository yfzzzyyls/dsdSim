# sim.py v1 minimal distributed speculative-decoding simulator
# deps: pip install simpy pyyaml

import argparse, random, simpy, yaml, math, json, itertools, copy, time
import time
from collections import deque, defaultdict
from types import MappingProxyType
from pathlib import Path

# Print SimPy version for debugging
print(f"SimPy version: {simpy.__version__}", flush=True)
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Sequence, Iterable, Tuple, Mapping
from dataclasses import dataclass

from performance import PhaseRequest, PerformanceModelConfig, create_performance_provider
from network.topology import build_latency_lookup, NetworkModelError
from network.fabric import NetworkFabric
from trace.trace_loader import iter_trace_records
from trace.types import TraceRecord, TraceParseError
from acceptance.regressor import AcceptanceRegressor

# ---------- Config & Types ----------

_ACCEPTANCE_MODEL_CACHE: Dict[str, AcceptanceRegressor] = {}
_GLOBAL_PROFILER = None

@dataclass
class TargetParams:
    id: str
    model: str = ""                   # optional metadata (e.g., "llama-3.1-8b")
    gpu: str = ""                     # optional metadata (e.g., "A100", "H100", "L4")
    weight: float = 1.0               # capacity weight (relative speed)
    batch_window_ms: float = 6.0      # Delta: wait to form a batch
    batch_size: int = 32              # B: max jobs per batch
    cluster: str = "default"
    kv_capacity_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    mode: str = "distributed"         # distributed | fused
    fused_draft_profile: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class GammaConversationStats:
    acceptance_ratio: float
    tokens_generated: int
    tokens_accepted: int


@dataclass(frozen=True)
class GammaContext:
    draft_id: str
    target_id: str
    context_length: int
    queue_depth: int

    acceptance_probabilities: Tuple[float, ...] = tuple()


class GammaPolicy:
    def required_depth(self, default_gamma: int) -> int:
        return default_gamma

    def select_gamma(
        self,
        draft_id: str,
        default_gamma: int,
        context: Optional[GammaContext] = None,
    ) -> int:
        return default_gamma

    def update_gamma(self, draft_id: str, stats: GammaConversationStats) -> None:
        return


class ConstantGammaPolicy(GammaPolicy):
    def __init__(self, default_gamma: int) -> None:
        self._default = max(1, int(default_gamma))

    def required_depth(self, default_gamma: int) -> int:
        return 0

    def select_gamma(
        self,
        draft_id: str,
        default_gamma: int,
        context: Optional[GammaContext] = None,
    ) -> int:
        return self._default



class SpecPPGammaPolicy(GammaPolicy):
    """Implements SpecDec++-style adaptive gamma selection."""

    def __init__(self, default_gamma: int, config: Mapping[str, Any]) -> None:
        self._default = max(1, int(default_gamma))
        self._min = max(1, int(config.get("min_gamma", 1)))
        self._max = max(self._min, int(config.get("max_gamma", self._default)))
        self._threshold = float(config.get("stop_threshold", 0.7))
        if not 0.0 <= self._threshold <= 1.0:
            raise ValueError("stop_threshold must be within [0, 1]")
        self._fallback = max(1, int(config.get("fallback_gamma", self._default)))

    def required_depth(self, default_gamma: int) -> int:
        return max(self._max, default_gamma, self._fallback)

    def select_gamma(
        self,
        draft_id: str,
        default_gamma: int,
        context: Optional[GammaContext] = None,
    ) -> int:
        if context is None:
            return self._fallback

        probabilities = list(context.acceptance_probabilities or [])
        if not probabilities:
            return self._fallback

        while len(probabilities) < self._max:
            probabilities.append(probabilities[-1])

        min_tokens = max(1, self._min)
        max_tokens = max(min_tokens, self._max)
        cum_accept = 1.0
        selected = min_tokens

        selected = min_tokens
        for idx in range(1, max_tokens + 1):
            prob = float(probabilities[idx - 1])
            prob = max(0.0, min(1.0, prob))
            cum_accept *= prob
            if idx < min_tokens:
                selected = idx
                continue
            rejection = 1.0 - cum_accept
            if rejection > self._threshold:
                # Roll back to the last depth that stayed under threshold.
                selected = max(min_tokens, idx - 1)
                break
            selected = idx

        return max(min_tokens, min(max_tokens, selected))

    def update_gamma(self, draft_id: str, stats: GammaConversationStats) -> None:
        return



class AcceptanceBackoffGammaPolicy(GammaPolicy):
    def __init__(self, default_gamma: int, config: Mapping[str, Any]) -> None:
        self._default = max(1, int(default_gamma))
        self._min = max(1, int(config.get("min_gamma", 1)))
        self._max = max(self._min, int(config.get("max_gamma", self._default)))
        self._low = float(config.get("low_acceptance", 0.3))
        self._high = float(config.get("high_acceptance", 0.6))
        self._last_ratio: Dict[str, float] = {}

    def required_depth(self, default_gamma: int) -> int:
        return 0

    def select_gamma(
        self,
        draft_id: str,
        default_gamma: int,
        context: Optional[GammaContext] = None,
    ) -> int:
        ratio = self._last_ratio.get(draft_id)
        if ratio is None:
            return self._default
        if ratio <= self._low:
            return self._min
        if ratio >= self._high:
            return self._max
        return self._default

    def update_gamma(self, draft_id: str, stats: GammaConversationStats) -> None:
        self._last_ratio[draft_id] = max(0.0, min(1.0, stats.acceptance_ratio))


@dataclass
class DraftParams:
    id: str
    capability: float = 1.0           # relative compute speed (affects generation rate)
    burst_factor: float = 1.0         # short-term burst multiplier
    reliability: float = 0.99         # connection reliability (0-1)
    cluster: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ReplayDraftSpec:
    """Metadata used to distribute trace arrivals across drafts."""
    draft_id: str
    capability: float = 1.0
    tier: str = "default"

@dataclass
class ConnectionParams:
    draft_id: str
    target_id: str
    forward_latency_ms: float         # draft -> target latency
    response_latency_ms: float        # target -> draft latency
    acceptance_rate: Optional[float] = None  # fallback acceptance probability when no model
    cluster: str = "default"
    network_forward_key: Optional[Tuple[str, str]] = None
    network_response_key: Optional[Tuple[str, str]] = None

@dataclass
class DraftChunk:
    chunk_id: int
    session_id: int
    draft_id: str
    target_id: str
    tokens: int                      # number of tokens (we don't simulate actual tokens)
    created_ms: float
    sent_ms: float
    expected_response_ms: float

@dataclass
class VerifyResult:
    chunk_id: int
    accepted_tokens: int
    rejected_tokens: int
    total_tokens: int


@dataclass
class BranchCandidate:
    branch_id: int
    depth: int
    score: float

@dataclass
class WorkloadCfg:
    arrival: str = "deterministic"    # "deterministic" | "poisson"
    interarrival_ms: float = 12.0     # used if deterministic
    rate_rps: float = 100.0           # used if poisson: mean rate


_FUSED_MODEL_DEFAULTS = {
    "meta-llama/Llama-2-70b-hf": "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-70B": "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen-72B": "Qwen/Qwen-7B",
}


def _default_fused_profile(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    info = entry.get("vidur") or entry.get("vidur_profile")
    if not info:
        return None
    model_name = info.get("model_name") or info.get("model")
    if not model_name:
        return None
    draft_model = _FUSED_MODEL_DEFAULTS.get(model_name)
    if not draft_model:
        return None
    profile: Dict[str, Any] = {"model_name": draft_model}
    device = info.get("device") or entry.get("gpu")
    if device:
        profile["device"] = device
    network_device = info.get("network_device") or entry.get("network_device")
    if network_device:
        profile["network_device"] = network_device
    for key in ("tensor_parallel", "pipeline_parallel", "scheduler", "chunk_size"):
        if key in info:
            profile[key] = info[key]
    return profile

@dataclass
class ThinkTimeConfig:
    enabled: bool = True
    distribution: str = "workload"  # workload | exponential | lognormal | constant
    mean_ms: float = 2000.0
    cv: float = 0.5                   # coefficient of variation for lognormal
    min_ms: float = 0.0

@dataclass
class Config:
    sim_time_ms: float = 10_000
    seed: int = 0
    execution_mode: str = "blocking"  # "blocking" | "speculative"
    gamma: int = 4                     # tokens per chunk
    
    # Conversation parameters (NEW)
    answer_length: int = 20           # tokens per answer (can be overridden by distribution)
    answer_length_mean: float = 400   # mean for normal distribution
    answer_length_std: float = 100    # std dev for normal distribution
    answer_length_min: int = 50       # minimum answer length
    answer_length_max: int = 800      # maximum answer length
    use_answer_distribution: bool = False  # use normal distribution for answer length
    prompt_length_min: int = 10       # minimum prompt length
    prompt_length_max: int = 200      # maximum prompt length
    prompt_scale_by_capability: bool = True  # scale prompt length by device capability
    
    # Batching parameters for mixed prefill/decode
    mixed_batching: bool = True  # Allow mixing prefill and decode in same batch
    
    router: str = "round_robin"       # "round_robin" | "jsq2" | "wjsq2"
    router_params: Dict[str, Any] = field(default_factory=lambda: {"d_choices": 2})
    scheduler_config: Dict[str, Any] = field(default_factory=dict)  # Scheduler configuration
    devices: List[Dict[str, Any]] = field(default_factory=list)   # list of dicts with role/params
    connections: List[Dict[str, Any]] = field(default_factory=list)  # draft-target connections
    cluster_router: Dict[str, str] = field(default_factory=dict)
    cluster_router_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    global_router: Optional[str] = None
    global_router_params: Dict[str, Any] = field(default_factory=dict)
    trace_path: Optional[str] = None
    trace_defaults: Dict[str, Any] = field(default_factory=dict)
    trace_replay: Dict[str, Any] = field(default_factory=dict)
    performance_model: PerformanceModelConfig = field(default_factory=PerformanceModelConfig)
    workload: WorkloadCfg = field(default_factory=WorkloadCfg)
    think_time: ThinkTimeConfig = field(default_factory=ThinkTimeConfig)
    burn_in_ms: float = 0.0           # Ignore first X ms for stats
    verbose: bool = True               # Print progress updates
    debug: bool = False                # Print detailed batch formation
    speculation_framework: str = "vanilla"
    speculation_execution_mode: str = "distributed"
    speculation_config: Dict[str, Any] = field(default_factory=dict)
    acceptance_model_path: Optional[str] = None
    acceptance_config: Dict[str, Any] = field(default_factory=dict)
    acceptance_model_disabled: bool = False
    acceptance_use_classifier: bool = True
    acceptance_context_bucket: int = 1
    acceptance_depth_bucket: int = 1
    acceptance_pending_bucket: int = 1
    acceptance_queue_bucket: int = 1
    acceptance_surrogate_config: Dict[str, Any] = field(default_factory=dict)
    acceptance_use_classifier: bool = True
    network_config: Dict[str, Any] = field(default_factory=dict)
    network_enabled: bool = True

@dataclass
class Job:
    jid: int
    created_ms: float
    draft_id: str
    job_type: str = "decode"  # "prefill" or "decode"
    token_count: int = 4  # Number of tokens to process
    started_ms: Optional[float] = None
    finished_ms: Optional[float] = None
    completion_event: Optional[Any] = None  # SimPy event signaled when job completes
    rtt_start_ms: Optional[float] = None  # When RTT measurement starts (before generation)
    rtt_end_ms: Optional[float] = None    # When RTT measurement ends (after response received)
    request_id: Optional[str] = None
    context_len: int = 0
    target_id: Optional[str] = None
    priority_class: str = "standard"
    priority: int = 100
    kv_tokens: int = 0
    phase: str = "decode"
    chunk_index: int = 0
    chunk_count: int = 1
    chunk_barrier: Optional["ChunkBarrier"] = None
    parallelism_plan: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    accepted_tokens: int = 0
    gamma_override: int = 0
    is_fused: bool = False


class ChunkBarrier:
    """Barrier that triggers a final event after a set number of chunk completions."""

    def __init__(self, env: simpy.Environment, chunk_count: int, final_event: Optional[Any]) -> None:
        self.env = env
        self.remaining = max(0, chunk_count)
        self.final_event = final_event

    def notify(self) -> None:
        if self.remaining <= 0:
            return
        self.remaining -= 1
        if self.remaining == 0 and self.final_event is not None and not self.final_event.triggered:
            self.final_event.succeed()

@dataclass
class TokenMetrics:
    """Track token generation and acceptance across all devices."""
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    total_generated_tokens: int = 0
    start_time_ms: float = 0
    end_time_ms: float = 0
    
    def get_effective_tokens_per_second(self) -> float:
        """Calculate the ONE metric that matters."""
        if self.end_time_ms <= self.start_time_ms:
            return 0
        duration_s = (self.end_time_ms - self.start_time_ms) / 1000.0
        return self.total_accepted_tokens / duration_s
    
    def get_acceptance_rate(self) -> float:
        if self.total_generated_tokens == 0:
            return 0
        return self.total_accepted_tokens / self.total_generated_tokens

class Metrics:
    def __init__(self, verbose: bool = True, burn_in_ms: float = 0.0) -> None:
        from collections import defaultdict
        self.completed: List[Job] = []
        self.verbose = verbose
        self.burn_in_ms = burn_in_ms
        self.token_metrics = TokenMetrics()  # Add token tracking
        self.connection_counts = defaultdict(int)     # (draft_id, target_id) -> hits
        self.tier_utilization = defaultdict(list)     # gpu/model tier -> samples
        self.conversations: list[dict[str, Any]] = []  # Completed conversation records

    def add(self, job: Job):
        self.completed.append(job)
        # Progress indicator every 100 jobs
        if self.verbose and len(self.completed) % 100 == 0:
            print(f"  Completed {len(self.completed)} jobs...")

    def record_conversation(
        self,
        conversation_id: str,
        *,
        start_ms: float,
        end_ms: float,
        draft_id: str,
        target_id: str,
        tokens_generated: int,
        tokens_accepted: int,
        answer_tokens: int,
        ttft_ms: Optional[float] = None,
        tpot_samples: Optional[Sequence[float]] = None,
        ttft_breakdown: Optional[Mapping[str, float]] = None,
        decode_breakdown: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Track per-conversation timing (only after burn-in)."""
        if start_ms < self.burn_in_ms:
            return
        duration = end_ms - start_ms
        if duration < 0:
            return
        record: Dict[str, Any] = {
            "id": conversation_id,
            "draft_id": draft_id,
            "target_id": target_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration,
            "tokens_generated": tokens_generated,
            "tokens_accepted": tokens_accepted,
            "answer_tokens": answer_tokens,
            "completed": True,
        }
        if ttft_ms is not None:
            record["ttft_ms"] = ttft_ms
        if tpot_samples:
            record["tpot_samples"] = list(tpot_samples)
        if ttft_breakdown:
            record["ttft_breakdown"] = dict(ttft_breakdown)
        if decode_breakdown:
            record["decode_breakdown"] = dict(decode_breakdown)
        self.conversations.append(record)

    def summary(self) -> Dict[str, float]:
        # Filter out burn-in period
        filtered = [j for j in self.completed if j.created_ms >= self.burn_in_ms]
        if not filtered:
            return {}
        
        # Target-side latency (queue + processing)
        lat = [j.finished_ms - j.created_ms for j in filtered]
        lat.sort()
        
        # Full RTT latency (generation + network + queue + processing + network)
        rtt_jobs = [j for j in filtered if j.rtt_start_ms is not None and j.rtt_end_ms is not None]
        rtt = [j.rtt_end_ms - j.rtt_start_ms for j in rtt_jobs] if rtt_jobs else []
        rtt.sort()
        
        def pct(p):
            i = int(round(p * (len(lat) - 1)))
            return lat[max(0, min(i, len(lat)-1))]
        
        def pct_rtt(p):
            if not rtt:
                return 0
            i = int(round(p * (len(rtt) - 1)))
            return rtt[max(0, min(i, len(rtt)-1))]
        
        # Correct span calculation using min/max of filtered jobs
        start = min(j.created_ms for j in filtered)
        end = max(j.finished_ms for j in filtered)
        span = end - start + 1e-9
        
        result = {
            "count": len(lat),
            "throughput_jobs_s": 1000.0 * len(lat) / span,
            "avg_ms": sum(lat)/len(lat) if lat else 0,
            "p50_ms": pct(0.50) if lat else 0, 
            "p95_ms": pct(0.95) if lat else 0, 
            "p99_ms": pct(0.99) if lat else 0,
            "observed_span_ms": span,
        }

        conv_records: Dict[str, Dict[str, Any]] = {}
        for entry in self.conversations:
            cid = entry.get("id")
            if not cid:
                continue
            conv_records[cid] = dict(entry)
            if entry.get("tpot_samples"):
                conv_records[cid]["tpot_samples"] = list(entry.get("tpot_samples", []))

        conv_map: Dict[str, Dict[str, Any]] = {}
        for job in filtered:
            rid = job.request_id
            if not rid:
                continue
            rec = conv_map.setdefault(
                rid,
                {
                    "start_ms": None,
                    "end_ms": None,
                    "jobs": 0,
                    "draft_id": job.draft_id,
                    "target_id": job.target_id,
                    "decode_tokens": 0,
                    "prompt_tokens": 0,
                    "tpot_samples": [],
                    "first_token_end_ms": None,
                },
            )
            if job.rtt_start_ms is not None:
                rec["start_ms"] = job.rtt_start_ms if rec["start_ms"] is None else min(rec["start_ms"], job.rtt_start_ms)
            if job.rtt_end_ms is not None:
                rec["end_ms"] = job.rtt_end_ms if rec["end_ms"] is None else max(rec["end_ms"], job.rtt_end_ms)
            rec["jobs"] += 1
            if job.job_type == "decode":
                rec["decode_tokens"] += job.token_count
                if job.accepted_tokens > 0 and job.rtt_end_ms is not None and job.rtt_start_ms is not None:
                    chunk_rtt = job.rtt_end_ms - job.rtt_start_ms
                    per_token = chunk_rtt / max(1, job.accepted_tokens)
                    rec.setdefault("tpot_samples", []).extend([per_token] * job.accepted_tokens)
                    first_end = rec.get("first_token_end_ms")
                    if first_end is None or job.rtt_end_ms < first_end:
                        rec["first_token_end_ms"] = job.rtt_end_ms
            elif job.job_type == "prefill":
                rec["prompt_tokens"] += job.token_count
            if job.target_id and not rec.get("target_id"):
                rec["target_id"] = job.target_id

        for rid, rec in conv_map.items():
            start = rec.get("start_ms")
            end = rec.get("end_ms")
            if start is None or end is None or end < start:
                continue
            duration = end - start
            existing = conv_records.get(rid)
            if existing:
                existing.setdefault("start_ms", start)
                existing.setdefault("end_ms", end)
                existing.setdefault("duration_ms", duration)
                existing.setdefault("decode_tokens", rec.get("decode_tokens", 0))
                existing.setdefault("prompt_tokens", rec.get("prompt_tokens", 0))
                existing["observed_jobs"] = rec.get("jobs", 0)
                if existing.get("ttft_ms") is None and rec.get("first_token_end_ms") is not None:
                    existing["ttft_ms"] = rec["first_token_end_ms"] - start if start is not None else None
                if rec.get("tpot_samples"):
                    existing.setdefault("tpot_samples", [])
                    existing["tpot_samples"].extend(rec.get("tpot_samples", []))
            else:
                ttft_ms = None
                if rec.get("first_token_end_ms") is not None:
                    ttft_ms = rec["first_token_end_ms"] - start if start is not None else None
                conv_records[rid] = {
                    "id": rid,
                    "draft_id": rec.get("draft_id"),
                    "target_id": rec.get("target_id"),
                    "start_ms": start,
                    "end_ms": end,
                    "duration_ms": duration,
                    "decode_tokens": rec.get("decode_tokens", 0),
                    "prompt_tokens": rec.get("prompt_tokens", 0),
                    "observed_jobs": rec.get("jobs", 0),
                    "completed": False,
                    "ttft_ms": ttft_ms,
                    "tpot_samples": list(rec.get("tpot_samples", [])),
                }

        observed_entries = [entry for entry in conv_records.values() if entry.get("duration_ms") is not None]
        completed_entries = [entry for entry in observed_entries if entry.get("completed")]

        def pct_conv(values: List[float], p: float) -> float:
            if not values:
                return 0.0
            idx = int(round(p * (len(values) - 1)))
            return values[max(0, min(idx, len(values) - 1))]

        def compute_stats(values: List[float], suffix: str) -> None:
            values_sorted = sorted(values)
            if not values_sorted:
                return
            result[f"avg_conversation_ms{suffix}"] = sum(values_sorted) / len(values_sorted)
            result[f"p50_conversation_ms{suffix}"] = pct_conv(values_sorted, 0.50)
            result[f"p95_conversation_ms{suffix}"] = pct_conv(values_sorted, 0.95)
            result[f"p99_conversation_ms{suffix}"] = pct_conv(values_sorted, 0.99)

        if observed_entries:
            observed_durations = [entry["duration_ms"] for entry in observed_entries]
            result["conversation_count"] = len(observed_entries)
            compute_stats(observed_durations, "")

        if completed_entries:
            completed_durations = [entry["duration_ms"] for entry in completed_entries]
            result["completed_conversation_count"] = len(completed_entries)
            result["conversation_completion_rate"] = len(completed_entries) / max(1, len(observed_entries))
            compute_stats(completed_durations, "_completed")
            # Prefer completed stats for primary fields
            result["avg_conversation_ms"] = result.get("avg_conversation_ms_completed", result.get("avg_conversation_ms", 0))
            result["p50_conversation_ms"] = result.get("p50_conversation_ms_completed", result.get("p50_conversation_ms", 0))
            result["p95_conversation_ms"] = result.get("p95_conversation_ms_completed", result.get("p95_conversation_ms", 0))
            result["p99_conversation_ms"] = result.get("p99_conversation_ms_completed", result.get("p99_conversation_ms", 0))

            ttft_components: Dict[str, float] = {}
            ttft_count = 0
            decode_components: Dict[str, float] = {}
            decode_count = 0
            for entry in completed_entries:
                comp = entry.get("ttft_breakdown")
                if comp:
                    ttft_count += 1
                    for k, v in comp.items():
                        ttft_components[k] = ttft_components.get(k, 0.0) + float(v)
                comp_dec = entry.get("decode_breakdown")
                if comp_dec:
                    decode_count += 1
                    for k, v in comp_dec.items():
                        decode_components[k] = decode_components.get(k, 0.0) + float(v)

            if ttft_count:
                for k, total in ttft_components.items():
                    result[f"ttft_breakdown_{k}_avg"] = total / ttft_count
            if decode_count:
                for k, total in decode_components.items():
                    result[f"decode_breakdown_{k}_avg"] = total / decode_count

        decode_jobs = [j for j in filtered if j.job_type == "decode"]
        result["decode_jobs_count"] = len(decode_jobs)
        if decode_jobs:
            queue_vals = []
            compute_vals = []
            for job in decode_jobs:
                created = job.created_ms or 0.0
                started = job.started_ms if job.started_ms is not None else created
                finished = job.finished_ms if job.finished_ms is not None else started
                queue_vals.append(max(0.0, started - created))
                compute_vals.append(max(0.0, finished - started))
            if queue_vals:
                result["decode_breakdown_queue_ms_avg"] = sum(queue_vals) / len(queue_vals)
            if compute_vals:
                result["decode_breakdown_compute_ms_avg"] = sum(compute_vals) / len(compute_vals)
        elif observed_entries:
            result["completed_conversation_count"] = 0
            result["conversation_completion_rate"] = 0.0
        else:
            result["conversation_count"] = 0
            result["completed_conversation_count"] = 0
            result["conversation_completion_rate"] = 0.0

        self.conversations = list(conv_records.values())

        # Add RTT metrics if available
        if rtt:
            result["rtt_avg_ms"] = sum(rtt)/len(rtt)
            result["rtt_p50_ms"] = pct_rtt(0.50)
            result["rtt_p95_ms"] = pct_rtt(0.95)
            result["rtt_p99_ms"] = pct_rtt(0.99)
            result["rtt_count"] = len(rtt)

        # Add TTFT/TPOT metrics
        ttft_values = [
            entry.get("ttft_ms") for entry in conv_records.values() if entry.get("ttft_ms") is not None
        ]
        tpot_samples: List[float] = []
        for entry in conv_records.values():
            samples = entry.get("tpot_samples") or []
            tpot_samples.extend(samples)

        def percentile(values: List[float], p: float) -> float:
            if not values:
                return 0.0
            values_sorted = sorted(values)
            idx = int(round(p * (len(values_sorted) - 1)))
            return values_sorted[max(0, min(idx, len(values_sorted) - 1))]

        if ttft_values:
            result["ttft_count"] = len(ttft_values)
            ttft_sorted = sorted(ttft_values)
            result["ttft_avg_ms"] = sum(ttft_sorted) / len(ttft_sorted)
            result["ttft_p50_ms"] = percentile(ttft_sorted, 0.50)
            result["ttft_p95_ms"] = percentile(ttft_sorted, 0.95)
            result["ttft_p99_ms"] = percentile(ttft_sorted, 0.99)
        else:
            result["ttft_count"] = 0
            result["ttft_avg_ms"] = 0.0
            result["ttft_p50_ms"] = 0.0
            result["ttft_p95_ms"] = 0.0
            result["ttft_p99_ms"] = 0.0

        if tpot_samples:
            result["tpot_count"] = len(tpot_samples)
            tpot_sorted = sorted(tpot_samples)
            result["tpot_avg_ms"] = sum(tpot_sorted) / len(tpot_sorted)
            result["tpot_p50_ms"] = percentile(tpot_sorted, 0.50)
            result["tpot_p95_ms"] = percentile(tpot_sorted, 0.95)
            result["tpot_p99_ms"] = percentile(tpot_sorted, 0.99)
        else:
            result["tpot_count"] = 0
            result["tpot_avg_ms"] = 0.0
            result["tpot_p50_ms"] = 0.0
            result["tpot_p95_ms"] = 0.0
            result["tpot_p99_ms"] = 0.0

        if ttft_values and tpot_samples:
            TTFT_SLO_MS = 300.0
            TPOT_SLO_MS = 35.0
            slo_meeting = 0
            for entry in conv_records.values():
                ttft = entry.get("ttft_ms")
                samples = entry.get("tpot_samples") or []
                if ttft is None or not samples:
                    continue
                if ttft <= TTFT_SLO_MS and max(samples) <= TPOT_SLO_MS:
                    slo_meeting += 1
            total_conv = len(conv_records)
            result["slo_meeting_count"] = slo_meeting
            result["slo_attainment_rate"] = slo_meeting / total_conv if total_conv else 0.0
            if filtered:
                span = max(j.finished_ms for j in filtered) - min(j.created_ms for j in filtered) + 1e-9
                result["goodput_rps"] = 1000.0 * slo_meeting / span
            else:
                result["goodput_rps"] = 0.0
        else:
            result["slo_meeting_count"] = 0
            result["slo_attainment_rate"] = 0.0
            result["goodput_rps"] = 0.0

        return result

# ---------- Helper Functions ----------

def get_typical_verify_ms(target_config: dict, gamma: int = 4) -> float:
    """Calculate typical verification time for mixed batching or return verify_latency_ms if available."""
    if 'verify_latency_ms' in target_config:
        return target_config['verify_latency_ms']
    # For mixed batching, use decode latency as typical case (most batches are decode-only)
    decode_per_token = target_config.get('decode_latency_per_token', 9.25)
    return gamma * decode_per_token


def _prepare_trace_records(records: Sequence[TraceRecord], draft_specs: Sequence[ReplayDraftSpec], cfg: "Config") -> Sequence[TraceRecord]:
    if not records or not draft_specs:
        return records
    horizon_ms = max(1.0, float(getattr(cfg, "sim_time_ms", 0.0) or 0.0))
    replay_cfg = dict(getattr(cfg, "trace_replay", {}) or {})
    per_conv = replay_cfg.get("per_draft_conversations")
    per_rps = replay_cfg.get("per_draft_rps")

    def _filtered(rs: Sequence[TraceRecord]) -> List[TraceRecord]:
        filtered = [r for r in rs if r.arrival_ms <= horizon_ms]
        return filtered if filtered else list(rs)

    if per_conv is None and per_rps is None:
        return _filtered(records)

    total_cap = sum(max(spec.capability, 0.0) for spec in draft_specs)
    if total_cap <= 0:
        total_cap = float(len(draft_specs)) or 1.0

    if per_conv is not None:
        required = int(math.ceil(float(per_conv) * total_cap))
    else:
        horizon_sec = horizon_ms / 1000.0
        required = int(math.ceil(float(per_rps) * horizon_sec * total_cap))

    required = max(required, len(draft_specs))

    template = _filtered(records)
    if not template:
        raise TraceParseError("trace template is empty after filtering")

    seed = getattr(cfg, "seed", 0)
    generated = _generate_scaled_trace(template, required, horizon_ms, seed)
    return generated


def _generate_scaled_trace(template: Sequence[TraceRecord], count: int, horizon_ms: float, seed: int) -> List[TraceRecord]:
    if count <= 0:
        return []
    rng = random.Random(seed ^ 0xC0FFEE)
    intervals = [rng.expovariate(1.0) for _ in range(count)]
    cumulative = list(itertools.accumulate(intervals))
    total_time = cumulative[-1] if cumulative else 1.0
    scale = horizon_ms / max(total_time, 1e-9)
    arrivals = [min(horizon_ms, t * scale) for t in cumulative]

    generated: List[TraceRecord] = []
    template_len = len(template)
    for idx in range(count):
        src = template[idx % template_len]
        meta = src.metadata if isinstance(src.metadata, Mapping) else {}
        new_meta = dict(meta) if meta else {}
        if "client_id" in new_meta:
            new_meta["client_id"] = f"{new_meta['client_id']}_{idx:05d}"
        else:
            new_meta["client_id"] = f"trace_client_{idx:05d}"
        record = TraceRecord(
            arrival_ms=arrivals[idx],
            prompt_tokens=src.prompt_tokens,
            target_tokens=src.target_tokens,
            draft_id=None,
            device_tier=src.device_tier or "default",
            slo_class=src.slo_class,
            mode_hint=src.mode_hint,
            seed=src.seed,
            metadata=new_meta,
            request_id=src.request_id or f"trace_{idx:05d}",
        )
        generated.append(record)
    return generated


class TraceSchedule:
    """Distribute trace records across drafts for replay."""

    _COHORT_META_KEYS = ("client_id", "user_id", "session_id", "session", "cohort", "trace_client", "device_id")

    def __init__(
        self,
        records: Sequence[TraceRecord],
        draft_specs: Sequence[ReplayDraftSpec],
        *,
        horizon_ms: Optional[float] = None,
    ) -> None:
        if not draft_specs:
            raise TraceParseError("trace playback requires at least one draft device")
        self._by_draft: Dict[str, List[TraceRecord]] = {spec.draft_id: [] for spec in draft_specs}

        assigner = _ReplayAssigner(draft_specs, self._COHORT_META_KEYS)
        assigned = assigner.assign(records, horizon_ms=horizon_ms)
        for draft_id, recs in assigned.items():
            recs.sort(key=lambda r: r.arrival_ms)
            self._by_draft[draft_id] = recs
        self.max_arrival_ms = assigner.max_arrival_ms

    @classmethod
    def _cohort_key(cls, tier: str, record: TraceRecord, index: int) -> str:
        meta = record.metadata or {}
        if isinstance(meta, Mapping):
            for key in cls._COHORT_META_KEYS:
                if key in meta and meta[key] is not None:
                    return f"{tier}:{meta[key]}"
        if record.request_id:
            return f"{tier}:{record.request_id}"
        if record.seed is not None:
            return f"{tier}:seed:{record.seed}"
        return f"{tier}:rr:{index}"

    def for_draft(self, draft_id: str) -> Sequence[TraceRecord]:
        return list(self._by_draft.get(draft_id, []))


class _ReplayAssigner:
    """Assign draft-agnostic trace arrivals to concrete drafts."""

    def __init__(self, drafts: Sequence[ReplayDraftSpec], session_keys: Sequence[str]):
        self._drafts = list(drafts)
        self._session_keys = tuple(session_keys)
        self._session_map: Dict[str, str] = {}
        self.max_arrival_ms: float = 0.0

    def assign(
        self,
        records: Sequence[TraceRecord],
        *,
        horizon_ms: Optional[float] = None,
    ) -> Dict[str, List[TraceRecord]]:
        by_draft: Dict[str, List[TraceRecord]] = {spec.draft_id: [] for spec in self._drafts}
        spec_map = {spec.draft_id: spec for spec in self._drafts}
        groups: Dict[str, List[tuple[int, TraceRecord]]] = defaultdict(list)

        for idx, record in enumerate(records):
            if horizon_ms is not None and record.arrival_ms > horizon_ms:
                continue
            self.max_arrival_ms = max(self.max_arrival_ms, record.arrival_ms)
            if record.draft_id and record.draft_id in spec_map:
                by_draft[record.draft_id].append(record)
                continue
            tier = str(record.device_tier) if record.device_tier is not None else "__all__"
            groups[tier].append((idx, record))

        for tier, entries in groups.items():
            candidates = [spec for spec in self._drafts if spec.tier == tier]
            if not candidates:
                candidates = self._drafts
            if not candidates:
                continue
            for draft_id, rec in self._assign_group(tier, entries, candidates):
                by_draft[draft_id].append(rec)

        return by_draft

    def _assign_group(
        self,
        tier: str,
        entries: Sequence[tuple[int, TraceRecord]],
        candidates: Sequence[ReplayDraftSpec],
    ) -> Iterable[tuple[str, TraceRecord]]:
        items = sorted(entries, key=lambda x: (x[1].arrival_ms, x[0]))
        if not items:
            return []
        slots = self._build_slots(len(items), candidates)
        spec_ids = {spec.draft_id for spec in candidates}
        assignments = []
        position = 0
        for original_idx, record in items:
            cohort_key = TraceSchedule._cohort_key(tier, record, original_idx)
            assigned = None
            if cohort_key in self._session_map:
                candidate_id = self._session_map[cohort_key]
                if candidate_id in spec_ids:
                    assigned = candidate_id
            if assigned is None:
                assigned = slots[position % len(slots)]
                position += 1
                self._session_map[cohort_key] = assigned
            assignments.append((assigned, record.with_draft(assigned)))
        return assignments

    def _build_slots(self, count: int, candidates: Sequence[ReplayDraftSpec]) -> List[str]:
        if count <= 0:
            return [candidates[0].draft_id]
        weights = [max(spec.capability, 0.0) for spec in candidates]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(candidates)
            total = float(len(candidates))
        raw = [count * w / total for w in weights]
        shares = [max(0, int(math.floor(r))) for r in raw]
        assigned = sum(shares)
        fractional = [r - math.floor(r) for r in raw]
        if assigned < count:
            order = sorted(range(len(candidates)), key=lambda i: (-fractional[i], i))
            idx_iter = iter(order)
            while assigned < count:
                try:
                    idx = next(idx_iter)
                except StopIteration:
                    idx_iter = iter(order)
                    idx = next(idx_iter)
                shares[idx] += 1
                assigned += 1
        elif assigned > count:
            order = sorted(range(len(candidates)), key=lambda i: (fractional[i], i))
            idx_iter = iter(order)
            while assigned > count and order:
                try:
                    idx = next(idx_iter)
                except StopIteration:
                    idx_iter = iter(order)
                    idx = next(idx_iter)
                if shares[idx] > 0:
                    shares[idx] -= 1
                    assigned -= 1
                else:
                    break
        if sum(shares) == 0:
            shares[0] = 1
        slots: List[str] = []
        max_share = max(shares) if shares else 0
        for round_idx in range(max_share):
            for spec, share in zip(candidates, shares):
                if round_idx < share:
                    slots.append(spec.draft_id)
        if not slots:
            slots = [candidates[0].draft_id]
        return slots
# ---------- Servers ----------

class TargetServer:
    """Continuous-batching-style target: collect for Delta or until B, then serve in fixed time."""
    def __init__(self, env: simpy.Environment, params: TargetParams, metrics: Metrics,
                 performance_provider, cfg: Config = None, debug: bool = False, router=None,
                 kv_manager: Optional["KVManager"] = None):
        self.env = env
        self.p = params
        self.cluster = params.cluster
        self.cfg = cfg or Config()  # Use default config if not provided
        self.scheduler_config = self.cfg.scheduler_config  # Get scheduler config
        self.metrics = metrics
        self.debug = debug
        self.metadata = params.metadata
        self.router = router  # Reference to router for JIQ notifications
        self.q = simpy.Store(env)           # FIFO queue of Job
        self.performance = performance_provider
        self.kv_manager = kv_manager
        self.kv_capacity_tokens = getattr(params, "kv_capacity_tokens", None)
        self.execution_mode = getattr(self.cfg, "speculation_execution_mode", "distributed").lower()
        self.framework = getattr(self.cfg, "speculation_framework", "vanilla").lower()
        self.fused_draft_profile = dict(params.fused_draft_profile or {})

        # CRITICAL FIX: Add Resource to enforce single-server processing
        # This ensures batch collection and processing cannot overlap
        self.server = simpy.Resource(env, capacity=1)
        
        self._busy = False                  # coarse busy flag (for rough work-left estimate)
        self._enqueued_count = 0            # track items in queue
        self._busy_until = 0.0              # when current batch finishes
        # Per-target metrics
        self.busy_ms = 0.0
        self.total_batches = 0
        self.total_batch_items = 0
        self.queue_samples = []             # Track queue length over time
        self.max_concurrency = 0            # Track max parallelism (should always be 1)
        
        # Track average batch latency for better ETA calculations in mixed batching
        self._recent_batch_latencies = []  # Keep last N batch latencies
        self._max_latency_history = 20  # Track last 20 batches
        self._avg_batch_latency_ms = None  # Running average
        self._pending_decode_tokens = 0  # Tokens awaiting verification (queued or inflight)
        self.total_processed_tokens = 0  # Total decode tokens processed by this target
        self.total_processed_jobs = 0    # Total decode jobs processed

        # Track when prefill jobs complete at target (for target throughput measurement)
        self.prefill_completions = []  # List of (timestamp, count) tuples
        self.prefill_service_ms = 0.0   # Total service time spent on prefill jobs (ms)
        self.prefill_jobs_processed = 0  # Number of prefill jobs processed

        self.proc = env.process(self._serve_loop())

    # queue helpers
    def enqueue(self, job: Job):
        self._enqueued_count += 1
        if job.job_type != "prefill":
            self._pending_decode_tokens += max(0, int(job.token_count or 0))
        draft_id = (job.draft_id or "").strip()
        if self.p.id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
            print(
                f"[{self.env.now:.1f}ms] Target {self.p.id}: enqueue {job.job_type} from {draft_id} (queue={self._enqueued_count})",
                flush=True,
            )
        self.q.put(job)
        # Debug: print if queue is getting very large
        if self._enqueued_count > 100 and self._enqueued_count % 50 == 0:
            print(f"[{self.env.now:.1f}ms] WARNING: Target {self.p.id} queue size: {self._enqueued_count}", flush=True)

    def pending_decode_tokens(self) -> int:
        return max(0, int(self._pending_decode_tokens))

    def queue_len(self) -> int:
        # Include the job currently in service so JSQ sees busy targets
        busy = 1 if self._busy else 0
        return self._enqueued_count + busy

    def _calculate_batch_latency(self, batch: List[Job]) -> float:
        """Calculate batch processing time with optional VIDUR support."""
        provider_latency = self._estimate_batch_latency_from_provider(batch)
        if provider_latency is not None and provider_latency > 0:
            return provider_latency

        max_latency = 0.0
        for job in batch:
            est = self._estimate_job_latency(job)
            max_latency = max(max_latency, est)
        if max_latency > 0:
            return max_latency

        if not self.cfg.mixed_batching:
            return 0.0

        return 0.0

    def _estimate_job_latency(self, job: Job) -> float:
        target_latency = self._estimate_target_latency(job)
        if getattr(job, "is_fused", False):
            phase = "prefill" if job.job_type == "prefill" else "decode"
            draft_latency = self._fused_phase_latency(job, phase)
            return draft_latency + target_latency
        return target_latency

    def _estimate_target_latency(self, job: Job) -> float:
        provider = getattr(self, "performance", None)
        if provider is None:
            return 0.0
        phase = "prefill" if job.job_type == "prefill" else "verify"
        gamma_override = getattr(job, "gamma_override", 0)
        fanout_value = gamma_override if (job.job_type != "prefill" and gamma_override > 0) else (self.cfg.gamma if job.job_type != "prefill" else 1)
        request = PhaseRequest(
            phase=phase,
            model=self.p.model,
            hardware=self.p.gpu,
            batch_size=1,
            microbatch_size=1,
            fanout=max(1, fanout_value),
            sequence_length=job.context_len or job.token_count,
            tokens_to_generate=job.token_count if job.job_type != "prefill" else 0,
            context_length=job.context_len,
            request_id=job.request_id,
            target_id=self.p.id,
            draft_id=job.draft_id,
        )
        metrics = provider.get_metrics(request)
        return metrics.latency_ms if metrics else 0.0

    def _estimate_batch_latency_from_provider(self, batch: List[Job]) -> Optional[float]:
        provider = getattr(self, "performance", None)
        if provider is None or not batch:
            return None
        if any(getattr(job, "is_fused", False) for job in batch):
            return None
        phase_groups = {
            "prefill": [job for job in batch if job.job_type == "prefill"],
            "decode": [job for job in batch if job.job_type != "prefill"],
        }
        latencies = []
        for phase_name, jobs in phase_groups.items():
            if not jobs:
                continue
            request = self._make_phase_request(phase_name, jobs)
            if request is None:
                continue
            metrics = provider.get_metrics(request)
            if metrics is None:
                continue
            latencies.append(max(0.0, metrics.latency_ms))
        if latencies:
            return max(latencies)
        return None

    def _make_phase_request(self, phase: str, jobs: Sequence[Job]) -> Optional[PhaseRequest]:
        if not jobs:
            return None
        batch_size = len(jobs)
        token_counts = [max(1, int(j.token_count or 0)) for j in jobs]
        context_lengths = [max(0, int(j.context_len or 0)) for j in jobs]
        avg_tokens = sum(token_counts) / batch_size if batch_size else 0.0
        avg_context = sum(context_lengths) / batch_size if batch_size else 0.0
        max_tokens = max(token_counts) if token_counts else 0
        max_context = max(context_lengths) if context_lengths else 0

        seq_length = int(round(avg_tokens if phase == "prefill" else avg_context))
        if seq_length <= 0:
            seq_length = max_tokens if phase == "prefill" else max_context
        seq_length = max(1, seq_length)

        ctx_length = int(round(avg_context)) if avg_context > 0 else max_context
        ctx_length = max(0, ctx_length)

        prompt_tokens = int(round(avg_tokens)) if phase == "prefill" else None

        metadata = MappingProxyType({
            "phase": phase,
            "batch_size": batch_size,
            "avg_tokens": avg_tokens,
            "avg_context": avg_context,
            "max_tokens": max_tokens,
            "max_context": max_context,
        })

        if phase != "prefill":
            gamma_values = []
            for job in jobs:
                override = getattr(job, "gamma_override", 0)
                gamma_values.append(override if override > 0 else self.cfg.gamma)
            fanout_value = max(1, int(max(gamma_values) if gamma_values else self.cfg.gamma))
        else:
            fanout_value = 1

        return PhaseRequest(
            phase=phase,
            model=self.p.model,
            hardware=self.p.gpu,
            batch_size=batch_size,
            microbatch_size=min(batch_size, self.p.batch_size),
            fanout=fanout_value,
            sequence_length=seq_length,
            tokens_to_generate=max_tokens if phase != "prefill" else 0,
            context_length=ctx_length,
            target_id=self.p.id,
            draft_id=jobs[0].draft_id if jobs and jobs[0].draft_id else None,
            prompt_tokens=prompt_tokens,
            context_tokens=ctx_length if ctx_length > 0 else None,
            tokens_per_request=tuple(token_counts),
            context_per_request=tuple(context_lengths),
            extra_metadata=metadata,
        )

    def _fused_phase_latency(self, job: Job, phase: str) -> float:
        profile = self.fused_draft_profile
        if not profile or self.performance is None:
            return 0.0
        model_name = profile.get("model_name") or profile.get("model")
        hardware = profile.get("device") or profile.get("hardware") or self.p.gpu
        if not model_name or not hardware:
            return 0.0
        request = PhaseRequest(
            phase=phase,
            model=model_name,
            hardware=hardware,
            batch_size=1,
            microbatch_size=1,
            fanout=1,
            sequence_length=job.context_len if phase != "prefill" else job.token_count,
            tokens_to_generate=job.token_count if phase != "prefill" else 0,
            context_length=job.context_len,
            request_id=job.request_id,
            target_id=self.p.id,
            draft_id=job.draft_id,
        )
        metrics = self.performance.get_metrics(request)
        return metrics.latency_ms if metrics else 0.0

    def fused_execute(self, job: Job) -> simpy.Event:
        event = job.completion_event or self.env.event()

        def _run():
            if job.job_type != "prefill":
                self._pending_decode_tokens += max(0, int(job.token_count or 0))

            job.started_ms = self.env.now
            total_latency = 0.0
            phase = "prefill" if job.job_type == "prefill" else "decode"
            if self.fused_draft_profile:
                total_latency += self._fused_phase_latency(job, phase)
            total_latency += self._estimate_job_latency(job)
            if total_latency > 0:
                yield self.env.timeout(total_latency)
            job.finished_ms = self.env.now
            if self.metrics:
                self.metrics.add(job)
            if job.job_type != "prefill":
                self._pending_decode_tokens = max(0, self._pending_decode_tokens - max(0, int(job.token_count or 0)))

            if not event.triggered:
                event.succeed(job)

        self.env.process(_run())
        return event

    def work_left_score(self) -> float:
        # Better ETA-based scoring
        B = max(1, self.p.batch_size)
        batches_queued = math.ceil(self._enqueued_count / B)
        # Time until current batch finishes
        remaining_ms = max(0, self._busy_until - self.env.now) if self._busy else 0
        
        # Use average batch latency if available (for mixed batching), otherwise use typical verify time
        if self._avg_batch_latency_ms is not None:
            batch_latency = self._avg_batch_latency_ms
        else:
            batch_latency = 37.0
        
        # Total estimated time to clear queue
        eta_ms = remaining_ms + batches_queued * batch_latency
        # Normalize by capacity weight
        return eta_ms / max(self.p.weight, 1e-6)

    # serving loop
    def _serve_loop(self):
        tp = self.p
        serve_count = 0
        while True:
            # CRITICAL: Acquire server resource BEFORE collecting batch
            # This ensures only one batch is being collected/processed at a time
            with self.server.request() as req:
                yield req  # Wait for server to be available
                
                # Track concurrency (should always be 1)
                self.max_concurrency = max(self.max_concurrency, len(self.server.users))
                
                # Debug around problem time
                if self.env.now > 1020 and self.env.now < 1030:
                    print(f"[{self.env.now:.1f}ms] Target {self.p.id}: Waiting for job, queue has {self._enqueued_count} items", flush=True)
                
                first = yield self.q.get()              # wait for first job
                self._enqueued_count -= 1
                first_draft = (first.draft_id or "").strip()
                if self.p.id == "llama_t01" or first_draft in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                    print(
                        f"[{self.env.now:.1f}ms] Target {self.p.id}: picked first job {first.job_type} from {first_draft} (queue={self._enqueued_count})",
                        flush=True,
                    )
                
                # JIQ: Mark as busy when we get first job
                if self.router and hasattr(self.router, 'mark_busy'):
                    self.router.mark_busy(self.p.id)
                
                batch = [first]
                t0 = self.env.now
                serve_count += 1
                
                # Scheduler config
                max_prefills = self.scheduler_config.get("max_prefills_per_batch")
                prefill_count = 1 if first.job_type == "prefill" else 0
                deferred_prefills = []  # Prefills to save for next batch
                
                if self.env.now > 1020 and self.env.now < 1030:
                    print(f"[{self.env.now:.1f}ms] Target {self.p.id}: Got first job, starting batch collection", flush=True)
                
                # Debug trace every 50 batches
                if serve_count % 50 == 0:
                    print(f"[{self.env.now:.1f}ms] Target {self.p.id}: Served {serve_count} batches, queue={self._enqueued_count}", flush=True)

                # collect more jobs up to window or size cap
                while len(batch) < tp.batch_size:
                    remaining = tp.batch_window_ms - (self.env.now - t0)
                    
                    # CRITICAL FIX: Check for timeout BEFORE doing anything else
                    # Use a small epsilon to avoid floating point issues
                    if remaining <= 0.001:  # 0.001ms epsilon for floating point safety
                        if self.env.now > 1020 and self.env.now < 1030:
                            print(f"[{self.env.now:.1f}ms] Target {self.p.id}: Batch window expired (remaining={remaining:.3f}), serving {len(batch)} jobs", flush=True)
                        break
                    
                    if self.env.now > 1020 and self.env.now < 1030:
                        print(f"[{self.env.now:.1f}ms] Target {self.p.id}: Waiting for more jobs, {remaining:.1f}ms left in window", flush=True)
                    
                    get_ev = self.q.get()
                    timeout_ev = self.env.timeout(remaining)
                    
                    # Back-compat for AnyOf
                    try:
                        got = yield self.env.any_of([get_ev, timeout_ev])
                    except AttributeError:
                        # Fallback for older SimPy versions
                        got = yield simpy.events.AnyOf(self.env, [get_ev, timeout_ev])
                    
                    if get_ev in got:
                        job = got[get_ev]
                        self._enqueued_count -= 1
                        
                        # SCHEDULER LOGIC: Decode-first
                        if max_prefills is not None and job.job_type == "prefill":
                            if prefill_count >= max_prefills:
                                draft_id = (job.draft_id or "").strip()
                                if self.p.id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                                    print(
                                        f"[{self.env.now:.1f}ms] Target {self.p.id}: deferring prefill from {draft_id} (prefill_count={prefill_count})",
                                        flush=True,
                                    )
                                deferred_prefills.append(job)
                                continue  # Skip this prefill for now
                        
                        batch.append(job)
                        draft_id = (job.draft_id or "").strip()
                        if self.p.id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                            print(
                                f"[{self.env.now:.1f}ms] Target {self.p.id}: added {job.job_type} from {draft_id} (batch={len(batch)})",
                                flush=True,
                            )
                        if job.job_type == "prefill":
                            prefill_count += 1
                    else:
                        # timeout fired; must cancel the pending get to avoid a dangling consumer
                        # (this is recommended in SimPy's shared-resource patterns)
                        if hasattr(get_ev, 'cancel'):
                            get_ev.cancel()

                # Re-queue deferred prefills at the front
                for job in reversed(deferred_prefills):
                    # Put back at front of queue (they'll be processed first next batch)
                    yield self.q.put(job)
                    self._enqueued_count += 1
                
                # Calculate batch processing time (max of all job latencies)
                batch_latency = self._calculate_batch_latency(batch)
                
                # Update running average of batch latencies for better ETA calculations
                self._recent_batch_latencies.append(batch_latency)
                if len(self._recent_batch_latencies) > self._max_latency_history:
                    self._recent_batch_latencies.pop(0)
                self._avg_batch_latency_ms = sum(self._recent_batch_latencies) / len(self._recent_batch_latencies)
                
                # "serve" the batch
                prefill_count = sum(1 for j in batch if j.job_type == "prefill")
                if self.cfg.verbose or (self.debug and self.total_batches < 5):  # Print if verbose or first 5 batches
                    decode_count = len(batch) - prefill_count
                    print(f"[{self.env.now:.1f}ms] Target {self.p.id}: Serving batch of {len(batch)} jobs "
                          f"({prefill_count} prefill, {decode_count} decode), latency={batch_latency:.1f}ms", flush=True)

                for j in batch:
                    j.started_ms = self.env.now
                self._busy = True
                self._busy_until = self.env.now + batch_latency
                start_time = self.env.now
                
                yield self.env.timeout(batch_latency)   # batch time = max of all job latencies
                
                self._busy = False
                self.busy_ms += self.env.now - start_time
                self.total_batches += 1
                self.total_batch_items += len(batch)
                self.queue_samples.append((self.env.now, self._enqueued_count))

                # Track prefill completions for target throughput measurement
                if prefill_count > 0:
                    self.prefill_completions.append((self.env.now, prefill_count))

                tdone = self.env.now
                for j in batch:
                    j.finished_ms = tdone
                    if self.kv_manager:
                        self.kv_manager.release(self.p.id, max(0, j.kv_tokens))
                    self.metrics.add(j)
                    draft_id = (j.draft_id or "").strip()
                    if self.p.id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                        print(
                            f"[{self.env.now:.1f}ms] Target {self.p.id}: completed {j.job_type} from {draft_id} (accepted={getattr(j, 'accepted_tokens', 0)}, retry={j.retry_count})",
                            flush=True,
                        )
                    if j.job_type != "prefill":
                        self._pending_decode_tokens = max(0, self._pending_decode_tokens - max(0, int(j.token_count or 0)))
                        tokens = max(0, int(j.token_count or 0))
                        self.total_processed_tokens += tokens
                        self.total_processed_jobs += 1
                    else:
                        service_ms = self._estimate_job_latency(j)
                        if service_ms <= 0.0:
                            service_ms = max(0.0, (j.finished_ms or 0) - (j.started_ms or 0))
                        self.prefill_service_ms += service_ms
                        self.prefill_jobs_processed += 1

                    # Signal completion to waiting draft
                    if j.completion_event and not j.completion_event.triggered:
                        j.completion_event.succeed()
                    if j.chunk_barrier is not None:
                        j.chunk_barrier.notify()
                
                # JIQ: Mark as idle if queue is empty after processing
                if self.router and hasattr(self.router, 'mark_idle'):
                    if self._enqueued_count == 0:
                        self.router.mark_idle(self.p.id)
                
                # Server resource automatically released at end of with block

class DraftServer:
    """Simulates draft model generating chunks and waiting for verification (blocking mode)."""
    def __init__(self, env: simpy.Environment, params: DraftParams, cfg: Config,
                 router, global_router=None, target_lookup: Optional[Dict[str, TargetServer]] = None,
                 connections: Dict[str, ConnectionParams] = None, total_capability: float = 1.0,
                 metrics: Metrics = None, scheduler: Optional["Scheduler"] = None,
                 trace_records: Optional[Sequence[TraceRecord]] = None,
                 performance_provider=None, network: Optional[NetworkFabric] = None,
                 acceptance_model: Optional[AcceptanceRegressor] = None,
                 gamma_policy: Optional[GammaPolicy] = None):
        self.env = env
        self.p = params
        self.id = params.id
        self.cluster = params.cluster
        self.cfg = cfg
        self.router = router
        self.global_router = global_router
        self._target_lookup = target_lookup or {}
        self.connections = connections or {}  # Map of target_id -> ConnectionParams
        self.total_capability = total_capability
        self.gamma = cfg.gamma  # tokens per chunk
        self.metrics = metrics  # Reference to global metrics
        self.performance = performance_provider
        self.metadata = params.metadata
        self._acceptance_model = acceptance_model
        self._acceptance_disabled = bool(getattr(cfg, "acceptance_model_disabled", False))
        if not self._acceptance_disabled and self._acceptance_model is None:
            raise RuntimeError("Acceptance model must be provided; fallbacks are not permitted")
        self._gamma_policy = gamma_policy or ConstantGammaPolicy(self.gamma)
        self._acceptance_use_classifier = bool(getattr(cfg, "acceptance_use_classifier", True))
        self._context_bucket = max(1, int(getattr(cfg, "acceptance_context_bucket", 1) or 1))
        self._depth_bucket = max(1, int(getattr(cfg, "acceptance_depth_bucket", 1) or 1))
        self._pending_bucket = max(1, int(getattr(cfg, "acceptance_pending_bucket", 1) or 1))
        self._queue_bucket = max(1, int(getattr(cfg, "acceptance_queue_bucket", 1) or 1))
        self._acceptance_cache: Dict[Tuple[Any, ...], Tuple[float, Tuple[float, ...]]] = {}
        self.execution_mode = getattr(cfg, "speculation_execution_mode", "distributed").lower()
        self.framework = getattr(cfg, "speculation_framework", "vanilla").lower()
        self._workload_enabled = bool(cfg.workload.rate_rps > 0) if cfg and cfg.workload else False
        self._trace_records = list(trace_records) if trace_records else None
        self._trace_index = 0
        self._trace_mode = self._trace_records is not None
        self.scheduler = scheduler
        if not getattr(cfg, "network_enabled", True):
            network = None
        self.network = network
        net_cfg = getattr(cfg, "network_config", {}) or {}
        self._bytes_per_token = float(net_cfg.get("bytes_per_token", 2.0) or 0.0)
        self._prefill_overhead_bytes = float(net_cfg.get("prefill_overhead_bytes", 512.0) or 0.0)
        self._decode_overhead_bytes = float(net_cfg.get("decode_overhead_bytes", 128.0) or 0.0)
        self._response_overhead_bytes = float(net_cfg.get("response_overhead_bytes", 256.0) or 0.0)
        self._eagle_cfg = dict(getattr(cfg, "speculation_config", {}).get("eagle", {}) or {})

        self._think_enabled = self.cfg.think_time.enabled
        self._next_available_ms = self.env.now
        if not self._trace_mode:
            initial_wait = 0.0
            if self._workload_enabled:
                initial_wait = max(initial_wait, self._ia())
            if self._think_enabled:
                initial_wait = max(initial_wait, self._sample_think_time())
            self._next_available_ms = self.env.now + initial_wait

        # Metrics
        self.chunks_sent = 0
        self.total_tokens_generated = 0
        self.total_tokens_accepted = 0
        self.total_tokens_rejected = 0
        self.total_round_trip_time = 0.0
        
        self.proc = env.process(self._generate_blocking())

    def _ia(self) -> float:
        """Inter-arrival time between generation sessions"""
        wl = self.cfg.workload
        if wl.arrival == "poisson":
            # Capability-weighted rate distribution
            my_share = self.p.capability / self.total_capability
            my_rate = max(wl.rate_rps * my_share, 1e-9)
            lam = my_rate / 1000.0  # events per ms
            return random.expovariate(lam)
        # default deterministic - also capability-weighted
        base_ia = self.cfg.workload.interarrival_ms
        return base_ia * self.total_capability / self.p.capability

    def _sample_think_time(self) -> float:
        cfg = self.cfg.think_time
        if not cfg.enabled:
            return 0.0
        dist = (cfg.distribution or "workload").lower()
        if dist == "workload":
            return max(0.0, self._ia())
        if dist == "exponential":
            mean = max(cfg.mean_ms, 1e-6)
            return max(cfg.min_ms, random.expovariate(1.0 / mean))
        if dist == "constant":
            return max(cfg.min_ms, cfg.mean_ms)
        if dist == "lognormal":
            mean = max(cfg.mean_ms, 1e-6)
            cv = max(cfg.cv, 1e-6)
            sigma = math.sqrt(math.log(1.0 + cv * cv))
            mu = math.log(mean) - 0.5 * sigma * sigma
            value = random.lognormvariate(mu, sigma)
            return max(cfg.min_ms, value)
        raise ValueError(f"Unsupported think-time distribution: {cfg.distribution}")

    def _get_eagle_param(self, key: str, default: Any) -> Any:
        return self._eagle_cfg.get(key, default)

    def _schedule_next_arrival(self, reference_ms: float) -> None:
        wait = 0.0
        if self._workload_enabled:
            wait = max(wait, self._ia())
        if self._think_enabled:
            wait = max(wait, self._sample_think_time())
        # Cap next arrival at sim_time_ms to prevent scheduling beyond simulation end
        self._next_available_ms = min(reference_ms + wait, self.cfg.sim_time_ms)

    def _select_target(self) -> tuple[str, ConnectionParams]:
        """Select a target among only those this draft can reach (connection-aware).
        
        THIS is where routing actually happens (not router.route()):
        1. Filter to reachable targets (allowed_ids from connections)
        2. Use router's weighted sampler to pick d=2 candidates
        3. Select the least loaded candidate
        4. Return target_id and connection params (latencies, acceptance rate)
        """
        allowed_ids = set(self.connections.keys())
        if not allowed_ids:
            raise ValueError(f"Draft {self.id} has no connections configured")

        # Try global router first for cross-cluster awareness
        if self.global_router is not None:
            global_choice = self.global_router.choose(self.id, allowed_ids)
            if global_choice is not None:
                target_id = global_choice.p.id
                if target_id in self.connections:
                    return target_id, self.connections[target_id]

        # Candidate pool restricted to allowed targets present in the router
        pool = [t for t in getattr(self.router, 'targets', []) if t.p.id in allowed_ids]
        if not pool:
            # Fallback: pick any allowed id if targets not yet attached
            tid = next(iter(allowed_ids))
            return tid, self.connections[tid]

        # Handle different router types
        if hasattr(self.router, 'random_select_filtered'):
            # Random router - no load awareness
            target = self.router.random_select_filtered(allowed_ids)
            if not target:
                target = pool[0]
            target_id = target.p.id
            return target_id, self.connections[target_id]
        elif hasattr(self.router, 'round_robin_select_filtered'):
            # Round-robin router - cycles through targets
            target = self.router.round_robin_select_filtered(self.id, allowed_ids)
            if not target:
                target = pool[0]
            target_id = target.p.id
            return target_id, self.connections[target_id]
        elif hasattr(self.router, 'jiq_select_filtered'):
            # JIQ router - selects idle targets from FIFO queue
            target = self.router.jiq_select_filtered(allowed_ids)
            if not target:
                target = pool[0]
            target_id = target.p.id
            return target_id, self.connections[target_id]
        elif hasattr(self.router, 'jsq_select_filtered'):
            target = self.router.jsq_select_filtered(allowed_ids)
            if not target:
                target = pool[0]
            target_id = target.p.id
            return target_id, self.connections[target_id]
        elif hasattr(self.router, 'semi_clairvoyant_select_filtered'):
            # Semi-Clairvoyant router - fairness based on progress
            target = self.router.semi_clairvoyant_select_filtered(self.id, allowed_ids)
            if not target:
                target = pool[0]
            target_id = target.p.id
            return target_id, self.connections[target_id]
        elif hasattr(self.router, '_weighted_sample_k_filtered'):
            # Weighted JSQ(d) router
            candidates = self.router._weighted_sample_k_filtered(getattr(self.router, 'd', 2), allowed_ids)
            if not candidates:
                candidates = pool
        elif hasattr(self.router, 'd') and isinstance(self.router, JSQ2Router):
            # Unweighted JSQ(d) router
            k = min(self.router.d, len(pool))
            candidates = random.sample(pool, k) if len(pool) > k else pool
        else:
            # Default: use all available
            candidates = pool

        # For non-random routers, select least loaded from candidates
        target = min(candidates, key=lambda t: t.work_left_score())
        target_id = target.p.id
        return target_id, self.connections[target_id]
    
    def _quantize_value(self, value: float, bucket: int) -> float:
        if bucket <= 1:
            return value
        return round(value / bucket) * bucket

    def _acceptance_cache_key(
        self,
        target_id: str,
        depth_q: int,
        context_q: float,
        pending_q: int,
        queue_q: int,
    ) -> Tuple[Any, ...]:
        return (
            target_id,
            depth_q,
            context_q,
            pending_q,
            queue_q,
            bool(self._acceptance_use_classifier),
        )

    def _build_acceptance_feature_context(self, target_id: str, spec_tokens: int) -> Dict[str, Any]:
        target = self._target_lookup.get(target_id)
        verifier_metadata = getattr(target.p, "metadata", {}) if target is not None else {}
        verifier_model = None
        if target is not None:
            verifier_model = target.p.model or verifier_metadata.get("model") or verifier_metadata.get("model_name")
        drafter_metadata = self.metadata or {}
        drafter_model = drafter_metadata.get("model") or drafter_metadata.get("model_name") or self.id
        pending_tokens = 0
        queue_depth = 0
        target = self._target_lookup.get(target_id)
        if target is not None:
            try:
                pending_tokens = target.pending_decode_tokens()
                queue_depth = target.queue_len()
            except AttributeError:
                pending_tokens = 0
                queue_depth = 0
        spec_tokens = int(max(1, spec_tokens))
        if self._depth_bucket > 1:
            spec_tokens = int(max(1, self._quantize_value(spec_tokens, self._depth_bucket)))
        pending_tokens = int(max(0, pending_tokens))
        queue_depth = int(max(0, queue_depth))
        if self._pending_bucket > 1:
            pending_tokens = int(max(0, self._quantize_value(pending_tokens, self._pending_bucket)))
        if self._queue_bucket > 1:
            queue_depth = int(max(0, self._quantize_value(queue_depth, self._queue_bucket)))
        return {
            "spec_tokens": spec_tokens,
            "drafter_model": drafter_model,
            "verifier_model": verifier_model or target_id,
            "pending_decode_tokens": int(pending_tokens),
            "target_queue_depth": int(queue_depth),
        }

    def _effective_acceptance_rate(self, conn: ConnectionParams, context_length: int, depth: int) -> float:
        if depth <= 0:
            return 0.0
        rate, _ = self._lookup_acceptance(conn, context_length, depth)
        return rate


    def _lookup_acceptance(self, conn: ConnectionParams, context_length: float, tokens: int) -> Tuple[float, Tuple[float, ...]]:
        tokens = int(max(1, tokens))
        if self._acceptance_disabled or self._acceptance_model is None:
            base = conn.acceptance_rate if conn and conn.acceptance_rate is not None else self.cfg.acceptance_config.get("default_rate", 0.75)
            rate = max(0.0, min(1.0, float(base)))
            return rate, tuple(rate for _ in range(tokens))

        depth_q = int(max(1, self._quantize_value(tokens, self._depth_bucket)))
        context_q = self._quantize_value(context_length, self._context_bucket)
        feature_context = self._build_acceptance_feature_context(conn.target_id, depth_q)
        pending_q = int(feature_context.get("pending_decode_tokens", 0))
        queue_q = int(feature_context.get("target_queue_depth", 0))
        key = self._acceptance_cache_key(conn.target_id, depth_q, context_q, pending_q, queue_q)

        cache_entry = self._acceptance_cache.get(key)
        if cache_entry is None:
            if _GLOBAL_PROFILER is not None:
                _GLOBAL_PROFILER["acceptance_cache_misses"] += 1

            # Skip expensive regressor when using classifier
            if self._acceptance_use_classifier:
                # Use default rate as fallback, classifier will provide actual probabilities
                rate = self.cfg.acceptance_config.get("default_rate", 0.75)
            else:
                # Only use regressor when classifier is disabled
                start = time.perf_counter()
                expected, used_surrogate = self._acceptance_model.predict_expected_accepts(
                    float(context_q),
                    feature_context=feature_context,
                )
                reg_duration_ms = (time.perf_counter() - start) * 1000.0
                if _GLOBAL_PROFILER is not None:
                    _GLOBAL_PROFILER["acceptance_calls"] += 1
                    if used_surrogate:
                        _GLOBAL_PROFILER["acceptance_surrogate_queries"] += 1
                    else:
                        _GLOBAL_PROFILER["acceptance_regressor_ms"] += reg_duration_ms
                rate = expected / max(1.0, float(depth_q))
                rate = max(0.0, min(1.0, float(rate)))

            if self._acceptance_use_classifier:
                start = time.perf_counter()
                probabilities = self._acceptance_model.position_probabilities(
                    context_length=float(context_q),
                    depth=depth_q,
                    default=rate,
                    feature_context=feature_context,
                )
                cls_duration_ms = (time.perf_counter() - start) * 1000.0
                if _GLOBAL_PROFILER is not None:
                    _GLOBAL_PROFILER["acceptance_classifier_ms"] += cls_duration_ms
                    _GLOBAL_PROFILER["acceptance_proba_ms"] += cls_duration_ms
                probabilities = tuple(max(0.0, min(1.0, float(p))) for p in probabilities[:depth_q])
            else:
                probabilities = tuple(rate for _ in range(depth_q))

            cache_entry = (rate, probabilities)
            self._acceptance_cache[key] = cache_entry
        else:
            if _GLOBAL_PROFILER is not None:
                _GLOBAL_PROFILER["acceptance_cache_hits"] += 1
                _GLOBAL_PROFILER["acceptance_cached"] += 1

        rate, base_probs = cache_entry
        if tokens <= len(base_probs):
            probs = base_probs[:tokens]
        else:
            probs = tuple(base_probs) + tuple(rate for _ in range(tokens - len(base_probs)))
        return rate, probs


    def _simulate_verification(self, tokens: int, conn: ConnectionParams, context_length: int) -> VerifyResult:
        """Simulate target verification of draft tokens."""
        tokens = max(0, int(tokens))
        if tokens == 0:
            return VerifyResult(chunk_id=self.chunks_sent, accepted_tokens=0, rejected_tokens=0, total_tokens=0)
        accepted = 0
        rate, probabilities = self._lookup_acceptance(conn, context_length, tokens)
        for prob in probabilities:
            prob = max(0.0, min(1.0, float(prob)))
            if random.random() < prob:
                accepted += 1
            else:
                break
        rejected = tokens - accepted
        return VerifyResult(
            chunk_id=self.chunks_sent,
            accepted_tokens=accepted,
            rejected_tokens=rejected,
            total_tokens=tokens,
        )

    def _predict_generation_latency(self, tokens: int, context_length: int) -> float:
        """Use the performance provider to estimate local draft generation time."""
        if self.performance is None or tokens <= 0:
            return 0.0

        metadata = self.metadata or {}
        profile = metadata.get("vidur_profile") or metadata.get("vidur") or {}
        model = metadata.get("model") or profile.get("model_name")
        hardware = metadata.get("gpu") or profile.get("device")

        if not model and not hardware and not profile:
            return 0.0

        request = PhaseRequest(
            phase="decode",
            model=model,
            hardware=hardware,
            batch_size=1,
            microbatch_size=1,
            fanout=max(1, tokens),
            sequence_length=max(1, context_length),
            tokens_to_generate=max(1, tokens),
            context_length=max(0, context_length),
            draft_id=self.id,
            target_id=self.id,
            extra_metadata=metadata,
        )
        metrics = self.performance.get_metrics(request)
        if metrics is None:
            return 0.0
        return max(0.0, metrics.latency_ms)
    
    def _sample_prompt_length(self) -> int:
        """Sample prompt length based on device capability"""
        if self.cfg.prompt_scale_by_capability:
            # Weaker devices get shorter prompts
            # Use capability directly (assumed to be in [0,1] range), not normalized by total
            # This ensures prompt lengths don't change as we add more drafts
            normalized_cap = min(1.0, self.p.capability)  # Cap at 1.0 for safety
            length_range = self.cfg.prompt_length_max - self.cfg.prompt_length_min
            prompt_length = int(self.cfg.prompt_length_min + normalized_cap * length_range)
        else:
            # For homogeneous experiments with fixed prompt length (min == max),
            # this gives a consistent prompt length
            if self.cfg.prompt_length_min == self.cfg.prompt_length_max:
                prompt_length = self.cfg.prompt_length_min
            else:
                # Uniform random between min and max
                prompt_length = random.randint(self.cfg.prompt_length_min, self.cfg.prompt_length_max)
        return prompt_length
    
    def _sample_answer_length(self) -> int:
        """Sample answer length, using normal distribution if configured."""
        if self.cfg.use_answer_distribution:
            # Sample from normal distribution
            length = int(random.gauss(self.cfg.answer_length_mean, self.cfg.answer_length_std))
            # Clamp to min/max bounds
            return max(self.cfg.answer_length_min, min(self.cfg.answer_length_max, length))
        else:
            # Use fixed length
            return self.cfg.answer_length

    def _estimate_payload_bytes(self, tokens: int, overhead_bytes: float) -> float:
        if self.network is None:
            return 0.0
        tokens = max(0, int(tokens))
        return max(0.0, float(overhead_bytes) + tokens * self._bytes_per_token)

    def _network_transfer(
        self,
        source_id: str,
        target_id: str,
        latency_ms: float,
        *,
        payload_bytes: float,
        link_key: Optional[Tuple[str, str]],
    ):
        if self.network is None:
            return self.env.timeout(max(0.0, float(latency_ms)))
        return self.network.transfer(
            source_id,
            target_id,
            payload_bytes=payload_bytes,
            link_key=link_key,
            fallback_latency_ms=float(latency_ms),
        )

    def _estimate_branch_score(self, depth: int, conn: ConnectionParams, context_length: int) -> float:
        base = self._effective_acceptance_rate(conn, context_length, depth)
        return base ** max(1, depth)

    def _run_eagle_conversation(
        self,
        conversation_id: str,
        conversation_start: float,
        prompt_length: int,
        answer_length: int,
        target_id: str,
        conn: ConnectionParams,
        priority_class: str,
        trace_record: Optional[TraceRecord],
    ) -> None:
        ttft_breakdown = {
            "prefill_forward_ms": 0.0,
            "prefill_queue_ms": 0.0,
            "prefill_compute_ms": 0.0,
            "prefill_response_ms": 0.0,
            "decode_generation_ms": 0.0,
            "decode_forward_ms": 0.0,
            "decode_queue_ms": 0.0,
            "decode_compute_ms": 0.0,
            "decode_response_ms": 0.0,
        }
        decode_breakdown = {
            "generation_ms": 0.0,
            "forward_ms": 0.0,
            "queue_ms": 0.0,
            "compute_ms": 0.0,
            "response_ms": 0.0,
        }

        fused_mode = self.execution_mode == "fused"
        target_server = self._target_lookup.get(target_id)
        if fused_mode and target_server is None:
            print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
            return

        prefill_completion = self.env.event()
        prefill_job = Job(
            jid=self.chunks_sent,
            created_ms=self.env.now,
            draft_id=self.id,
            job_type="prefill",
            token_count=prompt_length,
            completion_event=prefill_completion,
            rtt_start_ms=self.env.now,
            request_id=conversation_id,
            context_len=prompt_length,
            target_id=target_id,
            priority_class=priority_class,
            phase="prefill",
            is_fused=fused_mode,
        )
        self.chunks_sent += 1

        if fused_mode:
            forward_elapsed = 0.0
        else:
            prefill_payload = self._estimate_payload_bytes(prompt_length, self._prefill_overhead_bytes)
            fwd_start = self.env.now
            yield self._network_transfer(
                self.id,
                target_id,
                conn.forward_latency_ms,
                payload_bytes=prefill_payload,
                link_key=conn.network_forward_key,
            )
            forward_elapsed = self.env.now - fwd_start
        ttft_breakdown["prefill_forward_ms"] += forward_elapsed
        prefill_job.created_ms = self.env.now

        if self.scheduler is None:
            if target_server is None:
                print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
                return
            if fused_mode:
                wait_event = target_server.fused_execute(prefill_job)
            else:
                target_server.enqueue(prefill_job)
                wait_event = prefill_completion
        else:
            wait_event = self.scheduler.submit_job(prefill_job)

        yield wait_event

        if fused_mode:
            response_elapsed = 0.0
        else:
            rsp_start = self.env.now
            yield self._network_transfer(
                target_id,
                self.id,
                conn.response_latency_ms,
                payload_bytes=self._estimate_payload_bytes(prompt_length, self._response_overhead_bytes),
                link_key=conn.network_response_key,
            )
            response_elapsed = self.env.now - rsp_start
        ttft_breakdown["prefill_response_ms"] += response_elapsed
        prefill_job.rtt_end_ms = self.env.now

        prefill_started = prefill_job.started_ms if prefill_job.started_ms is not None else prefill_job.created_ms
        prefill_queue = max(0.0, (prefill_started - prefill_job.created_ms) if prefill_started is not None else 0.0)
        prefill_compute = max(0.0, (prefill_job.finished_ms - prefill_started) if prefill_started is not None and prefill_job.finished_ms is not None else 0.0)
        ttft_breakdown["prefill_queue_ms"] += prefill_queue
        ttft_breakdown["prefill_compute_ms"] += prefill_compute

        beam_width = max(1, int(self._get_eagle_param("beam_width", 4)))
        max_depth = max(1, int(self._get_eagle_param("max_depth", self.gamma)))
        min_depth = max(1, int(self._get_eagle_param("min_depth", 1)))
        prune_threshold = float(self._get_eagle_param("prune_score_threshold", 0.0))
        load_control_cfg = dict(self._eagle_cfg.get("load_control", {}) or {})

        tokens_generated = 0
        tokens_accepted = 0
        current_context_len = prompt_length
        conversation_tpot_samples: List[float] = []
        ttft_pending = True
        first_token_time_ms = None
        conversation_completed = False

        while tokens_accepted < answer_length and not conversation_completed:
            beam_current = beam_width
            max_depth_current = max_depth
            pending_tokens = target_server.pending_decode_tokens() if target_server is not None else 0
            queue_depth = target_server.queue_len() if target_server is not None else 0
            depth_scale = 1.0
            beam_ratio = 1.0
            if load_control_cfg.get('enabled'):
                tokens_low = float(load_control_cfg.get('tokens_low', 256.0))
                tokens_high = float(load_control_cfg.get('tokens_high', max(tokens_low * 2.0, tokens_low + 1.0)))
                min_depth_scale = float(load_control_cfg.get('min_depth_scale', 0.5))
                min_depth_scale = max(0.0, min(1.0, min_depth_scale))
                min_beam_width = max(1, int(load_control_cfg.get('min_beam_width', 1)))
                if tokens_high <= tokens_low:
                    beam_ratio = 0.0 if pending_tokens >= tokens_low else 1.0
                else:
                    if pending_tokens <= tokens_low:
                        beam_ratio = 1.0
                    elif pending_tokens >= tokens_high:
                        beam_ratio = 0.0
                    else:
                        beam_ratio = (tokens_high - pending_tokens) / (tokens_high - tokens_low)
                depth_scale = min_depth_scale + (1.0 - min_depth_scale) * beam_ratio
                max_depth_current = max(min_depth, max(1, int(max_depth * depth_scale)))
                if beam_width > min_beam_width:
                    beam_current = max(min_beam_width, int(round(min_beam_width + (beam_width - min_beam_width) * beam_ratio)))
                else:
                    beam_current = beam_width
            tokens_remaining = answer_length - tokens_accepted
            branch_depth = min(max_depth_current, tokens_remaining)
            if branch_depth < min_depth:
                branch_depth = tokens_remaining

            raw_candidates: List[BranchCandidate] = []
            candidates: List[BranchCandidate] = []
            generation_latency_total = 0.0
            pruned_candidates = 0
            for idx in range(beam_current):
                if fused_mode:
                    generation_latency = 0.0
                else:
                    generation_latency = self._predict_generation_latency(branch_depth, current_context_len)
                if generation_latency > 0:
                    yield self.env.timeout(generation_latency)
                generation_latency_total += generation_latency
                score = self._estimate_branch_score(branch_depth, conn, current_context_len)
                branch = BranchCandidate(idx, branch_depth, score)
                raw_candidates.append(branch)
                if prune_threshold > 0.0 and score < prune_threshold:
                    pruned_candidates += 1
                else:
                    candidates.append(branch)
                self.total_tokens_generated += branch_depth
                tokens_generated += branch_depth
            if not candidates and raw_candidates:
                fallback = max(raw_candidates, key=lambda c: c.score)
                candidates.append(fallback)
                if self.cfg.verbose or self.cfg.debug:
                    print(
                        f"[{self.env.now:.1f}ms] Draft {self.id}: pruning removed all candidates, using fallback branch (score={fallback.score:.4f})",
                        flush=True,
                    )
            decode_breakdown["generation_ms"] += generation_latency_total
            candidates.sort(key=lambda c: c.score, reverse=True)
            if (self.cfg.verbose or self.cfg.debug) and pruned_candidates > 0:
                print(
                    f"[{self.env.now:.1f}ms] Draft {self.id}: pruned {pruned_candidates}/{beam_current} branches (pending_tokens={pending_tokens}, queue={queue_depth})",
                    flush=True,
                )

            branch_selected = False
            for candidate in candidates:
                branch_start = self.env.now
                job = Job(
                    jid=self.chunks_sent,
                    created_ms=self.env.now,
                    draft_id=self.id,
                    job_type="decode",
                    token_count=candidate.depth,
                    completion_event=self.env.event(),
                    rtt_start_ms=branch_start,
                    request_id=conversation_id,
                    context_len=prompt_length + tokens_generated + candidate.depth,
                    target_id=target_id,
                    priority_class=priority_class,
                    phase="decode",
                    is_fused=fused_mode,
                )
                self.chunks_sent += 1

                if self.scheduler is None:
                    target = self._target_lookup.get(target_id)
                    if target is None:
                        print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
                        conversation_completed = True
                        break
                    if fused_mode:
                        wait_event = target.fused_execute(job)
                    else:
                        target.enqueue(job)
                        wait_event = job.completion_event
                else:
                    wait_event = self.scheduler.submit_job(job)

                if not fused_mode:
                    fwd_start = self.env.now
                    yield self._network_transfer(
                        self.id,
                        target_id,
                        conn.forward_latency_ms,
                        payload_bytes=self._estimate_payload_bytes(candidate.depth, self._decode_overhead_bytes),
                        link_key=conn.network_forward_key,
                    )
                    forward_elapsed = self.env.now - fwd_start
                else:
                    forward_elapsed = 0.0
                decode_breakdown["forward_ms"] += forward_elapsed
                if ttft_pending:
                    ttft_breakdown["decode_forward_ms"] += forward_elapsed

                yield wait_event

                queue_elapsed = 0.0
                compute_elapsed = 0.0
                if job.started_ms is not None and job.created_ms is not None:
                    queue_elapsed = max(0.0, job.started_ms - job.created_ms)
                if job.finished_ms is not None and job.started_ms is not None:
                    compute_elapsed = max(0.0, job.finished_ms - job.started_ms)
                decode_breakdown["queue_ms"] += queue_elapsed
                decode_breakdown["compute_ms"] += compute_elapsed
                if ttft_pending:
                    ttft_breakdown["decode_queue_ms"] += queue_elapsed
                    ttft_breakdown["decode_compute_ms"] += compute_elapsed

                if not fused_mode:
                    rsp_start = self.env.now
                    yield self._network_transfer(
                        target_id,
                        self.id,
                        conn.response_latency_ms,
                        payload_bytes=self._estimate_payload_bytes(candidate.depth, self._response_overhead_bytes),
                        link_key=conn.network_response_key,
                    )
                    response_elapsed = self.env.now - rsp_start
                else:
                    response_elapsed = 0.0
                decode_breakdown["response_ms"] += response_elapsed
                if ttft_pending:
                    ttft_breakdown["decode_response_ms"] += response_elapsed

                result = self._simulate_verification(
                    candidate.depth,
                    conn,
                    prompt_length + tokens_generated + candidate.depth)
                self.total_tokens_accepted += result.accepted_tokens
                self.total_tokens_rejected += result.rejected_tokens
                if hasattr(self.metrics, 'token_metrics'):
                    tm = self.metrics.token_metrics
                    tm.total_generated_tokens += candidate.depth
                    tm.total_accepted_tokens += result.accepted_tokens
                    tm.total_rejected_tokens += result.rejected_tokens

                rtt = self.env.now - branch_start
                self.total_round_trip_time += rtt
                if result.accepted_tokens > 0:
                    per_token = rtt / result.accepted_tokens
                    conversation_tpot_samples.extend([per_token] * result.accepted_tokens)
                    if first_token_time_ms is None:
                        first_token_time_ms = self.env.now
                        ttft_pending = False
                    tokens_accepted += result.accepted_tokens
                    current_context_len += result.accepted_tokens
                    branch_selected = True
                    break
                else:
                    conversation_completed = True

            if not branch_selected and not conversation_completed:
                conversation_completed = True

        conversation_time = self.env.now - conversation_start
        acceptance_ratio = (
            tokens_accepted / tokens_generated if tokens_generated > 0 else 0.0
        )

        if self.cfg.verbose or self.cfg.debug:
            print(
                f"[{self.env.now:.1f}ms] Draft {self.id}: EAGLE conversation completed - "
                f"prompt={prompt_length}, accepted={tokens_accepted}/{answer_length}, "
                f"time={conversation_time:.1f}ms, acceptance={acceptance_ratio:.2%}",
                flush=True,
            )

        if hasattr(self.metrics, "record_conversation"):
            ttft_ms = (
                (first_token_time_ms - conversation_start)
                if first_token_time_ms is not None
                else None
            )
            self.metrics.record_conversation(
                conversation_id,
                start_ms=conversation_start,
                end_ms=self.env.now,
                draft_id=self.id,
                target_id=target_id,
                tokens_generated=tokens_generated,
                tokens_accepted=tokens_accepted,
                answer_tokens=answer_length,
                ttft_ms=ttft_ms,
                tpot_samples=conversation_tpot_samples,
                ttft_breakdown=ttft_breakdown,
                decode_breakdown=decode_breakdown,
            )

    
    def _generate_blocking(self):
        """Blocking mode: generate full conversations with multiple speculation rounds"""
        if self.cfg.verbose:
            my_share = self.p.capability / self.total_capability
            my_rate = self.cfg.workload.rate_rps * my_share
            mode = "trace replay" if self._trace_mode else "blocking mode"
            print(
                f"[{self.env.now:.1f}ms] Draft {self.id} starting {mode} (gamma={self.gamma}, rate={my_rate:.1f} req/s)",
                flush=True,
            )

        conversation_count = 0
        records = self._trace_records or []
        if self.id in ["llama_d000", "llama_d001"] and records:
            print(f"[{self.env.now:.1f}ms] Draft {self.id}: Starting with {len(records)} trace records", flush=True)

        while self.env.now < self.cfg.sim_time_ms:
            trace_record: Optional[TraceRecord] = None
            if self._trace_mode:
                if self._trace_index >= len(records):
                    if self.id in ["llama_d000", "llama_d001"]:
                        print(f"[{self.env.now:.1f}ms] Draft {self.id}: No more trace records (index={self._trace_index})", flush=True)
                    break
                if self.id in ["llama_d000", "llama_d001"]:
                    print(f"[{self.env.now:.1f}ms] Draft {self.id}: fetching trace record #{self._trace_index}", flush=True)
                record = records[self._trace_index]
                self._trace_index += 1
                if record.draft_id and record.draft_id != self.id:
                    continue  # Skip mismatched records
                scheduled = max(0.0, record.arrival_ms)
                if self._think_enabled:
                    scheduled = max(scheduled, self._next_available_ms)
                if scheduled > self.env.now:
                    yield self.env.timeout(scheduled - self.env.now)
                if self._think_enabled:
                    self._next_available_ms = self.env.now
                conversation_start = self.env.now
                prompt_length = max(1, record.prompt_tokens)
                answer_length = max(1, record.target_tokens)
                request_id = record.request_id
                trace_record = record
                if self.id in ["llama_d000", "llama_d001"]:
                    print(f"[{self.env.now:.1f}ms] Draft {self.id}: Starting conversation with prompt={prompt_length}, answer={answer_length}", flush=True)
            else:
                scheduled = max(self._next_available_ms, self.env.now)
                if scheduled > self.env.now:
                    yield self.env.timeout(scheduled - self.env.now)
                conversation_start = self.env.now
                prompt_length = self._sample_prompt_length()
                answer_length = self._sample_answer_length()
                request_id = None

            conversation_count += 1
            conversation_id = request_id or f"{self.id}_conv{conversation_count}"

            conversation_target_id, conversation_conn = self._select_target()
            target_id = conversation_target_id
            conn = conversation_conn

            priority_class = trace_record.slo_class if trace_record is not None else None
            if not priority_class:
                priority_class = (self.cfg.trace_defaults or {}).get("slo_class")
            if not priority_class and self.scheduler is not None:
                priority_class = self.scheduler.default_priority_class
            priority_class = priority_class or "standard"

            if self.framework == "eagle":
                yield from self._run_eagle_conversation(
                    conversation_id,
                    conversation_start,
                    prompt_length,
                    answer_length,
                    target_id,
                    conn,
                    priority_class,
                    trace_record,
                )
                if not self._trace_mode:
                    self._schedule_next_arrival(self.env.now)
                elif self._trace_mode and self._think_enabled:
                    # In trace mode with think time, update _next_available_ms for the next trace record
                    think_time = self._sample_think_time()
                    self._next_available_ms = self.env.now + think_time
                continue

            ttft_breakdown = {
                "prefill_forward_ms": 0.0,
                "prefill_queue_ms": 0.0,
                "prefill_compute_ms": 0.0,
                "prefill_response_ms": 0.0,
                "decode_generation_ms": 0.0,
                "decode_forward_ms": 0.0,
                "decode_queue_ms": 0.0,
                "decode_compute_ms": 0.0,
                "decode_response_ms": 0.0,
            }
            decode_breakdown = {
                "generation_ms": 0.0,
                "forward_ms": 0.0,
                "queue_ms": 0.0,
                "compute_ms": 0.0,
                "response_ms": 0.0,
            }
            ttft_pending = True

            if hasattr(self.metrics, "connection_counts"):
                self.metrics.connection_counts[(self.id, target_id)] += 1

            if self.cfg.debug:
                label = f" req_id={request_id}" if request_id else ""
                print(
                    f"[{self.env.now:.1f}ms] Draft {self.id}: Starting conversation #{conversation_count}"
                    f" (prompt={prompt_length} tokens, answer={answer_length} tokens) with target {target_id}{label}",
                    flush=True,
                )

            target_server = self._target_lookup.get(target_id)
            fused_mode = self.framework == "vanilla" and self.execution_mode == "fused"

            prefill_rtt_start = self.env.now
            prefill_completion = self.env.event()
            prefill_job = Job(
                jid=self.chunks_sent,
                created_ms=self.env.now,
                draft_id=self.id,
                job_type="prefill",
                token_count=prompt_length,
                completion_event=prefill_completion,
                rtt_start_ms=prefill_rtt_start,
                request_id=conversation_id,
                context_len=prompt_length,
                target_id=target_id,
                priority_class=priority_class,
                phase="prefill",
                is_fused=fused_mode,
            )
            self.chunks_sent += 1

            # Send prefill request to target
            if fused_mode:
                forward_elapsed = 0.0
            else:
                prefill_payload = self._estimate_payload_bytes(prompt_length, self._prefill_overhead_bytes)
                fwd_start = self.env.now
                yield self._network_transfer(
                    self.id,
                    target_id,
                    conn.forward_latency_ms,
                    payload_bytes=prefill_payload,
                    link_key=conn.network_forward_key,
                )
                forward_elapsed = self.env.now - fwd_start
            ttft_breakdown["prefill_forward_ms"] += forward_elapsed
            prefill_job.created_ms = self.env.now

            if self.scheduler is None:
                if target_server is None:
                    print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
                    continue
                if fused_mode:
                    wait_event = target_server.fused_execute(prefill_job)
                else:
                    target_server.enqueue(prefill_job)
                    wait_event = prefill_completion
            else:
                wait_event = self.scheduler.submit_job(prefill_job)

            # Wait for actual job completion instead of synthetic sleep
            yield wait_event

            # Wait for response to travel back
            if fused_mode:
                response_elapsed = 0.0
            else:
                response_payload = self._estimate_payload_bytes(prompt_length, self._response_overhead_bytes)
                rsp_start = self.env.now
                yield self._network_transfer(
                    target_id,
                    self.id,
                    conn.response_latency_ms,
                    payload_bytes=response_payload,
                    link_key=conn.network_response_key,
                )
                response_elapsed = self.env.now - rsp_start
            ttft_breakdown["prefill_response_ms"] += response_elapsed

            # Mark RTT end for prefill
            prefill_job.rtt_end_ms = self.env.now

            prefill_started = prefill_job.started_ms if prefill_job.started_ms is not None else prefill_job.created_ms
            prefill_queue = max(0.0, (prefill_started - prefill_job.created_ms) if prefill_started is not None else 0.0)
            prefill_compute = max(0.0, (prefill_job.finished_ms - prefill_started) if prefill_started is not None and prefill_job.finished_ms is not None else 0.0)
            ttft_breakdown["prefill_queue_ms"] += prefill_queue
            ttft_breakdown["prefill_compute_ms"] += prefill_compute

            if self.cfg.debug:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Prefill completed for {prompt_length} tokens", flush=True)

            # Phase 2: Generate answer with multiple speculation rounds
            candidate_depth = self.gamma
            try:
                candidate_depth = int(self._gamma_policy.required_depth(self.gamma))
            except Exception:
                candidate_depth = self.gamma
            acceptance_probs: Tuple[float, ...] = tuple()
            if candidate_depth and candidate_depth > 0:
                candidate_depth = max(1, min(candidate_depth, 512))
                if not self._acceptance_disabled and self._acceptance_model is not None:
                    _, probs = self._lookup_acceptance(conn, prompt_length, candidate_depth)
                    acceptance_probs = tuple(probs)
                else:
                    rate = self._effective_acceptance_rate(conn, prompt_length, candidate_depth)
                    acceptance_probs = tuple(rate for _ in range(candidate_depth))
            queue_depth = target_server.queue_len() if target_server is not None else 0
            gamma_context = GammaContext(
                draft_id=self.id,
                target_id=target_id,
                context_length=prompt_length,
                queue_depth=queue_depth,
                acceptance_probabilities=acceptance_probs,
            )
            try:
                selected_gamma = self._gamma_policy.select_gamma(self.id, self.gamma, gamma_context)
            except Exception as exc:
                if self.cfg.debug:
                    print(f"[{self.env.now:.1f}ms] Draft {self.id}: gamma policy select failed: {exc}", flush=True)
                selected_gamma = self.gamma
            gamma_value = max(1, int(selected_gamma or 0))
            rounds_needed = (answer_length + gamma_value - 1) // gamma_value  # ceiling division
            if self.id in ["llama_d000", "llama_d001"]:
                print(
                    f"[{self.env.now:.1f}ms] Draft {self.id}: rounds_needed={rounds_needed} "
                    f"(answer={answer_length}, gamma={gamma_value})",
                    flush=True,
                )
            tokens_generated_in_conversation = 0
            tokens_accepted_in_conversation = 0
            first_token_time_ms = None  # Track TTFT
            conversation_tpot_samples: List[float] = []
            conversation_completed = False

            # Initialize progress for Semi-Clairvoyant router
            if hasattr(self.router, 'update_progress'):
                self.router.update_progress(self.id, 0, 0, answer_length)

            try:
                for round_num in range(rounds_needed):
                    round_start = self.env.now

                    # Determine how many tokens to generate in this round
                    tokens_remaining = answer_length - tokens_generated_in_conversation
                    tokens_this_round = min(gamma_value, tokens_remaining)
                    if self.id in ["llama_d000", "llama_d001"]:
                        print(f"[{self.env.now:.1f}ms] Draft {self.id}: Round {round_num+1}/{rounds_needed}, tokens_this_round={tokens_this_round}", flush=True)

                    current_context = prompt_length + tokens_generated_in_conversation
                    # Generate draft tokens using performance provider latency
                    if fused_mode:
                        generation_latency = 0.0
                    else:
                        generation_latency = self._predict_generation_latency(tokens_this_round, current_context)
                    if generation_latency > 0:
                        yield self.env.timeout(generation_latency)
                    decode_breakdown["generation_ms"] += generation_latency
                    if ttft_pending:
                        ttft_breakdown["decode_generation_ms"] += generation_latency

                    self.chunks_sent += 1
                    self.total_tokens_generated += tokens_this_round
                    tokens_generated_in_conversation += tokens_this_round

                    if self.cfg.debug:
                        print(f"[{self.env.now:.1f}ms] Draft {self.id}: Round {round_num+1}/{rounds_needed} - "
                              f"Generated {tokens_this_round} tokens for target {target_id}", flush=True)

                    # Create a decode job with completion event
                    decode_completion = self.env.event()
                    job = Job(
                        jid=self.chunks_sent,
                        created_ms=self.env.now,
                        draft_id=self.id,
                        job_type="decode",
                        token_count=tokens_this_round,
                        completion_event=decode_completion,
                        rtt_start_ms=round_start,
                        request_id=conversation_id,
                        context_len=current_context + tokens_this_round,
                        target_id=target_id,
                        priority_class=priority_class,
                        phase="decode",
                        gamma_override=gamma_value,
                        is_fused=fused_mode,
                    )

                    # Send chunk to target (forward latency)
                    decode_payload = self._estimate_payload_bytes(tokens_this_round, self._decode_overhead_bytes)
                    if fused_mode:
                        forward_elapsed = 0.0
                    else:
                        fwd_chunk_start = self.env.now
                        yield self._network_transfer(
                            self.id,
                            target_id,
                            conn.forward_latency_ms,
                            payload_bytes=decode_payload,
                            link_key=conn.network_forward_key,
                        )
                        forward_elapsed = self.env.now - fwd_chunk_start
                    decode_breakdown["forward_ms"] += forward_elapsed
                    if ttft_pending:
                        ttft_breakdown["decode_forward_ms"] += forward_elapsed

                    # Job arrives at target after network delay (or immediately in fused mode)
                    job.created_ms = self.env.now

                    if self.scheduler is None:
                        if target_server is None:
                            print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
                            conversation_completed = True
                            break
                        if fused_mode:
                            wait_event = target_server.fused_execute(job)
                        else:
                            target_server.enqueue(job)
                            wait_event = decode_completion
                    else:
                        wait_event = self.scheduler.submit_job(job)

                    # Wait for actual job completion instead of synthetic sleep
                    yield wait_event

                    # Wait for response to travel back
                    response_payload = self._estimate_payload_bytes(tokens_this_round, self._response_overhead_bytes)
                    if fused_mode:
                        response_elapsed = 0.0
                    else:
                        rsp_chunk_start = self.env.now
                        yield self._network_transfer(
                            target_id,
                            self.id,
                            conn.response_latency_ms,
                            payload_bytes=response_payload,
                            link_key=conn.network_response_key,
                        )
                        response_elapsed = self.env.now - rsp_chunk_start
                    decode_breakdown["response_ms"] += response_elapsed
                    if ttft_pending:
                        ttft_breakdown["decode_response_ms"] += response_elapsed

                    # Mark RTT end after response received
                    job.rtt_end_ms = self.env.now

                    # Queue wait and compute durations
                    job_started = job.started_ms if job.started_ms is not None else job.created_ms
                    queue_wait = max(0.0, (job_started - job.created_ms) if job_started is not None else 0.0)
                    compute_ms = max(0.0, (job.finished_ms - job_started) if job_started is not None and job.finished_ms is not None else 0.0)
                    decode_breakdown["queue_ms"] += queue_wait
                    decode_breakdown["compute_ms"] += compute_ms
                    if ttft_pending:
                        ttft_breakdown["decode_queue_ms"] += queue_wait
                        ttft_breakdown["decode_compute_ms"] += compute_ms

                    # Simulate verification result
                    result = self._simulate_verification(
                        tokens_this_round,
                        conn,
                        current_context + tokens_this_round)
                    job.accepted_tokens = result.accepted_tokens
                    self.total_tokens_accepted += result.accepted_tokens
                    self.total_tokens_rejected += result.rejected_tokens
                    tokens_accepted_in_conversation += result.accepted_tokens

                    # Calculate round-trip time for this chunk
                    rtt = self.env.now - round_start
                    self.total_round_trip_time += rtt

                    if self.env.now >= self.cfg.burn_in_ms:
                        self.metrics.token_metrics.total_generated_tokens += tokens_this_round
                        self.metrics.token_metrics.total_accepted_tokens += result.accepted_tokens
                        self.metrics.token_metrics.total_rejected_tokens += result.rejected_tokens

                    if result.accepted_tokens > 0:
                        per_token = rtt / max(1, result.accepted_tokens)
                        conversation_tpot_samples.extend([per_token] * result.accepted_tokens)
                        if first_token_time_ms is None:
                            first_token_time_ms = self.env.now
                            ttft_pending = False

                        # Update progress for Semi-Clairvoyant router with actual acceptance
                        if hasattr(self.router, 'update_progress'):
                            self.router.update_progress(self.id, tokens_generated_in_conversation,
                                                   tokens_accepted_in_conversation, answer_length)

                        if self.cfg.debug:
                            print(f"[{self.env.now:.1f}ms] Draft {self.id}: Round {round_num+1} result: "
                                  f"{result.accepted_tokens}/{tokens_this_round} accepted, RTT={rtt:.1f}ms", flush=True)

                        if tokens_accepted_in_conversation >= answer_length:
                            conversation_completed = True
                            break

                else:
                    conversation_completed = True
            finally:
                if conversation_completed and tokens_generated_in_conversation > 0:
                    conversation_time = self.env.now - conversation_start
                    conversation_acceptance = (
                        tokens_accepted_in_conversation / tokens_generated_in_conversation
                        if tokens_generated_in_conversation > 0 else 0
                    )

                    if self.cfg.verbose or self.cfg.debug:
                        print(
                            f"[{self.env.now:.1f}ms] Draft {self.id}: Conversation #{conversation_count} completed - "
                            f"prompt={prompt_length}, answer={tokens_generated_in_conversation}/{answer_length} tokens, "
                            f"acceptance={conversation_acceptance:.2%}, time={conversation_time:.1f}ms",
                            flush=True,
                        )

                    if hasattr(self.metrics, "record_conversation"):
                        ttft_ms = (
                            (first_token_time_ms - conversation_start)
                            if first_token_time_ms is not None else None
                        )
                        self.metrics.record_conversation(
                            conversation_id,
                            start_ms=conversation_start,
                            end_ms=self.env.now,
                            draft_id=self.id,
                            target_id=target_id,
                            tokens_generated=tokens_generated_in_conversation,
                            tokens_accepted=tokens_accepted_in_conversation,
                            answer_tokens=answer_length,
                            ttft_ms=ttft_ms,
                            tpot_samples=conversation_tpot_samples,
                            ttft_breakdown=ttft_breakdown,
                            decode_breakdown=decode_breakdown,
                        )

                    self._gamma_policy.update_gamma(
                        self.id,
                        GammaConversationStats(
                            acceptance_ratio=conversation_acceptance,
                            tokens_generated=tokens_generated_in_conversation,
                            tokens_accepted=tokens_accepted_in_conversation,
                        ),
                    )

            # Schedule next arrival for both think_enabled and workload-driven modes
            if not self._trace_mode:
                self._schedule_next_arrival(self.env.now)
            elif self._trace_mode and self._think_enabled:
                # In trace mode with think time, update _next_available_ms for the next trace record
                think_time = self._sample_think_time()
                self._next_available_ms = self.env.now + think_time
                if self.id in ["llama_d000", "llama_d001"]:
                    print(f"[{self.env.now:.1f}ms] Draft {self.id}: Conversation {conversation_count} done, "
                          f"trace_index={self._trace_index}/{len(records)}, think_time={think_time:.1f}ms, "
                          f"next_available={self._next_available_ms:.1f}ms", flush=True)

        # Final statistics
        if self.chunks_sent > 0:
            acceptance_rate = self.total_tokens_accepted / self.total_tokens_generated
            avg_rtt = self.total_round_trip_time / self.chunks_sent
        else:
            acceptance_rate = 0
            avg_rtt = 0

        print(f"[{self.env.now:.1f}ms] Draft {self.id} finished: conversations={conversation_count}, chunks={self.chunks_sent}, "
              f"tokens={self.total_tokens_generated}, accepted={self.total_tokens_accepted} ({acceptance_rate:.2%}), "
              f"avg RTT={avg_rtt:.1f}ms, scheduler={'Yes' if self.scheduler else 'No'}", flush=True)

# ---------- Scheduler Core ----------


@dataclass
class PhaseSettings:
    pool: str = "default"
    queue_policy: str = "priority"  # "fifo" or "priority"
    max_batch_requests: int = 32
    max_batch_tokens: int = 0
    max_wait_ms: float = 6.0
    chunk_tokens: int = 0
    mode: str = "batch"
    chunk_sequential: bool = False
    delayed_batch_ms: float = 0.0
    retry_backoff_ms: float = 1.0
    max_retries: int = 3
    parallelism_plan: Dict[str, Any] = field(default_factory=dict)
    dynamic_policy: Dict[str, Any] = field(default_factory=dict)
    max_queue_depth: int = 0
    backpressure_wait_ms: float = 0.1


@dataclass(order=True)
class QueueItem:
    sort_key: Tuple[float, int]
    job: Job = field(compare=False)


class KVManager:
    """Track KV utilization per target with optional paging penalties."""

    def __init__(
        self,
        capacities: Mapping[str, int],
        *,
        default_capacity_tokens: int = 1_000_000,
        max_utilization_pct: float = 100.0,
        paging: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.default_capacity_tokens = max(1, int(default_capacity_tokens))
        self.capacities: Dict[str, int] = {str(k): max(1, int(v)) for k, v in (capacities or {}).items()}
        self.usage: Dict[str, int] = {target_id: 0 for target_id in self.capacities.keys()}
        pct = max(0.0, min(float(max_utilization_pct), 100.0))
        self.max_utilization = pct / 100.0 if pct > 0 else 1.0

        paging = paging or {}
        self.paging_enabled = bool(paging.get("enabled", False))
        self.page_size_tokens = max(1, int(paging.get("page_size_tokens", 1024)))
        self.page_in_penalty_ms = float(paging.get("page_in_penalty_ms", 0.0))

    def register_target(self, target_id: str, capacity_tokens: Optional[int] = None) -> None:
        tid = str(target_id)
        if tid not in self.capacities:
            cap = capacity_tokens if capacity_tokens is not None else self.default_capacity_tokens
            self.capacities[tid] = max(1, int(cap))
            self.usage[tid] = 0
        elif tid not in self.usage:
            self.usage[tid] = 0

    def reserve(self, target_id: str, tokens: int) -> Tuple[bool, float]:
        """Attempt to reserve KV space. Returns (success, page_in_delay_ms)."""
        if tokens <= 0:
            return True, 0.0
        tid = str(target_id)
        if tid not in self.capacities:
            self.register_target(tid)
        capacity = self.capacities.get(tid, self.default_capacity_tokens)
        current = self.usage.get(tid, 0)
        limit = capacity * self.max_utilization
        new_total = current + tokens
        if limit <= 0:
            limit = capacity

        if new_total <= limit + 1e-6:
            self.usage[tid] = min(new_total, capacity)
            return True, 0.0

        if self.paging_enabled:
            overflow = max(0, new_total - limit)
            pages = math.ceil(overflow / max(1, self.page_size_tokens))
            penalty = pages * self.page_in_penalty_ms
            self.usage[tid] = min(new_total, capacity)
            return True, penalty

        # Cannot admit, leave usage unchanged
        return False, 0.0

    def release(self, target_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        tid = str(target_id)
        if tid not in self.usage:
            return
        self.usage[tid] = max(0, self.usage[tid] - tokens)


class PhaseScheduler:
    """Per-phase scheduler with priority queues, batching, and KV guards."""

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        phase: str,
        settings: PhaseSettings,
        *,
        target_lookup: Dict[str, TargetServer],
        pool_targets: List[TargetServer],
        kv_manager: Optional[KVManager],
        global_router,
        priority_map: Dict[str, int],
        default_priority_class: str,
        metrics: Metrics,
    ) -> None:
        self.env = env
        self.name = name
        self.phase = phase
        self.settings = settings
        self.store = simpy.PriorityStore(env)
        self._seq = 0
        self.target_lookup = target_lookup
        self.pool_targets = list(pool_targets) if pool_targets else list(target_lookup.values())
        if not self.pool_targets:
            raise ValueError(f"Scheduler pool '{settings.pool}' has no targets")
        self.pool_map = {t.p.id: t for t in self.pool_targets}
        self.kv_manager = kv_manager
        self.global_router = global_router
        self.priority_map = priority_map
        self.default_priority_class = default_priority_class
        self.metrics = metrics
        self.dynamic_policy = dict(settings.dynamic_policy or {})
        self._pending_enqueues = 0
        self._debug_sample = 0
        self._debug_enqueued = defaultdict(int)
        self._debug_dequeued = defaultdict(int)
        self._debug_batch_add = 0
        self.proc = env.process(self._run())

    def submit(self, job: Job) -> simpy.Event:
        job.phase = self.phase
        if not job.parallelism_plan and self.settings.parallelism_plan:
            job.parallelism_plan = dict(self.settings.parallelism_plan)
        if not job.priority_class:
            job.priority_class = self.default_priority_class
        job.priority = self.priority_map.get(job.priority_class, self.priority_map.get(self.default_priority_class, 100))

        if self.settings.chunk_tokens and job.token_count > self.settings.chunk_tokens:
            self._submit_chunked(job)
            return job.completion_event

        if job.kv_tokens <= 0:
            job.kv_tokens = self._estimate_kv_tokens(job)
        self._enqueue_job(job)
        return job.completion_event

    def _submit_chunked(self, job: Job) -> None:
        chunk_size = max(1, self.settings.chunk_tokens)
        total_chunks = math.ceil(job.token_count / chunk_size)
        barrier = ChunkBarrier(self.env, total_chunks, job.completion_event)
        base_tokens = job.token_count
        prev_completion = None
        for idx in range(total_chunks):
            if idx == total_chunks - 1:
                tokens = base_tokens - chunk_size * (total_chunks - 1)
            else:
                tokens = chunk_size
            chunk_completion = self.env.event()
            chunk_job = Job(
                jid=job.jid * 1000 + idx,
                created_ms=job.created_ms,
                draft_id=job.draft_id,
                job_type=job.job_type,
                token_count=tokens,
                completion_event=chunk_completion,
                rtt_start_ms=job.rtt_start_ms if idx == 0 else None,
                request_id=job.request_id,
                context_len=job.context_len,
                target_id=job.target_id,
                priority_class=job.priority_class,
                priority=job.priority,
                kv_tokens=tokens,
                phase=self.phase,
                chunk_index=idx,
                chunk_count=total_chunks,
                chunk_barrier=barrier,
                parallelism_plan=dict(self.settings.parallelism_plan),
            )
            if self.settings.chunk_sequential and prev_completion is not None:
                self._schedule_chunk_after(prev_completion, chunk_job)
            else:
                self._enqueue_job(chunk_job)
            prev_completion = chunk_completion

    def _schedule_chunk_after(self, prior_event: simpy.Event, job: Job) -> None:
        def _defer():
            yield prior_event
            if not getattr(prior_event, 'ok', True):
                barrier = getattr(job, 'chunk_barrier', None)
                final_event = getattr(barrier, 'final_event', None) if barrier else None
                if final_event is not None and not final_event.triggered:
                    err = getattr(prior_event, 'value', None)
                    if err is None:
                        err = RuntimeError('Upstream chunk failed before scheduling next chunk')
                    final_event.fail(err)
                return
            if job.kv_tokens <= 0:
                job.kv_tokens = self._estimate_kv_tokens(job)
            self._enqueue_job(job)
        self.env.process(_defer())

    def _enqueue_job(self, job: Job) -> None:
        self._seq += 1
        draft_id = (job.draft_id or "").strip()
        target_id = (job.target_id or "").strip()
        self._debug_enqueued[draft_id] += 1
        if target_id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
            print(
                f"[{self.env.now:.1f}ms] Scheduler {self.name}: enqueue {job.job_type} {draft_id}->{target_id} "
                f"(priority_class={job.priority_class}, priority={job.priority}, kv_tokens={job.kv_tokens})",
                flush=True,
            )
        if self.settings.queue_policy == "fifo":
            sort_key: Tuple[float, int] = (float(self._seq),)
        else:
            sort_key = (float(job.priority), self._seq)
        item = QueueItem(sort_key, job)

        if target_id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
            print(
                f"[{self.env.now:.1f}ms] Scheduler {self.name}: queue key={sort_key} depth={len(self.store.items)} pending={self._pending_enqueues}",
                flush=True,
            )

        if self.settings.max_queue_depth and self.settings.max_queue_depth > 0:
            if len(self.store.items) + self._pending_enqueues >= self.settings.max_queue_depth:
                self._schedule_backpressure(item)
                return

        self.store.put(item)
        if self._seq <= 50:
            print(
                f"[{self.env.now:.1f}ms] Scheduler {self.name}: sample enqueue seq={self._seq} draft={draft_id}->{target_id} priority={job.priority} class={job.priority_class} depth={len(self.store.items)}",
                flush=True,
            )
        if target_id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
            print(
                f"[{self.env.now:.1f}ms] Scheduler {self.name}: queue depth after put={len(self.store.items)} pending={self._pending_enqueues}",
                flush=True,
            )

            def _check_pending(item_ref: QueueItem, delay_ms: float, label: str):
                yield self.env.timeout(delay_ms)
                still_present = item_ref in self.store.items
                enq = self._debug_enqueued.get(draft_id, 0)
                deq = self._debug_dequeued.get(draft_id, 0)
                snapshot = [getattr(q, 'sort_key', None) for q in self.store.items]
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: pending check ({label}) for {draft_id}->{target_id} still_present={still_present} depth={len(self.store.items)} enq={enq} deq={deq} batch_add={self._debug_batch_add} snapshot={snapshot}",
                    flush=True,
                )

            self.env.process(_check_pending(item, 1.0, 'early'))
            self.env.process(_check_pending(item, 200.0, 'late'))

    def _run(self):
        while True:
            item = yield self.store.get()
            draft_id = (item.job.draft_id or "").strip()
            target_id = (item.job.target_id or "").strip()
            self._debug_dequeued[draft_id] += 1
            self._debug_sample += 1
            if self.name == "prefill" and self._debug_sample <= 80:
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: sample dequeue draft={draft_id!r} target={target_id!r} type={item.job.job_type} sample_idx={self._debug_sample}",
                    flush=True,
                )
            if draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: debug dequeue candidate {item.job.job_type} {draft_id!r}->{target_id!r}",
                    flush=True,
                )
            if target_id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: dequeued {item.job.job_type} {draft_id}->{target_id}",
                    flush=True,
                )
            batch = [item.job]
            token_total = item.job.token_count

            max_wait_ms, max_batch_requests, delay_ms = self._dynamic_batch_params(len(self.store.items))
            deadline = self.env.now + max(0.0, max_wait_ms)
            earliest_dispatch = self.env.now + max(0.0, delay_ms)

            while self._can_add_more(batch, token_total, max_batch_requests):
                now = self.env.now
                remaining_deadline = max(0.0, deadline - now)
                if remaining_deadline <= 0:
                    break
                target_time = deadline
                if len(batch) == 1 and self.settings.delayed_batch_ms > 0 and earliest_dispatch > now:
                    target_time = min(target_time, earliest_dispatch)
                remaining = max(0.0, target_time - now)
                if remaining <= 0:
                    break
                get_ev = self.store.get()
                timeout_ev = self.env.timeout(remaining)
                result = yield get_ev | timeout_ev
                if get_ev in result:
                    next_job = result[get_ev].job
                    self._debug_batch_add += 1
                    if (next_job.draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}
                            or (next_job.target_id or "").strip() == "llama_t01"
                            or self._debug_batch_add <= 40):
                        print(
                            f"[{self.env.now:.1f}ms] Scheduler {self.name}: batch add {next_job.job_type} {next_job.draft_id!r}->{next_job.target_id!r} key={result[get_ev].sort_key} queue_left={len(self.store.items)}",
                            flush=True,
                        )
                    batch.append(next_job)
                    token_total += next_job.token_count
                    if self.settings.max_batch_tokens and token_total >= self.settings.max_batch_tokens:
                        break
                else:
                    if hasattr(get_ev, 'cancel'):
                        get_ev.cancel()
                    break

            if any((job.draft_id or '').strip() in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"} or (job.target_id or '').strip() == "llama_t01" for job in batch):
                batch_ids = [(job.draft_id, job.target_id, job.job_type) for job in batch]
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: dispatching batch with items={batch_ids}",
                    flush=True,
                )
            yield self.env.process(self._dispatch_batch_proc(batch))

    def _dynamic_batch_params(self, queue_depth: int) -> Tuple[float, int, float]:
        dyn = self.dynamic_policy
        if not dyn or not dyn.get("enabled", False):
            return self.settings.max_wait_ms, self.settings.max_batch_requests, self.settings.delayed_batch_ms

        low_depth = max(0, int(dyn.get("low_queue_depth", 0)))
        high_depth = max(low_depth + 1, int(dyn.get("high_queue_depth", low_depth + 1)))

        def _lerp(low_val: float, high_val: float) -> float:
            if queue_depth <= low_depth:
                return low_val
            if queue_depth >= high_depth:
                return high_val
            span = high_depth - low_depth
            alpha = (queue_depth - low_depth) / span
            return low_val + alpha * (high_val - low_val)

        low_wait = float(dyn.get("low_wait_ms", self.settings.max_wait_ms))
        high_wait = float(dyn.get("high_wait_ms", self.settings.max_wait_ms))
        effective_wait = _lerp(low_wait, high_wait)

        low_batch = int(dyn.get("low_batch_requests", self.settings.max_batch_requests or 1))
        high_batch = int(dyn.get("high_batch_requests", self.settings.max_batch_requests or low_batch))
        effective_batch = int(round(_lerp(low_batch, high_batch)))
        if effective_batch <= 0:
            effective_batch = self.settings.max_batch_requests

        low_delay = float(dyn.get("low_delay_ms", self.settings.delayed_batch_ms))
        high_delay = float(dyn.get("high_delay_ms", self.settings.delayed_batch_ms))
        effective_delay = _lerp(low_delay, high_delay)

        return effective_wait, effective_batch, effective_delay

    def _can_add_more(self, batch: List[Job], token_total: int, max_batch_requests: int) -> bool:
        batch_limit = max_batch_requests if max_batch_requests and max_batch_requests > 0 else self.settings.max_batch_requests
        if batch_limit and len(batch) >= batch_limit:
            return False
        if self.settings.max_batch_tokens and token_total >= self.settings.max_batch_tokens:
            return False
        limit = batch_limit if batch_limit and batch_limit > 0 else max(1, self.settings.max_batch_requests or 1)
        return len(batch) < max(1, limit)

    def _schedule_backpressure(self, item: QueueItem) -> None:
        self._pending_enqueues += 1

        def _deferred_enqueue():
            while True:
                depth = len(self.store.items)
                limit = self.settings.max_queue_depth
                if not limit or depth < limit:
                    break
                wait = max(0.0, self.settings.backpressure_wait_ms)
                yield self.env.timeout(wait)
            self._pending_enqueues -= 1
            draft_id = (item.job.draft_id or "").strip()
            target_id = (item.job.target_id or "").strip()
            if self.name == "prefill" and (target_id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}):
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: deferred enqueue {item.job.job_type} {draft_id}->{target_id} (pending={self._pending_enqueues})",
                    flush=True,
                )
            self.store.put(item)

        self.env.process(_deferred_enqueue())

    def _dispatch_batch_proc(self, batch: List[Job]):
        groups: Dict[str, List[Job]] = {}
        for job in batch:
            target = self._choose_target(job)
            job.target_id = target.p.id
            groups.setdefault(target.p.id, []).append(job)

        for target_id, jobs in groups.items():
            target_name = (target_id or "").strip()
            target = self.pool_map.get(target_id) or self.target_lookup.get(target_id)
            if target is None:
                if target_name == "llama_t01":
                    print(
                        f"[{self.env.now:.1f}ms] Scheduler {self.name}: no target entry for {target_name}",
                        flush=True,
                    )
                continue
            accepted: List[Job] = []
            max_penalty = 0.0
            for job in jobs:
                draft_id = (job.draft_id or "").strip()
                if draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                    print(
                        f"[{self.env.now:.1f}ms] Scheduler {self.name}: group {draft_id}->{target_name} (job_type={job.job_type})",
                        flush=True,
                    )
                if job.kv_tokens <= 0:
                    job.kv_tokens = self._estimate_kv_tokens(job)
                if self.kv_manager is None:
                    accepted.append(job)
                    continue
                success, penalty = self.kv_manager.reserve(target_id, job.kv_tokens)
                if not success:
                    if target_name == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                        print(
                            f"[{self.env.now:.1f}ms] Scheduler {self.name}: KV reject {job.job_type} job from {draft_id} -> {target_name} (retry_count={job.retry_count})",
                            flush=True,
                        )
                    self._schedule_retry(job)
                    continue
                max_penalty = max(max_penalty, penalty)
                accepted.append(job)

            if not accepted:
                continue

            if max_penalty > 0:
                yield self.env.timeout(max_penalty)

            for job in accepted:
                job.created_ms = self.env.now
                draft_id = (job.draft_id or "").strip()
                if target_name == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                    print(
                        f"[{self.env.now:.1f}ms] Scheduler {self.name}: dispatching {job.job_type} job {draft_id}->{target_name} (priority={job.priority}, tokens={job.token_count})",
                        flush=True,
                    )
                target.enqueue(job)

    def _schedule_retry(self, job: Job) -> None:
        if self.settings.max_retries >= 0 and job.retry_count >= self.settings.max_retries:
            if self.metrics and self.metrics.verbose:
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: dropping job {job.jid} after retries",
                    flush=True,
                )
            if job.completion_event and not job.completion_event.triggered:
                job.completion_event.fail(RuntimeError("KV admission rejected job"))
            return

        draft_id = (job.draft_id or "").strip()
        target_id = (job.target_id or "").strip()
        if target_id == "llama_t01" or draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
            print(
                f"[{self.env.now:.1f}ms] Scheduler {self.name}: scheduling retry {job.job_type} {draft_id}->{target_id} (retry_count={job.retry_count + 1})",
                flush=True,
            )
        job.retry_count += 1

        def _retry_proc():
            yield self.env.timeout(max(0.0, self.settings.retry_backoff_ms * job.retry_count))
            self._enqueue_job(job)

        self.env.process(_retry_proc())

    def _choose_target(self, job: Job) -> TargetServer:
        if job.target_id and job.target_id in self.pool_map:
            if job.draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: using sticky target {job.target_id} for {job.draft_id}",
                    flush=True,
                )
            return self.pool_map[job.target_id]

        allowed_ids = list(self.pool_map.keys())
        chosen = None
        if self.global_router is not None:
            chosen = self.global_router.choose(job.draft_id, allowed_ids)
            if job.draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
                print(
                    f"[{self.env.now:.1f}ms] Scheduler {self.name}: global router choice {chosen} for {job.draft_id} among {allowed_ids}",
                    flush=True,
                )

        if chosen is not None:
            return chosen

        # Fall back to least work-left among pool
        target = min(self.pool_targets, key=lambda t: t.work_left_score())
        if job.draft_id in {"llama_d001", "llama_d011", "llama_d017", "llama_d031"}:
            print(
                f"[{self.env.now:.1f}ms] Scheduler {self.name}: fallback target {target.p.id} for {job.draft_id}",
                flush=True,
            )
        return target

    def _estimate_kv_tokens(self, job: Job) -> int:
        if job.phase == "prefill":
            return max(1, job.token_count)
        base = max(1, job.token_count)
        ctx = max(0, job.context_len)
        return base + ctx


class Scheduler:
    """Top-level scheduler coordinating prefill and decode lanes."""

    def __init__(
        self,
        env: simpy.Environment,
        cfg: Config,
        *,
        target_lookup: Dict[str, TargetServer],
        targets_by_cluster: Dict[str, List[TargetServer]],
        kv_manager: Optional[KVManager],
        global_router,
        metrics: Metrics,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.kv_manager = kv_manager
        self.global_router = global_router
        self.metrics = metrics

        sched_cfg = dict(cfg.scheduler_config or {})
        self.priority_map = self._build_priority_map(sched_cfg)
        self.default_priority_class = sched_cfg.get("default_priority_class", "standard")
        if self.default_priority_class not in self.priority_map:
            self.priority_map[self.default_priority_class] = self.priority_map.get("standard", 100)

        pools = self._resolve_pools(sched_cfg, target_lookup, targets_by_cluster)

        prefill_settings = self._phase_settings("prefill", sched_cfg)
        decode_settings = self._phase_settings("decode", sched_cfg)

        prefill_pool_targets = pools.get(prefill_settings.pool) or pools.get("default")
        decode_pool_targets = pools.get(decode_settings.pool) or pools.get("default")

        if prefill_pool_targets is None or decode_pool_targets is None:
            raise ValueError("Scheduler pools are not defined correctly")

        self.prefill_scheduler = PhaseScheduler(
            env,
            name="prefill",
            phase="prefill",
            settings=prefill_settings,
            target_lookup=target_lookup,
            pool_targets=prefill_pool_targets,
            kv_manager=kv_manager,
            global_router=global_router,
            priority_map=self.priority_map,
            default_priority_class=self.default_priority_class,
            metrics=metrics,
        )

        self.decode_scheduler = PhaseScheduler(
            env,
            name="decode",
            phase="decode",
            settings=decode_settings,
            target_lookup=target_lookup,
            pool_targets=decode_pool_targets,
            kv_manager=kv_manager,
            global_router=global_router,
            priority_map=self.priority_map,
            default_priority_class=self.default_priority_class,
            metrics=metrics,
        )

    def submit_job(self, job: Job) -> simpy.Event:
        if job.job_type == "prefill":
            return self.prefill_scheduler.submit(job)
        return self.decode_scheduler.submit(job)

    def priority_for(self, priority_class: Optional[str]) -> int:
        class_name = priority_class or self.default_priority_class
        return self.priority_map.get(class_name, self.priority_map.get(self.default_priority_class, 100))

    def _build_priority_map(self, sched_cfg: Dict[str, Any]) -> Dict[str, int]:
        raw = sched_cfg.get("priority_classes", {}) or {}
        if not raw:
            return {"high": 0, "standard": 100, "low": 200}
        result = {}
        for key, value in raw.items():
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        if "standard" not in result:
            result.setdefault("standard", min(result.values(), default=100) + 100)
        return result

    def _phase_settings(self, phase: str, sched_cfg: Dict[str, Any]) -> PhaseSettings:
        phase_cfg = dict(sched_cfg.get(phase, {}) or {})
        settings = PhaseSettings()
        settings.pool = phase_cfg.get("pool", "default")
        settings.queue_policy = phase_cfg.get("queue_policy", "priority")
        settings.max_batch_requests = int(phase_cfg.get("max_batch_requests", settings.max_batch_requests))
        settings.max_batch_tokens = int(phase_cfg.get("max_batch_tokens", settings.max_batch_tokens))
        settings.max_wait_ms = float(phase_cfg.get("max_wait_ms", settings.max_wait_ms))
        settings.delayed_batch_ms = float(phase_cfg.get("delayed_batch_ms", settings.delayed_batch_ms))
        settings.retry_backoff_ms = float(phase_cfg.get("retry_backoff_ms", settings.retry_backoff_ms))
        settings.max_retries = int(phase_cfg.get("max_retries", settings.max_retries))
        settings.parallelism_plan = dict(phase_cfg.get("parallelism", settings.parallelism_plan) or {})
        settings.dynamic_policy = dict(phase_cfg.get("dynamic_policy", settings.dynamic_policy) or {})
        settings.max_queue_depth = int(phase_cfg.get("max_queue_depth", settings.max_queue_depth))
        settings.backpressure_wait_ms = float(phase_cfg.get("backpressure_wait_ms", settings.backpressure_wait_ms))

        mode = str(phase_cfg.get("mode", settings.mode)).lower()
        settings.mode = mode

        chunk_tokens = phase_cfg.get("chunk_tokens")
        if chunk_tokens is None:
            chunk_tokens = self.cfg.scheduler_config.get(f"{phase}_chunk_tokens")
        if chunk_tokens:
            settings.chunk_tokens = max(1, int(chunk_tokens))

        if mode in {"continuous", "continuous_batching"}:
            settings.mode = "continuous"
            settings.chunk_sequential = bool(phase_cfg.get("chunk_sequential", True))
            if settings.chunk_tokens <= 0:
                settings.chunk_tokens = max(1, int(self.cfg.scheduler_config.get(f"{phase}_chunk_tokens", 1)))
        else:
            settings.chunk_sequential = bool(phase_cfg.get("chunk_sequential", False))

        return settings

    def _resolve_pools(
        self,
        sched_cfg: Dict[str, Any],
        target_lookup: Dict[str, TargetServer],
        targets_by_cluster: Dict[str, List[TargetServer]],
    ) -> Dict[str, List[TargetServer]]:
        pools_cfg = sched_cfg.get("pools", {}) or {}
        pools: Dict[str, List[TargetServer]] = {}

        if pools_cfg:
            for name, spec in pools_cfg.items():
                targets: List[TargetServer] = []
                if isinstance(spec, dict):
                    for tid in spec.get("targets", []) or []:
                        if tid in target_lookup:
                            targets.append(target_lookup[tid])
                    for cluster in spec.get("clusters", []) or []:
                        targets.extend(targets_by_cluster.get(cluster, []))
                elif isinstance(spec, list):
                    for tid in spec:
                        if tid in target_lookup:
                            targets.append(target_lookup[tid])
                elif isinstance(spec, str):
                    if spec in target_lookup:
                        targets.append(target_lookup[spec])
                    elif spec in targets_by_cluster:
                        targets.extend(targets_by_cluster[spec])
                if targets:
                    pools[name] = list(dict.fromkeys(targets))

        if "default" not in pools:
            pools["default"] = list(target_lookup.values())

        return pools

# ---------- Routers ----------

# Router classes act as weighted samplers for DraftServer._select_target()
# The route() method is NEVER used - drafts call _weighted_sample_k_filtered directly
# and enqueue jobs themselves to maintain connection-awareness.

class RandomRouter:
    """Pure random selection - no load awareness."""

    def __init__(self, targets: List[TargetServer], seed: Optional[int] = None):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self._rng = random.Random(seed)

    def random_select_filtered(self, allowed_ids) -> TargetServer:
        """Randomly select one target from allowed set."""
        pool = [t for t in self.targets if t.p.id in allowed_ids]
        if not pool:
            return None
        return self._rng.choice(pool)

class RoundRobinRouter:
    """Round-robin router with per-draft counters for connection-aware routing."""
    def __init__(self, targets: List[TargetServer]):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self._counters = {}  # Per-draft counters for independent round-robin
    
    def round_robin_select_filtered(self, draft_id: str, allowed_ids) -> TargetServer:
        """Round-robin selection from allowed targets, maintaining per-draft state."""
        pool = [t for t in self.targets if t.p.id in allowed_ids]
        if not pool:
            return None
        
        # Initialize counter for this draft if needed
        if draft_id not in self._counters:
            self._counters[draft_id] = 0
        
        # Select target based on counter
        idx = self._counters[draft_id] % len(pool)
        target = pool[idx]
        
        # Increment counter for next time
        self._counters[draft_id] += 1
        
        return target
class JSQRouter:
    """Join-the-Shortest-Queue router (global queue awareness)."""

    def __init__(self, targets: List[TargetServer], seed: Optional[int] = None):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self._rng = random.Random(seed)

    def jsq_select_filtered(self, allowed_ids) -> TargetServer:
        """Pick the target with the fewest queued jobs among the allowed set."""
        pool = [t for t in self.targets if t.p.id in allowed_ids]
        if not pool:
            return None
        lengths = [t.queue_len() for t in pool]
        min_len = min(lengths)
        candidates = [t for t, qlen in zip(pool, lengths) if qlen == min_len]
        if len(candidates) == 1:
            return candidates[0]
        return self._rng.choice(candidates)

class JIQRouter:
    """Join-Idle-Queue router with FIFO idle queue."""
    def __init__(self, targets: List[TargetServer]):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self.idle_queue = deque()  # FIFO queue of idle target IDs
        self.idle_set = set()  # For fast membership check
        
    def mark_idle(self, target_id: str):
        """Called by target when it becomes idle."""
        if target_id not in self.idle_set:
            self.idle_queue.append(target_id)
            self.idle_set.add(target_id)
    
    def mark_busy(self, target_id: str):
        """Called by target when it becomes busy."""
        # Just remove from set, leave in queue (will be skipped)
        self.idle_set.discard(target_id)
    
    def jiq_select_filtered(self, allowed_ids) -> TargetServer:
        """Select idle target from allowed set, or random if none idle."""
        # Try to find idle target in allowed set
        while self.idle_queue:
            # Pop from FIFO queue
            idle_id = self.idle_queue[0]
            
            # Check if still idle and allowed
            if idle_id in self.idle_set and idle_id in allowed_ids:
                # Found valid idle target
                self.idle_queue.popleft()
                self.idle_set.remove(idle_id)
                
                # Find the target server object
                for t in self.targets:
                    if t.p.id == idle_id:
                        return t
            else:
                # Not valid anymore, remove and continue
                self.idle_queue.popleft()
                if idle_id in self.idle_set:
                    self.idle_set.remove(idle_id)
        
        # No idle targets available, fallback to random
        pool = [t for t in self.targets if t.p.id in allowed_ids]
        if pool:
            return random.choice(pool)
        return None

class SemiClairvoyantRouter:
    """Semi-Clairvoyant router based on request progress (attained/remaining service).
    
    Implements fairness-based scheduling to prevent starvation by prioritizing
    requests with least relative progress. Adapted from the paper:
    "Semi-Clairvoyant Scheduling for Speculative Parallelism" 
    
    Uses acceptance rate to accurately estimate actual progress:
    Priority = tokens_accepted / answer_length
    Routes to target with lowest average priority (least progress).
    """
    def __init__(self, targets: List[TargetServer]):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self.request_progress = {}  # draft_id -> {tokens_generated, tokens_accepted, answer_length}
    
    def update_progress(self, draft_id: str, tokens_generated: int, tokens_accepted: int, answer_length: int):
        """Update progress tracking for a draft's conversation.
        Now tracks both generated and accepted tokens for accurate progress."""
        self.request_progress[draft_id] = {
            'tokens_generated': tokens_generated,
            'tokens_accepted': tokens_accepted,
            'answer_length': answer_length
        }
    
    def calculate_priority(self, draft_id: str) -> float:
        """Calculate priority based on ACTUAL progress (accepted tokens).
        Lower priority = less progress = should be prioritized."""
        if draft_id not in self.request_progress:
            return 0.5  # Default priority for unknown requests
        
        progress = self.request_progress[draft_id]
        tokens_accepted = progress.get('tokens_accepted', 0)
        answer_length = progress['answer_length']
        
        if tokens_accepted >= answer_length:  # Request complete
            return 1.0  # Lowest priority (done)
        
        if answer_length == 0:
            return 0.0  # Highest priority (just starting)
        
        # Priority = actual progress / total needed
        return tokens_accepted / answer_length
    
    def semi_clairvoyant_select_filtered(self, draft_id: str, allowed_ids) -> TargetServer:
        """Select target with least-progress requests (fairness-based)."""
        pool = [t for t in self.targets if t.p.id in allowed_ids]
        if not pool:
            return None
        
        # Calculate score for each target based on queue and our priority
        # Lower score = better choice
        best_target = None
        best_score = float('inf')
        
        for target in pool:
            # Use queue length as proxy for target load
            queue_len = target.queue_len()
            
            # Our request's priority (0=new, 1=complete)
            our_priority = self.calculate_priority(draft_id)
            
            # Fairness score: prioritize less loaded targets for low-progress requests
            # High-progress requests (priority near 1) get deprioritized
            score = queue_len * (2 - our_priority)  # Amplify effect for low progress
            
            if score < best_score:
                best_score = score
                best_target = target
        
        return best_target if best_target else pool[0]

class JSQ2Router:
    """Unweighted power-of-two choices sampler."""
    def __init__(self, targets: List[TargetServer], d_choices: int = 2):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self.d = min(max(1, d_choices), len(targets))

class WeightedJSQ2Router:
    """Weighted power-of-two choices sampler (main router used).
    
    DraftServer uses this as a weighted sampler via _weighted_sample_k_filtered(),
    which respects draft-target connectivity constraints.
    """
    def __init__(self, targets: List[TargetServer], d_choices: int = 2, debug: bool = False):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self.d = max(1, d_choices)
        self.debug = debug

    def _weighted_sample_k_filtered(self, k: int, allowed_ids) -> List[TargetServer]:
        """Weighted sampling without replacement restricted to allowed target IDs.
        
        This is the MAIN method used by DraftServer._select_target() to ensure
        connection-aware routing. Only samples from targets the draft can reach.
        """
        pool = [t for t in self.targets if t.p.id in allowed_ids]
        if not pool:
            return []
        k = min(k, len(pool))
        weights = [max(0.0, t.p.weight) for t in pool]
        if sum(weights) == 0:
            return random.sample(pool, k)
        keyed = []
        for t, w in zip(pool, weights):
            if w == 0:
                key = 0
            else:
                u = random.random()
                key = u ** (1.0 / w)
            keyed.append((key, t))
        keyed.sort(reverse=True)
        return [t for _, t in keyed[:k]]

class GlobalWeightedRouter:
    """Global router that chooses targets across clusters based on work-left scores."""
    def __init__(self, *, cluster_targets: Dict[str, List[TargetServer]],
                 target_lookup: Dict[str, TargetServer], params: Optional[Dict[str, Any]] = None) -> None:
        self._cluster_targets = cluster_targets
        self._target_lookup = target_lookup
        params = params or {}
        raw_weights = params.get('cluster_weights', {})
        default_weight = float(params.get('default_weight', 1.0))
        self._weights = {cluster: float(raw_weights.get(cluster, default_weight)) for cluster in cluster_targets.keys()}
        self._default_weight = max(default_weight, 1e-6)

    def choose(self, draft_id: str, allowed_target_ids: Iterable[str]) -> Optional[TargetServer]:
        allowed = set(allowed_target_ids)
        if not allowed:
            return None
        best_target: Optional[TargetServer] = None
        best_score = float('inf')
        for cluster, targets in self._cluster_targets.items():
            cluster_allowed = [t for t in targets if t.p.id in allowed]
            if not cluster_allowed:
                continue
            weight = max(self._weights.get(cluster, self._default_weight), 1e-6)
            for target in cluster_allowed:
                score = target.work_left_score() / weight
                if score < best_score:
                    best_score = score
                    best_target = target
        if best_target is not None:
            return best_target

        # Fallback: pick first known target from lookup if cluster filters missed it
        for target_id in allowed:
            candidate = self._target_lookup.get(target_id)
            if candidate is not None:
                return candidate
        return None

    def get_target(self, target_id: str) -> Optional[TargetServer]:
        return self._target_lookup.get(target_id)


GLOBAL_ROUTERS = {
    "weighted_cluster": GlobalWeightedRouter,
}

ROUTERS = {
    "random": RandomRouter,
    "round_robin": RoundRobinRouter,
    "jsq": JSQRouter,
    "jsq2": JSQ2Router,
    "wjsq2": WeightedJSQ2Router,
    "jiq": JIQRouter,
    "semi_clairvoyant": SemiClairvoyantRouter,
}

# ---------- Config I/O & Runner ----------


def _expand_auto_topology(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Expand auto_topology shorthand into full devices and connections lists."""
    import random
    from collections import defaultdict

    if not isinstance(raw, dict) or not raw.get("auto_topology"):
        return raw

    spec = raw["auto_topology"]
    rng = random.Random(raw.get("seed", 0))
    spec_exec_mode = str(raw.get("speculation", {}).get("execution_mode", "distributed")).lower()
    network_cfg = dict(raw.get("network", {}) or {})
    network_enabled = bool(network_cfg.get("enabled", True))

    def _sanitize_prefix(name: str) -> str:
        return str(name).replace(" ", "_")

    def _build_cluster(cluster_spec: Dict[str, Any], cluster_name: str):
        cluster_devices: list[Dict[str, Any]] = []
        cluster_connections: list[Dict[str, Any]] = []
        prefix = _sanitize_prefix(cluster_spec.get("id_prefix", cluster_name))

        # --- Targets ---
        t_spec = cluster_spec.get("targets", {})
        tiers = list(t_spec.get("tiers", []))
        inferred_count = int(t_spec.get("count", 0))
        if not tiers:
            tiers = [{
                "name": f"{cluster_name}_default",
                "ratio": 1.0,
            }]
        if inferred_count <= 0:
            explicit = sum(int(t.get("count", 0)) for t in tiers)
            inferred_count = explicit if explicit > 0 else sum(1 for _ in tiers)
        tier_counts = []
        remaining = inferred_count
        for idx, tier in enumerate(tiers):
            if "count" in tier:
                count = int(tier["count"])
            elif "ratio" in tier and inferred_count > 0:
                count = int(round(inferred_count * float(tier.get("ratio", 0.0))))
            else:
                count = int(round(inferred_count / len(tiers))) if inferred_count > 0 else 0
            if idx == len(tiers) - 1:
                count = remaining
            count = max(0, min(count, remaining))
            tier_counts.append(count)
            remaining -= count
        if tier_counts and remaining > 0:
            tier_counts[-1] += remaining

        targets_by_tier: defaultdict[str, list[str]] = defaultdict(list)
        tier_of: Dict[str, str] = {}
        target_entries: list[Dict[str, Any]] = []
        target_index = 0
        for tier, count in zip(tiers, tier_counts):
            tier_name = str(tier.get("name", f"{cluster_name}_tier"))
            for _ in range(count):
                tid = f"{prefix}_t{target_index:02d}"
                target_index += 1
                vidur_profile = dict(tier.get("vidur") or tier.get("vidur_profile") or {})
                entry: Dict[str, Any] = {
                    "id": tid,
                    "role": "target",
                    "model": tier.get("model") or vidur_profile.get("model_name", ""),
                    "gpu": tier.get("gpu") or vidur_profile.get("device", ""),
                    "weight": float(tier.get("weight", 1.0)),
                    "batch_window_ms": float(tier.get("batch_window_ms", 6.0)),
                    "batch_size": int(tier.get("batch_size", 32)),
                    "cluster": cluster_name,
                    "mode": str(tier.get("mode", "distributed")).lower(),
                }
                if "vidur" in tier:
                    entry["vidur"] = dict(tier["vidur"])
                if "vidur_profile" in tier:
                    entry["vidur_profile"] = dict(tier["vidur_profile"])
                if tier.get("draft_vidur"):
                    entry["fused_draft_profile"] = dict(tier["draft_vidur"])
                elif spec_exec_mode == "fused":
                    fused_profile = _default_fused_profile(entry)
                    if fused_profile is not None:
                        entry["fused_draft_profile"] = fused_profile
                if "metadata" in tier:
                    entry["metadata"] = dict(tier["metadata"])
                cluster_devices.append(entry)
                target_entries.append(entry)
                tier_of[tid] = tier_name
                targets_by_tier[tier_name].append(tid)
        all_tiers = list(targets_by_tier.keys())

        # --- Drafts ---
        d_spec = cluster_spec.get("drafts", {})
        d_count = int(d_spec.get("count", 0))
        capability_map = d_spec.get("capability_map", {})
        draft_meta = d_spec.get("metadata_by_label", {})
        reliability_map = d_spec.get("reliability", {})
        labels = list(d_spec.get("draft_bucket_labels", []))
        if not labels and capability_map:
            labels = [str(k) for k in capability_map.keys()]
        counts_by_label = d_spec.get("count_by_label", {})
        if d_count <= 0:
            if counts_by_label:
                d_count = sum(int(v) for v in counts_by_label.values())
            elif labels:
                d_count = len(labels)
        if d_count <= 0:
            d_count = 1
        if not labels:
            labels = [str(i) for i in range(d_count)]
        bucket_counts: List[int] = []
        remaining = d_count
        for idx, label in enumerate(labels):
            if label in counts_by_label:
                count = int(counts_by_label[label])
            else:
                count = int(round(d_count / len(labels))) if labels else d_count
            if idx == len(labels) - 1:
                count = remaining
            count = max(0, min(count, remaining))
            bucket_counts.append(count)
            remaining -= count
        if bucket_counts and remaining > 0:
            bucket_counts[-1] += remaining

        draft_entries: list[Dict[str, Any]] = []
        draft_index = 0
        for label, count in zip(labels, bucket_counts):
            for _ in range(count):
                did = f"{prefix}_d{draft_index:03d}"
                draft_index += 1
                capability = float(
                    capability_map.get(label, capability_map.get(str(label), d_spec.get("default_capability", 1.0)))
                )
                reliability = float(
                    reliability_map.get(label, reliability_map.get(str(label), d_spec.get("default_reliability", 0.99)))
                )
                combined_meta: Dict[str, Any] = {}
                if "metadata" in d_spec:
                    combined_meta.update(d_spec["metadata"])
                bucket_meta = draft_meta.get(label) or draft_meta.get(str(label))
                if isinstance(bucket_meta, Mapping):
                    combined_meta.update(bucket_meta)
                entry = {
                    "id": did,
                    "role": "draft",
                    "capability": capability,
                    "bucket": label,
                    "label": label,
                    "burst_factor": float(d_spec.get("burst_factor", 1.0)),
                    "reliability": reliability,
                    "cluster": cluster_name,
                }
                if combined_meta:
                    entry["metadata"] = combined_meta
                model_name = combined_meta.get("model") or combined_meta.get("model_name", "")
                hardware = combined_meta.get("hardware") or combined_meta.get("device", "")
                if model_name:
                    entry["model"] = model_name
                if hardware:
                    entry["gpu"] = hardware
                if "vidur_profile" in combined_meta:
                    entry["vidur_profile"] = dict(combined_meta["vidur_profile"])
                elif "vidur" in combined_meta:
                    entry["vidur_profile"] = dict(combined_meta["vidur"])
                cluster_devices.append(entry)
                draft_entries.append(entry)

        # --- Connectivity ---
        conn_spec = cluster_spec.get("connectivity", {})
        fanout_base = int(conn_spec.get("fanout_per_draft", 3))
        fanout_override = conn_spec.get("fanout_override", {})
        affinity = conn_spec.get("affinity_rules", {})
        net_ranges = conn_spec.get("net_ms_ranges", {})
        acc_tbl = conn_spec.get("acceptance_by_tier", {})
        jitter_pct = float(
            conn_spec.get(
                "link_jitter_pct",
                network_cfg.get("jitter_pct", 0.0),
            )
            or 0.0
        )
        drop_tbl = conn_spec.get("drop_rate", {})
        network_model_spec = None
        if network_enabled:
            network_model_spec = conn_spec.get("network_model")
            if not network_model_spec:
                network_model_spec = network_cfg.get("model")
        latency_lookup = None
        if network_enabled and network_model_spec:
            try:
                latency_lookup = build_latency_lookup(
                    drafts=draft_entries,
                    targets=target_entries,
                    spec=network_model_spec,
                )
            except NetworkModelError as exc:
                raise ValueError(
                    f"Cluster '{cluster_name}' network_model error: {exc}"
                ) from exc
            jitter_pct = float(network_model_spec.get("jitter_pct", jitter_pct))

        for draft in draft_entries:
            did = draft["id"]
            bucket = draft.get("bucket", 0)
            label = draft.get("label", str(bucket))
            allowed_tiers = affinity.get(label, all_tiers or list(targets_by_tier.keys()))
            candidates: list[str] = []
            for tier in allowed_tiers:
                candidates.extend(targets_by_tier.get(tier, []))
            if not candidates:
                candidates = list(tier_of.keys())
            rng.shuffle(candidates)
            fanout = max(1, int(fanout_override.get(label, fanout_base)))
            chosen = candidates[:fanout]
            for tid in chosen:
                t_tier = tier_of.get(tid)
                fwd_base = rsp_base = None
                if latency_lookup is not None:
                    base_latency = latency_lookup.get((did, tid))
                    if base_latency is None:
                        base_latency = latency_lookup.get((tid, did))
                    if base_latency is not None:
                        fwd_base = rsp_base = float(base_latency)

                if fwd_base is None or rsp_base is None:
                    fr = net_ranges.get(t_tier, net_ranges.get(label, [20, 40]))
                    fwd_base = rng.uniform(float(fr[0]), float(fr[1])) if fr else 20.0
                    rsp_base = rng.uniform(float(fr[0]), float(fr[1])) if fr else 20.0

                fwd = fwd_base
                rsp = rsp_base
                if jitter_pct:
                    jitter = rng.uniform(-jitter_pct, jitter_pct)
                    fwd = max(0.1, fwd_base * (1.0 + jitter))
                    jitter = rng.uniform(-jitter_pct, jitter_pct)
                    rsp = max(0.1, rsp_base * (1.0 + jitter))
                row = acc_tbl.get(str(bucket), acc_tbl.get(bucket, {}))
                base_acc = row.get(t_tier, 0.75)
                acc = max(0.5, min(0.99, base_acc + rng.uniform(-0.03, 0.03)))
                connection: Dict[str, Any] = {
                    "draft": did,
                    "target": tid,
                    "forward_latency_ms": float(fwd),
                    "response_latency_ms": float(rsp),
                    "acceptance_rate": float(acc),
                    "cluster": cluster_name,
                    "base_forward_latency_ms": float(fwd_base),
                    "base_response_latency_ms": float(rsp_base),
                }
                if jitter_pct:
                    connection["jitter_pct"] = jitter_pct
                if "bandwidth_mbps" in conn_spec:
                    connection["bandwidth_mbps"] = float(conn_spec.get("bandwidth_mbps", 0.0))
                if "response_bandwidth_mbps" in conn_spec:
                    connection["response_bandwidth_mbps"] = float(conn_spec.get("response_bandwidth_mbps", 0.0))
                if "link_capacity" in conn_spec:
                    connection["link_capacity"] = int(conn_spec.get("link_capacity", 1))
                drop_rate = drop_tbl.get(label, drop_tbl.get(str(bucket), None))
                if drop_rate is not None:
                    connection["drop_rate"] = float(drop_rate)
                cluster_connections.append(connection)

        if raw.get("verbose", False):
            target_count = sum(1 for dev in cluster_devices if dev['role'] == 'target')
            draft_count = sum(1 for dev in cluster_devices if dev['role'] == 'draft')
            avg_connections = len(cluster_connections) / max(1, len(draft_entries))
            print(
                f"Cluster '{cluster_name}': {target_count} targets, {draft_count} drafts\n"
                f"  Connections: {len(cluster_connections)} (avg {avg_connections:.2f} per draft)",
                flush=True,
            )

        router_override = cluster_spec.get("router")
        router_params = dict(cluster_spec.get("router_params", {})) if cluster_spec.get("router_params") else {}
        return cluster_devices, cluster_connections, router_override, router_params

    if isinstance(spec, dict) and spec.get("clusters"):
        devices: list[Dict[str, Any]] = []
        connections: list[Dict[str, Any]] = []
        cluster_router: Dict[str, str] = {}
        cluster_router_params: Dict[str, Dict[str, Any]] = {}
        for idx, cluster_spec in enumerate(spec.get("clusters", [])):
            cluster_name = str(cluster_spec.get("name", f"cluster_{idx}") or f"cluster_{idx}")
            cluster_devs, cluster_conns, router_override, router_params = _build_cluster(cluster_spec, cluster_name)
            devices.extend(cluster_devs)
            connections.extend(cluster_conns)
            if router_override:
                cluster_router[cluster_name] = router_override
            if router_params:
                cluster_router_params[cluster_name] = router_params
        raw = dict(raw)
        raw["devices"] = devices
        raw["connections"] = connections
        if cluster_router:
            raw["cluster_router"] = cluster_router
        if cluster_router_params:
            raw["cluster_router_params"] = cluster_router_params
        return raw

    # Legacy single-cluster behaviour
    devices, connections, router_override, router_params = _build_cluster(spec, "default")
    raw = dict(raw)
    raw["devices"] = devices
    raw["connections"] = connections
    if router_override:
        raw.setdefault("cluster_router", {})["default"] = router_override
    if router_params:
        raw.setdefault("cluster_router_params", {})["default"] = router_params
    return raw

def _build_config_from_mapping(data: Mapping[str, Any]) -> Config:
    raw_input = copy.deepcopy(dict(data))
    global _GLOBAL_PROFILER
    _GLOBAL_PROFILER = {
        "vidur_realtime_ms": 0.0,
        "vidur_cache_hits": 0,
        "vidur_calls": 0,
        "acceptance_proba_ms": 0.0,
        "acceptance_regressor_ms": 0.0,
        "acceptance_classifier_ms": 0.0,
        "acceptance_calls": 0,
        "acceptance_cached": 0,
        "acceptance_cache_hits": 0,
        "acceptance_cache_misses": 0,
        "acceptance_surrogate_ms": 0.0,
        "acceptance_surrogate_build_ms": 0.0,
        "acceptance_surrogate_hits": 0,
        "acceptance_surrogate_queries": 0,
        "build_ms": 0.0,
        "simulation_ms": 0.0,
    }
    raw = _expand_auto_topology(raw_input)  # Expand auto-topology if present
    wl = WorkloadCfg(**(raw.get("workload", {}) or {}))
    pm = PerformanceModelConfig(**(raw.get("performance_model", {}) or {}))
    network_cfg = dict(raw.get("network", {}) or {})
    network_enabled = bool(network_cfg.get("enabled", True))
    spec_cfg = dict(raw.get("speculation", {}) or {})
    framework = str(spec_cfg.get("framework", "vanilla")).lower()
    execution_mode = str(spec_cfg.get("execution_mode", "distributed")).lower()
    acceptance_cfg = dict(spec_cfg.get("acceptance", {}) or {})
    disable_acceptance_model = bool(acceptance_cfg.get("disable_model"))
    acceptance_model = None if disable_acceptance_model else (spec_cfg.get("acceptance_model") or acceptance_cfg.get("model") or acceptance_cfg.get("file"))
    surrogate_cfg = dict(acceptance_cfg.get("surrogate") or {})
    context_bucket = int(acceptance_cfg.get("context_bucket", 1) or 1)
    depth_bucket = int(acceptance_cfg.get("depth_bucket", 1) or 1)
    pending_bucket = int(acceptance_cfg.get("pending_bucket", 1) or 1)
    queue_bucket = int(acceptance_cfg.get("queue_bucket", 1) or 1)
    cfg = Config(
        sim_time_ms=raw.get("sim_time_ms", 10_000),
        seed=raw.get("seed", 0),
        execution_mode=raw.get("execution_mode", "blocking"),
        gamma=raw.get("gamma", 4),
        # Conversation parameters - IMPORTANT: read from YAML
        answer_length=raw.get("answer_length", 20),
        answer_length_mean=raw.get("answer_length_mean", 400),
        answer_length_std=raw.get("answer_length_std", 100),
        answer_length_min=raw.get("answer_length_min", 50),
        answer_length_max=raw.get("answer_length_max", 800),
        use_answer_distribution=raw.get("use_answer_distribution", False),
        prompt_length_min=raw.get("prompt_length_min", 10),
        prompt_length_max=raw.get("prompt_length_max", 200),
        prompt_scale_by_capability=raw.get("prompt_scale_by_capability", True),
        mixed_batching=raw.get("mixed_batching", True),
        router=raw.get("router", "round_robin"),
        router_params=raw.get("router_params", {"d_choices": 2}),
        scheduler_config=raw.get("scheduler", {}),  # Parse scheduler config
        devices=raw.get("devices", []),
        connections=raw.get("connections", []),
        cluster_router=dict(raw.get("cluster_router", {}) or {}),
        cluster_router_params={k: dict(v) for k, v in (raw.get("cluster_router_params", {}) or {}).items()},
        global_router=raw.get("global_router"),
        global_router_params=dict(raw.get("global_router_params", {}) or {}),
        trace_path=raw.get("trace_path"),
        trace_defaults=dict(raw.get("trace_defaults", {}) or {}),
        trace_replay=dict(raw.get("trace_replay", {}) or {}),
        performance_model=pm,
        workload=wl,
        think_time=ThinkTimeConfig(**(raw.get("think_time", {}) or {})),
        burn_in_ms=raw.get("burn_in_ms", 0.0),
        verbose=raw.get("verbose", True),
        debug=raw.get("debug", False),
        network_config=network_cfg,
        network_enabled=network_enabled,
        speculation_framework=framework,
        speculation_execution_mode=execution_mode,
        speculation_config=spec_cfg,
        acceptance_model_path=str(acceptance_model) if acceptance_model else None,
        acceptance_config=acceptance_cfg,
        acceptance_model_disabled=disable_acceptance_model,
        acceptance_use_classifier=bool(acceptance_cfg.get("use_classifier", True)),
        acceptance_context_bucket=context_bucket,
        acceptance_depth_bucket=depth_bucket,
        acceptance_pending_bucket=pending_bucket,
        acceptance_queue_bucket=queue_bucket,
        acceptance_surrogate_config=surrogate_cfg,
    )
    if not cfg.think_time.enabled:
        raise ValueError("Think time must be enabled for closed-loop operation")
    if str(cfg.execution_mode).lower() != 'blocking':
        raise ValueError("Execution mode must be 'blocking' for closed-loop operation")

    return cfg


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _build_config_from_mapping(raw)


def load_config_from_mapping(data: Mapping[str, Any]) -> Config:
    return _build_config_from_mapping(data)

def heartbeat_monitor(env: simpy.Environment, metrics: Metrics, targets: List[TargetServer], cfg, interval_ms: float = 1000):
    """Periodic monitor to show simulation progress"""
    print(f"[{env.now:.0f}ms] Heartbeat monitor started", flush=True)
    count = 0
    while env.now < cfg.sim_time_ms:
        yield env.timeout(interval_ms)
        count += 1
        target_info = [f"{t.p.id}:{t.queue_len()}" for t in targets]
        print(f"[{env.now:.0f}ms] HB#{count}: completed={len(metrics.completed)} queues={target_info}", flush=True)



def _build_gamma_policy(cfg: Config) -> GammaPolicy:
    spec_cfg = dict(getattr(cfg, "speculation_config", {}) or {})
    policy_cfg = dict(spec_cfg.get("gamma_policy", {}) or {})
    policy_type = str(policy_cfg.get("type", "constant")).lower()
    if policy_type in {"specpp", "spec++"}:
        return SpecPPGammaPolicy(cfg.gamma, policy_cfg)
    if policy_type == "acceptance_backoff":
        return AcceptanceBackoffGammaPolicy(cfg.gamma, policy_cfg)
    return ConstantGammaPolicy(cfg.gamma)


def build(env: simpy.Environment, cfg: Config):
    build_start = time.perf_counter()
    random.seed(cfg.seed)
    metrics = Metrics(verbose=cfg.verbose, burn_in_ms=cfg.burn_in_ms)
    targets: List[TargetServer] = []
    drafts: List[DraftServer] = []
    targets_by_cluster: Dict[str, List[TargetServer]] = defaultdict(list)
    target_cluster_map: Dict[str, str] = {}
    draft_cluster_map: Dict[str, str] = {}
    target_lookup: Dict[str, TargetServer] = {}
    target_router_map: Dict[str, object] = {}
    performance_provider = create_performance_provider(cfg.performance_model)
    trace_schedule: Optional[TraceSchedule] = None
    scheduler_cfg = dict(cfg.scheduler_config or {})
    kv_cfg = dict(scheduler_cfg.get("kv", {}) or {})
    kv_manager = KVManager(
        capacities=kv_cfg.get("capacity_tokens", {}) or {},
        default_capacity_tokens=kv_cfg.get("default_capacity_tokens", 1_000_000),
        max_utilization_pct=kv_cfg.get("max_utilization_pct", 100.0),
        paging=kv_cfg.get("paging"),
    )
    network_fabric: Optional[NetworkFabric] = None
    if cfg.network_enabled:
        network_fabric = NetworkFabric(env, cfg.network_config)

    # Require devices to be specified in config
    if not cfg.devices:
        raise ValueError("No devices specified in config. Please define at least one target and one draft device.")

    # Build from devices list
    draft_configs = []
    
    for d in cfg.devices:
        role = d.get("role", "target")
        if role == "target":
            # Calculate typical verify latency if not provided (for mixed batching)
            cluster_name = str(d.get("cluster", "default"))
            params = TargetParams(
                id=d["id"],
                model=str(d.get("model", "")),
                gpu=str(d.get("gpu", "")),
                weight=float(d.get("weight", 1.0)),
                batch_window_ms=float(d.get("batch_window_ms", 6.0)),
                batch_size=int(d.get("batch_size", 32)),
                cluster=cluster_name,
                kv_capacity_tokens=d.get("kv_capacity_tokens"),
                metadata=dict(d),
                mode=str(d.get("mode", "distributed")).lower(),
                fused_draft_profile=dict(d.get("fused_draft_profile", {})) if d.get("fused_draft_profile") else None,
            )
            target = TargetServer(
                env,
                params,
                metrics,
                performance_provider,
                cfg=cfg,
                debug=cfg.debug,
                router=None,
                kv_manager=kv_manager,
            )
            targets.append(target)
            targets_by_cluster[cluster_name].append(target)
            target_lookup[params.id] = target
            target_cluster_map[params.id] = cluster_name
            kv_manager.register_target(params.id, params.kv_capacity_tokens)
            performance_provider.register_target(
                target_id=params.id,
                model=params.model,
                hardware=params.gpu,
                metadata=dict(d),
            )
        elif role == "draft":
            # Store draft configs for later processing
            d.setdefault("cluster", "default")
            draft_configs.append(d)
            if d.get("id"):
                draft_cluster_map[str(d["id"])] = str(d.get("cluster", "default"))

    acceptance_model = None
    if getattr(cfg, "acceptance_model_disabled", False):
        if cfg.verbose:
            print("Acceptance model disabled; using fixed acceptance rates.", flush=True)
    else:
        if not cfg.acceptance_model_path:
            raise RuntimeError('Acceptance model path is required for simulation')
        try:
            acceptance_model = _ACCEPTANCE_MODEL_CACHE.get(cfg.acceptance_model_path)
            if acceptance_model is None:
                acceptance_model = AcceptanceRegressor.from_file(
                    cfg.acceptance_model_path,
                    surrogate_config=cfg.acceptance_surrogate_config,
                )
                _ACCEPTANCE_MODEL_CACHE[cfg.acceptance_model_path] = acceptance_model
                if cfg.verbose:
                    meta = getattr(acceptance_model, 'metadata', {}).get('name') or cfg.acceptance_model_path
                    print(f"Loaded acceptance model: {meta}", flush=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load acceptance model '{cfg.acceptance_model_path}': {exc}") from exc

    draft_specs: List[ReplayDraftSpec] = []
    for spec in draft_configs:
        did = spec.get('id')
        if not isinstance(did, str):
            continue
        tier = (spec.get('device_tier') or spec.get('tier') or spec.get('label') or spec.get('bucket'))
        meta = spec.get('metadata') if isinstance(spec.get('metadata'), Mapping) else {}
        if tier is None and meta:
            tier = (meta.get('device_tier') or meta.get('tier') or meta.get('bucket') or meta.get('label')
                    or meta.get('model_name') or meta.get('model') or meta.get('hardware'))
        tier_str = str(tier) if tier is not None else 'default'
        capability = float(spec.get('capability', 1.0))
        draft_specs.append(ReplayDraftSpec(draft_id=did, capability=capability, tier=tier_str))

    draft_ids: List[str] = [spec.draft_id for spec in draft_specs]

    if cfg.trace_path:
        records = list(iter_trace_records(cfg.trace_path, defaults=cfg.trace_defaults))
        if not draft_ids:
            raise ValueError("At least one draft device is required when replaying traces")
        records = _prepare_trace_records(records, draft_specs, cfg)
        trace_schedule = TraceSchedule(
            records,
            draft_specs,
            horizon_ms=cfg.sim_time_ms,
        )
        if trace_schedule.max_arrival_ms > cfg.sim_time_ms:
            cfg.sim_time_ms = max(cfg.sim_time_ms, trace_schedule.max_arrival_ms + 5_000)
        if cfg.verbose:
            print(
                f"Loaded trace with {len(records)} records (max arrival {trace_schedule.max_arrival_ms:.1f}ms)",
                flush=True,
            )

    router_instances: Dict[str, object] = {}
    for cluster_idx, (cluster_name, cluster_targets) in enumerate(targets_by_cluster.items()):
        router_name = cfg.cluster_router.get(cluster_name, cfg.router)
        RouterCls = ROUTERS.get(router_name, RoundRobinRouter)
        params = dict(cfg.router_params or {})
        params.update(cfg.cluster_router_params.get(cluster_name, {}) or {})
        if RouterCls == WeightedJSQ2Router:
            router = RouterCls(cluster_targets, d_choices=int(params.get("d_choices", 2)), debug=cfg.debug)
        elif RouterCls == JSQ2Router:
            router = RouterCls(cluster_targets, d_choices=int(params.get("d_choices", 2)))
        elif RouterCls == JSQRouter:
            seed = (cfg.seed if isinstance(cfg.seed, int) else 0) + cluster_idx * 9973
            router = RouterCls(cluster_targets, seed=seed)
        elif RouterCls == RandomRouter:
            seed = (cfg.seed if isinstance(cfg.seed, int) else 0) + cluster_idx * 9973
            router = RouterCls(cluster_targets, seed=seed)
        else:
            router = RouterCls(cluster_targets)
        router_instances[cluster_name] = router
        for target in cluster_targets:
            target.router = router
            target_router_map[target.p.id] = router

    if not router_instances and targets:
        router_name = cfg.cluster_router.get("default", cfg.router)
        RouterCls = ROUTERS.get(router_name, RoundRobinRouter)
        params = dict(cfg.router_params or {})
        params.update(cfg.cluster_router_params.get("default", {}) or {})
        if RouterCls == WeightedJSQ2Router:
            router = RouterCls(targets, d_choices=int(params.get("d_choices", 2)), debug=cfg.debug)
        elif RouterCls == JSQ2Router:
            router = RouterCls(targets, d_choices=int(params.get("d_choices", 2)))
        elif RouterCls == JSQRouter:
            seed = (cfg.seed if isinstance(cfg.seed, int) else 0)
            router = RouterCls(targets, seed=seed)
        elif RouterCls == RandomRouter:
            seed = (cfg.seed if isinstance(cfg.seed, int) else 0)
            router = RouterCls(targets, seed=seed)
        else:
            router = RouterCls(targets)
        router_instances["default"] = router
        for target in targets:
            target.router = router
            target_router_map[target.p.id] = router

    global_router = None
    global_router_name = (cfg.global_router or "weighted_cluster") if cfg.global_router is not None else "weighted_cluster"
    if str(global_router_name).lower() not in {"none", "disabled", "off"}:
        GlobalRouterCls = GLOBAL_ROUTERS.get(global_router_name, GlobalWeightedRouter)
        global_router = GlobalRouterCls(
            cluster_targets=targets_by_cluster,
            target_lookup=target_lookup,
            params=cfg.global_router_params,
        )

    scheduler = Scheduler(
        env,
        cfg,
        target_lookup=target_lookup,
        targets_by_cluster=targets_by_cluster,
        kv_manager=kv_manager,
        global_router=global_router,
        metrics=metrics,
    )

    # Validate we have at least one target and one draft
    if not targets:
        raise ValueError("No target devices specified in config. At least one target is required.")
    if not draft_configs:
        raise ValueError("No draft devices specified in config. At least one draft is required.")
    
    # Calculate total draft capability for weighted distribution
    total_capability = sum(float(d.get("capability", 1.0)) for d in draft_configs)
    if total_capability <= 0:
        raise ValueError("Total draft capability must be positive.")
    
    # Build connection matrix
    connection_map = {}  # draft_id -> {target_id -> ConnectionParams}
    for conn_cfg in cfg.connections:
        draft_id = conn_cfg.get("draft")
        target_id = conn_cfg.get("target")
        if not draft_id or not target_id:
            continue
            
        if draft_id not in connection_map:
            connection_map[draft_id] = {}

        conn_cluster = str(conn_cfg.get("cluster", target_cluster_map.get(target_id, draft_cluster_map.get(draft_id, "default"))))
        forward_latency = float(conn_cfg.get("forward_latency_ms", 0.0))
        response_latency = float(conn_cfg.get("response_latency_ms", 0.0))
        forward_base = float(conn_cfg.get("base_forward_latency_ms", forward_latency))
        response_base = float(conn_cfg.get("base_response_latency_ms", response_latency))
        bandwidth_mbps = conn_cfg.get("bandwidth_mbps")
        response_bandwidth_mbps = conn_cfg.get("response_bandwidth_mbps", bandwidth_mbps)
        link_capacity = conn_cfg.get("link_capacity")
        jitter_pct = conn_cfg.get("jitter_pct")
        if bandwidth_mbps is None:
            bandwidth_mbps = cfg.network_config.get("bandwidth_mbps") if cfg.network_config else None
        if response_bandwidth_mbps is None:
            response_bandwidth_mbps = cfg.network_config.get("response_bandwidth_mbps") if cfg.network_config else bandwidth_mbps
        if link_capacity is None:
            link_capacity = cfg.network_config.get("link_capacity") if cfg.network_config else None
        network_forward_key = None
        network_response_key = None
        if network_fabric is not None:
            network_forward_key = network_fabric.register_link(
                draft_id,
                target_id,
                base_latency_ms=forward_base,
                bandwidth_mbps=float(bandwidth_mbps) if bandwidth_mbps is not None else None,
                jitter_pct=jitter_pct,
                capacity=link_capacity,
            )
            network_response_key = network_fabric.register_link(
                target_id,
                draft_id,
                base_latency_ms=response_base,
                bandwidth_mbps=float(response_bandwidth_mbps) if response_bandwidth_mbps is not None else None,
                jitter_pct=jitter_pct,
                capacity=link_capacity,
            )
        connection_map[draft_id][target_id] = ConnectionParams(
            draft_id=draft_id,
            target_id=target_id,
            forward_latency_ms=forward_latency,
            response_latency_ms=response_latency,
            acceptance_rate=float(conn_cfg.get("acceptance_rate", 0.8)),
            cluster=conn_cluster,
            network_forward_key=network_forward_key,
            network_response_key=network_response_key,
        )
    
    # Create draft servers with connections
    gamma_policy = _build_gamma_policy(cfg)

    for d in draft_configs:
        cluster_name = draft_cluster_map.get(d.get("id"), str(d.get("cluster", "default")))
        params = DraftParams(
            id=d["id"],
            capability=float(d.get("capability", 1.0)),
            burst_factor=float(d.get("burst_factor", 1.0)),
            reliability=float(d.get("reliability", 0.99)),
            cluster=cluster_name,
            metadata=dict(d),
        )

        draft_connections_all = connection_map.get(d["id"], {})
        draft_connections = {tid: conn for tid, conn in draft_connections_all.items() if conn.cluster == cluster_name}
        if not draft_connections:
            draft_connections = draft_connections_all
        if not draft_connections:
            print(f"Warning: Draft {d['id']} has no connections configured", flush=True)
        draft_router = router_instances.get(cluster_name) or router_instances.get("default")
        if draft_router is None:
            raise ValueError(f"No router available for cluster '{cluster_name}' (draft {d['id']})")

        trace_records = trace_schedule.for_draft(d["id"]) if trace_schedule else None

        performance_provider.register_target(
            target_id=params.id,
            model=str(params.metadata.get("model", "")),
            hardware=str(params.metadata.get("gpu", "")),
            metadata=dict(d),
        )
        drafts.append(
            DraftServer(
                env=env,
                params=params,
                cfg=cfg,
                router=draft_router,
                global_router=global_router,
                target_lookup=target_lookup,
                connections=draft_connections,
                total_capability=total_capability,
                metrics=metrics,
                scheduler=scheduler,
                trace_records=trace_records,
                performance_provider=performance_provider,
                network=network_fabric,
                acceptance_model=acceptance_model,
                gamma_policy=gamma_policy,
            )
        )


    # Start heartbeat monitor
    if cfg.verbose:
        env.process(heartbeat_monitor(env, metrics, targets, cfg))
    if _GLOBAL_PROFILER is not None:
        _GLOBAL_PROFILER["build_ms"] += (time.perf_counter() - build_start) * 1000.0
    return metrics, targets, drafts, performance_provider, scheduler

def run(cfg: Config):
    env = simpy.Environment()
    result = build(env, cfg)
    perf_provider = None
    scheduler = None
    if isinstance(result, tuple):
        if len(result) == 5:
            metrics, targets, drafts, perf_provider, scheduler = result
        elif len(result) == 4:
            metrics, targets, drafts, perf_provider = result
        elif len(result) == 3:
            metrics, targets, drafts = result
        elif len(result) == 2:
            metrics, targets = result
            drafts = []
        else:
            metrics = result
            targets = []
            drafts = []
    else:
        metrics = result
        targets = []
        drafts = []
    
    # Progress reporting with correct estimation
    if cfg.verbose:
        print(f"Starting simulation for {cfg.sim_time_ms:.0f}ms...", flush=True)
        if cfg.workload.arrival == "poisson":
            expected = cfg.workload.rate_rps * cfg.sim_time_ms / 1000.0
            rate_desc = f"{cfg.workload.rate_rps} req/s"
        else:
            expected = cfg.sim_time_ms / max(cfg.workload.interarrival_ms, 1e-9)
            rate_desc = f"1 every {cfg.workload.interarrival_ms} ms"
        print(f"Generating ~{expected:.0f} requests at {rate_desc}", flush=True)
        if cfg.burn_in_ms > 0:
            print(f"Burn-in period: {cfg.burn_in_ms:.0f}ms", flush=True)
    
    # Initialize token metrics timing
    metrics.token_metrics.start_time_ms = cfg.burn_in_ms  # Start after burn-in
    
    start_real_time = time.time()
    sim_start = time.perf_counter()
    print(f"Starting env.run(until={cfg.sim_time_ms})...", flush=True)
    
    # Run with intermediate checkpoints to see progress
    checkpoints = [1000, 5000, 10000, 20000, 30000]
    last_checkpoint = 0
    for checkpoint in checkpoints:
        if checkpoint <= cfg.sim_time_ms:
            print(f"About to run until {checkpoint}ms (currently at {env.now}ms)...", flush=True)
            try:
                env.run(until=checkpoint)
                print(f"[{env.now:.0f}ms] Checkpoint reached, jobs completed: {len(metrics.completed)}", flush=True)
            except Exception as e:
                print(f"ERROR at {env.now}ms: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break
            if env.now == last_checkpoint:
                print(f"WARNING: Simulation stuck at {env.now}ms!", flush=True)
                # Print debugging info
                print(f"Active processes: {len(env.active_process)}")
                print(f"Event queue empty: {len(env._queue) == 0}")
                break
            last_checkpoint = env.now
    
    # Run to the full simulation time if we haven't reached it yet
    if env.now < cfg.sim_time_ms:
        print(f"Running final segment from {env.now}ms to {cfg.sim_time_ms}ms...", flush=True)
        try:
            env.run(until=cfg.sim_time_ms)
            print(f"[{env.now:.0f}ms] Simulation complete, jobs completed: {len(metrics.completed)}", flush=True)
        except Exception as e:
            print(f"ERROR at {env.now}ms: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    # Drain any in-flight work that was queued before sim_time_ms
    def _pending_work() -> bool:
        if scheduler is not None:
            if getattr(scheduler.prefill_scheduler.store, "items", None):
                if len(scheduler.prefill_scheduler.store.items) > 0:
                    return True
            if getattr(scheduler.decode_scheduler.store, "items", None):
                if len(scheduler.decode_scheduler.store.items) > 0:
                    return True
        for target in targets:
            if getattr(target, "_enqueued_count", 0) > 0:
                return True
            if getattr(target, "_busy", False):
                return True
            if hasattr(target, "pending_decode_tokens") and target.pending_decode_tokens() > 0:
                return True
        return False

    if _pending_work():
        print(f"Draining in-flight work after {cfg.sim_time_ms}ms horizon...", flush=True)
        drain_step_ms = max(1.0, float(getattr(cfg, "drain_step_ms", 50.0)))
        drain_deadline = env.now + max(float(getattr(cfg, "drain_timeout_ms", 5000.0)), float(cfg.sim_time_ms or 0))
        while _pending_work():
            current = env.now
            next_until = min(current + drain_step_ms, drain_deadline)
            if next_until <= current:
                break
            env.run(until=next_until)
            if env.now == current:
                print("Drain loop made no progress; breaking to avoid stall", flush=True)
                break
            if env.now >= drain_deadline and _pending_work():
                print("Drain time exceeded; pending work may remain", flush=True)
                break
        print(f"[DRAIN] env.run() completed at {env.now:.0f}ms (jobs completed: {len(metrics.completed)})", flush=True)
    else:
        print(f"env.run() completed at {env.now:.0f}ms", flush=True)

    elapsed_real_time = time.time() - start_real_time
    
    # Finalize token metrics
    metrics.token_metrics.end_time_ms = env.now
    if perf_provider is not None:
        try:
            perf_provider.flush()
        except Exception as exc:  # pragma: no cover - flushing should not break the run
            if cfg.verbose:
                print(f"Warning: failed to flush performance provider cache: {exc}", flush=True)
    
    if cfg.verbose:
        print(f"\nSimulation complete in {elapsed_real_time:.2f}s real time")
        print(f"Processed {len(metrics.completed)} total jobs")
    if _GLOBAL_PROFILER is not None:
        _GLOBAL_PROFILER["simulation_ms"] += (time.perf_counter() - sim_start) * 1000.0
    return metrics, targets, drafts

def get_target_metrics(targets: List[TargetServer], sim_time_ms: float) -> Dict[str, Dict[str, float]]:
    """Extract per-target metrics."""
    results = {}
    for t in targets:
        utilization = t.busy_ms / sim_time_ms if sim_time_ms > 0 else 0
        avg_batch_size = t.total_batch_items / t.total_batches if t.total_batches > 0 else 0
        
        # Queue length stats
        queue_lengths = [ql for _, ql in t.queue_samples]
        avg_queue = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0
        p95_queue = sorted(queue_lengths)[int(0.95 * len(queue_lengths))] if queue_lengths else 0
        
        results[t.p.id] = {
            "utilization": utilization,
            "avg_batch_size": avg_batch_size,
            "total_batches": t.total_batches,
            "avg_queue_len": avg_queue,
            "p95_queue_len": p95_queue,
        }
        
        # Track tier utilization if available
        tier = getattr(t.p, "gpu", "") or getattr(t.p, "model", "") or t.p.id
        if hasattr(targets[0].metrics, "tier_utilization"):
            targets[0].metrics.tier_utilization[tier].append(utilization)
    
    return results

def get_draft_metrics(drafts: List[DraftServer]) -> Dict[str, Dict[str, Any]]:
    """Extract per-draft metrics for speculative decoding."""
    results = {}
    for d in drafts:
        generated = max(0, d.total_tokens_generated)
        if d.chunks_sent > 0 and generated > 0:
            acceptance_rate = d.total_tokens_accepted / generated
            avg_rtt = d.total_round_trip_time / d.chunks_sent
        elif d.chunks_sent > 0:
            acceptance_rate = 0.0
            avg_rtt = d.total_round_trip_time / d.chunks_sent
        else:
            acceptance_rate = 0.0
            avg_rtt = 0.0
            
        results[d.p.id] = {
            "capability": d.p.capability,
            "chunks_sent": d.chunks_sent,
            "tokens_generated": generated,
            "tokens_accepted": d.total_tokens_accepted,
            "tokens_rejected": d.total_tokens_rejected,
            "acceptance_rate": acceptance_rate,
            "avg_rtt_ms": avg_rtt,
        }
    return results

def _unpack_run_result(result):
    if isinstance(result, tuple) and len(result) == 3:
        return result
    if isinstance(result, tuple) and len(result) == 2:
        metrics, targets = result
        return metrics, targets, []
    return result, [], []


def _collect_metrics_json(cfg: Config, metrics, summary: Dict[str, float], targets: List[TargetServer]) -> Dict[str, float]:
    import statistics

    metrics_json = {
        "scheduler": cfg.scheduler_config.get("type", "baseline"),
        "seed": cfg.seed,
        "load_rps": cfg.workload.rate_rps,
        "avg_latency_ms": summary.get("avg_ms", 0),
        "p50_latency_ms": summary.get("p50_ms", 0),
        "p95_latency_ms": summary.get("p95_ms", 0),
        "p99_latency_ms": summary.get("p99_ms", 0),
        "rtt_avg_ms": summary.get("rtt_avg_ms", 0),
        "rtt_p50_ms": summary.get("rtt_p50_ms", 0),
        "rtt_p95_ms": summary.get("rtt_p95_ms", 0),
        "rtt_p99_ms": summary.get("rtt_p99_ms", 0),
        "throughput_jobs_s": summary.get("throughput_jobs_s", 0),
        "throughput_busy_jobs_s": 0.0,
        "target_tokens_per_s": 0.0,
        "acceptance_rate": metrics.token_metrics.get_acceptance_rate() if hasattr(metrics, 'token_metrics') else 0,
        "effective_tok_s": metrics.token_metrics.get_effective_tokens_per_second() if hasattr(metrics, 'token_metrics') else 0,
        "avg_conversation_ms": summary.get("avg_conversation_ms", 0),
        "p50_conversation_ms": summary.get("p50_conversation_ms", 0),
        "p95_conversation_ms": summary.get("p95_conversation_ms", 0),
        "p99_conversation_ms": summary.get("p99_conversation_ms", 0),
        "conversation_count": summary.get("conversation_count", 0),
        "completed_conversation_count": summary.get("completed_conversation_count", 0),
        "conversation_completion_rate": summary.get("conversation_completion_rate", 0),
        "ttft_avg_ms": summary.get("ttft_avg_ms", 0),
        "ttft_p50_ms": summary.get("ttft_p50_ms", 0),
        "ttft_p95_ms": summary.get("ttft_p95_ms", 0),
        "ttft_p99_ms": summary.get("ttft_p99_ms", 0),
        "ttft_count": summary.get("ttft_count", 0),
        "tpot_avg_ms": summary.get("tpot_avg_ms", 0),
        "tpot_p50_ms": summary.get("tpot_p50_ms", 0),
        "tpot_p95_ms": summary.get("tpot_p95_ms", 0),
        "tpot_p99_ms": summary.get("tpot_p99_ms", 0),
        "tpot_count": summary.get("tpot_count", 0),
        "slo_attainment_rate": summary.get("slo_attainment_rate", 0),
        "slo_meeting_count": summary.get("slo_meeting_count", 0),
        "goodput_rps": summary.get("goodput_rps", 0),
        "conversation_throughput_rps": 0.0,
        "observed_span_ms": summary.get("observed_span_ms", cfg.sim_time_ms),
    }

    for key in (
        "avg_conversation_ms_completed",
        "p50_conversation_ms_completed",
        "p95_conversation_ms_completed",
        "p99_conversation_ms_completed",
    ):
        if key in summary:
            metrics_json[key] = summary[key]

    span_ms = metrics_json["observed_span_ms"] if metrics_json["observed_span_ms"] is not None else cfg.sim_time_ms
    if span_ms and span_ms > 0:
        metrics_json["conversation_throughput_rps"] = summary.get("completed_conversation_count", 0) / (span_ms / 1000.0)

    for key, value in summary.items():
        if key.startswith("ttft_breakdown_") or key.startswith("decode_breakdown_"):
            metrics_json[key] = value

    # Calculate target throughput as actual service rate of prefill jobs
    # Use total service time spent processing prefills to remove queueing artifacts
    target_throughput_jobs_s = 0.0
    if targets:
        total_prefill_jobs = 0
        total_prefill_service_ms = 0.0
        for t in targets:
            total_prefill_jobs += getattr(t, "prefill_jobs_processed", 0)
            total_prefill_service_ms += getattr(t, "prefill_service_ms", 0.0)

        if total_prefill_service_ms > 0:
            target_throughput_jobs_s = 1000.0 * total_prefill_jobs / total_prefill_service_ms

    metrics_json["target_throughput_jobs_s"] = target_throughput_jobs_s

    if targets:
        all_batch_sizes = []
        prefill_counts = []
        decode_counts = []
        total_busy_ms = 0.0
        total_processed_tokens = 0
        for t in targets:
            total_busy_ms += getattr(t, "busy_ms", 0.0)
            total_processed_tokens += getattr(t, "total_processed_tokens", 0)
            if hasattr(t, 'batch_history'):
                for batch in t.batch_history:
                    all_batch_sizes.append(len(batch))
                    prefill_counts.append(sum(1 for j in batch if j.job_type == "prefill"))
                    decode_counts.append(sum(1 for j in batch if j.job_type == "decode"))

        if all_batch_sizes:
            metrics_json["avg_batch_size"] = statistics.mean(all_batch_sizes)
            if prefill_counts:
                metrics_json["avg_prefills_per_batch"] = statistics.mean(prefill_counts)
            if decode_counts:
                metrics_json["avg_decodes_per_batch"] = statistics.mean(decode_counts)
        span_ms = metrics_json.get("observed_span_ms") or cfg.sim_time_ms
        target_count = len(targets)
        if span_ms and span_ms > 0 and target_count > 0:
            utilization = total_busy_ms / (span_ms * target_count)
            metrics_json["target_utilization_pct"] = max(0.0, min(utilization, 1.0))
        else:
            metrics_json["target_utilization_pct"] = 0.0

        decode_jobs = float(summary.get("decode_jobs_count", 0))
        if total_busy_ms > 0:
            metrics_json["throughput_busy_jobs_s"] = (decode_jobs * 1000.0) / total_busy_ms
            metrics_json["target_tokens_per_s"] = (total_processed_tokens * 1000.0) / total_busy_ms
        else:
            metrics_json["throughput_busy_jobs_s"] = 0.0
            metrics_json["target_tokens_per_s"] = 0.0
    else:
        metrics_json["target_utilization_pct"] = 0.0
        metrics_json["throughput_busy_jobs_s"] = 0.0
        metrics_json["target_tokens_per_s"] = 0.0

    return metrics_json


def _print_report(cfg: Config, metrics, summary: Dict[str, float], targets: List[TargetServer], drafts: List[DraftServer], metrics_json: Dict[str, float]) -> None:
    # Print results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    # THE ONE METRIC THAT MATTERS
    if hasattr(metrics, 'token_metrics') and metrics.token_metrics.total_generated_tokens > 0:
        effective_tps = metrics.token_metrics.get_effective_tokens_per_second()
        acceptance_rate = metrics.token_metrics.get_acceptance_rate()
        print("\n🎯 TOKEN PERFORMANCE (THE KEY METRIC):")
        print(f"  Effective Tokens/Second: {effective_tps:.1f} tok/s")
        print(f"  Acceptance Rate: {acceptance_rate:.1%}")
        print(f"  Total Accepted: {metrics.token_metrics.total_accepted_tokens}")
        print(f"  Total Generated: {metrics.token_metrics.total_generated_tokens}")
        print(f"  Wasted Work: {metrics.token_metrics.total_rejected_tokens} ({metrics.token_metrics.total_rejected_tokens/max(1,metrics.token_metrics.total_generated_tokens)*100:.1f}%)")
    
    print("\nSummary Metrics:")
    for key, value in summary.items():
        if key == "count":
            print(f"  {key}: {value}")
        elif "ms" in key:
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:.2f}")
    
    if targets:
        print("\nPer-Target Metrics:")
        target_metrics = get_target_metrics(targets, cfg.sim_time_ms)
        for tid, metrics in target_metrics.items():
            print(f"  {tid}:")
            print(f"    Utilization: {metrics['utilization']:.2%}")
            print(f"    Avg batch size: {metrics['avg_batch_size']:.1f}")
            print(f"    Total batches: {metrics['total_batches']}")
            print(f"    Avg queue len: {metrics['avg_queue_len']:.1f}")
            print(f"    P95 queue len: {metrics['p95_queue_len']:.0f}")
        
        # CRITICAL: Verify single-server enforcement
        print("\n🔍 Concurrency Check (MUST be 1 for correct simulation):")
        for target in targets:
            print(f"  {target.p.id}: max_concurrency = {target.max_concurrency}")
            if target.max_concurrency > 1:
                print(f"    ⚠️ WARNING: Parallelism detected! Fix required.")
    
    if drafts:
        print("\nPer-Draft Metrics (Speculative Decoding):")
        draft_metrics = get_draft_metrics(drafts)
        for did, metrics in draft_metrics.items():
            print(f"  {did}:")
            print(f"    Capability: {metrics['capability']:.1f}x")
            print(f"    Chunks sent: {metrics['chunks_sent']}")
            print(f"    Tokens generated: {metrics['tokens_generated']}")
            print(f"    Tokens accepted: {metrics['tokens_accepted']}")
            print(f"    Acceptance rate: {metrics['acceptance_rate']:.2%}")
            print(f"    Avg RTT: {metrics['avg_rtt_ms']:.1f}ms")
    
    # Final reminder of the key metric
    if hasattr(metrics, 'token_metrics') and metrics.token_metrics.total_generated_tokens > 0:
        print("\n" + "="*60)
        print(f"🚀 FINAL RESULT: {metrics.token_metrics.get_effective_tokens_per_second():.1f} tokens per second")
        print("="*60)
    
    # Output JSON metrics for analysis
    print("\n===METRICS_JSON===")
    print(json.dumps(metrics_json, indent=2))
    print("===END_METRICS_JSON===")


def simulate_config_obj(cfg: Config, *, emit_output: bool = True) -> Dict[str, float]:
    global _GLOBAL_PROFILER
    result = run(cfg)
    metrics, targets, drafts = _unpack_run_result(result)
    summary = metrics.summary() if hasattr(metrics, 'summary') else {}
    metrics_json = _collect_metrics_json(cfg, metrics, summary, targets)
    if _GLOBAL_PROFILER:
        metrics_json.setdefault("profiler", {}).update(_GLOBAL_PROFILER)
    if emit_output:
        _print_report(cfg, metrics, summary, targets, drafts, metrics_json)
    return metrics_json


def simulate_config(config_path: str, *, emit_output: bool = True) -> Dict[str, float]:
    cfg = load_config(config_path)
    return simulate_config_obj(cfg, emit_output=emit_output)


def _maybe_plot(results: List[tuple[str, Dict[str, float]]], output_path: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot") from exc

    labels = [label for label, _ in results]
    avg_latencies = [metrics.get("avg_latency_ms", 0.0) for _, metrics in results]
    rtt_latencies = [metrics.get("rtt_avg_ms", 0.0) for _, metrics in results]

    x = range(len(labels))
    bar_width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar([i - bar_width / 2 for i in x], avg_latencies, width=bar_width, label="Target avg latency (ms)")
    plt.bar([i + bar_width / 2 for i in x], rtt_latencies, width=bar_width, label="Conversation RTT avg (ms)")
    plt.ylabel("Latency (ms)")
    plt.xticks(list(x), labels, rotation=15)
    plt.title("Average Latency Comparison")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main():
    repo_root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/config.yaml")
    ap.add_argument("--label", help="Optional label for the primary config when plotting.")
    ap.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Additional configs to compare, formatted as label=path/to/config.yaml",
    )
    ap.add_argument(
        "--plot",
        nargs="?",
        const="AUTO",
        help="Generate latency comparison plot (optionally provide output path).",
    )
    ap.add_argument(
        "--metrics-json",
        help="Optional path to write aggregated metrics JSON when using --plot or --compare.",
    )

    args = ap.parse_args()

    config_entries: List[tuple[str, Path]] = []
    primary_path = Path(args.config)
    primary_label = args.label or primary_path.stem
    config_entries.append((primary_label, primary_path))

    for entry in args.compare:
        if "=" not in entry:
            ap.error(f"Invalid --compare entry '{entry}'. Expected label=path format.")
        label, path = entry.split("=", 1)
        config_entries.append((label, Path(path).resolve()))

    if args.plot and len(config_entries) == 1 and not args.compare:
        default_compare = repo_root / "explorer" / "output" / "baseline_vs_disagg" / "disagg_baseline" / "configs" / "disagg_baseline.yaml"
        if default_compare.exists():
            config_entries.append(("disagg_baseline", default_compare))

    aggregated: List[tuple[str, Dict[str, float]]] = []
    for idx, (label, path) in enumerate(config_entries):
        emit = not args.plot or idx == 0
        metrics_json = simulate_config(str(path), emit_output=emit)
        aggregated.append((label, metrics_json))

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump({label: metrics for label, metrics in aggregated}, fh, indent=2)

    if args.plot:
        output_path = Path(args.plot if args.plot != "AUTO" else repo_root / "experiments" / "results" / "latency_comparison.png")
        plot_path = _maybe_plot(aggregated, output_path.resolve())
        print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
