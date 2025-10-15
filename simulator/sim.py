# sim.py v1 minimal distributed speculative-decoding simulator
# deps: pip install simpy pyyaml

import argparse, random, simpy, yaml, math, json
import time
from collections import deque, defaultdict
from types import MappingProxyType
from pathlib import Path

# Print SimPy version for debugging
print(f"SimPy version: {simpy.__version__}", flush=True)
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Sequence, Iterable, Tuple, Mapping

from performance import PhaseRequest, PerformanceModelConfig, create_performance_provider
from network.topology import build_latency_lookup, NetworkModelError
from network.fabric import NetworkFabric
from trace.trace_loader import iter_trace_records
from trace.types import TraceRecord, TraceParseError

# ---------- Config & Types ----------

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

@dataclass
class DraftParams:
    id: str
    capability: float = 1.0           # relative compute speed (affects generation rate)
    burst_factor: float = 1.0         # short-term burst multiplier
    reliability: float = 0.99         # connection reliability (0-1)
    cluster: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConnectionParams:
    draft_id: str
    target_id: str
    forward_latency_ms: float         # draft -> target latency
    response_latency_ms: float        # target -> draft latency
    acceptance_rate: float            # probability each token is accepted
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
class WorkloadCfg:
    arrival: str = "deterministic"    # "deterministic" | "poisson"
    interarrival_ms: float = 12.0     # used if deterministic
    rate_rps: float = 100.0           # used if poisson: mean rate

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
    performance_model: PerformanceModelConfig = field(default_factory=PerformanceModelConfig)
    workload: WorkloadCfg = field(default_factory=WorkloadCfg)
    think_time: ThinkTimeConfig = field(default_factory=ThinkTimeConfig)
    burn_in_ms: float = 0.0           # Ignore first X ms for stats
    verbose: bool = True               # Print progress updates
    debug: bool = False                # Print detailed batch formation
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


class TraceSchedule:
    """Distribute trace records across drafts for replay."""

    def __init__(self, records: Sequence[TraceRecord], known_drafts: Sequence[str]):
        self._by_draft: Dict[str, List[TraceRecord]] = defaultdict(list)
        self.max_arrival_ms = 0.0
        known = set(known_drafts)
        for record in records:
            draft_id = record.draft_id
            if draft_id is None:
                raise TraceParseError("trace playback requires draft_id for each record")
            if draft_id not in known:
                raise TraceParseError(f"trace references unknown draft_id '{draft_id}'")
            self._by_draft[draft_id].append(record)
            self.max_arrival_ms = max(self.max_arrival_ms, record.arrival_ms)
        for recs in self._by_draft.values():
            recs.sort(key=lambda r: r.arrival_ms)

    def for_draft(self, draft_id: str) -> Sequence[TraceRecord]:
        return list(self._by_draft.get(draft_id, []))

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
        
        self.proc = env.process(self._serve_loop())

    # queue helpers
    def enqueue(self, job: Job):
        self._enqueued_count += 1
        self.q.put(job)
        # Debug: print if queue is getting very large
        if self._enqueued_count > 100 and self._enqueued_count % 50 == 0:
            print(f"[{self.env.now:.1f}ms] WARNING: Target {self.p.id} queue size: {self._enqueued_count}", flush=True)

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
        provider = getattr(self, "performance", None)
        if provider is None:
            return 0.0
        phase = "prefill" if job.job_type == "prefill" else "verify"
        request = PhaseRequest(
            phase=phase,
            model=self.p.model,
            hardware=self.p.gpu,
            batch_size=1,
            microbatch_size=1,
            fanout=max(1, self.cfg.gamma if job.job_type != "prefill" else 1),
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

        return PhaseRequest(
            phase=phase,
            model=self.p.model,
            hardware=self.p.gpu,
            batch_size=batch_size,
            microbatch_size=min(batch_size, self.p.batch_size),
            fanout=self.cfg.gamma if phase != "prefill" else 1,
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
                                deferred_prefills.append(job)
                                continue  # Skip this prefill for now
                        
                        batch.append(job)
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
                if self.cfg.verbose or (self.debug and self.total_batches < 5):  # Print if verbose or first 5 batches
                    prefill_count = sum(1 for j in batch if j.job_type == "prefill")
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
                
                tdone = self.env.now
                for j in batch:
                    j.finished_ms = tdone
                    if self.kv_manager:
                        self.kv_manager.release(self.p.id, max(0, j.kv_tokens))
                    self.metrics.add(j)
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
                 performance_provider=None, network: Optional[NetworkFabric] = None):
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
    
    def _simulate_verification(self, tokens: int, acceptance_rate: float) -> VerifyResult:
        """Simulate target verification of draft tokens"""
        accepted = 0
        for i in range(tokens):
            if random.random() < acceptance_rate:
                accepted += 1
            else:
                # First rejection stops acceptance
                break
        
        rejected = tokens - accepted
        return VerifyResult(
            chunk_id=self.chunks_sent,
            accepted_tokens=accepted,
            rejected_tokens=rejected,
            total_tokens=tokens
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
            fanout=max(1, self.gamma),
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

        while self.env.now < self.cfg.sim_time_ms:
            trace_record: Optional[TraceRecord] = None
            if self._trace_mode:
                if self._trace_index >= len(records):
                    break
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

            priority_class = trace_record.slo_class if trace_record is not None else None
            if not priority_class:
                priority_class = (self.cfg.trace_defaults or {}).get("slo_class")
            if not priority_class and self.scheduler is not None:
                priority_class = self.scheduler.default_priority_class
            priority_class = priority_class or "standard"

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
            )
            self.chunks_sent += 1

            # Send prefill request to target
            prefill_payload = self._estimate_payload_bytes(prompt_length, self._prefill_overhead_bytes)
            fwd_start = self.env.now
            yield self._network_transfer(
                self.id,
                target_id,
                conn.forward_latency_ms,
                payload_bytes=prefill_payload,
                link_key=conn.network_forward_key,
            )
            ttft_breakdown["prefill_forward_ms"] += self.env.now - fwd_start
            prefill_job.created_ms = self.env.now

            if self.scheduler is None:
                target_server = self._target_lookup.get(target_id)
                if target_server is None:
                    print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
                    continue
                target_server.enqueue(prefill_job)
                wait_event = prefill_completion
            else:
                wait_event = self.scheduler.submit_job(prefill_job)

            # Wait for actual job completion instead of synthetic sleep
            yield wait_event

            # Wait for response to travel back
            response_payload = self._estimate_payload_bytes(prompt_length, self._response_overhead_bytes)
            rsp_start = self.env.now
            yield self._network_transfer(
                target_id,
                self.id,
                conn.response_latency_ms,
                payload_bytes=response_payload,
                link_key=conn.network_response_key,
            )
            ttft_breakdown["prefill_response_ms"] += self.env.now - rsp_start

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
            rounds_needed = (answer_length + self.gamma - 1) // self.gamma  # ceiling division
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
                    tokens_this_round = min(self.gamma, tokens_remaining)

                    current_context = prompt_length + tokens_generated_in_conversation
                    # Generate draft tokens using performance provider latency
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
                    )

                    # Send chunk to target (forward latency)
                decode_payload = self._estimate_payload_bytes(tokens_this_round, self._decode_overhead_bytes)
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

                    # Job arrives at target after network delay
                    job.created_ms = self.env.now

                    if self.scheduler is None:
                        target_server = self._target_lookup.get(target_id)
                        if target_server is None:
                            print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
                            break
                        target_server.enqueue(job)
                        wait_event = decode_completion
                    else:
                        wait_event = self.scheduler.submit_job(job)

                    # Wait for actual job completion instead of synthetic sleep
                yield wait_event

                # Wait for response to travel back
                response_payload = self._estimate_payload_bytes(tokens_this_round, self._response_overhead_bytes)
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
                    result = self._simulate_verification(tokens_this_round, conn.acceptance_rate)
                    job.accepted_tokens = result.accepted_tokens
                    self.total_tokens_accepted += result.accepted_tokens
                    self.total_tokens_rejected += result.rejected_tokens
                    tokens_accepted_in_conversation += result.accepted_tokens

                    # Calculate round-trip time for this chunk
                    rtt = self.env.now - round_start
                    self.total_round_trip_time += rtt

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

                    # Update global token metrics (only count after burn-in)
                    if self.env.now >= self.cfg.burn_in_ms:
                        self.metrics.token_metrics.total_generated_tokens += tokens_this_round
                        self.metrics.token_metrics.total_accepted_tokens += result.accepted_tokens
                        self.metrics.token_metrics.total_rejected_tokens += result.rejected_tokens

                    if self.cfg.debug:
                        print(f"[{self.env.now:.1f}ms] Draft {self.id}: Round {round_num+1} result: "
                              f"{result.accepted_tokens}/{tokens_this_round} accepted, RTT={rtt:.1f}ms", flush=True)
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

            # Schedule next arrival for both think_enabled and workload-driven modes
            if not self._trace_mode:
                self._schedule_next_arrival(self.env.now)

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
        if self.settings.queue_policy == "fifo":
            sort_key: Tuple[float, int] = (float(self._seq),)
        else:
            sort_key = (float(job.priority), self._seq)
        item = QueueItem(sort_key, job)

        if self.settings.max_queue_depth and self.settings.max_queue_depth > 0:
            if len(self.store.items) + self._pending_enqueues >= self.settings.max_queue_depth:
                self._schedule_backpressure(item)
                return

        self.store.put(item)

    def _run(self):
        while True:
            item = yield self.store.get()
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
                    batch.append(next_job)
                    token_total += next_job.token_count
                    if self.settings.max_batch_tokens and token_total >= self.settings.max_batch_tokens:
                        break
                else:
                    break

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
            self.store.put(item)

        self.env.process(_deferred_enqueue())

    def _dispatch_batch_proc(self, batch: List[Job]):
        groups: Dict[str, List[Job]] = {}
        for job in batch:
            target = self._choose_target(job)
            job.target_id = target.p.id
            groups.setdefault(target.p.id, []).append(job)

        for target_id, jobs in groups.items():
            target = self.pool_map.get(target_id) or self.target_lookup.get(target_id)
            if target is None:
                continue
            accepted: List[Job] = []
            max_penalty = 0.0
            for job in jobs:
                if job.kv_tokens <= 0:
                    job.kv_tokens = self._estimate_kv_tokens(job)
                if self.kv_manager is None:
                    accepted.append(job)
                    continue
                success, penalty = self.kv_manager.reserve(target_id, job.kv_tokens)
                if not success:
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

        job.retry_count += 1

        def _retry_proc():
            yield self.env.timeout(max(0.0, self.settings.retry_backoff_ms * job.retry_count))
            self._enqueue_job(job)

        self.env.process(_retry_proc())

    def _choose_target(self, job: Job) -> TargetServer:
        if job.target_id and job.target_id in self.pool_map:
            return self.pool_map[job.target_id]

        allowed_ids = list(self.pool_map.keys())
        chosen = None
        if self.global_router is not None:
            chosen = self.global_router.choose(job.draft_id, allowed_ids)

        if chosen is not None:
            return chosen

        # Fall back to least work-left among pool
        return min(self.pool_targets, key=lambda t: t.work_left_score())

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
                cfg.network_config.get("jitter_pct", 0.0),
            )
            or 0.0
        )
        drop_tbl = conn_spec.get("drop_rate", {})
        network_model_spec = None
        if cfg.network_enabled:
            network_model_spec = conn_spec.get("network_model")
            if not network_model_spec:
                network_model_spec = cfg.network_config.get("model")
        latency_lookup = None
        if cfg.network_enabled and network_model_spec:
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

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}  # safe_load is the secure choice for untrusted YAML
    raw = _expand_auto_topology(raw)  # Expand auto-topology if present
    wl = WorkloadCfg(**(raw.get("workload", {}) or {}))
    pm = PerformanceModelConfig(**(raw.get("performance_model", {}) or {}))
    network_cfg = dict(raw.get("network", {}) or {})
    network_enabled = bool(network_cfg.get("enabled", True))
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
        performance_model=pm,
        workload=wl,
        think_time=ThinkTimeConfig(**(raw.get("think_time", {}) or {})),
        burn_in_ms=raw.get("burn_in_ms", 0.0),
        verbose=raw.get("verbose", True),
        debug=raw.get("debug", False),
        network_config=network_cfg,
        network_enabled=network_enabled,
    )
    return cfg

def heartbeat_monitor(env: simpy.Environment, metrics: Metrics, targets: List[TargetServer], cfg, interval_ms: float = 1000):
    """Periodic monitor to show simulation progress"""
    print(f"[{env.now:.0f}ms] Heartbeat monitor started", flush=True)
    count = 0
    while env.now < cfg.sim_time_ms:
        yield env.timeout(interval_ms)
        count += 1
        target_info = [f"{t.p.id}:{t.queue_len()}" for t in targets]
        print(f"[{env.now:.0f}ms] HB#{count}: completed={len(metrics.completed)} queues={target_info}", flush=True)

def build(env: simpy.Environment, cfg: Config):
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

    if cfg.trace_path:
        records = list(iter_trace_records(cfg.trace_path, defaults=cfg.trace_defaults))
        draft_ids = [d.get("id") for d in draft_configs]
        if not all(isinstance(x, str) for x in draft_ids):
            raise ValueError("All draft devices must have an 'id' when replaying traces")
        trace_schedule = TraceSchedule(records, [x for x in draft_ids if isinstance(x, str)])
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
            )
        )


    # Start heartbeat monitor
    if cfg.verbose:
        env.process(heartbeat_monitor(env, metrics, targets, cfg))
    
    return metrics, targets, drafts, performance_provider

def run(cfg: Config):
    env = simpy.Environment()
    result = build(env, cfg)
    perf_provider = None
    if isinstance(result, tuple):
        if len(result) == 4:
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
    
    elapsed_real_time = time.time() - start_real_time
    print(f"env.run() completed at {env.now:.0f}ms", flush=True)
    
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

    if targets:
        all_batch_sizes = []
        prefill_counts = []
        decode_counts = []
        for t in targets:
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


def simulate_config(config_path: str, *, emit_output: bool = True) -> Dict[str, float]:
    cfg = load_config(config_path)
    result = run(cfg)
    metrics, targets, drafts = _unpack_run_result(result)
    summary = metrics.summary() if hasattr(metrics, 'summary') else {}
    metrics_json = _collect_metrics_json(cfg, metrics, summary, targets)
    if emit_output:
        _print_report(cfg, metrics, summary, targets, drafts, metrics_json)
    return metrics_json


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
