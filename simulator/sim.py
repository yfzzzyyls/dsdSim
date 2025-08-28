# sim.py v1 minimal distributed speculative-decoding simulator
# deps: pip install simpy pyyaml

import argparse, random, simpy, yaml, math
import time
from collections import deque

# Print SimPy version for debugging
print(f"SimPy version: {simpy.__version__}", flush=True)
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# ---------- Config & Types ----------

@dataclass
class TargetParams:
    id: str
    model: str = ""                   # optional metadata (e.g., "llama-3.1-8b")
    gpu: str = ""                     # optional metadata (e.g., "A100", "H100", "L4")
    weight: float = 1.0               # capacity weight (relative speed)
    batch_window_ms: float = 6.0      # Delta: wait to form a batch
    batch_size: int = 32              # B: max jobs per batch
    verify_latency_ms: float = 8.0    # fixed service time per batch (legacy, for non-mixed mode)
    prefill_latency_per_token: float = 0.5  # ms per token for prefill requests
    decode_latency_per_token: float = 2.5   # ms per token for decode requests

@dataclass
class DraftParams:
    id: str
    capability: float = 1.0           # relative compute speed (affects generation rate)
    generation_latency_ms: float = 0.0  # time to generate gamma tokens (0 = instant)
    burst_factor: float = 1.0         # short-term burst multiplier
    reliability: float = 0.99         # connection reliability (0-1)

@dataclass
class ConnectionParams:
    draft_id: str
    target_id: str
    forward_latency_ms: float         # draft -> target latency
    response_latency_ms: float        # target -> draft latency
    acceptance_rate: float            # probability each token is accepted

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
    workload: WorkloadCfg = field(default_factory=WorkloadCfg)
    burn_in_ms: float = 0.0           # Ignore first X ms for stats
    verbose: bool = True               # Print progress updates
    debug: bool = False                # Print detailed batch formation

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
        if self.end_time_ms == 0 or self.start_time_ms == 0:
            return 0
        duration_s = (self.end_time_ms - self.start_time_ms) / 1000.0
        if duration_s <= 0:
            return 0
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

    def add(self, job: Job):
        self.completed.append(job)
        # Progress indicator every 100 jobs
        if self.verbose and len(self.completed) % 100 == 0:
            print(f"  Completed {len(self.completed)} jobs...")

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
        }
        
        # Add RTT metrics if available
        if rtt:
            result["rtt_avg_ms"] = sum(rtt)/len(rtt)
            result["rtt_p50_ms"] = pct_rtt(0.50)
            result["rtt_p95_ms"] = pct_rtt(0.95)
            result["rtt_p99_ms"] = pct_rtt(0.99)
            result["rtt_count"] = len(rtt)
        
        return result

# ---------- Helper Functions ----------

def get_typical_verify_ms(target_config: dict, gamma: int = 4) -> float:
    """Calculate typical verification time for mixed batching or return verify_latency_ms if available."""
    if 'verify_latency_ms' in target_config:
        return target_config['verify_latency_ms']
    # For mixed batching, use decode latency as typical case (most batches are decode-only)
    decode_per_token = target_config.get('decode_latency_per_token', 9.25)
    return gamma * decode_per_token

# ---------- Servers ----------

class TargetServer:
    """Continuous-batching-style target: collect for Delta or until B, then serve in fixed time."""
    def __init__(self, env: simpy.Environment, params: TargetParams, metrics: Metrics, cfg: Config = None, debug: bool = False, router=None):
        self.env = env
        self.p = params
        self.cfg = cfg or Config()  # Use default config if not provided
        self.scheduler_config = self.cfg.scheduler_config  # Get scheduler config
        self.metrics = metrics
        self.debug = debug
        self.router = router  # Reference to router for JIQ notifications
        self.q = simpy.Store(env)           # FIFO queue of Job
        
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
        return self._enqueued_count

    def _calculate_batch_latency(self, batch: List[Job]) -> float:
        """Calculate batch processing time as max of all job latencies.
        Each job's latency depends on its type and token count."""
        if not self.cfg.mixed_batching:
            # Use uniform latency if not mixing
            return self.p.verify_latency_ms
        
        max_latency = 0
        for job in batch:
            if job.job_type == "prefill":
                # Prefill latency = tokens * per-token latency
                latency = job.token_count * self.p.prefill_latency_per_token
            else:
                # Decode latency = tokens * per-token latency  
                latency = job.token_count * self.p.decode_latency_per_token
            max_latency = max(max_latency, latency)
        return max_latency
    
    def work_left_score(self) -> float:
        # Better ETA-based scoring
        B = max(1, self.p.batch_size)
        batches_queued = math.ceil(self._enqueued_count / B)
        # Time until current batch finishes
        remaining_ms = max(0, self._busy_until - self.env.now) if self._busy else 0
        
        # Use average batch latency if available (for mixed batching), otherwise use typical verify time
        if self._avg_batch_latency_ms is not None:
            batch_latency = self._avg_batch_latency_ms
        elif hasattr(self.p, 'verify_latency_ms'):
            batch_latency = self.p.verify_latency_ms
        else:
            # Estimate based on decode latency (most common case)
            batch_latency = self.cfg.gamma * self.p.decode_latency_per_token if hasattr(self.p, 'decode_latency_per_token') else 37.0
        
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
                    self.metrics.add(j)
                    # Signal completion to waiting draft
                    if j.completion_event and not j.completion_event.triggered:
                        j.completion_event.succeed()
                
                # JIQ: Mark as idle if queue is empty after processing
                if self.router and hasattr(self.router, 'mark_idle'):
                    if self._enqueued_count == 0:
                        self.router.mark_idle(self.p.id)
                
                # Server resource automatically released at end of with block

class DraftServer:
    """Simulates draft model generating chunks and waiting for verification (blocking mode)."""
    def __init__(self, env: simpy.Environment, params: DraftParams, cfg: Config, 
                 router, connections: Dict[str, ConnectionParams], total_capability: float = 1.0, metrics: Metrics = None):
        self.env = env
        self.p = params
        self.id = params.id
        self.cfg = cfg
        self.router = router
        self.connections = connections  # Map of target_id -> ConnectionParams
        self.total_capability = total_capability
        self.gamma = cfg.gamma  # tokens per chunk
        self.metrics = metrics  # Reference to global metrics
        
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
    
    def _generate_blocking(self):
        """Blocking mode: generate full conversations with multiple speculation rounds"""
        if self.cfg.verbose:
            my_share = self.p.capability / self.total_capability
            my_rate = self.cfg.workload.rate_rps * my_share
            gen_info = f", gen_latency={self.p.generation_latency_ms:.0f}ms" if self.p.generation_latency_ms > 0 else ""
            print(f"[{self.env.now:.1f}ms] Draft {self.id} starting blocking mode (gamma={self.gamma}, rate={my_rate:.1f} req/s{gen_info})", flush=True)
        
        conversation_count = 0
        
        while self.env.now < self.cfg.sim_time_ms:
            # Start a new conversation
            conversation_count += 1
            conversation_start = self.env.now
            
            # Sample prompt length for this conversation
            prompt_length = self._sample_prompt_length()
            
            # Sample answer length for this conversation
            answer_length = self._sample_answer_length()
            
            # Select target for this entire conversation
            target_id, conn = self._select_target()
            
            # Track connection usage in metrics
            if hasattr(self.metrics, "connection_counts"):
                self.metrics.connection_counts[(self.id, target_id)] += 1
            
            if self.cfg.debug:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Starting conversation #{conversation_count} "
                      f"(prompt={prompt_length} tokens, answer={answer_length} tokens) with target {target_id}", flush=True)
            
            # Phase 1: Send prefill request to target
            if self.cfg.debug:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Sending prefill request ({prompt_length} tokens) to target {target_id}", flush=True)
            
            # Track RTT start for prefill
            prefill_rtt_start = self.env.now
            
            # Create prefill job with completion event
            prefill_completion = self.env.event()
            prefill_job = Job(
                jid=self.chunks_sent,
                created_ms=self.env.now,
                draft_id=self.id,
                job_type="prefill",
                token_count=prompt_length,
                completion_event=prefill_completion,
                rtt_start_ms=prefill_rtt_start
            )
            self.chunks_sent += 1
            
            # Send prefill request to target
            yield self.env.timeout(conn.forward_latency_ms)
            prefill_job.created_ms = self.env.now
            
            # Enqueue prefill job at target
            target_server = None
            for target in self.router.targets:
                if target.p.id == target_id:
                    target_server = target
                    target.enqueue(prefill_job)
                    break
            
            if not target_server:
                print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
                continue
            
            # Wait for actual job completion instead of synthetic sleep
            yield prefill_completion
            
            # Wait for response to travel back
            yield self.env.timeout(conn.response_latency_ms)
            
            # Mark RTT end for prefill
            prefill_job.rtt_end_ms = self.env.now
            
            if self.cfg.debug:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Prefill completed for {prompt_length} tokens", flush=True)
            
            # Phase 2: Generate answer with multiple speculation rounds
            rounds_needed = (answer_length + self.gamma - 1) // self.gamma  # ceiling division
            tokens_generated_in_conversation = 0
            tokens_accepted_in_conversation = 0
            
            # Initialize progress for Semi-Clairvoyant router
            if hasattr(self.router, 'update_progress'):
                self.router.update_progress(self.id, 0, 0, answer_length)
            
            for round_num in range(rounds_needed):
                round_start = self.env.now
                
                # Determine how many tokens to generate in this round
                tokens_remaining = answer_length - tokens_generated_in_conversation
                tokens_this_round = min(self.gamma, tokens_remaining)
                
                # Generate draft tokens (takes time on edge device!)
                if self.p.generation_latency_ms > 0:
                    yield self.env.timeout(self.p.generation_latency_ms)
                
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
                    rtt_start_ms=round_start  # Track RTT from start of generation
                )
                
                # Send chunk to target (forward latency)
                yield self.env.timeout(conn.forward_latency_ms)
                
                # Job arrives at target after network delay
                job.created_ms = self.env.now
                
                # Find the target server and enqueue the job
                target_server = None
                for target in self.router.targets:
                    if target.p.id == target_id:
                        target_server = target
                        target.enqueue(job)
                        break
                
                if not target_server:
                    print(f"Warning: Target {target_id} not found for draft {self.id}", flush=True)
                    break
                
                # Wait for actual job completion instead of synthetic sleep
                yield decode_completion
                
                # Wait for response to travel back
                yield self.env.timeout(conn.response_latency_ms)
                
                # Mark RTT end after response received
                job.rtt_end_ms = self.env.now
                
                # Simulate verification result
                result = self._simulate_verification(tokens_this_round, conn.acceptance_rate)
                self.total_tokens_accepted += result.accepted_tokens
                self.total_tokens_rejected += result.rejected_tokens
                tokens_accepted_in_conversation += result.accepted_tokens
                
                # Update progress for Semi-Clairvoyant router with actual acceptance
                if hasattr(self.router, 'update_progress'):
                    self.router.update_progress(self.id, tokens_generated_in_conversation, 
                                               tokens_accepted_in_conversation, answer_length)
                
                # Update global token metrics (only count after burn-in)
                if self.env.now >= self.cfg.burn_in_ms:
                    self.metrics.token_metrics.total_generated_tokens += tokens_this_round
                    self.metrics.token_metrics.total_accepted_tokens += result.accepted_tokens
                    self.metrics.token_metrics.total_rejected_tokens += result.rejected_tokens
                
                # Calculate round-trip time
                rtt = self.env.now - round_start
                self.total_round_trip_time += rtt
                
                if self.cfg.debug:
                    print(f"[{self.env.now:.1f}ms] Draft {self.id}: Round {round_num+1} result: "
                          f"{result.accepted_tokens}/{tokens_this_round} accepted, RTT={rtt:.1f}ms", flush=True)
            
            # Conversation completed
            conversation_time = self.env.now - conversation_start
            conversation_acceptance = tokens_accepted_in_conversation / tokens_generated_in_conversation if tokens_generated_in_conversation > 0 else 0
            
            if self.cfg.verbose or self.cfg.debug:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Conversation #{conversation_count} completed - "
                      f"prompt={prompt_length}, answer={tokens_generated_in_conversation}/{answer_length} tokens, "
                      f"acceptance={conversation_acceptance:.2%}, time={conversation_time:.1f}ms", flush=True)
            
            # Wait before starting next conversation (inter-arrival time)
            ia = self._ia()
            yield self.env.timeout(ia)
        
        # Final statistics
        if self.chunks_sent > 0:
            acceptance_rate = self.total_tokens_accepted / self.total_tokens_generated
            avg_rtt = self.total_round_trip_time / self.chunks_sent
        else:
            acceptance_rate = 0
            avg_rtt = 0
        
        print(f"[{self.env.now:.1f}ms] Draft {self.id} finished: chunks={self.chunks_sent}, "
              f"tokens={self.total_tokens_generated}, accepted={self.total_tokens_accepted} ({acceptance_rate:.2%}), "
              f"avg RTT={avg_rtt:.1f}ms", flush=True)

# ---------- Routers ----------

# Router classes act as weighted samplers for DraftServer._select_target()
# The route() method is NEVER used - drafts call _weighted_sample_k_filtered directly
# and enqueue jobs themselves to maintain connection-awareness.

class RandomRouter:
    """Pure random selection - no load awareness."""
    def __init__(self, targets: List[TargetServer]):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
    
    def random_select_filtered(self, allowed_ids) -> TargetServer:
        """Randomly select one target from allowed set."""
        pool = [t for t in self.targets if t.p.id in allowed_ids]
        if not pool:
            return None
        return random.choice(pool)

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

ROUTERS = {
    "random": RandomRouter,
    "round_robin": RoundRobinRouter,
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
    
    rng = random.Random(raw.get("seed", 0))
    spec = raw["auto_topology"]
    
    # --- Generate Targets ---
    t_spec = spec.get("targets", {})
    t_count = int(t_spec.get("count", 20))
    tiers = t_spec.get("tiers", [])
    if not tiers:
        tiers = [{
            "name": "default", "ratio": 1.0, "model": "llama-3.1-8b", "gpu": "A100",
            "weight": 1.0, "batch_window_ms": 6.0, "batch_size": 48,
            "prefill_latency_per_token": 0.45, "decode_latency_per_token": 1.80
        }]
    
    # Distribute counts per tier
    remaining = t_count
    tier_counts = []
    for i, t in enumerate(tiers):
        c = t_count - sum(tier_counts) if i == len(tiers)-1 else int(round(t_count * float(t.get("ratio", 0))))
        c = max(0, c)
        tier_counts.append(c)
    
    devices, target_ids, tier_of = [], [], {}
    for t, c in zip(tiers, tier_counts):
        for _ in range(c):
            tid = f"t{len(target_ids):02d}"
            target_ids.append(tid)
            tier_of[tid] = t.get("name", "default")
            devices.append({
                "id": tid, "role": "target",
                "model": t.get("model", ""), "gpu": t.get("gpu", ""),
                "weight": float(t.get("weight", 1.0)),
                "batch_window_ms": float(t.get("batch_window_ms", 6.0)),
                "batch_size": int(t.get("batch_size", 32)),
                "prefill_latency_per_token": float(t.get("prefill_latency_per_token", 0.5)),
                "decode_latency_per_token": float(t.get("decode_latency_per_token", 2.5)),
            })
    
    # Build targets_by_tier mapping
    targets_by_tier = defaultdict(list)
    for tid, tname in tier_of.items():
        targets_by_tier[tname].append(tid)
    all_tiers = list(targets_by_tier.keys())
    
    # --- Generate Drafts ---
    d_spec = spec.get("drafts", {})
    d_count = int(d_spec.get("count", 100))
    gens = d_spec.get("gens_ms_per_gamma", [[0,0],[100,160],[240,420]])
    caps = d_spec.get("capability_map", {0:2.0, 1:1.0, 2:0.6})
    labels = d_spec.get("draft_bucket_labels", ["datacenter", "edge", "user"])
    
    # Store draft gen midpoints for bucket assignment
    gen_mids = [(a+b)/2.0 for a,b in gens]
    
    # Distribute drafts evenly across buckets (33/33/34 for 100 drafts)
    drafts_per_bucket = d_count // len(gens)
    
    for i in range(d_count):
        # Assign bucket deterministically for even distribution
        if i < drafts_per_bucket * len(gens):
            bucket = i // drafts_per_bucket
        else:
            bucket = len(gens) - 1  # Put remainder in last bucket
        
        lo, hi = gens[bucket]
        gen_ms = rng.uniform(float(lo), float(hi))
        devices.append({
            "id": f"d{i:02d}", "role": "draft",
            "capability": float(caps.get(bucket, caps.get(str(bucket), 1.0))),
            "generation_latency_ms": float(gen_ms),
            "bucket": bucket,  # Store the bucket assignment
            "burst_factor": 1.0, "reliability": 0.99
        })
    
    # --- Generate Connectivity ---
    conn_spec = spec.get("connectivity", {})
    fanout_base = int(conn_spec.get("fanout_per_draft", 3))
    fanout_override = conn_spec.get("fanout_override", {})
    affinity = conn_spec.get("affinity_rules", {})
    net_ranges = conn_spec.get("net_ms_ranges", {})
    acc_tbl = conn_spec.get("acceptance_by_tier", {})
    
    connections = []
    for i in range(d_count):
        did = f"d{i:02d}"
        
        # Use the stored bucket assignment
        draft_device = devices[len(target_ids)+i]
        bidx = draft_device.get("bucket", 0)  # Use stored bucket
        label = labels[bidx] if bidx < len(labels) else str(bidx)
        
        # Get allowed targets from affinity rules
        allowed_tiers = affinity.get(label, all_tiers)
        candidates = []
        for t in allowed_tiers:
            candidates.extend(targets_by_tier.get(t, []))
        
        # Determine fanout
        fanout = int(fanout_override.get(label, fanout_base))
        
        # Special case for user drafts - stable "nearest" L4s
        if label == "user" and len(targets_by_tier.get("edge", [])) >= 2:
            l4s = targets_by_tier.get("edge", [])
            base = abs(hash(did)) % len(l4s)
            chosen = [l4s[base], l4s[(base+1) % len(l4s)]]
        else:
            # Randomize candidate selection to avoid bias
            rng.shuffle(candidates)
            # FIXED: Always limit to exactly fanout connections
            chosen = candidates[:fanout]
        
        # Create connections with per-edge properties
        for tid in chosen:
            t_tier = tier_of[tid]
            
            # Network latencies based on tier
            fr = net_ranges.get(t_tier, [20, 40])
            fwd = rng.uniform(float(fr[0]), float(fr[1]))
            rsp = rng.uniform(float(fr[0]), float(fr[1]))
            
            # Acceptance rate based on draft bucket and target tier
            row = acc_tbl.get(str(bidx), acc_tbl.get(bidx, {}))
            base_acc = row.get(t_tier, 0.75)
            acc = max(0.5, min(0.95, base_acc + rng.uniform(-0.03, 0.03)))
            
            connections.append({
                "draft": did, "target": tid,
                "forward_latency_ms": float(fwd),
                "response_latency_ms": float(rsp),
                "acceptance_rate": float(acc)
            })
    
    # Update raw config with generated devices and connections
    raw = dict(raw)
    raw["devices"] = devices
    raw["connections"] = connections
    
    # Add topology statistics for debugging
    if raw.get("verbose", False):
        print(f"Auto-topology generated: {len(devices)} devices ({t_count} targets, {d_count} drafts)")
        print(f"                        {len(connections)} connections (avg {len(connections)/d_count:.1f} per draft)")
        for tier_name, count in zip([t["name"] for t in tiers], tier_counts):
            print(f"  Target tier '{tier_name}': {count} devices")
    
    return raw

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}  # safe_load is the secure choice for untrusted YAML
    raw = _expand_auto_topology(raw)  # Expand auto-topology if present
    wl = WorkloadCfg(**(raw.get("workload", {}) or {}))
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
        workload=wl,
        burn_in_ms=raw.get("burn_in_ms", 0.0),
        verbose=raw.get("verbose", True),
        debug=raw.get("debug", False),
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

    # Require devices to be specified in config
    if not cfg.devices:
        raise ValueError("No devices specified in config. Please define at least one target and one draft device.")

    # Build from devices list
    draft_configs = []
    
    for d in cfg.devices:
        role = d.get("role", "target")
        if role == "target":
            # Calculate typical verify latency if not provided (for mixed batching)
            if "verify_latency_ms" in d:
                verify_ms = float(d["verify_latency_ms"])
            else:
                # Use decode latency as typical case (most batches are decode-only)
                verify_ms = cfg.gamma * float(d.get("decode_latency_per_token", 9.25))
            
            params = TargetParams(
                id=d["id"],
                model=str(d.get("model", "")),
                gpu=str(d.get("gpu", "")),
                weight=float(d.get("weight", 1.0)),
                batch_window_ms=float(d.get("batch_window_ms", 6.0)),
                batch_size=int(d.get("batch_size", 32)),
                verify_latency_ms=verify_ms,
                prefill_latency_per_token=float(d.get("prefill_latency_per_token", 0.5)),
                decode_latency_per_token=float(d.get("decode_latency_per_token", 2.5)),
            )
            targets.append(TargetServer(env, params, metrics, cfg=cfg, debug=cfg.debug))
        elif role == "draft":
            # Store draft configs for later processing
            draft_configs.append(d)

    RouterCls = ROUTERS.get(cfg.router, RoundRobinRouter)
    if RouterCls == WeightedJSQ2Router:
        router = RouterCls(targets, d_choices=int(cfg.router_params.get("d_choices", 2)), debug=cfg.debug)
    elif RouterCls == JSQ2Router:
        router = RouterCls(targets, d_choices=int(cfg.router_params.get("d_choices", 2)))
    else:
        router = RouterCls(targets)
    
    # Set router reference on targets for JIQ notifications
    for target in targets:
        target.router = router

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
            
        connection_map[draft_id][target_id] = ConnectionParams(
            draft_id=draft_id,
            target_id=target_id,
            forward_latency_ms=float(conn_cfg.get("forward_latency_ms", 20.0)),
            response_latency_ms=float(conn_cfg.get("response_latency_ms", 20.0)),
            acceptance_rate=float(conn_cfg.get("acceptance_rate", 0.8)),
        )
    
    # Create draft servers with connections
    for d in draft_configs:
        params = DraftParams(
            id=d["id"],
            capability=float(d.get("capability", 1.0)),
            generation_latency_ms=float(d.get("generation_latency_ms", 0.0)),
            burst_factor=float(d.get("burst_factor", 1.0)),
            reliability=float(d.get("reliability", 0.99)),
        )
        
        # Get connections for this draft
        draft_connections = connection_map.get(d["id"], {})
        if not draft_connections:
            print(f"Warning: Draft {d['id']} has no connections configured", flush=True)
        
        drafts.append(DraftServer(env, params, cfg, router, draft_connections, total_capability, metrics))

    # Start heartbeat monitor
    if cfg.verbose:
        env.process(heartbeat_monitor(env, metrics, targets, cfg))
    
    return metrics, targets, drafts

def run(cfg: Config):
    env = simpy.Environment()
    result = build(env, cfg)
    if isinstance(result, tuple) and len(result) == 3:
        metrics, targets, drafts = result
    elif isinstance(result, tuple) and len(result) == 2:
        metrics, targets = result
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
        if d.chunks_sent > 0:
            acceptance_rate = d.total_tokens_accepted / d.total_tokens_generated
            avg_rtt = d.total_round_trip_time / d.chunks_sent
        else:
            acceptance_rate = 0
            avg_rtt = 0
            
        results[d.p.id] = {
            "capability": d.p.capability,
            "chunks_sent": d.chunks_sent,
            "tokens_generated": d.total_tokens_generated,
            "tokens_accepted": d.total_tokens_accepted,
            "tokens_rejected": d.total_tokens_rejected,
            "acceptance_rate": acceptance_rate,
            "avg_rtt_ms": avg_rtt,
        }
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    
    result = run(cfg)
    if isinstance(result, tuple) and len(result) == 3:
        metrics, targets, drafts = result
    elif isinstance(result, tuple) and len(result) == 2:
        metrics, targets = result
        drafts = []
    else:
        metrics = result
        targets = []
        drafts = []
    
    # Get the summary from metrics
    summary = metrics.summary() if hasattr(metrics, 'summary') else {}
    
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
    import json
    import statistics
    print("\n===METRICS_JSON===")
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
    }
    
    # Add batch composition if available
    if targets and len(targets) > 0:
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
            metrics_json["avg_prefills_per_batch"] = statistics.mean(prefill_counts) if prefill_counts else 0
            metrics_json["avg_decodes_per_batch"] = statistics.mean(decode_counts) if decode_counts else 0
    
    print(json.dumps(metrics_json, indent=2))
    print("===END_METRICS_JSON===")

if __name__ == "__main__":
    main()