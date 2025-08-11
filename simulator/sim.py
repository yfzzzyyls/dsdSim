# sim.py  v1 minimal distributed speculative-decoding simulator
# deps: pip install simpy pyyaml

import argparse, random, simpy, yaml, math
import time

# Print SimPy version for debugging
print(f"SimPy version: {simpy.__version__}", flush=True)
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# ---------- Config & Types ----------

@dataclass
class TargetParams:
    id: str
    weight: float = 1.0               # capacity weight (relative speed)
    batch_window_ms: float = 6.0      # Delta: wait to form a batch
    batch_size: int = 32              # B: max jobs per batch
    verify_latency_ms: float = 8.0    # fixed service time per batch

@dataclass
class DraftParams:
    id: str
    capability: float = 1.0           # relative compute speed (affects generation rate)
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
    router: str = "round_robin"       # "round_robin" | "jsq2" | "wjsq2"
    router_params: Dict[str, Any] = field(default_factory=lambda: {"d_choices": 2})
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
    started_ms: Optional[float] = None
    finished_ms: Optional[float] = None

class Metrics:
    def __init__(self, verbose: bool = True, burn_in_ms: float = 0.0) -> None:
        self.completed: List[Job] = []
        self.verbose = verbose
        self.burn_in_ms = burn_in_ms

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
        lat = [j.finished_ms - j.created_ms for j in filtered]
        lat.sort()
        def pct(p):
            i = int(round(p * (len(lat) - 1)))
            return lat[max(0, min(i, len(lat)-1))]
        # Correct span calculation using min/max of filtered jobs
        start = min(j.created_ms for j in filtered)
        end = max(j.finished_ms for j in filtered)
        span = end - start + 1e-9
        return {
            "count": len(lat),
            "throughput_jobs_s": 1000.0 * len(lat) / span,
            "avg_ms": sum(lat)/len(lat) if lat else 0,
            "p50_ms": pct(0.50) if lat else 0, 
            "p95_ms": pct(0.95) if lat else 0, 
            "p99_ms": pct(0.99) if lat else 0,
        }

# ---------- Servers ----------

class TargetServer:
    """Continuous-batching-style target: collect for Delta or until B, then serve in fixed time."""
    def __init__(self, env: simpy.Environment, params: TargetParams, metrics: Metrics, debug: bool = False):
        self.env = env
        self.p = params
        self.metrics = metrics
        self.debug = debug
        self.q = simpy.Store(env)           # FIFO queue of Job
        self._busy = False                  # coarse busy flag (for rough work-left estimate)
        self._enqueued_count = 0            # track items in queue
        self._busy_until = 0.0              # when current batch finishes
        # Per-target metrics
        self.busy_ms = 0.0
        self.total_batches = 0
        self.total_batch_items = 0
        self.queue_samples = []             # Track queue length over time
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

    def work_left_score(self) -> float:
        # Better ETA-based scoring
        B = max(1, self.p.batch_size)
        batches_queued = math.ceil(self._enqueued_count / B)
        # Time until current batch finishes
        remaining_ms = max(0, self._busy_until - self.env.now) if self._busy else 0
        # Total estimated time to clear queue
        eta_ms = remaining_ms + batches_queued * self.p.verify_latency_ms
        # Normalize by capacity weight
        return eta_ms / max(self.p.weight, 1e-6)

    # serving loop
    def _serve_loop(self):
        tp = self.p
        serve_count = 0
        while True:
            # Debug around problem time
            if self.env.now > 1020 and self.env.now < 1030:
                print(f"[{self.env.now:.1f}ms] Target {self.p.id}: Waiting for job, queue has {self._enqueued_count} items", flush=True)
            
            first = yield self.q.get()              # wait for first job
            self._enqueued_count -= 1
            batch = [first]
            t0 = self.env.now
            serve_count += 1
            
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
                    batch.append(job)
                else:
                    # timeout fired; must cancel the pending get to avoid a dangling consumer
                    # (this is recommended in SimPy's shared-resource patterns)
                    if hasattr(get_ev, 'cancel'):
                        get_ev.cancel()

            # "serve" the batch with fixed latency
            if self.debug and self.total_batches < 5:  # Only print first 5 batches
                print(f"[{self.env.now:.1f}ms] Target {self.p.id}: Serving batch of {len(batch)} jobs", flush=True)
            
            for j in batch:
                j.started_ms = self.env.now
            self._busy = True
            self._busy_until = self.env.now + tp.verify_latency_ms
            start_time = self.env.now
            
            yield self.env.timeout(tp.verify_latency_ms)   # fixed service time per batch
            
            self._busy = False
            self.busy_ms += self.env.now - start_time
            self.total_batches += 1
            self.total_batch_items += len(batch)
            self.queue_samples.append((self.env.now, self._enqueued_count))
            
            tdone = self.env.now
            for j in batch:
                j.finished_ms = tdone
                self.metrics.add(j)

class DraftServer:
    """Simulates draft model generating chunks and waiting for verification (blocking mode)."""
    def __init__(self, env: simpy.Environment, params: DraftParams, cfg: Config, 
                 router, connections: Dict[str, ConnectionParams], total_capability: float = 1.0):
        self.env = env
        self.p = params
        self.id = params.id
        self.cfg = cfg
        self.router = router
        self.connections = connections  # Map of target_id -> ConnectionParams
        self.total_capability = total_capability
        self.gamma = cfg.gamma  # tokens per chunk
        
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
        """Select a target server using router logic and return its ID and connection params"""
        # Use the router to select the best target
        # Create a dummy job for routing decision
        dummy_job = Job(jid=-1, created_ms=self.env.now, draft_id=self.id)
        
        # Get the target selection from router (but don't actually enqueue)
        if hasattr(self.router, '_weighted_sample_k'):
            # For wJSQ2 router
            candidates = self.router._weighted_sample_k(self.router.d)
            target = min(candidates, key=lambda t: t.work_left_score())
        elif hasattr(self.router, 'targets'):
            # For round-robin or other routers, just pick based on queue length
            target = min(self.router.targets, key=lambda t: t.queue_len())
        else:
            # Fallback to random
            target_ids = list(self.connections.keys())
            if not target_ids:
                raise ValueError(f"Draft {self.id} has no connections configured")
            target_id = random.choice(target_ids)
            return target_id, self.connections[target_id]
        
        target_id = target.p.id
        if target_id not in self.connections:
            # If no connection configured for this target, pick first available
            target_ids = list(self.connections.keys())
            if target_ids:
                target_id = target_ids[0]
            else:
                raise ValueError(f"Draft {self.id} has no connections configured")
        
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
    
    def _generate_blocking(self):
        """Blocking mode: generate chunk, send, wait for response, then continue"""
        if self.cfg.verbose:
            my_share = self.p.capability / self.total_capability
            my_rate = self.cfg.workload.rate_rps * my_share
            print(f"[{self.env.now:.1f}ms] Draft {self.id} starting blocking mode (gamma={self.gamma}, rate={my_rate:.1f} req/s)", flush=True)
        
        session_count = 0
        
        while self.env.now < self.cfg.sim_time_ms:
            # Start a new generation session
            session_count += 1
            session_start = self.env.now
            
            # Select target for this session
            target_id, conn = self._select_target()
            
            # Generate a chunk of gamma tokens locally (no delay for local generation)
            self.chunks_sent += 1
            self.total_tokens_generated += self.gamma
            
            if self.cfg.debug:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Generating chunk #{self.chunks_sent} ({self.gamma} tokens) for target {target_id}", flush=True)
            
            # Create a job representing this chunk verification request
            job = Job(jid=self.chunks_sent, created_ms=self.env.now, draft_id=self.id)
            
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
                continue
            
            # In blocking mode, draft waits for:
            # 1. Target to process the verification (target's verify_latency_ms handles this)
            # 2. Response to travel back (response_latency_ms)
            
            # The target will process this job in its batch window + verify_latency
            # For now, we simulate the total round-trip time
            # In reality, the target processing happens asynchronously
            
            # Wait for response (response latency after target processes)
            yield self.env.timeout(conn.response_latency_ms)
            
            # Simulate verification result
            result = self._simulate_verification(self.gamma, conn.acceptance_rate)
            self.total_tokens_accepted += result.accepted_tokens
            self.total_tokens_rejected += result.rejected_tokens
            
            # Calculate round-trip time
            rtt = self.env.now - session_start
            self.total_round_trip_time += rtt
            
            if self.cfg.debug or (self.chunks_sent % 10 == 0):
                acceptance_rate = self.total_tokens_accepted / self.total_tokens_generated if self.total_tokens_generated > 0 else 0
                avg_rtt = self.total_round_trip_time / self.chunks_sent if self.chunks_sent > 0 else 0
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Chunk #{self.chunks_sent} result: {result.accepted_tokens}/{self.gamma} accepted, "
                      f"overall acceptance={acceptance_rate:.2%}, avg RTT={avg_rtt:.1f}ms", flush=True)
            
            # Wait before starting next chunk (inter-arrival time)
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

class BaseRouter:
    def route(self, job: Job): raise NotImplementedError

class RoundRobinRouter(BaseRouter):
    def __init__(self, targets: List[TargetServer]):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self._i = 0
    def route(self, job: Job):
        tgt = self.targets[self._i % len(self.targets)]
        self._i += 1
        tgt.enqueue(job)

class JSQ2Router(BaseRouter):
    """Unweighted power-of-two choices (classic JSQ(d))."""
    def __init__(self, targets: List[TargetServer], d_choices: int = 2):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self.d = min(max(1, d_choices), len(targets))
    
    def route(self, job: Job):
        # Sample d random targets
        if len(self.targets) <= self.d:
            candidates = self.targets
        else:
            candidates = random.sample(self.targets, self.d)
        # Route to the one with shortest queue
        tgt = min(candidates, key=lambda t: t.queue_len())
        tgt.enqueue(job)

class WeightedJSQ2Router(BaseRouter):
    """Power-of-two choices with capacity-aware sampling (weights)."""
    def __init__(self, targets: List[TargetServer], d_choices: int = 2, debug: bool = False):
        if not targets:
            raise ValueError("Router needs at least one target")
        self.targets = targets
        self.d = max(1, d_choices)
        self.debug = debug
        self.route_count = 0

    def _weighted_sample_k(self, k: int) -> List[TargetServer]:
        """Weighted sampling without replacement using Efraimidis-Spirakis algorithm"""
        k = min(k, len(self.targets))
        weights = [max(0.0, t.p.weight) for t in self.targets]
        
        # Fallback if all weights are zero
        if sum(weights) == 0:
            return random.sample(self.targets, k)
        
        # Efraimidis-Spirakis: weighted sampling without replacement
        # Draw k items with keys = u^(1/w_i) where u ~ U(0,1)
        # Items with higher weights more likely to get larger keys
        keyed = []
        for t, w in zip(self.targets, weights):
            # If w==0, it will never be chosen (unless all are 0, handled above)
            if w == 0:
                key = 0  # Will sort to the end
            else:
                u = random.random()
                key = u ** (1.0 / w)
            keyed.append((key, t))
        
        # Take top-k by key (highest keys = most likely to be selected)
        keyed.sort(reverse=True)
        return [t for _, t in keyed[:k]]

    def route(self, job: Job):
        self.route_count += 1
        
        # Debug for specific time window
        debug_this = self.debug and self.route_count <= 10
        if job.created_ms > 1020 and job.created_ms < 1030:
            print(f"[{job.created_ms:.1f}ms] Router: Starting route for job from {job.draft_id}", flush=True)
            debug_this = True
        
        cands = self._weighted_sample_k(self.d)
        tgt = min(cands, key=lambda t: t.work_left_score())
        
        if debug_this:
            cand_info = [(t.p.id, f"{t.work_left_score():.1f}") for t in cands]
            print(f"  Router: job {job.jid} -> {tgt.p.id} (candidates: {cand_info})", flush=True)
        
        tgt.enqueue(job)

ROUTERS = {
    "round_robin": RoundRobinRouter,
    "jsq2": JSQ2Router,
    "wjsq2": WeightedJSQ2Router,
}

# ---------- Config I/O & Runner ----------

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}  # safe_load is the secure choice for untrusted YAML
    wl = WorkloadCfg(**(raw.get("workload", {}) or {}))
    cfg = Config(
        sim_time_ms=raw.get("sim_time_ms", 10_000),
        seed=raw.get("seed", 0),
        execution_mode=raw.get("execution_mode", "blocking"),
        gamma=raw.get("gamma", 4),
        router=raw.get("router", "round_robin"),
        router_params=raw.get("router_params", {"d_choices": 2}),
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
            params = TargetParams(
                id=d["id"],
                weight=float(d.get("weight", 1.0)),
                batch_window_ms=float(d.get("batch_window_ms", 6.0)),
                batch_size=int(d.get("batch_size", 32)),
                verify_latency_ms=float(d.get("verify_latency_ms", 8.0)),
            )
            targets.append(TargetServer(env, params, metrics, debug=cfg.debug))
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
            burst_factor=float(d.get("burst_factor", 1.0)),
            reliability=float(d.get("reliability", 0.99)),
        )
        
        # Get connections for this draft
        draft_connections = connection_map.get(d["id"], {})
        if not draft_connections:
            print(f"Warning: Draft {d['id']} has no connections configured", flush=True)
        
        drafts.append(DraftServer(env, params, cfg, router, draft_connections, total_capability))

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
    
    elapsed_real_time = time.time() - start_real_time
    print(f"env.run() completed at {env.now:.0f}ms", flush=True)
    
    if cfg.verbose:
        print(f"\nSimulation complete in {elapsed_real_time:.2f}s real time")
        print(f"Processed {len(metrics.completed)} total jobs")
    
    return metrics.summary(), targets, drafts

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
    ap.add_argument("--config", "-c", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    
    result = run(cfg)
    if isinstance(result, tuple) and len(result) == 3:
        summary, targets, drafts = result
    elif isinstance(result, tuple) and len(result) == 2:
        summary, targets = result
        drafts = []
    else:
        summary = result
        targets = []
        drafts = []
    
    # Print results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
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

if __name__ == "__main__":
    main()