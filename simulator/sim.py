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
class WorkloadCfg:
    arrival: str = "deterministic"    # "deterministic" | "poisson"
    interarrival_ms: float = 12.0     # used if deterministic
    rate_rps: float = 100.0           # used if poisson: mean rate

@dataclass
class Config:
    sim_time_ms: float = 10_000
    seed: int = 0
    router: str = "round_robin"       # "round_robin" | "jsq2" | "wjsq2"
    router_params: Dict[str, Any] = field(default_factory=lambda: {"d_choices": 2})
    devices: List[Dict[str, Any]] = field(default_factory=list)   # list of dicts with role/params
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
    """Generates verify jobs according to the workload model."""
    def __init__(self, env: simpy.Environment, draft_id: str, cfg: Config, router, num_drafts: int = 1):
        self.env = env
        self.id = draft_id
        self.cfg = cfg
        self.router = router
        self.num_drafts = num_drafts  # Total number of draft servers
        self.jid = 0
        self.proc = env.process(self._generate())

    def _ia(self) -> float:
        wl = self.cfg.workload
        if wl.arrival == "poisson":
            # Divide rate by number of drafts to get per-draft rate
            per_draft_rate = max(wl.rate_rps / self.num_drafts, 1e-9)
            lam = per_draft_rate / 1000.0  # events per ms
            return random.expovariate(lam)
        # default deterministic - also divide by number of drafts
        return self.cfg.workload.interarrival_ms * self.num_drafts

    def _generate(self):
        if self.cfg.debug and self.id == "d0":  # Only print for first draft to reduce noise
            print(f"[{self.env.now:.1f}ms] Draft {self.id} starting generation (rate per draft: {self.cfg.workload.rate_rps/self.num_drafts:.1f} req/s)", flush=True)
        
        job_count = 0
        while self.env.now < self.cfg.sim_time_ms:
            job = Job(jid=self.jid, created_ms=self.env.now, draft_id=self.id)
            self.jid += 1
            job_count += 1
            
            if self.cfg.debug and self.jid <= 3:  # Print first 3 jobs only
                print(f"[{self.env.now:.1f}ms] Draft {self.id} generated job {self.jid}", flush=True)
            
            # Add periodic debug even when debug is off
            if job_count % 50 == 0 and job_count > 0:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Generated {job_count} jobs so far", flush=True)
            
            # Debug after 1000ms
            if self.env.now > 1000 and job_count % 10 == 0:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Still generating, job #{job_count}", flush=True)
            
            # Debug: check if routing is slow
            if self.env.now > 1020 and self.env.now < 1030:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: About to route job {job_count}", flush=True)
            
            self.router.route(job)
            
            if self.env.now > 1020 and self.env.now < 1030:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: Routing complete", flush=True)
            
            ia = self._ia()
            
            if self.cfg.debug and self.jid <= 2:  # Print interarrival for first 2
                print(f"  Draft {self.id} waiting {ia:.1f}ms until next job", flush=True)
            
            yield self.env.timeout(ia)
            
            # Extra debug if we're past 1000ms and haven't printed in a while
            if self.env.now > 1000 and self.env.now < 1100:
                print(f"[{self.env.now:.1f}ms] Draft {self.id}: About to loop, job_count={job_count}, next check at {self.env.now + ia:.1f}ms", flush=True)
        
        print(f"[{self.env.now:.1f}ms] Draft {self.id}: Finished generating (total: {job_count} jobs)", flush=True)

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
        router=raw.get("router", "round_robin"),
        router_params=raw.get("router_params", {"d_choices": 2}),
        devices=raw.get("devices", []),
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

    # If devices list is empty, fall back to 2 drafts + 2 targets
    if not cfg.devices:
        targets = [TargetServer(env, TargetParams(id=f"t{i}"), metrics, debug=cfg.debug) for i in range(2)]
        router = ROUTERS[cfg.router](targets, **cfg.router_params) if cfg.router != "round_robin" \
            else ROUTERS[cfg.router](targets)
        num_drafts = 2
        drafts = [DraftServer(env, f"d{i}", cfg, router, num_drafts) for i in range(num_drafts)]
        # Start heartbeat monitor
        if cfg.verbose:
            env.process(heartbeat_monitor(env, metrics, targets, cfg))
        return metrics, targets

    # Build from devices list
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

    RouterCls = ROUTERS.get(cfg.router, RoundRobinRouter)
    if RouterCls == WeightedJSQ2Router:
        router = RouterCls(targets, d_choices=int(cfg.router_params.get("d_choices", 2)), debug=cfg.debug)
    elif RouterCls == JSQ2Router:
        router = RouterCls(targets, d_choices=int(cfg.router_params.get("d_choices", 2)))
    else:
        router = RouterCls(targets)

    # Count drafts first
    num_drafts = sum(1 for d in cfg.devices if d.get("role") == "draft")
    
    for d in cfg.devices:
        role = d.get("role", "target")
        if role == "draft":
            drafts.append(DraftServer(env, d["id"], cfg, router, num_drafts))

    # Start heartbeat monitor
    if cfg.verbose:
        env.process(heartbeat_monitor(env, metrics, targets, cfg))
    
    return metrics, targets

def run(cfg: Config):
    env = simpy.Environment()
    result = build(env, cfg)
    if isinstance(result, tuple):
        metrics, targets = result
    else:
        metrics = result
        targets = []
    
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
    
    return metrics.summary(), targets

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    
    result = run(cfg)
    if isinstance(result, tuple):
        summary, targets = result
    else:
        summary = result
        targets = []
    
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

if __name__ == "__main__":
    main()