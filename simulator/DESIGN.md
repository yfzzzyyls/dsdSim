# Distributed Speculative Decoding Simulator — System Design

## 0. Background & Inspirations

This design builds on lessons from recent serving and simulation systems:

- **ORCA** (OSDI 2022): introduced iteration-level scheduling with selective batching to respect per-token heterogeneity and KV constraints in large-model serving.
- **DistServe** (OSDI 2024): demonstrated disaggregated prefill/decoding, SLO-aware goodput planning, and the need to co-design compute/network placement under strict TTFT/TPOT objectives.
- **VIDUR** microprofiling workflows and subsequent speculative decoding surveys motivate LUT-driven accuracy, acceptance modelling, and adaptive speculation policies.

## 1. Objectives

We are building a trace-driven, heterogeneity-aware simulator for **distributed speculative decoding (SD)**. The framework must:

- Model heterogeneous draft devices (phones, laptops, edge GPUs) and target servers (datacenter GPUs) with realistic compute, energy, memory, and network characteristics.
- Support both **distributed SD** (draft generates speculative tokens, target verifies) and **fused SD** (target performs draft + verify locally).
- Scale to thousands of concurrent requests using iteration-level (token-level) scheduling with explicit SLO targets for TTFT, TPOT, and TVT (time-to-verify-token).
- Plan GPU pool sizes, microbatch sizes, and speculation parameters to maximize **SLO goodput** (requests that meet TTFT/TPOT/TVT per GPU-second).
- Produce accurate performance estimates using **lookup tables (LUTs)** populated from VIDUR/microbenchmarks.
- Offer pluggable scheduling, speculation, and admission policies to evaluate new algorithms quickly.
- Treat **acceptance behavior** as a conditional random variable over method, context, draft depth, and temperature so controllers can adapt dynamically, while keeping energy/cost and fairness as first-class signals.
- Track **tail behavior (P95/P99)** and queueing effects explicitly so iteration-level decisions show up in latency distributions, not just means.
- Emulate realistic **network behavior** including RTT, bandwidth sharing, jitter, serialization overheads, loss, and link failures so distributed paths are faithfully modelled.
- Guarantee **reproducibility** via explicit configuration/version pinning (LUT, policy, scenario), deterministic seeds, and compact internal-state logging/regression checks.

This document lays out the core components and specifies the auxiliary design files that developers will implement:

1. `SCENARIO.md` (describes the YAML/JSON scenario schema and example snippets).
2. `device_profiles/README.md` (device profile schema and conventions).
3. `lut_schema.json` (formal schema for performance LUT entries).
4. `lut_population/PLAN.md` (VIDUR-based population workflow & validation steps).
5. Section 6 (scheduler & planner APIs / pseudocode kept inline).

## 2. Architecture Overview

```
+----------------+        +------------------+        +--------------------+
| Scenario Input | -----> | Simulator Driver | -----> | Metrics & Outputs  |
+----------------+        +------------------+        +--------------------+
         |                       |
         v                       v
 +-------------------+    +------------------+
 | Device Registry   |    | Workload Trace   |
 |  - Draft devices  |    |  - Arrivals      |
 |  - Target servers |    |  - Prompts       |
 +-------------------+    |  - SLO targets   |
         |                +------------------+
         v
 +-------------------+
 | Cold-Start Manager|
 +-------------------+
         |
         v
 +-------------------+    +-----------------------+
 | LUT Manager       |    | Event Engine          |
 |  - VIDUR-backed   |    |  - Prefill queue      |
 |  - Interpolator   |    |  - Draft decode queue |
 +-------------------+    |  - Verify queue       |
         |                |  - Network events     |
         v                +-----------------------+
 +-------------------+              |
 | Queueing Model    | <------------+
 +-------------------+              |
         |                           v
         |                +-----------------------+
         +--------------> | Scheduler & Planner   |
                          |  - Adaptive k         |
                          |  - Pool sizing        |
                          |  - Admission control  |
                          |  - Mode selection     |
                          +-----------------------+
                          ^
                          |
                  +-------------------+
                  | Acceptance Est.   |
                  +-------------------+
                          |
                          v
                +-----------------------+
                | Reproducibility Log   |
                +-----------------------+
```

### Key modules

| Module              | Responsibilities |
|---------------------|------------------|
| Scenario Parser     | Load YAML/JSON inputs defining topology, devices, workloads, SD modes, policies. |
| Device Registry     | Create device instances with performance models, energy, availability schedules, KV capacities. |
| LUT Manager         | Load, validate, and interpolate VIDUR-produced LUT entries. Provide per-operation latency, memory, energy lookups. |
| Workload Manager    | Replays trace `W` (arrival time, prompt length, output length, SLO class) or synthesizes workloads. |
| Network Model       | Simulates RTT, bandwidth sharing, jitter, serialization overhead, loss, and optional link failures for prompt/speculative/result transfers. |
| KV Manager          | Track per-request KV cache allocations, handle eviction/pinning policies. |
| Event Engine        | Drive discrete events: arrivals, prefill completion, draft iteration completion, verification completion, network transfers, timeouts. |
| Scheduler           | Token-level scheduling for draft and verify phases (EDF/least slack), adaptive fanout `k`, target vs fused mode choice, admission control. |
| Planner             | DistServe-style goodput maximization: search pool sizes, microbatch sizes, draft-model choices, and speculation parameters. |
| Acceptance Estimator| Maintain acceptance curves, update online with trace feedback, and provide per-request predictions to schedulers and planners. |
| Mode Selector       | Decide between distributed, fused, or hybrid execution per request/class using network/queue telemetry and acceptance forecasts. |
| Queueing Model      | Maintain per-phase queues with LUT-derived service distributions; expose backpressure (wait, utilization) to policies and planner. |
| Cold-Start Manager  | Track checkpoint warm/cold state, locality, and load costs for targets; inform mode selection and TTFT estimates. |
| Reproducibility & Logger | Manage RNG seeds, LUT/policy/scenario version metadata, structured logs, and regression baselines for deterministic replay and auditing. |
| Metrics Reporter    | Produce latency distributions, goodput, throughput, fairness, energy, KV utilization, cost per 1k tokens. |

## 3. Simulation Flow

### 3.1 Plain-Language Execution Walkthrough

1. **Start**
   - The simulator boots with time set to zero, empty queues, and no requests in flight.

2. **Load System Config**
   - Read the scenario configuration to identify devices, hardware profiles, policies, mode selector thresholds, parallelism options, network topology, and SLO targets.

3. **Instantiate Modules**
   - Construct the core components: `Device Registry`, `Cold-Start Manager`, `LUT Manager`, `Network Model`, `Queueing Model`, `Acceptance Estimator`, `Scheduler`, `Planner` (including parallelism controls), `Mode Selector`, `KV Manager`, `Metrics Reporter`, and the `Reproducibility Log` wiring.

4. **Load Trace W**
   - Read or synthesize the workload trace so arrival times, prompts, and request goals are available to the driver.

5. **Initialize Event Queue**
   - Seed the `Event Engine` with initial arrival events drawn from the trace; the priority queue is ordered by event timestamp.

6. **Main Loop: Pop Next Event**
   - Repeatedly dequeue the earliest event and dispatch it to the appropriate handler (arrival, service completion, planner tick, timeout, etc.).

7. **Arrival Event Handling**
   - Use the `Mode Selector` (and planner-provided policies) to decide whether the request runs distributed, fused, or hybrid.
   - Admit the request into the correct queue via the `Scheduler` and `Queueing Model`, optionally waiting for batching or microbatch opportunities.
   - Consult the `LUT Manager` to estimate service latency under the chosen device, context length, and parallelism settings.
   - Schedule downstream service events with the `Event Engine` at the predicted completion time.

8. **Service-Complete Event Handling**
   - Update per-request state (context length, generated tokens, TTFT markers) and release device/KV resources through the `KV Manager` and `Queueing Model`.
   - Consult the `Acceptance Estimator` (and speculation policies) to decide whether more work is needed or the request is finished.
   - If complete, emit metrics through the `Metrics Reporter` and log outcomes for reproducibility; otherwise re-enqueue the request for the next stage.

9. **Planner Tick Event**
   - Periodically trigger the `Planner` to evaluate alternative pool sizes, microbatch limits, draft/verify splits, and mode thresholds using recent statistics plus LUT forecasts.
   - Apply the best configuration by updating the `Scheduler`, `Mode Selector`, and related controllers.

10. **Repeat**
   - Continue processing events until the trace is exhausted and no queues contain pending work.

11. **Finalization & Reporting**
   - Aggregate per-request metrics, compute goodput and tail latency summaries, and write logs/reports via `Metrics Reporter` and `Run Reports / Logs`.
   - Flush the `Reproducibility Log` with the final state so runs can be replayed.

**Step → Modules**

| Step | Key Modules |
| --- | --- |
| Load System Config | Scenario Parser, Device Registry, Mode Selector, Planner |
| Instantiate Modules | Device Registry, Cold-Start Manager, LUT Manager, Network Model, Queueing Model, Acceptance Estimator, Scheduler, Planner, Mode Selector, KV Manager, Metrics Reporter, Reproducibility Log |
| Load Trace W | Workload Trace Reader, Event Engine |
| Init Event Queue | Event Engine |
| Arrival Event | Event Engine, Mode Selector, Scheduler, Queueing Model, LUT Manager |
| Service Complete Event | Event Engine, Queueing Model, KV Manager, Scheduler, Acceptance Estimator |
| Planner Tick Event | Event Engine, Planner, Scheduler, LUT Manager, Mode Selector |
| Logging / Finalization | Metrics Reporter, Reproducibility Log, Queueing Model, Device Registry, Acceptance Estimator |

## Status

### Completed

#### Multi-Cluster Routing & Per-Cluster Routers

- Auto-topology expansion now emits cluster-aware devices and connections via `_expand_auto_topology`, so configs can define `auto_topology.clusters[]`, `cluster_router`, and `cluster_router_params` for per-cluster policies.
- During bootstrap the sim instantiates a dedicated router per cluster and binds drafts to the matching router, preserving cluster affinities on devices and connections (`simulator/sim.py`).
- Scenario templates such as `simulator/configs/config.yaml` demonstrate independent clusters (core, regional, edge) with heterogeneous router selections.

#### Per-Draft Closed-Loop Think Time

- `ThinkTimeConfig` is parsed from config files and stored on each draft to enable log-normal/exponential/constant/workload pacing choices (`simulator/sim.py`).
- Drafts track `_next_available_ms`, sample think time after completions, and clamp trace replay arrivals with `max(trace_arrival, next_available_ms)` for closed-loop behaviour.
- Toggling `think_time.enabled` reverts to the original open-loop behaviour for stress-style workloads.

#### Queueing & Resource Modeling Improvements

- The phase scheduler now relies on `simpy.PriorityStore` so arrivals are ordered by priority class instead of FIFO, enabling SLO-aware reordering (`PhaseScheduler` in `simulator/sim.py`).
- KV usage is tracked explicitly through `KVManager`, enforcing per-target token budgets and paging penalties before admitting batches.

### Remaining TODO

- Extend resource modeling beyond KV to cover GPU execution pools (SMs, HBM, KV bandwidth) so contention flows through the performance model.
- Revisit the SimPy kernel choice (PriorityResource, ProcessPool, or an internal dispatcher) once VIDUR integration lands for multi-server targets and GPU sharing.

### 3.2 Event-Driven Loop (Implementation Notes)

1. **Initialization**
   - Parse scenario file.
   - Load device profiles and LUTs (validate compatibility).
   - Seed RNGs, capture scenario/LUT/policy versions, and initialize reproducibility logs.
   - Load acceptance tables or predictors and warm any learned controllers.
   - Initialize workload trace reader and network topology.
   - Configure scheduling/policy modules.

2. **Main event loop**
   - Inject arrivals from trace (add to draft prefill queue). Each arrival carries metadata: prompt length, desired tokens, SLO class, device of origin.
   - When scheduler triggers, form batches/microbatches:
     - Prefill (draft or target) batches use **size/timeout** rule constrained by TTFT SLO.
     - Draft decode iterations use adaptive fanout `k` + selective batching.
       - Acceptance forecasts and verify queue depths jointly tune `k`, draft depth, and drafter model choice each iteration.
       - Maintain portfolios of draft models (tiny/mid/large) and escalate mid-stream when acceptance drops or prompt difficulty rises.
     - Verify iterations group requests by context length where possible; otherwise fallback to per-request attention.
       - Use shape-aware micro-cohorting and prefix-similarity hints to reuse KV/cache state when prompts overlap.
       - Support partial/early-exit verification policies (e.g., first-token or sampled verification) with rollback paths when mis-speculation occurs.
     - After each iteration, materialize acceptance outcomes, update estimators, and feed signals into adaptive `k` and drafter selection controllers.
   - Query LUT manager to obtain per-iteration latency, memory, energy for specified hardware, parallelism, context length, microbatch, fanout.
   - Tail-aware timing: sample service duration from LUT-provided distributions (e.g., mean + P95 or parametric fit) so iteration-level choices impact latency tails.
   - Priority/preemption hook: allow verify scheduler to preempt or reorder (EDF/least-slack-first) when SLO risk is high; log preemptions for fairness/tail diagnostics.
   - Generate network events for prompt/speculative bundle uploads and verification responses, drawing delays from RTT/bandwidth/jitter/loss models per link (including retries when losses occur).
   - Update KV manager for allocations and releases after each iteration.
     - Enforce KV-first admission control, eviction, and spill-to-fused rules when memory headroom shrinks.
   - Record per-request metrics (TTFT, TPOT, TVT, energy, KV footprint).
   - Append structured traces (queues, acceptance samples, mode decisions) to reproducibility logs for regression comparison.

3. **Planner loop** (periodic)
   - Sample recent queue/backpressure statistics (wait time, utilization), acceptance rate, and SLO attainment.
   - Enumerate candidate pool splits `(num_draft_replicas, num_verify_replicas)` and parallelism configs.
   - Estimate multi-objective outcomes (goodput, tail latency, fairness, energy/$) via short "what-if" windows or analytic approximations.
   - Produce Pareto candidates and respect explicit SLO/constraint guards before committing.
   - Apply best configuration (possibly adjusting `k`, draft-model choice, microbatch size, mode mix).

4. **Completion**
   - Aggregate metrics: mean/median/P95/P99 TTFT/TPOT/TVT, throughput, goodput, fairness, energy/token, GPU-hours, cost per 1k tokens.
   - Generate scenario report (JSON + plots).
   - Emit run manifest (seed, versions, acceptance snapshot) for reproducibility audits.

### 3.3 Event Granularity & Dynamic Adaptation

#### Event Granularity (Iteration-Level vs Request-Level)

- Events are predominantly **iteration-level**, representing single decoding steps (one token), prefill segments, or verification phases rather than whole-request lifetimes.
- This granularity lets the `Scheduler` finish requests as soon as their final iterations complete, preventing head-of-line blocking inside batches, echoing ORCA’s iteration-oriented design [oai_citation:0‡USENIX](https://www.usenix.org/conference/osdi22/presentation/yu?utm_source=chatgpt.com).
- Iteration events capture evolving context length, microbatch composition, and device assignments, enabling accurate TTFT/TPOT accounting and queueing delay modeling per token.
- The simulator may adjust batching, admission, and speculation choices between successive iterations by observing up-to-date system state.

#### Planner Tick Behavior (Dynamic Configuration Adaptation)

- `Planner Tick` events fire on a fixed cadence or when triggers (queue backlog, utilization swings, acceptance degradation) are detected.
- Each tick, the `Scheduler & Planner` gathers fresh telemetry from cooperating modules:
  - `Queueing Model` — queue depths, wait times, drop/deferral statistics.
  - `Device Registry` & parallelism controls — replica utilization, KV/memory headroom.
  - `Acceptance Estimator` — realized acceptance ratios across drafter tiers and `k` settings.
  - `Network Model` — observed link latency/bandwidth if distributed execution is active.
- Candidate configurations span pool sizing (draft vs verify vs fused), tensor/pipeline parallelism, batch/microbatch limits, and speculation knobs (fanout `k`, drafter tier, verification policy).
- For each candidate, the `Planner` combines recent measurements with LUT projections to forecast TTFT, TPOT, throughput, goodput, tail latency, and resource costs, discarding options that violate scenario constraints.
- The configuration that best fits the declared optimization targets is applied through the `Scheduler`, `Mode Selector`, and `Parallelism Controller`. In-flight iterations continue under the previous settings; newly scheduled work adopts the updated plan, a strategy aligned with DistServe’s adaptive planning insights [oai_citation:1‡USENIX](https://www.usenix.org/conference/osdi22/presentation/yu?utm_source=chatgpt.com) [oai_citation:2‡USENIX](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf?utm_source=chatgpt.com).
## 4. Scenario Schema (YAML/JSON)

We define the scenario schema here; `SCENARIO.md` will document it with full examples. High-level structure:

```yaml
scenario_id: mobile_cloud_sd_v1
metadata:
  description: "Mobile drafts offload to two target clusters"
  references: ["DistServe OSDI'24", "Orca OSDI'22"]

workload:
  source: trace://traces/web_mix.jsonl
  arrival_scale: 1.0
  prompt_length_override: null   # optional distribution override
  output_length_override: null
  load_model:
    arrivals: poisson            # poisson | replay
    burst_factor: 1.0            # >1.0 to inject bursts
  slo:
    ttft_ms: 300
    tpot_ms: 35
    tvt_ms: 25

network:
  links:
    - {src: phone_A, dst: target_cluster_1, rtt_ms: 40, bw_mbps: 10}
    - {src: laptop_B, dst: target_cluster_2, rtt_ms: 8,  bw_mbps: 200}
    - {src: target_cluster_1, dst: target_cluster_2, rtt_ms: 3, bw_mbps: 400}
  jitter:
    enabled: true
    rtt_sigma_ms: 2
    bw_sigma_pct: 10
  serialization_overhead_us: 50
  loss_pct: 0.0
  failure_schedule: null          # optional cron or trace path for link outages

hardware_profiles:
  - {id: phone_A,  type: draft,  profile: profiles/draft/mobile_adreno.yaml}
  - {id: laptop_B, type: draft,  profile: profiles/draft/rtx3060.yaml}
  - {id: target_cluster_1, type: target, profile: profiles/target/a100x4.yaml}
  - {id: target_cluster_2, type: target, profile: profiles/target/h100x8.yaml}

cold_start:
  load_ms: 1500
  warm_retention_s: 600
  locality: near_gpu_store       # on | off | near_gpu_store

speculation:
  mode: distributed   # distributed | fused | hybrid
  drafter_models:
    - {name: llama7b,  lut: lut/llama7b_mobile.csv, tier: tiny, max_draft_depth: 3}
    - {name: llama13b, lut: lut/llama13b_rtx3060.csv, tier: medium, max_draft_depth: 6}
  verifier_models:
    - {name: llama70b, lut: lut/llama70b_a100x4.csv, parallelism: {tensor:2, pipeline:2}}
  fused_models:
    - {name: llama34b_fused, lut: lut/llama34b_a100x4_fused.csv}
  acceptance_models:
    - {name: base, file: acceptance/base.yaml, temperature: 0.7}
  acceptance_feedback:
    warmup_requests: 200
    smoothing: ema_0_2
  adaptive_k:
    enabled: true
    k_min: 1
    k_max: 6
    controller: backpressure  # backpressure | pid | bandit
  partial_verification:
    mode: sampled                 # off | first_token | sampled
    sample_rate: 0.25
    rollback_penalty_ms: 8
  mode_switch:
    policy: rtt_load_adaptive
    distributed_threshold_ms: 60
    fused_threshold_ms: 120

scheduler:
  draft:
    policy: iteration_level
    batching:
      max_batch: 32
      max_wait_ms: 30
    drafter_selection:
      strategy: acceptance_portfolio
      escalation_slack_ms: 15
  verify:
    policy: kv_aware_edf
    microbatch:
      max_microbatch: 4
      fallback: per_request
    selective_verification: true
  admission_control:
    policy: slo_guard
  kv_admission:
    max_kv_utilization_pct: 85
    spill_mode: fused

planner:
  enabled: true
  interval_s: 10
  objectives: ["goodput", "p95_tpot"]
  candidates:
    draft_replicas: [2, 4, 6]
    verify_replicas: [4, 8, 12]
    fused_fraction: [0.0, 0.3]
  constraints:
    ttft_p95_ms: 320
    fairness_gap_pct: 10

classes:
  - name: device_tier
    selector: {device_type: ["phone", "laptop", "server"]}
  - name: prompt_bin
    selector: {prompt_len_bins: ["0-256", "257-1024", "1025+"]}

reproducibility:
  random_seed: 1337
  lut_version: vidur_v1_20250101
  policy_version: sched_v2
  scenario_commit: abc1234

metrics:
  output_dir: results/mobile_cloud_sd_v1
  record_energy: true
  record_cost: true
  fairness_classes: ["device_type", "slo_class"]
  track_pareto_frontier: true
```

**Scenario schema check-list** (covered in `SCENARIO.md`):
- Top-level keys: `scenario_id`, `metadata`, `workload`, `network`, `hardware_profiles`, `speculation`, `scheduler`, `planner`, `metrics`, `reproducibility`.
- Every `profile:` path must point to a device profile file (see Section 5.2).
- LUT file references must match the schema defined in Section 5.3.
- Acceptance models describe acceptance-rate curves per `(context_len, method, temperature)` and feed the adaptive `k` / drafter portfolio logic.
- `speculation` may optionally configure `acceptance_feedback`, `partial_verification`, and `mode_switch` behavior.
- `scheduler` can enable `drafter_selection`, `selective_verification`, and `kv_admission` policies.
- `planner.constraints` sets explicit tails/fairness bounds; `metrics.track_pareto_frontier` toggles frontier reporting.
- `workload.load_model` controls arrival process/burstiness; `network.serialization_overhead_us` and `network.loss_pct` capture RPC effects.
- `network.failure_schedule` captures planned link outages; `cold_start` defines warm vs cold load costs; `classes` enumerates fairness cohorts surfaced in reporting.
- `reproducibility` pins run seeds and version metadata for regression testing.

## 5. Data Models

### 5.1 Device Profile Schema

Documented in `device_profiles/README.md`. Each profile captures hardware capabilities and dynamic behavior.

```yaml
id: mobile_adreno
vendor: Qualcomm
role: draft
compute:
  lut_ids: ["llama7b_mobile", "phi3_mini_mobile"]
  max_parallel_tokens: 1
  sustained_tflops: 1.5
  thermal:
    throttle_after_s: 120
    throttle_factor: 0.7
memory:
  ram_gb: 8
  kv_cache_limit_mb: 2048
network:
  default_link: wifi
availability:
  schedule: cron://profiles/availability/mobile.json
energy:
  joules_per_token: 0.05
notes: |
  Use caution for long outputs; thermal throttling kicks in quickly.
```

Target device profiles additionally specify tensor/pipeline parallel limits, GPU memory per replica, and NCCL bandwidth parameters.

Profiles may also encode dynamic throughput modifiers (thermal throttling curves, background load factors), availability schedules, and cold-start costs for model weights. These signals let the simulator model time-varying device performance and checkpoint locality decisions used in mode switching.

### 5.2 LUT Schema

Formalized in `lut_schema.json`. Key fields:

```json
{
  "type": "object",
  "required": ["phase", "model", "hardware", "tensor_parallel", "pipeline_parallel", "context_len", "microbatch", "fanout_k", "dtype", "latency_ms", "peak_mem_mb", "kv_delta_mb", "energy_mj"],
  "properties": {
    "phase": {"enum": ["prefill", "draft_decode", "verify", "fused_decode"]},
    "model": {"type": "string"},
    "hardware": {"type": "string"},
    "tensor_parallel": {"type": "integer", "minimum": 1},
    "pipeline_parallel": {"type": "integer", "minimum": 1},
    "context_len": {"type": "integer", "minimum": 1},
    "microbatch": {"type": "integer", "minimum": 1},
    "fanout_k": {"type": "integer", "minimum": 1},
    "dtype": {"enum": ["fp16", "bf16", "fp8"]},
    "latency_ms": {"type": "number", "minimum": 0},
    "latency_p95_ms": {"type": "number", "minimum": 0},
    "peak_mem_mb": {"type": "number", "minimum": 0},
    "kv_delta_mb": {"type": "number"},
    "energy_mj": {"type": "number", "minimum": 0},
    "cost_nonatt_ms": {"type": "number", "minimum": 0},
    "cost_attn_ms": {"type": "number", "minimum": 0},
    "attn_mode": {"enum": ["batched", "per_request"]},
    "source": {"type": "string"},
    "notes": {"type": "string"}
  }
}
```

LUT files are CSV/Parquet with columns matching this schema. The simulator loads them into the LUT Manager with interpolation on `context_len` and `microbatch`. `source` field captures `"vidur_v1"`, `"microbench"`, etc., and context bins should remain log-spaced to avoid excessive table growth.

`fanout_k` distinguishes speculative depth; `kv_delta_mb` records per-iteration KV growth so admission control can reason about headroom. `latency_p95_ms` (when populated) lets the engine sample service times for tail-aware timing. `cost_nonatt_ms` / `cost_attn_ms` (or, alternatively, `attn_mode`) enable selective-batching cost reconstruction by separating batched non-attention work from per-request attention. Energy entries enable cost or energy-per-token optimization. Separate rows exist for `prefill`, `draft_decode`, `verify`, and `fused_decode` phases to support mode switching and partial verification.

### 5.3 LUT Population Plan

Outlined in `lut_population/PLAN.md`:

1. **Preparation**
   - List models + hardware combos needed by upcoming scenarios.
   - For each combination, enumerate `context_len` grid (e.g., {128, 256, 512, 1k, 2k, 4k, 8k}) and `microbatch` grid.
   - For draft/verify phases include `fanout_k` grid (1..k_max).

2. **VIDUR runs**
   - Configure VIDUR to benchmark each (model, hardware, tensor×pipeline, context_len, microbatch, fanout).
   - Capture latencies, peak memory, energy, KV allocation.
   - Export to CSV; annotate `source`="vidur_{date}".

3. **Sanity checks**
   - Validate monotonicity (latency increases with context_len).
   - Verify KV deltas align with expected (fanout×num_layers×head_dim).
   - Compare spot-checks against real hardware measurements when available.

4. **Aggregation & interpolation**
   - Combine CSVs into canonical LUT files per model+hardware pair.
   - Provide metadata (profiling date, commit SHA) in `lut/README.md`.

5. **Regression detection**
   - Automated script flags >5% deviation from previous LUT version to catch measurement drift.

### 5.4 Acceptance Model Schema

Acceptance curves live in `acceptance/*.yaml` and map `(method, context_len, temperature)` to acceptance probabilities.

```yaml
method: eagle
bins:
  - context_len: 128
    acceptance:
      k=1: {mean: 0.92, p95: 0.98}
      k=2: {mean: 0.85, p95: 0.93}
      k=4: {mean: 0.72, p95: 0.88}
  - context_len: 512
    acceptance:
      k=1: {mean: 0.89, p95: 0.97}
      k=2: {mean: 0.78, p95: 0.90}
      k=4: {mean: 0.63, p95: 0.82}
interpolation: linear
source: "SpecBench24"
notes: "Acceptance curves derived from offline evaluation on dataset X."
```

Each acceptance entry may provide `mean` and optional tail statistics (e.g., `p95`) so controllers can reason about rejection risk and adapt `k` accordingly, and may condition on prompt attributes such as `temperature`, `top_p`, or domain labels when available. Histograms or Gaussian-fit parameters can be supplied when richer modelling is required.

### 5.5 Policy Configuration Schema

Scenario `speculation`, `scheduler`, and `planner` sections bind to the following configuration structures:

- **Drafter selection** — `strategy` (`acceptance_portfolio`, `fixed`, `bandit`), optional `escalation_slack_ms`, and per-tier priorities. Policies consume device priors plus live acceptance estimates.
- **Adaptive `k` & depth** — `controller` type with gain parameters (PID/bandit/backpressure) and smoothing factors defined in `acceptance_feedback`.
- **Partial verification** — `mode` (`all`, `sampled`, `first_token`), `sample_rate`, and `rollback_penalty_ms` to capture mis-speculation recovery cost.
- **KV admission** — `max_kv_utilization_pct`, `spill_mode` (`fused`, `queue`), and eviction ordering (`least_slack_first`, `oldest_context`).
- **Mode switch** — thresholds on RTT/load or a policy identifier (e.g., `rtt_load_adaptive`) that maps to custom logic selecting distributed vs fused execution per request.
- **Fairness classes** — explicit grouping keys so metrics can report attainment gaps per device/workload cohort.

### 5.6 Reproducibility Manifest

Runs provide a manifest capturing deterministic replay settings:

```yaml
reproducibility:
  random_seed: 1337             # simulator RNG seed
  lut_version: vidur_v1_20250101 # LUT bundle or git hash
  policy_version: sched_v2       # scheduler/planner code revision
  scenario_commit: abc1234       # scenario/trace snapshot identifier
```

Validators lock these fields into run metadata, enabling regression comparison across commits and CI.

## 6. Scheduling & Planner Design

### Scheduler Configuration Knobs and Trade-offs


### Scheduler Explorer Modes

#### Mode A – Fixed Hardware, Sweep Policies
- **Inputs**: draft/target capacities (per cluster), router/pool assignments, batching knobs, speculation parameters (`k`, stop rule `h`), KV utilization rules, per-phase parallelism (TP/PP/CP/MoE, collective algorithm).
- **Objective**: maximize SLO-goodput subject to TTFT/TPOT tail constraints (p95/p99) and optional cost/GPU-hour bounds.
- **Search**: grid/random/Bayesian sweeps with early pruning (queueing bounds, micro-sims), leveraging parallel experiment execution (Multiverse-style) to accelerate runs.
- **Outputs**: top-k policy bundles per hardware point plus Pareto curves (goodput vs TTFT/TPOT tails vs cost) for comparing co-location vs disaggregation, chunk sizes, paging, etc.

#### Mode B – Fixed Policy, Find Max Stable Load
- **Inputs**: selected scheduler/policy, hardware topology, workload generator (open-loop trace scaling or closed-loop session generator with think time).
- **Method**: scale offered load (QPS multiplier or active sessions) via binary search; warm-up, measure steady-state queues and SLO metrics.
- **Stability criteria**: declare instability when queues diverge or TTFT/TPOT tails exceed thresholds; otherwise continue.
- **Outputs**: capacity card summarising max SLO-goodput, throughput/latency curve, pool utilization, and interference (prefill vs decode).

#### Explorer Plumbing
- Experiment spec describing policy sweeps or load ramps.
- Runner with warm-up + measurement windows, metrics collector, and early pruner for failing configs.
- Aggregator producing tables/plots (Pareto frontiers, capacity cards) for reporting.


- **Queue Topology**
  - Maintain separate iteration queues for prefill and decode; each queue may target a dedicated GPU pool (disaggregated) or share devices (co-located).
  - Queue policies (FIFO, EDF/least-slack-first, age-based) exposed via `scheduler.prefill.queue_policy` and `scheduler.decode.queue_policy`.
  - Support multiple SLO classes: `scheduler.priority_classes[*]` defines weights, max wait, slack thresholds.

- **Pool Assignment (Co-locate vs Disaggregate)**
  - `scheduler.prefill.pool` and `scheduler.decode.pool` reference cluster names defined in `auto_topology.clusters`. Flip these to compare co-location versus DistServe-style disaggregation.

- **Batch Formation**
  - Continuous batching for decode (`scheduler.decode.continuous=true`), with controls for `max_batch_tokens`, `max_batch_requests`, and `max_wait_us`.
  - Chunked prefill (`scheduler.prefill.chunk_tokens`) and selective batching (`scheduler.decode.selective_ops`) to avoid prompt-induced stalls and respect shape-aware batching.

- **Delay Budget & Admission Timing**
  - `max_wait_us` per phase/SLO bucket bounds batching delay; scheduler chooses eligible jobs each tick based on arrival ≤ now, KV headroom, SLO slack, and shape bucket.
  - Config knobs live under `scheduler.decode.selection` / `scheduler.prefill.selection`.

- **KV & Memory Rules**
  - `scheduler.kv.max_utilization_pct` enforces KV headroom. Optional paging (`scheduler.kv.paging`) declares page size, page-in penalty, and eviction policy for vLLM-style demand paging.

- **Parallelism Plans & Collectives**
  - Phase-specific parallelism (`scheduler.prefill.parallelism`, `scheduler.decode.parallelism`) describe TP/PP/CP/MoE degrees and collective algorithm (`ring`, `tree`, `multishot`). Planner can swap plans dynamically.

- **Preemption & Fairness**
  - Soft preemption between iterations (`scheduler.decode.preemption.enabled`, `slack_threshold_ms`) plus age/EDF weighting (`scheduler.decode.age_weight`) to keep tails in check.

- **Router ↔ Scheduler Feedback**
  - Router performs global target selection; per-GPU scheduler applies the above knobs. Optional feedback (`scheduler.router_feedback`) exports queue depth/slack/acceptance for router load balancing.

- **Objective Controls**
  - `scheduler.objective` chooses throughput-centric, latency-centric, or balanced goals; telemetry logs TTFT/TPOT/throughput so experiments (e.g., co-located vs disaggregated, chunk sizes, paging toggles) can be compared.

These knobs mirror the mechanisms used in ORCA, Sarathi-Serve, DistServe, Triton, vLLM, and LLMServingSim, letting us toggle features without code changes.

### 6.1 Draft Scheduler

- Policy: **Iteration-level** (ORCA-inspired) with one tick per speculative iteration.
  - Each tick drains completed work, re-evaluates queue slack, and forms new microbatches subject to TTFT SLO forecasts.
- Batching: `max_batch` + `max_wait_ms`, ensuring predicted TTFT ≤ SLO using LUT latency plus network RTT.
- Adaptive speculation: controllers combine verify queue depth, KV headroom, and acceptance predictions to adjust fanout `k` and draft depth every tick.
  - Load-aware controller: shrink `k` or depth when verify queue wait or KV utilization crosses thresholds; expand when lightly loaded and acceptance confidence is high.
  - Network-aware guard: when outbound link utilization or RTT spikes, bias toward fused mode or lower `k` to avoid saturating slow paths.
- Drafter portfolio: maintain multiple draft models (tiny/mid/large) and choose per request using prompt features, live acceptance, and escalation slack.
- Mid-decode escalation: optionally switch a request to a larger drafter at iteration boundaries when observed acceptance falls below targets.
- Prefix reuse: optionally reuse draft KV for requests sharing prompt prefixes; scheduler emits reuse hints to KV manager.

### 6.2 Verify Scheduler

- Policy: **KV-aware EDF (earliest deadline first)** with selective batching.
- Microbatching: attempt `max_microbatch`; fallback to per-request when context lengths diverge or acceptance indicates high error risk.
- Partial verification: support `all`, `sampled`, or `first_token` verification; when sampling, add rollback work upon rejection.
- Shape/content cohorting: cluster requests by context length buckets and optional prefix hashes to maximize shared attention computation.
- Cost modelling: use LUT `cost_nonatt_ms` for batched feed-forward/MLP/norm work and apply `cost_attn_ms` per request unless context bins align (then reuse batched-attention entries).
- Preemption: when EDF slack turns negative for a request, preempt current verify work and service the at-risk request first.
- KV guard: only admit batch if total KV footprint ≤ available memory; otherwise delay or spill to fused mode per `kv_admission` policy, optionally invoking eviction ordering (`least_slack_first`, `oldest_context`).

### 6.3 Planner (Goodput Maximizer)

- Inputs: recent measurements (goodput, queue lengths, acceptance, KV utilization, link utilization/RTT, energy), candidate pool sizes, parallelism, `k`, microbatch, mode mix.
- Objective: maximize `goodput = (TTFT_OK & TPOT_OK & TVT_OK requests) / (num_GPUs × time)` while respecting constraint bounds (tail latency, fairness gap, energy budget) and optionally minimizing cost.
- Algorithm: iterative search (grid, hill-climb, or bandit). Build Pareto frontier subject to compute **and** network bandwidth constraints, then pick the best feasible configuration.

```
def plan(pools, candidates, metrics):
    best = None
    for cfg in candidates:
        est = simulate_short(cfg, metrics.recent_trace, constraints=metrics.constraints)
        best = choose_frontier(best, (cfg, est))
    apply(best.cfg)
```

- Requires fast-forward simulation for each candidate (short horizon using LUTs) and optional analytic approximations for energy/cost.

`choose_frontier` retains configurations that satisfy constraints and dominate the current Pareto set; fallback tie-breakers favor higher goodput, then lower cost/energy.

### 6.4 Mode Selector & Cold-Start Handling

- Inputs: network RTT/BW, device availability, queue pressure, and acceptance forecasts.
- Policy: threshold-based (`rtt_load_adaptive`) or learned classifier that chooses distributed vs fused execution per request class.
- Cold starts: models may incur load latency; scheduler amortizes by pre-warming pools or accounting for warmup in TTFT predictions.
- Elastic scaling: when planner changes pool sizes, mode selector considers checkpoint locality (ServerlessLLM-style) and prefetch status before routing traffic.

## 7. Execution Modes

1. **Distributed SD**
   - Draft devices generate speculative bundles; target verifies.
   - Network events: prompt upload → speculative bundle upload → verified token stream back.

2. **Fused SD**
   - Target server runs fused draft+verify model; draft device submits prompt only; results returned after verification.
   - No speculative bundle network traffic; larger server compute cost.

3. **Hybrid**
   - Policy decides per request which mode to use (e.g., high RTT → fused; fast draft device → distributed).
   - May route to already-warm fused targets when cold-start cost elsewhere would violate TTFT.

Simulator must support all three; scenario config selects default mode (`speculation.mode`) and optional mix.

Mode switching policies consume telemetry from the mode selector and may prefetch checkpoints or reserve KV to minimize cold-start penalties. Partial verification settings apply to both distributed and fused paths, with LUT phase data informing latency/energy estimates under each mode.

## 8. Metrics & Reporting

Metrics collected per scenario:

- **Latency**: TTFT, per-token TPOT, per-request TVT (max over tokens), distribution (avg, P90, P95, P99).
- **Throughput & Goodput**: total requests/sec, SLO-satisfying requests/sec, goodput per GPU.
- **Queueing**: per-phase wait-time distributions, utilization, and defer/dropped counts due to KV or SLO guards.
- **Fairness**: SLO attainment per device class / workload class; Jain’s fairness index; configurable attainment gaps.
- **Resource & Cost**: GPU utilization, KV occupancy, energy/token, joules/request, cost/1k tokens, network bandwidth usage.
- **Acceptance & Speculation**: acceptance rate vs context length, adaptive `k` timeline, drafter switching events, partial verification savings vs rollback penalties.
- **Tail Plots**: P95/P99 TTFT/TPOT/TVT per fairness class, correlated with queue depth and mode decisions.
- **Pareto Frontiers**: trade-off curves across goodput, tail latency, energy/cost, and fairness.
- **Network Diagnostics**: link utilization, serialization overhead, retry counts, and time spent in network queues.
- **Reproducibility**: captured seed/version manifest, run-to-run variance against baseline, regression alerts.
- **Policy diagnostics**: queue lengths, mode selections (distributed/fused), KV eviction/spill events, cold-start occurrences.

Outputs: JSON summary, CSV details, optional plots (Matplotlib/Altair).

## 9. Implementation Roadmap (files to create)

| File / Directory                         | Purpose |
|------------------------------------------|---------|
| `SCENARIO.md`                            | Detailed schema explanation + full YAML example + validation rules. |
| `device_profiles/README.md`              | Device profile schema, templates for draft & target devices. |
| `device_profiles/*.yaml`                 | Concrete device profiles (mobile, laptop, GPU cluster). |
| `lut_schema.json`                        | JSON schema for LUT files (Section 5.2). |
| `lut_population/PLAN.md`                 | VIDUR workflow, measurement plan, regression tests (Section 5.3). |
| `acceptance/README.md`                   | Acceptance model schema and examples. |
| `DESIGN.md` §6                           | Scheduler & planner APIs, pseudocode for baseline + advanced policies (kept inline). |
| `scenarios/examples/*.yaml`              | Example scenario files (mobile_cloud, low_rtt_datacenter, mixed). |
| `scripts/validate_scenario.py`           | Schema validator for scenarios & LUTs. |
| `scripts/plot_metrics.py`                | Plot generation utilities. |

## 10. References (for developers)

- **DistServe** (OSDI 2024) — workload disaggregation, goodput planning.
- **Orca** (OSDI 2022) — iteration-level scheduling, selective batching, KV awareness.
- **FedScale / FLUTE** (PMLR 2022) — trace-driven heterogeneous simulation principles.
- **Speculative Decoding Surveys** (2024–2025) — acceptance-rate modelling, method comparisons.
- **ServerlessLLM** (OSDI 2024) — checkpoint locality & cold-start considerations (elastic targets).

---

This DESIGN document is the authoritative specification for the simulator implementation. As related components are developed, update the referenced auxiliary documents to keep schemas and workflows synchronized.

## Appendix A — Additional Features & Refinements

1. **Priority / Preemption** — Enable preemptive verify scheduling (EDF / least-slack-first) to shield SLOs during bursts; log preemptions for fairness and tail analysis (DistServe-style SLO focus).
2. **Dynamic / Online Adaptation** — Allow policies to update controller parameters online from telemetry (acceptance, queues) without re-profiling; expose a “live policy state” hook in the scheduler to support adaptive `k` (speculative decoding surveys).
3. **Multi-level Speculation** — Optionally chain multiple drafters (tiny → mid → heavy) before verification to explore hierarchical SD strategies.
4. **Arrival & Burst Modeling** — Support configurable arrival processes (Poisson, replay) and burst factors; surface per-phase queue growth to study instability thresholds (DistServe workload sensitivity).
5. **Shape Divergence Modeling** — Track context-length divergence inside cohorts to decide when attention can be batched or must remain per-request (ORCA selective batching).
6. **Cold-Start & Locality** — Model warm vs cold checkpoint loads and near-GPU storage; route traffic toward warm pools to curb TTFT (ServerlessLLM locality insights).
7. **Tail-aware LUTs** — Optionally store `latency_p95_ms` or distribution parameters per bin to sample realistic service times (VIDUR LUT accuracy practice).
