# Distributed Speculative Decoding Simulator — System Design

## 1. Objectives

We are building a trace-driven, heterogeneity-aware simulator for **distributed speculative decoding (SD)**. The framework must:

- Model heterogeneous draft devices (phones, laptops, edge GPUs) and target servers (datacenter GPUs) with realistic compute, energy, memory, and network characteristics.
- Support both **distributed SD** (draft generates speculative tokens, target verifies) and **fused SD** (target performs draft + verify locally).
- Scale to thousands of concurrent requests using iteration-level (token-level) scheduling with explicit SLO targets for TTFT, TPOT, and TVT (time-to-verify-token).
- Plan GPU pool sizes, microbatch sizes, and speculation parameters to maximize **SLO goodput** (requests that meet TTFT/TPOT/TVT per GPU-second).
- Produce accurate performance estimates using **lookup tables (LUTs)** populated from VIDUR/microbenchmarks.
- Offer pluggable scheduling, speculation, and admission policies to evaluate new algorithms quickly.

This document lays out the core components and specifies the auxiliary design files that developers will implement:

1. `SCENARIO.md` (describes the YAML/JSON scenario schema and example snippets).
2. `device_profiles/README.md` (device profile schema and conventions).
3. `lut_schema.json` (formal schema for performance LUT entries).
4. `lut_population/PLAN.md` (VIDUR-based population workflow & validation steps).
5. `policies/README.md` (scheduler & planner APIs / pseudocode).

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
                           +------------------+
         |                       |
         v                       v
 +-------------------+    +-----------------------+
 | LUT Manager       |    | Event Engine          |
 |  - VIDUR-backed   |    |  - Prefill queue      |
 |  - Interpolator   |    |  - Draft decode queue |
 +-------------------+    |  - Verify queue       |
                           |  - Network events     |
                           +-----------------------+
                                        |
                                        v
                            +-----------------------+
                            | Scheduler & Planner   |
                            |  - Adaptive k         |
                            |  - Pool sizing        |
                            |  - Admission control  |
                            +-----------------------+
```

### Key modules

| Module              | Responsibilities |
|---------------------|------------------|
| Scenario Parser     | Load YAML/JSON inputs defining topology, devices, workloads, SD modes, policies. |
| Device Registry     | Create device instances with performance models, energy, availability schedules, KV capacities. |
| LUT Manager         | Load, validate, and interpolate VIDUR-produced LUT entries. Provide per-operation latency, memory, energy lookups. |
| Workload Manager    | Replays trace `W` (arrival time, prompt length, output length, SLO class) or synthesizes workloads. |
| Network Model       | Simulates RTT, bandwidth, loss/jitter for prompt upload, speculative bundle upload, and result streaming. |
| KV Manager          | Track per-request KV cache allocations, handle eviction/pinning policies. |
| Event Engine        | Drive discrete events: arrivals, prefill completion, draft iteration completion, verification completion, network transfers, timeouts. |
| Scheduler           | Token-level scheduling for draft and verify phases (EDF/least slack), adaptive fanout `k`, target vs fused mode choice, admission control. |
| Planner             | DistServe-style goodput maximization: search pool sizes, microbatch sizes, draft-model choices, and speculation parameters. |
| Metrics & Logger    | Produce latency distributions, goodput, throughput, fairness, energy, KV utilization, cost per 1k tokens. |

## 3. Simulation Flow

1. **Initialization**
   - Parse scenario file.
   - Load device profiles and LUTs (validate compatibility).
   - Initialize workload trace reader and network topology.
   - Configure scheduling/policy modules.

2. **Main event loop**
   - Inject arrivals from trace (add to draft prefill queue). Each arrival carries metadata: prompt length, desired tokens, SLO class, device of origin.
   - When scheduler triggers, form batches/microbatches:
     - Prefill (draft or target) batches use **size/timeout** rule constrained by TTFT SLO.
     - Draft decode iterations use adaptive fanout `k` + selective batching.
     - Verify iterations group requests by context length where possible; otherwise fallback to per-request attention.
   - Query LUT manager to obtain per-iteration latency, memory, energy for specified hardware, parallelism, context length, microbatch, fanout.
   - Generate network events for prompt/speculative bundle uploads and verification responses.
   - Update KV manager for allocations and releases after each iteration.
   - Record per-request metrics (TTFT, TPOT, TVT, energy, KV footprint).

3. **Planner loop** (periodic)
   - Sample recent statistics (queue lengths, acceptance rate, success rate).
   - Enumerate candidate pool splits `(num_draft_replicas, num_verify_replicas)` and parallelism configs.
   - Estimate goodput via short “what-if” simulation windows or analytic approximations.
   - Apply best configuration (possibly adjusting `k`, draft-model choice, microbatch size).

4. **Completion**
   - Aggregate metrics: mean/median/P95/P99 TTFT/TPOT/TVT, throughput, goodput, fairness, energy/token, GPU-hours, cost per 1k tokens.
   - Generate scenario report (JSON + plots).

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

hardware_profiles:
  - {id: phone_A,  type: draft,  profile: profiles/draft/mobile_adreno.yaml}
  - {id: laptop_B, type: draft,  profile: profiles/draft/rtx3060.yaml}
  - {id: target_cluster_1, type: target, profile: profiles/target/a100x4.yaml}
  - {id: target_cluster_2, type: target, profile: profiles/target/h100x8.yaml}

speculation:
  mode: distributed   # distributed | fused | hybrid
  drafter_models:
    - {name: llama7b,  lut: lut/llama7b_mobile.csv}
    - {name: llama13b, lut: lut/llama13b_rtx3060.csv}
  verifier_models:
    - {name: llama70b, lut: lut/llama70b_a100x4.csv, parallelism: {tensor:2, pipeline:2}}
  fused_models:
    - {name: llama34b_fused, lut: lut/llama34b_a100x4_fused.csv}
  acceptance_models:
    - {name: base, file: acceptance/base.yaml}
  adaptive_k:
    enabled: true
    k_min: 1
    k_max: 6
    controller: backpressure  # backpressure | pid | bandit

scheduler:
  draft:
    policy: iteration_level
    batching:
      max_batch: 32
      max_wait_ms: 30
  verify:
    policy: kv_aware_edf
    microbatch:
      max_microbatch: 4
      fallback: per_request
  admission_control:
    policy: slo_guard

planner:
  enabled: true
  interval_s: 10
  objectives: ["goodput", "p95_tpot"]
  candidates:
    draft_replicas: [2, 4, 6]
    verify_replicas: [4, 8, 12]
    fused_fraction: [0.0, 0.3]

metrics:
  output_dir: results/mobile_cloud_sd_v1
  record_energy: true
  record_cost: true
```

**Scenario schema check-list** (covered in `SCENARIO.md`):
- Top-level keys: `scenario_id`, `metadata`, `workload`, `network`, `hardware_profiles`, `speculation`, `scheduler`, `planner`, `metrics`.
- Every `profile:` path must point to a device profile file (see Section 5.2).
- LUT file references must match the schema defined in Section 5.3.
- Acceptance models describe acceptance-rate curves per `(context_len, method, temperature)`.

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
    "peak_mem_mb": {"type": "number", "minimum": 0},
    "kv_delta_mb": {"type": "number"},
    "energy_mj": {"type": "number", "minimum": 0},
    "source": {"type": "string"},
    "notes": {"type": "string"}
  }
}
```

LUT files are CSV/Parquet with columns matching this schema. The simulator loads them into the LUT Manager with interpolation on `context_len` and `microbatch`. `source` field captures `"vidur_v1"`, `"microbench"`, etc.

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
  - {context_len: 128, acceptance: {k=1: 0.92, k=2: 0.85, k=4: 0.72}}
  - {context_len: 512, acceptance: {k=1: 0.89, k=2: 0.78, k=4: 0.63}}
interpolation: linear
source: "SpecBench24"
notes: "Acceptance curves derived from offline evaluation on dataset X."
```

## 6. Scheduling & Planner Design

### 6.1 Draft Scheduler

- Policy: **Iteration-level** (ORCA-inspired).
- Batching: `max_batch` + `max_wait_ms`, ensuring predicted TTFT ≤ SLO.
- Adaptive fanout `k`: controller monitors verify queue, acceptance rate; reduces `k` when verify backpressure high.
- Draft model selection: per-request policy chooses among available draft models (size vs predicted accuracy) using prompt features and acceptance prior.

### 6.2 Verify Scheduler

- Policy: **KV-aware EDF (earliest deadline first)**.
- Microbatching: attempt `max_microbatch`; fallback to per-request when context lengths diverge or acceptance indicates high error risk.
- KV guard: only admit batch if total KV footprint ≤ available memory; otherwise delay or spill to fused mode.

### 6.3 Planner (Goodput Maximizer)

- Inputs: recent measurements (goodput, queue lengths, acceptance), candidate pool sizes, parallelism, `k`, microbatch.
- Objective: maximize `goodput = (TTFT_OK & TPOT_OK & TVT_OK requests) / (num_GPUs × time)` with penalty terms for fairness or energy if configured.
- Algorithm: iterative search (grid or hill-climb). Example pseudocode:

```
def plan(pools, candidates, metrics):
    best = None
    for cfg in candidates:
        est = simulate_short(cfg, metrics.recent_trace)
        if better(est, best):
            best = (cfg, est)
    apply(best.cfg)
```

- Requires fast-forward simulation for each candidate (short horizon using LUTs).

## 7. Execution Modes

1. **Distributed SD**
   - Draft devices generate speculative bundles; target verifies.
   - Network events: prompt upload → speculative bundle upload → verified token stream back.

2. **Fused SD**
   - Target server runs fused draft+verify model; draft device submits prompt only; results returned after verification.
   - No speculative bundle network traffic; larger server compute cost.

3. **Hybrid**
   - Policy decides per request which mode to use (e.g., high RTT → fused; fast draft device → distributed).

Simulator must support all three; scenario config selects default mode (`speculation.mode`) and optional mix.

## 8. Metrics & Reporting

Metrics collected per scenario:

- **Latency**: TTFT, per-token TPOT, per-request TVT (max over tokens), distribution (avg, P90, P95, P99).
- **Throughput & Goodput**: total requests/sec, SLO-satisfying requests/sec, goodput per GPU.
- **Fairness**: SLO attainment per device class / workload class; Jain’s fairness index.
- **Resource**: GPU utilization, KV occupancy, energy/token, cost/1k tokens, network bandwidth usage.
- **Policy diagnostics**: acceptance rate vs context length, adaptive `k` timeline, queue lengths.

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
| `policies/README.md`                     | Scheduler & planner APIs, pseudocode for baseline + advanced policies. |
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
