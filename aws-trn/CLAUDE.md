# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements distributed and fused speculative decoding for AWS Trainium using Meta LLaMA models. It provides two architectures for optimizing inference throughput:
- **Distributed Architecture**: Client-side draft model with multiple round-trips to target server
- **Fused Architecture**: Server-side fused model with single round-trip

## Development Environment Setup

### AWS Trainium Environment
Before running any code, ensure you're in the correct environment:
```bash
source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate
```

### Required Environment Variables
```bash
export NEURON_CC_FLAGS="--model-type=transformer"
export NEURON_RT_NUM_CORES="2"
```

### Dependencies
```bash
pip install grpcio==1.71.0 grpcio-tools==1.66.2
pip install gevent
pip install --upgrade transformers-neuronx

# For simulator
pip install simpy pyyaml matplotlib numpy
```

## Common Commands

### Running Distributed Architecture

Start target server:
```bash
python main.py --role target --model /home/ubuntu/models/llama-3.1-8b/ --port 50051 --sequence_length 128 --batch 2 --profile --top_p 1.0
```

Run draft client:
```bash
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b/ --target_host <HOST_IP> --port 50051 --prompt_text prompt.txt --sequence_length 128 --profile --top_p 1.0 --temperature 0.9
```

### Running Fused Architecture

Start fused server:
```bash
python main.py --role fused_target --draft_model /home/ubuntu/models/llama-3.2-1b --target_model /home/ubuntu/models/llama-3.1-8b --port 50051 --sequence_length 128 --batch 2 --tp_degree 2 --speculation_length 5
```

Run fused client:
```bash
python main.py --role fused_client --target_host <HOST_IP> --port 50051 --prompt_text prompt.txt --temperature 0.9 --top_p 1.0 --speculation_length 5 --profile
```

### Model Verification
Test individual models without speculative decoding:
```bash
# Verify target model
python main.py --role verify_target --model /home/ubuntu/models/llama-3.1-8b --prompt_text prompt.txt --sequence_length 128 --profile

# Verify draft model
python main.py --role verify_draft --model /home/ubuntu/models/llama-3.2-1b --prompt_text prompt.txt --sequence_length 128 --profile
```

### Performance Comparison (compare_architectures.py mentioned in README but file doesn't exist)
```bash
# Note: compare_architectures.py script is referenced in README but not present in the repository
# python compare_architectures.py --prompt-file prompt.txt --num-runs 3
```

### Regenerate gRPC Code
```bash
cd grpc_comm
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
# After generation, fix import in inference_pb2_grpc.py:
# Change: import inference_pb2 as inference__pb2
# To: from . import inference_pb2 as inference__pb2
```

### Clean Neuron Compilation Cache
```bash
rm -r /var/tmp/neuron-compile-cache
```

## High-Level Architecture

### Core Components

**main.py**: Entry point handling all roles (target, draft, verify_target, verify_draft, fused_target, fused_client). Parses arguments and launches appropriate workers. Key responsibilities:
- Role dispatch based on --role argument
- Model loading via `run_model()` for verification roles
- Worker initialization for distributed/fused modes

**inference/model_loader.py**: Handles model loading and compilation for Neuron. Key functions:
- `load_model()`: Loads and optionally compiles models for Neuron
- `compile_model()`: Compiles LLaMA models with specific Neuron configurations
- `load_fused_speculative_model()`: Creates fused speculative decoder combining draft and target models
- `NeuronHFAdapterWrap`: Wrapper managing KV-cache pointers for batched inference
- Bucket management: Speculation lengths [3, 5] and batch sizes [1, 2, 3]

**gRPC Communication (grpc_comm/)**: Protocol buffer definitions for client-server communication supporting both distributed and fused architectures. Key messages:
- `StartRequest/Response`: Session initialization
- `VerifyRequest/Response`: Token verification (distributed)
- `GenerateRequest/Response`: Full generation (fused)
- `VerifyBatchRequest/Response`: Batched verification for multiple sessions

### Distributed Architecture Flow
1. Draft client (`inference/draft_worker.py`) generates speculative tokens using draft model
2. Sends token batches to target server for verification via gRPC
3. Target server (`inference/target_worker.py`) accepts/rejects tokens and returns results
4. Process repeats until generation completes

### Fused Architecture Flow
1. Client (`inference/fused_draft_worker.py`) sends single request with prompt
2. Server (`inference/fused_target_worker.py`) uses FusedSpeculativeDecoder (both models on server)
3. Server returns complete generated text in single response

## Key Implementation Details

- Models are compiled for specific sequence lengths (default: 128) and batch sizes
- Speculation length buckets: [3, 5] to support different gamma values
- Batch size buckets: [1, 2, 3] for dynamic batching
- Performance profiling saves to CSV/JSON files when --profile flag is used
- Multiple prompts can be processed concurrently from prompt text files (prompt1.txt - prompt7.txt)
- Tensor parallelism degree controlled by NEURON_RT_NUM_CORES environment variable

## Critical Parameters

- `--sequence_length`: Must match compiled model or will trigger recompilation
- `--gamma`: Draft tokens per verification (distributed only, default: 4)
- `--speculation_length`: Tokens to speculate at once (fused only, default: 5)
- `--tp_degree`: Tensor parallelism degree (default: 4, use 2 for trn1.2xlarge)
- `--batch`: Batch size for compilation (affects memory usage)
- `--temperature`: Sampling temperature (default: 1.0)
- `--top_p`: Nucleus sampling parameter (default: 0.9)

## Performance Considerations

- Distributed architecture: ~34 tokens/second throughput
- Fused architecture: ~19 tokens/second throughput
- Model compilation is cached in /var/tmp/neuron-compile-cache
- First run will be slow due to compilation
- Use same sequence_length across runs to avoid recompilation
- Fused models always use batch_size=1 internally

## Simulator (simulator/)

### Overview
Comprehensive system-level simulator for distributed speculative decoding using SimPy. Models mixed prefill/decode batching, conversation-based workloads, and realistic queueing dynamics.

### Running the Simulator
```bash
# Run single configuration
python simulator/sim.py --config simulator/config.yaml

# Run homogeneous scaling experiment (1-100 drafts)
cd simulator/scripts
python homogeneous_draft.py

# Analyze existing results
python simulator/scripts/analyze_existing_data.py
python simulator/scripts/measure_arrival_rates.py
python simulator/scripts/theoretical_analysis.py

# Results saved to simulator/results/
```

### Key Configuration (simulator/config.yaml)
- **sim_time_ms**: Simulation duration (default: 60000ms)
- **burn_in_ms**: Warm-up period excluded from metrics (default: 1000ms)
- **gamma**: Draft tokens per chunk (default: 4)
- **router**: Load balancing strategy (round_robin, jsq2, wjsq2)
- **devices**: Target and draft device specifications
- **connections**: Network latencies and acceptance rates between draft-target pairs
- **workload**: Arrival patterns (poisson/deterministic) and rates

### Simulator Architecture
- **SimPy Process Model**: Event-driven simulation with virtual time
- **Job Types**: Prefill (initial prompt) and decode (speculative verification)
- **Batching**: Mixed prefill/decode with head-of-line blocking
- **Routing**: JSQ2 (Join Shortest Queue with 2 choices) for load balancing
- **Metrics**: Throughput, latency, queue depths, batch utilization

## Important Development Principles

### Evidence-Based Debugging
**Principle:** Comments from colleagues, professors, or experts should be treated as hypotheses to test, not truths to justify. Always collect concrete evidence before accepting explanations.

**Real Example - Simulator Step Pattern Debugging:**
- **Initial hypothesis**: Head-of-line blocking from mixing 50ms prefills with 37ms decodes caused step pattern
- **Evidence revealed**: Batch-filling threshold with 10ms window was the actual cause
  - Below 52 drafts: ~45% batch utilization
  - At 52-53 drafts: Jump to 100% utilization
- **Lesson**: The proposed cause was entirely wrong. Evidence trumped expertise.

**Always follow this process:**
1. Treat external input as hypothesis to test, not fact to prove
2. Design experiments to test the hypothesis
3. Collect actual metrics/statistics/data
4. Let evidence speak for itself, even if it contradicts the hypothesis
5. Be willing to conclude "the hypothesis was wrong"

## Core Research Focus: Large-Scale Distributed SD Simulation Framework

### Research Objective
Building an accurate and efficient simulation framework for large-scale heterogeneous distributed speculative decoding systems, capable of determining optimal scheduling algorithms under various system configurations.

### Key System Characteristics
- **Heterogeneous Devices**: Draft clients (phones, laptops) and target servers with varying capabilities
- **Multiple Architectures**: Support both distributed (draft computes, target verifies) and fused (server does everything) modes
- **Large Scale**: Many draft and target servers communicating with each other
- **Complex Variables**: Queue policies, device counts, input distributions, processing powers

### Framework Design Goals

#### 1. Large-Scale Simulation Support
- **Shared process on single GPU**: Not all devices need resources simultaneously
- **Dynamic resource allocation**: Allocate only when needed, keep alive and reuse
- **Pipelining**: Next job uses resource when previous finishes
- **Trace-based simulation**: Fast-forward using offline traces (like DistServe)

#### 2. Flexibility & Modularity
- **Swappable algorithms**: Easy testing of different scheduling policies
- **Modular design**: Quick experimentation with new components
- **Support different optimization schemes**: Device pools, split prefill/decode queues
- **Multiple optimization targets**: Throughput, latency, fairness, goodput

#### 3. Accuracy & Efficiency
- **Performance lookup tables**: Use VIDUR for latency profiling, create lookup tables
- **Realistic workloads**: Trace-based inputs with real user behavior patterns
- **Heterogeneous modeling**: Non-IID data, devices with different capacities
- **Communication modeling**: Network latencies, device availability patterns

#### 4. Input Configuration Structure
```yaml
Model Configuration:
  - Model dimensions: d_model, n_layers, n_heads, head_dim, ffn_dim, vocab_size
  - Or simply: model_name (e.g., "llama-3.1-8b")
  - Parallelism: intra_op, dtype (fp16/bf16), activation_checkpointing

Workload Configuration:
  - Sequence/batch: prefill_lengths, decode_context_lengths, microbatch_sizes
  - Trace workload: arrival_rate, prompt_length_distribution, output_lengths
  - SLOs: TTFT (Time-To-First-Token), TPOT (Time-Per-Output-Token)
  - Topology: same-node vs cross-node, bandwidth specifications
```

#### 5. Key Technical Approaches
- **Pre-loading/caching**: Load data beforehand for faster simulation
- **Simulate critical components only**: Focus on performance-critical paths
- **Offline trace collection**: Communication patterns, device status information
- **No real network simulation**: Use traces to model network delays

### Papers to Leverage

**Read and Applied:**
- DistServe (OSDI'24): Disaggregated prefill/decode, trace-based simulation
- Orca (OSDI'22): Continuous batching, iteration-level scheduling
- Splitwise: Heterogeneous speculative decoding
- FedScale (MLR'22): Large-scale FL simulation techniques
- Branch-Train-Merge: Federated learning with heterogeneous models

**To Read:**
- LoongServe (OSDI'24): Potential scheduling insights
- Heterogeneous scheduling papers: More on device heterogeneity
- Additional systems papers for simulation techniques

### Expected Contributions
1. **First large-scale heterogeneous SD simulator**: Supporting both distributed and fused modes
2. **Optimal scheduling algorithms**: For heterogeneous draft-target pairs
3. **Performance characterization**: Under various system configurations
4. **Practical deployment insights**: For real-world heterogeneous SD systems