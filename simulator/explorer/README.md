# Explorer Guide

This module orchestrates repeatable experiments against the simulator.

## Usage

```bash
python -m explorer.runner --list                    # show available manifests
python -m explorer.runner --experiment baseline_vs_disagg
python -m explorer.runner --experiment scheduler_search --dry-run
```

Generated configs and logs are written to `explorer/output/<experiment>/<run>/`.

### Manifest layout

Each manifest in `explorer/experiments/` contains:

```yaml
experiment:
  name: scheduler_search
  description: "Search simple router and batching variants"
  runs:
    - name: router_random
      config: configs/explorer/baseline.yaml
      overrides:
        router: random
```

`config` points at a base YAML file under `configs/`. `overrides` is merged
recursively before the temporary config is written.

## Experiments

1. **baseline_vs_disagg.yaml** – co-located vs disaggregated pools.
2. **scheduler_search.yaml** – small sweep over routing/batching policies on a fixed topology.
3. **capacity_search.yaml** – scale the offered rate under a fixed scheduler to find the knee.

Each manifest documents the exact knobs being varied so results remain reproducible.
