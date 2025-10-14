#!/usr/bin/env python3
"""Sweep fused vs distributed mixes while keeping total draft capacity fixed."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_PATH = REPO_ROOT / "simulator" / "sim.py"

CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
DISTRIBUTED_CONFIG = CONFIG_ROOT / "explorer" / "baseline.yaml"
FUSED_CONFIG = CONFIG_ROOT / "explorer" / "fused.yaml"
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "fused_vs_distributed"

TOTAL_RPS = 40.0
FUSED_FRACTIONS = [i / 100 for i in range(0, 101, 2)]  # sample every 2 fused drafts
SEEDS = [123, 211, 347]


class SimulationError(RuntimeError):
    pass


def _load_raw_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    if "auto_topology" not in data:
        raise SimulationError(f"Config {path} must use auto_topology")
    return data


def _total_drafts(raw_cfg: dict) -> int:
    total = 0
    for cluster in raw_cfg.get("auto_topology", {}).get("clusters", []):
        total += int(cluster.get("drafts", {}).get("count", 0))
    return total


def _write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _parse_metrics(stdout: str) -> Dict[str, float]:
    start = stdout.find("===METRICS_JSON===")
    end = stdout.find("===END_METRICS_JSON===")
    if start == -1 or end == -1:
        raise SimulationError("METRICS_JSON block not found in simulator output")
    payload = stdout[start + len("===METRICS_JSON==="): end].strip()
    metrics = json.loads(payload)
    metrics.pop("_stdout", None)
    return metrics


def _run_sim(config_path: Path) -> Dict[str, float]:
    result = subprocess.run(
        ["python", str(SIM_PATH), "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SimulationError(
            f"Simulation failed for {config_path} (exit {result.returncode})\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return _parse_metrics(result.stdout)


def _run_with_rps(base_cfg: dict, rps: float, seeds: List[int], label: str) -> Dict[str, float]:
    if rps <= 1e-6:
        return {"throughput_jobs_s": 0.0, "avg_latency_ms": 0.0}

    aggregate: Dict[str, float] = {}
    template: Dict[str, float] = {}
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        cfg.setdefault("workload", {})["rate_rps"] = float(rps)

        cfg_path = OUTPUT_DIR / "configs" / f"{label}_rps{rps:.1f}_seed{seed}.yaml"
        _write_config(cfg, cfg_path)
        metrics = _run_sim(cfg_path)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                aggregate[key] = aggregate.get(key, 0.0) + float(value)
            elif key not in template:
                template[key] = value

    averaged: Dict[str, float] = dict(template)
    for key, total in aggregate.items():
        averaged[key] = total / len(seeds)
    return averaged


def main() -> None:
    distributed_raw = _load_raw_config(DISTRIBUTED_CONFIG)
    fused_raw = _load_raw_config(FUSED_CONFIG)
    total_drafts = _total_drafts(distributed_raw)
    if total_drafts <= 0:
        raise SimulationError("Distributed config must have at least one draft")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float]] = []
    for fraction in FUSED_FRACTIONS:
        fused_rps = TOTAL_RPS * fraction
        distributed_rps = TOTAL_RPS - fused_rps

        fused_metrics = _run_with_rps(fused_raw, fused_rps, SEEDS, label="fused")
        dist_metrics = _run_with_rps(distributed_raw, distributed_rps, SEEDS, label="distributed")

        fused_throughput = fused_metrics.get("throughput_jobs_s", 0.0)
        dist_throughput = dist_metrics.get("throughput_jobs_s", 0.0)
        total_throughput = fused_throughput + dist_throughput

        if total_throughput > 0:
            avg_latency = (
                fused_metrics.get("avg_latency_ms", 0.0) * fused_throughput
                + dist_metrics.get("avg_latency_ms", 0.0) * dist_throughput
            ) / total_throughput
        else:
            avg_latency = 0.0

        results.append(
            {
                "fraction_fused": fraction,
                "fused_drafts": fraction * total_drafts,
                "throughput_jobs_s": total_throughput,
                "avg_latency_ms": avg_latency,
                "throughput_fused": fused_throughput,
                "throughput_distributed": dist_throughput,
                "avg_latency_fused": fused_metrics.get("avg_latency_ms", 0.0),
                "avg_latency_distributed": dist_metrics.get("avg_latency_ms", 0.0),
            }
        )

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    fractions = [r["fraction_fused"] for r in results]
    fused_counts = [r["fused_drafts"] for r in results]
    throughput = [r["throughput_jobs_s"] for r in results]
    latency = [r["avg_latency_ms"] for r in results]

    fig, ax_throughput = plt.subplots(figsize=(7, 4.5))
    ax_latency = ax_throughput.twinx()

    ax_throughput.plot(fused_counts, throughput, marker="o", color="#1f77b4", label="Throughput (jobs/s)")
    ax_latency.plot(fused_counts, latency, marker="s", color="#d62728", label="Avg latency (ms)")

    ax_throughput.set_xlabel("Fused draft clients")
    ax_throughput.set_ylabel("Throughput (jobs/s)", color="#1f77b4")
    ax_latency.set_ylabel("Average latency (ms)", color="#d62728")
    ax_throughput.grid(True, alpha=0.3)

    handles = [
        ax_throughput.lines[0],
        ax_latency.lines[0],
    ]
    ax_throughput.legend(handles, [h.get_label() for h in handles], loc="upper right")
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "fused_vs_distributed.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Summary written to {metrics_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
