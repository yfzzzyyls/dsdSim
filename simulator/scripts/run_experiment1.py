#!/usr/bin/env python3
"""Experiment 1: router / scheduler control matrix for the new baseline."""

from __future__ import annotations

import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "explorer" / "baseline.yaml"
OUTPUT_ROOT = REPO_ROOT / "scripts_output" / "experiment1"
CONFIG_DIR = OUTPUT_ROOT / "configs"
PLOT_DIR = OUTPUT_ROOT / "plots"
METRICS_PATH = OUTPUT_ROOT / "experiment1_metrics.json"

SLO_LIMITS = {
    "p95_latency_ms": 220.0,
    "p99_latency_ms": 280.0,
    "throughput_jobs_s": 90.0,
}

VariantSpec = Dict[str, object]

VARIANTS: List[VariantSpec] = [
    {
        "name": "random_fifo",
        "description": "Baseline random routing with FIFO batching.",
        "overrides": {},
    },
    {
        "name": "round_robin_priority",
        "description": "Round-robin routing with priority queues.",
        "overrides": {
            "router": "round_robin",
            "router_params": {},
            "scheduler": {
                "prefill": {"queue_policy": "priority", "max_wait_ms": 0.4},
                "decode": {"queue_policy": "priority", "max_wait_ms": 0.4},
            },
        },
    },
    {
        "name": "jsq_priority",
        "description": "Join-the-shortest-queue routing with priority queues.",
        "overrides": {
            "router": "jsq",
            "router_params": {},
            "scheduler": {
                "prefill": {"queue_policy": "priority", "max_wait_ms": 0.4},
                "decode": {"queue_policy": "priority", "max_wait_ms": 0.4},
            },
        },
    },
]


def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if "auto_topology" not in data:
        raise ValueError("Experiment 1 expects baseline to use auto_topology.")
    return data


def _deep_update(base: Dict[str, object], overrides: Dict[str, object]) -> Dict[str, object]:
    result = deepcopy(base)
    stack: List[Tuple[Dict[str, object], Dict[str, object]]] = [(result, overrides)]
    while stack:
        target, patch = stack.pop()
        for key, value in patch.items():
            if isinstance(value, dict):
                sub = target.setdefault(key, {})
                if isinstance(sub, dict):
                    stack.append((sub, value))
                else:
                    target[key] = deepcopy(value)
            else:
                target[key] = deepcopy(value)
    return result


def _write_config(name: str, data: Dict[str, object]) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _run_sim(config_path: Path) -> Dict[str, float]:
    cmd = ["python", str(REPO_ROOT / "sim.py"), "--config", str(config_path)]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Simulation failed for {config_path} (exit {result.returncode})\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    stdout = result.stdout
    start = stdout.find("===METRICS_JSON===")
    end = stdout.find("===END_METRICS_JSON===")
    if start == -1 or end == -1:
        raise RuntimeError(f"Metrics block not found for {config_path}")
    payload = stdout[start + len("===METRICS_JSON===") : end].strip()
    metrics = json.loads(payload)
    metrics["_stdout"] = stdout
    return metrics


def _evaluate_slo(metrics: Dict[str, float]) -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    p95 = metrics.get("p95_latency_ms")
    p99 = metrics.get("p99_latency_ms")
    thr = metrics.get("throughput_jobs_s")
    status["p95_latency_ms"] = p95 is not None and p95 <= SLO_LIMITS["p95_latency_ms"]
    status["p99_latency_ms"] = p99 is not None and p99 <= SLO_LIMITS["p99_latency_ms"]
    status["throughput_jobs_s"] = thr is not None and thr >= SLO_LIMITS["throughput_jobs_s"]
    status["overall"] = all(status.values())
    return status


def _plot_metrics(results: Dict[str, Dict[str, float]]) -> None:
    if not results:
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(results.keys())
    p95_values = [results[name].get("p95_latency_ms", math.nan) for name in labels]
    throughput_values = [results[name].get("throughput_jobs_s", math.nan) for name in labels]

    def _bar_plot(values: List[float], title: str, ylabel: str, slo_line: float | None, slo_label: str, output: Path):
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 4.2))
        ax.bar(labels, values, color="#1f77b4")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if slo_line is not None:
            ax.axhline(slo_line, color="#d62728", linestyle="--", linewidth=1.5, label=slo_label)
            ax.legend()
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)

    _bar_plot(
        p95_values,
        "Experiment 1: P95 Latency by Policy",
        "Latency (ms)",
        SLO_LIMITS["p95_latency_ms"],
        "P95 SLO",
        PLOT_DIR / "p95_latency.png",
    )
    _bar_plot(
        throughput_values,
        "Experiment 1: Throughput by Policy",
        "Jobs / second",
        SLO_LIMITS["throughput_jobs_s"],
        "Throughput SLO",
        PLOT_DIR / "throughput.png",
    )


def run() -> None:
    base_cfg = _load_config(BASE_CONFIG)
    summary: Dict[str, Dict[str, object]] = {}

    for variant in VARIANTS:
        name = str(variant["name"])
        overrides = variant.get("overrides", {})
        desc = variant.get("description", "")
        cfg = _deep_update(base_cfg, overrides) if overrides else deepcopy(base_cfg)
        cfg.setdefault("metadata", {})["variant_description"] = desc
        config_path = _write_config(name, cfg)
        metrics = _run_sim(config_path)
        slo = _evaluate_slo(metrics)
        summary[name] = {
            "description": desc,
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "metrics": metrics,
            "slo": slo,
        }
        print(f"[exp1] {name}: overall_slo={'PASS' if slo['overall'] else 'FAIL'}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    condensed = {name: data["metrics"] for name, data in summary.items()}
    _plot_metrics(condensed)
    print(f"[exp1] metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    run()
