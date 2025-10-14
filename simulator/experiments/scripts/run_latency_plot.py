#!/usr/bin/env python3
"""Run specified simulator configs and plot average latency results."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_PATH = REPO_ROOT / "simulator" / "sim.py"

DEFAULT_RUNS = {
    "baseline": REPO_ROOT / "experiments" / "configs" / "explorer" / "baseline.yaml",
    "disagg_baseline": REPO_ROOT / "simulator" / "explorer" / "output" / "baseline_vs_disagg" / "disagg_baseline" / "configs" / "disagg_baseline.yaml",
}


class MetricsError(RuntimeError):
    pass


def run_sim(config: Path) -> Dict[str, float]:
    if not SIM_PATH.exists():
        raise MetricsError(f"Simulator entrypoint not found: {SIM_PATH}")
    if not config.exists():
        raise MetricsError(f"Config not found: {config}")

    cmd = ["python", str(SIM_PATH), "--config", str(config)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        raise MetricsError(f"Simulator failed for {config}:\n{result.stderr}")

    stdout = result.stdout
    marker_start = "===METRICS_JSON==="
    marker_end = "===END_METRICS_JSON==="
    start = stdout.find(marker_start)
    end = stdout.find(marker_end)
    if start == -1 or end == -1:
        raise MetricsError(f"METRICS_JSON block not found in output for {config}")

    payload = stdout[start + len(marker_start):end].strip()
    try:
        metrics = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MetricsError(f"Failed to parse metrics JSON for {config}: {exc}")

    metrics["_stdout"] = stdout
    return metrics


def plot_latency(results: Dict[str, Dict[str, float]], output: Path) -> Path:
    labels = list(results.keys())
    avg_latencies = [results[name]["avg_latency_ms"] for name in labels]
    rtt_latencies = [results[name].get("rtt_avg_ms", 0.0) for name in labels]

    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    bar_width = 0.35
    plt.bar([i - bar_width / 2 for i in x], avg_latencies, width=bar_width, label="Target avg latency (ms)")
    plt.bar([i + bar_width / 2 for i in x], rtt_latencies, width=bar_width, label="Conversation RTT avg (ms)")

    plt.ylabel("Latency (ms)")
    plt.xticks(list(x), labels, rotation=15)
    plt.title("Average Latency Comparison")
    plt.tight_layout()
    plt.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sim configs and plot average latency")
    parser.add_argument("configs", nargs="*", help="Pairs of name=config_path entries (e.g., baseline=path.yaml)")
    parser.add_argument("--output", default=REPO_ROOT / "experiments" / "results" / "latency_comparison.png", type=Path, help="Path to save the plot")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional path to dump raw metrics JSON")

    args = parser.parse_args()

    runs: Dict[str, Path]
    if args.configs:
        runs = {}
        for entry in args.configs:
            if "=" not in entry:
                parser.error(f"Invalid config entry '{entry}'. Expected name=path.yaml")
            name, path = entry.split("=", 1)
            runs[name] = Path(path).resolve()
    else:
        runs = DEFAULT_RUNS

    results: Dict[str, Dict[str, float]] = {}
    for name, cfg in runs.items():
        metrics = run_sim(cfg if cfg.is_absolute() else (REPO_ROOT / cfg))
        results[name] = metrics
        print(f"{name}: avg_latency={metrics['avg_latency_ms']:.2f}ms rtt_avg={metrics.get('rtt_avg_ms', 0.0):.2f}ms")

    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_json.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    plot_path = plot_latency(results, args.output if args.output.is_absolute() else args.output.resolve())
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
