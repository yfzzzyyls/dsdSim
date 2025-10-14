#!/usr/bin/env python3
"""Sweep system load for fixed topology and compare routing algorithms."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_PATH = REPO_ROOT / "simulator" / "sim.py"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "explorer" / "router_sweep.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "router_load_sweep"
ROUTERS = ["random", "round_robin", "jsq"]
LOAD_POINTS = [float(x) for x in range(60, 242, 2)]
SIM_TIME_MS = 300_000  # 5 minutes for proper saturation measurements
DEFAULT_SEEDS = [123]


class SweepError(RuntimeError):
    pass


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise SweepError(f"Baseline config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if "auto_topology" not in data:
        raise SweepError("Load sweep expects config with auto_topology.")
    return data


def _write_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _run_sim(config_path: Path) -> Dict[str, float]:
    cmd = ["python", str(SIM_PATH), "--config", str(config_path)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SweepError(f"Simulation failed ({config_path})\n{result.stderr}")
    stdout = result.stdout
    start = stdout.find("===METRICS_JSON===")
    end = stdout.find("===END_METRICS_JSON===")
    if start == -1 or end == -1:
        raise SweepError(f"METRICS_JSON block not found for {config_path}")
    payload = stdout[start + len("===METRICS_JSON==="):end].strip()
    metrics = json.loads(payload)
    return metrics


def sweep_loads(
    base_cfg: dict,
    load_points: List[float],
    routers: List[str],
    output_dir: Path,
    seeds: List[int],
) -> Dict[str, Dict[float, Dict[str, float]]]:
    results: Dict[str, Dict[float, Dict[str, float]]] = {router: {} for router in routers}

    for router in routers:
        for load in load_points:
            aggregate: Dict[str, float] = {}
            samples = 0

            for seed in seeds:
                cfg = deepcopy(base_cfg)
                cfg["seed"] = seed
                cfg.setdefault("router_params", {})
                cfg["router"] = router
                cfg["router_params"] = {}
                cfg["global_router"] = "disabled"
                cfg["global_router_params"] = {}
                clusters = cfg.get("auto_topology", {}).get("clusters", [])
                for cluster_spec in clusters:
                    cluster_spec["router"] = router
                    if "router_params" in cluster_spec:
                        cluster_spec["router_params"] = {}
                cfg.setdefault("workload", {})["rate_rps"] = float(load)
                cfg["sim_time_ms"] = SIM_TIME_MS
                cfg["burn_in_ms"] = 0
                cfg["verbose"] = False

                cfg_path = (output_dir / "configs" / f"{router}_load{load}_seed{seed}.yaml").resolve()
                _write_config(cfg, cfg_path)

                metrics = _run_sim(cfg_path)
                samples += 1
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        aggregate[key] = aggregate.get(key, 0.0) + float(value)

            averaged = {k: v / samples for k, v in aggregate.items()}
            results[router][load] = averaged
            p95_value = averaged.get("p95_conversation_ms", averaged.get("p95_latency_ms", 0.0))
            conv_throughput = averaged.get("conversation_throughput_rps", 0.0)
            print(
                f"router={router:>12} load={load:5.1f} jobs_thr={averaged.get('throughput_jobs_s',0):6.2f} "
                f"conv_thr={conv_throughput:6.2f} p95_conv={p95_value:6.2f}"
            )

    return results


def plot_results(results: Dict[str, Dict[float, Dict[str, float]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loads = sorted(next(iter(results.values())).keys())

    # Plot 1: Throughput & Goodput
    plt.figure(figsize=(7, 4))
    for router, data in results.items():
        throughput = [data[load].get("throughput_jobs_s", 0.0) for load in loads]
        throughput = [data[load].get("throughput_jobs_s", 0.0) for load in loads]
        plt.plot(loads, throughput, marker="o", label=f"{router} (job throughput)", linestyle="-")

        conv_throughput = [data[load].get("conversation_throughput_rps", 0.0) for load in loads]
        plt.plot(loads, conv_throughput, marker="^", label=f"{router} (conversation throughput)", linestyle=":", alpha=0.9)
    plt.xlabel("Offered load (requests/s)")
    plt.ylabel("Requests/s")
    plt.title("Throughput & Goodput vs Load")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_goodput_vs_load.png", dpi=150)
    plt.close()

    # Plot 2: TTFT P95
    plt.figure(figsize=(7, 4))
    for router, data in results.items():
        y = [data[load].get("ttft_p95_ms", 0.0) for load in loads]
        plt.plot(loads, y, marker="o", label=router)
    plt.xlabel("Offered load (requests/s)")
    plt.ylabel("TTFT P95 (ms)")
    plt.title("Time To First Token (P95) vs Load")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ttft_p95_vs_load.png", dpi=150)
    plt.close()

    # Plot 3: TPOT P95
    plt.figure(figsize=(7, 4))
    for router, data in results.items():
        y = [data[load].get("tpot_p95_ms", 0.0) for load in loads]
        plt.plot(loads, y, marker="o", label=router)
    plt.xlabel("Offered load (requests/s)")
    plt.ylabel("TPOT P95 (ms)")
    plt.title("Time Per Output Token (P95) vs Load")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tpot_p95_vs_load.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep load for fixed topology across routers")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_CONFIG, help="Baseline config path")
    parser.add_argument("--loads", type=float, nargs="*", help="Explicit load points (RPS)")
    parser.add_argument("--start", type=float, help="Start load (inclusive) if --loads omitted")
    parser.add_argument("--end", type=float, help="End load (inclusive) if --loads omitted")
    parser.add_argument("--step", type=float, help="Step size when generating loads")
    parser.add_argument("--routers", nargs="*", default=ROUTERS, help="Routers to compare")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to store outputs")
    parser.add_argument("--metrics-json", type=Path, help="Optional path to dump metrics JSON")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS, help="Random seeds to average over")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation for faster iteration")

    args = parser.parse_args()

    if args.loads:
        load_points = sorted(set(args.loads))
    else:
        start = args.start if args.start is not None else LOAD_POINTS[0]
        end = args.end if args.end is not None else LOAD_POINTS[-1]
        step = args.step if args.step is not None else (LOAD_POINTS[1] - LOAD_POINTS[0])
        if step <= 0:
            parser.error("--step must be positive")
        load_points = []
        cur = start
        while cur <= end + 1e-9:
            load_points.append(round(cur, 3))
            cur += step

    base_cfg = _load_config(args.baseline if args.baseline.is_absolute() else Path.cwd() / args.baseline)
    routers = args.routers or ROUTERS
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    seeds = sorted(set(seeds))

    results = sweep_loads(base_cfg, load_points, routers, output_dir, seeds)

    if not args.no_plots:
        plot_results(results, output_dir / "plots")

    if args.metrics_json:
        metrics_path = args.metrics_json if args.metrics_json.is_absolute() else Path.cwd() / args.metrics_json
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
