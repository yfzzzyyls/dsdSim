#!/usr/bin/env python3
"""Sweep draft counts across router algorithms and plot latency/throughput."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = REPO_ROOT / "sim.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "explorer" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts_output" / "router_draft_sweep"
# Default sweep: 50 → 200 in increments of 5
DEFAULT_DRAFT_COUNTS = list(range(50, 201, 5))
ROUTERS = ["random", "round_robin", "semi_clairvoyant"]

class SweepError(RuntimeError):
    pass


def _load_raw_config(path: Path) -> dict:
    if not path.exists():
        raise SweepError(f"Baseline config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not data:
        raise SweepError("Baseline configuration is empty")
    if "auto_topology" not in data:
        raise SweepError("Draft sweep expects baseline with auto_topology")
    return data


def _expand(cfg: dict) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from sim import _expand_auto_topology  # type: ignore

    return _expand_auto_topology(dict(cfg))


def _base_cluster_counts(raw_cfg: dict) -> Dict[str, int]:
    clusters = raw_cfg.get("auto_topology", {}).get("clusters", [])
    return {cluster.get("name", f"cluster_{idx}"): int(cluster.get("drafts", {}).get("count", 0)) for idx, cluster in enumerate(clusters)}


def _scaled_cluster_counts(base_counts: Dict[str, int], total: int) -> Dict[str, int]:
    names = list(base_counts.keys())
    base_total = sum(base_counts.values())
    if base_total <= 0:
        raise SweepError("Baseline must have at least one draft")
    if total <= 0:
        raise SweepError("Draft count must be positive")
    scaled = {}
    remainders: List[Tuple[float, str]] = []
    remaining = total
    for name in names:
        base = base_counts[name]
        frac = base / base_total
        exact = frac * total
        count = max(1, int(exact))
        scaled[name] = count
        remaining -= count
        remainders.append((exact - count, name))
    if remaining != 0:
        remainders.sort(reverse=True)
        idx = 0
        while remaining > 0 and idx < len(remainders):
            _, name = remainders[idx]
            scaled[name] += 1
            remaining -= 1
            idx += 1
        while remaining < 0 and idx < len(remainders):
            _, name = remainders[idx]
            if scaled[name] > 1:
                scaled[name] -= 1
                remaining += 1
            idx += 1
        if remaining != 0:
            name = names[-1]
            scaled[name] += remaining
    return scaled


def _write_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _run_sim(config_path: Path) -> Dict[str, float]:
    cmd = ["python", str(SIM_PATH), "--config", str(config_path)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SweepError(f"Simulation failed for {config_path} (exit {result.returncode})\n{result.stderr}")
    stdout = result.stdout
    start = stdout.find("===METRICS_JSON===")
    end = stdout.find("===END_METRICS_JSON===")
    if start == -1 or end == -1:
        raise SweepError(f"METRICS_JSON block not found for {config_path}")
    payload = stdout[start + len("===METRICS_JSON==="):end].strip()
    metrics = json.loads(payload)
    metrics["_stdout"] = stdout
    return metrics


def sweep_runs(
    raw_cfg: dict,
    draft_counts: List[int],
    routers: List[str],
    output_dir: Path,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    base_counts = _base_cluster_counts(raw_cfg)
    base_total = sum(base_counts.values())
    base_rate = float(raw_cfg.get("workload", {}).get("rate_rps", 0.0))
    rate_per_draft = base_rate / base_total if base_total else 0.0

    results: Dict[str, Dict[int, Dict[str, float]]] = {router: {} for router in routers}

    for draft_count in draft_counts:
        scaled_counts = _scaled_cluster_counts(base_counts, draft_count)

        for router in routers:
            cfg = deepcopy(raw_cfg)
            cfg.setdefault("router_params", {})
            cfg["router"] = router
            cfg["router_params"] = {}
            cfg["global_router"] = "disabled"
            cfg["global_router_params"] = {}
            cfg.setdefault("workload", {})["rate_rps"] = round(rate_per_draft * draft_count, 3)
            cfg["burn_in_ms"] = 0
            cfg["verbose"] = False

            clusters = cfg.get("auto_topology", {}).get("clusters", [])
            for cluster in clusters:
                name = cluster.get("name")
                if name in scaled_counts:
                    cluster.setdefault("drafts", {})["count"] = scaled_counts[name]
                cluster["router"] = router
                cluster["router_params"] = {}

            if router == "semi_clairvoyant":
                sched = cfg.setdefault("scheduler", {})
                prefill = sched.setdefault("prefill", {})
                decode = sched.setdefault("decode", {})
                for section in (prefill, decode):
                    section["queue_policy"] = "priority"
                    section["mode"] = "continuous"
                    section["chunk_tokens"] = 24
                    section["chunk_sequential"] = True
                    section["max_wait_ms"] = 0.4
            else:
                sched = cfg.setdefault("scheduler", {})
                prefill = sched.setdefault("prefill", {})
                decode = sched.setdefault("decode", {})
                prefill.setdefault("queue_policy", "priority")
                decode.setdefault("queue_policy", "priority")
                prefill.setdefault("max_wait_ms", 0.4)
                decode.setdefault("max_wait_ms", 0.4)
                for section in (prefill, decode):
                    section.pop("mode", None)
                    section.pop("chunk_tokens", None)
                    section.pop("chunk_sequential", None)

            cfg_path = (output_dir / "configs" / f"{router}_{draft_count}.yaml").resolve()
            _write_config(cfg, cfg_path)

            metrics = _run_sim(cfg_path)
            results[router][draft_count] = metrics
            print(
                f"router={router:>12} drafts={draft_count:4d} avg_latency={metrics['avg_latency_ms']:.2f}ms throughput={metrics['throughput_jobs_s']:.2f}"
            )
    return results


def plot_results(results: Dict[str, Dict[int, Dict[str, float]]], draft_counts: List[int], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latency_fig = output_dir / "latency_vs_drafts.png"
    throughput_fig = output_dir / "throughput_vs_drafts.png"

    plt.figure(figsize=(7, 4))
    for router, data in results.items():
        vals = [data[count]["avg_latency_ms"] for count in draft_counts]
        plt.plot(draft_counts, vals, marker="o", label=router)
    plt.xlabel("Draft count")
    plt.ylabel("Average latency (ms)")
    plt.title("Latency vs. draft count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(latency_fig, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    for router, data in results.items():
        vals = [data[count]["throughput_jobs_s"] for count in draft_counts]
        plt.plot(draft_counts, vals, marker="o", label=router)
    plt.xlabel("Draft count")
    plt.ylabel("Throughput (jobs/s)")
    plt.title("Throughput vs. draft count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(throughput_fig, dpi=150)
    plt.close()

    return latency_fig, throughput_fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare routing algorithms across draft counts")
    parser.add_argument("--baseline", default=DEFAULT_CONFIG, type=Path, help="Baseline config to use as template")
    parser.add_argument("--draft-counts", type=int, nargs="*", help="Explicit draft counts to evaluate")
    parser.add_argument("--start", type=int, help="Start draft count (inclusive) if --draft-counts omitted")
    parser.add_argument("--end", type=int, help="End draft count (inclusive) if --draft-counts omitted")
    parser.add_argument("--step", type=int, help="Step when generating draft counts (default 5)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write configs/plots")
    parser.add_argument("--routers", nargs="*", default=ROUTERS, help="Routers to compare")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional path to dump metrics JSON")

    args = parser.parse_args()

    baseline_path = args.baseline if args.baseline.is_absolute() else Path.cwd() / args.baseline
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    routers = args.routers or ROUTERS

    if args.draft_counts:
        draft_counts = sorted(set(args.draft_counts))
    else:
        start = args.start if args.start is not None else 50
        end = args.end if args.end is not None else 200
        step = args.step if args.step is not None else 5
        if step <= 0:
            parser.error("--step must be positive")
        if end < start:
            parser.error("--end must be >= --start")
        draft_counts = list(range(start, end + 1, step))

    raw_cfg = _load_raw_config(baseline_path)

    results = sweep_runs(raw_cfg, draft_counts, routers, output_dir)
    latency_fig, throughput_fig = plot_results(results, draft_counts, output_dir)

    if args.metrics_json:
        metrics_path = args.metrics_json if args.metrics_json.is_absolute() else Path.cwd() / args.metrics_json
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    print(f"Latency plot: {latency_fig}")
    print(f"Throughput plot: {throughput_fig}")


if __name__ == "__main__":
    main()
