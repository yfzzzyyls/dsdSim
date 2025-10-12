#!/usr/bin/env python3
"""Sweep draft client counts to observe saturation in blocking mode."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = REPO_ROOT / "sim.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "explorer" / "router_sweep.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts_output" / "draft_count_sweep"
DEFAULT_SEEDS = [123]


class SweepError(RuntimeError):
    pass


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise SweepError(f"Baseline config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if "auto_topology" not in data:
        raise SweepError("draft_count_sweep expects config with auto_topology.")
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


def _rebalance_label_counts(count_by_label: Dict[str, int], desired_total: int) -> Dict[str, int]:
    """Scale count_by_label to sum to desired_total while preserving ratios."""
    if not count_by_label:
        return {}
    labels = list(count_by_label.keys())
    if len(labels) == 1:
        return {labels[0]: desired_total}

    original_total = sum(max(1, int(v)) for v in count_by_label.values()) or 1
    provisional: Dict[str, int] = {}
    accumulated = 0
    for idx, label in enumerate(labels):
        if idx == len(labels) - 1:
            provisional[label] = desired_total - accumulated
        else:
            scaled = int(round(desired_total * count_by_label[label] / original_total))
            scaled = max(1, scaled)
            provisional[label] = scaled
            accumulated += scaled

    # Correct any rounding drift
    final_total = sum(provisional.values())
    if final_total != desired_total:
        delta = desired_total - final_total
        first_label = labels[0]
        provisional[first_label] = max(1, provisional.get(first_label, 1) + delta)

    return provisional


def sweep_draft_counts(
    base_cfg: dict,
    counts: Iterable[int],
    routers: Iterable[str],
    seeds: Iterable[int],
    output_dir: Path,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    results: Dict[str, Dict[int, Dict[str, float]]] = {router: {} for router in routers}

    for router in routers:
        for draft_count in counts:
            aggregate: Dict[str, float] = {}
            samples = 0

            for seed in seeds:
                cfg = deepcopy(base_cfg)
                cfg["seed"] = int(seed)
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

                    drafts = cluster_spec.get("drafts", {})
                    if drafts:
                        drafts["count"] = int(draft_count)
                        labels = drafts.get("count_by_label", {})
                        new_counts = _rebalance_label_counts(labels, int(draft_count))
                        if new_counts:
                            drafts["count_by_label"] = new_counts

                cfg.setdefault("workload", {})["rate_rps"] = 0.0  # ensure blocking-driven load
                cfg["sim_time_ms"] = cfg.get("sim_time_ms", 300_000)

                cfg_path = (
                    output_dir / "configs" / f"{router}_drafts{draft_count}_seed{seed}.yaml"
                ).resolve()
                _write_config(cfg, cfg_path)

                metrics = _run_sim(cfg_path)
                samples += 1
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        aggregate[key] = aggregate.get(key, 0.0) + float(value)

            averaged = {k: v / samples for k, v in aggregate.items()} if samples else {}
            results[router][int(draft_count)] = averaged
            ttft_p95 = averaged.get("ttft_p95_ms", averaged.get("ttft_avg_ms", 0.0))
            conv_thr = averaged.get("conversation_throughput_rps", 0.0)
            print(
                f"router={router:>10} drafts={draft_count:4d} conv_thr={conv_thr:6.2f} "
                f"ttft_p95={ttft_p95:7.2f}"
            )

    return results


def parse_counts(args) -> List[int]:
    if args.counts:
        return sorted({int(c) for c in args.counts if int(c) > 0})
    if args.start is None or args.end is None or args.step is None:
        raise SystemExit("Either --counts or all of --start/--end/--step must be provided.")
    start, end, step = args.start, args.end, args.step
    if step <= 0:
        raise SystemExit("--step must be positive")
    counts = []
    cur = start
    while cur <= end + 1e-9:
        counts.append(int(round(cur)))
        cur += step
    return sorted({c for c in counts if c > 0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep draft client counts for blocking-mode saturation.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_CONFIG, help="Baseline config path.")
    parser.add_argument("--counts", type=int, nargs="*", help="Explicit draft counts to evaluate.")
    parser.add_argument("--start", type=int, help="Start draft count (inclusive) if --counts omitted.")
    parser.add_argument("--end", type=int, help="End draft count (inclusive) if --counts omitted.")
    parser.add_argument("--step", type=int, help="Step size when generating draft counts.")
    parser.add_argument("--routers", nargs="*", default=["jsq"], help="Routers to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for configs & logs.")
    parser.add_argument("--metrics-json", type=Path, help="Optional path to dump metrics JSON.")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS, help="Random seeds to average.")

    args = parser.parse_args()

    counts = parse_counts(args)
    if not counts:
        raise SystemExit("No draft counts specified.")

    routers = args.routers or ["jsq"]
    routers = sorted(set(routers))
    seeds = args.seeds or DEFAULT_SEEDS

    base_cfg = _load_config(args.baseline if args.baseline.is_absolute() else Path.cwd() / args.baseline)
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir

    results = sweep_draft_counts(base_cfg, counts, routers, seeds, output_dir)

    if args.metrics_json:
        metrics_path = args.metrics_json if args.metrics_json.is_absolute() else Path.cwd() / args.metrics_json
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
