#!/usr/bin/env python3
"""Baseline draft roster sweep for speculative decoding simulator.

This utility clones ``configs/explorer/baseline.yaml`` for a roster of draft
counts (default: 50→1000), runs ``sim.py`` for each generated config, and
records latency metrics plus SLO status so they can be plotted later.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

# Default sweep: 50 to 1000 drafts (roughly logarithmic coverage)
DEFAULT_DRAFT_COUNTS: List[int] = [50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]
DEFAULT_SLO_TARGET_MS: float = 120.0  # Average latency per conversation

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ENTRYPOINT = REPO_ROOT / "simulator" / "sim.py"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
BASELINE_CONFIG = CONFIG_ROOT / "explorer" / "baseline.yaml"


class SweepError(RuntimeError):
    """Raised when the sweep cannot proceed."""


def _unique_sorted(values: Iterable[int]) -> List[int]:
    seen: Dict[int, None] = {}
    for val in values:
        if val <= 0:
            raise SweepError(f"Draft counts must be positive integers (got {val}).")
        seen[val] = None
    return sorted(seen.keys())


def load_baseline_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SweepError(f"Baseline config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "devices" not in data:
        raise SweepError("Baseline config is missing a devices section.")
    return data


def build_scaled_config(base_cfg: Dict[str, Any], draft_count: int) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)

    devices = cfg.get("devices", [])
    targets = [deepcopy(dev) for dev in devices if dev.get("role") == "target"]
    draft_templates = [dev for dev in devices if dev.get("role") == "draft"]
    if not targets:
        raise SweepError("Baseline config must define at least one target device.")
    if not draft_templates:
        raise SweepError("Baseline config must define at least one draft device.")

    # Cycle through available draft templates so heterogeneity is preserved
    new_drafts: List[Dict[str, Any]] = []
    for idx in range(draft_count):
        template = deepcopy(draft_templates[idx % len(draft_templates)])
        template_id = f"d{idx}"  # Simple incremental IDs (d0, d1, …)
        template["id"] = template_id
        new_drafts.append(template)

    cfg["devices"] = targets + new_drafts

    # Build connection templates per target from the baseline
    base_connections = cfg.get("connections", [])
    if not base_connections:
        raise SweepError("Baseline config must define connections between drafts and targets.")

    template_by_target: Dict[str, Dict[str, Any]] = {}
    for conn in base_connections:
        target_id = conn.get("target") or conn.get("target_id")
        if target_id and target_id not in template_by_target:
            template_by_target[target_id] = deepcopy(conn)

    missing_template_targets = [t["id"] for t in targets if t["id"] not in template_by_target]
    if missing_template_targets:
        raise SweepError(
            "Baseline connections do not cover all targets (missing templates for "
            + ", ".join(missing_template_targets)
            + ")"
        )

    connections: List[Dict[str, Any]] = []
    for draft in new_drafts:
        for target in targets:
            template = deepcopy(template_by_target[target["id"]])
            template["draft"] = draft["id"]
            template["target"] = target["id"]
            connections.append(template)
    cfg["connections"] = connections

    return cfg


def run_simulation(config_path: Path) -> Dict[str, Any]:
    if not SIM_ENTRYPOINT.exists():
        raise SweepError(f"Simulator entrypoint missing: {SIM_ENTRYPOINT}")

    cmd = ["python", str(SIM_ENTRYPOINT), "--config", str(config_path)]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    log_text = result.stdout
    if result.returncode != 0:
        raise SweepError(
            f"Simulator failed for {config_path.name} (exit {result.returncode})\n" + result.stderr
        )

    metrics = extract_metrics(log_text)
    metrics["stdout"] = log_text
    metrics["stderr"] = result.stderr
    return metrics


def extract_metrics(stdout: str) -> Dict[str, Any]:
    marker_start = "===METRICS_JSON==="
    marker_end = "===END_METRICS_JSON==="
    start_idx = stdout.find(marker_start)
    end_idx = stdout.find(marker_end)
    if start_idx == -1 or end_idx == -1:
        raise SweepError("Failed to locate METRICS_JSON block in simulator output.")

    json_payload = stdout[start_idx + len(marker_start):end_idx].strip()
    if not json_payload:
        raise SweepError("METRICS_JSON block was empty.")

    try:
        metrics = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        raise SweepError(f"Could not parse METRICS_JSON payload: {exc}") from exc

    if "avg_latency_ms" not in metrics:
        raise SweepError("avg_latency_ms missing from simulator metrics output.")

    return metrics


def write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["draft_count", "avg_latency_ms", "slo_target_ms", "slo_met", "config", "log"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline draft-count sweep experiments.")
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=BASELINE_CONFIG,
        help="Path to the baseline YAML configuration (default: configs/explorer/baseline.yaml)",
    )
    parser.add_argument(
        "--draft-counts",
        type=int,
        nargs="+",
        default=DEFAULT_DRAFT_COUNTS,
        help="List of draft counts to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--slo-target-ms",
        type=float,
        default=DEFAULT_SLO_TARGET_MS,
        help="Target average latency per conversation in milliseconds (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "baseline_draft_sweep",
        help="Directory where configs, logs, and metrics will be stored.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and manifest without executing simulations.",
    )

    args = parser.parse_args()

    try:
        draft_counts = _unique_sorted(args.draft_counts)
        slo_target = float(args.slo_target_ms)
        if slo_target <= 0:
            raise SweepError("SLO target must be positive.")

        base_cfg = load_baseline_config(args.baseline_config)

        output_dir = args.output_dir.resolve()
        configs_dir = output_dir / "configs"
        logs_dir = output_dir / "logs"
        metrics_dir = output_dir / "metrics"
        for folder in (configs_dir, logs_dir, metrics_dir):
            folder.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, Any] = {
            "experiment": "baseline_draft_sweep",
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "baseline_config": str(args.baseline_config.resolve()),
            "sim_entrypoint": str(SIM_ENTRYPOINT),
            "slo_target_ms": slo_target,
            "draft_counts": draft_counts,
            "dry_run": bool(args.dry_run),
            "runs": [],
        }

        summary_rows: List[Dict[str, Any]] = []

        for draft_count in draft_counts:
            scaled_cfg = build_scaled_config(base_cfg, draft_count)
            config_path = configs_dir / f"baseline_drafts_{draft_count:04d}.yaml"
            with config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(scaled_cfg, fh, sort_keys=False)

            run_entry: Dict[str, Any] = {
                "draft_count": draft_count,
                "config": str(config_path),
            }

            if args.dry_run:
                manifest["runs"].append(run_entry)
                continue

            metrics = run_simulation(config_path)

            log_path = logs_dir / f"baseline_drafts_{draft_count:04d}.log"
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write(metrics.pop("stdout", ""))
            if metrics.get("stderr"):
                stderr_path = logs_dir / f"baseline_drafts_{draft_count:04d}.err"
                with stderr_path.open("w", encoding="utf-8") as fh:
                    fh.write(metrics.pop("stderr", ""))
                run_entry["stderr"] = str(stderr_path)

            metrics_path = metrics_dir / f"baseline_drafts_{draft_count:04d}.json"
            with metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2, sort_keys=True)

            avg_latency = float(metrics.get("avg_latency_ms", 0.0))
            slo_met = avg_latency <= slo_target

            run_entry.update(
                {
                    "metrics": str(metrics_path),
                    "log": str(log_path),
                    "avg_latency_ms": avg_latency,
                    "slo_met": slo_met,
                }
            )

            summary_rows.append(
                {
                    "draft_count": draft_count,
                    "avg_latency_ms": avg_latency,
                    "slo_target_ms": slo_target,
                    "slo_met": slo_met,
                    "config": str(config_path),
                    "log": str(log_path),
                }
            )

            manifest["runs"].append(run_entry)
            print(
                f"drafts={draft_count:4d} avg_latency={avg_latency:7.2f}ms SLO({'met' if slo_met else 'miss'})",
                flush=True,
            )

        manifest_path_yaml = output_dir / "manifest.yaml"
        manifest_path_json = output_dir / "manifest.json"
        with manifest_path_yaml.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(manifest, fh, sort_keys=False)
        with manifest_path_json.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        if summary_rows:
            results_csv = output_dir / "results.csv"
            write_results_csv(results_csv, summary_rows)

            results_json = output_dir / "results.json"
            with results_json.open("w", encoding="utf-8") as fh:
                json.dump(summary_rows, fh, indent=2)

    except SweepError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")


if __name__ == "__main__":
    main()
