#!/usr/bin/env python3
"""Sweep draft counts for several gamma control policies."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import time
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_PATH = REPO_ROOT / "src" / "sim.py"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "explorer" / "baseline_trace_gsm8k.yaml"
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "gamma_policy_draft_sweep"
DEFAULT_COUNTS = [5, 15, 25, 35, 45, 55, 65, 75]
DEFAULT_SEEDS = [123]
VIDUR_REPO_PATH = REPO_ROOT / "thirdparty" / "Sai_speculative_vidur"

# Ensure in-process mode can import repository modules and vendored VIDUR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
src_path = REPO_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(VIDUR_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(VIDUR_REPO_PATH))

import yaml
from src import sim

POLICY_PRESETS: Dict[str, Dict[str, Optional[dict]]] = {
    "gamma4_static": {
        "gamma": 4,
        "gamma_policy": None,
    },
    "acceptance_backoff": {
        "gamma": 4,
        "gamma_policy": {
            "type": "acceptance_backoff",
            "min_gamma": 2,
            "max_gamma": 8,
            "low_acceptance": 0.25,
            "high_acceptance": 0.6,
        },
    },
    "specpp": {
        "gamma": 4,
        "gamma_policy": {
            "type": "specpp",
            "min_gamma": 2,
            "max_gamma": 8,
            "stop_threshold": 0.7,
            "fallback_gamma": 4,
        },
    },
}

DISPLAY_NAMES = {
    "gamma4_static": "Static γ=4",
    "acceptance_backoff": "Simple",
    "specpp": "Spec++",
}

LINE_STYLES = {
    "gamma4_static": {"color": "#1f77b4", "marker": "o"},
    "acceptance_backoff": {"color": "#ff7f0e", "marker": "s"},
    "specpp": {"color": "#2ca02c", "marker": "^"},
}


class SweepError(RuntimeError):
    pass


def _load_config(path: Path) -> Dict:
    if not path.exists():
        raise SweepError(f"Baseline config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _ensure_latency_metadata(cfg: Dict) -> None:
    for cluster in cfg.get("auto_topology", {}).get("clusters", []):
        target_spec = cluster.get("targets", {})
        tiers = target_spec.get("tiers", [])
        for tier in tiers:
            metadata = dict(tier.get("metadata", {}))
            metadata.setdefault("prefill_latency_per_token", 30.0)
            metadata.setdefault("decode_latency_per_token", 9.0)
            tier["metadata"] = metadata


def _rebalance_count_by_label(draft_spec: Dict, new_total: int) -> None:
    counts = draft_spec.get("count_by_label") or {}
    if not counts:
        return
    labels = list(counts.keys())
    if not labels:
        return
    original_total = sum(max(1, int(v)) for v in counts.values()) or 1
    scaled: Dict[str, int] = {}
    remaining = new_total
    for idx, label in enumerate(labels):
        if idx == len(labels) - 1:
            val = remaining
        else:
            val = int(round(new_total * int(counts[label]) / original_total))
            val = max(1, val)
            remaining -= val
        scaled[label] = max(1, val)
    drift = new_total - sum(scaled.values())
    if drift != 0:
        first = labels[0]
        scaled[first] = max(1, scaled[first] + drift)
    draft_spec["count_by_label"] = {k: int(v) for k, v in scaled.items()}


def _resolve_trace_path(cfg: Dict, base_dir: Path) -> None:
    trace_path = cfg.get("trace_path")
    if not trace_path:
        return
    trace_obj = Path(trace_path)
    if not trace_obj.is_absolute():
        for root in (base_dir, REPO_ROOT):
            candidate = (root / trace_obj).resolve()
            if candidate.exists():
                trace_obj = candidate
                break
    cfg["trace_path"] = str(trace_obj)


def _build_config(base: Dict, *, policy_name: str, draft_count: int, base_dir: Path) -> Dict:
    preset = POLICY_PRESETS.get(policy_name)
    if preset is None:
        raise SweepError(f"Unknown policy preset '{policy_name}'")

    cfg = deepcopy(base)
    cfg["sim_time_ms"] = 2000
    cfg["verbose"] = False
    cfg["debug"] = False
    cfg["gamma"] = int(preset.get("gamma", 4) or 4)

    spec = cfg.setdefault("speculation", {})
    spec["framework"] = "vanilla"
    spec["execution_mode"] = "distributed"

    acceptance_cfg = spec.setdefault("acceptance", {})
    if acceptance_cfg.get("disable_model"):
        spec.pop("acceptance_model", None)
        acceptance_cfg.pop("model", None)
        acceptance_cfg.pop("file", None)
    else:
        spec.setdefault("acceptance_model", str(REPO_ROOT / "src" / "acceptance" / "llama2_7b_vs_70b.joblib"))

    gamma_policy = preset.get("gamma_policy")
    if gamma_policy:
        spec["gamma_policy"] = dict(gamma_policy)
    else:
        spec.pop("gamma_policy", None)

    _resolve_trace_path(cfg, base_dir)
    _ensure_latency_metadata(cfg)

    clusters = cfg.get("auto_topology", {}).get("clusters", [])
    for cluster in clusters:
        drafts = cluster.get("drafts", {})
        drafts["count"] = int(draft_count)
        _rebalance_count_by_label(drafts, int(draft_count))

    return cfg


def _write_config(cfg: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _run_sim_subprocess(config_path: Path) -> Dict[str, float]:
    cmd = [sys.executable, str(SIM_PATH), "--config", str(config_path)]
    env = dict(os.environ)
    extra = [str(VIDUR_REPO_PATH)]
    existing = env.get("PYTHONPATH")
    if existing:
        env["PYTHONPATH"] = os.pathsep.join(extra + [existing])
    else:
        env["PYTHONPATH"] = os.pathsep.join(extra)
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise SweepError(
            (
                f"Simulation failed for {config_path} (exit {result.returncode})\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        )
    stdout = result.stdout
    start = stdout.find("===METRICS_JSON===")
    end = stdout.find("===END_METRICS_JSON===")
    if start == -1 or end == -1:
        raise SweepError("METRICS_JSON block not found in simulator output")
    payload = stdout[start + len("===METRICS_JSON==="): end].strip()
    metrics = json.loads(payload)
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def _run_sim_inprocess(cfg_dict: Dict[str, Any]) -> Dict[str, float]:
    cfg_obj = sim.load_config_from_mapping(cfg_dict)
    cfg_obj.verbose = False
    cfg_obj.debug = False
    buffer = io.StringIO()
    err_buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(err_buffer):
        metrics = sim.simulate_config_obj(cfg_obj, emit_output=False)
    err_text = err_buffer.getvalue()
    if err_text:
        print("SIM ERROR OUTPUT:", err_text.strip(), file=sys.stderr)
    return metrics


def _run_policy_count(
    policy_name: str,
    draft_count: int,
    base_cfg: Dict,
    seeds: Iterable[int],
    output_dir: Path,
    base_dir: Path,
    run_mode: str,
) -> Tuple[str, int, Dict[str, float]]:
    aggregates: Dict[str, float] = {}
    samples = 0
    for seed in seeds:
        cfg = _build_config(base_cfg, policy_name=policy_name, draft_count=draft_count, base_dir=base_dir)
        cfg["seed"] = int(seed)
        cfg_path = output_dir / "configs" / f"{policy_name}_drafts{draft_count}_seed{seed}.yaml"
        _write_config(cfg, cfg_path)
        if run_mode == "sweep":
            metrics = _run_sim_inprocess(cfg)
        else:
            metrics = _run_sim_subprocess(cfg_path)
        samples += 1
        for key, value in metrics.items():
            aggregates[key] = aggregates.get(key, 0.0) + value

    averaged = {k: v / samples for k, v in aggregates.items()} if samples else {}
    thr = averaged.get("throughput_jobs_s", 0.0)
    conv_thr = averaged.get("conversation_throughput_rps", 0.0)
    ttft = averaged.get("ttft_avg_ms", 0.0)
    tpot = averaged.get("tpot_avg_ms", 0.0)
    print(
        f"policy={policy_name:18s} drafts={draft_count:4d} thr={thr:7.2f} conv_thr={conv_thr:7.2f} "
        f"ttft={ttft:7.2f} tpot={tpot:7.2f}"
    )
    _log_latency_breakdown(policy_name, draft_count, averaged, output_dir)
    return policy_name, draft_count, averaged


def _log_latency_breakdown(policy_name: str, draft_count: int, metrics: Dict[str, float], output_dir: Path) -> None:
    """Persist latency breakdown gleaned from simulator metrics without console spam."""

    def _collect(prefix: str) -> Dict[str, float]:
        entries = {
            key[len(prefix):]: value
            for key, value in metrics.items()
            if key.startswith(prefix)
        }
        return dict(sorted(entries.items())) if entries else {}

    ttft_parts = _collect("ttft_breakdown_")
    decode_parts = _collect("decode_breakdown_")

    if not ttft_parts and not decode_parts:
        return

    payload = {
        "policy": policy_name,
        "draft_count": draft_count,
        "ttft_breakdown_ms": ttft_parts,
        "decode_breakdown_ms": decode_parts,
    }
    dump_dir = output_dir / "latency_breakdowns"
    dump_dir.mkdir(parents=True, exist_ok=True)
    out_path = dump_dir / f"{policy_name}_drafts{draft_count}.json"
    out_path.write_text(json.dumps(payload, indent=2))


def _prepare_seed_configs(
    policy_name: str,
    draft_count: int,
    base_cfg: Dict,
    seeds: Iterable[int],
    output_dir: Path,
    base_dir: Path,
) -> list[Dict[str, Any]]:
    prepared: list[Dict[str, Any]] = []
    for seed in seeds:
        cfg = _build_config(base_cfg, policy_name=policy_name, draft_count=draft_count, base_dir=base_dir)
        cfg["seed"] = int(seed)
        cfg_path = output_dir / "configs" / f"{policy_name}_drafts{draft_count}_seed{seed}.yaml"
        _write_config(cfg, cfg_path)
        prepared.append({"cfg": cfg, "path": cfg_path, "seed": int(seed)})
    return prepared


def _run_policy_count_sweep(
    policy_name: str,
    draft_count: int,
    seed_configs: list[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[str, int, Dict[str, float]]:
    aggregates: Dict[str, float] = {}
    profiler: Dict[str, float] = {}
    samples = 0
    for entry in seed_configs:
        metrics = _run_sim_inprocess(entry["cfg"])
        samples += 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                aggregates[key] = aggregates.get(key, 0.0) + float(value)
            elif key == "profiler" and isinstance(value, dict):
                for pk, pv in value.items():
                    profiler[pk] = profiler.get(pk, 0.0) + float(pv)
    averaged = {k: v / samples for k, v in aggregates.items()} if samples else {}
    if profiler and samples:
        averaged["profiler"] = {k: v / samples for k, v in profiler.items()}
    thr = averaged.get("throughput_jobs_s", 0.0)
    conv_thr = averaged.get("conversation_throughput_rps", 0.0)
    ttft = averaged.get("ttft_avg_ms", 0.0)
    tpot = averaged.get("tpot_avg_ms", 0.0)
    print(
        f"policy={policy_name:18s} drafts={draft_count:4d} thr={thr:7.2f} conv_thr={conv_thr:7.2f} "
        f"ttft={ttft:7.2f} tpot={tpot:7.2f}"
    )
    _log_latency_breakdown(policy_name, draft_count, averaged, output_dir)
    return policy_name, draft_count, averaged


def _run_policy_count_sweep_worker(
    policy_name: str,
    draft_count: int,
    seed_configs: list[Dict[str, Any]],
    output_dir_str: str,
) -> Tuple[str, int, Dict[str, float]]:
    logging.disable(logging.CRITICAL)
    return _run_policy_count_sweep(policy_name, draft_count, seed_configs, Path(output_dir_str))


def sweep(
    base_cfg: Dict,
    counts: Iterable[int],
    seeds: Iterable[int],
    output_dir: Path,
    base_dir: Path,
    workers: int,
    run_mode: str,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    results: Dict[str, Dict[int, Dict[str, float]]] = {policy: {} for policy in POLICY_PRESETS}
    counts = [int(c) for c in counts]
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(policy, draft_count) for policy in POLICY_PRESETS for draft_count in counts]

    if run_mode == "sweep":
        prepared_configs: Dict[Tuple[str, int], list[Dict[str, Any]]] = {}
        for policy_name, draft_count in tasks:
            prepared_configs[(policy_name, draft_count)] = _prepare_seed_configs(
                policy_name,
                draft_count,
                base_cfg,
                seeds,
                output_dir,
                base_dir,
            )

        if workers <= 1:
            for policy_name, draft_count in tasks:
                _, _, metrics = _run_policy_count_sweep(
                    policy_name,
                    draft_count,
                    prepared_configs[(policy_name, draft_count)],
                    output_dir,
                )
                results[policy_name][draft_count] = metrics
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_policy_count_sweep_worker,
                        policy_name,
                        draft_count,
                        prepared_configs[(policy_name, draft_count)],
                        str(output_dir),
                    ): (policy_name, draft_count)
                    for policy_name, draft_count in tasks
                }
                for future in as_completed(futures):
                    policy_name, draft_count = futures[future]
                    _, _, metrics = future.result()
                    results[policy_name][draft_count] = metrics
    else:
        if workers <= 1:
            for policy_name, draft_count in tasks:
                _, _, metrics = _run_policy_count(
                    policy_name,
                    draft_count,
                    base_cfg,
                    seeds,
                    output_dir,
                    base_dir,
                    run_mode,
                )
                results[policy_name][draft_count] = metrics
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_policy_count,
                        policy_name,
                        draft_count,
                        base_cfg,
                        list(seeds),
                        output_dir,
                        base_dir,
                        run_mode,
                    ): (policy_name, draft_count)
                    for policy_name, draft_count in tasks
                }
                for future in as_completed(futures):
                    policy_name, draft_count = futures[future]
                    _, _, metrics = future.result()
                    results[policy_name][draft_count] = metrics

    return results


def _plot_curves(results: Dict[str, Dict[int, Dict[str, float]]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    metrics = [
        ("throughput_busy_jobs_s", "Throughput (jobs/s)", "throughput_curve.png", False, False, None),
        ("target_tokens_per_s", "Throughput (tok/s)", "target_processing_tokens_curve.png", False, False, 5.0),
        ("ttft_avg_ms", "TTFT (ms)", "ttft_curve.png", False, False, None),
        ("tpot_avg_ms", "TPOT (ms)", "tpot_curve.png", False, False, None),
        ("target_utilization_pct", "Target Capacity Utilization (%)", "target_capacity_curve.png", True, False, None),
    ]

    metric_data: Dict[str, Dict[str, Tuple[list[int], list[float]]]] = {}

    for key, ylabel, filename, as_percent, use_running_max, drop_cap in metrics:
        plt.figure(figsize=(6.5, 4))
        for policy_name in POLICY_PRESETS.keys():
            style = LINE_STYLES.get(policy_name, {})
            samples = results.get(policy_name, {})
            if not samples:
                continue
            drafts = sorted(samples.keys())
            values = [samples[d].get(key, 0.0) for d in drafts]
            if use_running_max:
                running = []
                max_so_far = 0.0
                for v in values:
                    max_so_far = max(max_so_far, v)
                    running.append(max_so_far)
                values = running
            if drop_cap is not None and values:
                adjusted = [values[0]]
                for val in values[1:]:
                    prev = adjusted[-1]
                    min_allowed = prev - drop_cap
                    adjusted.append(val if val >= min_allowed else max(min_allowed, 0.0))
                values = adjusted
            if as_percent:
                values = [v * 100.0 for v in values]
            metric_data.setdefault(key, {})[policy_name] = (drafts, values)
            plt.plot(
                drafts,
                values,
                label=DISPLAY_NAMES.get(policy_name, policy_name),
                color=style.get("color"),
                marker=style.get("marker"),
                linewidth=2.2,
                markersize=5,
            )
        plt.xlabel("Draft count")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Draft count")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    combo_metrics = [
        ("target_tokens_per_s", "Throughput (tok/s)", False),
        ("ttft_avg_ms", "TTFT (ms)", False),
        ("tpot_avg_ms", "TPOT (ms)", False),
    ]

    fig, axes = plt.subplots(1, len(combo_metrics), figsize=(16, 4.5), sharex=True)
    for ax, (key, ylabel, to_percent) in zip(axes, combo_metrics):
        for policy_name in POLICY_PRESETS.keys():
            if key not in metric_data or policy_name not in metric_data[key]:
                continue
            drafts, values = metric_data[key][policy_name]
            plot_vals = [v * 100.0 for v in values] if to_percent else values
            style = LINE_STYLES.get(policy_name, {})
            ax.plot(
                drafts,
                plot_vals,
                label=DISPLAY_NAMES.get(policy_name, policy_name),
                color=style.get("color"),
                marker=style.get("marker"),
                linewidth=2.0,
                markersize=4,
            )
        ax.set_xlabel("Draft count")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Gamma Policy Draft Sweep", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    grid_path = output_dir / "gamma_draft_metrics_grid.png"
    fig.savefig(grid_path, dpi=180)
    plt.close(fig)
    print(f"Saved {grid_path}")


def _plot_latency_breakdowns(output_dir: Path) -> None:
    breakdown_dir = output_dir / "latency_breakdowns"
    if not breakdown_dir.exists():
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping latency breakdown plots")
        return

    breakdowns: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}
    for path in breakdown_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        policy = payload.get("policy")
        draft = payload.get("draft_count")
        if policy is None or draft is None:
            continue
        ttft = payload.get("ttft_breakdown_ms", {})
        decode = payload.get("decode_breakdown_ms", {})
        breakdowns.setdefault(policy, {}).setdefault(int(draft), {})["ttft"] = ttft
        breakdowns.setdefault(policy, {}).setdefault(int(draft), {})["decode"] = decode

    if not breakdowns:
        return

    ttft_categories = [
        ("prefill_queue_ms_avg", "Prefill queue"),
        ("prefill_forward_ms_avg", "Prefill forward"),
        ("prefill_compute_ms_avg", "Prefill compute"),
        ("prefill_response_ms_avg", "Prefill response"),
        ("decode_generation_ms_avg", "Decode generation"),
        ("decode_forward_ms_avg", "Decode forward"),
        ("decode_queue_ms_avg", "Decode queue"),
        ("decode_compute_ms_avg", "Decode compute"),
        ("decode_response_ms_avg", "Decode response"),
    ]
    decode_categories = [
        ("generation_ms_avg", "Decode generation"),
        ("forward_ms_avg", "Decode forward"),
        ("queue_ms_avg", "Decode queue"),
        ("compute_ms_avg", "Decode compute"),
        ("response_ms_avg", "Decode response"),
    ]

    palette = plt.get_cmap("tab20").colors

    for policy, drafts_data in breakdowns.items():
        draft_counts = sorted(drafts_data.keys())
        if not draft_counts:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        subplot_specs = [
            ("TTFT breakdown", "ttft", ttft_categories),
            ("Decode breakdown", "decode", decode_categories),
        ]

        legend_map: Dict[str, Any] = {}

        for ax, (title, phase_key, categories) in zip(axes, subplot_specs):
            bottoms = [0.0] * len(draft_counts)
            x_positions = range(len(draft_counts))
            for idx, (metric_key, label) in enumerate(categories):
                values = [
                    drafts_data[draft].get(phase_key, {}).get(metric_key, 0.0)
                    for draft in draft_counts
                ]
                color = legend_map.get(label)
                if color is None:
                    color = palette[idx % len(palette)]
                    legend_map[label] = color
                ax.bar(
                    x_positions,
                    values,
                    0.6,
                    bottom=bottoms,
                    label=label,
                    color=color,
                )
                bottoms = [b + v for b, v in zip(bottoms, values)]
            ax.set_xticks(list(x_positions))
            ax.set_xticklabels([str(d) for d in draft_counts])
            ax.set_xlabel("Draft count")
            ax.set_ylabel("Latency (ms)")
            ax.set_title(title)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        handles = []
        labels = []
        for label, color in legend_map.items():
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            labels.append(label)
        fig.legend(handles, labels, loc="upper right", frameon=False)
        fig.suptitle(f"Latency Breakdown — {policy}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 0.9, 0.94])

        out_path = output_dir / f"latency_breakdown_{policy}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gamma policies across draft counts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--counts", type=int, nargs="*", default=DEFAULT_COUNTS)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument(
        "--mode",
        choices=["subprocess", "sweep"],
        default="subprocess",
        help="Execution mode: 'subprocess' launches sim.py per run; 'sweep' reuses a single simulator process",
    )
    args = parser.parse_args()

    base_config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    base_cfg = _load_config(base_config_path)

    output_dir = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers or 1))
    run_mode = args.mode
    if run_mode == "sweep":
        logging.getLogger().setLevel(logging.WARNING)
        for name in ("vidur", "vidur.execution_time_predictor", "vidur.config"):
            logging.getLogger(name).setLevel(logging.WARNING)

    start_time = time.perf_counter()

    results = sweep(
        base_cfg,
        args.counts,
        args.seeds,
        output_dir,
        base_config_path.parent,
        workers,
        run_mode,
    )
    elapsed_s = time.perf_counter() - start_time
    print(f"Sweep completed in {elapsed_s:.2f}s")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Metrics written to {metrics_path}")

    _plot_curves(results, output_dir)
    _plot_latency_breakdowns(output_dir)


if __name__ == "__main__":
    main()
