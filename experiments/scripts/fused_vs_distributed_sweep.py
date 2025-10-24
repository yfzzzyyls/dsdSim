#!/usr/bin/env python3
"""Compare fused vs distributed speculative execution across draft counts."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_PATH = REPO_ROOT / "src" / "sim.py"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "explorer" / "baseline_trace_gsm8k.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "results" / "fused_vs_distributed"
DEFAULT_COUNTS = [200, 300, 400, 600, 800, 1000, 1500, 2048, 3072, 4096]
DEFAULT_SEEDS = [123]
VIDUR_REPO_PATH = REPO_ROOT / "thirdparty" / "Sai_speculative_vidur"

EXECUTION_MODES: Dict[str, Dict[str, Any]] = {
    "distributed": {"label": "Distributed", "style": {"color": "#1f77b4", "marker": "o"}},
    "fused": {"label": "Fused", "style": {"color": "#ff7f0e", "marker": "s"}},
}

# Ensure in-process mode can import repository modules and vendored VIDUR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(VIDUR_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(VIDUR_REPO_PATH))

import yaml  # type: ignore
from src import sim  # type: ignore


class SweepError(RuntimeError):
    pass


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SweepError(f"Baseline config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _ensure_latency_metadata(cfg: Dict[str, Any]) -> None:
    for cluster in cfg.get("auto_topology", {}).get("clusters", []):
        target_spec = cluster.get("targets", {})
        for tier in target_spec.get("tiers", []):
            metadata = dict(tier.get("metadata", {}))
            metadata.setdefault("prefill_latency_per_token", 30.0)
            metadata.setdefault("decode_latency_per_token", 9.0)
            tier["metadata"] = metadata


def _rebalance_count_by_label(draft_spec: Dict[str, Any], new_total: int) -> None:
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


def _resolve_trace_path(cfg: Dict[str, Any], base_dir: Path) -> None:
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


def _build_config(
    base: Dict[str, Any],
    *,
    execution_mode: str,
    draft_count: int,
    base_dir: Path,
    gamma: Optional[int],
) -> Dict[str, Any]:
    cfg = deepcopy(base)
    cfg["sim_time_ms"] = 2000
    cfg["verbose"] = False
    cfg["debug"] = False
    if gamma is not None:
        cfg["gamma"] = int(gamma)

    spec = cfg.setdefault("speculation", {})
    spec["framework"] = spec.get("framework", "vanilla")
    spec["execution_mode"] = execution_mode

    acceptance_cfg = spec.setdefault("acceptance", {})
    if acceptance_cfg.get("disable_model"):
        spec.pop("acceptance_model", None)
        acceptance_cfg.pop("model", None)
        acceptance_cfg.pop("file", None)
    else:
        model_path = (
            spec.get("acceptance_model")
            or acceptance_cfg.get("model")
            or acceptance_cfg.get("file")
        )
        if not model_path:
            raise SweepError(
                "Acceptance model not specified; please set "
                "`speculation.acceptance.model` (or disable the model explicitly)."
            )

    _resolve_trace_path(cfg, base_dir)
    _ensure_latency_metadata(cfg)

    for cluster in cfg.get("auto_topology", {}).get("clusters", []):
        drafts = cluster.get("drafts", {})
        drafts["count"] = int(draft_count)
        _rebalance_count_by_label(drafts, int(draft_count))

    return cfg


def _write_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _run_sim_subprocess(config_path: Path) -> Dict[str, float]:
    cmd = [sys.executable, str(SIM_PATH), "--config", str(config_path)]
    env = dict(os.environ)
    extra = [str(VIDUR_REPO_PATH)]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(extra + [existing]) if existing else os.pathsep.join(extra)
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


def _log_latency_breakdown(mode: str, draft_count: int, metrics: Dict[str, float], output_dir: Path) -> None:
    def _collect(prefix: str) -> Dict[str, float]:
        entries = {key[len(prefix):]: value for key, value in metrics.items() if key.startswith(prefix)}
        return dict(sorted(entries.items())) if entries else {}

    ttft_parts = _collect("ttft_breakdown_")
    decode_parts = _collect("decode_breakdown_")
    if not ttft_parts and not decode_parts:
        return

    payload = {
        "mode": mode,
        "draft_count": draft_count,
        "ttft_breakdown_ms": ttft_parts,
        "decode_breakdown_ms": decode_parts,
    }
    dump_dir = output_dir / "latency_breakdowns"
    dump_dir.mkdir(parents=True, exist_ok=True)
    out_path = dump_dir / f"{mode}_drafts{draft_count}.json"
    out_path.write_text(json.dumps(payload, indent=2))


def _prepare_seed_configs(
    mode: str,
    draft_count: int,
    base_cfg: Dict[str, Any],
    seeds: Iterable[int],
    output_dir: Path,
    base_dir: Path,
    gamma: Optional[int],
) -> list[Dict[str, Any]]:
    prepared: list[Dict[str, Any]] = []
    for seed in seeds:
        cfg = _build_config(base_cfg, execution_mode=mode, draft_count=draft_count, base_dir=base_dir, gamma=gamma)
        cfg["seed"] = int(seed)
        cfg_path = output_dir / "configs" / f"{mode}_drafts{draft_count}_seed{seed}.yaml"
        _write_config(cfg, cfg_path)
        prepared.append({"cfg": cfg, "path": cfg_path, "seed": int(seed)})
    return prepared


def _run_mode_count(
    mode: str,
    draft_count: int,
    base_cfg: Dict[str, Any],
    seeds: Iterable[int],
    output_dir: Path,
    base_dir: Path,
    runner: str,
    gamma: Optional[int],
) -> Tuple[str, int, Dict[str, float]]:
    aggregates: Dict[str, float] = {}
    samples = 0
    for seed in seeds:
        cfg = _build_config(base_cfg, execution_mode=mode, draft_count=draft_count, base_dir=base_dir, gamma=gamma)
        cfg["seed"] = int(seed)
        cfg_path = output_dir / "configs" / f"{mode}_drafts{draft_count}_seed{seed}.yaml"
        _write_config(cfg, cfg_path)
        metrics = _run_sim_inprocess(cfg) if runner == "sweep" else _run_sim_subprocess(cfg_path)
        samples += 1
        for key, value in metrics.items():
            aggregates[key] = aggregates.get(key, 0.0) + float(value)

    averaged = {k: v / samples for k, v in aggregates.items()} if samples else {}
    thr = averaged.get("throughput_jobs_s", 0.0)
    ttft = averaged.get("ttft_avg_ms", 0.0)
    tpot = averaged.get("tpot_avg_ms", 0.0)
    print(f"mode={mode:12s} drafts={draft_count:4d} thr={thr:7.2f} ttft={ttft:7.2f} tpot={tpot:7.2f}")
    _log_latency_breakdown(mode, draft_count, averaged, output_dir)
    return mode, draft_count, averaged


def _run_mode_count_sweep(
    mode: str,
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
    ttft = averaged.get("ttft_avg_ms", 0.0)
    tpot = averaged.get("tpot_avg_ms", 0.0)
    print(f"mode={mode:12s} drafts={draft_count:4d} thr={thr:7.2f} ttft={ttft:7.2f} tpot={tpot:7.2f}")
    _log_latency_breakdown(mode, draft_count, averaged, output_dir)
    return mode, draft_count, averaged


def _run_mode_count_sweep_worker(
    mode: str,
    draft_count: int,
    seed_configs: list[Dict[str, Any]],
    output_dir_str: str,
) -> Tuple[str, int, Dict[str, float]]:
    logging.disable(logging.CRITICAL)
    return _run_mode_count_sweep(mode, draft_count, seed_configs, Path(output_dir_str))


def compare_modes(
    base_cfg: Dict[str, Any],
    counts: Iterable[int],
    seeds: Iterable[int],
    output_dir: Path,
    base_dir: Path,
    workers: int,
    runner: str,
    modes: Iterable[str],
    gamma: Optional[int],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    selected_modes = [m for m in modes if m in EXECUTION_MODES]
    if not selected_modes:
        raise SweepError("No valid execution modes selected.")

    counts = [int(c) for c in counts]
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[int, Dict[str, float]]] = {mode: {} for mode in selected_modes}
    tasks = [(mode, draft_count) for mode in selected_modes for draft_count in counts]

    if runner == "sweep":
        prepared: Dict[Tuple[str, int], list[Dict[str, Any]]] = {}
        for mode, draft_count in tasks:
            prepared[(mode, draft_count)] = _prepare_seed_configs(
                mode, draft_count, base_cfg, seeds, output_dir, base_dir, gamma
            )

        if workers <= 1:
            for mode, draft_count in tasks:
                _, _, metrics = _run_mode_count_sweep(mode, draft_count, prepared[(mode, draft_count)], output_dir)
                results[mode][draft_count] = metrics
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_mode_count_sweep_worker,
                        mode,
                        draft_count,
                        prepared[(mode, draft_count)],
                        str(output_dir),
                    ): (mode, draft_count)
                    for mode, draft_count in tasks
                }
                for future in as_completed(futures):
                    mode, draft_count = futures[future]
                    _, _, metrics = future.result()
                    results[mode][draft_count] = metrics
    else:
        if workers <= 1:
            for mode, draft_count in tasks:
                _, _, metrics = _run_mode_count(
                    mode,
                    draft_count,
                    base_cfg,
                    seeds,
                    output_dir,
                    base_dir,
                    runner,
                    gamma,
                )
                results[mode][draft_count] = metrics
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_mode_count,
                        mode,
                        draft_count,
                        base_cfg,
                        list(seeds),
                        output_dir,
                        base_dir,
                        runner,
                        gamma,
                    ): (mode, draft_count)
                    for mode, draft_count in tasks
                }
                for future in as_completed(futures):
                    mode, draft_count = futures[future]
                    _, _, metrics = future.result()
                    results[mode][draft_count] = metrics

    return results


def _plot_comparison(results: Dict[str, Dict[int, Dict[str, float]]], output_dir: Path, modes: Iterable[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    selected_modes = [m for m in modes if m in EXECUTION_MODES]
    if not selected_modes:
        return

    metric_specs = [
        ("throughput_jobs_s", "Throughput (jobs/s)"),
        ("ttft_avg_ms", "TTFT (ms)"),
        ("tpot_avg_ms", "TPOT (ms)"),
    ]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(15, 4.5), sharex=False)
    for ax, (metric_key, ylabel) in zip(axes, metric_specs):
        for mode in selected_modes:
            samples = results.get(mode, {})
            if not samples:
                continue
            counts = sorted(samples.keys())
            values = [samples[count].get(metric_key, 0.0) for count in counts]
            style = EXECUTION_MODES[mode]["style"]
            label = EXECUTION_MODES[mode]["label"]
            ax.plot(
                counts,
                values,
                label=label,
                color=style.get("color"),
                marker=style.get("marker"),
                linewidth=2.2,
                markersize=5,
            )
        ax.set_xlabel("Draft count")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Fused vs Distributed Speculation", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / "fused_vs_distributed_metrics.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


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
        mode = payload.get("mode")
        draft = payload.get("draft_count")
        if mode is None or draft is None:
            continue
        ttft = payload.get("ttft_breakdown_ms", {})
        decode = payload.get("decode_breakdown_ms", {})
        breakdowns.setdefault(mode, {}).setdefault(int(draft), {})["ttft"] = ttft
        breakdowns.setdefault(mode, {}).setdefault(int(draft), {})["decode"] = decode

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

    for mode, drafts_data in breakdowns.items():
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

        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for label, color in legend_map.items()]
        labels = list(legend_map.keys())
        fig.legend(handles, labels, loc="upper right", frameon=False)
        mode_label = EXECUTION_MODES.get(mode, {}).get("label", mode)
        fig.suptitle(f"Latency Breakdown — {mode_label}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 0.9, 0.94])

        out_path = output_dir / f"latency_breakdown_{mode}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fused vs distributed speculation across draft counts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base simulator config.")
    parser.add_argument("--counts", type=int, nargs="*", default=DEFAULT_COUNTS, help="Draft counts to sweep.")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS, help="Random seeds to average.")
    parser.add_argument("--gamma", type=int, default=None, help="Override gamma (tokens per chunk).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for results.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for simulations.")
    parser.add_argument(
        "--runner",
        choices=["subprocess", "sweep"],
        default="subprocess",
        help="Execution strategy: launch sim.py per run or reuse in-process simulator.",
    )
    parser.add_argument(
        "--modes",
        choices=sorted(EXECUTION_MODES.keys()),
        nargs="*",
        default=list(EXECUTION_MODES.keys()),
        help="Execution modes to compare.",
    )
    args = parser.parse_args()

    base_config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    base_cfg = _load_config(base_config_path)

    output_dir = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers or 1))
    runner = args.runner
    if runner == "sweep":
        logging.getLogger().setLevel(logging.WARNING)
        for name in ("vidur", "vidur.execution_time_predictor", "vidur.config"):
            logging.getLogger(name).setLevel(logging.WARNING)

    start_time = time.perf_counter()
    results = compare_modes(
        base_cfg,
        args.counts,
        args.seeds,
        output_dir,
        base_config_path.parent,
        workers,
        runner,
        args.modes,
        args.gamma,
    )
    elapsed_s = time.perf_counter() - start_time
    print(f"Sweep completed in {elapsed_s:.2f}s")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Metrics written to {metrics_path}")

    _plot_comparison(results, output_dir, args.modes)
    _plot_latency_breakdowns(output_dir)


if __name__ == "__main__":
    main()
