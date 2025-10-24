#!/usr/bin/env python3
"""Sweep RTT between drafts and targets to compare fused vs distributed execution."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.fused_vs_distributed_sweep import (  # type: ignore
    EXECUTION_MODES,
    compare_modes,
)

import yaml  # type: ignore


def _cluster_index_map(clusters: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(cluster.get("name", "")): cluster for cluster in clusters}


def _override_latency(
    base_cfg: Dict[str, Any],
    latency_ms: float,
    jitter: float | None,
    mode: str,
) -> Dict[str, Any]:
    """Return a copy of config with draft<->target RTT adjusted according to the provided mode."""
    updated = deepcopy(base_cfg)
    auto = updated.get("auto_topology", {})
    clusters = auto.get("clusters", [])
    if not clusters:
        raise ValueError("Configuration does not define auto_topology clusters for connectivity override.")

    base_clusters = _cluster_index_map(base_cfg.get("auto_topology", {}).get("clusters", []))  # type: ignore[call-overload]

    for cluster in clusters:
        name = str(cluster.get("name", ""))
        base_cluster = base_clusters.get(name, {})
        connectivity = cluster.setdefault("connectivity", {})
        if jitter is not None:
            connectivity["link_jitter_pct"] = float(jitter)
        net_ranges = connectivity.setdefault("net_ms_ranges", {})
        targets_spec = cluster.get("targets", {})
        tiers: Iterable[Dict[str, Any]] = targets_spec.get("tiers", []) or []
        for tier in tiers:
            tier_name = tier.get("name")
            if not tier_name:
                continue
            base_range = None
            if isinstance(base_cluster, Mapping):
                base_conn = base_cluster.get("connectivity", {})
                if isinstance(base_conn, Mapping):
                    base_ranges = base_conn.get("net_ms_ranges", {})
                    if isinstance(base_ranges, Mapping):
                        base_range = base_ranges.get(tier_name)

            if mode == "offset" and base_range:
                try:
                    base_min, base_max = float(base_range[0]), float(base_range[1])
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Invalid base net_ms_ranges entry for tier '{tier_name}': {base_range}") from exc
                net_ranges[tier_name] = [base_min + float(latency_ms), base_max + float(latency_ms)]
            else:
                net_ranges[tier_name] = [float(latency_ms), float(latency_ms)]

    return updated


def _plot_metrics_vs_latency(
    aggregate: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    latencies: Sequence[float],
    counts: Sequence[int],
    modes: Sequence[str],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping latency sweep plots")
        return

    lat_keys = sorted(aggregate.keys(), key=lambda key: float(key))
    if not lat_keys:
        return

    metrics = [
        ("throughput_jobs_s", "Throughput (jobs/s)"),
        ("ttft_avg_ms", "TTFT (ms)"),
        ("tpot_avg_ms", "TPOT (ms)"),
    ]

    counts_sorted = sorted({int(c) for c in counts})
    modes = [m for m in modes if m in EXECUTION_MODES]
    if not counts_sorted or not modes:
        return

    n_rows = len(counts_sorted)
    n_cols = len(metrics)
    figsize = (5 * n_cols, 3.2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    lat_map: Dict[float, Dict[str, Dict[str, float]]] = {float(k): v for k, v in aggregate.items()}
    lat_sorted = sorted(lat_map.keys())
    if not lat_sorted:
        return

    for row_idx, count in enumerate(counts_sorted):
        for col_idx, (metric_key, ylabel) in enumerate(metrics):
            ax = axes[row_idx][col_idx]

            for mode in modes:
                style = EXECUTION_MODES.get(mode, {})
                color = style.get("color")
                marker = style.get("marker")
                label = EXECUTION_MODES.get(mode, {}).get("label", mode)

                lat_vals: List[float] = []
                metric_vals: List[float] = []

                for lat in lat_sorted:
                    mode_bucket = lat_map.get(lat, {}).get(mode, {})
                    entry = mode_bucket.get(str(count))
                    if entry and metric_key in entry:
                        lat_vals.append(lat)
                        metric_vals.append(float(entry[metric_key]))

                if not lat_vals:
                    continue

                ax.plot(
                    lat_vals,
                    metric_vals,
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=2.5,
                    markersize=8,
                )
                ax.scatter(
                    lat_vals,
                    metric_vals,
                    color=color,
                    marker=marker,
                    s=70,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
                for x_val, y_val in zip(lat_vals, metric_vals):
                    ax.text(
                        x_val,
                        y_val,
                        f"{y_val:.0f}",
                        fontsize=7,
                        color=color,
                        ha="center",
                        va="bottom",
                        zorder=4,
                    )

            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)
            if row_idx == n_rows - 1:
                ax.set_xlabel("RTT (ms)")

            if lat_sorted:
                x_min = min(lat_sorted)
                x_max = max(lat_sorted)
                if x_min == x_max:
                    delta = max(1.0, abs(x_min) * 0.05 + 1.0)
                    ax.set_xlim(x_min - delta, x_max + delta)
                else:
                    pad = max(0.5, (x_max - x_min) * 0.05)
                    ax.set_xlim(x_min - pad, x_max + pad)

            lines = ax.get_lines()
            all_y = []
            for line in lines:
                all_y.extend(line.get_ydata())
            if all_y:
                y_min = min(all_y)
                y_max = max(all_y)
                if y_min == y_max:
                    delta = max(1.0, abs(y_min) * 0.05 + 1.0)
                    ax.set_ylim(y_min - delta, y_max + delta)
                else:
                    pad = max(1.0, (y_max - y_min) * 0.1)
                    ax.set_ylim(y_min - pad, y_max + pad)

            if col_idx == 0:
                ax.set_title(f"Drafts = {count}", loc="left", fontsize=11)
            if col_idx == n_cols - 1:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, frameon=False, loc="best")

    fig.suptitle("Fused vs Distributed Performance vs RTT", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / "metrics_vs_latency.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def _plot_per_count_panels(
    aggregate: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    latencies: Sequence[float],
    counts: Sequence[int],
    modes: Sequence[str],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping per-draft plots")
        return

    lat_map: Dict[float, Dict[str, Dict[str, float]]] = {float(k): v for k, v in aggregate.items()}
    lat_sorted = sorted(lat_map.keys())
    if not lat_sorted:
        return

    metrics = [
        ("throughput_jobs_s", "Throughput (jobs/s)"),
        ("ttft_avg_ms", "TTFT (ms)"),
        ("tpot_avg_ms", "TPOT (ms)"),
    ]

    counts_sorted = sorted({int(c) for c in counts})
    modes = [m for m in modes if m in EXECUTION_MODES]
    if not counts_sorted or not modes:
        return

    for count in counts_sorted:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.4), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        title_prefix = f"Drafts = {count}"

        for ax, (metric_key, ylabel) in zip(axes, metrics):
            for mode in modes:
                style = EXECUTION_MODES.get(mode, {})
                color = style.get("color")
                marker = style.get("marker")
                label = EXECUTION_MODES.get(mode, {}).get("label", mode)

                lat_vals: List[float] = []
                metric_vals: List[float] = []
                for lat in lat_sorted:
                    entry = lat_map.get(lat, {}).get(mode, {}).get(str(count))
                    if entry and metric_key in entry:
                        lat_vals.append(lat)
                        metric_vals.append(float(entry[metric_key]))

                if not lat_vals:
                    continue

                ax.plot(
                    lat_vals,
                    metric_vals,
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=2.5,
                    markersize=8,
                )
                ax.scatter(
                    lat_vals,
                    metric_vals,
                    color=color,
                    marker=marker,
                    s=70,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
                for x_val, y_val in zip(lat_vals, metric_vals):
                    ax.text(
                        x_val,
                        y_val,
                        f"{y_val:.0f}",
                        fontsize=7,
                        color=color,
                        ha="center",
                        va="bottom",
                        zorder=4,
                    )

            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlabel("RTT (ms)")

            if lat_sorted:
                x_min = min(lat_sorted)
                x_max = max(lat_sorted)
                if x_min == x_max:
                    delta = max(1.0, abs(x_min) * 0.05 + 1.0)
                    ax.set_xlim(x_min - delta, x_max + delta)
                else:
                    pad = max(0.5, (x_max - x_min) * 0.05)
                    ax.set_xlim(x_min - pad, x_max + pad)

            lines = ax.get_lines()
            all_y = []
            for line in lines:
                all_y.extend(line.get_ydata())
            if all_y:
                y_min = min(all_y)
                y_max = max(all_y)
                if y_min == y_max:
                    delta = max(1.0, abs(y_min) * 0.05 + 1.0)
                    ax.set_ylim(y_min - delta, y_max + delta)
                else:
                    pad = max(1.0, (y_max - y_min) * 0.1)
                    ax.set_ylim(y_min - pad, y_max + pad)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, frameon=False, loc="best")

        fig.suptitle(f"{title_prefix}: SLOs vs RTT", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        count_dir = output_dir / f"drafts_{count}"
        count_dir.mkdir(parents=True, exist_ok=True)
        output_path = count_dir / "metrics_vs_rtt.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

        metadata = {
            "draft_count": count,
            "latencies_ms": lat_sorted,
            "modes": modes,
        }
        (count_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"Saved {output_path}")


def _parse_latencies(values: Iterable[str]) -> List[float]:
    latencies: List[float] = []
    for entry in values:
        try:
            latencies.append(float(entry))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid latency '{entry}': {exc}") from exc
    if not latencies:
        raise argparse.ArgumentTypeError("at least one latency must be provided")
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep RTT between drafts and targets for fused vs distributed comparison.")
    parser.add_argument("--config", type=Path, required=True, help="Base simulator config.")
    parser.add_argument(
        "--latencies",
        type=str,
        nargs="+",
        required=True,
        help="Latency values (ms) to sweep; forward/response latencies are set to this value.",
    )
    parser.add_argument("--counts", type=int, nargs="*", default=[600], help="Draft counts to evaluate.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[123], help="Random seeds to average.")
    parser.add_argument("--gamma", type=int, default=None, help="Override gamma (tokens per chunk).")
    parser.add_argument("--output", type=Path, required=True, help="Directory for aggregated results.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for simulations.")
    parser.add_argument(
        "--runner",
        choices=["subprocess", "sweep"],
        default="sweep",
        help="Execution strategy: launch sim.py per run or reuse in-process simulator.",
    )
    parser.add_argument(
        "--modes",
        choices=sorted(EXECUTION_MODES.keys()),
        nargs="*",
        default=list(EXECUTION_MODES.keys()),
        help="Execution modes to compare.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=None,
        help="Override link jitter percentage; omit to retain values from the base config.",
    )
    parser.add_argument(
        "--mode",
        choices=["absolute", "offset"],
        default="absolute",
        help="Interpret provided latencies as absolute RTTs or as additional delay added to existing ranges.",
    )
    args = parser.parse_args()

    base_config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    base_dir = base_config_path.parent
    base_cfg = yaml.safe_load(base_config_path.read_text()) or {}

    latencies = _parse_latencies(args.latencies)
    counts = [int(c) for c in (args.counts or [])]
    if not counts:
        raise SystemExit("At least one draft count must be provided.")
    seeds = [int(s) for s in (args.seeds or [])]

    output_root = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers or 1))
    modes = args.modes

    aggregate: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}

    runs_root = output_root / "runs"

    for latency in latencies:
        override_cfg = _override_latency(base_cfg, latency_ms=latency, jitter=args.jitter, mode=args.mode)
        latency_dir = runs_root / f"lat_{latency:.2f}ms"
        latency_dir.mkdir(parents=True, exist_ok=True)

        results = compare_modes(
            override_cfg,
            counts,
            seeds,
            latency_dir,
            base_dir,
            workers,
            args.runner,
            modes,
            args.gamma,
        )
        aggregate[f"{latency:.2f}"] = results

        summary_path = latency_dir / "latency_metadata.json"
        summary_path.write_text(
            json.dumps(
                {
                    "latency_ms": latency,
                    "counts": counts,
                    "modes": modes,
                    "seeds": seeds,
                    "mode": args.mode,
                    "jitter": args.jitter,
                },
                indent=2,
            )
        )

    combined_path = output_root / "metrics_by_latency.json"
    combined_path.write_text(json.dumps(aggregate, indent=2))
    print(f"Aggregated metrics written to {combined_path}")

    _plot_metrics_vs_latency(aggregate, latencies, counts, modes, output_dir=output_root)
    _plot_per_count_panels(aggregate, latencies, counts, modes, output_root)


if __name__ == "__main__":
    main()
