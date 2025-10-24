#!/usr/bin/env python3
"""Sweep routing algorithms over a draft-count range and compare metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml

from gamma_policy_draft_sweep import (
    POLICY_PRESETS,
    DISPLAY_NAMES,
    sweep as base_sweep,
)

ROUTER_CONFIGS = {
    "jsq": Path("experiments/configs/routing/routing_jsq.yaml"),
    "round_robin": Path("experiments/configs/routing/routing_round_robin.yaml"),
    "random": Path("experiments/configs/routing/routing_random.yaml"),
}


def sweep_routing(
    routers: Iterable[str],
    counts: Iterable[int],
    seeds: Iterable[int],
    output_root: Path,
    workers: int,
    run_mode: str,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Run base sweep per router and collate results."""
    results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for router in routers:
        cfg_path = ROUTER_CONFIGS[router]
        router_dir = output_root / router
        router_dir.mkdir(parents=True, exist_ok=True)
        router_results = base_sweep(
            base_cfg=yaml.safe_load(cfg_path.read_text()),
            counts=counts,
            seeds=seeds,
            output_dir=router_dir,
            base_dir=Path.cwd(),
            workers=workers,
            run_mode=run_mode,
        )
        results[router] = router_results["gamma4_static"]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50, 60, 70, 80],
        help="Draft counts to sweep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for all runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/routing_study"),
        help="Output directory root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum parallel workers.",
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "subprocess"],
        default="sweep",
        help="Run mode (delegated to the base sweep).",
    )
    args = parser.parse_args()

    routers = ["jsq", "round_robin", "random"]
    seeds = [args.seed]

    sweep_routing(
        routers=routers,
        counts=args.counts,
        seeds=seeds,
        output_root=args.output,
        workers=args.workers,
        run_mode=args.mode,
    )


if __name__ == "__main__":
    main()
