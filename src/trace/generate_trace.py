#!/usr/bin/env python3
"""Generate synthetic traces based on an existing simulator configuration."""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path
import argparse
import gzip
import json
from typing import Dict, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import load_config
from trace.synthetic_trace import (
    LengthDistribution,
    SyntheticTraceConfig,
    SyntheticTraceGenerator,
    build_device_mix_from_specs,
)


def _parse_kv(entries: Iterable[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(f"metadata entry '{entry}' must be KEY=VALUE")
        key, value = entry.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _build_prompt_distribution(args, cfg) -> LengthDistribution:
    if args.prompt_dist == "fixed":
        fixed = args.prompt_fixed if args.prompt_fixed is not None else (
            cfg.prompt_length_min if cfg.prompt_length_min == cfg.prompt_length_max else cfg.prompt_length_min
        )
        return LengthDistribution(kind="fixed", fixed=int(fixed))
    minimum = args.prompt_min if args.prompt_min is not None else cfg.prompt_length_min
    maximum = args.prompt_max if args.prompt_max is not None else cfg.prompt_length_max
    return LengthDistribution(kind="uniform", minimum=int(minimum), maximum=int(maximum))


def _build_answer_distribution(args, cfg) -> LengthDistribution:
    if args.answer_dist == "fixed":
        fixed = args.answer_fixed if args.answer_fixed is not None else cfg.answer_length
        return LengthDistribution(kind="fixed", fixed=int(fixed))
    mean = args.answer_mean if args.answer_mean is not None else cfg.answer_length_mean
    std = args.answer_std if args.answer_std is not None else cfg.answer_length_std
    minimum = args.answer_min if args.answer_min is not None else cfg.answer_length_min
    maximum = args.answer_max if args.answer_max is not None else cfg.answer_length_max
    return LengthDistribution(kind="normal", mean=float(mean), stddev=float(std), minimum=int(minimum), maximum=int(maximum))


def _write_records(generator: SyntheticTraceGenerator, output_path: Path) -> int:
    count = 0
    if output_path.suffix == ".gz":
        opener = lambda p: gzip.open(p, "wt", encoding="utf-8")  # pragma: no cover
    else:
        opener = lambda p: p.open("w", encoding="utf-8")
    with opener(output_path) as fh:  # type: ignore[misc]
        for record in generator.iter_records():
            fh.write(json.dumps(record.to_dict(), separators=(",", ":")))
            fh.write("\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic trace for the simulator")
    parser.add_argument("--config", required=True, help="Simulation config (supports auto_topology)")
    parser.add_argument("--output", required=True, help="Destination JSONL/JSONL.GZ path")
    parser.add_argument("--duration-ms", type=float, default=60_000, help="Trace duration in milliseconds")
    parser.add_argument("--start-ms", type=float, default=0.0, help="Start timestamp for first request")
    parser.add_argument("--arrival", choices=["poisson", "deterministic"], default="poisson")
    parser.add_argument("--rate-rps", type=float, default=None, help="Mean request rate (req/s) for Poisson arrivals")
    parser.add_argument("--interarrival-ms", type=float, default=None, help="Interarrival time for deterministic arrivals")
    parser.add_argument("--burst-factor", type=float, default=1.0, help="Optional burst multiplier")
    parser.add_argument("--max-requests", type=int, default=None, help="Cap the number of generated requests")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (defaults to config seed)")
    parser.add_argument("--request-prefix", default="req", help="Prefix for generated request IDs")
    parser.add_argument("--no-request-ids", action="store_true", help="Disable automatic request IDs")
    parser.add_argument("--no-assign-seeds", action="store_true", help="Disable per-request RNG seeds")
    parser.add_argument("--uniform-weights", action="store_true", help="Ignore capability when weighting drafts")
    parser.add_argument("--default-weight", type=float, default=1.0, help="Fallback weight when capability missing")
    parser.add_argument("--prompt-dist", choices=["uniform", "fixed"], default="uniform")
    parser.add_argument("--prompt-min", type=int, default=None)
    parser.add_argument("--prompt-max", type=int, default=None)
    parser.add_argument("--prompt-fixed", type=int, default=None)
    parser.add_argument("--answer-dist", choices=["normal", "fixed"], default="normal")
    parser.add_argument("--answer-mean", type=float, default=None)
    parser.add_argument("--answer-std", type=float, default=None)
    parser.add_argument("--answer-min", type=int, default=None)
    parser.add_argument("--answer-max", type=int, default=None)
    parser.add_argument("--answer-fixed", type=int, default=None)
    parser.add_argument("--default-slo", default=None, help="Default SLO class to embed in trace records")
    parser.add_argument("--default-mode", default=None, help="Default mode hint to embed in trace records")
    parser.add_argument("--metadata", nargs="*", default=[], help="Additional metadata entries (KEY=VALUE)")
    parser.add_argument("--omit-draft-ids", action="store_true", help="Emit device tiers without binding records to specific draft IDs.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    draft_specs = [d for d in cfg.devices if d.get("role", "target") == "draft"]
    if not draft_specs:
        raise SystemExit("configuration does not define any draft devices")

    weight_key = None if args.uniform_weights else "capability"
    device_mix = build_device_mix_from_specs(
        draft_specs,
        weight_key=weight_key,
        default_weight=args.default_weight,
    )

    prompt_dist = _build_prompt_distribution(args, cfg)
    answer_dist = _build_answer_distribution(args, cfg)

    rate_rps = args.rate_rps if args.rate_rps is not None else cfg.workload.rate_rps
    interarrival_ms = args.interarrival_ms if args.interarrival_ms is not None else cfg.workload.interarrival_ms

    metadata = {
        "scenario": Path(args.config).name,
        "arrival_process": args.arrival,
        "generated_by": "trace/generate_trace.py",
    }
    metadata.update(_parse_kv(args.metadata))

    trace_cfg = SyntheticTraceConfig(
        duration_ms=args.duration_ms,
        start_ms=args.start_ms,
        arrival_process=args.arrival,
        rate_rps=rate_rps,
        interarrival_ms=interarrival_ms,
        burst_factor=args.burst_factor,
        prompt=prompt_dist,
        target=answer_dist,
        device_mix=device_mix,
        default_slo_class=args.default_slo,
        default_mode_hint=args.default_mode,
        metadata=metadata,
        max_requests=args.max_requests,
        assign_request_ids=not args.no_request_ids,
        request_id_prefix=args.request_prefix,
        assign_request_seeds=not args.no_assign_seeds,
        assign_draft_ids=not args.omit_draft_ids,
        seed=args.seed if args.seed is not None else cfg.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = SyntheticTraceGenerator(trace_cfg)
    count = _write_records(generator, output_path)
    print(f"Generated {count} trace records at {output_path}")


if __name__ == "__main__":
    main()
