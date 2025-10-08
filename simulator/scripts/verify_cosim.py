#!/usr/bin/env python3
"""Co-simulation verifier: compare vLLM measurements with simulator output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from simulator.verification.cosim import ActualRun, compare_runs, load_actual_run


def _print_report(report) -> None:
    print(f"Trace written to: {report.trace_path}")
    print("\nActual summary:")
    for key, value in sorted(report.actual_summary.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nSimulator summary:")
    for key, value in sorted(report.simulated_summary.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nDifferences:")
    header = f"{'metric':<24}{'actual':>14}{'simulated':>14}{'delta':>14}{'rel.%':>10}"
    print(header)
    print("-" * len(header))
    for name, actual, simulated, delta, pct in report.differences:
        a = f"{actual:.4f}" if isinstance(actual, (int, float)) else "-"
        b = f"{simulated:.4f}" if isinstance(simulated, (int, float)) else "-"
        d = f"{delta:.4f}" if isinstance(delta, (int, float)) else "-"
        p = f"{pct:.2f}" if isinstance(pct, (int, float)) else "-"
        print(f"{name:<24}{a:>14}{b:>14}{d:>14}{p:>10}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("actual", type=Path, help="Path to vLLM (or real run) metrics JSON/JSONL")
    ap.add_argument("config", type=Path, help="Simulator config YAML to evaluate")
    ap.add_argument("--workdir", type=Path, help="Directory for generated trace and artifacts")
    ap.add_argument("--emit-sim-output", action="store_true", help="Print full simulator report")
    ap.add_argument("--report-json", type=Path, help="Optional path to write the comparison report JSON")
    args = ap.parse_args()

    actual_run: ActualRun = load_actual_run(args.actual)
    report = compare_runs(actual_run, args.config, workdir=args.workdir, emit_output=args.emit_sim_output)

    _print_report(report)

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nReport written to {args.report_json}")


if __name__ == "__main__":
    main()
