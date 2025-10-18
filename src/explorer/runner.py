"""CLI for running experiment manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

from .manifest import ExperimentManifest, RunManifest
from .utils import deep_update


def _discover_manifests(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.yaml"))


def _load_manifest(path: Path) -> ExperimentManifest:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return ExperimentManifest.from_dict(data, path)


def _apply_overrides(base_path: Path, overrides: dict, output_dir: Path, run_name: str) -> Path:
    with base_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if overrides:
        deep_update(config, overrides)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_name}.yaml"
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return out_path


def _run_sim(sim_path: Path, config_path: Path, log_dir: Path, dry_run: bool) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(sim_path), "--config", str(config_path)]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return 0

    print("EXEC:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=sim_path.parent, capture_output=True, text=True)
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    (log_dir / "returncode.json").write_text(json.dumps({"returncode": proc.returncode}, indent=2))
    if proc.returncode != 0:
        print(f"Run failed (rc={proc.returncode}) for {config_path}", file=sys.stderr)
    return proc.returncode


def run_experiment(manifest_path: Path, *, output_root: Path, dry_run: bool) -> None:
    manifest = _load_manifest(manifest_path)
    print(f"Experiment: {manifest.name}\nDescription: {manifest.description}\nRuns: {len(manifest.runs)}")

    sim_script = (manifest_path.parents[2] / "sim.py").resolve()
    if not sim_script.exists():
        raise FileNotFoundError(f"Cannot locate sim.py at {sim_script}")

    exp_root = output_root / manifest.name
    exp_root.mkdir(parents=True, exist_ok=True)

    for run in manifest.runs:
        _execute_run(run, sim_script, exp_root, dry_run)


def _execute_run(run: RunManifest, sim_script: Path, exp_root: Path, dry_run: bool) -> None:
    if not run.base_config.exists():
        raise FileNotFoundError(f"Base config not found: {run.base_config}")

    run_output = exp_root / run.name
    config_output_dir = run_output / "configs"
    generated_config = _apply_overrides(run.base_config, run.overrides, config_output_dir, run.name)
    run.output_config = generated_config
    rc = _run_sim(sim_script, generated_config, run_output / "logs", dry_run)
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"Run {run.name}: {status}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explorer experiment runner")
    parser.add_argument("--experiment", "-e", help="Name of the experiment manifest to run")
    parser.add_argument("--manifests", default="explorer/experiments", help="Directory with experiment manifests")
    parser.add_argument("--output", default="explorer/output", help="Directory for generated configs and logs")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing sim.py")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifests_dir = Path(args.manifests).resolve()
    if not manifests_dir.exists():
        print(f"Manifests directory not found: {manifests_dir}", file=sys.stderr)
        return 1

    available = list(_discover_manifests(manifests_dir))
    if args.list or not args.experiment:
        print("Available experiments:")
        for path in available:
            print("  ", path.stem)
        return 0

    match = None
    for path in available:
        if path.stem == args.experiment:
            match = path
            break
    if match is None:
        print(f"Experiment '{args.experiment}' not found in {manifests_dir}", file=sys.stderr)
        return 2

    run_experiment(match, output_root=Path(args.output).resolve(), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
