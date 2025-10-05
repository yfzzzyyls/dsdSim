"""Experiment manifest dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunManifest:
    name: str
    base_config: Path
    overrides: Dict[str, object] = field(default_factory=dict)
    output_config: Optional[Path] = None


@dataclass
class ExperimentManifest:
    name: str
    description: str
    runs: List[RunManifest]

    @staticmethod
    def from_dict(raw: Dict[str, object], manifest_path: Path) -> "ExperimentManifest":
        exp = raw.get("experiment")
        if not isinstance(exp, dict):
            raise ValueError(f"Invalid manifest {manifest_path}: missing 'experiment' section")

        name = str(exp.get("name")) if exp.get("name") else manifest_path.stem
        description = str(exp.get("description", ""))
        run_entries = exp.get("runs")
        if not isinstance(run_entries, list) or not run_entries:
            raise ValueError(f"Experiment {name} has no runs defined")

        runs: List[RunManifest] = []
        for idx, entry in enumerate(run_entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Run entry #{idx} in {name} is not a mapping")
            run_name = str(entry.get("name")) if entry.get("name") else f"run_{idx:02d}"
            base_cfg = entry.get("config") or entry.get("base_config")
            if not base_cfg:
                raise ValueError(f"Run '{run_name}' in {name} needs a 'config' path")
            base_path = (manifest_path.parent / Path(str(base_cfg))).resolve()
            overrides = entry.get("overrides") or {}
            if not isinstance(overrides, dict):
                raise ValueError(f"Run '{run_name}' overrides must be a mapping")
            runs.append(RunManifest(name=run_name, base_config=base_path, overrides=overrides))

        return ExperimentManifest(name=name, description=description, runs=runs)
