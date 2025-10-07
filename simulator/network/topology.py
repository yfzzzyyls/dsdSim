"""Generate network-aware latency estimates using NetworkX topologies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None  # type: ignore


class NetworkModelError(RuntimeError):
    """Raised when a network topology specification is invalid."""


@dataclass(frozen=True)
class _Device:
    id: str
    role: str


def build_latency_lookup(
    *,
    drafts: Sequence[Mapping[str, object]],
    targets: Sequence[Mapping[str, object]],
    spec: Mapping[str, object],
) -> Dict[Tuple[str, str], float]:
    """Return draft→target latency (ms) using the configured network model.

    Parameters
    ----------
    drafts / targets: iterable of device dictionaries containing at least ``id``
        and ``role``.
    spec: mapping describing the chosen topology. ``type`` selects which
        NetworkX structure to instantiate. Currently supported types:
        - ``clos`` / ``two_level_clos``: two-tier leaf/spine fabric.
        - ``complete``: constant-latency fully connected fabric.

    Returns
    -------
    dict[(draft_id, target_id)] -> latency_ms
    """

    if nx is None:
        raise NetworkModelError(
            "networkx is required for network_model but is not installed. "
            "Install the 'networkx' package or disable the network model."
        )

    model_type = str(spec.get("type", "clos")).lower()
    draft_devices = [_Device(id=str(d["id"]), role="draft") for d in drafts]
    target_devices = [_Device(id=str(t["id"]), role="target") for t in targets]

    if not draft_devices or not target_devices:
        raise NetworkModelError("network_model requires at least one draft and one target")

    if model_type in {"clos", "two_level_clos", "leaf_spine"}:
        latency = _build_two_level_clos(draft_devices, target_devices, spec)
    elif model_type in {"complete", "fully_connected", "static"}:
        latency = _build_complete_graph(draft_devices, target_devices, spec)
    else:
        raise NetworkModelError(f"Unsupported network_model.type '{model_type}'")

    return latency


def _build_complete_graph(
    drafts: Sequence[_Device],
    targets: Sequence[_Device],
    spec: Mapping[str, object],
) -> Dict[Tuple[str, str], float]:
    base_latency = float(spec.get("latency_ms", 20.0))
    per_hop = float(spec.get("per_hop_ms", 0.0))

    graph = nx.Graph()
    for device in list(drafts) + list(targets):
        graph.add_node(device.id)

    for i in range(len(graph.nodes)):
        for j in range(i + 1, len(graph.nodes)):
            u = list(graph.nodes)[i]
            v = list(graph.nodes)[j]
            graph.add_edge(u, v, weight=base_latency + per_hop)

    return _pairwise_latencies(graph, drafts, targets)


def _build_two_level_clos(
    drafts: Sequence[_Device],
    targets: Sequence[_Device],
    spec: Mapping[str, object],
) -> Dict[Tuple[str, str], float]:
    hop_latency = float(spec.get("hop_latency_ms", spec.get("leaf_spine_latency_ms", 0.35)))
    edge_latency = float(spec.get("device_edge_latency_ms", spec.get("edge_latency_ms", 0.15)))
    spine_count = max(1, int(spec.get("spine_count", 4)))
    leaf_count = max(1, int(spec.get("leaf_count", max(2, math.ceil((len(drafts) + len(targets)) / 4)))))

    graph = nx.Graph()

    spines = [f"_spine_{i}" for i in range(spine_count)]
    leaves = [f"_leaf_{i}" for i in range(leaf_count)]

    for spine in spines:
        graph.add_node(spine)
    for leaf in leaves:
        graph.add_node(leaf)
        for spine in spines:
            graph.add_edge(leaf, spine, weight=hop_latency)

    # Round-robin placement keeps devices distributed across leaves.
    all_devices = list(targets) + list(drafts)
    assignment: MutableMapping[str, str] = {}
    for idx, device in enumerate(all_devices):
        leaf = leaves[idx % leaf_count]
        assignment[device.id] = leaf
        graph.add_node(device.id)
        graph.add_edge(device.id, leaf, weight=edge_latency)

    return _pairwise_latencies(graph, drafts, targets)


def _pairwise_latencies(
    graph: "nx.Graph",
    drafts: Sequence[_Device],
    targets: Sequence[_Device],
) -> Dict[Tuple[str, str], float]:
    latencies: Dict[Tuple[str, str], float] = {}
    for draft in drafts:
        if draft.id not in graph:
            raise NetworkModelError(f"Draft '{draft.id}' missing from topology graph")
        for target in targets:
            if target.id not in graph:
                raise NetworkModelError(f"Target '{target.id}' missing from topology graph")
            try:
                dist = nx.shortest_path_length(graph, draft.id, target.id, weight="weight")
            except nx.NetworkXNoPath as exc:
                raise NetworkModelError(
                    f"No path between draft '{draft.id}' and target '{target.id}' in network_model"
                ) from exc
            latencies[(draft.id, target.id)] = float(dist)
    return latencies
