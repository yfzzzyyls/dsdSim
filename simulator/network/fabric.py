"""Lightweight network fabric for draft⇄target communication."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import simpy


def _mbps_to_bytes_per_ms(mbps: float) -> float:
    """Convert megabits/second to bytes/millisecond."""
    if mbps <= 0:
        return 0.0
    # 1 megabit = 1_000_000 bits. Divide by 8 for bytes, divide by 1000 for milliseconds.
    return mbps * 1_000_000 / 8.0 / 1000.0


@dataclass
class LinkStats:
    transfers: int = 0
    queue_wait_ms: float = 0.0
    transfer_ms: float = 0.0

    def record(self, queue_wait: float, transfer_time: float) -> None:
        self.transfers += 1
        self.queue_wait_ms += max(0.0, queue_wait)
        self.transfer_ms += max(0.0, transfer_time)


@dataclass
class Link:
    env: simpy.Environment
    key: Tuple[str, str]
    base_latency_ms: float
    bandwidth_bytes_per_ms: float
    jitter_pct: float
    capacity: int
    stats: LinkStats = field(default_factory=LinkStats)

    def __post_init__(self) -> None:
        self.base_latency_ms = max(0.0, float(self.base_latency_ms))
        self.bandwidth_bytes_per_ms = max(0.0, float(self.bandwidth_bytes_per_ms))
        self.jitter_pct = max(0.0, float(self.jitter_pct))
        self.capacity = max(1, int(self.capacity))
        self._resource = simpy.Resource(self.env, capacity=self.capacity)

    def request(self):
        return self._resource.request()


class NetworkFabric:
    """Minimal SimPy-based network fabric shared by drafts."""

    def __init__(self, env: simpy.Environment, config: Optional[Dict[str, object]] = None) -> None:
        self.env = env
        config = dict(config or {})
        self.default_bandwidth_mbps = float(config.get("bandwidth_mbps", 0.0))
        self.default_capacity = int(config.get("link_capacity", 1) or 1)
        self.default_jitter_pct = float(config.get("jitter_pct", 0.0) or 0.0)
        self.default_latency_ms = float(config.get("default_latency_ms", 0.0) or 0.0)
        seed = config.get("seed")
        self._rng = random.Random(seed)
        self.bytes_per_token = float(config.get("bytes_per_token", 2.0) or 0.0)
        self.prefill_overhead_bytes = float(config.get("prefill_overhead_bytes", 512.0) or 0.0)
        self.decode_overhead_bytes = float(config.get("decode_overhead_bytes", 128.0) or 0.0)
        self.response_overhead_bytes = float(config.get("response_overhead_bytes", 256.0) or 0.0)
        self.links: Dict[Tuple[str, str], Link] = {}

    def register_link(
        self,
        source: str,
        target: str,
        *,
        base_latency_ms: float,
        bandwidth_mbps: Optional[float] = None,
        jitter_pct: Optional[float] = None,
        capacity: Optional[int] = None,
    ) -> Tuple[str, str]:
        key = (str(source), str(target))
        bw = (
            _mbps_to_bytes_per_ms(float(bandwidth_mbps))
            if bandwidth_mbps is not None
            else _mbps_to_bytes_per_ms(self.default_bandwidth_mbps)
        )
        jitter = self.default_jitter_pct if jitter_pct is None else float(jitter_pct)
        cap = self.default_capacity if capacity is None else int(capacity)
        link = Link(
            env=self.env,
            key=key,
            base_latency_ms=float(base_latency_ms),
            bandwidth_bytes_per_ms=bw,
            jitter_pct=jitter,
            capacity=cap,
        )
        self.links[key] = link
        return key

    def transfer(
        self,
        source: str,
        target: str,
        *,
        payload_bytes: float = 0.0,
        link_key: Optional[Tuple[str, str]] = None,
        fallback_latency_ms: float = 0.0,
    ):
        key = link_key if link_key is not None else (str(source), str(target))
        link = self.links.get(key)
        if link is None:
            # Fall back to a simple timeout with the provided latency.
            return self.env.timeout(max(0.0, float(fallback_latency_ms or self.default_latency_ms)))
        return self.env.process(self._transfer_proc(link, payload_bytes))

    def _transfer_proc(self, link: Link, payload_bytes: float):
        start_wait = self.env.now
        with link.request() as req:
            yield req
            queue_wait = self.env.now - start_wait
            transfer_ms = link.base_latency_ms
            if link.bandwidth_bytes_per_ms > 0.0 and payload_bytes > 0.0:
                serialization_ms = payload_bytes / link.bandwidth_bytes_per_ms
                transfer_ms += max(0.0, serialization_ms)
            jitter = link.jitter_pct
            if jitter > 0.0:
                jitter_scale = 1.0 + self._rng.uniform(-jitter, jitter)
                transfer_ms = max(0.0, transfer_ms * jitter_scale)
            link.stats.record(queue_wait, transfer_ms)
            yield self.env.timeout(transfer_ms)

    def link_metrics(self) -> Dict[Tuple[str, str], LinkStats]:
        return {key: link.stats for key, link in self.links.items()}
