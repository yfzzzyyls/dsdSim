"""Minimal wandb stub for offline Vidur integration."""

from __future__ import annotations

class _DummyRun:
    def log(self, *args, **kwargs):
        return None

    def finish(self) -> None:
        return None


def init(*args, **kwargs):  # type: ignore[override]
    return _DummyRun()


class Table:  # pragma: no cover - stub
    def __init__(self, *args, **kwargs) -> None:
        self.data = []

    def add_data(self, *args, **kwargs) -> None:
        self.data.append((args, kwargs))


class Image:  # pragma: no cover - stub
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def log(*args, **kwargs) -> None:  # type: ignore[override]
    return None


def finish() -> None:  # type: ignore[override]
    return None
