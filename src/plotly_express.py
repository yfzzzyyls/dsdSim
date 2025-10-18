"""Minimal plotly express stub for Vidur dependency."""

from __future__ import annotations

class _Figure:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def write_html(self, *args, **kwargs):  # pragma: no cover - stub
        return None

    def to_html(self, *args, **kwargs):  # pragma: no cover - stub
        return ""


def bar(*args, **kwargs):  # type: ignore[override]
    return _Figure(*args, **kwargs)
