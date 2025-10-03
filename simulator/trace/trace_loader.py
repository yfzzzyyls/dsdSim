"""Streaming utilities for reading trace files into TraceRecord objects."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, TextIO, Union

from .types import TraceParseError, TraceRecord, TraceRecordDict

TraceIterable = Iterable[Union[TraceRecordDict, TraceRecord]]
TraceSource = Union[str, Path, TextIO, TraceIterable]


def iter_trace_records(
    source: TraceSource,
    defaults: Optional[Mapping[str, object]] = None,
    *,
    strict: bool = True,
) -> Iterator[TraceRecord]:
    """Yield trace records from a JSONL/JSONL.GZ path or iterable."""

    defaults = defaults or {}

    if isinstance(source, (str, Path)):
        yield from _iter_from_path(Path(source), defaults=defaults, strict=strict)
    elif isinstance(source, io.TextIOBase):
        yield from _iter_from_stream(source, defaults=defaults, strict=strict)
    else:
        for row in source:
            try:
                if isinstance(row, TraceRecord):
                    base_payload: Mapping[str, object] = row.to_dict()
                elif isinstance(row, Mapping):
                    base_payload = row
                else:
                    raise TraceParseError(f"unsupported trace row type: {type(row)!r}")
                payload = _merge_defaults(base_payload, defaults)
                yield TraceRecord.from_dict(payload)
            except TraceParseError:
                if strict:
                    raise
                continue


def load_trace(
    source: TraceSource,
    defaults: Optional[Mapping[str, object]] = None,
    *,
    strict: bool = True,
) -> list[TraceRecord]:
    """Materialise all records from ``source`` into a list."""

    return list(iter_trace_records(source, defaults=defaults, strict=strict))


def _iter_from_path(
    path: Path,
    *,
    defaults: Mapping[str, object],
    strict: bool,
) -> Iterator[TraceRecord]:
    with _open_text(path) as stream:
        yield from _iter_from_stream(stream, defaults=defaults, strict=strict)


def _iter_from_stream(
    stream: TextIO,
    *,
    defaults: Mapping[str, object],
    strict: bool,
) -> Iterator[TraceRecord]:
    for line_no, line in enumerate(stream, start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if strict:
                raise TraceParseError(f"invalid JSON on line {line_no}: {exc.msg}") from exc
            continue

        try:
            record_map = _merge_defaults(payload, defaults)
            yield TraceRecord.from_dict(record_map)
        except TraceParseError:
            if strict:
                raise
            continue


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _merge_defaults(
    payload: Mapping[str, object],
    defaults: Mapping[str, object],
) -> TraceRecordDict:
    merged: dict[str, object] = dict(payload)
    for key, value in defaults.items():
        if key == "metadata":
            base_meta = value if isinstance(value, Mapping) else {}
            payload_meta = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
            meta: dict[str, object] = {}
            meta.update(base_meta)  # type: ignore[arg-type]
            meta.update(payload_meta)  # type: ignore[arg-type]
            if meta:
                merged["metadata"] = meta
        else:
            merged.setdefault(key, value)
    return merged  # type: ignore[return-value]
