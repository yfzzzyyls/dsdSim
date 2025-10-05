"""Utility helpers for experiment runner."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, MutableMapping


def deep_update(base: MutableMapping, overrides: Mapping) -> MutableMapping:
    """Recursively merge ``overrides`` into ``base`` and return the result."""

    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = deepcopy(value)
    return base
