"""Parsing utilities for patient2gene package."""

from __future__ import annotations

import typing as t


def flatten(nested_items: t.Iterable[t.Iterable[t.Any]]) -> t.Iterable[t.Any]:
    """Flatten nested iterable."""
    return [item for nested_item in nested_items for item in nested_item]
