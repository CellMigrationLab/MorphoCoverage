"""Typed identity of FOVs, blocks, conditions and repeats (private; issues 283, 319).

An identity value is a scalar that tells one thing from another: integer ``1`` and text
``"1"`` are different values, a numpy scalar is the Python value it holds. Display text
(``str``, ``repr``) is never the identity: two distinct values can print alike, and an order
or a grouping keyed by their text would fall back to the table's row order (or a hash seed)
for them. ``identity_key`` is the typed key: sortable without the values being mutually
orderable, hashable, and equal only for equal values of equal type.

``canonical_order`` and ``canonical_codes`` order identities by their text first and their type
second, so an ordinary column (all text, or all numbers) keeps the order it always had, and
mixed ones are ordered by something other than row order.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd

IdentityKey = tuple[str, str]


def identity_key(value: Any) -> IdentityKey:
    """``(text, type)`` of one identity value; a numpy scalar keys as the Python value it holds.
    ``None`` is the explicit "no such level" (an analysis without a repeat column has repeat
    ``None``); any other missing value is refused (an identity is never missing)."""
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, Hashable):
        raise TypeError(f"{value!r} cannot identify a field, block or group: it is not hashable")
    if (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        raise ValueError("an identity value is never missing")
    return str(value), type(value).__name__


def row_keys(table: pd.DataFrame, columns: Sequence[str]) -> list[tuple[IdentityKey, ...]]:
    """The typed key of every row over ``columns`` (a categorical column by its values)."""
    columns = [table[c].astype(object).tolist() for c in columns]
    return [tuple(identity_key(v) for v in row) for row in zip(*columns, strict=True)]


def canonical_order(table: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """The rows of ``table`` in a canonical order of their identity over ``columns``: by the
    typed key, so neither the row order nor a categorical column's category order moves it."""
    keys = row_keys(table, columns)
    return np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)


def _natural(value: Any) -> tuple[int, Any, str, str]:
    """A sort key that never compares a number with text: numbers by value, then text by
    itself, then anything else by its text; the key's text and type break every tie."""
    value = value.item() if isinstance(value, np.generic) else value
    key = identity_key(value)
    if isinstance(value, bool):
        return 2, key[0], *key
    if isinstance(value, int | float):
        return 0, value, *key
    if isinstance(value, str):
        return 1, value, *key
    return 2, key[0], *key


def canonical_codes(values: pd.Series) -> tuple[np.ndarray, list[IdentityKey]]:
    """One integer per row for the labels of ``values``, numbered in the natural order of the
    distinct labels (numbers by value, then text; ties by typed key), and those keys. Labels
    need not be mutually orderable (``1`` and ``"a"`` are fine); the numbering never depends
    on the row order."""
    labels = values.astype(object).tolist()
    distinct = sorted({identity_key(v): v for v in labels}.items(), key=lambda kv: _natural(kv[1]))
    code = {key: i for i, (key, _) in enumerate(distinct)}
    keys = [identity_key(v) for v in labels]
    return np.array([code[k] for k in keys], dtype=np.int64), [k for k, _ in distinct]


def canonical_row_codes(table: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """One integer per row for the typed key over ``columns``, numbered in the canonical order
    of the distinct keys (one field, one code, whatever the row order)."""
    keys = row_keys(table, columns)
    code = {key: i for i, key in enumerate(sorted(set(keys)))}
    return np.array([code[key] for key in keys], dtype=np.int64)


def display(keys: Sequence[IdentityKey]) -> dict[IdentityKey, str]:
    """A label per key for text: the value's text, with its type added when two distinct keys
    would print alike (``1 (int)`` and ``1 (str)``), so a sentence never merges them."""
    texts = [text for text, _ in keys]
    return {key: key[0] if texts.count(key[0]) == 1 else f"{key[0]} ({key[1]})" for key in keys}


def key_labels(table: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """One text label per row for messages, the whole key written out (``condition=a,
    biological_repeat=R1, position_id=f3``): a FOV identifier alone repeats across repeats, so
    the last key column never names a field. Rendering only; matching uses ``row_keys``."""
    return [
        ", ".join(f"{column}={text}" for column, (text, _) in zip(columns, row, strict=True))
        for row in row_keys(table, columns)
    ]


__all__ = [
    "IdentityKey",
    "canonical_codes",
    "canonical_order",
    "canonical_row_codes",
    "display",
    "identity_key",
    "key_labels",
    "row_keys",
]
