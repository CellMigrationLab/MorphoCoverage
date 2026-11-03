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

from ._seeds import plain

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


def typed_frame(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``table[columns]`` with each value as its typed key in one text (``"1\x1eint"``), for
    matching rows of two tables by text without joining ``1`` and ``"1"`` (#420); never shown.
    Missing values are refused, as identities are never missing."""
    return pd.DataFrame(
        {c: ["\x1e".join(identity_key(v)) for v in table[c].tolist()] for c in columns},
        index=table.index,
    )


def typed_labels(values: pd.Series) -> np.ndarray:
    """One text per row of ``values`` that tells typed identities apart: the value's text, with
    its type added when two distinct values of ``values`` print alike (``display``), so
    ``1`` and ``"1"`` are ``"1 (int)"`` and ``"1 (str)"``; a missing value is its text. Equal
    to ``values.astype(str)`` whenever no two distinct values print alike (#420)."""
    raw = values.astype(object).tolist()
    keys = [None if _missing(v) else identity_key(v) for v in raw]
    shown = display(list(dict.fromkeys(k for k in keys if k is not None)))
    return np.array([str(v) if k is None else shown[k] for v, k in zip(raw, keys, strict=True)])


def _missing(value: Any) -> bool:
    """Whether ``value`` is a missing value (``None``, ``NaN``, ``pd.NA``, ``NaT``)."""
    return value is None or value is pd.NA or bool(pd.isna(value)) if np.ndim(value) == 0 else False


def key_labels(table: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """One text label per row for messages, the whole key written out (``condition=a,
    biological_repeat=R1, position_id=f3``): a FOV identifier alone repeats across repeats, so
    the last key column never names a field. Rendering only; matching uses ``row_keys``."""
    return [
        ", ".join(f"{column}={text}" for column, (text, _) in zip(columns, row, strict=True))
        for row in row_keys(table, columns)
    ]


def ambiguous_values(values: Sequence[Any]) -> list[tuple[Any, ...]]:
    """The groups of distinct identity values in ``values`` that Python equality would merge.

    ``True == 1 == 1.0`` and ``0.0 == -0.0``, and they hash alike, so a set, a dict, a merge
    or a groupby joins them although their typed keys (``identity_key``) differ (#420): an
    observation of FOV ``True`` was placed in FOV ``1``. Only numbers collide this way:
    ``1`` and ``"1"`` are never equal and stay two values, told apart by type wherever they
    are keyed (issues 283, 284). Each group lists its values in order of first appearance;
    empty when no two distinct values are equal. Missing values are left out here."""
    merged: dict[Any, dict[IdentityKey, Any]] = {}
    for value in map(plain, values):
        if isinstance(value, bool | int | float) and not (
            isinstance(value, float) and np.isnan(value)
        ):
            merged.setdefault(value, {}).setdefault(identity_key(value), value)
    return [tuple(kinds.values()) for kinds in merged.values() if len(kinds) > 1]


# Inferred kinds whose distinct values are all of one type that no other value equals.
_ONE_TYPE = {"string", "integer", "boolean", "bytes"}
# The separator of joined keys: never inside an identity value, so no two keys join alike.
KEY_SEPARATOR = "\x1f"


def _ambiguity(present: Sequence[pd.Series]) -> tuple[Any, ...] | None:
    """The first ambiguous group of the values of one column over several tables, or a value
    holding ``KEY_SEPARATOR`` (alone), or ``None``."""
    # pandas' C type inference first: values of one such type never collide.
    kinds = {pd.api.types.infer_dtype(values, skipna=True) for values in present} - {"empty"}
    if len(kinds) <= 1 and kinds <= _ONE_TYPE:
        if kinds != {"string"}:
            return None
        for series in present:
            held = series[series.str.contains(KEY_SEPARATOR, regex=False, na=False)]
            if len(held):
                return (held.iloc[0],)
        return None
    if kinds == {"floating"}:  # only 0.0 and -0.0 collide: a vectorised check
        return _signed_zeros(present)
    values = [v for series in present for v in series.astype(object).tolist()]
    found = [(v,) for v in values if isinstance(v, str) and KEY_SEPARATOR in v]
    groups = ambiguous_values(values)
    return groups[0] if groups else found[0] if found else None


def _signed_zeros(present: Sequence[pd.Series]) -> tuple[float, float] | None:
    """``(0.0, -0.0)`` in order of first appearance when the float values of ``present``
    hold both zeros, which equality merges; else ``None``."""
    values = np.concatenate([s.to_numpy(dtype=float, na_value=np.nan) for s in present])
    signs = np.signbit(values[values == 0])
    if signs.any() and not signs.all():
        first = float(values[values == 0][0])
        return first, -first
    return None


def check_one_type(tables: Sequence[tuple[str, pd.DataFrame]], columns: Sequence[str]) -> None:
    """Refuse an identity column whose distinct values equality would merge (#420).

    ``columns`` name the FOV identity (the field identifier and its group columns); each is
    read over every table of ``tables`` (``(name, table)`` pairs: the observations and the
    FOV table) together, since a value of the one is matched with the other's. A value
    holding the control character ``KEY_SEPARATOR`` is refused too, so no two composite keys
    joined by it read alike. ``ValueError`` naming the column, the values and their types."""
    for column in dict.fromkeys(columns):
        present = [table[column] for _, table in tables if column in table]
        found = _ambiguity(present)
        if found is None:
            continue
        where = " and ".join(name for name, table in tables if column in table)
        if len(found) == 1:
            raise ValueError(
                f"{where} column {column!r} holds the identity value {found[0]!r}, which "
                "contains the control character \\x1f; remove it from the identifier"
            )
        shown = ", ".join(f"{v!r} ({type(v).__name__})" for v in found)
        raise ValueError(
            f"{where} column {column!r} holds distinct identity values that are equal as "
            f"numbers ({shown}): matching would merge them into one field or group. Write "
            "each identifier with one type in every table (for example all integers, or all "
            "text)"
        )


__all__ = [
    "KEY_SEPARATOR",
    "IdentityKey",
    "ambiguous_values",
    "canonical_codes",
    "canonical_order",
    "canonical_row_codes",
    "check_one_type",
    "display",
    "identity_key",
    "key_labels",
    "row_keys",
    "typed_frame",
    "typed_labels",
]
