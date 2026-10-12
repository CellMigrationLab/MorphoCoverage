"""Groups in the report's text: counted and merged by identity, never by rendered label.

Two distinct groups can print alike (the integer ``1`` and the text ``"1"``). The text layer
therefore keys every count and deduplication on the typed identity of the group
(``_identity``) and writes labels only afterwards, adding the value's type when two distinct
groups would otherwise read the same (issue 284).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from .._identity import IdentityKey, identity_key
from .._style import group_label

GroupIdentity = tuple[IdentityKey, ...]


def group_key(row: Mapping[str, Any], keys: Sequence[str]) -> GroupIdentity:
    """The typed identity of a row's group over ``keys``."""
    return tuple(identity_key(row[key]) for key in keys)


def count_groups(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int:
    """How many distinct groups the rows hold, by identity."""
    return len({group_key(row, keys) for row in rows})


def _typed(identity: GroupIdentity) -> str:
    """``int`` or ``int, str``: the types of a group's values, to tell alike labels apart."""
    return ", ".join(kind for _, kind in identity)


def namer(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Callable[[Mapping[str, Any]], str]:
    """A function writing a row's group label, the type added only where two groups of
    ``rows`` would print alike. Rows outside ``rows`` are written plainly."""
    labels: dict[GroupIdentity, str] = {}
    for row in rows:
        labels.setdefault(group_key(row, keys), group_label({key: row[key] for key in keys}))
    texts = list(labels.values())
    written = {
        identity: label if texts.count(label) == 1 else f"{label} ({_typed(identity)})"
        for identity, label in labels.items()
    }

    def name(row: Mapping[str, Any]) -> str:
        identity = group_key(row, keys)
        return written.get(identity) or group_label({key: row[key] for key in keys})

    return name


def distinct_labels(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[str]:
    """One label per distinct group of ``rows``, in order of first appearance."""
    name = namer(rows, keys)
    seen: dict[GroupIdentity, str] = {}
    for row in rows:
        seen.setdefault(group_key(row, keys), name(row))
    return list(seen.values())


def skipped_label(skipped: Mapping[str, Any], every: Sequence[Mapping[str, Any]]) -> str:
    """A skipped group's label (recorded as ``label`` and typed ``identity`` by the check),
    with its types added when another skipped group prints alike."""
    label = skipped.get("label") or skipped["group"]
    alike = [o for o in every if (o.get("label") or o["group"]) == label]
    distinct = {tuple(map(tuple, o.get("identity", ()))) for o in alike}
    if len(distinct) < 2:
        return str(label)
    return f"{label} ({_typed(tuple(map(tuple, skipped['identity'])))})"


def present(value: Any) -> bool:
    """Whether a table cell holds a value (not None or missing)."""
    return value is not None and not pd.isna(value)


def row_label(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    """A row's group over ``keys``, by its values (``_style.group_label``)."""
    return group_label({key: row[key] for key in keys})


def selector_values(selector: Mapping[str, Any] | None) -> str:
    """``MOCK`` for ``{"condition": ["MOCK"]}``: the values a selector accepts; empty when
    nothing was declared."""
    if not selector:
        return ""
    parts = []
    for value in selector.values():
        values = value if isinstance(value, list | tuple | set) else [value]
        parts.append("/".join(str(v) for v in values))
    return " · ".join(parts)
