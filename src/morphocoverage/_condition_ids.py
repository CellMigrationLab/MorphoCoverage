"""Conditions are named by their ``condition_id``, never by folder label (MC-1, #628).

The MorphoBricks FOV key keeps the source folder label of each condition in ``condition``; the
condition ID of the description is added beside it as ``condition_id`` (MorphoBricks-Spec
experiment-v1 section 3.4). ``checked_condition_ids`` reads the folder -> ID map, refusing a
field without an ID, a folder with two IDs, an ID naming two folders, and tables that disagree.
A declared condition (a control, a positive control, an experiment mapping's keys, a contrast
table's conditions) names a condition by its ID: a folder label that is not an ID is refused
with the ID to use (``refuse_folder_names``), and a name that matches no condition at all is
refused listing the conditions (``refuse_unmatched``), so a selector never selects nothing
without a word.

Split out of ``_repeat_groups``, which re-exports every name here; shared-private: used by
``_repeat_groups`` and, through it, by the paired analyses and the workflows.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from .metadata import CONDITION_COLUMN, CONDITION_ID_COLUMN, empty_values

# Why conditions cannot be named by their IDs, by code (``PairingRefused.code``).
ID_EMPTY = "condition_id_empty"
ID_SPLIT = "condition_id_split"
ID_MERGED = "condition_id_merged"
ID_DISAGREES = "condition_id_disagrees"
ID_NO_SOURCE = "condition_id_without_condition"
FOLDER_CONTROL = "condition_named_by_folder_label"
NOT_FOUND = "condition_not_found"


class PairingRefused(ValueError):
    """The biological repeats cannot be paired across conditions; ``code`` names the reason
    (one of ``CODES``) and the message says what to fix."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _listed(values: Sequence[Any]) -> str:
    """``'Day1', 'Day2'``, in a stable order."""
    return ", ".join(repr(v) for v in sorted(values, key=str))


def checked_condition_ids(
    objects: pd.DataFrame, fovs: pd.DataFrame, condition_col: str = CONDITION_COLUMN
) -> dict[Any, Any]:
    """Each condition folder's ``condition_id`` (every folder), empty without the column or
    with a condition column of the caller's own; refused (``PairingRefused``) when a field has
    none, a folder has two, an ID names two folders or the observation table disagrees."""
    if condition_col != CONDITION_COLUMN:
        return {}
    source = fovs
    if CONDITION_ID_COLUMN not in fovs.columns and CONDITION_ID_COLUMN in objects.columns:
        source = objects  # the observation table may carry it alone
    if CONDITION_ID_COLUMN not in source.columns:
        return {}
    if condition_col not in source.columns:
        raise PairingRefused(
            ID_NO_SOURCE,
            f"the tables have a {CONDITION_ID_COLUMN!r} column but no {condition_col!r} column: "
            "the FOV key keeps the source folder label of each condition beside its ID; add "
            f"the {condition_col!r} column",
        )
    empty = empty_values(source[CONDITION_ID_COLUMN])
    if empty.any():
        first = source.loc[empty, condition_col].iloc[0]
        raise PairingRefused(
            ID_EMPTY,
            f"{int(empty.sum())} of the {len(source)} rows have no {CONDITION_ID_COLUMN!r} "
            f"value (the first: condition {first!r}), so the condition is unknown: fill it in "
            "for every field",
        )
    ids = _one_to_one(source[[condition_col, CONDITION_ID_COLUMN]].drop_duplicates())
    if source is fovs and {condition_col, CONDITION_ID_COLUMN} <= set(objects.columns):
        held = objects[[condition_col, CONDITION_ID_COLUMN]].drop_duplicates()
        for folder, cid in held.itertuples(index=False):
            if folder in ids and str(ids[folder]) != str(cid):
                raise PairingRefused(
                    ID_DISAGREES,
                    f"the observation table gives the condition folder {folder!r} the ID "
                    f"{cid!r} and the FOV table {ids[folder]!r}: give both tables the same "
                    f"{CONDITION_ID_COLUMN!r}",
                )
    return ids


def _one_to_one(pairs: pd.DataFrame) -> dict[Any, Any]:
    """The folder -> ID pairs, refused unless each folder has one ID and each ID one folder."""
    ids: dict[Any, list[Any]] = {}
    folders: dict[Any, list[Any]] = {}
    for folder, cid in pairs.itertuples(index=False):
        ids.setdefault(folder, []).append(cid)
        folders.setdefault(str(cid), []).append(folder)
    for folder, found in ids.items():
        if len(found) > 1:
            raise PairingRefused(
                ID_SPLIT,
                f"the condition folder {folder!r} has {len(found)} IDs ({_listed(found)}): a "
                f"folder is one condition; give it one {CONDITION_ID_COLUMN!r}",
            )
    for cid, found in folders.items():
        if len(found) > 1:
            raise PairingRefused(
                ID_MERGED,
                f"the condition ID {cid!r} names {len(found)} folders ({_listed(found)}): each "
                f"condition of the description has one source folder; give each folder its "
                f"own {CONDITION_ID_COLUMN!r}",
            )
    return {folder: found[0] for folder, found in ids.items()}


def declared_conditions(control: Any, condition_col: str = CONDITION_COLUMN) -> list[Any]:
    """The condition names a control declaration holds under ``condition_col``: the value of
    ``{"condition": "ctrl"}``, and every key and value of the forms of several controls
    (``{"condition": {control: {treatment: condition}}}``, ``[conditions]``)."""
    if not isinstance(control, Mapping) or condition_col not in control:
        return []
    return list(_names_in(control[condition_col]))


def _names_in(value: Any) -> Iterable[Any]:
    """The hashable leaves and keys of a nested declaration."""
    if isinstance(value, Mapping):
        for key, inner in value.items():
            yield from _names_in(key)
            yield from _names_in(inner)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for inner in value:
            yield from _names_in(inner)
    elif isinstance(value, Hashable) and value is not None:
        yield value


def refuse_folder_names(names: Iterable[Any], ids: Mapping[Any, Any]) -> None:
    """Refuse a declared condition that is a folder label, not an ID (``FOLDER_CONTROL``),
    naming the ID to use; an ID that is also another folder's label is the ID."""
    known = {str(cid) for cid in ids.values()}
    for name in names:
        if str(name) in known:
            continue
        cid = next((c for folder, c in ids.items() if str(folder) == str(name)), None)
        if cid is not None:
            raise PairingRefused(
                FOLDER_CONTROL,
                f"{name!r} is a condition folder label: conditions are named by their ID "
                f"({CONDITION_ID_COLUMN!r}), so declare it as {cid!r} (the folder label stays "
                "in the 'condition' column and in the outputs beside its ID)",
            )


def held_conditions(
    objects: pd.DataFrame, fovs: pd.DataFrame, ids: Mapping[Any, Any], condition_col: str
) -> list[Any]:
    """The conditions the tables hold, by their IDs (``ids``, from ``checked_condition_ids``)
    when they have them, else the values of ``condition_col`` (the FOV table's when it has the
    column); none without the column."""
    if ids:
        return list(dict.fromkeys(ids.values()))
    for table in (fovs, objects):
        if condition_col in table.columns:
            return [c for c in table[condition_col].drop_duplicates() if isinstance(c, Hashable)]
    return []


def refuse_unmatched(names: Iterable[Any], conditions: Sequence[Any]) -> None:
    """Refuse a declared condition that matches none of ``conditions`` (``NOT_FOUND``), listing
    them: a selector that selects nothing must not pass without a word (#628)."""
    known = {str(c) for c in conditions}
    for name in names:
        if str(name) not in known:
            held = _listed(conditions) if conditions else "none"
            raise PairingRefused(
                NOT_FOUND,
                f"{name!r} names no condition of the tables (the conditions, by their "
                f"{CONDITION_ID_COLUMN!r} when the tables hold it: {held}): declare one of "
                "them (values are matched exactly, case and spaces included)",
            )


__all__ = [
    "FOLDER_CONTROL",
    "ID_DISAGREES",
    "ID_EMPTY",
    "ID_MERGED",
    "ID_NO_SOURCE",
    "ID_SPLIT",
    "NOT_FOUND",
    "PairingRefused",
    "checked_condition_ids",
    "declared_conditions",
    "held_conditions",
    "refuse_folder_names",
    "refuse_unmatched",
]
