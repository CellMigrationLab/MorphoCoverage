"""Biological repeats are paired by their confirmed repeat group, never by folder label (#617).

The MorphoBricks FOV key keeps the **source** repeat, the folder label, in
``biological_repeat``; the canonical repeat ID the user confirmed is added beside it as
``repeat_group`` (MorphoBricks-Spec experiment-v1 section 3.4, ADR 0001 AQ5). Two folders of
different conditions may be one repeat under different names (``CTRL/rep1`` with
``DrugA/experiment_23``: fixture ``v03``), and one folder name may be two repeats (``rep1`` of
``CTRL`` in ``Day1``, ``rep1`` of ``DrugA`` in ``Day2``: fixture ``v30``). Grouping and pairing
must use ``repeat_group``: the pairing of conditions and families and every statement about
biological N. A consumer must not pair by ``biological_repeat`` and must not fall back to
folder-name grouping.

``paired_tables`` is the one place that reads the column. When the FOV table has a
``biological_repeat`` column it requires ``repeat_group`` beside it and checks it: a value in
every field, one repeat group per folder of a condition (``repeat_group`` is a function of the
key) and at most one folder of a condition per repeat group (otherwise the pairing is
ambiguous), and the observation table's copy, when it has one, equal to the FOV table's.
Anything else is refused (``PairingRefused``, its ``code`` naming the reason); nothing is
guessed and nothing falls back to the folder label. A valid column then replaces the folder
label by the repeat group in working copies of the tables, so every paired analysis behind it
pairs, normalises and counts repeats by the repeat group. When every folder label already
equals its repeat group the tables are returned as given, so those analyses are unchanged.

Within a condition a folder and its repeat group correspond one to one (the checks above), so
the working copies identify every field exactly as the given tables do.

Shared-private: called by the paired analyses (``sampling_unit``, ``retrospective``,
``normalize``) and the workflows.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._repeat_alias import refuse_repeat_alias
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN, empty_values

# The canonical repeat ID beside the FOV key (experiment-v1 section 3.4); the column name of
# preprocessing's experiment_design.csv and of MorphoFeature's coverage export (MF-5).
REPEAT_GROUP_COLUMN = "repeat_group"

# Why a pairing is refused, by code (``PairingRefused.code``).
ABSENT = "repeat_group_absent"
EMPTY = "repeat_group_empty"
SPLIT = "repeat_group_split"
MERGED = "repeat_group_merged"
DISAGREES = "repeat_group_disagrees"
NO_SOURCE = "repeat_group_without_biological_repeat"
CODES = (ABSENT, EMPTY, SPLIT, MERGED, DISAGREES, NO_SOURCE)

_BY_GROUP = (
    f"repeats are paired by their confirmed repeat group ({REPEAT_GROUP_COLUMN!r}), never by "
    f"the folder label in {REPEAT_COLUMN!r}"
)


class PairingRefused(ValueError):
    """The biological repeats cannot be paired across conditions; ``code`` names the reason
    (one of ``CODES``) and the message says what to fix."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Pairing:
    """How the repeats were paired: ``renamed`` maps each ``(condition, folder label)`` whose
    repeat group differs from its label to that group (empty when every label is its group,
    or when nothing was paired)."""

    renamed: dict[tuple[Any, Any], Any]


def paired_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    condition_col: str = CONDITION_COLUMN,
    others: Sequence[pd.DataFrame] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame], Pairing]:
    """The tables with each folder label replaced by its repeat group, and the pairing.

    Without a ``biological_repeat`` column in the FOV table the tables are returned as given
    (one implicit repeat, as before). With one, ``repeat_group`` must be beside it and valid
    (module docstring), else ``PairingRefused``. ``others`` (feature tables holding the FOV
    key) are relabelled the same way. When every label is its group the tables are returned
    as given (not copied).
    """
    groups = _checked_groups(objects, fovs, condition_col)
    renamed = _renamed(groups)
    if not renamed:
        return objects, fovs, list(others), Pairing({})
    tables = [_relabelled(t, groups, condition_col) for t in (objects, fovs, *others)]
    return tables[0], tables[1], tables[2:], Pairing(renamed)


def pair_by_repeat_group(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    repeat_col: str | None,
    condition_col: str = CONDITION_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``paired_tables`` for an analysis that pairs by ``repeat_col``: only when it is the
    MorphoBricks ``biological_repeat`` column. Another column named explicitly is the
    caller's own declaration of the repeats, and ``None`` compares within the whole table, so
    both are left as given."""
    if repeat_col != REPEAT_COLUMN:
        return objects, fovs
    paired = paired_tables(objects, fovs, condition_col=condition_col)
    return paired[0], paired[1]


def repeat_groups(
    objects: pd.DataFrame, fovs: pd.DataFrame, *, condition_col: str = CONDITION_COLUMN
) -> dict[tuple[Any, Any], Any]:
    """Each ``(condition, folder label)`` whose repeat group is not its label, mapped to the
    group; refused (``PairingRefused``) as ``paired_tables`` says. A table using the former
    name ``repeat`` is refused first, naming ``biological_repeat``."""
    return _renamed(_checked_groups(objects, fovs, condition_col))


def _renamed(groups: dict[tuple[Any, Any], Any]) -> dict[tuple[Any, Any], Any]:
    """The folders whose repeat group is not their label."""
    return {key: group for key, group in groups.items() if str(key[-1]) != str(group)}


def _checked_groups(
    objects: pd.DataFrame, fovs: pd.DataFrame, condition_col: str
) -> dict[tuple[Any, Any], Any]:
    """The repeat group of every ``(condition, folder label)``, checked (``repeat_groups``)."""
    refuse_repeat_alias(objects=objects, fovs=fovs)
    if REPEAT_COLUMN not in fovs.columns:
        if REPEAT_GROUP_COLUMN in fovs.columns:
            raise PairingRefused(
                NO_SOURCE,
                f"the FOV table has a {REPEAT_GROUP_COLUMN!r} column but no {REPEAT_COLUMN!r} "
                "column: the FOV key keeps the source folder label of each repeat beside its "
                f"repeat group; add the {REPEAT_COLUMN!r} column",
            )
        return {}
    if REPEAT_GROUP_COLUMN not in fovs.columns and {REPEAT_COLUMN, REPEAT_GROUP_COLUMN} <= set(
        objects.columns
    ):
        # The observation table may carry it alone; its folders' groups are read there.
        fovs = objects
    if REPEAT_GROUP_COLUMN not in fovs.columns:
        raise PairingRefused(
            ABSENT,
            f"the FOV table has no {REPEAT_GROUP_COLUMN!r} column, so the biological repeats "
            f"cannot be paired across conditions: {_BY_GROUP}. Add a {REPEAT_GROUP_COLUMN!r} "
            "column, the same value for the folders that are one biological repeat (the "
            "MorphoBricks export writes it from the experiment description; in a table of your "
            f"own, set it to {REPEAT_COLUMN!r} only where the folder labels are the repeats)",
        )
    keys = [c for c in (condition_col,) if c in fovs.columns] + [REPEAT_COLUMN]
    _refuse_empty(fovs, keys)
    groups = _groups_of(fovs, keys)
    _refuse_merged(groups, keys)
    if REPEAT_GROUP_COLUMN in objects.columns:
        _refuse_disagreeing(objects, keys, groups)
    return groups


def _refuse_empty(fovs: pd.DataFrame, keys: list[str]) -> None:
    """Refuse fields without a repeat group: their repeat is unknown."""
    empty = empty_values(fovs[REPEAT_GROUP_COLUMN])
    if empty.any():
        first = fovs.loc[empty, keys].iloc[0]
        where = ", ".join(f"{c} {first[c]!r}" for c in keys)
        raise PairingRefused(
            EMPTY,
            f"{int(empty.sum())} of the {len(fovs)} fields have no {REPEAT_GROUP_COLUMN!r} "
            f"value (the first: {where}), so their biological repeat is unknown: {_BY_GROUP}; "
            "fill it in for every field",
        )


def _groups_of(fovs: pd.DataFrame, keys: list[str]) -> dict[tuple[Any, Any], Any]:
    """The repeat group of each folder (``keys``: condition and label); a folder in two
    groups is refused."""
    found: dict[tuple[Any, Any], list[Any]] = {}
    for row in fovs[[*keys, REPEAT_GROUP_COLUMN]].drop_duplicates().itertuples(index=False):
        values = tuple(row)
        key = (values[0], values[1]) if len(keys) == 2 else (None, values[0])
        found.setdefault(key, []).append(values[-1])
    for (condition, label), groups in found.items():
        if len(groups) > 1:
            raise PairingRefused(
                SPLIT,
                f"the folder {label!r}{_of(condition)} is in {len(groups)} repeat groups "
                f"({_listed(groups)}): a folder belongs to exactly one biological repeat, so "
                f"its fields cannot be paired; give each folder one {REPEAT_GROUP_COLUMN!r}",
            )
    return {key: groups[0] for key, groups in found.items()}


def _refuse_merged(groups: dict[tuple[Any, Any], Any], keys: list[str]) -> None:
    """Refuse a repeat group that holds two folders of one condition: which of them pairs
    with the other conditions' folder is ambiguous."""
    folders: dict[tuple[Any, Any], list[Any]] = {}
    for (condition, label), group in groups.items():
        folders.setdefault((condition, group), []).append(label)
    for (condition, group), labels in folders.items():
        if len(labels) > 1:
            raise PairingRefused(
                MERGED,
                f"the repeat group {group!r} holds {len(labels)} folders{_of(condition)} "
                f"({_listed(labels)}): a repeat group holds at most one folder per condition, "
                "so which of them pairs with the other conditions is ambiguous; give each "
                "folder its own repeat group, or merge the folders",
            )


def _refuse_disagreeing(
    objects: pd.DataFrame, keys: list[str], groups: dict[tuple[Any, Any], Any]
) -> None:
    """Refuse an observation table whose ``repeat_group`` differs from the FOV table's."""
    held = [c for c in keys if c in objects.columns]
    if held != keys:
        return  # the observations take their repeat from the FOV table
    rows = objects[[*keys, REPEAT_GROUP_COLUMN]].drop_duplicates()
    for values in rows.itertuples(index=False):
        values = tuple(values)
        key = (values[0], values[1]) if len(keys) == 2 else (None, values[0])
        if key in groups and str(groups[key]) != str(values[-1]):
            raise PairingRefused(
                DISAGREES,
                f"the observation table puts the folder {key[1]!r}{_of(key[0])} in repeat "
                f"group {values[-1]!r} and the FOV table in {groups[key]!r}: give both tables "
                f"the same {REPEAT_GROUP_COLUMN!r}",
            )


def _relabelled(
    table: pd.DataFrame, groups: dict[tuple[Any, Any], Any], condition_col: str
) -> pd.DataFrame:
    """``table`` with each folder label replaced by its repeat group (a copy); ``groups`` holds
    every folder, so a table without the condition column is relabelled only where each label
    has one group whatever the condition."""
    if REPEAT_COLUMN not in table.columns:
        return table
    labels = table[REPEAT_COLUMN]
    if condition_col in table.columns:
        pairs = list(zip(table[condition_col], labels, strict=True))
    else:
        by_label = _by_label(groups)
        pairs = [(None, label) for label in labels]
        groups = {(None, label): group for label, group in by_label.items()}
    relabelled = [groups.get(_pair(pair), pair[1]) for pair in pairs]
    out = table.copy()
    out[REPEAT_COLUMN] = pd.Series(relabelled, index=table.index, dtype=object)
    return out


def _by_label(groups: dict[tuple[Any, Any], Any]) -> dict[Any, Any]:
    """The repeat group by folder label alone, for a table without the condition column;
    refused when one label is in different groups in two conditions (one of them may keep
    the label as its group)."""
    by_label: dict[Any, Any] = {}
    for (_, label), group in groups.items():
        if str(by_label.setdefault(label, group)) != str(group):
            raise PairingRefused(
                SPLIT,
                f"a table without the condition column holds the folder {label!r}, which is "
                "in different repeat groups in different conditions: add the condition column",
            )
    return by_label


def _pair(pair: tuple[Any, Any]) -> tuple[Any, Any]:
    """A ``(condition, label)`` key, hashable whatever the values."""
    return tuple(value if isinstance(value, Hashable) else str(value) for value in pair)


def _of(condition: Any) -> str:
    """`` of condition 'CTRL'`` (empty without a condition column)."""
    return "" if condition is None else f" of condition {condition!r}"


def _listed(values: Sequence[Any]) -> str:
    """``'Day1', 'Day2'``, in a stable order."""
    return ", ".join(repr(v) for v in sorted(values, key=str))


def pairing_note(pairing: Pairing) -> str | None:
    """One sentence naming the folders paired under another name (``None`` when none is)."""
    if not pairing.renamed:
        return None
    parts = [
        f"{label}{'' if condition is None else f' ({condition})'} = {group}"
        for (condition, label), group in sorted(pairing.renamed.items(), key=str)
    ]
    return (
        f"Repeats are paired by their repeat group ({REPEAT_GROUP_COLUMN}), not by folder "
        f"label: {'; '.join(parts)}."
    )


__all__ = [
    "ABSENT",
    "CODES",
    "DISAGREES",
    "EMPTY",
    "MERGED",
    "NO_SOURCE",
    "REPEAT_GROUP_COLUMN",
    "SPLIT",
    "Pairing",
    "PairingRefused",
    "pair_by_repeat_group",
    "paired_tables",
    "pairing_note",
    "repeat_groups",
]
