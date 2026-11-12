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

Conditions are named by their ``condition_id`` the same way (MC-1): with that column beside the
MorphoBricks ``condition`` folder label, each folder must have one ID in every field and each ID
one folder, and the working copies name every condition by its ID, so a control is declared by
its ID (``ctrl``, not the folder ``control``: a folder label that is not an ID is refused with
the ID to use). The analyses that pair take ``repeat_col="repeat_group"`` by default
(``pair_by_repeat_group``); naming ``repeat_col="biological_repeat"`` while the tables hold
``repeat_group`` asks to pair by folder label and is refused. The given tables are never changed,
and ``Pairing`` keeps every folder label beside its ID, for the outputs to name both
(experiment-v1 section 3.4: the source label is never replaced in what the user reads).
The IDs are read and the declared names checked in ``_condition_ids`` (re-exported here);
``named_by_id`` names the conditions by their IDs, the repeats as given, for the analyses
that do not pair (#628).

Shared-private: called by the paired analyses (``sampling_unit``, ``retrospective``,
``normalize``) and the workflows.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ._condition_ids import (
    FOLDER_CONTROL,
    ID_DISAGREES,
    ID_EMPTY,
    ID_MERGED,
    ID_NO_SOURCE,
    ID_SPLIT,
    NOT_FOUND,
    PairingRefused,
    _listed,
    checked_condition_ids,
    declared_conditions,
    held_conditions,
    refuse_folder_names,
    refuse_unmatched,
)
from ._control import VS
from ._repeat_alias import refuse_repeat_alias
from .metadata import (
    CONDITION_COLUMN,
    CONDITION_ID_COLUMN,
    REPEAT_COLUMN,
    REPEAT_GROUP_COLUMN,
    empty_values,
)

# Why a pairing is refused, by code (``PairingRefused.code``).
ABSENT = "repeat_group_absent"
EMPTY = "repeat_group_empty"
SPLIT = "repeat_group_split"
MERGED = "repeat_group_merged"
DISAGREES = "repeat_group_disagrees"
NO_SOURCE = "repeat_group_without_biological_repeat"
FOLDER_PAIRING = "repeat_col_pairs_by_folder_label"
CODES = (
    ABSENT,
    EMPTY,
    SPLIT,
    MERGED,
    DISAGREES,
    NO_SOURCE,
    FOLDER_PAIRING,
    ID_EMPTY,
    ID_SPLIT,
    ID_MERGED,
    ID_DISAGREES,
    ID_NO_SOURCE,
    FOLDER_CONTROL,
    NOT_FOUND,
)

_BY_GROUP = (
    f"repeats are paired by their confirmed repeat group ({REPEAT_GROUP_COLUMN!r}), never by "
    f"the folder label in {REPEAT_COLUMN!r}"
)


@dataclass(frozen=True)
class Pairing:
    """How the repeats were paired and the conditions named: ``renamed`` maps each
    ``(condition folder, repeat folder)`` whose repeat group differs from its label to that
    group, ``conditions`` each condition folder whose ``condition_id`` differs from it to the
    ID (both empty when every label is its ID, or when nothing was paired), and ``folders``
    each ``(condition ID, repeat group)`` of the working tables to its ``(condition folder,
    repeat folder)`` when anything was renamed (the repeat ``None`` without repeats)."""

    renamed: dict[tuple[Any, Any], Any]
    conditions: dict[Any, Any] = field(default_factory=dict)
    folders: dict[tuple[Any, Any], tuple[Any, Any]] = field(default_factory=dict)

    def folder_of(self, condition: Any, repeat: Any = None) -> str | None:
        """``"control/rep1"``: the folders of a working-table condition and repeat when they
        differ from it, else ``None`` (also for a pair the tables do not hold)."""
        path = _folder_path(self, condition, repeat)
        same = "/".join(str(part) for part in (condition, repeat) if part is not None)
        return None if path is None or path == same else path


def paired_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    condition_col: str = CONDITION_COLUMN,
    others: Sequence[pd.DataFrame] = (),
    names: Iterable[Any] = (),
    selected: Iterable[Any] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame], Pairing]:
    """The tables with each folder label replaced by its repeat group and each condition
    folder by its ``condition_id``, and the pairing.

    Without a ``biological_repeat`` column in the FOV table the repeats are left as given
    (one implicit repeat, as before). With one, ``repeat_group`` must be beside it and valid
    (module docstring), else ``PairingRefused``; so must ``condition_id`` when the tables hold
    it. ``names`` are the conditions the caller declared (a control): a folder label that is
    not an ID is refused (``FOLDER_CONTROL``). ``selected`` are declared conditions that must
    each select one (a positive control, an experiment mapping's keys): one that matches no
    condition is refused (``NOT_FOUND``) after the folder labels. ``others`` (feature tables
    holding the FOV key) are relabelled the same way. When every label is its ID the tables
    are returned as given (not copied).
    """
    groups = _checked_groups(objects, fovs, condition_col)
    ids = checked_condition_ids(objects, fovs, condition_col)
    selected = list(selected)
    refuse_folder_names([*names, *selected], ids)
    refuse_unmatched(selected, held_conditions(objects, fovs, ids, condition_col))
    pairing = _pairing(groups, ids)
    if not pairing.renamed and not pairing.conditions:
        return objects, fovs, list(others), pairing
    tables = [_relabelled(t, groups, ids, condition_col) for t in (objects, fovs, *others)]
    return tables[0], tables[1], tables[2:], pairing


def pair_by_repeat_group(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    repeat_col: str | None,
    condition_col: str = CONDITION_COLUMN,
    control: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None, Pairing]:
    """The tables an analysis pairing by ``repeat_col`` reads, the repeat column to use in
    them, and the pairing.

    ``repeat_group`` (the default of the analyses) pairs by the repeat groups
    (``paired_tables``): the working tables carry each group in ``biological_repeat``, which
    is returned as the column to use. ``biological_repeat`` named explicitly pairs by folder
    label, refused while the tables hold ``repeat_group`` (``FOLDER_PAIRING``), else read as
    before (refused when the tables hold ``biological_repeat`` without ``repeat_group``).
    Another column is the caller's own declaration of the repeats, and ``None`` compares
    within the whole table: both are used as given, and only the conditions are named by
    their IDs. ``control`` names the control (by ``condition_id`` when the tables hold it).
    """
    names = declared_conditions(control, condition_col)
    if repeat_col == REPEAT_COLUMN and any(
        REPEAT_GROUP_COLUMN in table.columns for table in (objects, fovs)
    ):
        raise PairingRefused(
            FOLDER_PAIRING,
            f"repeat_col={REPEAT_COLUMN!r} pairs by folder name, while the tables hold "
            f"{REPEAT_GROUP_COLUMN!r}: {_BY_GROUP}. Leave repeat_col at its default, "
            f"{REPEAT_GROUP_COLUMN!r}, to pair the repeats as confirmed",
        )
    if repeat_col in (REPEAT_GROUP_COLUMN, REPEAT_COLUMN):
        paired = paired_tables(objects, fovs, condition_col=condition_col, names=names)
        return paired[0], paired[1], REPEAT_COLUMN, paired[3]
    objects, fovs, pairing = named_by_id(objects, fovs, control, condition_col)
    return objects, fovs, repeat_col, pairing


def named_by_id(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None = None,
    control: Mapping[str, Any] | None = None,
    condition_col: str = CONDITION_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Pairing]:
    """The tables with each condition folder replaced by its ``condition_id`` (the repeats as
    given), and the pairing, for an analysis that reads conditions without pairing repeats
    (``check_tables``, ``scan_design``, the figures): ``control`` names conditions by their IDs,
    a folder label that is not an ID refused (``FOLDER_CONTROL``). Without the column, or with
    a condition column of the caller's own, the tables are returned as given. ``fovs`` may be
    ``None`` (the observation table then carries the IDs); tables that are not DataFrames are
    returned as given, for the analysis's own check to refuse."""
    if not isinstance(objects, pd.DataFrame) or not isinstance(fovs, (pd.DataFrame, type(None))):
        return objects, fovs, Pairing({})
    ids = checked_condition_ids(objects, objects if fovs is None else fovs, condition_col)
    refuse_folder_names(declared_conditions(control, condition_col), ids)
    pairing = _pairing({}, ids)
    if not pairing.conditions:
        return objects, fovs, pairing
    named = _relabelled(objects, {}, ids, condition_col)
    return named, None if fovs is None else _relabelled(fovs, {}, ids, condition_col), pairing


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


def _pairing(groups: dict[tuple[Any, Any], Any], ids: dict[Any, Any]) -> Pairing:
    """The ``Pairing`` of the checked repeat groups and condition IDs (every folder)."""
    renamed = _renamed(groups)
    conditions = {folder: cid for folder, cid in ids.items() if str(folder) != str(cid)}
    if not renamed and not conditions:
        return Pairing({})
    folders = {
        (ids.get(condition, condition), group): (condition, label)
        for (condition, label), group in groups.items()
    }
    for folder, cid in ids.items():
        folders.setdefault((cid, None), (folder, None))
    return Pairing(renamed, conditions, folders)


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
    table: pd.DataFrame,
    groups: dict[tuple[Any, Any], Any],
    ids: dict[Any, Any],
    condition_col: str,
) -> pd.DataFrame:
    """``table`` with each folder label replaced by its repeat group and each condition folder
    by its ID (a copy); ``groups`` holds every folder, so a table without the condition column
    is relabelled only where each label has one group whatever the condition."""
    has_repeat = REPEAT_COLUMN in table.columns and bool(groups)
    has_condition = condition_col in table.columns and bool(ids)
    if not has_repeat and not has_condition:
        return table
    out = table.copy()
    if has_repeat:
        labels = table[REPEAT_COLUMN]
        if condition_col in table.columns:
            pairs = list(zip(table[condition_col], labels, strict=True))
        else:
            by_label = _by_label(groups)
            pairs = [(None, label) for label in labels]
            groups = {(None, label): group for label, group in by_label.items()}
        relabelled = [groups.get(_pair(pair), pair[1]) for pair in pairs]
        out[REPEAT_COLUMN] = pd.Series(relabelled, index=table.index, dtype=object)
    if has_condition:
        named = [ids.get(c, c) if isinstance(c, Hashable) else c for c in table[condition_col]]
        out[condition_col] = pd.Series(named, index=table.index, dtype=object)
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


def with_folders(table: pd.DataFrame, pairing: Pairing) -> pd.DataFrame:
    """A per-repeat table (``condition``, ``biological_repeat``, and ``control_condition`` when
    it has one) with each row's folders beside its IDs, ``folder`` and ``control_folder``
    (``"DrugA/experiment_23"``), when the pairing renamed anything; else the table as given."""
    held = {CONDITION_COLUMN, REPEAT_COLUMN} <= set(table.columns)
    if not pairing.folders or not held or "folder" in table.columns:
        return table
    out = table.copy()
    position = list(out.columns).index(REPEAT_COLUMN) + 1
    sides = [("folder", CONDITION_COLUMN), ("control_folder", "control_condition")]
    references = out.get("control_condition", pd.Series([None] * len(out), index=out.index))
    for name, column in sides:
        if column not in out.columns:
            continue
        compared = column == CONDITION_COLUMN
        rows = zip(out[column], out[REPEAT_COLUMN], references, strict=True)
        folders = [
            _row_folder(pairing, *row) if compared else _folder_path(pairing, *row[:2])
            for row in rows
        ]
        out.insert(position, name, folders)
        position += 1
    return out


def _row_folder(pairing: Pairing, condition: Any, repeat: Any, reference: Any) -> str | None:
    """The folders of a per-repeat row's condition; a comparison's row, labelled ``"DrugA vs
    CTRL"`` (``_control.comparison_label``), is its treated condition's."""
    found = _folder_path(pairing, condition, repeat)
    suffix = f"{VS}{reference}"
    if found is None and isinstance(condition, str) and condition.endswith(suffix):
        found = _folder_path(pairing, condition[: -len(suffix)], repeat)
    return found


def _folder_path(pairing: Pairing, condition: Any, repeat: Any) -> str | None:
    """``"DrugA/experiment_23"``: the folders of a working-table condition and repeat, ``None``
    for a pair the tables do not hold (never the IDs written as if they were folders)."""
    found = pairing.folders.get((condition, repeat)) or pairing.folders.get((condition, None))
    if found is None:
        return None
    return "/".join(str(part) for part in found if part is not None)


def pairing_note(pairing: Pairing) -> str | None:
    """The sentences naming the folders paired, and the conditions named, under another name
    (``None`` when none is): each folder label beside its ID."""
    sentences = []
    if pairing.renamed:
        parts = [
            f"{label}{'' if condition is None else f' ({condition})'} = {group}"
            for (condition, label), group in sorted(pairing.renamed.items(), key=str)
        ]
        sentences.append(
            f"Repeats are paired by their repeat group ({REPEAT_GROUP_COLUMN}), not by folder "
            f"label: {'; '.join(parts)}."
        )
    if pairing.conditions:
        parts = [
            f"{cid} (folder {folder})"
            for folder, cid in sorted(pairing.conditions.items(), key=str)
        ]
        sentences.append(
            f"Conditions are named by their ID ({CONDITION_ID_COLUMN}), each beside its folder "
            f"label: {'; '.join(parts)}."
        )
    return " ".join(sentences) or None


__all__ = [
    "ABSENT",
    "CODES",
    "DISAGREES",
    "EMPTY",
    "FOLDER_CONTROL",
    "FOLDER_PAIRING",
    "ID_DISAGREES",
    "ID_EMPTY",
    "ID_MERGED",
    "ID_NO_SOURCE",
    "ID_SPLIT",
    "MERGED",
    "NOT_FOUND",
    "NO_SOURCE",
    "REPEAT_GROUP_COLUMN",
    "SPLIT",
    "Pairing",
    "PairingRefused",
    "checked_condition_ids",
    "declared_conditions",
    "held_conditions",
    "named_by_id",
    "pair_by_repeat_group",
    "paired_tables",
    "pairing_note",
    "refuse_folder_names",
    "refuse_unmatched",
    "repeat_groups",
    "with_folders",
]
