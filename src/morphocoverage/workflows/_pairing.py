"""The workflows pair biological repeats by their repeat group, or say why they cannot (#617).

Every workflow reads the tables through ``paired_inputs`` first (``_repeat_groups``): with a
valid ``repeat_group`` column the folder labels are replaced by the repeat groups in working
copies of the tables (the feature tables too, so they still match their observations), and
every analysis behind the workflow pairs conditions, normalises to the control of the same
repeat and counts biological repeats by the repeat group. Tables whose labels already are their
groups are used as given.

When the repeats cannot be paired (no ``repeat_group`` column, or an ambiguous one) and the
Setup compares conditions or reads values relative to a control, ``report`` and the figures
run what needs no pairing (``unpaired_setup``: the declaration without its control, contrasts,
normalisation and positive control) and the report says, in each section that would have
paired, that the conditions were not compared and why; the planning workflows refuse
(``PairingRefused``), since a target they were asked to judge would otherwise be dropped. A
Setup that pairs nothing reads the tables as given.

Shared-private: used by ``retrospective``, ``figures``, ``_html_figures`` and ``benchmarks``.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from typing import Any, NamedTuple

import pandas as pd

from .._repeat_groups import (
    FOLDER_CONTROL,
    NOT_FOUND,
    Pairing,
    PairingRefused,
    declared_conditions,
    paired_tables,
    pairing_note,
    with_folders,
)
from ..metadata import CONDITION_COLUMN
from ._contrasts import declared_conditions as contrast_conditions
from ._described import check_described
from .setup import Setup

# The sections of the report that pair repeats: the Foundation's object rule and the unusual
# observations (against the control of the same repeat), and the conditions (question 9).
PAIRED_SECTIONS = ("unit", "profile", "conditions")


class Paired(NamedTuple):
    """The tables a workflow reads, the Setup it runs, the pairing, why the repeats could not
    be paired (``None`` when they were, or when nothing needed pairing) and the Setup given."""

    objects: pd.DataFrame
    fovs: pd.DataFrame
    feature_tables: Mapping[str, Any] | None
    setup: Setup
    pairing: Pairing
    refused: PairingRefused | None
    given: Setup


def needs_pairing(setup: Setup) -> bool:
    """Whether the Setup pairs repeats: a control or contrasts (the tests, and every analysis
    against the control of the same repeat), a normalisation, or a positive control."""
    return (
        setup.compares
        or setup.reference_control is not None
        or setup.normalize is not None
        or setup.positive is not None
    )


def paired_inputs(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    feature_tables: Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
) -> Paired:
    """The workflow's tables paired by repeat group (module docstring). A refusal raises with
    ``strict``, else gives the unpaired Setup and the refusal."""
    if not isinstance(objects, pd.DataFrame) or not isinstance(fovs, pd.DataFrame):
        return Paired(
            objects, fovs, feature_tables, setup, Pairing({}), None, setup
        )  # refused later
    check_described(setup, fovs)  # the description read beside these fields (MC-2)
    names, frames = _frames(feature_tables)
    try:
        paired_objects, paired_fovs, others, pairing = paired_tables(
            objects,
            fovs,
            others=frames,
            names=setup_conditions(setup),
            selected=selected_conditions(setup),
        )
    except PairingRefused as refused:
        if refused.code in (FOLDER_CONTROL, NOT_FOUND):
            raise  # a declaration to correct, not a property of the tables
        if not needs_pairing(setup):
            return Paired(objects, fovs, feature_tables, setup, Pairing({}), None, setup)
        if strict:
            raise
        unpaired = unpaired_setup(setup)
        return Paired(objects, fovs, feature_tables, unpaired, Pairing({}), refused, setup)
    tables = _rebuilt(feature_tables, names, others)
    return Paired(paired_objects, paired_fovs, tables, setup, pairing, None, setup)


def setup_conditions(setup: Setup) -> list[Any]:
    """The conditions a Setup names: its control declaration, master control and parents, its
    positive control, the conditions of its experiment mapping, of its confirmed targets and of
    its contrast table
    (conditions are named by their ``condition_id`` when the tables hold it). The workflows
    read the Setup on tables whose conditions are already named by their IDs, where a folder
    label would select nothing without a word, so each is refused here, with the ID to use."""
    names = declared_conditions(setup.control, CONDITION_COLUMN)
    if setup.master_control is not None:
        names += declared_conditions({CONDITION_COLUMN: setup.master_control})
    if setup.control_parents:
        names += declared_conditions({CONDITION_COLUMN: dict(setup.control_parents)})
    if setup.positive is not None:
        names += declared_conditions(setup.positive.selector, CONDITION_COLUMN)
    if isinstance(setup.experiment, Mapping):
        names += declared_conditions({CONDITION_COLUMN: list(setup.experiment)})
    names += list(setup.targets or ())
    return names + contrast_conditions(setup)


def selected_conditions(setup: Setup) -> list[Any]:
    """The conditions a Setup selects, each of which must be in the tables: its positive
    control's, the keys of its experiment mapping and of its confirmed targets (MC-2). One that
    matches no condition would
    leave the positive control unassessed, or the condition in a correction family of its own,
    without a word, so it is refused (#628). The control declaration is checked by the
    analyses that read it (its several-control form also holds treatment keys)."""
    names = []
    if setup.positive is not None:
        names += declared_conditions(setup.positive.selector, CONDITION_COLUMN)
    if isinstance(setup.experiment, Mapping):
        names += declared_conditions({CONDITION_COLUMN: list(setup.experiment)})
    return names + list(setup.targets or ())


def unpaired_setup(setup: Setup) -> Setup:
    """The Setup without what pairs repeats: its control, contrasts, normalisation and
    positive control (what needs no pairing runs as declared)."""
    return replace(
        setup,
        control=None,
        contrasts=None,
        normalize=None,
        positive=None,
        master_control=None,
        control_parents=None,
        experiment=None,
        scales=None,
    )


def refusal_reasons(refused: PairingRefused) -> dict[str, str]:
    """The reason each paired section of the report gives: not compared, and why."""
    reason = f"not compared ({refused.code}): {refused}"
    return dict.fromkeys(PAIRED_SECTIONS, reason)


def pairing_said(
    paired: Paired, declared: Setup, settings: dict[str, Any], reasons: dict[str, str]
) -> Setup:
    """What the report records of the pairing: the folders paired under another name in
    ``settings["repeat_pairing"]`` (absent when none is), and when the repeats could not be
    paired, the reason in each paired section; returns the declaration to keep (the Setup
    given, when the report ran without its pairing)."""
    note = pairing_note(paired.pairing)
    if note is not None:
        settings["repeat_pairing"] = note
        # Each working (condition, repeat) with its folders, to write code on the given tables.
        settings["folder_labels"] = [
            (*working, *folders) for working, folders in paired.pairing.folders.items()
        ]
    if paired.refused is None:
        return declared
    reasons.update(refusal_reasons(paired.refused))
    return paired.given


def with_folder_labels(results: MutableMapping[str, Any], pairing: Pairing) -> None:
    """Name each folder beside its IDs in the report's results, when the pairing renamed any
    (experiment-v1 section 3.4): the per-repeat rows of the condition contrast gain ``folder``
    and ``control_folder``, and the fields to check name their group's folders
    (``settings["folders"]``, ``inspection_text.field_group``)."""
    if not pairing.folders:
        return
    contrast = results.get("contrast")
    if contrast is not None:
        results["contrast"] = replace(contrast, table=with_folders(contrast.table, pairing))
    inspection = results.get("inspection")
    if inspection is not None:
        folders = [
            (condition, repeat, folder)
            for (condition, repeat) in pairing.folders
            if (folder := pairing.folder_of(condition, repeat)) is not None
        ]
        results["inspection"] = replace(
            inspection, settings={**inspection.settings, "folders": folders}
        )


def _frames(
    feature_tables: Mapping[str, Any] | None,
) -> tuple[list[str], list[pd.DataFrame]]:
    """The feature tables' names and frames (a table given with its tag gives its frame)."""
    if not isinstance(feature_tables, Mapping):
        return [], []
    names, frames = [], []
    for name, given in feature_tables.items():
        frame = given[0] if isinstance(given, tuple) and given else given
        if isinstance(frame, pd.DataFrame):
            names.append(name)
            frames.append(frame)
    return names, frames


def _rebuilt(
    feature_tables: Mapping[str, Any] | None, names: list[str], frames: list[pd.DataFrame]
) -> Mapping[str, Any] | None:
    """``feature_tables`` with each frame replaced by its paired copy, tags kept."""
    if not names or feature_tables is None:
        return feature_tables
    paired = dict(zip(names, frames, strict=True))
    rebuilt: dict[str, Any] = {}
    for name, given in feature_tables.items():
        if name not in paired:
            rebuilt[name] = given
        elif isinstance(given, tuple):
            rebuilt[name] = (paired[name], *given[1:])
        else:
            rebuilt[name] = paired[name]
    return rebuilt


__all__ = [
    "PAIRED_SECTIONS",
    "Paired",
    "needs_pairing",
    "paired_inputs",
    "pairing_said",
    "refusal_reasons",
    "selected_conditions",
    "setup_conditions",
    "unpaired_setup",
    "with_folder_labels",
]
