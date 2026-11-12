"""The experiment description of a MorphoFeature export: default controls and suggestions (MC-2).

MorphoFeature (MF-5b, ``coverage.json`` ``schema_version`` 3) ends each exported set's
``coverage.json`` with an ``experiment`` block, copied from the published experiment
description the run read: its SHA-256, the ``conditions`` and ``repeats`` digests, the
conditions with their source folders, each condition's ``reference`` (its control), the repeat
groups with their source repeats, and the descriptive ``target``/``treatment_key`` of each
condition (format: MorphoFeature's design note
``docs/restart/designs/coverage-experiment-block-2026-11-12.md``, section 2). The block names
identities only; nothing here is ever inferred from a folder name.

``read_experiment(coverage, fovs)`` reads the ``coverage.json`` of every export of one
analysis beside its FOV table and returns a ``DescribedExperiment``:

- **the default controls** are the description's ``reference`` graph (Stage 2 plan, MC-2;
  owner answer 1): one control when every other condition references it (``{"condition":
  "CTRL"}``), else one ``own_control`` row per condition and its reference, so a chain
  (``v10``: ``VEC_SiCol6a1 -> VEC_SiNC -> VEC``) is compared as a chain, each condition with
  its own control and a control with its parent (``default_control``). A condition whose
  control has no field in this export is not compared, and one that shares no biological
  repeat with its control is not paired (Spec warning ``W_REFERENCE_UNPAIRED``); both are
  said (``notes``), neither is compared with another control. The workflows take the default
  through ``workflows.described_setup``; a control the user declares instead overrides it,
  and the report states the override beside the description's reference;
- **the target suggestions** (``target_suggestions``) prefill the reagent grouping and are
  never used as one: ``Setup(targets=...)`` refuses them until the user confirms them
  (``TargetSuggestions.confirm``), and the confirmed mapping, a copy, is all the analysis
  reads, so a later change of the description changes no confirmed grouping. They are never a
  grouping, join, pairing, contrast or cache key, even when two conditions carry equal values
  (``v27``).

Refused (``DescriptionRefused``, its ``code`` naming the reason; note section 4), never read
"best effort" and never replaced by a guess from the folder names:

1. a ``coverage.json`` ``schema_version`` above 3, or a description ``schema_version`` other
   than 1 (``COVERAGE_VERSION_UNKNOWN``, ``DESCRIPTION_VERSION_UNKNOWN``); an unexported set
   (``NOT_EXPORTED``);
2. a missing block, when the FOV table has ``condition_id`` (a ``schema_version`` 2 export or
   one without the block): the controls are not defaulted (``MISSING``, raised by
   ``described_setup`` when no control is declared), and controls the user declares by ID
   are still accepted (owner decision, 12 Nov 2026);
3. a malformed block (``MALFORMED``, ``_described_shape``);
4. a block that disagrees with the FOV table (``DISAGREES``): a row whose ``condition`` is not
   the source folder of its ``condition_id``, whose ``repeat_group`` is not a repeat group of
   the block, or whose ``biological_repeat`` is not that group's source repeat for its
   condition;
5. exports of two descriptions in one analysis (``TWO_DESCRIPTIONS``, owner decision, 12 Nov
   2026);
6. a suggestion used before it is confirmed (``UNCONFIRMED``, ``Setup(targets=...)``).

A control named by its folder label is refused by the workflows, as before
(``condition_named_by_folder_label``, MC-1): the IDs are matched exactly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from ._described_shape import (
    COVERAGE_VERSION,
    COVERAGE_VERSION_UNKNOWN,
    DESCRIPTION_VERSION_UNKNOWN,
    DISAGREES,
    MALFORMED,
    MISSING,
    NOT_EXPORTED,
    TWO_DESCRIPTIONS,
    UNCONFIRMED,
    DescriptionRefused,
    checked_version,
    parsed_block,
)
from ._wording import listing
from .metadata import (
    CONDITION_COLUMN,
    CONDITION_ID_COLUMN,
    REPEAT_COLUMN,
    REPEAT_GROUP_COLUMN,
)

COVERAGE_FILE = "coverage.json"
# The FOV-table columns a block is checked against (note section 4.4).
CHECKED_COLUMNS = (CONDITION_COLUMN, CONDITION_ID_COLUMN, REPEAT_COLUMN, REPEAT_GROUP_COLUMN)
UNPAIRED = "W_REFERENCE_UNPAIRED"


@dataclass(frozen=True)
class ExperimentBlock:
    """The ``experiment`` block of one ``coverage.json``, as written (strings kept exactly):
    ``conditions`` maps each condition ID to its source folder, ``references`` each to its
    reference (``None`` without), ``repeat_groups`` each repeat ID to its members' source
    repeats, ``suggestions`` each condition to its ``(target, treatment_key)``."""

    description_sha256: str
    description_schema_version: int
    section_digests: Mapping[str, str]
    conditions: Mapping[str, str]
    references: Mapping[str, str | None]
    repeat_groups: Mapping[str, Mapping[str, str]]
    suggestions: Mapping[str, tuple[str | None, str | None]]

    def check_fovs(self, fovs: pd.DataFrame, source: str = COVERAGE_FILE) -> None:
        """Refuse (``DISAGREES``) a FOV table that disagrees with the block (module, 4)."""
        missing = [c for c in CHECKED_COLUMNS if c not in fovs.columns]
        if missing:
            raise DescriptionRefused(
                DISAGREES,
                f"{source} describes the experiment, but the FOV table has no {missing} "
                "column: give the FOV table MorphoFeature exported beside it",
            )
        for row in fovs[list(CHECKED_COLUMNS)].drop_duplicates().itertuples(index=False):
            problem = self._row_problem(*(str(v) for v in row))
            if problem is not None:
                raise DescriptionRefused(
                    DISAGREES,
                    f"{source}: the FOV table disagrees with the experiment description: "
                    f"{problem}; give the FOV table and the coverage.json of one export",
                )

    def _row_problem(self, folder: str, cid: str, repeat: str, group: str) -> str | None:
        """Why one FOV-table row disagrees with the block, or ``None``."""
        where = f"the fields of {folder}/{repeat}"
        if cid not in self.conditions:
            return f"{where} have the condition_id {cid!r}, which the description does not declare"
        if self.conditions[cid] != folder:
            return (
                f"{where} have the condition_id {cid!r}, whose source folder is "
                f"{self.conditions[cid]!r}, not {folder!r}"
            )
        if group not in self.repeat_groups:
            return (
                f"{where} have the repeat_group {group!r}, which the description does not declare"
            )
        members = self.repeat_groups[group]
        if cid not in members:
            return f"{where} are in the repeat group {group!r}, which does not hold {cid!r}"
        if members[cid] != repeat:
            return (
                f"{where} are in the repeat group {group!r}, whose source repeat of {cid!r} is "
                f"{members[cid]!r}, not {repeat!r}"
            )
        return None


@dataclass(frozen=True)
class CoverageRecord:
    """One ``coverage.json``: where it was read (``source``), its ``schema_version`` and its
    experiment block (``None`` without one)."""

    source: str
    schema_version: int
    experiment: ExperimentBlock | None


def _frozen_block(fields: dict[str, Any]) -> ExperimentBlock:
    """The block with read-only mappings."""
    groups = {rid: MappingProxyType(dict(m)) for rid, m in fields["repeat_groups"].items()}
    return ExperimentBlock(
        description_sha256=fields["description_sha256"],
        description_schema_version=fields["description_schema_version"],
        section_digests=MappingProxyType(fields["section_digests"]),
        conditions=MappingProxyType(fields["conditions"]),
        references=MappingProxyType(fields["references"]),
        repeat_groups=MappingProxyType(groups),
        suggestions=MappingProxyType(fields["suggestions"]),
    )


def parse_coverage(record: Any, source: str = COVERAGE_FILE) -> CoverageRecord:
    """A parsed ``coverage.json`` record, its version and block checked (see the module)."""
    if not isinstance(record, Mapping):
        raise DescriptionRefused(
            MALFORMED, f"{source}: a coverage.json record is a JSON object, not {record!r}"
        )
    version = checked_version(record, source)
    block = record.get("experiment")
    experiment = None if block is None else _frozen_block(parsed_block(block, source))
    return CoverageRecord(source, version, experiment)


def read_coverage(path: str | Path) -> CoverageRecord:
    """The ``coverage.json`` at ``path``, or in the export folder ``path`` (see the module)."""
    path = Path(path)
    if path.is_dir():
        path = path / COVERAGE_FILE
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DescriptionRefused(MALFORMED, f"{path}: not valid JSON ({error})") from error
    return parse_coverage(record, str(path))


class TargetSuggestions:
    """The description's ``target`` of each condition that has one (``suggested``): a prefill
    for the reagent grouping, never a grouping itself, and deliberately not a mapping, so no
    analysis reads it as one. ``Setup(targets=...)`` refuses it until the user confirms it:
    ``confirm()`` as suggested, or ``confirm({"DrugB": "SRC"})`` with the user's edits (a
    condition mapped to ``None`` is left out). ``treatment_keys`` holds the descriptive
    ``treatment_key`` of each condition that has one, shown, never read."""

    def __init__(self, targets: Mapping[str, str], treatment_keys: Mapping[str, str]) -> None:
        self.suggested = MappingProxyType(dict(targets))
        self.treatment_keys = MappingProxyType(dict(treatment_keys))

    def __repr__(self) -> str:
        return f"TargetSuggestions({dict(self.suggested)!r}, unconfirmed)"

    def __bool__(self) -> bool:
        return bool(self.suggested)

    def lines(self) -> list[str]:
        """Each suggested target with its conditions, as written (``SRC: DrugA, DrugC``)."""
        by_target: dict[str, list[str]] = {}
        for condition, target in self.suggested.items():
            by_target.setdefault(target, []).append(condition)
        return [f"{t}: {listing(c, 8)}" for t, c in by_target.items()]

    def confirm(self, edits: Mapping[str, str | None] | None = None) -> dict[str, str]:
        """The grouping the user confirms: a plain copy of the suggestions with ``edits``
        applied, for ``Setup(targets=...)``; later changes of the suggestions change it not."""
        confirmed: dict[str, str | None] = {**self.suggested, **dict(edits or {})}
        return {c: t for c, t in confirmed.items() if t is not None}


@dataclass(frozen=True)
class DescribedExperiment:
    """What an analysis takes from the ``coverage.json`` of its exports (see the module).

    ``block`` is the description (``None`` without one); ``missing`` says why the controls
    are not defaulted when the FOV table is described (``condition_id``) but an export carries
    no block; ``sources`` the records read; ``present`` the condition IDs with a field in the
    FOV table; ``pairs`` the description's ``(condition, reference)`` comparisons on these
    fields and ``notes`` the conditions not compared, and why.
    """

    block: ExperimentBlock | None
    missing: str | None
    sources: tuple[str, ...]
    present: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def described(self) -> bool:
        """Whether the FOV table comes from a described run (a block, or one missing)."""
        return self.block is not None or self.missing is not None

    def default_control(self) -> dict[str, Any]:
        """The default ``Setup`` fields: ``{"control": {"condition": X}}`` when every other
        condition with a field references ``X``, else ``{"contrasts": table}`` (one
        ``own_control`` row per pair), ``{}`` without a reference; refused (``MISSING``)
        without a block."""
        if self.missing is not None:
            raise DescriptionRefused(MISSING, self.missing)
        if not self.pairs:
            return {}
        references = {r for _, r in self.pairs}
        if len(references) == 1:
            (control,) = references
            treated = {c for c, _ in self.pairs}
            if treated == set(self.present) - {control}:
                return {"control": {CONDITION_COLUMN: control}}
        rows = [{"kind": "own_control", "condition": c, "reference": r} for c, r in self.pairs]
        return {"contrasts": pd.DataFrame(rows, columns=["kind", "condition", "reference"])}

    @property
    def target_suggestions(self) -> TargetSuggestions | None:
        """The description's targets, unconfirmed (``None`` without a block)."""
        if self.block is None:
            return None
        suggested = self.block.suggestions
        targets = {c: t for c, (t, _) in suggested.items() if t is not None}
        keys = {c: k for c, (_, k) in suggested.items() if k is not None}
        return TargetSuggestions(targets, keys)

    def summary(self) -> str:
        """The description in a few lines: its SHA-256, the default comparisons, the notes."""
        if self.block is None:
            return self.missing or "No experiment description: the export is not described."
        compared = [f"{c} vs {r}" for c, r in self.pairs] or ["none (no reference declared)"]
        lines = [
            (
                f"Experiment description {self.block.description_sha256[:12]} "
                f"({listing(list(self.sources), 3)})."
            ),
            f"Default controls (the description's references): {listing(compared, 8)}.",
        ]
        return "\n".join([*lines, *self.notes])


@dataclass(frozen=True)
class ControlSource:
    """Where a Setup's controls come from, for the report to state (``Setup.described``):
    the ``experiment``, and ``chosen``, the user's controls in words when they override the
    description's references (``None`` when the controls are the description's)."""

    experiment: DescribedExperiment
    chosen: str | None = None

    def sentences(self) -> list[str]:
        """The controls' source in sentences, the override named beside the reference."""
        experiment = self.experiment
        if experiment.block is None:
            return [
                f"Controls: declared by the user ({self.chosen}); none is defaulted, since "
                + (experiment.missing or "the export is not described")
                + "."
            ]
        given = [f"{c} vs {r}" for c, r in experiment.pairs] or ["no comparison"]
        sha = experiment.block.description_sha256[:12]
        if self.chosen is None:
            head = (
                f"Controls: the experiment description's references (description {sha}): "
                f"{listing(given, 8)}."
            )
        else:
            head = (
                f"Controls: chosen by the user, overriding the experiment description "
                f"(description {sha}): the description gives {listing(given, 8)}; the user "
                f"chose {self.chosen}."
            )
        return [head, *experiment.notes]


def confirmed_targets(targets: Any) -> dict[Any, str] | None:
    """``Setup.targets``: the reagent grouping the user confirmed, as a copy (condition ID ->
    target), ``None`` without one; refused (``UNCONFIRMED``) when given the description's
    suggestions themselves, and ``TypeError`` for anything but a mapping to strings."""
    if targets is None:
        return None
    if isinstance(targets, TargetSuggestions):
        raise DescriptionRefused(
            UNCONFIRMED,
            "targets holds the description's suggestions, which are not a grouping until you "
            "confirm them: check them and pass targets=suggestions.confirm() (or with your "
            "edits, confirm({'DrugB': 'SRC'})); they are compared exactly, case included",
        )
    if not isinstance(targets, Mapping):
        raise TypeError("targets must map each condition ID to its target, as confirmed")
    bad = [c for c, t in targets.items() if not isinstance(t, str) or not t]
    if bad:
        raise TypeError(f"targets must give each condition a target name; {bad[0]!r} has none")
    return dict(targets)


def targets_line(targets: Mapping[Any, str] | None) -> str | None:
    """The confirmed reagent grouping in a sentence, or ``None`` without one."""
    if not targets:
        return None
    by_target: dict[str, list[str]] = {}
    for condition, target in targets.items():
        by_target.setdefault(target, []).append(str(condition))
    named = "; ".join(f"{t} ({listing(c, 8)})" for t, c in by_target.items())
    return f"Targets: as confirmed by the user: {named}."


def _records(coverage: Any) -> list[CoverageRecord]:
    """The records of ``coverage``: one or several paths or ``CoverageRecord`` s."""
    many = isinstance(coverage, Sequence) and not isinstance(coverage, str)
    items = list(coverage) if many else [coverage]
    if not items:
        raise ValueError("read_experiment needs at least one coverage.json")
    return [i if isinstance(i, CoverageRecord) else read_coverage(i) for i in items]


def _one_description(records: list[CoverageRecord]) -> ExperimentBlock | None:
    """The one description of the records, refused (``TWO_DESCRIPTIONS``) when two differ."""
    shas = {r.experiment.description_sha256: r for r in records if r.experiment is not None}
    if len(shas) > 1:
        named = [f"{r.source} ({sha[:12]})" for sha, r in shas.items()]
        raise DescriptionRefused(
            TWO_DESCRIPTIONS,
            f"the exports come from {len(shas)} experiment descriptions: {listing(named, 6)}; "
            "one analysis reads one description: analyse each apart, or re-export them from "
            "one preprocessing run",
        )
    return next((r.experiment for r in records if r.experiment is not None), None)


def _missing_reason(records: list[CoverageRecord]) -> str:
    """Why the controls are not defaulted: the records without a block."""
    named = [f"{r.source} (schema_version {r.schema_version})" for r in records if not r.experiment]
    return (
        f"{listing(named, 4)} has no experiment block, but its FOV table is described "
        f"({CONDITION_ID_COLUMN!r}): the controls are not defaulted and never guessed from the "
        "folder names; declare every control by its ID, or re-export with MorphoFeature after "
        f"MF-5b (coverage.json schema_version {COVERAGE_VERSION}) to default them to the "
        "description's references"
    )


def _comparisons(
    block: ExperimentBlock, fovs: pd.DataFrame
) -> tuple[tuple[str, ...], list[tuple[str, str]], list[str]]:
    """The conditions with a field, the description's pairs on them, and the notes."""
    rows = fovs[[CONDITION_ID_COLUMN, REPEAT_GROUP_COLUMN]].astype(str).drop_duplicates()
    groups = rows.groupby(CONDITION_ID_COLUMN)[REPEAT_GROUP_COLUMN].agg(set).to_dict()
    present = tuple(c for c in block.conditions if c in groups)
    pairs, notes = [], []
    for cid, reference in block.references.items():
        if reference is None or cid not in groups:
            continue
        if reference not in groups:
            notes.append(
                f"{cid}: not compared: its control {reference} has no field in this export."
            )
        elif not groups[cid] & groups[reference]:
            notes.append(
                f"{cid}: not paired with its control {reference}: they share no biological "
                f"repeat ({UNPAIRED})."
            )
        else:
            pairs.append((cid, reference))
    referenced = {r for r in block.references.values() if r is not None}
    notes += [
        f"{cid}: not compared: the description declares no reference for it."
        for cid in present
        if block.references[cid] is None and cid not in referenced
    ]
    return present, pairs, notes


def read_experiment(coverage: Any, fovs: pd.DataFrame) -> DescribedExperiment:
    """The experiment description of an analysis (see the module): ``coverage`` is the
    ``coverage.json`` (a path, an export folder or a ``CoverageRecord``, or a sequence of them,
    one per export), ``fovs`` the FOV table the analysis reads. Refused
    (``DescriptionRefused``) as the module lists."""
    if not isinstance(fovs, pd.DataFrame):
        raise TypeError(f"fovs must be a pandas DataFrame, not {type(fovs).__name__}")
    records = _records(coverage)
    sources = tuple(r.source for r in records)
    block = _one_description(records)
    for record in records:
        if record.experiment is not None:
            record.experiment.check_fovs(fovs, record.source)
    without = any(r.experiment is None for r in records)
    if without and CONDITION_ID_COLUMN in fovs.columns:
        return DescribedExperiment(block, _missing_reason(records), sources)
    if block is None:
        return DescribedExperiment(None, None, sources)
    present, pairs, notes = _comparisons(block, fovs)
    return DescribedExperiment(block, None, sources, present, tuple(pairs), tuple(notes))


__all__ = [
    "COVERAGE_VERSION_UNKNOWN",
    "DESCRIPTION_VERSION_UNKNOWN",
    "DISAGREES",
    "MALFORMED",
    "MISSING",
    "NOT_EXPORTED",
    "TWO_DESCRIPTIONS",
    "UNCONFIRMED",
    "UNPAIRED",
    "ControlSource",
    "CoverageRecord",
    "DescribedExperiment",
    "DescriptionRefused",
    "ExperimentBlock",
    "TargetSuggestions",
    "confirmed_targets",
    "parse_coverage",
    "read_coverage",
    "read_experiment",
    "targets_line",
]
