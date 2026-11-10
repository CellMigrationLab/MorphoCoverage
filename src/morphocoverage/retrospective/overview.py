"""What is in the dataset: the first section of the microscopist's report.

Before any question is answered the report says what it was given: how many
objects and acquired fields (and how many fields hold no object), what one
row is, which feature columns are the phenotype representation, which
conditions and repeats exist, how many technical repeats (wells, dishes:
the ``technical_repeat`` column) each holds, how the fields fall across them,
and whether the values were normalised to the control. Long feature lists are cut after
``max_features_listed`` names. The declared control condition is listed first,
then the other conditions and the repeats in sorted order (never in the order
of the table's rows); labels that print alike (``'drug'`` and ``'drug '``) are
printed with their quotes. With a ``data_check`` (``check_tables``, which the
workflows run) its warnings are listed too. Everything here is counted from
the validated tables; nothing is estimated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .._acquisition import ACQUISITION, acquisition_identity
from .._control import control_name
from .._data_rules import _loose
from .._frozen import owned
from .._repeat_alias import refuse_repeat_alias
from .._style import Status, check_detail, group_columns, low_to_high, wrap
from .._style import table as style_table
from .._wording import column_word, count, listing
from ..data_check import DataCheck, Problem, validate_input_tables
from ..metadata import (
    CONDITION_COLUMN,
    MORPHOBRICKS_GROUP_COLUMNS,
    TECHNICAL_REPEAT_COLUMN,
    WITHIN_FOV_METADATA_COLUMNS,
    column_names,
    sorted_labels,
)
from ..sampling_unit._experiments import experiments
from ..schema import ValidatedInput, fov_positions, observation_words
from ._value_checks import value_checks
from .report_status import data_check_status

MAX_FEATURES_LISTED = 12
MAX_VALUES_LISTED = 12


def _shown(names: Sequence[Any]) -> list[str]:
    """The names as text, all quoted when two of them would print alike (``drug``, ``drug ``)."""
    loose = [_loose(n) for n in dict.fromkeys(names)]
    alike = len(set(loose)) < len(loose)
    return [repr(n) if alike else str(n) for n in names]


def _listed(names: Sequence[Any], limit: int) -> str:
    """The first ``limit`` names, then how many more; quoted when two would print alike."""
    names = _shown(names)
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit} more"


def _sub_experiments(family: dict[Any, int], compared: dict[Any, list[Any]], control: Any) -> str:
    """``2 sub-experiments, each of 2 conditions and its own siCTRL in its own 3 biological
    repeats (siCCT7#3 and siCCT7#6; siCCT8#5 and siCCT8#8)``; the conditions named for at
    most 3 sub-experiments, and each one's size when they differ."""
    members: dict[int, list[Any]] = {}
    for condition, number in family.items():
        members.setdefault(number, []).append(condition)
    repeats = {n: len({r for c in names for r in compared[c]}) for n, names in members.items()}
    sizes = {(len(names), repeats[n]) for n, names in members.items()}
    head = f"{len(members)} sub-experiments, each"
    if len(sizes) == 1:
        conditions, held = next(iter(sizes))
        text = (
            f"{head} of {count(conditions, 'condition')} and its own {control} in its own "
            f"{count(held, 'biological repeat')}"
        )
        if len(members) <= 3:
            text += f" ({'; '.join(listing(names, 4) for names in members.values())})"
        return text
    each = "; ".join(f"{listing(names, 4)} in {repeats[n]}" for n, names in members.items())
    return f"{head} with its own {control} in its own repeats: {each}"


# An embedding-only report's observation table (owner, 2026-10-06).
NO_FEATURES = "No explainable features: the other feature tables alone are read."


@owned("counts", "layout", "settings")
@dataclass(frozen=True)
class DatasetOverview:
    """Counts of the validated tables and the layout of fields by condition and repeat.

    - ``counts``: objects, fields, empty fields, features, conditions,
      repeats, technical repeats (distinct within their condition and repeat).
    - ``layout``: one row per condition and repeat with its fields, empty
      fields, objects and, when the column exists, technical repeats (empty
      when neither the condition nor the repeat column exists).
    - ``features``: the feature column names, in order.
    - ``settings``: the observation unit and whether it was declared, the
      condition, repeat and technical-repeat columns, the analysis groups,
      the values each within-field coordinate takes per field, and the
      normalisation applied.
    - ``warnings``: the data check's warnings (``check_tables``), when one was given.
    - ``value_checks``: the checks of the values that found something (``_value_checks``:
      fields more alike than chance, features with few distinct values, one sentence per
      group of duplicated or derived features), as sentences.
    """

    counts: dict[str, int]
    layout: pd.DataFrame
    features: tuple[str, ...]
    settings: dict[str, Any]
    warnings: tuple[Problem, ...] = ()
    value_checks: tuple[str, ...] = ()

    def summary(self, max_features_listed: int = MAX_FEATURES_LISTED, detail: str = "short") -> str:
        """``"short"``: what the tables hold in a few lines, with the data checks' marker;
        ``"full"``: plain sentences, an aligned fields-by-condition-and-repeat table when there
        is one, and every warning of the data check."""
        if check_detail(detail) == "short":
            return self._short(max_features_listed)
        return self._full(max_features_listed)

    def _short(self, max_features_listed: int) -> str:
        """Three to six lines: counts, features and values, conditions and repeats, checks."""
        lines = self.short_lines(max_features_listed)
        return "\n".join(wrap(line, 0, hang=2) for line in lines)

    def short_lines(self, max_features_listed: int = MAX_FEATURES_LISTED) -> list[str]:
        """The short form's sentences, one per line, not wrapped (a report wraps them under its
        own indent): counts, features and values, the design, planes, the data checks."""
        c, settings = self.counts, self.settings
        unit, words = observation_words(settings)
        declared = "" if settings["observation_unit_declared"] else " (assumed, not declared)"
        empty = f", {c['empty_fields']} without {words}" if c["empty_fields"] else ", none empty"
        objects = f"{count(c['objects'], unit, words)}{declared}"
        one = c["objects"] == c["fields"] and c["fields"] > 0
        # The layout's own levels need no words; another field key is said.
        standard = set(settings["fov_key"]) <= {*MORPHOBRICKS_GROUP_COLUMNS, "position_id"}
        key = "" if standard else f"; a field is {group_columns(settings['fov_key'])}"
        each = f" (one {unit} per field)" if one else ""
        lines = [f"{objects} in {count(c['fields'], 'field')}{each}{empty}{key}."]
        normalized = settings.get("normalized")
        values = f"values: {normalized}" if normalized else "values as given (not normalised)"
        lines.append(
            f"{count(c['features'], 'feature')} "
            f"({_listed(self.features, min(3, max_features_listed))}); {values}."
            if self.features
            else NO_FEATURES
        )
        lines += list(settings.get("feature_tables", ()))
        design = self._design_line()
        if design:
            lines.append(design)
        for axis, (low, high) in settings.get("planes", {}).items():
            lines.append(f"{axis!r}: {low_to_high(low, high)} planes per field, one field each.")
        status, text = data_check_status(self)
        lines.append(f"{status.marker} {text[:1].upper()}{text[1:]}.")
        lines += [f"{Status.CAUTION.marker} {check}" for check in self.value_checks]
        return lines

    def _design_line(self) -> str | None:
        """Conditions, repeats and technical repeats, and the fields per condition and repeat."""
        c, settings = self.counts, self.settings
        condition_col, repeat_col = settings["condition_col"], settings["repeat_col"]
        parts = []
        if condition_col is not None:
            names = self.layout[condition_col].drop_duplicates().tolist()
            parts.append(f"{count(c['conditions'], 'condition')} ({_listed(names, 6)})")
        if repeat_col is not None:
            names = self.layout[repeat_col].drop_duplicates().tolist()
            parts.append(f"{count(c['repeats'], 'biological repeat')} ({_listed(names, 6)})")
        arrangement = self.arrangement()
        if arrangement:
            parts = [", ".join(parts) + f"; {arrangement}"]
        if settings["technical_col"] is not None:
            parts.append(self._wells(apart=bool(arrangement)))
        group_cols = settings.get("group_cols", ())
        groups = (
            "each well analysed apart"
            if group_cols and group_cols[-1] == settings["technical_col"]
            else f"{count(c['groups'], 'analysis group')} ({group_columns(group_cols)}), each "
            "analysed apart"
            if group_cols
            else "every field in one group (pooled)"
        )
        per = self.layout["fields"] if len(self.layout) and "fields" in self.layout else []
        spread = ""
        if len(per):
            low, high = int(min(per)), int(max(per))
            fields = f"{low}" if low == high else f"{low} to {high}"
            # One observation per field: the fields are the cells a biologist counts.
            one = c["objects"] == c["fields"]
            word = observation_words(settings)[1] if one else "fields"
            spread = f"; {fields} {word} per condition and repeat"
        joined = "; ".join(parts) if arrangement else ", ".join(parts)
        return f"{joined + '; ' if parts else ''}{groups}{spread}."

    def arrangement(self) -> str:
        """How the conditions fall across the repeats when not every condition is in every
        repeat, empty when they are: the sub-experiments when the conditions do not all share
        their repeats and control (``2 sub-experiments, each of 2 conditions and its own
        siCTRL in its own 3 biological repeats (siCCT7#3 and siCCT7#6; siCCT8#5 and
        siCCT8#8)``, ``sampling_unit._experiments``), else the repeats each condition holds
        (``cond1 in 3, cond2 and cond3 each in 2 of the 4 repeats``)."""
        condition_col, repeat_col = self.settings["condition_col"], self.settings["repeat_col"]
        layout = self.layout
        if condition_col is None or repeat_col is None or not len(layout):
            return ""
        held = {
            c: list(dict.fromkeys(block[repeat_col]))
            for c, block in layout.groupby(condition_col, sort=False, dropna=False)
        }
        every = list(dict.fromkeys(layout[repeat_col]))
        if all(len(repeats) == len(every) for repeats in held.values()):
            return ""
        control = self.settings.get("control")
        own = set(held.get(control, ()))
        compared = {
            c: [r for r in repeats if r in own] for c, repeats in held.items() if c != control
        }
        alone = [c for c, repeats in compared.items() if not repeats] if own else []
        never = f"; {listing(alone, 4)} in no repeat with {control}" if alone else ""
        family = experiments({c: r for c, r in compared.items() if r})
        if len(set(family.values())) > 1:
            return _sub_experiments(family, compared, control) + never
        by_size: dict[int, list[Any]] = {}
        for condition, repeats in held.items():
            if len(repeats) < len(every):
                by_size.setdefault(len(repeats), []).append(condition)
        uneven = ", ".join(
            f"{listing(names, 4)} {'each ' if len(names) > 1 else ''}in {size}"
            for size, names in sorted(by_size.items())[::-1]
        )
        return f"{uneven} of the {len(every)} repeats{never}"

    def _wells(self, apart: bool = False) -> str:
        """The technical repeats in lab words: ``one well per condition in each repeat (9
        wells)`` (``apart``: ``..., 9 wells``, after the sub-experiments' parentheses), or
        ``14 wells (technical repeats; 1 to 2 per condition and repeat)``."""
        c, layout = self.counts, self.layout
        per = layout.get("technical_repeats", [])
        wells = count(c["technical_repeats"], "well")
        if len(per) and int(min(per)) == int(max(per)) == 1:
            each = "one well per condition in each repeat"
            return f"{each}, {wells}" if apart else f"{each} ({wells})"
        low, high = (int(min(per)), int(max(per))) if len(per) else (0, 0)
        each = f"{low}" if low == high else f"{low} to {high}"
        spread = f"; {each} per condition and repeat" if len(per) else ""
        return f"{wells} (technical repeats{spread})"

    def _full(self, max_features_listed: int) -> str:
        """Plain sentences and a fields-by-condition-and-repeat table when there is one."""
        c = self.counts
        unit, words = observation_words(self.settings)
        declared = "" if self.settings["observation_unit_declared"] else " (assumed, not declared)"
        empty = f", {c['empty_fields']} of them without {words}" if c["empty_fields"] else ""
        lines = [
            (
                f"{count(c['objects'], unit, words)}{declared} in "
                f"{count(c['fields'], 'acquired field')}{empty}; "
                f"a field is identified by {', '.join(self.settings['fov_key'])}."
            ),
            (
                f"{count(c['features'], 'feature column')}: "
                f"{_listed(self.features, max_features_listed)}."
                if self.features
                else NO_FEATURES
            ),
        ]
        technical_col = self.settings["technical_col"]
        if technical_col is not None:
            lines.append(self._technical_line(technical_col))
        for axis, (low, high) in self.settings.get("planes", {}).items():
            lines.append(
                f"{axis!r} takes {low_to_high(low, high)} values per field: {words} of several "
                "planes, "
                "channels or frames of one field (declared as plane-level rows)."
            )
        lines += self._grouping_lines(words)
        lines += self._closing_lines()
        return "\n".join(lines)

    def _technical_line(self, technical_col: str) -> str:
        """The technical repeats: how many, and how many per condition and repeat."""
        per = self.layout["technical_repeats"] if len(self.layout) else pd.Series(dtype=int)
        spread = (
            f", {int(per.min())} to {int(per.max())} per condition and repeat"
            if len(per) and per.min() != per.max()
            else (f", {int(per.iloc[0])} per condition and repeat" if len(per) else "")
        )
        return (
            f"{count(self.counts['technical_repeats'], 'technical repeat')} ({technical_col}, "
            f"a well or a dish within a repeat{spread})."
        )

    def _grouping_lines(self, words: str) -> list[str]:
        """How the fields are grouped: the analysis groups, or the conditions and repeats with
        their arrangement and a fields-by-condition-and-repeat table."""
        c = self.counts
        condition_col, repeat_col = self.settings["condition_col"], self.settings["repeat_col"]
        group_cols = self.settings.get("group_cols", ())
        if condition_col is None and repeat_col is None and group_cols:
            text = (
                f"No condition or repeat column; the analysis groups are {', '.join(group_cols)} "
                f"({count(c['groups'], 'group')}), each analysed apart."
            )
            return [text]
        if condition_col is None and repeat_col is None:
            return ["No condition or repeat column: every field is one group (all fields pooled)."]
        lines = [self._columns_line()]
        arrangement = self.arrangement()
        if arrangement:
            lines.append(f"{arrangement}.")  # it may start with a label: never capitalised
        lines.append(self._layout_table(words))
        return lines

    def _columns_line(self) -> str:
        """``3 conditions (condition: ...); 2 repeats (repeat: ...).``"""
        c = self.counts
        condition_col, repeat_col = self.settings["condition_col"], self.settings["repeat_col"]
        parts = []
        if condition_col is not None:
            names = self.layout[condition_col].drop_duplicates().tolist()
            parts.append(
                f"{count(c['conditions'], 'condition')} "
                f"({condition_col}: {_listed(names, MAX_VALUES_LISTED)})"
            )
        else:
            parts.append("no condition column")
        if repeat_col is not None:
            names = self.layout[repeat_col].drop_duplicates().tolist()
            parts.append(
                f"{count(c['repeats'], 'repeat')} "
                f"({repeat_col}: {_listed(names, MAX_VALUES_LISTED)})"
            )
        else:
            parts.append("no repeat column (one implicit repeat)")
        return "; ".join(parts) + "."

    def _layout_table(self, words: str) -> str:
        """The fields, empty fields and observations per condition and repeat (and technical
        repeats), or why there are too many combinations to list."""
        condition_col, repeat_col = self.settings["condition_col"], self.settings["repeat_col"]
        technical_col = self.settings["technical_col"]
        table = self.layout.copy()
        if len(table) > 40:
            return (
                f"  ({len(table)} condition-and-repeat combinations: too many to list; "
                "DatasetOverview.layout holds them)"
            )
        header = [col for col in (condition_col, repeat_col) if col is not None]
        columns = ["fields", "empty", words]
        keys = ["fields", "empty_fields", "objects"]
        if technical_col is not None:
            columns.append("technical repeats")
            keys.append("technical_repeats")
        for col in header:
            table[col] = _shown(table[col].tolist())
        rows = [
            [row[col] for col in header] + [row[key] for key in keys]
            for row in (r._asdict() for r in table.itertuples(index=False))
        ]
        align = "l" * len(header) + "r" * len(keys)
        return style_table(rows, [*header, *columns], indent=2, align=align)

    def _closing_lines(self) -> list[str]:
        """How the values were normalised, the other feature tables, the data check's
        warnings (or its status) and the checks of the values."""
        normalized = self.settings.get("normalized")
        if normalized:
            lines = [f"Values: {normalized}."]
        else:
            lines = ["Values: as given (not normalised)."]
        lines += list(self.settings.get("feature_tables", ()))
        status, text = data_check_status(self)
        if self.warnings:
            lines.append(
                f"{status.marker} The data check found {count(len(self.warnings), 'warning')} "
                "(check_tables says where and how to fix each):"
            )
            lines += [f"  - {problem.title}" for problem in self.warnings]
        else:
            lines.append(f"{status.marker} {text[:1].upper()}{text[1:]}.")
        lines += [f"{Status.CAUTION.marker} {check}" for check in self.value_checks]
        return lines


def _within_field_values(data: Any) -> dict[str, tuple[int, int]]:
    """For each within-field coordinate that varies within a field: its fewest and most values."""
    planes = {}
    key = list(data.fov_key)
    for axis in WITHIN_FOV_METADATA_COLUMNS:
        if axis not in data.objects.columns or axis in data.feature_cols:
            continue
        per_field = data.objects.groupby(key, sort=False, dropna=False)[axis].nunique()
        if len(per_field) and int(per_field.max()) > 1:
            planes[axis] = (int(per_field.min()), int(per_field.max()))
    return planes


def _layout(
    table: pd.DataFrame, keys: list[str], technical_col: str | None, first: Any
) -> pd.DataFrame:
    """The fields, empty fields, observations (and technical repeats) per condition and
    repeat (``keys``), the control ``first``, then sorted labels."""
    aggregations = {
        "fields": ("objects", "size"),
        "empty_fields": ("empty_fields", "sum"),
        "objects": ("objects", "sum"),
    }
    if technical_col is not None:
        aggregations["technical_repeats"] = (technical_col, "nunique")
    if not keys:
        return pd.DataFrame(columns=list(aggregations))
    layout = table.groupby(keys, sort=False, dropna=False).agg(**aggregations).reset_index()
    layout["empty_fields"] = layout["empty_fields"].astype(int)
    # The control first, then sorted labels: the row order of the tables is not data.
    ranks = [
        layout[col].map({v: i for i, v in enumerate(sorted_labels(layout[col], first))})
        for col in keys
    ]
    order = sorted(range(len(layout)), key=lambda i: tuple(int(r.iloc[i]) for r in ranks))
    return layout.iloc[order].reset_index(drop=True)


def _overview_counts(
    data: ValidatedInput,
    table: pd.DataFrame,
    keys: list[str],
    columns: tuple[str | None, str | None, str | None],
) -> dict[str, int]:
    """The counts of the overview; ``columns`` are the condition, repeat and technical-repeat
    columns found (None when absent)."""
    condition_col, repeat_col, technical_col = columns
    if technical_col is None:
        technical_repeats = 0
    else:
        technical_repeats = len(table[[*keys, technical_col]].drop_duplicates())
    return {
        "objects": len(data.objects),
        "fields": len(data.fovs),
        "empty_fields": int(table["empty_fields"].sum()),
        "features": len(data.feature_cols),
        "conditions": int(table[condition_col].nunique(dropna=False)) if condition_col else 0,
        "repeats": int(table[repeat_col].nunique(dropna=False)) if repeat_col else 0,
        "technical_repeats": technical_repeats,
        "groups": len(data.fovs.loc[:, list(data.group_cols)].drop_duplicates())
        if data.group_cols
        else 1,
    }


def dataset_overview(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    condition_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
    normalized: str | None = None,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    control: Mapping[str, Any] | None = None,
    data_check: DataCheck | None = None,
    feature_tables: Sequence[str] = (),
    control_experiments: Mapping[Any, Sequence[Any]] | None = None,
) -> DatasetOverview:
    """Count what the tables hold (see the module). ``normalized`` is a sentence to print, and
    so is each of ``feature_tables`` (the other feature tables, ``workflows._feature_space``).
    ``control_experiments`` (each of several controls' conditions) also checks the features
    for duplicated or derived columns within each experiment (``_feature_relations``).

    Each column argument is used when the FOV table has it; a technical
    repeat is counted within its condition and repeat (well J23 of two
    plates is two technical repeats). ``control`` (``{"condition": ...}``)
    puts the control condition first; ``data_check`` adds its warnings.
    """
    refuse_repeat_alias(repeat_col=repeat_col, control=control)
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=list(column_names(feature_cols, "feature_cols")),
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    condition_col = condition_col if condition_col in data.fovs.columns else None
    repeat_col = repeat_col if repeat_col in data.fovs.columns else None
    technical_col = technical_col if technical_col in data.fovs.columns else None
    positions = fov_positions(data, data.fovs)
    per_field = pd.Series(positions[positions >= 0]).value_counts()
    table = data.fovs.copy()
    table["objects"] = [int(per_field.get(i, 0)) for i in range(len(table))]
    table["empty_fields"] = table["objects"] == 0
    keys = [col for col in (condition_col, repeat_col) if col is not None]
    layout = _layout(table, keys, technical_col, dict(control or {}).get(condition_col))
    counts = _overview_counts(data, table, keys, (condition_col, repeat_col, technical_col))
    settings: dict[str, Any] = {
        ACQUISITION: acquisition_identity(data),
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "fov_key": list(data.fov_key),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "group_cols": tuple(data.group_cols),
        "planes": _within_field_values(data),
        "normalized": normalized,
        "feature_tables": list(feature_tables),
        "control": dict(control or {}).get(condition_col),
        # Whether a data check was given: without one, "no warning" is not known.
        "data_checked": data_check is not None,
    }
    warnings = tuple(data_check.warnings) if data_check is not None else ()
    checks = value_checks(data, control_experiments, condition_col or CONDITION_COLUMN)
    return DatasetOverview(counts, layout, tuple(data.feature_cols), settings, warnings, checks)


def normalized_sentence(
    control: Mapping[str, Any], repeat_col: str | None, observations: str = "objects"
) -> str:
    """How the values were normalised, for the overview and the declaration; ``observations``
    is the declared unit's plural (``images``), ``objects`` when none was declared."""
    where = f"the same {column_word(repeat_col)}" if repeat_col else "the whole table"
    named = control_name(control)
    return f"every feature divided by the mean of the control {named} {observations} of {where}"


__all__ = ["DatasetOverview", "dataset_overview", "normalized_sentence"]
