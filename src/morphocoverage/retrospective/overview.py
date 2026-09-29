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

from .._data_rules import _loose
from .._style import check_detail, group_columns, wrap
from .._style import table as style_table
from .._wording import count
from ..data_check import DataCheck, Problem, validate_input_tables
from ..metadata import (
    MORPHOBRICKS_GROUP_COLUMNS,
    TECHNICAL_REPEAT_COLUMN,
    WITHIN_FOV_METADATA_COLUMNS,
    sorted_labels,
)
from ..schema import fov_positions, unit_plural
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
    """

    counts: dict[str, int]
    layout: pd.DataFrame
    features: tuple[str, ...]
    settings: dict[str, Any]
    warnings: tuple[Problem, ...] = ()

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
        unit = settings["observation_unit"]
        words = unit_plural(unit)
        declared = "" if settings["observation_unit_declared"] else " (assumed, not declared)"
        empty = f", {c['empty_fields']} without {words}" if c["empty_fields"] else ", none empty"
        objects = f"{count(c['objects'], unit, words)}{declared}"
        key = group_columns(settings["fov_key"])
        lines = [f"{objects} in {count(c['fields'], 'field')}{empty}; a field is {key}."]
        normalized = settings.get("normalized")
        values = f"values: {normalized}" if normalized else "values as given (not normalised)"
        lines.append(
            f"{count(c['features'], 'feature')} "
            f"({_listed(self.features, min(3, max_features_listed))}); {values}."
        )
        design = self._design_line()
        if design:
            lines.append(design)
        for axis, (low, high) in settings.get("planes", {}).items():
            lines.append(f"{axis!r}: {low} to {high} planes per field, one field each.")
        status, text = data_check_status(self)
        lines.append(f"{status.marker} {text[:1].upper()}{text[1:]}.")
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
            parts.append(f"{count(c['repeats'], 'repeat')} ({_listed(names, 6)})")
        if settings["technical_col"] is not None:
            parts.append(count(c["technical_repeats"], "technical repeat"))
        group_cols = settings.get("group_cols", ())
        groups = (
            f"{count(c['groups'], 'analysis group')} ({group_columns(group_cols)}), each "
            "analysed apart"
            if group_cols
            else "every field in one group (pooled)"
        )
        per = self.layout["fields"] if len(self.layout) and "fields" in self.layout else []
        spread = ""
        if len(per):
            low, high = int(min(per)), int(max(per))
            fields = f"{low}" if low == high else f"{low} to {high}"
            spread = f"; {fields} fields per condition and repeat"
        return f"{', '.join(parts) + '; ' if parts else ''}{groups}{spread}."

    def _full(self, max_features_listed: int) -> str:
        """Plain sentences and a fields-by-condition-and-repeat table when there is one."""
        c = self.counts
        unit = self.settings["observation_unit"]
        words = unit_plural(unit)
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
            ),
        ]
        condition_col, repeat_col = self.settings["condition_col"], self.settings["repeat_col"]
        technical_col = self.settings["technical_col"]
        if technical_col is not None:
            per = self.layout["technical_repeats"] if len(self.layout) else pd.Series(dtype=int)
            spread = (
                f", {int(per.min())} to {int(per.max())} per condition and repeat"
                if len(per) and per.min() != per.max()
                else (f", {int(per.iloc[0])} per condition and repeat" if len(per) else "")
            )
            lines.append(
                f"{count(c['technical_repeats'], 'technical repeat')} ({technical_col}, a well "
                f"or a dish within a repeat{spread})."
            )
        for axis, (low, high) in self.settings.get("planes", {}).items():
            lines.append(
                f"{axis!r} takes {low} to {high} values per field: {words} of several planes, "
                "channels or frames of one field (declared as plane-level rows)."
            )
        group_cols = self.settings.get("group_cols", ())
        if condition_col is None and repeat_col is None and group_cols:
            lines.append(
                f"No condition or repeat column; the analysis groups are {', '.join(group_cols)} "
                f"({count(c['groups'], 'group')}), each analysed apart."
            )
        elif condition_col is None and repeat_col is None:
            lines.append(
                "No condition or repeat column: every field is one group (all fields pooled)."
            )
        else:
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
            lines.append("; ".join(parts) + ".")
            table = self.layout.copy()
            if len(table) <= 40:
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
                lines.append(style_table(rows, [*header, *columns], indent=2, align=align))
            else:
                lines.append(
                    f"  ({len(table)} condition-and-repeat combinations; see the layout table)"
                )
        normalized = self.settings.get("normalized")
        if normalized:
            lines.append(f"Values: {normalized}.")
        else:
            lines.append("Values: as given (not normalised).")
        status, text = data_check_status(self)
        if self.warnings:
            lines.append(
                f"{status.marker} The data check found {count(len(self.warnings), 'warning')} "
                "(check_tables says where and how to fix each):"
            )
            lines += [f"  - {problem.title}" for problem in self.warnings]
        else:
            lines.append(f"{status.marker} {text[:1].upper()}{text[1:]}.")
        return "\n".join(lines)


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
) -> DatasetOverview:
    """Count what the tables hold (see the module). ``normalized`` is a sentence to print.

    Each column argument is used when the FOV table has it; a technical
    repeat is counted within its condition and repeat (well J23 of two
    plates is two technical repeats). ``control`` (``{"condition": ...}``)
    puts the control condition first; ``data_check`` adds its warnings.
    """
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=list(feature_cols),
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
    aggregations = {
        "fields": ("objects", "size"),
        "empty_fields": ("empty_fields", "sum"),
        "objects": ("objects", "sum"),
    }
    if technical_col is not None:
        aggregations["technical_repeats"] = (technical_col, "nunique")
    if keys:
        layout = table.groupby(keys, sort=False, dropna=False).agg(**aggregations).reset_index()
        layout["empty_fields"] = layout["empty_fields"].astype(int)
        # The control first, then sorted labels: the row order of the tables is not data.
        first = dict(control or {}).get(condition_col)
        ranks = [
            layout[col].map({v: i for i, v in enumerate(sorted_labels(layout[col], first))})
            for col in keys
        ]
        order = sorted(range(len(layout)), key=lambda i: tuple(int(r.iloc[i]) for r in ranks))
        layout = layout.iloc[order].reset_index(drop=True)
    else:
        layout = pd.DataFrame(columns=list(aggregations))
    if technical_col is None:
        technical_repeats = 0
    else:
        technical_repeats = len(table[[*keys, technical_col]].drop_duplicates())
    counts = {
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
    settings: dict[str, Any] = {
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "fov_key": list(data.fov_key),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "group_cols": tuple(data.group_cols),
        "planes": _within_field_values(data),
        "normalized": normalized,
        # Whether a data check was given: without one, "no warning" is not known.
        "data_checked": data_check is not None,
    }
    warnings = tuple(data_check.warnings) if data_check is not None else ()
    return DatasetOverview(counts, layout, tuple(data.feature_cols), settings, warnings)


def normalized_sentence(control: Mapping[str, Any], repeat_col: str | None) -> str:
    """How the values were normalised, for the overview and the declaration."""
    where = f"the same {repeat_col}" if repeat_col else "the whole table"
    return f"every feature divided by the mean of the control {dict(control)} objects of {where}"


__all__ = ["DatasetOverview", "dataset_overview", "normalized_sentence"]
