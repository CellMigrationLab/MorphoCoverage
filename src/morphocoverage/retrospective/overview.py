"""What is in the dataset: the first section of the microscopist's report.

Before any question is answered the report says what it was given: how many
objects and acquired fields (and how many fields hold no object), what one
row is, which feature columns are the phenotype representation, which
conditions and repeats exist, how many technical repeats (wells, dishes:
the ``technical_repeat`` column) each holds, how the fields fall across them,
and whether the values were normalised to the control. Long feature lists are cut after
``max_features_listed`` names. Everything here is counted from the validated
tables; nothing is estimated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..data_check import validate_input_tables
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from ..schema import fov_positions, unit_plural

MAX_FEATURES_LISTED = 12
MAX_VALUES_LISTED = 12


def _listed(names: Sequence[Any], limit: int) -> str:
    """The first ``limit`` names, then how many more."""
    names = [str(n) for n in names]
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
      condition, repeat and technical-repeat columns, and the normalisation
      applied.
    """

    counts: dict[str, int]
    layout: pd.DataFrame
    features: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self, max_features_listed: int = MAX_FEATURES_LISTED) -> str:
        """Plain sentences and a fields-by-condition-and-repeat table when there is one."""
        c = self.counts
        unit = self.settings["observation_unit"]
        words = unit_plural(unit)
        declared = "" if self.settings["observation_unit_declared"] else " (assumed, not declared)"
        empty = f", {c['empty_fields']} of them without {words}" if c["empty_fields"] else ""
        lines = [
            (
                f"{c['objects']} {words}{declared} in {c['fields']} acquired fields{empty}; "
                f"a field is identified by {', '.join(self.settings['fov_key'])}."
            ),
            (
                f"{c['features']} feature column{'s' if c['features'] != 1 else ''}: "
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
                f"{c['technical_repeats']} technical repeat"
                f"{'s' if c['technical_repeats'] != 1 else ''} ({technical_col}, a well or a "
                f"dish within a repeat{spread})."
            )
        if condition_col is None and repeat_col is None:
            lines.append(
                "No condition or repeat column: every field is one group (all fields pooled)."
            )
        else:
            parts = []
            if condition_col is not None:
                names = self.layout[condition_col].drop_duplicates().tolist()
                parts.append(
                    f"{c['conditions']} condition{'s' if c['conditions'] != 1 else ''} "
                    f"({condition_col}: {_listed(names, MAX_VALUES_LISTED)})"
                )
            else:
                parts.append("no condition column")
            if repeat_col is not None:
                names = self.layout[repeat_col].drop_duplicates().tolist()
                parts.append(
                    f"{c['repeats']} repeat{'s' if c['repeats'] != 1 else ''} "
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
                lines.append("  " + "  ".join([*header, *columns]))
                for row in table.itertuples(index=False):
                    row = row._asdict()
                    cells = [str(row[col]) for col in header] + [str(row[key]) for key in keys]
                    lines.append("  " + "  ".join(cells))
            else:
                lines.append(
                    f"  ({len(table)} condition-and-repeat combinations; see the layout table)"
                )
        normalized = self.settings.get("normalized")
        if normalized:
            lines.append(f"Values: {normalized}.")
        else:
            lines.append("Values: as given (not normalised).")
        return "\n".join(lines)


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
) -> DatasetOverview:
    """Count what the tables hold (see the module). ``normalized`` is a sentence to print.

    Each column argument is used when the FOV table has it; a technical
    repeat is counted within its condition and repeat (well J23 of two
    plates is two technical repeats).
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
    }
    settings: dict[str, Any] = {
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "fov_key": list(data.fov_key),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "normalized": normalized,
    }
    return DatasetOverview(counts, layout, tuple(data.feature_cols), settings)


def normalized_sentence(control: Mapping[str, Any], repeat_col: str | None) -> str:
    """How the values were normalised, for the overview and the declaration."""
    where = f"the same {repeat_col}" if repeat_col else "the whole table"
    return f"every feature divided by the mean of the control {dict(control)} objects of {where}"


__all__ = ["DatasetOverview", "dataset_overview", "normalized_sentence"]
